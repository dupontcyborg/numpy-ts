//! WASM element-wise exp2 kernels for float types.
//!
//! Unary: out[i] = exp2(a[i])
//! Operates on contiguous 1D buffers of length N.
//! Only float types — integer inputs handled via JS-side conversion.
//!
//! Like exp, there is no exp2 CPU opcode anywhere — it's a software polynomial.
//! 2^x = 2^n · 2^r with n = round(x) and r = x − n ∈ [-0.5, 0.5], and 2^r =
//! exp(r·ln2) lands in the same tiny range a degree-13 Taylor poly nails to
//! machine precision. The float lanes (f64x2 / f32x4) vectorize the whole thing,
//! with the Horner FMAs lowering to relaxed_madd on +relaxed_simd builds.

const math = @import("std").math;
const simd = @import("simd.zig");

const LN2_F64: f64 = 0.6931471805599453;
const EXP2_MAX_F64: f64 = 1024.0; // 2^x overflows above this
const EXP2_MIN_F64: f64 = -1074.0; // 2^x underflows below this

/// Vectorized exp2 for a 2-wide f64 lane.
inline fn exp2v_f64(x_in: simd.V2f64) simd.V2f64 {
    const maxv: simd.V2f64 = @splat(EXP2_MAX_F64);
    const minv: simd.V2f64 = @splat(EXP2_MIN_F64);
    const x = simd.max_f64x2(simd.min_f64x2(x_in, maxv), minv);

    // n = round(x); r = (x - n)·ln2 ∈ [-ln2/2, ln2/2]
    const n = @floor(x + @as(simd.V2f64, @splat(0.5)));
    const r = (x - n) * @as(simd.V2f64, @splat(LN2_F64));

    // exp(r) via degree-13 Taylor–Horner (≈ machine precision on this range).
    var expr: simd.V2f64 = @splat(1.6059043836821613e-10); // 1/13!
    expr = simd.mulAdd_f64x2(expr, r, @splat(2.08767569878681e-9)); // 1/12!
    expr = simd.mulAdd_f64x2(expr, r, @splat(2.5052108385441720e-8)); // 1/11!
    expr = simd.mulAdd_f64x2(expr, r, @splat(2.7557319223985893e-7)); // 1/10!
    expr = simd.mulAdd_f64x2(expr, r, @splat(2.7557319223985888e-6)); // 1/9!
    expr = simd.mulAdd_f64x2(expr, r, @splat(2.4801587301587302e-5)); // 1/8!
    expr = simd.mulAdd_f64x2(expr, r, @splat(1.9841269841269841e-4)); // 1/7!
    expr = simd.mulAdd_f64x2(expr, r, @splat(1.3888888888888889e-3)); // 1/6!
    expr = simd.mulAdd_f64x2(expr, r, @splat(8.3333333333333332e-3)); // 1/5!
    expr = simd.mulAdd_f64x2(expr, r, @splat(4.1666666666666664e-2)); // 1/4!
    expr = simd.mulAdd_f64x2(expr, r, @splat(1.6666666666666666e-1)); // 1/3!
    expr = simd.mulAdd_f64x2(expr, r, @splat(5.0e-1)); // 1/2!
    expr = simd.mulAdd_f64x2(expr, r, @splat(1.0)); // 1/1!
    expr = simd.mulAdd_f64x2(expr, r, @splat(1.0)); // 1/0!

    const ni: simd.V2i64 = @intFromFloat(n);
    const biased = (ni +% @as(simd.V2i64, @splat(1023))) << @as(simd.V2i64, @splat(52));
    const pow2n: simd.V2f64 = @bitCast(biased);

    var result = expr * pow2n;
    result = @select(f64, x_in > maxv, @as(simd.V2f64, @splat(math.inf(f64))), result);
    result = @select(f64, x_in < minv, @as(simd.V2f64, @splat(0.0)), result);
    return result;
}

const LN2_F32: f32 = 0.6931471805599453;
const EXP2_MAX_F32: f32 = 128.0; // 2^x overflows above this (f32)
const EXP2_MIN_F32: f32 = -149.0; // 2^x underflows below this (f32)

/// Vectorized exp2 for a 4-wide f32 lane (true f32x4, ~1 ulp f32).
inline fn exp2v_f32(x_in: simd.V4f32) simd.V4f32 {
    const maxv: simd.V4f32 = @splat(EXP2_MAX_F32);
    const minv: simd.V4f32 = @splat(EXP2_MIN_F32);
    const x = simd.max_f32x4(simd.min_f32x4(x_in, maxv), minv);

    const n = @floor(x + @as(simd.V4f32, @splat(0.5)));
    const r = (x - n) * @as(simd.V4f32, @splat(LN2_F32));

    // exp(r) via Cephes single-precision degree-6 poly.
    var p: simd.V4f32 = @splat(1.9875691500e-4);
    p = simd.mulAdd_f32x4(p, r, @splat(1.3981999507e-3));
    p = simd.mulAdd_f32x4(p, r, @splat(8.3334519073e-3));
    p = simd.mulAdd_f32x4(p, r, @splat(4.1665795894e-2));
    p = simd.mulAdd_f32x4(p, r, @splat(1.6666665459e-1));
    p = simd.mulAdd_f32x4(p, r, @splat(5.0000001201e-1));
    var expr = simd.mulAdd_f32x4(p, r * r, r); // p·r² + r
    expr = expr + @as(simd.V4f32, @splat(1.0));

    const ni: simd.V4i32 = @intFromFloat(n);
    const biased = (ni +% @as(simd.V4i32, @splat(127))) << @as(simd.V4i32, @splat(23));
    const pow2n: simd.V4f32 = @bitCast(biased);

    var result = expr * pow2n;
    result = @select(f32, x_in > maxv, @as(simd.V4f32, @splat(math.inf(f32))), result);
    result = @select(f32, x_in < minv, @as(simd.V4f32, @splat(0.0)), result);
    return result;
}

/// Element-wise exp2 for f64 using 2-wide SIMD: out[i] = exp2(a[i]).
export fn exp2_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, exp2v_f64(simd.load2_f64(a, i)));
    }
    while (i < N) : (i += 1) {
        out[i] = exp2_via_exp(a[i]);
    }
}

/// Element-wise exp2 for f32 using 4-wide SIMD: out[i] = exp2(a[i]).
export fn exp2_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_f32(out, i, exp2v_f32(simd.load4_f32(a, i)));
    }
    while (i < N) : (i += 1) {
        out[i] = @floatCast(exp2_via_exp(a[i]));
    }
}

/// Element-wise exp2 for i64 → f64 output.
export fn exp2_i64_f64(a: [*]const i64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = exp2_via_exp(@floatFromInt(a[i]));
    }
}

/// Element-wise exp2 for u64 → f64 output.
export fn exp2_u64_f64(a: [*]const u64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = exp2_via_exp(@floatFromInt(a[i]));
    }
}

/// Compute exp2 via exp(x * ln2) to avoid ldexp dependency in WASM.
inline fn exp2_via_exp(x: f64) f64 {
    return math.exp(x * math.ln2);
}

/// Element-wise exp2 for i32 → f64 output.
export fn exp2_i32_f64(a: [*]const i32, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = exp2_via_exp(@floatFromInt(a[i]));
}

/// Element-wise exp2 for u32 → f64 output.
export fn exp2_u32_f64(a: [*]const u32, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = exp2_via_exp(@floatFromInt(a[i]));
}

/// Element-wise exp2 for i16 → f32 output.
export fn exp2_i16_f32(a: [*]const i16, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = @floatCast(exp2_via_exp(@floatFromInt(a[i])));
}

/// Element-wise exp2 for u16 → f32 output.
export fn exp2_u16_f32(a: [*]const u16, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = @floatCast(exp2_via_exp(@floatFromInt(a[i])));
}

/// Element-wise exp2 for i8 → f32 output.
export fn exp2_i8_f32(a: [*]const i8, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = @floatCast(exp2_via_exp(@floatFromInt(a[i])));
}

/// Element-wise exp2 for u8 → f32 output.
export fn exp2_u8_f32(a: [*]const u8, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = @floatCast(exp2_via_exp(@floatFromInt(a[i])));
}

// --- Tests ---

test "exp2_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0.0, 1.0, 3.0 };
    var out: [3]f64 = undefined;
    exp2_f64(&a, &out, 3);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 8.0, 1e-10);
}

test "exp2_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 0.0, 1.0, 3.0 };
    var out: [3]f32 = undefined;
    exp2_f32(&a, &out, 3);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-5);
    try testing.expectApproxEqAbs(out[2], 8.0, 1e-5);
}

test "exp2_i64_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{0};
    var out: [1]f64 = undefined;
    exp2_i64_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
}

test "exp2_u64_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]u64{0};
    var out: [1]f64 = undefined;
    exp2_u64_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
}

test "exp2_i32_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{0};
    var out: [1]f64 = undefined;
    exp2_i32_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
}

test "exp2_u32_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]u32{0};
    var out: [1]f64 = undefined;
    exp2_u32_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
}

test "exp2_i16_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{0};
    var out: [1]f32 = undefined;
    exp2_i16_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
}

test "exp2_u16_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{0};
    var out: [1]f32 = undefined;
    exp2_u16_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
}

test "exp2_i8_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{0};
    var out: [1]f32 = undefined;
    exp2_i8_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
}

test "exp2_u8_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{0};
    var out: [1]f32 = undefined;
    exp2_u8_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
}
