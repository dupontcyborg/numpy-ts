//! WASM element-wise exp kernels for float types.
//!
//! Unary: out[i] = exp(a[i])
//! Operates on contiguous 1D buffers of length N.
//! Only float types — integer inputs handled via JS-side conversion.
//!
//! WASM SIMD has no native exp instruction, but exp is a software polynomial on
//! every ISA. We range-reduce x = n·ln2 + r, evaluate exp(r) with a Cephes
//! Taylor polynomial (≈1 ulp) using fused multiply-add, then scale by 2^n built
//! from the exponent bits. Lanes are f64x2 / f32x4; on +relaxed_simd the FMAs
//! lower to relaxed_madd (1 op). Matches std.math.exp to the unit-test tol.

const math = @import("std").math;
const simd = @import("simd.zig");

// --- Cephes exp(x), double precision (~1 ulp) ---
const LOG2E_F64: f64 = 1.4426950408889634073599;
const EXP_C1_F64: f64 = 6.93145751953125e-1; // ln2 high part
const EXP_C2_F64: f64 = 1.42860682030941723212e-6; // ln2 low part
const EXP_MAXLOG_F64: f64 = 7.09782712893383996843e2; // exp overflows above this
const EXP_MINLOG_F64: f64 = -7.08396418532264106224e2; // exp underflows below this

/// Vectorized exp for a 2-wide f64 lane.
inline fn expv_f64(x_in: simd.V2f64) simd.V2f64 {
    const maxv: simd.V2f64 = @splat(EXP_MAXLOG_F64);
    const minv: simd.V2f64 = @splat(EXP_MINLOG_F64);
    const x = simd.max_f64x2(simd.min_f64x2(x_in, maxv), minv);

    // n = round(x / ln2); r = x - n·ln2  (two-part ln2 for accuracy)
    const n = @floor(simd.mulAdd_f64x2(x, @as(simd.V2f64, @splat(LOG2E_F64)), @as(simd.V2f64, @splat(0.5))));
    var r = simd.nmulAdd_f64x2(n, @splat(EXP_C1_F64), x); // x - n·C1
    r = simd.nmulAdd_f64x2(n, @splat(EXP_C2_F64), r); // r - n·C2

    // exp(r) on r ∈ [-ln2/2, ln2/2] via a degree-13 Taylor–Horner polynomial.
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

    // Scale by 2^n: pow2n = bitcast((n + 1023) << 52).
    const ni: simd.V2i64 = @intFromFloat(n);
    const biased = (ni +% @as(simd.V2i64, @splat(1023))) << @as(simd.V2i64, @splat(52));
    const pow2n: simd.V2f64 = @bitCast(biased);

    var result = expr * pow2n;
    // Restore exact saturation behaviour at the extremes.
    result = @select(f64, x_in > maxv, @as(simd.V2f64, @splat(math.inf(f64))), result);
    result = @select(f64, x_in < minv, @as(simd.V2f64, @splat(0.0)), result);
    return result;
}

// --- Cephes exp(x), single precision (~1 ulp f32) ---
const LOG2E_F32: f32 = 1.44269504088896341;
const EXP_C1_F32: f32 = 0.693359375;
const EXP_C2_F32: f32 = -2.12194440e-4;
const EXP_MAXLOG_F32: f32 = 88.3762626647949;
const EXP_MINLOG_F32: f32 = -87.3365447504019;

/// Vectorized exp for a 4-wide f32 lane.
inline fn expv_f32(x_in: simd.V4f32) simd.V4f32 {
    const maxv: simd.V4f32 = @splat(EXP_MAXLOG_F32);
    const minv: simd.V4f32 = @splat(EXP_MINLOG_F32);
    const x = simd.max_f32x4(simd.min_f32x4(x_in, maxv), minv);

    const n = @floor(simd.mulAdd_f32x4(x, @as(simd.V4f32, @splat(LOG2E_F32)), @as(simd.V4f32, @splat(0.5))));
    var r = simd.nmulAdd_f32x4(n, @splat(EXP_C1_F32), x);
    r = simd.nmulAdd_f32x4(n, @splat(EXP_C2_F32), r);

    // Horner minimax poly for exp(r), degree 6.
    const rr = r * r;
    var p: simd.V4f32 = @splat(1.9875691500e-4);
    p = simd.mulAdd_f32x4(p, r, @splat(1.3981999507e-3));
    p = simd.mulAdd_f32x4(p, r, @splat(8.3334519073e-3));
    p = simd.mulAdd_f32x4(p, r, @splat(4.1665795894e-2));
    p = simd.mulAdd_f32x4(p, r, @splat(1.6666665459e-1));
    p = simd.mulAdd_f32x4(p, r, @splat(5.0000001201e-1));
    var expr = simd.mulAdd_f32x4(p, rr, r); // p·r² + r
    expr = expr + @as(simd.V4f32, @splat(1.0));

    // Scale by 2^n: pow2n = bitcast((n + 127) << 23). f32x4 <-> i32x4 is native SIMD.
    const ni: simd.V4i32 = @intFromFloat(n);
    const biased = (ni +% @as(simd.V4i32, @splat(127))) << @as(simd.V4i32, @splat(23));
    const pow2n: simd.V4f32 = @bitCast(biased);

    var result = expr * pow2n;
    result = @select(f32, x_in > maxv, @as(simd.V4f32, @splat(math.inf(f32))), result);
    result = @select(f32, x_in < minv, @as(simd.V4f32, @splat(0.0)), result);
    return result;
}

/// Element-wise exp for f64 using 2-wide SIMD: out[i] = exp(a[i]).
export fn exp_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, expv_f64(simd.load2_f64(a, i)));
    }
    while (i < N) : (i += 1) {
        out[i] = math.exp(a[i]);
    }
}

/// Element-wise exp for f32 using 4-wide SIMD: out[i] = exp(a[i]).
export fn exp_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_f32(out, i, expv_f32(simd.load4_f32(a, i)));
    }
    while (i < N) : (i += 1) {
        out[i] = @floatCast(math.exp(@as(f64, a[i])));
    }
}

/// Element-wise exp for i64 → f64 output. Scalar (no i64 SIMD in WASM).
export fn exp_i64_f64(a: [*]const i64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = math.exp(@as(f64, @floatFromInt(a[i])));
    }
}

/// Element-wise exp for u64 → f64 output. Scalar (no u64 SIMD in WASM).
export fn exp_u64_f64(a: [*]const u64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = math.exp(@as(f64, @floatFromInt(a[i])));
    }
}

/// Element-wise exp for i32 → f64 output.
export fn exp_i32_f64(a: [*]const i32, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = math.exp(@as(f64, @floatFromInt(a[i])));
}

/// Element-wise exp for u32 → f64 output.
export fn exp_u32_f64(a: [*]const u32, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = math.exp(@as(f64, @floatFromInt(a[i])));
}

/// Element-wise exp for i16 → f32 output.
export fn exp_i16_f32(a: [*]const i16, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = @floatCast(math.exp(@as(f64, @floatFromInt(a[i]))));
}

/// Element-wise exp for u16 → f32 output.
export fn exp_u16_f32(a: [*]const u16, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = @floatCast(math.exp(@as(f64, @floatFromInt(a[i]))));
}

/// Element-wise exp for i8 → f32 output.
export fn exp_i8_f32(a: [*]const i8, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = @floatCast(math.exp(@as(f64, @floatFromInt(a[i]))));
}

/// Element-wise exp for u8 → f32 output.
export fn exp_u8_f32(a: [*]const u8, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = @floatCast(math.exp(@as(f64, @floatFromInt(a[i]))));
}

// --- Tests ---

test "exp_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0.0, 1.0 };
    var out: [2]f64 = undefined;
    exp_f64(&a, &out, 2);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 2.718281828459045, 1e-10);
}

test "exp_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 0.0, 1.0 };
    var out: [2]f32 = undefined;
    exp_f32(&a, &out, 2);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 2.7183, 1e-4);
}

test "exp_i64_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{0};
    var out: [1]f64 = undefined;
    exp_i64_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
}

test "exp_u64_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]u64{0};
    var out: [1]f64 = undefined;
    exp_u64_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
}

test "exp_i32_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{0};
    var out: [1]f64 = undefined;
    exp_i32_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
}

test "exp_u32_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]u32{0};
    var out: [1]f64 = undefined;
    exp_u32_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
}

test "exp_i16_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{0};
    var out: [1]f32 = undefined;
    exp_i16_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
}

test "exp_u16_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{0};
    var out: [1]f32 = undefined;
    exp_u16_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
}

test "exp_i8_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{0};
    var out: [1]f32 = undefined;
    exp_i8_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
}

test "exp_u8_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{0};
    var out: [1]f32 = undefined;
    exp_u8_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
}
