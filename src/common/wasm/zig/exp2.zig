//! WASM element-wise exp2 kernels for float / int / complex types.
//!
//! Unary: out[i] = exp2(a[i]) = 2^x. 2^x = 2^n·2^r with n = round(x), and 2^r =
//! exp(r·ln2) over a tiny range nailed by a degree-13 Taylor poly. The exp2v
//! cores are op-specific (local); complex uses 2^z = exp(z·ln2) via the shared
//! transcend cores.

const math = @import("std").math;
const simd = @import("simd.zig");
const t = @import("transcend.zig");

const LN2_F64: f64 = 0.6931471805599453;
const EXP2_MAX_F64: f64 = 1024.0;
const EXP2_MIN_F64: f64 = -1074.0;

inline fn exp2v_f64(x_in: simd.V2f64) simd.V2f64 {
    const maxv: simd.V2f64 = @splat(EXP2_MAX_F64);
    const minv: simd.V2f64 = @splat(EXP2_MIN_F64);
    const x = simd.max_f64x2(simd.min_f64x2(x_in, maxv), minv);

    const n = @floor(x + @as(simd.V2f64, @splat(0.5)));
    const r = (x - n) * @as(simd.V2f64, @splat(LN2_F64));

    var expr: simd.V2f64 = @splat(1.6059043836821613e-10);
    expr = simd.mulAdd_f64x2(expr, r, @splat(2.08767569878681e-9));
    expr = simd.mulAdd_f64x2(expr, r, @splat(2.5052108385441720e-8));
    expr = simd.mulAdd_f64x2(expr, r, @splat(2.7557319223985893e-7));
    expr = simd.mulAdd_f64x2(expr, r, @splat(2.7557319223985888e-6));
    expr = simd.mulAdd_f64x2(expr, r, @splat(2.4801587301587302e-5));
    expr = simd.mulAdd_f64x2(expr, r, @splat(1.9841269841269841e-4));
    expr = simd.mulAdd_f64x2(expr, r, @splat(1.3888888888888889e-3));
    expr = simd.mulAdd_f64x2(expr, r, @splat(8.3333333333333332e-3));
    expr = simd.mulAdd_f64x2(expr, r, @splat(4.1666666666666664e-2));
    expr = simd.mulAdd_f64x2(expr, r, @splat(1.6666666666666666e-1));
    expr = simd.mulAdd_f64x2(expr, r, @splat(5.0e-1));
    expr = simd.mulAdd_f64x2(expr, r, @splat(1.0));
    expr = simd.mulAdd_f64x2(expr, r, @splat(1.0));

    const ni: simd.V2i64 = @intFromFloat(n);
    const biased = (ni +% @as(simd.V2i64, @splat(1023))) << @as(simd.V2i64, @splat(52));
    const pow2n: simd.V2f64 = @bitCast(biased);

    var result = expr * pow2n;
    result = @select(f64, x_in > maxv, @as(simd.V2f64, @splat(math.inf(f64))), result);
    result = @select(f64, x_in < minv, @as(simd.V2f64, @splat(0.0)), result);
    return result;
}

const LN2_F32: f32 = 0.6931471805599453;
const EXP2_MAX_F32: f32 = 128.0;
const EXP2_MIN_F32: f32 = -149.0;

inline fn exp2v_f32(x_in: simd.V4f32) simd.V4f32 {
    const maxv: simd.V4f32 = @splat(EXP2_MAX_F32);
    const minv: simd.V4f32 = @splat(EXP2_MIN_F32);
    const x = simd.max_f32x4(simd.min_f32x4(x_in, maxv), minv);

    const n = @floor(x + @as(simd.V4f32, @splat(0.5)));
    const r = (x - n) * @as(simd.V4f32, @splat(LN2_F32));

    var p: simd.V4f32 = @splat(1.9875691500e-4);
    p = simd.mulAdd_f32x4(p, r, @splat(1.3981999507e-3));
    p = simd.mulAdd_f32x4(p, r, @splat(8.3334519073e-3));
    p = simd.mulAdd_f32x4(p, r, @splat(4.1665795894e-2));
    p = simd.mulAdd_f32x4(p, r, @splat(1.6666665459e-1));
    p = simd.mulAdd_f32x4(p, r, @splat(5.0000001201e-1));
    var expr = simd.mulAdd_f32x4(p, r * r, r);
    expr = expr + @as(simd.V4f32, @splat(1.0));

    const ni: simd.V4i32 = @intFromFloat(n);
    const biased = (ni +% @as(simd.V4i32, @splat(127))) << @as(simd.V4i32, @splat(23));
    const pow2n: simd.V4f32 = @bitCast(biased);

    var result = expr * pow2n;
    result = @select(f32, x_in > maxv, @as(simd.V4f32, @splat(math.inf(f32))), result);
    result = @select(f32, x_in < minv, @as(simd.V4f32, @splat(0.0)), result);
    return result;
}

/// Compute exp2 via exp(x · ln2) for the scalar tails.
inline fn exp2_via_exp(x: f64) f64 {
    return math.exp(x * math.ln2);
}

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

// --- Integer inputs ---
inline fn exp2Int_f32(comptime I: type, a: [*]const I, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        const vi = @as(*align(1) const @Vector(4, I), @ptrCast(a + i)).*;
        simd.store4_f32(out, i, exp2v_f32(@floatFromInt(vi)));
    }
    while (i < N) : (i += 1) {
        out[i] = @floatCast(exp2_via_exp(@floatFromInt(a[i])));
    }
}

inline fn exp2Int_f64(comptime I: type, a: [*]const I, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const vi = @as(*align(1) const @Vector(2, I), @ptrCast(a + i)).*;
        const xf: simd.V2f64 = @floatFromInt(vi);
        simd.store2_f64(out, i, exp2v_f64(xf));
    }
    while (i < N) : (i += 1) {
        out[i] = exp2_via_exp(@floatFromInt(a[i]));
    }
}

export fn exp2_i64_f64(a: [*]const i64, out: [*]f64, N: u32) void {
    exp2Int_f64(i64, a, out, N);
}
export fn exp2_u64_f64(a: [*]const u64, out: [*]f64, N: u32) void {
    exp2Int_f64(u64, a, out, N);
}
export fn exp2_i32_f64(a: [*]const i32, out: [*]f64, N: u32) void {
    exp2Int_f64(i32, a, out, N);
}
export fn exp2_u32_f64(a: [*]const u32, out: [*]f64, N: u32) void {
    exp2Int_f64(u32, a, out, N);
}
export fn exp2_i16_f32(a: [*]const i16, out: [*]f32, N: u32) void {
    exp2Int_f32(i16, a, out, N);
}
export fn exp2_u16_f32(a: [*]const u16, out: [*]f32, N: u32) void {
    exp2Int_f32(u16, a, out, N);
}
export fn exp2_i8_f32(a: [*]const i8, out: [*]f32, N: u32) void {
    exp2Int_f32(i8, a, out, N);
}
export fn exp2_u8_f32(a: [*]const u8, out: [*]f32, N: u32) void {
    exp2Int_f32(u8, a, out, N);
}

// --- Complex: 2^(a+bi) = exp(a·ln2)·(cos(b·ln2) + i·sin(b·ln2)) ---
inline fn opExp2(re: simd.V2f64, im: simd.V2f64, out_re: *simd.V2f64, out_im: *simd.V2f64) void {
    const ln2: simd.V2f64 = @splat(LN2_F64);
    const e = t.expv_f64(re * ln2);
    const br = im * ln2;
    out_re.* = e * t.cosv_f64(br);
    out_im.* = e * t.sinv_f64(br);
}

export fn exp2_c128(a: [*]const f64, out: [*]f64, N: u32) void {
    t.cdrive_c128(a, out, N, opExp2);
}
export fn exp2_c64(a: [*]const f32, out: [*]f32, N: u32) void {
    t.cdrive_c64(a, out, N, opExp2);
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

test "exp2_i32_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 0, 4 };
    var out: [2]f64 = undefined;
    exp2_i32_f64(&a, &out, 2);
    try testing.expectApproxEqAbs(out[1], 16.0, 1e-10);
}

test "exp2_c128 matches 2^a(cos(b ln2), sin(b ln2))" {
    const testing = @import("std").testing;
    const a = [_]f64{ 3.0, 1.0 };
    var out: [2]f64 = undefined;
    exp2_c128(&a, &out, 1);
    const e = @exp(3.0 * math.ln2);
    try testing.expectApproxEqRel(out[0], e * @cos(1.0 * math.ln2), 1e-12);
    try testing.expectApproxEqRel(out[1], e * @sin(1.0 * math.ln2), 1e-12);
}

test "exp2_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 0.0, 1.0, 3.0, 4.0 };
    var out: [4]f32 = undefined;
    exp2_f32(&a, &out, 4);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-4);
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-4);
    try testing.expectApproxEqAbs(out[2], 8.0, 1e-3);
    try testing.expectApproxEqAbs(out[3], 16.0, 1e-3);
}

test "exp2 int variants" {
    const testing = @import("std").testing;
    const ai = [_]i64{ 0, 4 };
    const au = [_]u64{ 0, 4 };
    const au32 = [_]u32{ 0, 4 };
    var o64: [2]f64 = undefined;
    exp2_i64_f64(&ai, &o64, 2);
    try testing.expectApproxEqAbs(o64[1], 16.0, 1e-10);
    exp2_u64_f64(&au, &o64, 2);
    try testing.expectApproxEqAbs(o64[1], 16.0, 1e-10);
    exp2_u32_f64(&au32, &o64, 2);
    try testing.expectApproxEqAbs(o64[1], 16.0, 1e-10);

    const ai16 = [_]i16{ 0, 1, 4, 3 };
    const au16 = [_]u16{ 0, 1, 4, 3 };
    const ai8 = [_]i8{ 0, 1, 4, 3 };
    const au8 = [_]u8{ 0, 1, 4, 3 };
    var o32: [4]f32 = undefined;
    exp2_i16_f32(&ai16, &o32, 4);
    try testing.expectApproxEqAbs(o32[2], 16.0, 1e-3);
    exp2_u16_f32(&au16, &o32, 4);
    try testing.expectApproxEqAbs(o32[2], 16.0, 1e-3);
    exp2_i8_f32(&ai8, &o32, 4);
    try testing.expectApproxEqAbs(o32[2], 16.0, 1e-3);
    exp2_u8_f32(&au8, &o32, 4);
    try testing.expectApproxEqAbs(o32[2], 16.0, 1e-3);
}

test "exp2_c64 matches identity" {
    const testing = @import("std").testing;
    const a = [_]f32{ 3.0, 1.0 };
    var out: [2]f32 = undefined;
    exp2_c64(&a, &out, 1);
    const e = @exp(@as(f32, 3.0) * @as(f32, math.ln2));
    try testing.expectApproxEqAbs(out[0], e * @cos(@as(f32, 1.0) * @as(f32, math.ln2)), 1e-3);
    try testing.expectApproxEqAbs(out[1], e * @sin(@as(f32, 1.0) * @as(f32, math.ln2)), 1e-3);
}
