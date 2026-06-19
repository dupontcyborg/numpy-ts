//! WASM element-wise exp kernels for float / int / complex types.
//!
//! Unary: out[i] = exp(a[i])
//! Float/int fast paths range-reduce x = n·ln2 + r and evaluate exp(r) with a
//! Cephes Taylor polynomial (FMA, ≈1 ulp), then scale by 2^n. The f64 core is
//! shared via transcend.zig; the f32 core is local (not shared). Complex uses
//! exp(a+bi) = e^a·(cos b + i·sin b) composed from the shared cores.

const math = @import("std").math;
const simd = @import("simd.zig");
const t = @import("transcend.zig");

// --- f32 core (local; complex routes through transcend's f64 core) ---
const EXP_LOG2E_F32: f32 = 1.44269504088896341;
const EXP_C1_F32: f32 = 0.693359375;
const EXP_C2_F32: f32 = -2.12194440e-4;
const EXP_MAXLOG_F32: f32 = 88.3762626647949;
const EXP_MINLOG_F32: f32 = -87.3365447504019;

inline fn expv_f32(x_in: simd.V4f32) simd.V4f32 {
    const maxv: simd.V4f32 = @splat(EXP_MAXLOG_F32);
    const minv: simd.V4f32 = @splat(EXP_MINLOG_F32);
    const x = simd.max_f32x4(simd.min_f32x4(x_in, maxv), minv);

    const n = @floor(simd.mulAdd_f32x4(x, @as(simd.V4f32, @splat(EXP_LOG2E_F32)), @as(simd.V4f32, @splat(0.5))));
    var r = simd.nmulAdd_f32x4(n, @splat(EXP_C1_F32), x);
    r = simd.nmulAdd_f32x4(n, @splat(EXP_C2_F32), r);

    const rr = r * r;
    var p: simd.V4f32 = @splat(1.9875691500e-4);
    p = simd.mulAdd_f32x4(p, r, @splat(1.3981999507e-3));
    p = simd.mulAdd_f32x4(p, r, @splat(8.3334519073e-3));
    p = simd.mulAdd_f32x4(p, r, @splat(4.1665795894e-2));
    p = simd.mulAdd_f32x4(p, r, @splat(1.6666665459e-1));
    p = simd.mulAdd_f32x4(p, r, @splat(5.0000001201e-1));
    var expr = simd.mulAdd_f32x4(p, rr, r);
    expr = expr + @as(simd.V4f32, @splat(1.0));

    const ni: simd.V4i32 = @intFromFloat(n);
    const biased = (ni +% @as(simd.V4i32, @splat(127))) << @as(simd.V4i32, @splat(23));
    const pow2n: simd.V4f32 = @bitCast(biased);

    var result = expr * pow2n;
    result = @select(f32, x_in > maxv, @as(simd.V4f32, @splat(math.inf(f32))), result);
    result = @select(f32, x_in < minv, @as(simd.V4f32, @splat(0.0)), result);
    return result;
}

/// Element-wise exp for f64 using 2-wide SIMD.
export fn exp_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, t.expv_f64(simd.load2_f64(a, i)));
    }
    while (i < N) : (i += 1) {
        out[i] = math.exp(a[i]);
    }
}

/// Element-wise exp for f32 using 4-wide SIMD.
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

// --- Integer inputs: widen to float in SIMD, then the same poly core. ---
// i8/i16/i32/u8/u16/u32 widen with native SIMD converts; i64/u64 scalarize the
// widen but still share the vectorized polynomial.

/// i8/u8/i16/u16 → f32 output, 4-wide.
inline fn expInt_f32(comptime I: type, a: [*]const I, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        const vi = @as(*align(1) const @Vector(4, I), @ptrCast(a + i)).*;
        simd.store4_f32(out, i, expv_f32(@floatFromInt(vi)));
    }
    while (i < N) : (i += 1) {
        out[i] = @floatCast(math.exp(@as(f64, @floatFromInt(a[i]))));
    }
}

/// i32/u32/i64/u64 → f64 output, 2-wide.
inline fn expInt_f64(comptime I: type, a: [*]const I, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const vi = @as(*align(1) const @Vector(2, I), @ptrCast(a + i)).*;
        const xf: simd.V2f64 = @floatFromInt(vi);
        simd.store2_f64(out, i, t.expv_f64(xf));
    }
    while (i < N) : (i += 1) {
        out[i] = math.exp(@as(f64, @floatFromInt(a[i])));
    }
}

export fn exp_i64_f64(a: [*]const i64, out: [*]f64, N: u32) void {
    expInt_f64(i64, a, out, N);
}
export fn exp_u64_f64(a: [*]const u64, out: [*]f64, N: u32) void {
    expInt_f64(u64, a, out, N);
}
export fn exp_i32_f64(a: [*]const i32, out: [*]f64, N: u32) void {
    expInt_f64(i32, a, out, N);
}
export fn exp_u32_f64(a: [*]const u32, out: [*]f64, N: u32) void {
    expInt_f64(u32, a, out, N);
}
export fn exp_i16_f32(a: [*]const i16, out: [*]f32, N: u32) void {
    expInt_f32(i16, a, out, N);
}
export fn exp_u16_f32(a: [*]const u16, out: [*]f32, N: u32) void {
    expInt_f32(u16, a, out, N);
}
export fn exp_i8_f32(a: [*]const i8, out: [*]f32, N: u32) void {
    expInt_f32(i8, a, out, N);
}
export fn exp_u8_f32(a: [*]const u8, out: [*]f32, N: u32) void {
    expInt_f32(u8, a, out, N);
}

// --- Complex: exp(a+bi) = e^a·(cos b + i·sin b) ---
inline fn opExp(re: simd.V2f64, im: simd.V2f64, out_re: *simd.V2f64, out_im: *simd.V2f64) void {
    const e = t.expv_f64(re);
    out_re.* = e * t.cosv_f64(im);
    out_im.* = e * t.sinv_f64(im);
}

export fn exp_c128(a: [*]const f64, out: [*]f64, N: u32) void {
    t.cdrive_c128(a, out, N, opExp);
}
export fn exp_c64(a: [*]const f32, out: [*]f32, N: u32) void {
    t.cdrive_c64(a, out, N, opExp);
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

test "exp_i32_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 0, 1, 2 };
    var out: [3]f64 = undefined;
    exp_i32_f64(&a, &out, 3);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    try testing.expectApproxEqRel(out[2], @exp(2.0), 1e-12);
}

test "exp_c128 matches e^a(cos b, sin b)" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1.0, 2.0, -0.5, 1.5 };
    var out: [4]f64 = undefined;
    exp_c128(&a, &out, 2);
    try testing.expectApproxEqRel(out[0], @exp(1.0) * @cos(2.0), 1e-12);
    try testing.expectApproxEqRel(out[1], @exp(1.0) * @sin(2.0), 1e-12);
    try testing.expectApproxEqRel(out[2], @exp(-0.5) * @cos(1.5), 1e-12);
    try testing.expectApproxEqRel(out[3], @exp(-0.5) * @sin(1.5), 1e-12);
}

test "exp int variants" {
    const testing = @import("std").testing;
    const ai = [_]i64{ 0, 2 };
    const au = [_]u64{ 0, 2 };
    const au32 = [_]u32{ 0, 2 };
    var o64: [2]f64 = undefined;
    exp_i64_f64(&ai, &o64, 2);
    try testing.expectApproxEqRel(o64[1], @exp(2.0), 1e-12);
    exp_u64_f64(&au, &o64, 2);
    try testing.expectApproxEqRel(o64[1], @exp(2.0), 1e-12);
    exp_u32_f64(&au32, &o64, 2);
    try testing.expectApproxEqRel(o64[1], @exp(2.0), 1e-12);

    const ai16 = [_]i16{ 0, 1, 2, 3 };
    const au16 = [_]u16{ 0, 1, 2, 3 };
    const ai8 = [_]i8{ 0, 1, 2, 3 };
    const au8 = [_]u8{ 0, 1, 2, 3 };
    var o32: [4]f32 = undefined;
    exp_i16_f32(&ai16, &o32, 4);
    try testing.expectApproxEqAbs(o32[2], @exp(@as(f32, 2.0)), 1e-4);
    exp_u16_f32(&au16, &o32, 4);
    try testing.expectApproxEqAbs(o32[2], @exp(@as(f32, 2.0)), 1e-4);
    exp_i8_f32(&ai8, &o32, 4);
    try testing.expectApproxEqAbs(o32[2], @exp(@as(f32, 2.0)), 1e-4);
    exp_u8_f32(&au8, &o32, 4);
    try testing.expectApproxEqAbs(o32[2], @exp(@as(f32, 2.0)), 1e-4);
}

test "exp_c64 matches e^a(cos b, sin b)" {
    const testing = @import("std").testing;
    const a = [_]f32{ 1.0, 2.0 };
    var out: [2]f32 = undefined;
    exp_c64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], @exp(@as(f32, 1.0)) * @cos(@as(f32, 2.0)), 1e-4);
    try testing.expectApproxEqAbs(out[1], @exp(@as(f32, 1.0)) * @sin(@as(f32, 2.0)), 1e-4);
}
