//! WASM element-wise logarithm kernels (natural, base-2, base-10) for floats.
//!
//! Unary: out[i] = log(a[i]) / log2(a[i]) / log10(a[i])
//! Operates on contiguous 1D buffers of length N. Float inputs only — integer
//! inputs keep the JS fallback path (the wrapper returns null for them).
//!
//! There is no log CPU opcode on any ISA; it's a software polynomial. We use the
//! classic Cephes scheme: split x = m·2^e with m ∈ [√½, √2), then approximate
//! log(m) with a minimax rational and add e·ln2. log2/log10 just rescale. The
//! float lanes (f64x2 / f32x4) vectorize the mantissa/exponent bit-twiddling,
//! the polynomial, and the divide — all base SIMD128 ops, with the polynomial
//! Horner chains lowering to relaxed_madd on +relaxed_simd builds.

const math = @import("std").math;
const simd = @import("simd.zig");

const SQRTH_F64: f64 = 0.70710678118654752440;
const LN2_HI_F64: f64 = 0.693359375;
const LN2_LO_F64: f64 = -2.121944400546905827679e-4;
const LOG2E_F64: f64 = 1.4426950408889634073599; // 1/ln2
const LOG10E_F64: f64 = 0.43429448190325182765; // 1/ln10

/// Vectorized natural log for a 2-wide f64 lane (Cephes, ≈1 ulp on normals).
inline fn logv_f64(x: simd.V2f64) simd.V2f64 {
    // frexp: m ∈ [0.5, 1), e = exponent, via exponent-field bit twiddling.
    const bits: simd.V2u64 = @bitCast(x);
    const raw_e: simd.V2u64 = (bits >> @as(simd.V2u64, @splat(52))) & @as(simd.V2u64, @splat(0x7ff));
    const ei: simd.V2i64 = @as(simd.V2i64, @bitCast(raw_e)) -% @as(simd.V2i64, @splat(1022));
    const mant_bits = (bits & @as(simd.V2u64, @splat(0x000fffffffffffff))) |
        @as(simd.V2u64, @splat(0x3fe0000000000000));
    var m: simd.V2f64 = @bitCast(mant_bits);
    var e: simd.V2f64 = @floatFromInt(ei);

    // Keep the reduced argument near 0: if m < √½, use 2m-1 and drop one exponent.
    const lo = m < @as(simd.V2f64, @splat(SQRTH_F64));
    e = @select(f64, lo, e - @as(simd.V2f64, @splat(1.0)), e);
    m = @select(f64, lo, m + m - @as(simd.V2f64, @splat(1.0)), m - @as(simd.V2f64, @splat(1.0)));

    const z = m * m;
    // P(m): degree-5 Horner.
    var p: simd.V2f64 = @splat(1.01875663804580931796e-4);
    p = simd.mulAdd_f64x2(p, m, @splat(4.97494994976747001425e-1));
    p = simd.mulAdd_f64x2(p, m, @splat(4.70579119878881725854e0));
    p = simd.mulAdd_f64x2(p, m, @splat(1.44989225341610930846e1));
    p = simd.mulAdd_f64x2(p, m, @splat(1.79368678507819816313e1));
    p = simd.mulAdd_f64x2(p, m, @splat(7.70838733755885391666e0));
    // Q(m): degree-5 monic Horner (leading 1).
    var q: simd.V2f64 = m + @as(simd.V2f64, @splat(1.12873587189167450590e1));
    q = simd.mulAdd_f64x2(q, m, @splat(4.52279145837532221105e1));
    q = simd.mulAdd_f64x2(q, m, @splat(8.29875266912776603211e1));
    q = simd.mulAdd_f64x2(q, m, @splat(7.11544750618563894466e1));
    q = simd.mulAdd_f64x2(q, m, @splat(2.31251620126765340583e1));

    var y = m * (z * (p / q));
    y = simd.mulAdd_f64x2(e, @splat(LN2_LO_F64), y); // y + e·ln2_lo
    y = simd.nmulAdd_f64x2(@splat(0.5), z, y); // y - 0.5·z
    const result = m + y;
    return simd.mulAdd_f64x2(e, @splat(LN2_HI_F64), result); // result + e·ln2_hi
}

/// Apply the natural-log core then scale, with NumPy-faithful domain handling:
/// x<0 → NaN, x==0 → -inf, x==+inf → +inf, NaN → NaN.
inline fn logScaled_f64(x: simd.V2f64, scale: f64) simd.V2f64 {
    var result = logv_f64(x) * @as(simd.V2f64, @splat(scale));
    const zero: simd.V2f64 = @splat(0.0);
    result = @select(f64, x < zero, @as(simd.V2f64, @splat(math.nan(f64))), result);
    result = @select(f64, x == zero, @as(simd.V2f64, @splat(-math.inf(f64))), result);
    result = @select(f64, x == @as(simd.V2f64, @splat(math.inf(f64))), @as(simd.V2f64, @splat(math.inf(f64))), result);
    result = @select(f64, x != x, x, result); // propagate NaN inputs
    return result;
}

fn logBase_f64(a: [*]const f64, out: [*]f64, N: u32, comptime scale: f64) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, logScaled_f64(simd.load2_f64(a, i), scale));
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = .{ a[i], a[i] };
        out[i] = logScaled_f64(v, scale)[0];
    }
}

export fn log_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    logBase_f64(a, out, N, 1.0);
}
export fn log2_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    logBase_f64(a, out, N, LOG2E_F64);
}
export fn log10_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    logBase_f64(a, out, N, LOG10E_F64);
}

// --- f32 (4-wide) ---

const SQRTH_F32: f32 = 0.707106781186547524;
const LN2_HI_F32: f32 = 0.693359375;
const LN2_LO_F32: f32 = -2.12194440e-4;
const LOG2E_F32: f32 = 1.44269504088896341;
const LOG10E_F32: f32 = 0.43429448190325182765;

/// Vectorized natural log for a 4-wide f32 lane (Cephes single, ≈1 ulp f32).
inline fn logv_f32(x: simd.V4f32) simd.V4f32 {
    const bits: simd.V4u32 = @bitCast(x);
    const raw_e: simd.V4u32 = (bits >> @as(simd.V4u32, @splat(23))) & @as(simd.V4u32, @splat(0xff));
    const ei: simd.V4i32 = @as(simd.V4i32, @bitCast(raw_e)) -% @as(simd.V4i32, @splat(126));
    const mant_bits = (bits & @as(simd.V4u32, @splat(0x007fffff))) |
        @as(simd.V4u32, @splat(0x3f000000));
    var m: simd.V4f32 = @bitCast(mant_bits);
    var e: simd.V4f32 = @floatFromInt(ei);

    const lo = m < @as(simd.V4f32, @splat(SQRTH_F32));
    e = @select(f32, lo, e - @as(simd.V4f32, @splat(1.0)), e);
    m = @select(f32, lo, m + m - @as(simd.V4f32, @splat(1.0)), m - @as(simd.V4f32, @splat(1.0)));

    const z = m * m;
    var p: simd.V4f32 = @splat(7.0376836292e-2);
    p = simd.mulAdd_f32x4(p, m, @splat(-1.1514610310e-1));
    p = simd.mulAdd_f32x4(p, m, @splat(1.1676998740e-1));
    p = simd.mulAdd_f32x4(p, m, @splat(-1.2420140846e-1));
    p = simd.mulAdd_f32x4(p, m, @splat(1.4249322787e-1));
    p = simd.mulAdd_f32x4(p, m, @splat(-1.6668057665e-1));
    p = simd.mulAdd_f32x4(p, m, @splat(2.0000714765e-1));
    p = simd.mulAdd_f32x4(p, m, @splat(-2.4999993993e-1));
    p = simd.mulAdd_f32x4(p, m, @splat(3.3333331174e-1));

    var y = p * m * z;
    y = simd.mulAdd_f32x4(e, @splat(LN2_LO_F32), y); // y + e·ln2_lo
    y = simd.nmulAdd_f32x4(@splat(0.5), z, y); // y - 0.5·z
    const result = m + y;
    return simd.mulAdd_f32x4(e, @splat(LN2_HI_F32), result); // result + e·ln2_hi
}

inline fn logScaled_f32(x: simd.V4f32, scale: f32) simd.V4f32 {
    var result = logv_f32(x) * @as(simd.V4f32, @splat(scale));
    const zero: simd.V4f32 = @splat(0.0);
    result = @select(f32, x < zero, @as(simd.V4f32, @splat(math.nan(f32))), result);
    result = @select(f32, x == zero, @as(simd.V4f32, @splat(-math.inf(f32))), result);
    result = @select(f32, x == @as(simd.V4f32, @splat(math.inf(f32))), @as(simd.V4f32, @splat(math.inf(f32))), result);
    result = @select(f32, x != x, x, result);
    return result;
}

fn logBase_f32(a: [*]const f32, out: [*]f32, N: u32, comptime scale: f32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_f32(out, i, logScaled_f32(simd.load4_f32(a, i), scale));
    }
    while (i < N) : (i += 1) {
        const v: simd.V4f32 = .{ a[i], a[i], a[i], a[i] };
        out[i] = logScaled_f32(v, scale)[0];
    }
}

export fn log_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    logBase_f32(a, out, N, 1.0);
}
export fn log2_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    logBase_f32(a, out, N, LOG2E_F32);
}
export fn log10_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    logBase_f32(a, out, N, LOG10E_F32);
}

// --- Tests ---

test "log_f64 matches std.math.log" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1.0, 2.0, 0.5, 10.0, 100.0, 0.001, 1234.5, 1e-8 };
    var out: [8]f64 = undefined;
    log_f64(&a, &out, 8);
    for (a, 0..) |x, i| {
        try testing.expectApproxEqRel(out[i], @log(x), 1e-12);
    }
}

test "log_f64 domain edges" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0.0, -1.0 };
    var out: [2]f64 = undefined;
    log_f64(&a, &out, 2);
    try testing.expect(out[0] == -math.inf(f64));
    try testing.expect(math.isNan(out[1]));
}

test "log2_f64 / log10_f64" {
    const testing = @import("std").testing;
    const a = [_]f64{ 8.0, 1000.0 };
    var out2: [2]f64 = undefined;
    var out10: [2]f64 = undefined;
    log2_f64(&a, &out2, 2);
    log10_f64(&a, &out10, 2);
    try testing.expectApproxEqAbs(out2[0], 3.0, 1e-12);
    try testing.expectApproxEqAbs(out10[1], 3.0, 1e-12);
}

test "log_f32 matches std.math.log" {
    const testing = @import("std").testing;
    const a = [_]f32{ 1.0, 2.0, 0.5, 10.0, 100.0, 0.01, 1234.5, 7.0 };
    var out: [8]f32 = undefined;
    log_f32(&a, &out, 8);
    for (a, 0..) |x, i| {
        try testing.expectApproxEqRel(out[i], @as(f32, @floatCast(@log(@as(f64, x)))), 1e-5);
    }
}
