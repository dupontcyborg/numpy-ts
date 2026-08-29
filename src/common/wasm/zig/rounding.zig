//! WASM element-wise rounding kernels: floor, ceil, trunc, rint, around.
//!
//! `floor`, `ceil` and `trunc` map straight onto `f64x2.floor` / `.ceil` /
//! `.trunc` (and the `f32x4` equivalents) — one instruction per lane group.
//!
//! NumPy's `rint` and `round` are both round-half-to-even, which Zig 0.16 has
//! no builtin for (`@round` is round-half-away-from-zero, wrong on exact ties).
//! So `rint` uses the add-magic/subtract-magic identity instead: adding 2^52 to
//! a non-negative double shifts its mantissa so one ulp is exactly 1.0, and
//! WASM's fixed round-to-nearest-even float ops round the value to an integer
//! in the process; subtracting 2^52 back leaves that integer exactly. The whole
//! thing is add/sub/abs/compare/select/bitwise-or, so it stays 2-wide with no
//! scalar tail logic. Inputs at or above the magic constant pass through
//! untouched (this also covers the infinities), and NaN propagates because it
//! compares false against the magic constant.

const simd = @import("simd.zig");

/// 2^52 — the smallest f64 whose ulp is 1.0.
const MAGIC_F64: f64 = 4503599627370496.0;
/// 2^23 — the smallest f32 whose ulp is 1.0.
const MAGIC_F32: f32 = 8388608.0;

const SIGN_F64: @Vector(2, u64) = @splat(0x8000000000000000);
const SIGN_F32: @Vector(4, u32) = @splat(0x80000000);

/// Round-half-to-even, 2 lanes of f64. See the module comment.
inline fn rint2(x: simd.V2f64) simd.V2f64 {
    const magic: simd.V2f64 = @splat(MAGIC_F64);
    const ax = @abs(x);
    // Rounds to integer under round-to-nearest-even, then shifts back down.
    const r = (ax + magic) - magic;
    // Re-apply the original sign bit rather than negating, so -0.4 rounds to
    // -0.0 and not +0.0 (NumPy keeps the sign of zero here).
    const signed: simd.V2f64 = @bitCast(@as(@Vector(2, u64), @bitCast(r)) | (@as(@Vector(2, u64), @bitCast(x)) & SIGN_F64));
    return @select(f64, ax >= magic, x, signed);
}

/// Round-half-to-even, 4 lanes of f32.
inline fn rint4(x: simd.V4f32) simd.V4f32 {
    const magic: simd.V4f32 = @splat(MAGIC_F32);
    const ax = @abs(x);
    const r = (ax + magic) - magic;
    const signed: simd.V4f32 = @bitCast(@as(@Vector(4, u32), @bitCast(r)) | (@as(@Vector(4, u32), @bitCast(x)) & SIGN_F32));
    return @select(f32, ax >= magic, x, signed);
}

/// Scalar round-half-to-even for the loop tail, same identity as rint2.
inline fn rintScalar(comptime T: type, x: T) T {
    const magic: T = if (T == f64) MAGIC_F64 else MAGIC_F32;
    const ax = @abs(x);
    if (!(ax < magic)) return x; // >= magic, or NaN (comparison is false)
    const r = (ax + magic) - magic;
    return if (x < 0 or (x == 0 and 1.0 / x < 0)) -r else r;
}

// --- floor / ceil / trunc: one native instruction each ---

export fn floor_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) simd.store2_f64(out, i, @floor(simd.load2_f64(a, i)));
    while (i < N) : (i += 1) out[i] = @floor(a[i]);
}

export fn floor_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) simd.store4_f32(out, i, @floor(simd.load4_f32(a, i)));
    while (i < N) : (i += 1) out[i] = @floor(a[i]);
}

export fn ceil_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) simd.store2_f64(out, i, @ceil(simd.load2_f64(a, i)));
    while (i < N) : (i += 1) out[i] = @ceil(a[i]);
}

export fn ceil_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) simd.store4_f32(out, i, @ceil(simd.load4_f32(a, i)));
    while (i < N) : (i += 1) out[i] = @ceil(a[i]);
}

export fn trunc_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) simd.store2_f64(out, i, @trunc(simd.load2_f64(a, i)));
    while (i < N) : (i += 1) out[i] = @trunc(a[i]);
}

export fn trunc_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) simd.store4_f32(out, i, @trunc(simd.load4_f32(a, i)));
    while (i < N) : (i += 1) out[i] = @trunc(a[i]);
}

// --- rint: round-half-to-even ---

export fn rint_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) simd.store2_f64(out, i, rint2(simd.load2_f64(a, i)));
    while (i < N) : (i += 1) out[i] = rintScalar(f64, a[i]);
}

export fn rint_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) simd.store4_f32(out, i, rint4(simd.load4_f32(a, i)));
    while (i < N) : (i += 1) out[i] = rintScalar(f32, a[i]);
}

// --- around: rint(x * m) / m, matching the JS formula for `decimals` ---
//
// The caller passes 10^decimals. For the common decimals=0 case m is 1.0, and
// the multiply/divide are exact no-ops rather than a separate code path.

export fn around_f64(a: [*]const f64, out: [*]f64, N: u32, m: f64) void {
    const mv: simd.V2f64 = @splat(m);
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, rint2(simd.load2_f64(a, i) * mv) / mv);
    }
    while (i < N) : (i += 1) out[i] = rintScalar(f64, a[i] * m) / m;
}

export fn around_f32(a: [*]const f32, out: [*]f32, N: u32, m: f32) void {
    const mv: simd.V4f32 = @splat(m);
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_f32(out, i, rint4(simd.load4_f32(a, i) * mv) / mv);
    }
    while (i < N) : (i += 1) out[i] = rintScalar(f32, a[i] * m) / m;
}

// --- tests ---

test "floor/ceil/trunc f64 including the scalar tail" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1.5, -1.5, 2.7, -2.7, 0.0, -0.5, 3.0 }; // odd length -> tail
    var out = [_]f64{0} ** 7;
    floor_f64(&a, &out, 7);
    try testing.expectEqualSlices(f64, &[_]f64{ 1, -2, 2, -3, 0, -1, 3 }, &out);
    ceil_f64(&a, &out, 7);
    try testing.expectEqualSlices(f64, &[_]f64{ 2, -1, 3, -2, 0, -0.0, 3 }, &out);
    trunc_f64(&a, &out, 7);
    try testing.expectEqualSlices(f64, &[_]f64{ 1, -1, 2, -2, 0, -0.0, 3 }, &out);
}

test "rint_f64 breaks exact ties toward even" {
    const testing = @import("std").testing;
    // Every .5 here must go to the even neighbour, in both signs.
    const a = [_]f64{ 0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5, 4.5 };
    var out = [_]f64{0} ** 9;
    rint_f64(&a, &out, 9);
    try testing.expectEqualSlices(
        f64,
        &[_]f64{ 0, 2, 2, 4, -0.0, -2, -2, -4, 4 },
        &out,
    );
}

test "rint_f64 does not treat near-ties as ties" {
    const testing = @import("std").testing;
    // These are not exact ties, so NumPy rounds them away from 2 rather than to it.
    const a = [_]f64{ 2.5000000000001, -2.5000000000001, 2.4999999999999, 3 };
    var out = [_]f64{0} ** 4;
    rint_f64(&a, &out, 4);
    try testing.expectEqualSlices(f64, &[_]f64{ 3, -3, 2, 3 }, &out);
}

test "rint_f64 passes through large values, infinities and NaN" {
    const testing = @import("std").testing;
    const big: f64 = 4503599627370496.0; // 2^52, already integral
    const inf = @import("std").math.inf(f64);
    const a = [_]f64{ big, big + 2.0, inf, -inf, 1e300, -1e300 };
    var out = [_]f64{0} ** 6;
    rint_f64(&a, &out, 6);
    try testing.expectEqualSlices(f64, &a, &out);

    const nan = @import("std").math.nan(f64);
    const b = [_]f64{ nan, 1.5 };
    var o2 = [_]f64{ 0, 0 };
    rint_f64(&b, &o2, 2);
    try testing.expect(@import("std").math.isNan(o2[0]));
    try testing.expectEqual(@as(f64, 2), o2[1]);
}

test "rint_f64 keeps the sign of zero" {
    const testing = @import("std").testing;
    const a = [_]f64{ -0.0, -0.4, 0.4, 0.0 };
    var out = [_]f64{0} ** 4;
    rint_f64(&a, &out, 4);
    // All four are zeros; the sign must survive, which is why rint2 re-applies
    // the sign bit instead of negating.
    for (out, 0..) |v, i| {
        try testing.expectEqual(@as(f64, 0), @abs(v));
        try testing.expectEqual(@import("std").math.signbit(a[i]), @import("std").math.signbit(v));
    }
}

test "rint_f32 breaks exact ties toward even" {
    const testing = @import("std").testing;
    const a = [_]f32{ 0.5, 1.5, 2.5, 3.5, -2.5, 8388608.0 };
    var out = [_]f32{0} ** 6;
    rint_f32(&a, &out, 6);
    try testing.expectEqualSlices(f32, &[_]f32{ 0, 2, 2, 4, -2, 8388608.0 }, &out);
}

test "around_f64 scales, rounds half-to-even, unscales" {
    const testing = @import("std").testing;
    // Expected values taken from NumPy 2.3.1: np.around(a, 2).
    const a = [_]f64{ 1.25, 1.35, -1.25, 2.675, 0.125 };
    var out = [_]f64{0} ** 5;
    around_f64(&a, &out, 5, 100.0); // decimals = 2
    // 125.0 and 135.0 are already integers after scaling — no tie to break.
    try testing.expectApproxEqAbs(@as(f64, 1.25), out[0], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.35), out[1], 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -1.25), out[2], 1e-12);
    // 2.675 * 100 is exactly 267.5, so the tie rule picks the even 268.
    try testing.expectApproxEqAbs(@as(f64, 2.68), out[3], 1e-12);
    // 0.125 * 100 is exactly 12.5 -> even -> 12.
    try testing.expectApproxEqAbs(@as(f64, 0.12), out[4], 1e-12);
}

test "around_f64 with m = 1 matches rint" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0.5, 1.5, 2.5, -1.5, 7.25 };
    var viaAround = [_]f64{0} ** 5;
    var viaRint = [_]f64{0} ** 5;
    around_f64(&a, &viaAround, 5, 1.0);
    rint_f64(&a, &viaRint, 5);
    try testing.expectEqualSlices(f64, &viaRint, &viaAround);
}
