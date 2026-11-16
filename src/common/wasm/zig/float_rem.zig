//! Float remainder fix-ups shared by the modulo and divmod kernels.
//!
//! The kernels compute r = v − q·s from a SIMD floor or trunc quotient, which
//! is NaN for an infinite divisor (0·inf). A scalar divisor is checked once
//! before the loop and sent to the exact path when infinite. With an array
//! divisor, the SIMD loop stores r unchecked and ORs a NaN flag; when it is
//! set, a second pass recomputes the elements that are out of range or wrongly
//! signed from @rem, which is C fmod. Any per-lane check in the loop slows
//! these kernels noticeably, so the flag is the cheapest one that catches the
//! infinite divisor. Quotients past the mantissa are not caught there.

const std = @import("std");

/// NumPy's floor remainder: fmod, then moved into the divisor's sign. A zero
/// result takes the divisor's sign.
pub inline fn floorRem(comptime T: type, a: T, b: T) T {
    if (b == 0) return std.math.nan(T);
    const r = @rem(a, b);
    if (r == 0) return if (b < 0) -0.0 else 0.0;
    return if ((r < 0) != (b < 0)) r + b else r;
}

/// C fmod: the result, zero included, takes the dividend's sign.
pub inline fn truncRem(comptime T: type, a: T, b: T) T {
    if (b == 0) return std.math.nan(T);
    return @rem(a, b);
}

/// True for ±inf. A scalar divisor is checked once with this, so only array
/// divisors pay for flag in the loop.
pub inline fn isInf(comptime T: type, x: T) bool {
    return @abs(x) == std.math.inf(T);
}

/// Flag lanes whose fast remainder r is NaN, which is what an infinite divisor
/// produces. A NaN input or zero divisor trips it too and is repaired to NaN.
pub inline fn flag(comptime V: type, r: V, bad: *@Vector(@typeInfo(V).vector.len, bool)) void {
    bad.* = bad.* | (r != r);
}

/// Recompute every out[i], i < n, that is out of range or not signed like the
/// divisor, which covers every lane flag can raise. The divisor
/// for element i is b[i * b_stride], so a scalar divisor passes stride 0.
pub fn repairFloor(comptime T: type, a: [*]const T, b: [*]const T, b_stride: u32, out: [*]T, n: u32) void {
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        const s = b[i * b_stride];
        const r = out[i];
        if (!(@abs(r) < @abs(s) and (r == 0 or (r < 0) == (s < 0)))) out[i] = floorRem(T, a[i], s);
    }
}

/// Recompute every out[i], i < n, that is out of range or not signed like the
/// dividend.
pub fn repairTrunc(comptime T: type, a: [*]const T, b: [*]const T, b_stride: u32, out: [*]T, n: u32) void {
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        const s = b[i * b_stride];
        const r = out[i];
        if (!(@abs(r) < @abs(s) and (r == 0 or (r < 0) == (a[i] < 0)))) out[i] = truncRem(T, a[i], s);
    }
}

test "floorRem matches NumPy edge cases" {
    const t = std.testing;
    const inf = std.math.inf(f64);
    try t.expectEqual(@as(f64, 5), floorRem(f64, 5, inf));
    try t.expectEqual(inf, floorRem(f64, -5, inf));
    try t.expectEqual(-inf, floorRem(f64, 5, -inf));
    try t.expectEqual(@as(f64, -5), floorRem(f64, -5, -inf));
    try t.expectEqual(@as(f64, 5), floorRem(f64, 5, 1e17));
    try t.expect(std.math.signbit(floorRem(f64, 3, -3)));
    try t.expect(std.math.isNan(floorRem(f64, 3, 0)));
}

test "flag catches an infinite divisor and repairFloor fixes only that lane" {
    const t = std.testing;
    const V = @Vector(2, f64);
    const inf = std.math.inf(f64);
    const a = [_]f64{ 7, -5 };
    const b = [_]f64{ 3, inf };
    const v: V = a;
    const s: V = b;
    var bad: @Vector(2, bool) = @splat(false);
    const q = @floor(v / s);
    var out: [2]f64 = v - q * s;
    flag(V, out, &bad);
    try t.expect(@reduce(.Or, bad));
    repairFloor(f64, &a, &b, 1, &out, 2);
    try t.expectEqual(@as(f64, 1), out[0]);
    try t.expectEqual(inf, out[1]);
}

test "flag and repairTrunc with an infinite scalar divisor" {
    const t = std.testing;
    const V = @Vector(4, f32);
    const inf = std.math.inf(f32);
    const a = [_]f32{ 7, -7, -5, 5 };
    const s_scalar = [_]f32{inf};
    const v: V = a;
    const s: V = @splat(inf);
    var bad: @Vector(4, bool) = @splat(false);
    const q = @trunc(v / s);
    var out: [4]f32 = v - q * s;
    flag(V, out, &bad);
    try t.expect(@reduce(.Or, bad));
    repairTrunc(f32, &a, &s_scalar, 0, &out, 4);
    try t.expectEqualSlices(f32, &[_]f32{ 7, -7, -5, 5 }, &out);
}
