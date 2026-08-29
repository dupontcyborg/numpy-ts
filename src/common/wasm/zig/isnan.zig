//! WASM SIMD isnan kernels: out[i] = 1 if a[i] is NaN, else 0.
//!
//! For the float types this is a self-comparison — NaN is the only value that is
//! not equal to itself — which is one instruction per vector. The f16 entry point
//! takes raw u16 bits rather than a float type, so it keeps the bit test:
//! NaN iff (bits & 0x7FFF) > inf_bits.

/// NaN is the only value not equal to itself, so `v != v` is the whole test: one
/// `f64x2.ne` / `f32x4.ne` instead of the mask-and-compare bit trick documented
/// above. `@intFromBool` on the vector yields one byte per lane, which is already
/// the bool output layout, so nothing needs packing.
inline fn isnanFloat(comptime T: type, comptime L: comptime_int, a: [*]const T, out: [*]u8, N: u32) void {
    const V = @Vector(L, T);
    const n_simd = N & ~@as(u32, L - 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += L) {
        const v = @as(*align(1) const V, @ptrCast(a + i)).*;
        const m: @Vector(L, u8) = @intFromBool(v != v);
        @as(*align(1) [L]u8, @ptrCast(out + i)).* = m;
    }
    while (i < N) : (i += 1) out[i] = @intFromBool(a[i] != a[i]);
}

/// isnan for f64 — 2-wide.
export fn isnan_f64(a: [*]const f64, out: [*]u8, N: u32) void {
    isnanFloat(f64, 2, a, out, N);
}

/// isnan for f32 — 4-wide.
export fn isnan_f32(a: [*]const f32, out: [*]u8, N: u32) void {
    isnanFloat(f32, 4, a, out, N);
}

/// isnan for f16, taking raw u16 bit patterns — 8-wide.
///
/// This one keeps the bit trick: the input is not a float type, so there is
/// nothing to compare against itself without converting first. NaN iff
/// (bits & 0x7FFF) > 0x7C00; signed i16 GT is safe because masking clears the
/// sign bit, leaving both operands positive.
export fn isnan_u16(a: [*]const u16, out: [*]u8, N: u32) void {
    const a_i16: [*]const i16 = @ptrCast(a);
    const V = @Vector(8, i16);
    const abs_v: V = @splat(0x7FFF);
    const inf_v: V = @splat(0x7C00);
    const n8 = N & ~@as(u32, 7);
    var i: u32 = 0;
    while (i < n8) : (i += 8) {
        const bits = @as(*align(1) const V, @ptrCast(a_i16 + i)).*;
        const m: @Vector(8, u8) = @intFromBool((bits & abs_v) > inf_v);
        @as(*align(1) [8]u8, @ptrCast(out + i)).* = m;
    }
    while (i < N) : (i += 1) {
        out[i] = @intFromBool((a_i16[i] & 0x7FFF) > 0x7C00);
    }
}

// --- Tests ---

test "isnan_f64" {
    const testing = @import("std").testing;
    const nan: f64 = @bitCast(@as(u64, 0x7FF8000000000000));
    const inf: f64 = @bitCast(@as(u64, 0x7FF0000000000000));
    const a = [_]f64{ 1.0, nan, inf, -inf, 0.0, nan };
    var out: [6]u8 = undefined;
    isnan_f64(&a, &out, 6);
    try testing.expectEqual(out[0], 0); // 1.0
    try testing.expectEqual(out[1], 1); // NaN
    try testing.expectEqual(out[2], 0); // inf
    try testing.expectEqual(out[3], 0); // -inf
    try testing.expectEqual(out[4], 0); // 0.0
    try testing.expectEqual(out[5], 1); // NaN
}

test "isnan_f32" {
    const testing = @import("std").testing;
    const nan: f32 = @bitCast(@as(u32, 0x7FC00000));
    const inf: f32 = @bitCast(@as(u32, 0x7F800000));
    const a = [_]f32{ 1.0, nan, inf, -inf, 0.0, nan, 3.0, -nan };
    var out: [8]u8 = undefined;
    isnan_f32(&a, &out, 8);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 0);
    try testing.expectEqual(out[3], 0);
    try testing.expectEqual(out[4], 0);
    try testing.expectEqual(out[5], 1);
    try testing.expectEqual(out[6], 0);
    try testing.expectEqual(out[7], 1);
}

test "isnan_u16 (f16 bits)" {
    const testing = @import("std").testing;
    // f16 NaN: 0x7E00, f16 inf: 0x7C00, f16 1.0: 0x3C00, f16 0: 0x0000
    const a = [_]u16{ 0x3C00, 0x7E00, 0x7C00, 0xFC00, 0x0000, 0x7E01, 0x0001, 0x7C01 };
    var out: [8]u8 = undefined;
    isnan_u16(&a, &out, 8);
    try testing.expectEqual(out[0], 0); // 1.0
    try testing.expectEqual(out[1], 1); // NaN
    try testing.expectEqual(out[2], 0); // +inf
    try testing.expectEqual(out[3], 0); // -inf
    try testing.expectEqual(out[4], 0); // 0.0
    try testing.expectEqual(out[5], 1); // NaN
    try testing.expectEqual(out[6], 0); // subnormal
    try testing.expectEqual(out[7], 1); // NaN (negative)
}

test "isnan v != v is not folded away" {
    const testing = @import("std").testing;
    // The whole kernel rests on `v != v` surviving optimisation. Zig does not
    // enable fast-math, so it must — but assert it directly, across the vector
    // body and the scalar tail, rather than trusting that.
    const nan64: f64 = @bitCast(@as(u64, 0x7FF8000000000000));
    const nan32: f32 = @bitCast(@as(u32, 0x7FC00000));
    // Odd length so the tail runs too; NaN in both halves.
    var a64: [7]f64 = .{ 1.0, nan64, 2.0, nan64, 3.0, 4.0, nan64 };
    var o64 = [_]u8{9} ** 7;
    isnan_f64(&a64, &o64, 7);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 1, 0, 1, 0, 0, 1 }, &o64);

    var a32: [9]f32 = .{ 1.0, nan32, 2.0, nan32, 3.0, 4.0, 5.0, 6.0, nan32 };
    var o32 = [_]u8{9} ** 9;
    isnan_f32(&a32, &o32, 9);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 1, 0, 1, 0, 0, 0, 0, 1 }, &o32);
}

test "isnan distinguishes NaN from infinities and signalling NaN" {
    const testing = @import("std").testing;
    const qnan: f64 = @bitCast(@as(u64, 0x7FF8000000000000));
    const snan: f64 = @bitCast(@as(u64, 0x7FF0000000000001)); // signalling
    const nnan: f64 = @bitCast(@as(u64, 0xFFF8000000000000)); // negative NaN
    const inf: f64 = @bitCast(@as(u64, 0x7FF0000000000000));
    const a = [_]f64{ qnan, snan, nnan, inf, -inf, 0.0, -0.0, 1e308 };
    var out = [_]u8{9} ** 8;
    isnan_f64(&a, &out, 8);
    try testing.expectEqualSlices(u8, &[_]u8{ 1, 1, 1, 0, 0, 0, 0, 0 }, &out);
}
