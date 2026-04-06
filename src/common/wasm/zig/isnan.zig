//! WASM SIMD isnan kernels: out[i] = 1 if a[i] is NaN, else 0.
//!
//! NaN detection via exponent+mantissa bit logic:
//!   f64 NaN: exponent = 0x7FF (bits 62-52 all set) AND mantissa != 0
//!   f32 NaN: exponent = 0xFF  (bits 30-23 all set) AND mantissa != 0
//!   f16 NaN: exponent = 0x1F  (bits 14-10 all set) AND mantissa != 0
//!
//! SIMD approach: reinterpret as integer, mask exponent, compare, mask mantissa, compare.

const simd = @import("simd.zig");

/// isnan for f64 — scalar (no i64x2 GT in WASM SIMD).
/// NaN iff abs(bits) > inf: (bits & 0x7FFF...) > 0x7FF0...
export fn isnan_f64(a: [*]const f64, out: [*]u8, N: u32) void {
    const a_u64: [*]const u64 = @ptrCast(a);
    const abs_mask: u64 = 0x7FFF_FFFF_FFFF_FFFF;
    const inf_bits: u64 = 0x7FF0_0000_0000_0000;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if ((a_u64[i] & abs_mask) > inf_bits) 1 else 0;
    }
}

/// isnan for f32 — 4-wide SIMD: 1 AND + 1 GT per vector.
/// NaN iff abs(bits) > inf: (bits & 0x7FFFFFFF) > 0x7F800000.
/// Signed i32 GT works because both operands are positive after masking.
export fn isnan_f32(a: [*]const f32, out: [*]u8, N: u32) void {
    const a_i32: [*]const i32 = @ptrCast(a);
    const n4 = N & ~@as(u32, 3);
    var i: u32 = 0;

    const abs_v: @Vector(4, i32) = @splat(0x7FFF_FFFF);
    const inf_v: @Vector(4, i32) = @splat(0x7F80_0000);

    while (i < n4) : (i += 4) {
        const bits: @Vector(4, i32) = @as(*align(1) const @Vector(4, i32), @ptrCast(a_i32 + i)).*;
        const is_nan = (bits & abs_v) > inf_v;
        inline for (0..4) |lane| {
            out[i + lane] = @intFromBool(is_nan[lane]);
        }
    }
    while (i < N) : (i += 1) {
        out[i] = if ((a_i32[i] & 0x7FFF_FFFF) > 0x7F80_0000) 1 else 0;
    }
}

/// isnan for f16 (as raw u16 bit patterns) — 8-wide SIMD: 1 AND + 1 GT per vector.
/// NaN iff (bits & 0x7FFF) > 0x7C00. Signed i16 GT works (both operands positive).
export fn isnan_u16(a: [*]const u16, out: [*]u8, N: u32) void {
    const a_i16: [*]const i16 = @ptrCast(a);
    const n8 = N & ~@as(u32, 7);
    var i: u32 = 0;

    const abs_v: @Vector(8, i16) = @splat(0x7FFF);
    const inf_v: @Vector(8, i16) = @splat(0x7C00);

    while (i < n8) : (i += 8) {
        const bits: @Vector(8, i16) = @as(*align(1) const @Vector(8, i16), @ptrCast(a_i16 + i)).*;
        const is_nan = (bits & abs_v) > inf_v;
        inline for (0..8) |lane| {
            out[i + lane] = @intFromBool(is_nan[lane]);
        }
    }
    while (i < N) : (i += 1) {
        out[i] = if ((a_i16[i] & 0x7FFF) > 0x7C00) 1 else 0;
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
