//! WASM SIMD isfinite kernels: out[i] = 1 if a[i] is finite, else 0.
//!
//! A value is finite iff its exponent bits are NOT all ones.
//!   f64: (bits & 0x7FF0000000000000) != 0x7FF0000000000000
//!   f32: (bits & 0x7F800000) != 0x7F800000
//!   f16: (bits & 0x7C00) != 0x7C00
//!
//! This catches both NaN and ±Inf in a single comparison (no mantissa check needed).

const simd = @import("simd.zig");

/// isfinite for f64 — 2-wide SIMD: AND + NE (WASM has i64x2.eq).
export fn isfinite_f64(a: [*]const f64, out: [*]u8, N: u32) void {
    const a_i64: [*]const i64 = @ptrCast(a);
    const n2 = N & ~@as(u32, 1);
    var i: u32 = 0;

    const exp_v: @Vector(2, i64) = @splat(@as(i64, @bitCast(@as(u64, 0x7FF0_0000_0000_0000))));

    while (i < n2) : (i += 2) {
        const bits: @Vector(2, i64) = @as(*align(1) const @Vector(2, i64), @ptrCast(a_i64 + i)).*;
        const finite = (bits & exp_v) != exp_v;
        out[i] = @intFromBool(finite[0]);
        out[i + 1] = @intFromBool(finite[1]);
    }
    if (i < N) {
        out[i] = if ((@as(u64, @bitCast(a_i64[i])) & 0x7FF0_0000_0000_0000) != 0x7FF0_0000_0000_0000) 1 else 0;
    }
}

/// isfinite for f32 — 4-wide SIMD via u32x4 bit logic.
export fn isfinite_f32(a: [*]const f32, out: [*]u8, N: u32) void {
    const a_u32: [*]const u32 = @ptrCast(a);
    const exp_scalar: u32 = 0x7F80_0000;
    const n4 = N & ~@as(u32, 3);
    var i: u32 = 0;

    const exp_v: @Vector(4, u32) = @splat(exp_scalar);

    while (i < n4) : (i += 4) {
        const bits: @Vector(4, u32) = @as(*align(1) const @Vector(4, u32), @ptrCast(a_u32 + i)).*;
        const not_special = (bits & exp_v) != exp_v; // true where exponent is NOT all-ones
        out[i] = @intFromBool(not_special[0]);
        out[i + 1] = @intFromBool(not_special[1]);
        out[i + 2] = @intFromBool(not_special[2]);
        out[i + 3] = @intFromBool(not_special[3]);
    }
    while (i < N) : (i += 1) {
        out[i] = if ((a_u32[i] & exp_scalar) != exp_scalar) 1 else 0;
    }
}

/// isfinite for f16 (as raw u16 bit patterns) — 8-wide SIMD via u16x8 bit logic.
export fn isfinite_u16(a: [*]const u16, out: [*]u8, N: u32) void {
    const exp_scalar: u16 = 0x7C00;
    const n8 = N & ~@as(u32, 7);
    var i: u32 = 0;

    const exp_v: @Vector(8, u16) = @splat(exp_scalar);

    while (i < n8) : (i += 8) {
        const bits: @Vector(8, u16) = @as(*align(1) const @Vector(8, u16), @ptrCast(a + i)).*;
        const finite = (bits & exp_v) != exp_v;
        inline for (0..8) |lane| {
            out[i + lane] = @intFromBool(finite[lane]);
        }
    }
    while (i < N) : (i += 1) {
        out[i] = if ((a[i] & exp_scalar) != exp_scalar) 1 else 0;
    }
}

// --- Tests ---

test "isfinite_f64" {
    const testing = @import("std").testing;
    const nan: f64 = @bitCast(@as(u64, 0x7FF8000000000000));
    const inf: f64 = @bitCast(@as(u64, 0x7FF0000000000000));
    const a = [_]f64{ 1.0, nan, inf, -inf, 0.0, -3.14 };
    var out: [6]u8 = undefined;
    isfinite_f64(&a, &out, 6);
    try testing.expectEqual(out[0], 1); // 1.0 finite
    try testing.expectEqual(out[1], 0); // NaN
    try testing.expectEqual(out[2], 0); // inf
    try testing.expectEqual(out[3], 0); // -inf
    try testing.expectEqual(out[4], 1); // 0.0 finite
    try testing.expectEqual(out[5], 1); // -3.14 finite
}

test "isfinite_f32" {
    const testing = @import("std").testing;
    const nan: f32 = @bitCast(@as(u32, 0x7FC00000));
    const inf: f32 = @bitCast(@as(u32, 0x7F800000));
    const a = [_]f32{ 1.0, nan, inf, -inf, 0.0, nan, 3.0, -42.0 };
    var out: [8]u8 = undefined;
    isfinite_f32(&a, &out, 8);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], 0);
    try testing.expectEqual(out[3], 0);
    try testing.expectEqual(out[4], 1);
    try testing.expectEqual(out[5], 0);
    try testing.expectEqual(out[6], 1);
    try testing.expectEqual(out[7], 1);
}

test "isfinite_u16 (f16 bits)" {
    const testing = @import("std").testing;
    const a = [_]u16{ 0x3C00, 0x7E00, 0x7C00, 0xFC00, 0x0000, 0x7E01, 0x0001, 0x4200 };
    var out: [8]u8 = undefined;
    isfinite_u16(&a, &out, 8);
    try testing.expectEqual(out[0], 1); // 1.0
    try testing.expectEqual(out[1], 0); // NaN
    try testing.expectEqual(out[2], 0); // +inf
    try testing.expectEqual(out[3], 0); // -inf
    try testing.expectEqual(out[4], 1); // 0.0
    try testing.expectEqual(out[5], 0); // NaN
    try testing.expectEqual(out[6], 1); // subnormal
    try testing.expectEqual(out[7], 1); // 3.0
}
