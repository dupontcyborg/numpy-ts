//! WASM element-wise right shift kernels for all integer types.
//!
//! Binary: out[i] = a[i] >> b[i]  (arithmetic shift for signed)
//! Scalar: out[i] = a[i] >> scalar
//! Both operate on contiguous 1D buffers of length N.

const simd = @import("simd.zig");

/// Element-wise right shift for i64: out[i] = a[i] >> b[i].
/// No SIMD — WASM has no variable-shift for i64x2.
export fn right_shift_i64(a: [*]const i64, b: [*]const i64, out: [*]i64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const shift: u6 = @intCast(@as(u64, @bitCast(b[i])) & 63);
        out[i] = a[i] >> shift;
    }
}

/// Element-wise right shift scalar for i64: out[i] = a[i] >> scalar.
export fn right_shift_scalar_i64(a: [*]const i64, out: [*]i64, N: u32, scalar: i64) void {
    const shift: u6 = @intCast(@as(u64, @bitCast(scalar)) & 63);
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = a[i] >> shift;
    }
}

/// Element-wise right shift for i32: out[i] = a[i] >> b[i].
/// Scalar loop — WASM SIMD has no per-lane variable shift.
export fn right_shift_i32(a: [*]const i32, b: [*]const i32, out: [*]i32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const shift: u5 = @intCast(@as(u32, @bitCast(b[i])) & 31);
        out[i] = a[i] >> shift;
    }
}

/// Element-wise right shift scalar for i32 using 4-wide SIMD: out[i] = a[i] >> scalar.
export fn right_shift_scalar_i32(a: [*]const i32, out: [*]i32, N: u32, scalar: i32) void {
    const shift: u5 = @intCast(@as(u32, @bitCast(scalar)) & 31);
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_i32(out, i, simd.load4_i32(a, i) >> @splat(shift));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] >> shift;
    }
}

/// Element-wise right shift for i16: out[i] = a[i] >> b[i].
/// Scalar loop — WASM SIMD has no per-lane variable shift for i16x8.
export fn right_shift_i16(a: [*]const i16, b: [*]const i16, out: [*]i16, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const shift: u4 = @intCast(@as(u16, @bitCast(b[i])) & 15);
        out[i] = a[i] >> shift;
    }
}

/// Element-wise right shift scalar for i16 using 8-wide SIMD: out[i] = a[i] >> scalar.
export fn right_shift_scalar_i16(a: [*]const i16, out: [*]i16, N: u32, scalar: i16) void {
    const shift: u4 = @intCast(@as(u16, @bitCast(scalar)) & 15);
    const n_simd = N & ~@as(u32, 7);
    var i: u32 = 0;
    while (i < n_simd) : (i += 8) {
        simd.store8_i16(out, i, simd.load8_i16(a, i) >> @splat(shift));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] >> shift;
    }
}

/// Element-wise right shift for i8: out[i] = a[i] >> b[i].
/// Scalar loop — WASM SIMD has no per-lane variable shift for i8x16.
export fn right_shift_i8(a: [*]const i8, b: [*]const i8, out: [*]i8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const shift: u3 = @intCast(@as(u8, @bitCast(b[i])) & 7);
        out[i] = a[i] >> shift;
    }
}

/// Element-wise right shift scalar for i8 using 16-wide SIMD: out[i] = a[i] >> scalar.
export fn right_shift_scalar_i8(a: [*]const i8, out: [*]i8, N: u32, scalar: i8) void {
    const shift: u3 = @intCast(@as(u8, @bitCast(scalar)) & 7);
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    while (i < n_simd) : (i += 16) {
        simd.store16_i8(out, i, simd.load16_i8(a, i) >> @splat(shift));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] >> shift;
    }
}

// --- Unsigned right shift (logical shift) kernels ---

/// Element-wise logical right shift for u64: out[i] = a[i] >>> b[i].
export fn right_shift_u64(a: [*]const u64, b: [*]const u64, out: [*]u64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const shift: u6 = @intCast(b[i] & 63);
        out[i] = a[i] >> shift;
    }
}

/// Element-wise logical right shift scalar for u64: out[i] = a[i] >>> scalar.
export fn right_shift_scalar_u64(a: [*]const u64, out: [*]u64, N: u32, scalar: u64) void {
    const shift: u6 = @intCast(scalar & 63);
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = a[i] >> shift;
    }
}

/// Element-wise logical right shift for u32 using 4-wide SIMD: out[i] = a[i] >>> scalar.
export fn right_shift_u32(a: [*]const u32, b: [*]const u32, out: [*]u32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const shift: u5 = @intCast(b[i] & 31);
        out[i] = a[i] >> shift;
    }
}

/// Element-wise logical right shift scalar for u32 using 4-wide SIMD.
export fn right_shift_scalar_u32(a: [*]const u32, out: [*]u32, N: u32, scalar: u32) void {
    const shift: u5 = @intCast(scalar & 31);
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_u32(out, i, simd.load4_u32(a, i) >> @splat(shift));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] >> shift;
    }
}

/// Element-wise logical right shift for u16.
export fn right_shift_u16(a: [*]const u16, b: [*]const u16, out: [*]u16, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const shift: u4 = @intCast(b[i] & 15);
        out[i] = a[i] >> shift;
    }
}

/// Element-wise logical right shift scalar for u16 using 8-wide SIMD.
export fn right_shift_scalar_u16(a: [*]const u16, out: [*]u16, N: u32, scalar: u16) void {
    const shift: u4 = @intCast(scalar & 15);
    const n_simd = N & ~@as(u32, 7);
    var i: u32 = 0;
    while (i < n_simd) : (i += 8) {
        simd.store8_u16(out, i, simd.load8_u16(a, i) >> @splat(shift));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] >> shift;
    }
}

/// Element-wise logical right shift for u8.
export fn right_shift_u8(a: [*]const u8, b: [*]const u8, out: [*]u8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const shift: u3 = @intCast(b[i] & 7);
        out[i] = a[i] >> shift;
    }
}

/// Element-wise logical right shift scalar for u8 using 16-wide SIMD.
export fn right_shift_scalar_u8(a: [*]const u8, out: [*]u8, N: u32, scalar: u8) void {
    const shift: u3 = @intCast(scalar & 7);
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    while (i < n_simd) : (i += 16) {
        simd.store16_u8(out, i, simd.load16_u8(a, i) >> @splat(shift));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] >> shift;
    }
}

// --- Tests ---

test "right_shift_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 16, 32, 64, -8, 0 };
    const b = [_]i32{ 1, 2, 3, 1, 4 };
    var out: [5]i32 = undefined;
    right_shift_i32(&a, &b, &out, 5);
    try testing.expectEqual(out[0], 8);
    try testing.expectEqual(out[1], 8);
    try testing.expectEqual(out[2], 8);
    try testing.expectEqual(out[3], -4); // arithmetic shift
    try testing.expectEqual(out[4], 0);
}

test "right_shift_scalar_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 4, 8, 16, 32, 64, 128, 256 };
    var out: [7]i32 = undefined;
    right_shift_scalar_i32(&a, &out, 7, 2);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 2);
    try testing.expectEqual(out[2], 4);
    try testing.expectEqual(out[3], 8);
    try testing.expectEqual(out[4], 16);
    try testing.expectEqual(out[5], 32);
    try testing.expectEqual(out[6], 64);
}

test "right_shift_scalar_i8 SIMD boundary N=17" {
    const testing = @import("std").testing;
    var a: [17]i8 = undefined;
    for (0..17) |idx| {
        a[idx] = @intCast(idx * 4);
    }
    var out: [17]i8 = undefined;
    right_shift_scalar_i8(&a, &out, 17, 2);
    for (0..17) |idx| {
        const v: i8 = @intCast(idx * 4);
        try testing.expectEqual(out[idx], v >> 2);
    }
}

test "right_shift_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 64, -8 };
    const b = [_]i64{ 2, 1 };
    var out: [2]i64 = undefined;
    right_shift_i64(&a, &b, &out, 2);
    try testing.expectEqual(out[0], 16);
    try testing.expectEqual(out[1], -4); // arithmetic shift
}

test "right_shift_scalar_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 64, 128, -32 };
    var out: [3]i64 = undefined;
    right_shift_scalar_i64(&a, &out, 3, 2);
    try testing.expectEqual(out[0], 16);
    try testing.expectEqual(out[1], 32);
    try testing.expectEqual(out[2], -8);
}

test "right_shift_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{ 64, -8 };
    const b = [_]i16{ 2, 1 };
    var out: [2]i16 = undefined;
    right_shift_i16(&a, &b, &out, 2);
    try testing.expectEqual(out[0], 16);
    try testing.expectEqual(out[1], -4);
}

test "right_shift_scalar_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{ 64, 128 };
    var out: [2]i16 = undefined;
    right_shift_scalar_i16(&a, &out, 2, 2);
    try testing.expectEqual(out[0], 16);
    try testing.expectEqual(out[1], 32);
}

test "right_shift_i8 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{ 64, -8 };
    const b = [_]i8{ 2, 1 };
    var out: [2]i8 = undefined;
    right_shift_i8(&a, &b, &out, 2);
    try testing.expectEqual(out[0], 16);
    try testing.expectEqual(out[1], -4);
}

test "right_shift_u64 basic" {
    const testing = @import("std").testing;
    const a = [_]u64{ 64, 128 };
    const b = [_]u64{ 2, 3 };
    var out: [2]u64 = undefined;
    right_shift_u64(&a, &b, &out, 2);
    try testing.expectEqual(out[0], 16);
    try testing.expectEqual(out[1], 16);
}

test "right_shift_scalar_u64 basic" {
    const testing = @import("std").testing;
    const a = [_]u64{ 64, 128 };
    var out: [2]u64 = undefined;
    right_shift_scalar_u64(&a, &out, 2, 2);
    try testing.expectEqual(out[0], 16);
    try testing.expectEqual(out[1], 32);
}

test "right_shift_u32 basic" {
    const testing = @import("std").testing;
    const a = [_]u32{ 64, 128 };
    const b = [_]u32{ 2, 3 };
    var out: [2]u32 = undefined;
    right_shift_u32(&a, &b, &out, 2);
    try testing.expectEqual(out[0], 16);
    try testing.expectEqual(out[1], 16);
}

test "right_shift_scalar_u32 basic" {
    const testing = @import("std").testing;
    const a = [_]u32{ 64, 128 };
    var out: [2]u32 = undefined;
    right_shift_scalar_u32(&a, &out, 2, 2);
    try testing.expectEqual(out[0], 16);
    try testing.expectEqual(out[1], 32);
}

test "right_shift_u16 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{ 64, 128 };
    const b = [_]u16{ 2, 3 };
    var out: [2]u16 = undefined;
    right_shift_u16(&a, &b, &out, 2);
    try testing.expectEqual(out[0], 16);
    try testing.expectEqual(out[1], 16);
}

test "right_shift_scalar_u16 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{ 64, 128 };
    var out: [2]u16 = undefined;
    right_shift_scalar_u16(&a, &out, 2, 2);
    try testing.expectEqual(out[0], 16);
    try testing.expectEqual(out[1], 32);
}

test "right_shift_u8 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{ 64, 128 };
    const b = [_]u8{ 2, 3 };
    var out: [2]u8 = undefined;
    right_shift_u8(&a, &b, &out, 2);
    try testing.expectEqual(out[0], 16);
    try testing.expectEqual(out[1], 16);
}

test "right_shift_scalar_u8 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{ 64, 128 };
    var out: [2]u8 = undefined;
    right_shift_scalar_u8(&a, &out, 2, 2);
    try testing.expectEqual(out[0], 16);
    try testing.expectEqual(out[1], 32);
}
