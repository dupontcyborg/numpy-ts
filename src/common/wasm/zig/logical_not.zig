//! WASM element-wise logical NOT kernels for all numeric types.
//!
//! Unary: out[i] = (a[i] == 0) ? 1 : 0
//! Output is always u8 (0 or 1). Operates on contiguous 1D buffers of length N.

const simd = @import("simd.zig");

const V2 = @Vector(2, i64);
const V2Z: V2 = @splat(0);

/// Truthiness of two i64 lanes as a bool vector.
/// `@intFromBool` then yields one byte per lane, matching the bool output.
inline fn truthy2(p: [*]const i64, i: u32) @Vector(2, bool) {
    return @as(*align(1) const V2, @ptrCast(p + i)).* != V2Z;
}

inline fn store2(out: [*]u8, i: u32, m: @Vector(2, bool)) void {
    @as(*align(1) @Vector(2, u8), @ptrCast(out + i)).* = @intFromBool(m);
}

/// Element-wise logical NOT for f64: out[i] = (a[i] == 0) ? 1 : 0.
export fn logical_not_f64(a: [*]const f64, out: [*]u8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (a[i] == 0) 1 else 0;
    }
}

/// Element-wise logical NOT for f32: out[i] = (a[i] == 0) ? 1 : 0.
export fn logical_not_f32(a: [*]const f32, out: [*]u8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (a[i] == 0) 1 else 0;
    }
}

/// Float16 logical not: (u16 & 0x7FFF == 0) ? 1 : 0. Handles -0.
export fn logical_not_f16(a: [*]const u16, out: [*]u8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = if (a[i] & 0x7FFF == 0) 1 else 0;
}

/// Element-wise logical NOT for i64 — 2-wide.
export fn logical_not_i64(a: [*]const i64, out: [*]u8, N: u32) void {
    const n2 = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n2) : (i += 2) {
        const v = @as(*align(1) const V2, @ptrCast(a + i)).*;
        store2(out, i, v == V2Z);
    }
    while (i < N) : (i += 1) out[i] = @intFromBool(a[i] == 0);
}

/// Element-wise logical NOT for i32 using 4-wide SIMD: out[i] = (a[i] == 0) ? 1 : 0.
export fn logical_not_i32(a: [*]const i32, out: [*]u8, N: u32) void {
    const zero: simd.V4i32 = @splat(0);
    const one: simd.V4u32 = @splat(1);
    const zero_u32: simd.V4u32 = @splat(0);
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        const v = simd.load4_i32(a, i);
        const result: simd.V4u32 = @select(u32, v == zero, one, zero_u32);
        // Pack 4 x u32 down to 4 bytes
        out[i] = @truncate(result[0]);
        out[i + 1] = @truncate(result[1]);
        out[i + 2] = @truncate(result[2]);
        out[i + 3] = @truncate(result[3]);
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] == 0) 1 else 0;
    }
}

/// Element-wise logical NOT for i16 using 8-wide SIMD: out[i] = (a[i] == 0) ? 1 : 0.
export fn logical_not_i16(a: [*]const i16, out: [*]u8, N: u32) void {
    const zero: simd.V8i16 = @splat(0);
    const one: simd.V8u16 = @splat(1);
    const zero_u16: simd.V8u16 = @splat(0);
    const n_simd = N & ~@as(u32, 7);
    var i: u32 = 0;
    while (i < n_simd) : (i += 8) {
        const v = simd.load8_i16(a, i);
        const result: simd.V8u16 = @select(u16, v == zero, one, zero_u16);
        // Pack 8 x u16 down to 8 bytes
        inline for (0..8) |j| {
            out[i + j] = @truncate(result[j]);
        }
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] == 0) 1 else 0;
    }
}

/// Element-wise logical NOT for i8 using 16-wide SIMD: out[i] = (a[i] == 0) ? 1 : 0.
/// Input and output are both byte-width, enabling natural 16-wide vectorization.
export fn logical_not_i8(a: [*]const i8, out: [*]u8, N: u32) void {
    const zero: simd.V16i8 = @splat(0);
    const one: simd.V16u8 = @splat(1);
    const zero_u8: simd.V16u8 = @splat(0);
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    while (i < n_simd) : (i += 16) {
        simd.store16_u8(out, i, @select(u8, simd.load16_i8(a, i) == zero, one, zero_u8));
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] == 0) 1 else 0;
    }
}

// --- Tests ---

test "logical_not_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0.0, 1.5, 0.0, -2.0 };
    var out: [4]u8 = undefined;
    logical_not_f64(&a, &out, 4);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 0);
}

test "logical_not_i8 large SIMD" {
    const testing = @import("std").testing;
    var a: [20]i8 = undefined;
    for (0..20) |idx| {
        a[idx] = if (idx % 3 == 0) 0 else @intCast(idx);
    }
    var out: [20]u8 = undefined;
    logical_not_i8(&a, &out, 20);
    for (0..20) |idx| {
        const expected: u8 = if (idx % 3 == 0) 1 else 0;
        try testing.expectEqual(out[idx], expected);
    }
}

test "logical_not_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 0, 5, -3, 0, 7 };
    var out: [5]u8 = undefined;
    logical_not_i32(&a, &out, 5);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], 0);
    try testing.expectEqual(out[3], 1);
    try testing.expectEqual(out[4], 0);
}

test "logical_not_f64 edge zero types" {
    const testing = @import("std").testing;
    const a = [_]f64{ -0.0, 0.0, 1e-300 };
    var out: [3]u8 = undefined;
    logical_not_f64(&a, &out, 3);
    try testing.expectEqual(out[0], 1); // -0.0 == 0
    try testing.expectEqual(out[1], 1); // 0.0 == 0
    try testing.expectEqual(out[2], 0); // tiny but nonzero
}

test "logical_not_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 0.0, 1.0, -1.0, 0.0, 0.5 };
    var out: [5]u8 = undefined;
    logical_not_f32(&a, &out, 5);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], 0);
    try testing.expectEqual(out[3], 1);
    try testing.expectEqual(out[4], 0);
}

test "logical_not_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 0, 1, -1, 0, 100 };
    var out: [5]u8 = undefined;
    logical_not_i64(&a, &out, 5);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], 0);
    try testing.expectEqual(out[3], 1);
    try testing.expectEqual(out[4], 0);
}

test "logical_not_i16 SIMD boundary N=9" {
    const testing = @import("std").testing;
    const a = [_]i16{ 0, 1, 0, -1, 0, 5, 0, -3, 7 };
    var out: [9]u8 = undefined;
    logical_not_i16(&a, &out, 9);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 0);
    try testing.expectEqual(out[4], 1);
    try testing.expectEqual(out[8], 0);
}

test "logical_not_f16 basic" {
    const testing = @import("std").testing;
    // 1.0=0x3C00, 0.0=0x0000, -0.0=0x8000, -1.0=0xBC00
    const a = [_]u16{ 0x3C00, 0x0000, 0x8000, 0xBC00 };
    var out: [4]u8 = undefined;
    logical_not_f16(&a, &out, 4);
    try testing.expectEqual(out[0], 0); // !1.0 = 0
    try testing.expectEqual(out[1], 1); // !0.0 = 1
    try testing.expectEqual(out[2], 1); // !(-0.0) = 1
    try testing.expectEqual(out[3], 0); // !(-1.0) = 0
}

test "logical_not_i64 odd length exercises the 2-wide body and the tail" {
    const testing = @import("std").testing;
    const MIN = @import("std").math.minInt(i64);
    const MAX = @import("std").math.maxInt(i64);
    const a = [_]i64{ 0, 1, MIN, MAX, 0, -1, 0 };
    var out = [_]u8{9} ** 7;
    logical_not_i64(&a, &out, 7);
    try testing.expectEqualSlices(u8, &[_]u8{ 1, 0, 0, 0, 1, 0, 1 }, &out);
}
