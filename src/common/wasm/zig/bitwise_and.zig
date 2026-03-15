//! WASM element-wise bitwise AND kernels for all integer types.
//!
//! Binary: out[i] = a[i] & b[i]
//! Both operate on contiguous 1D buffers of length N.

const simd = @import("simd.zig");

/// Element-wise bitwise AND for i64 using 2-wide SIMD.
/// Handles both signed (i64) and unsigned (u64).
export fn bitwise_and_i64(a: [*]const i64, b: [*]const i64, out: [*]i64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_i64(out, i, simd.load2_i64(a, i) & simd.load2_i64(b, i));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] & b[i];
    }
}

/// Element-wise bitwise AND for i32 using 4-wide SIMD.
/// Handles both signed (i32) and unsigned (u32).
export fn bitwise_and_i32(a: [*]const i32, b: [*]const i32, out: [*]i32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_i32(out, i, simd.load4_i32(a, i) & simd.load4_i32(b, i));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] & b[i];
    }
}

/// Element-wise bitwise AND for i16 using 8-wide SIMD.
/// Handles both signed (i16) and unsigned (u16).
export fn bitwise_and_i16(a: [*]const i16, b: [*]const i16, out: [*]i16, N: u32) void {
    const n_simd = N & ~@as(u32, 7);
    var i: u32 = 0;
    while (i < n_simd) : (i += 8) {
        simd.store8_i16(out, i, simd.load8_i16(a, i) & simd.load8_i16(b, i));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] & b[i];
    }
}

/// Element-wise bitwise AND for i8 using 16-wide SIMD.
/// Handles both signed (i8) and unsigned (u8).
export fn bitwise_and_i8(a: [*]const i8, b: [*]const i8, out: [*]i8, N: u32) void {
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    while (i < n_simd) : (i += 16) {
        simd.store16_i8(out, i, simd.load16_i8(a, i) & simd.load16_i8(b, i));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] & b[i];
    }
}

// --- Tests ---

test "bitwise_and_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 0xFF, 0x0F, 0xAA, 0x55, 0x00 };
    const b = [_]i32{ 0x0F, 0xFF, 0x55, 0xAA, 0xFF };
    var out: [5]i32 = undefined;
    bitwise_and_i32(&a, &b, &out, 5);
    try testing.expectEqual(out[0], 0x0F);
    try testing.expectEqual(out[1], 0x0F);
    try testing.expectEqual(out[2], 0x00);
    try testing.expectEqual(out[3], 0x00);
    try testing.expectEqual(out[4], 0x00);
}

test "bitwise_and_i8 SIMD boundary N=17" {
    const testing = @import("std").testing;
    var a: [17]i8 = undefined;
    var b: [17]i8 = undefined;
    for (0..17) |idx| {
        a[idx] = @intCast(idx);
        b[idx] = 0x0F;
    }
    var out: [17]i8 = undefined;
    bitwise_and_i8(&a, &b, &out, 17);
    for (0..17) |idx| {
        const expected: i8 = @intCast(idx & 0x0F);
        try testing.expectEqual(out[idx], expected);
    }
}

test "bitwise_and_i16 SIMD boundary N=9" {
    const testing = @import("std").testing;
    var a: [9]i16 = undefined;
    var b: [9]i16 = undefined;
    for (0..9) |idx| {
        a[idx] = @intCast(idx * 0x111);
        b[idx] = 0x0FF;
    }
    var out: [9]i16 = undefined;
    bitwise_and_i16(&a, &b, &out, 9);
    for (0..9) |idx| {
        const expected: i16 = @intCast((idx * 0x111) & 0x0FF);
        try testing.expectEqual(out[idx], expected);
    }
}

test "bitwise_and_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 0xFF00FF, -1, 0 };
    const b = [_]i64{ 0x00FF00, -1, -1 };
    var out: [3]i64 = undefined;
    bitwise_and_i64(&a, &b, &out, 3);
    try testing.expectEqual(out[0], 0x000000);
    try testing.expectEqual(out[1], -1);
    try testing.expectEqual(out[2], 0);
}
