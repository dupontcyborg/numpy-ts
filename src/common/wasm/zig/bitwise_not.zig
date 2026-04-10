//! WASM element-wise bitwise NOT kernels for all integer types.
//!
//! Unary: out[i] = ~a[i]
//! Operates on contiguous 1D buffers of length N.

const simd = @import("simd.zig");

/// Element-wise bitwise NOT for i64 using 2-wide SIMD.
/// Handles both signed (i64) and unsigned (u64).
export fn bitwise_not_i64(a: [*]const i64, out: [*]i64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_i64(out, i, ~simd.load2_i64(a, i));
    }
    while (i < N) : (i += 1) {
        out[i] = ~a[i];
    }
}

/// Element-wise bitwise NOT for i32 using 4-wide SIMD.
/// Handles both signed (i32) and unsigned (u32).
export fn bitwise_not_i32(a: [*]const i32, out: [*]i32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_i32(out, i, ~simd.load4_i32(a, i));
    }
    while (i < N) : (i += 1) {
        out[i] = ~a[i];
    }
}

/// Element-wise bitwise NOT for i16 using 8-wide SIMD.
/// Handles both signed (i16) and unsigned (u16).
export fn bitwise_not_i16(a: [*]const i16, out: [*]i16, N: u32) void {
    const n_simd = N & ~@as(u32, 7);
    var i: u32 = 0;
    while (i < n_simd) : (i += 8) {
        simd.store8_i16(out, i, ~simd.load8_i16(a, i));
    }
    while (i < N) : (i += 1) {
        out[i] = ~a[i];
    }
}

/// Element-wise bitwise NOT for i8 using 16-wide SIMD.
/// Handles both signed (i8) and unsigned (u8).
export fn bitwise_not_i8(a: [*]const i8, out: [*]i8, N: u32) void {
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    while (i < n_simd) : (i += 16) {
        simd.store16_i8(out, i, ~simd.load16_i8(a, i));
    }
    while (i < N) : (i += 1) {
        out[i] = ~a[i];
    }
}

// --- Tests ---

test "bitwise_not_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 0, -1, 0x0F, 0x55 };
    var out: [4]i32 = undefined;
    bitwise_not_i32(&a, &out, 4);
    try testing.expectEqual(out[0], -1);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], ~@as(i32, 0x0F));
    try testing.expectEqual(out[3], ~@as(i32, 0x55));
}

test "bitwise_not_i8 SIMD boundary N=17" {
    const testing = @import("std").testing;
    var a: [17]i8 = undefined;
    for (0..17) |idx| {
        a[idx] = @intCast(idx);
    }
    var out: [17]i8 = undefined;
    bitwise_not_i8(&a, &out, 17);
    for (0..17) |idx| {
        const expected: i8 = ~@as(i8, @intCast(idx));
        try testing.expectEqual(out[idx], expected);
    }
}

test "bitwise_not_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 0, -1, 0xFF };
    var out: [3]i64 = undefined;
    bitwise_not_i64(&a, &out, 3);
    try testing.expectEqual(out[0], -1);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], ~@as(i64, 0xFF));
}

test "bitwise_not_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{0};
    var out: [1]i16 = undefined;
    bitwise_not_i16(&a, &out, 1);
    try testing.expectEqual(out[0], -1);
}
