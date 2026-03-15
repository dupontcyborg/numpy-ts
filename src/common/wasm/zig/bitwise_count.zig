//! WASM population count (bitwise_count) kernels for all integer types.
//!
//! Unary: out[i] = popcount(a[i])
//! For signed types, counts bits of abs(value) to match NumPy behavior.
//! For unsigned types, counts bits of the raw value.
//! Output is always u8.

const simd = @import("simd.zig");

/// Population count for a single u64 value.
inline fn popcount_u64(x: u64) u8 {
    // Parallel bit counting (Hamming weight)
    var v = x;
    v = v - ((v >> 1) & 0x5555555555555555);
    v = (v & 0x3333333333333333) + ((v >> 2) & 0x3333333333333333);
    v = (v + (v >> 4)) & 0x0F0F0F0F0F0F0F0F;
    return @intCast((v *% 0x0101010101010101) >> 56);
}

/// Population count for a single u32 value.
inline fn popcount_u32(x: u32) u8 {
    var v = x;
    v = v - ((v >> 1) & 0x55555555);
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
    v = (v + (v >> 4)) & 0x0F0F0F0F;
    return @intCast((v *% 0x01010101) >> 24);
}

/// Bitwise count for signed i64 — counts bits of abs(value).
export fn bitwise_count_i64(a: [*]const i64, out: [*]u8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const v = a[i];
        const abs_v: u64 = @bitCast(if (v < 0) -%v else v);
        out[i] = popcount_u64(abs_v);
    }
}

/// Bitwise count for unsigned u64 — counts bits of raw value.
export fn bitwise_count_u64(a: [*]const u64, out: [*]u8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = popcount_u64(a[i]);
    }
}

/// Bitwise count for signed i32 — counts bits of abs(value).
export fn bitwise_count_i32(a: [*]const i32, out: [*]u8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const v = a[i];
        const abs_v: u32 = @bitCast(if (v < 0) -%v else v);
        out[i] = popcount_u32(abs_v);
    }
}

/// Bitwise count for unsigned u32 — counts bits of raw value.
export fn bitwise_count_u32(a: [*]const u32, out: [*]u8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = popcount_u32(a[i]);
    }
}

/// Bitwise count for signed i16 — counts bits of abs(value).
export fn bitwise_count_i16(a: [*]const i16, out: [*]u8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const v = a[i];
        const abs_v: u16 = @bitCast(if (v < 0) -%v else v);
        out[i] = @popCount(abs_v);
    }
}

/// Bitwise count for unsigned u16 — counts bits of raw value.
export fn bitwise_count_u16(a: [*]const u16, out: [*]u8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = @popCount(a[i]);
    }
}

/// Bitwise count for signed i8 — counts bits of abs(value).
export fn bitwise_count_i8(a: [*]const i8, out: [*]u8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const v = a[i];
        const abs_v: u8 = @bitCast(if (v < 0) -%v else v);
        out[i] = @popCount(abs_v);
    }
}

/// Bitwise count for unsigned u8 — counts bits of raw value.
export fn bitwise_count_u8(a: [*]const u8, out: [*]u8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = @popCount(a[i]);
    }
}

// --- Tests ---

test "bitwise_count_u8 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{ 0, 1, 255, 128, 15 };
    var out: [5]u8 = undefined;
    bitwise_count_u8(&a, &out, 5);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 8);
    try testing.expectEqual(out[3], 1);
    try testing.expectEqual(out[4], 4);
}

test "bitwise_count_i8 signed" {
    const testing = @import("std").testing;
    // NumPy: bitwise_count on signed counts bits of abs(value)
    const a = [_]i8{ 0, 1, -1, -128, 127 };
    var out: [5]u8 = undefined;
    bitwise_count_i8(&a, &out, 5);
    try testing.expectEqual(out[0], 0); // abs(0) = 0
    try testing.expectEqual(out[1], 1); // abs(1) = 1 bit
    try testing.expectEqual(out[2], 1); // abs(-1) = 1 bit
    try testing.expectEqual(out[3], 1); // abs(-128) = 128 = 1 bit
    try testing.expectEqual(out[4], 7); // abs(127) = 0b1111111 = 7 bits
}

test "bitwise_count_i32 signed" {
    const testing = @import("std").testing;
    const a = [_]i32{ 0, 1, -1, 7, -2147483648 };
    var out: [5]u8 = undefined;
    bitwise_count_i32(&a, &out, 5);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 1); // abs(-1) = 1 bit
    try testing.expectEqual(out[3], 3);
    try testing.expectEqual(out[4], 1); // abs(-2147483648) = 2147483648 = 1 bit
}

test "bitwise_count_u32 basic" {
    const testing = @import("std").testing;
    const a = [_]u32{ 0, 1, 0xFFFFFFFF, 0x80000000 };
    var out: [4]u8 = undefined;
    bitwise_count_u32(&a, &out, 4);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 32);
    try testing.expectEqual(out[3], 1);
}

test "bitwise_count_i64 signed" {
    const testing = @import("std").testing;
    const a = [_]i64{ 0, -1, 255 };
    var out: [3]u8 = undefined;
    bitwise_count_i64(&a, &out, 3);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 1); // abs(-1) = 1 bit
    try testing.expectEqual(out[2], 8);
}
