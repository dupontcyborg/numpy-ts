//! WASM batched vector dot product kernels for integer types (i64, i32, i16, i8).
//!
//! Computes out[i] = sum_k a[i*K+k] * b[i*K+k] for batch size B and vector length K.
//! Both a and b are accessed sequentially — ideal for SIMD.
//! This is the "paired" dot product (vecdot), not the all-pairs inner product.

const simd = @import("simd.zig");

/// Computes the dot product of B pairs of i64 vectors of length K.
/// out[i] = sum_k a[i*K+k] * b[i*K+k]
/// Handles both signed (i64) and unsigned (u64) with wrapping arithmetic.
export fn vecdot_i64(a: [*]const i64, b: [*]const i64, out: [*]i64, B: u32, K: u32) void {
    const k_simd = K & ~@as(u32, 1); // floor to V2i64 (2-wide)
    for (0..B) |i| {
        const row = i * K;
        var acc: simd.V2i64 = @splat(0);
        var k: u32 = 0;
        // SIMD loop: 2 i64s per iteration
        while (k < k_simd) : (k += 2) {
            acc +%= simd.load2_i64(a, row + k) *% simd.load2_i64(b, row + k);
        }
        // Horizontal sum + scalar remainder
        var sum: i64 = acc[0] +% acc[1];
        while (k < K) : (k += 1) {
            sum +%= a[row + k] *% b[row + k];
        }
        out[i] = sum;
    }
}

/// Computes the dot product of B pairs of i32 vectors of length K.
/// out[i] = sum_k a[i*K+k] * b[i*K+k]
/// Handles both signed (i32) and unsigned (u32) with wrapping arithmetic.
export fn vecdot_i32(a: [*]const i32, b: [*]const i32, out: [*]i32, B: u32, K: u32) void {
    const k_simd = K & ~@as(u32, 3); // floor to V4i32 (4-wide)
    for (0..B) |i| {
        const row = i * K;
        var acc: simd.V4i32 = @splat(0);
        var k: u32 = 0;
        // SIMD loop: 4 i32s per iteration
        while (k < k_simd) : (k += 4) {
            acc +%= simd.load4_i32(a, row + k) *% simd.load4_i32(b, row + k);
        }
        // Horizontal sum + scalar remainder
        var sum: i32 = acc[0] +% acc[1] +% acc[2] +% acc[3];
        while (k < K) : (k += 1) {
            sum +%= a[row + k] *% b[row + k];
        }
        out[i] = sum;
    }
}

/// Computes the dot product of B pairs of i16 vectors of length K.
/// out[i] = sum_k a[i*K+k] * b[i*K+k]
/// Handles both signed (i16) and unsigned (u16) with wrapping arithmetic.
/// Uses i32x4.dot_i16x8_s for pairwise multiply-accumulate into i32.
export fn vecdot_i16(a: [*]const i16, b: [*]const i16, out: [*]i16, B: u32, K: u32) void {
    const k_simd = K & ~@as(u32, 7); // floor to V8i16 (8-wide)
    for (0..B) |i| {
        const row = i * K;
        var acc: simd.V4i32 = @splat(0);
        var k: u32 = 0;
        // SIMD loop: 8 i16s per iteration via dot_i16x8_s
        while (k < k_simd) : (k += 8) {
            acc +%= simd.dot_i16x8_s(simd.load8_i16(a, row + k), simd.load8_i16(b, row + k));
        }
        // Horizontal sum + scalar remainder
        var sum: i32 = acc[0] +% acc[1] +% acc[2] +% acc[3];
        while (k < K) : (k += 1) {
            sum +%= @as(i32, a[row + k]) *% @as(i32, b[row + k]);
        }
        out[i] = @truncate(sum);
    }
}

/// Computes the dot product of B pairs of i8 vectors of length K.
/// out[i] = sum_k a[i*K+k] * b[i*K+k]
/// Handles both signed (i8) and unsigned (u8) with wrapping arithmetic.
/// Widens i8→i16, uses i32x4.dot_i16x8_s for pairwise accumulation into i32.
export fn vecdot_i8(a: [*]const i8, b: [*]const i8, out: [*]i8, B: u32, K: u32) void {
    const k_simd = K & ~@as(u32, 15); // floor to V16i8 (16-wide)
    for (0..B) |i| {
        const row = i * K;
        var acc: simd.V4i32 = @splat(0);
        var k: u32 = 0;
        // SIMD loop: 16 i8s per iteration via widening dot into i32
        while (k < k_simd) : (k += 16) {
            acc +%= simd.dot_i8x16_to_i32x4(simd.load16_i8(a, row + k), simd.load16_i8(b, row + k));
        }
        // Horizontal sum + scalar remainder
        var sum: i32 = acc[0] +% acc[1] +% acc[2] +% acc[3];
        while (k < K) : (k += 1) {
            sum +%= @as(i32, a[row + k]) *% @as(i32, b[row + k]);
        }
        out[i] = @truncate(sum);
    }
}

// --- Tests ---

test "vecdot_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 1, 2, 3, 4, 5, 6 };
    const b = [_]i64{ 7, 8, 9, 10, 11, 12 };
    var out: [3]i64 = undefined;
    vecdot_i64(&a, &b, &out, 3, 2);
    try testing.expectEqual(out[0], 23);
    try testing.expectEqual(out[1], 67);
    try testing.expectEqual(out[2], 127);
}

test "vecdot_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 1, 2, 3, 4, 5, 6 };
    const b = [_]i32{ 7, 8, 9, 10, 11, 12 };
    var out: [3]i32 = undefined;
    vecdot_i32(&a, &b, &out, 3, 2);
    try testing.expectEqual(out[0], 23);
    try testing.expectEqual(out[1], 67);
    try testing.expectEqual(out[2], 127);
}

test "vecdot_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{ 1, 2, 3, 4, 5, 6 };
    const b = [_]i16{ 7, 8, 9, 10, 11, 12 };
    var out: [3]i16 = undefined;
    vecdot_i16(&a, &b, &out, 3, 2);
    try testing.expectEqual(out[0], 23);
    try testing.expectEqual(out[1], 67);
    try testing.expectEqual(out[2], 127);
}

test "vecdot_i8 wrapping" {
    const testing = @import("std").testing;
    const a = [_]i8{ 10, 10, 10, 10 };
    const b = [_]i8{ 10, 10, 10, 10 };
    var out: [2]i8 = undefined;
    vecdot_i8(&a, &b, &out, 2, 2);
    const expected: i8 = @truncate(@as(i32, 10) * 10 * 2);
    try testing.expectEqual(out[0], expected);
    try testing.expectEqual(out[1], expected);
}
