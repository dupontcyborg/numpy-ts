//! WASM 1D dot product kernels for integer types (i64, i32, i16, i8).
//!
//! Computes out[0] = sum_k a[k] * b[k] for vectors of length K..
//! Both a and b are accessed sequentially — ideal for SIMD.

const simd = @import("simd.zig");

/// Computes the dot product of two i64 vectors of length K.
/// out[0] = sum_k a[k] * b[k]
/// Handles both signed (i64) and unsigned (u64) with wrapping arithmetic.
export fn dot_i64(a: [*]const i64, b: [*]const i64, out: [*]i64, K: u32) void {
    const k_simd = K & ~@as(u32, 1); // floor to V2i64 (2-wide)
    var acc: simd.V2i64 = @splat(0);

    // SIMD loop: 2 i64s per iteration
    var k: u32 = 0;
    while (k < k_simd) : (k += 2) {
        acc +%= simd.load2_i64(a, k) *% simd.load2_i64(b, k);
    }

    // Horizontal sum + scalar remainder
    var sum: i64 = acc[0] +% acc[1];
    while (k < K) : (k += 1) {
        sum +%= a[k] *% b[k];
    }
    out[0] = sum;
}

/// Computes the dot product of two i64 vectors of length K.
/// out[0] = sum_k a[k] * b[k]
/// Handles both signed (i32) and unsigned (u32) with wrapping arithmetic.
export fn dot_i32(a: [*]const i32, b: [*]const i32, out: [*]i32, K: u32) void {
    const k_simd = K & ~@as(u32, 3); // floor to V4i32 (4-wide)
    var acc: simd.V4i32 = @splat(0);

    // SIMD loop: 4 i32s per iteration
    var k: u32 = 0;
    while (k < k_simd) : (k += 4) {
        acc +%= simd.load4_i32(a, k) *% simd.load4_i32(b, k);
    }

    // Horizontal sum + scalar remainder
    var sum: i32 = acc[0] +% acc[1] +% acc[2] +% acc[3];
    while (k < K) : (k += 1) {
        sum +%= a[k] *% b[k];
    }
    out[0] = sum;
}

/// Computes the dot product of two i16 vectors of length K.
/// out[0] = sum_k a[k] * b[k]
/// Handles both signed (i16) and unsigned (u16) with wrapping arithmetic.
/// Uses i32x4.dot_i16x8_s for pairwise multiply-accumulate into i32.
export fn dot_i16(a: [*]const i16, b: [*]const i16, out: [*]i16, K: u32) void {
    const k_simd = K & ~@as(u32, 7); // floor to V8i16 (8-wide)
    var acc: simd.V4i32 = @splat(0);

    // SIMD loop: 8 i16s per iteration via dot_i16x8_s
    var k: u32 = 0;
    while (k < k_simd) : (k += 8) {
        acc +%= simd.dot_i16x8_s(simd.load8_i16(a, k), simd.load8_i16(b, k));
    }

    // Horizontal sum + scalar remainder
    var sum: i32 = acc[0] +% acc[1] +% acc[2] +% acc[3];
    while (k < K) : (k += 1) {
        sum +%= @as(i32, a[k]) *% @as(i32, b[k]);
    }
    out[0] = @truncate(sum);
}

/// Computes the dot product of two i8 vectors of length K.
/// out[0] = sum_k a[k] * b[k]
/// Handles both signed (i8) and unsigned (u8) with wrapping arithmetic.
/// Widens i8→i16, uses i32x4.dot_i16x8_s for pairwise accumulation into i32.
export fn dot_i8(a: [*]const i8, b: [*]const i8, out: [*]i8, K: u32) void {
    const k_simd = K & ~@as(u32, 15); // floor to V16i8 (16-wide)
    var acc: simd.V4i32 = @splat(0);

    // SIMD loop: 16 i8s per iteration via widening dot into i32
    var k: u32 = 0;
    while (k < k_simd) : (k += 16) {
        acc +%= simd.dot_i8x16_to_i32x4(simd.load16_i8(a, k), simd.load16_i8(b, k));
    }

    // Horizontal sum + scalar remainder
    var sum: i32 = acc[0] +% acc[1] +% acc[2] +% acc[3];
    while (k < K) : (k += 1) {
        sum +%= @as(i32, a[k]) *% @as(i32, b[k]);
    }
    out[0] = @truncate(sum);
}

// --- Tests ---

test "dot_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 1, 2, 3 };
    const b = [_]i64{ 4, 5, 6 };
    var out: [1]i64 = undefined;
    dot_i64(&a, &b, &out, 3);
    // 1*4 + 2*5 + 3*6 = 32
    try testing.expectEqual(out[0], 32);
}

test "dot_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 1, 2, 3 };
    const b = [_]i32{ 4, 5, 6 };
    var out: [1]i32 = undefined;
    dot_i32(&a, &b, &out, 3);
    try testing.expectEqual(out[0], 32);
}

test "dot_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const b = [_]i16{ 8, 7, 6, 5, 4, 3, 2, 1 };
    var out: [1]i16 = undefined;
    dot_i16(&a, &b, &out, 8);
    // 1*8+2*7+3*6+4*5+5*4+6*3+7*2+8*1 = 120
    try testing.expectEqual(out[0], 120);
}

test "dot_i8 wrapping" {
    const testing = @import("std").testing;
    const a = [_]i8{ 10, 10, 10, 10 };
    const b = [_]i8{ 10, 10, 10, 10 };
    var out: [1]i8 = undefined;
    dot_i8(&a, &b, &out, 4);
    const expected: i8 = @truncate(@as(i32, 10) * 10 * 4);
    try testing.expectEqual(out[0], expected);
}
