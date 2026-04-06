//! WASM reduction count-nonzero kernels for all numeric types.
//!
//! Reduction: result = count of a[i] != 0 for i in 0..N
//! No unsigned variants needed — non-zero check is sign-agnostic.

const simd = @import("simd.zig");

/// Returns the count of non-zero f64 elements.
/// Note: NaN is considered non-zero, so reduce_count_nz_f64([NaN], 1) returns 1.
export fn reduce_count_nz_f64(a: [*]const f64, N: u32) u32 {
    var count: u32 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (a[i] != 0) count += 1;
    }
    return count;
}

/// Returns the count of non-zero f32 elements.
/// Note: NaN is considered non-zero, so reduce_count_nz_f32([NaN], 1) returns 1.
export fn reduce_count_nz_f32(a: [*]const f32, N: u32) u32 {
    var count: u32 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (a[i] != 0) count += 1;
    }
    return count;
}

/// Returns the count of non-zero i64 elements.
/// Handles both signed (i64) and unsigned (u64).
export fn reduce_count_nz_i64(a: [*]const i64, N: u32) u32 {
    var count: u32 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (a[i] != 0) count += 1;
    }
    return count;
}

/// Returns the count of non-zero i32 elements.
/// Handles both signed (i32) and unsigned (u32).
export fn reduce_count_nz_i32(a: [*]const i32, N: u32) u32 {
    const zero_v: simd.V4i32 = @splat(0);
    const ones: simd.V4i32 = @splat(1);
    var acc: simd.V4i32 = @splat(0);
    const n_simd = N & ~@as(u32, 3);
    var k: u32 = 0;
    while (k < n_simd) : (k += 4) {
        const v = simd.load4_i32(a, k);
        acc += @select(i32, v != zero_v, ones, zero_v);
    }
    var count: u32 = @intCast(acc[0] + acc[1] + acc[2] + acc[3]);
    // Scalar tail
    while (k < N) : (k += 1) {
        if (a[k] != 0) count += 1;
    }
    return count;
}

/// Returns the count of non-zero i16 elements.
/// Handles both signed (i16) and unsigned (u16).
/// Accumulates 0/1 mask in V8i16, horizontal sum at end (safe for N < 262144).
export fn reduce_count_nz_i16(a: [*]const i16, N: u32) u32 {
    const zero_v: simd.V8i16 = @splat(0);
    const ones: simd.V8i16 = @splat(1);
    var acc: simd.V8i16 = @splat(0);
    const n_simd = N & ~@as(u32, 7);
    var k: u32 = 0;
    while (k < n_simd) : (k += 8) {
        acc +%= @select(i16, simd.load8_i16(a, k) != zero_v, ones, zero_v);
    }
    var count: u32 = 0;
    inline for (0..8) |lane| {
        count += @intCast(@as(u16, @bitCast(acc[lane])));
    }
    while (k < N) : (k += 1) {
        if (a[k] != 0) count += 1;
    }
    return count;
}

/// Returns the count of non-zero i8 elements.
/// Handles both signed (i8) and unsigned (u8).
/// Accumulates in V16u8 with periodic drain to avoid u8 overflow.
export fn reduce_count_nz_i8(a: [*]const i8, N: u32) u32 {
    const zero_v: simd.V16i8 = @splat(0);
    const ones_i8: simd.V16i8 = @splat(1);
    const n_simd = N & ~@as(u32, 15);
    var count: u32 = 0;
    var k: u32 = 0;
    while (k < n_simd) {
        var acc: simd.V16u8 = @splat(0);
        // Drain every 255 iterations to avoid u8 overflow
        const batch_end = @min(k + 255 * 16, n_simd);
        while (k < batch_end) : (k += 16) {
            acc +%= @bitCast(@select(i8, simd.load16_i8(a, k) != zero_v, ones_i8, zero_v));
        }
        inline for (0..16) |lane| {
            count += acc[lane];
        }
    }
    while (k < N) : (k += 1) {
        if (a[k] != 0) count += 1;
    }
    return count;
}

// --- Tests ---

test "reduce_count_nz_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1.0, 0.0, 3.0, 0.0, 5.0 };
    try testing.expectEqual(reduce_count_nz_f64(&a, 5), 3);
}

test "reduce_count_nz_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 0, 0, 3, 0, 5 };
    try testing.expectEqual(reduce_count_nz_i32(&a, 5), 2);
}

test "reduce_count_nz_f64 all zero" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0.0, 0.0, 0.0 };
    try testing.expectEqual(reduce_count_nz_f64(&a, 3), 0);
}

test "reduce_count_nz_f64 all nonzero" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1.0, 2.0, 3.0 };
    try testing.expectEqual(reduce_count_nz_f64(&a, 3), 3);
}

test "reduce_count_nz_f64 empty" {
    const testing = @import("std").testing;
    const a = [_]f64{};
    try testing.expectEqual(reduce_count_nz_f64(&a, 0), 0);
}

test "reduce_count_nz_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 0.0, 1.0, 0.0, 2.0 };
    try testing.expectEqual(reduce_count_nz_f32(&a, 4), 2);
}

test "reduce_count_nz_i64 negatives are nonzero" {
    const testing = @import("std").testing;
    const a = [_]i64{ -1, 0, -2, 0 };
    try testing.expectEqual(reduce_count_nz_i64(&a, 4), 2);
}

test "reduce_count_nz_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{ 0, 100, 0, 200, 0 };
    try testing.expectEqual(reduce_count_nz_i16(&a, 5), 2);
}

test "reduce_count_nz_i8 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{ 1, 0, -1, 0 };
    try testing.expectEqual(reduce_count_nz_i8(&a, 4), 2);
}
