//! WASM reduction all (logical AND) kernels for all numeric types.
//!
//! Reduction: result = 1 if 0 if any a[i] == 0, else 1.
//! No unsigned variants needed — non-zero check is sign-agnostic.
//! Early-exit on first zero element.

const simd = @import("simd.zig");

/// Returns 1 if all f64 elements are non-zero, else 0.
export fn reduce_all_f64(a: [*]const f64, N: u32) u32 {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (a[i] == 0) return 0;
    }
    return 1;
}

/// Returns 1 if all f32 elements are non-zero, else 0.
export fn reduce_all_f32(a: [*]const f32, N: u32) u32 {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (a[i] == 0) return 0;
    }
    return 1;
}

/// Returns 1 if all i64 elements are non-zero, else 0.
/// Handles both signed (i64) and unsigned (u64).
export fn reduce_all_i64(a: [*]const i64, N: u32) u32 {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (a[i] == 0) return 0;
    }
    return 1;
}

/// Returns 1 if all i32 elements are non-zero, else 0.
/// Handles both signed (i32) and unsigned (u32).
export fn reduce_all_i32(a: [*]const i32, N: u32) u32 {
    const zero_v: simd.V4i32 = @splat(0);
    var any_zero: simd.V4i32 = @splat(0);
    const n_simd = N & ~@as(u32, 3);
    var k: u32 = 0;
    while (k < n_simd) : (k += 4) {
        const v = simd.load4_i32(a, k);
        any_zero |= @select(i32, v == zero_v, @as(simd.V4i32, @splat(-1)), zero_v);
    }
    if (any_zero[0] != 0 or any_zero[1] != 0 or any_zero[2] != 0 or any_zero[3] != 0) return 0;
    // Scalar tail
    while (k < N) : (k += 1) {
        if (a[k] == 0) return 0;
    }
    return 1;
}

/// Returns 1 if all i16 elements are non-zero, else 0.
/// Handles both signed (i16) and unsigned (u16).
export fn reduce_all_i16(a: [*]const i16, N: u32) u32 {
    const zero_v: simd.V8i16 = @splat(0);
    var any_zero: simd.V8i16 = @splat(0);
    const n_simd = N & ~@as(u32, 7);
    var k: u32 = 0;
    while (k < n_simd) : (k += 8) {
        const v = simd.load8_i16(a, k);
        any_zero |= @select(i16, v == zero_v, @as(simd.V8i16, @splat(-1)), zero_v);
    }
    inline for (0..8) |lane| {
        if (any_zero[lane] != 0) return 0;
    }
    // Scalar tail
    while (k < N) : (k += 1) {
        if (a[k] == 0) return 0;
    }
    return 1;
}

/// Returns 1 if all i8 elements are non-zero, else 0.
/// Handles both signed (i8) and unsigned (u8).
export fn reduce_all_i8(a: [*]const i8, N: u32) u32 {
    const zero_v: simd.V16i8 = @splat(0);
    var any_zero: simd.V16i8 = @splat(0);
    const n_simd = N & ~@as(u32, 15);
    var k: u32 = 0;
    while (k < n_simd) : (k += 16) {
        const v = simd.load16_i8(a, k);
        any_zero |= @select(i8, v == zero_v, @as(simd.V16i8, @splat(-1)), zero_v);
    }
    inline for (0..16) |lane| {
        if (any_zero[lane] != 0) return 0;
    }
    // Scalar tail
    while (k < N) : (k += 1) {
        if (a[k] == 0) return 0;
    }
    return 1;
}

// --- Tests ---

test "reduce_all_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1.0, 2.0, 0.0, 4.0 };
    try testing.expectEqual(reduce_all_f64(&a, 4), 0);
    const b = [_]f64{ 1.0, 2.0, 3.0 };
    try testing.expectEqual(reduce_all_f64(&b, 3), 1);
}

test "reduce_all_f64 single element" {
    const testing = @import("std").testing;
    const a = [_]f64{5.0};
    try testing.expectEqual(reduce_all_f64(&a, 1), 1);
    const b = [_]f64{0.0};
    try testing.expectEqual(reduce_all_f64(&b, 1), 0);
}

test "reduce_all_f64 empty" {
    const testing = @import("std").testing;
    const a = [_]f64{};
    try testing.expectEqual(reduce_all_f64(&a, 0), 1);
}

test "reduce_all_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    try testing.expectEqual(reduce_all_f32(&a, 3), 1);
    const b = [_]f32{ 1.0, 0.0, 3.0 };
    try testing.expectEqual(reduce_all_f32(&b, 3), 0);
}

test "reduce_all_i64 negatives are nonzero" {
    const testing = @import("std").testing;
    const a = [_]i64{ -1, -2, -3 };
    try testing.expectEqual(reduce_all_i64(&a, 3), 1);
    const b = [_]i64{ -1, 0, -3 };
    try testing.expectEqual(reduce_all_i64(&b, 3), 0);
}

test "reduce_all_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 1, 2, 3 };
    try testing.expectEqual(reduce_all_i32(&a, 3), 1);
    const b = [_]i32{ 1, 2, 0 };
    try testing.expectEqual(reduce_all_i32(&b, 3), 0);
}

test "reduce_all_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{ 10, 20, 30 };
    try testing.expectEqual(reduce_all_i16(&a, 3), 1);
    const b = [_]i16{ 10, 0, 30 };
    try testing.expectEqual(reduce_all_i16(&b, 3), 0);
}

test "reduce_all_i8 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{ 1, 2, 3 };
    try testing.expectEqual(reduce_all_i8(&a, 3), 1);
    const b = [_]i8{ 1, 0, 3 };
    try testing.expectEqual(reduce_all_i8(&b, 3), 0);
}

test "reduce_all early exit on zero at start" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0.0, 1.0, 2.0 };
    try testing.expectEqual(reduce_all_f64(&a, 3), 0);
}
