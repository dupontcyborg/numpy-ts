//! WASM reduction any (logical OR) kernels for all numeric types.
//!
//! Reduction: result = 1 if any a[i] != 0, else 0.
//! No unsigned variants needed — non-zero check is sign-agnostic.
//! Early-exit on first non-zero element.

const simd = @import("simd.zig");

/// Returns 1 if any f64 element is non-zero, else 0.
/// Note: NaN is considered non-zero, so reduce_any_f64([NaN], 1) returns 1.
export fn reduce_any_f64(a: [*]const f64, N: u32) u32 {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (a[i] != 0) return 1;
    }
    return 0;
}

/// Returns 1 if any f32 element is non-zero, else 0.
/// Note: NaN is considered non-zero, so reduce_any_f32([NaN], 1) returns 1.
export fn reduce_any_f32(a: [*]const f32, N: u32) u32 {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (a[i] != 0) return 1;
    }
    return 0;
}

/// Returns 1 if any i64 element is non-zero, else 0.
/// Handles both signed (i64) and unsigned (u64).
export fn reduce_any_i64(a: [*]const i64, N: u32) u32 {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (a[i] != 0) return 1;
    }
    return 0;
}

/// Returns 1 if any i32 element is non-zero, else 0.
/// Handles both signed (i32) and unsigned (u32).
export fn reduce_any_i32(a: [*]const i32, N: u32) u32 {
    var any_nz: simd.V4i32 = @splat(0);
    const n_simd = N & ~@as(u32, 3);
    var k: u32 = 0;
    while (k < n_simd) : (k += 4) {
        const v = simd.load4_i32(a, k);
        any_nz |= v;
    }
    if (any_nz[0] != 0 or any_nz[1] != 0 or any_nz[2] != 0 or any_nz[3] != 0) return 1;
    // Scalar tail
    while (k < N) : (k += 1) {
        if (a[k] != 0) return 1;
    }
    return 0;
}

/// Returns 1 if any i16 element is non-zero, else 0.
/// Handles both signed (i16) and unsigned (u16).
export fn reduce_any_i16(a: [*]const i16, N: u32) u32 {
    var any_nz: simd.V8i16 = @splat(0);
    const n_simd = N & ~@as(u32, 7);
    var k: u32 = 0;
    while (k < n_simd) : (k += 8) {
        const v = simd.load8_i16(a, k);
        any_nz |= v;
    }
    inline for (0..8) |lane| {
        if (any_nz[lane] != 0) return 1;
    }
    // Scalar tail
    while (k < N) : (k += 1) {
        if (a[k] != 0) return 1;
    }
    return 0;
}

/// Returns 1 if any i8 element is non-zero, else 0.
/// Handles both signed (i8) and unsigned (u8).
export fn reduce_any_i8(a: [*]const i8, N: u32) u32 {
    var any_nz: simd.V16i8 = @splat(0);
    const n_simd = N & ~@as(u32, 15);
    var k: u32 = 0;
    while (k < n_simd) : (k += 16) {
        const v = simd.load16_i8(a, k);
        any_nz |= v;
    }
    inline for (0..16) |lane| {
        if (any_nz[lane] != 0) return 1;
    }
    // Scalar tail
    while (k < N) : (k += 1) {
        if (a[k] != 0) return 1;
    }
    return 0;
}

// --- Tests ---

test "reduce_any_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0.0, 0.0, 3.0, 0.0 };
    try testing.expectEqual(reduce_any_f64(&a, 4), 1);
    const b = [_]f64{ 0.0, 0.0, 0.0 };
    try testing.expectEqual(reduce_any_f64(&b, 3), 0);
}

test "reduce_any_f64 single and empty" {
    const testing = @import("std").testing;
    const a = [_]f64{1.0};
    try testing.expectEqual(reduce_any_f64(&a, 1), 1);
    const b = [_]f64{0.0};
    try testing.expectEqual(reduce_any_f64(&b, 1), 0);
    const c = [_]f64{};
    try testing.expectEqual(reduce_any_f64(&c, 0), 0);
}

test "reduce_any_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 0.0, 0.0, 0.0 };
    try testing.expectEqual(reduce_any_f32(&a, 3), 0);
    const b = [_]f32{ 0.0, 1.0, 0.0 };
    try testing.expectEqual(reduce_any_f32(&b, 3), 1);
}

test "reduce_any_i64 negative is nonzero" {
    const testing = @import("std").testing;
    const a = [_]i64{ 0, 0, -1 };
    try testing.expectEqual(reduce_any_i64(&a, 3), 1);
    const b = [_]i64{ 0, 0, 0 };
    try testing.expectEqual(reduce_any_i64(&b, 3), 0);
}

test "reduce_any_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 0, 5, 0 };
    try testing.expectEqual(reduce_any_i32(&a, 3), 1);
    const b = [_]i32{ 0, 0, 0 };
    try testing.expectEqual(reduce_any_i32(&b, 3), 0);
}

test "reduce_any_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{ 0, 0, 0 };
    try testing.expectEqual(reduce_any_i16(&a, 3), 0);
    const b = [_]i16{ 0, 100, 0 };
    try testing.expectEqual(reduce_any_i16(&b, 3), 1);
}

test "reduce_any_i8 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{ 0, 0, 1 };
    try testing.expectEqual(reduce_any_i8(&a, 3), 1);
    const b = [_]i8{ 0, 0, 0 };
    try testing.expectEqual(reduce_any_i8(&b, 3), 0);
}
