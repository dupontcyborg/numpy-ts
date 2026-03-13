//! WASM element-wise absolute value kernels for all numeric types.
//!
//! Unary: out[i] = |a[i]|
//! Operates on contiguous 1D buffers of length N.
//! For complex types, use the JS path (magnitude = sqrt(re²+im²)).

const simd = @import("simd.zig");

/// Element-wise absolute value for f64 using 2-wide SIMD: out[i] = |a[i]|.
/// Uses bitwise AND to clear the sign bit (faster than compare+select).
export fn abs_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const v = simd.load2_f64(a, i);
        const mask: @Vector(2, u64) = @splat(0x7FFFFFFFFFFFFFFF);
        simd.store2_f64(out, i, @bitCast(@as(@Vector(2, u64), @bitCast(v)) & mask));
    }
    while (i < N) : (i += 1) {
        out[i] = @abs(a[i]);
    }
}

/// Element-wise absolute value for f32 using 4-wide SIMD.
/// Uses bitwise AND to clear the sign bit.
export fn abs_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        const v = simd.load4_f32(a, i);
        const mask: @Vector(4, u32) = @splat(0x7FFFFFFF);
        simd.store4_f32(out, i, @bitCast(@as(@Vector(4, u32), @bitCast(v)) & mask));
    }
    while (i < N) : (i += 1) {
        out[i] = @abs(a[i]);
    }
}

/// Element-wise absolute value for i64, scalar loop.
/// No i64x2 compare in WASM SIMD, so no vectorization.
export fn abs_i64(a: [*]const i64, out: [*]i64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const v = a[i];
        out[i] = if (v < 0) -%v else v;
    }
}

/// Element-wise absolute value for i32 using 4-wide SIMD.
/// Uses compare+select: neg = 0 -% v, then select(v < 0, neg, v).
export fn abs_i32(a: [*]const i32, out: [*]i32, N: u32) void {
    const zero: simd.V4i32 = @splat(0);
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        const v = simd.load4_i32(a, i);
        const neg = zero -% v;
        simd.store4_i32(out, i, @select(i32, v < zero, neg, v));
    }
    while (i < N) : (i += 1) {
        const v = a[i];
        out[i] = if (v < 0) -%v else v;
    }
}

/// Element-wise absolute value for i16 using 8-wide SIMD.
/// Uses compare+select: neg = 0 -% v, then select(v < 0, neg, v).
export fn abs_i16(a: [*]const i16, out: [*]i16, N: u32) void {
    const zero: simd.V8i16 = @splat(0);
    const n_simd = N & ~@as(u32, 7);
    var i: u32 = 0;
    while (i < n_simd) : (i += 8) {
        const v = simd.load8_i16(a, i);
        const neg = zero -% v;
        simd.store8_i16(out, i, @select(i16, v < zero, neg, v));
    }
    while (i < N) : (i += 1) {
        const v = a[i];
        out[i] = if (v < 0) -%v else v;
    }
}

/// Element-wise absolute value for i8 using 16-wide SIMD.
/// Uses compare+select: neg = 0 -% v, then select(v < 0, neg, v).
export fn abs_i8(a: [*]const i8, out: [*]i8, N: u32) void {
    const zero: simd.V16i8 = @splat(0);
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    while (i < n_simd) : (i += 16) {
        const v = simd.load16_i8(a, i);
        const neg = zero -% v;
        simd.store16_i8(out, i, @select(i8, v < zero, neg, v));
    }
    while (i < N) : (i += 1) {
        const v = a[i];
        out[i] = if (v < 0) -%v else v;
    }
}

// --- Tests ---

test "abs_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ -1, 2, -3, 0 };
    var out: [4]f64 = undefined;
    abs_f64(&a, &out, 4);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 0.0, 1e-10);
}

test "abs_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ -1, 2, -3, 0, 5 };
    var out: [5]i32 = undefined;
    abs_i32(&a, &out, 5);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 2);
    try testing.expectEqual(out[2], 3);
    try testing.expectEqual(out[3], 0);
    try testing.expectEqual(out[4], 5);
}

test "abs_i8 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{ -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16, -17 };
    var out: [17]i8 = undefined;
    abs_i8(&a, &out, 17);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 2);
    try testing.expectEqual(out[16], 17);
}
