//! WASM element-wise left shift kernels for all integer types.
//!
//! Binary: out[i] = a[i] << b[i]
//! Scalar: out[i] = a[i] << scalar
//! Both operate on contiguous 1D buffers of length N.

const simd = @import("simd.zig");

/// Element-wise left shift for i64: out[i] = a[i] << b[i].
/// No SIMD — WASM has no variable-shift for i64x2.
export fn left_shift_i64(a: [*]const i64, b: [*]const i64, out: [*]i64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const shift: u6 = @intCast(@as(u64, @bitCast(b[i])) & 63);
        out[i] = a[i] << shift;
    }
}

/// Element-wise left shift scalar for i64: out[i] = a[i] << scalar.
export fn left_shift_scalar_i64(a: [*]const i64, out: [*]i64, N: u32, scalar: i64) void {
    const shift: u6 = @intCast(@as(u64, @bitCast(scalar)) & 63);
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = a[i] << shift;
    }
}

/// Element-wise left shift for i32 using 4-wide SIMD: out[i] = a[i] << b[i].
/// WASM SIMD supports i32x4.shl with a scalar shift amount, but not per-lane
/// variable shifts. We use scalar loop for variable shifts.
export fn left_shift_i32(a: [*]const i32, b: [*]const i32, out: [*]i32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const shift: u5 = @intCast(@as(u32, @bitCast(b[i])) & 31);
        out[i] = a[i] << shift;
    }
}

/// Element-wise left shift scalar for i32 using 4-wide SIMD: out[i] = a[i] << scalar.
export fn left_shift_scalar_i32(a: [*]const i32, out: [*]i32, N: u32, scalar: i32) void {
    const shift: u5 = @intCast(@as(u32, @bitCast(scalar)) & 31);
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_i32(out, i, simd.load4_i32(a, i) << @splat(shift));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] << shift;
    }
}

/// Element-wise left shift for i16: out[i] = a[i] << b[i].
/// Scalar loop — WASM SIMD has no per-lane variable shift for i16x8.
export fn left_shift_i16(a: [*]const i16, b: [*]const i16, out: [*]i16, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const shift: u4 = @intCast(@as(u16, @bitCast(b[i])) & 15);
        out[i] = a[i] << shift;
    }
}

/// Element-wise left shift scalar for i16 using 8-wide SIMD: out[i] = a[i] << scalar.
export fn left_shift_scalar_i16(a: [*]const i16, out: [*]i16, N: u32, scalar: i16) void {
    const shift: u4 = @intCast(@as(u16, @bitCast(scalar)) & 15);
    const n_simd = N & ~@as(u32, 7);
    var i: u32 = 0;
    while (i < n_simd) : (i += 8) {
        simd.store8_i16(out, i, simd.load8_i16(a, i) << @splat(shift));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] << shift;
    }
}

/// Element-wise left shift for i8: out[i] = a[i] << b[i].
/// Scalar loop — WASM SIMD has no per-lane variable shift for i8x16.
export fn left_shift_i8(a: [*]const i8, b: [*]const i8, out: [*]i8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const shift: u3 = @intCast(@as(u8, @bitCast(b[i])) & 7);
        out[i] = a[i] << shift;
    }
}

/// Element-wise left shift scalar for i8 using 16-wide SIMD: out[i] = a[i] << scalar.
export fn left_shift_scalar_i8(a: [*]const i8, out: [*]i8, N: u32, scalar: i8) void {
    const shift: u3 = @intCast(@as(u8, @bitCast(scalar)) & 7);
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    while (i < n_simd) : (i += 16) {
        simd.store16_i8(out, i, simd.load16_i8(a, i) << @splat(shift));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] << shift;
    }
}

// --- Tests ---

test "left_shift_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 1, 2, 3, 4, 5 };
    const b = [_]i32{ 1, 2, 3, 0, 4 };
    var out: [5]i32 = undefined;
    left_shift_i32(&a, &b, &out, 5);
    try testing.expectEqual(out[0], 2);
    try testing.expectEqual(out[1], 8);
    try testing.expectEqual(out[2], 24);
    try testing.expectEqual(out[3], 4);
    try testing.expectEqual(out[4], 80);
}

test "left_shift_scalar_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 1, 2, 3, 4, 5, 6, 7 };
    var out: [7]i32 = undefined;
    left_shift_scalar_i32(&a, &out, 7, 2);
    for (0..7) |idx| {
        const expected: i32 = @as(i32, @intCast(idx + 1)) << 2;
        try testing.expectEqual(out[idx], expected);
    }
}

test "left_shift_scalar_i8 SIMD boundary N=17" {
    const testing = @import("std").testing;
    var a: [17]i8 = undefined;
    for (0..17) |idx| {
        a[idx] = @intCast(idx);
    }
    var out: [17]i8 = undefined;
    left_shift_scalar_i8(&a, &out, 17, 1);
    for (0..17) |idx| {
        const v: i8 = @intCast(idx);
        try testing.expectEqual(out[idx], v << 1);
    }
}
