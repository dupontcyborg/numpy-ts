//! Fused divmod scalar kernels: compute floor quotient and floor remainder in one pass.
//!
//! out_q[i] = floor(a[i] / scalar)
//! out_r[i] = a[i] - out_q[i] * scalar   (floor modulo, same sign as divisor)
//!
//! For float types, uses @floor. For integer types, reads native int, converts
//! to f64, computes, then writes f64 results (NumPy promotes int→float64).

const simd = @import("simd.zig");

/// Fused divmod scalar for f64: 2-wide SIMD.
export fn divmod_scalar_f64(a: [*]const f64, out_q: [*]f64, out_r: [*]f64, N: u32, scalar: f64) void {
    const s: simd.V2f64 = @splat(scalar);
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const v = simd.load2_f64(a, i);
        const q = @floor(v / s);
        simd.store2_f64(out_q, i, q);
        simd.store2_f64(out_r, i, v - q * s);
    }
    while (i < N) : (i += 1) {
        const q = @floor(a[i] / scalar);
        out_q[i] = q;
        out_r[i] = a[i] - q * scalar;
    }
}

/// Fused divmod scalar for f32: 4-wide SIMD.
export fn divmod_scalar_f32(a: [*]const f32, out_q: [*]f32, out_r: [*]f32, N: u32, scalar: f32) void {
    const s: simd.V4f32 = @splat(scalar);
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        const v = simd.load4_f32(a, i);
        const q = @floor(v / s);
        simd.store4_f32(out_q, i, q);
        simd.store4_f32(out_r, i, v - q * s);
    }
    while (i < N) : (i += 1) {
        const q = @floor(a[i] / scalar);
        out_q[i] = q;
        out_r[i] = a[i] - q * scalar;
    }
}

/// Integer divmod scalar for i64. Keeps i64 dtype.
export fn divmod_scalar_i64(a: [*]const i64, out_q: [*]i64, out_r: [*]i64, N: u32, scalar: i64) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (scalar != 0) {
            out_q[i] = @divFloor(a[i], scalar);
            out_r[i] = @mod(a[i], scalar);
        } else {
            out_q[i] = 0;
            out_r[i] = 0;
        }
    }
}

/// Integer divmod scalar for u64. Keeps u64 dtype.
export fn divmod_scalar_u64(a: [*]const u64, out_q: [*]u64, out_r: [*]u64, N: u32, scalar: u64) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (scalar != 0) {
            out_q[i] = a[i] / scalar;
            out_r[i] = a[i] % scalar;
        } else {
            out_q[i] = 0;
            out_r[i] = 0;
        }
    }
}

/// Integer divmod scalar for i32: q = @divFloor(a, s), r = @mod(a, s). Keeps i32 dtype.
export fn divmod_scalar_i32(a: [*]const i32, out_q: [*]i32, out_r: [*]i32, N: u32, scalar: i32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (scalar != 0) {
            out_q[i] = @divFloor(a[i], scalar);
            out_r[i] = @mod(a[i], scalar);
        } else {
            out_q[i] = 0;
            out_r[i] = 0;
        }
    }
}

/// Integer divmod scalar for u32: q = a / s, r = a % s. Keeps u32 dtype.
export fn divmod_scalar_u32(a: [*]const u32, out_q: [*]u32, out_r: [*]u32, N: u32, scalar: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (scalar != 0) {
            out_q[i] = a[i] / scalar;
            out_r[i] = a[i] % scalar;
        } else {
            out_q[i] = 0;
            out_r[i] = 0;
        }
    }
}

/// Integer divmod scalar for i16. Keeps i16 dtype.
export fn divmod_scalar_i16(a: [*]const i16, out_q: [*]i16, out_r: [*]i16, N: u32, scalar: i16) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (scalar != 0) {
            out_q[i] = @divFloor(a[i], scalar);
            out_r[i] = @mod(a[i], scalar);
        } else {
            out_q[i] = 0;
            out_r[i] = 0;
        }
    }
}

/// Integer divmod scalar for u16. Keeps u16 dtype.
export fn divmod_scalar_u16(a: [*]const u16, out_q: [*]u16, out_r: [*]u16, N: u32, scalar: u16) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (scalar != 0) {
            out_q[i] = a[i] / scalar;
            out_r[i] = a[i] % scalar;
        } else {
            out_q[i] = 0;
            out_r[i] = 0;
        }
    }
}

/// Integer divmod scalar for i8. Keeps i8 dtype.
export fn divmod_scalar_i8(a: [*]const i8, out_q: [*]i8, out_r: [*]i8, N: u32, scalar: i8) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (scalar != 0) {
            out_q[i] = @divFloor(a[i], scalar);
            out_r[i] = @mod(a[i], scalar);
        } else {
            out_q[i] = 0;
            out_r[i] = 0;
        }
    }
}

/// Integer divmod scalar for u8. Keeps u8 dtype.
export fn divmod_scalar_u8(a: [*]const u8, out_q: [*]u8, out_r: [*]u8, N: u32, scalar: u8) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (scalar != 0) {
            out_q[i] = a[i] / scalar;
            out_r[i] = a[i] % scalar;
        } else {
            out_q[i] = 0;
            out_r[i] = 0;
        }
    }
}

// --- Array / array (same dtype, same shape) ---

/// Fused divmod for f64 arrays: 2-wide SIMD.
export fn divmod_f64(a: [*]const f64, b: [*]const f64, out_q: [*]f64, out_r: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const v = simd.load2_f64(a, i);
        const s = simd.load2_f64(b, i);
        const q = @floor(v / s);
        simd.store2_f64(out_q, i, q);
        simd.store2_f64(out_r, i, v - q * s);
    }
    while (i < N) : (i += 1) {
        const q = @floor(a[i] / b[i]);
        out_q[i] = q;
        out_r[i] = a[i] - q * b[i];
    }
}

/// Fused divmod for f32 arrays: 4-wide SIMD.
export fn divmod_f32(a: [*]const f32, b: [*]const f32, out_q: [*]f32, out_r: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        const v = simd.load4_f32(a, i);
        const s = simd.load4_f32(b, i);
        const q = @floor(v / s);
        simd.store4_f32(out_q, i, q);
        simd.store4_f32(out_r, i, v - q * s);
    }
    while (i < N) : (i += 1) {
        const q = @floor(a[i] / b[i]);
        out_q[i] = q;
        out_r[i] = a[i] - q * b[i];
    }
}

/// Integer divmod for arrays. Division by zero writes 0/0, matching the
/// scalar kernels and NumPy's integer behaviour.
inline fn divmodIntArr(comptime T: type, a: [*]const T, b: [*]const T, out_q: [*]T, out_r: [*]T, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (b[i] != 0) {
            out_q[i] = @divFloor(a[i], b[i]);
            out_r[i] = @mod(a[i], b[i]);
        } else {
            out_q[i] = 0;
            out_r[i] = 0;
        }
    }
}
export fn divmod_i64(a: [*]const i64, b: [*]const i64, out_q: [*]i64, out_r: [*]i64, N: u32) void {
    divmodIntArr(i64, a, b, out_q, out_r, N);
}
export fn divmod_u64(a: [*]const u64, b: [*]const u64, out_q: [*]u64, out_r: [*]u64, N: u32) void {
    divmodIntArr(u64, a, b, out_q, out_r, N);
}
export fn divmod_i32(a: [*]const i32, b: [*]const i32, out_q: [*]i32, out_r: [*]i32, N: u32) void {
    divmodIntArr(i32, a, b, out_q, out_r, N);
}
export fn divmod_u32(a: [*]const u32, b: [*]const u32, out_q: [*]u32, out_r: [*]u32, N: u32) void {
    divmodIntArr(u32, a, b, out_q, out_r, N);
}
export fn divmod_i16(a: [*]const i16, b: [*]const i16, out_q: [*]i16, out_r: [*]i16, N: u32) void {
    divmodIntArr(i16, a, b, out_q, out_r, N);
}
export fn divmod_u16(a: [*]const u16, b: [*]const u16, out_q: [*]u16, out_r: [*]u16, N: u32) void {
    divmodIntArr(u16, a, b, out_q, out_r, N);
}
export fn divmod_i8(a: [*]const i8, b: [*]const i8, out_q: [*]i8, out_r: [*]i8, N: u32) void {
    divmodIntArr(i8, a, b, out_q, out_r, N);
}
export fn divmod_u8(a: [*]const u8, b: [*]const u8, out_q: [*]u8, out_r: [*]u8, N: u32) void {
    divmodIntArr(u8, a, b, out_q, out_r, N);
}

// --- Tests ---

test "divmod_scalar_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 7, -7, 7.5, -7.5 };
    var q: [4]f64 = undefined;
    var r: [4]f64 = undefined;
    divmod_scalar_f64(&a, &q, &r, 4, 3);
    try testing.expectApproxEqAbs(q[0], 2.0, 1e-10);
    try testing.expectApproxEqAbs(r[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(q[1], -3.0, 1e-10);
    try testing.expectApproxEqAbs(r[1], 2.0, 1e-10);
}

test "divmod_scalar_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 7, 10, 3 };
    var q: [3]f32 = undefined;
    var r: [3]f32 = undefined;
    divmod_scalar_f32(&a, &q, &r, 3, 3);
    try testing.expectApproxEqAbs(q[0], 2.0, 1e-5);
    try testing.expectApproxEqAbs(r[0], 1.0, 1e-5);
    try testing.expectApproxEqAbs(q[1], 3.0, 1e-5);
    try testing.expectApproxEqAbs(r[1], 1.0, 1e-5);
}

test "divmod_scalar_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 7, -7 };
    var q: [2]i64 = undefined;
    var r: [2]i64 = undefined;
    divmod_scalar_i64(&a, &q, &r, 2, 3);
    try testing.expectEqual(q[0], 2);
    try testing.expectEqual(r[0], 1);
    try testing.expectEqual(q[1], -3); // floor(-7/3) = -3
    try testing.expectEqual(r[1], 2); // -7 - (-3*3) = 2
}

test "divmod_scalar_u64 basic" {
    const testing = @import("std").testing;
    const a = [_]u64{7};
    var q: [1]u64 = undefined;
    var r: [1]u64 = undefined;
    divmod_scalar_u64(&a, &q, &r, 1, 3);
    try testing.expectEqual(q[0], 2);
    try testing.expectEqual(r[0], 1);
}

test "divmod_scalar_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 7, -7 };
    var q: [2]i32 = undefined;
    var r: [2]i32 = undefined;
    divmod_scalar_i32(&a, &q, &r, 2, 3);
    try testing.expectEqual(q[0], 2);
    try testing.expectEqual(r[0], 1);
    try testing.expectEqual(q[1], -3);
    try testing.expectEqual(r[1], 2);
}

test "divmod_scalar_u32 basic" {
    const testing = @import("std").testing;
    const a = [_]u32{7};
    var q: [1]u32 = undefined;
    var r: [1]u32 = undefined;
    divmod_scalar_u32(&a, &q, &r, 1, 3);
    try testing.expectEqual(q[0], 2);
    try testing.expectEqual(r[0], 1);
}

test "divmod_scalar_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{7};
    var q: [1]i16 = undefined;
    var r: [1]i16 = undefined;
    divmod_scalar_i16(&a, &q, &r, 1, 3);
    try testing.expectEqual(q[0], 2);
    try testing.expectEqual(r[0], 1);
}

test "divmod_scalar_u16 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{7};
    var q: [1]u16 = undefined;
    var r: [1]u16 = undefined;
    divmod_scalar_u16(&a, &q, &r, 1, 3);
    try testing.expectEqual(q[0], 2);
    try testing.expectEqual(r[0], 1);
}

test "divmod_scalar_i8 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{7};
    var q: [1]i8 = undefined;
    var r: [1]i8 = undefined;
    divmod_scalar_i8(&a, &q, &r, 1, 3);
    try testing.expectEqual(q[0], 2);
    try testing.expectEqual(r[0], 1);
}

test "divmod_scalar_u8 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{7};
    var q: [1]u8 = undefined;
    var r: [1]u8 = undefined;
    divmod_scalar_u8(&a, &q, &r, 1, 3);
    try testing.expectEqual(q[0], 2);
    try testing.expectEqual(r[0], 1);
}

test "divmod_f64 array/array with a scalar tail" {
    const testing = @import("std").testing;
    const a = [_]f64{ 7, -7, 7.5, -7.5, 3.0 }; // odd length -> tail
    const b = [_]f64{ 3, 3, 2, 2, 1 };
    var q: [5]f64 = undefined;
    var r: [5]f64 = undefined;
    divmod_f64(&a, &b, &q, &r, 5);
    try testing.expectEqualSlices(f64, &[_]f64{ 2, -3, 3, -4, 3 }, &q);
    try testing.expectEqualSlices(f64, &[_]f64{ 1, 2, 1.5, 0.5, 0 }, &r);
}

test "divmod_f32 array/array with a scalar tail" {
    const testing = @import("std").testing;
    const a = [_]f32{ 7, -7, 7.5, -7.5, 3.0 };
    const b = [_]f32{ 3, 3, 2, 2, 1 };
    var q: [5]f32 = undefined;
    var r: [5]f32 = undefined;
    divmod_f32(&a, &b, &q, &r, 5);
    try testing.expectEqualSlices(f32, &[_]f32{ 2, -3, 3, -4, 3 }, &q);
    try testing.expectEqualSlices(f32, &[_]f32{ 1, 2, 1.5, 0.5, 0 }, &r);
}

test "divmod array/array over the signed integer dtypes" {
    const testing = @import("std").testing;
    // The quotient floors and the remainder takes the sign of the divisor.
    const eq = testing.expectEqualSlices;
    const a64 = [_]i64{ 7, -7, 8, -8, 5 };
    const b64 = [_]i64{ 3, 3, -3, -3, 1 };
    var q64: [5]i64 = undefined;
    var r64: [5]i64 = undefined;
    divmod_i64(&a64, &b64, &q64, &r64, 5);
    try eq(i64, &[_]i64{ 2, -3, -3, 2, 5 }, &q64);
    try eq(i64, &[_]i64{ 1, 2, -1, -2, 0 }, &r64);

    const a32 = [_]i32{ 7, -7, 8, -8, 5 };
    const b32 = [_]i32{ 3, 3, -3, -3, 1 };
    var q32: [5]i32 = undefined;
    var r32: [5]i32 = undefined;
    divmod_i32(&a32, &b32, &q32, &r32, 5);
    try eq(i32, &[_]i32{ 2, -3, -3, 2, 5 }, &q32);
    try eq(i32, &[_]i32{ 1, 2, -1, -2, 0 }, &r32);

    const a16 = [_]i16{ 7, -7, 8, -8, 5 };
    const b16 = [_]i16{ 3, 3, -3, -3, 1 };
    var q16: [5]i16 = undefined;
    var r16: [5]i16 = undefined;
    divmod_i16(&a16, &b16, &q16, &r16, 5);
    try eq(i16, &[_]i16{ 2, -3, -3, 2, 5 }, &q16);
    try eq(i16, &[_]i16{ 1, 2, -1, -2, 0 }, &r16);

    const a8 = [_]i8{ 7, -7, 8, -8, 5 };
    const b8 = [_]i8{ 3, 3, -3, -3, 1 };
    var q8: [5]i8 = undefined;
    var r8: [5]i8 = undefined;
    divmod_i8(&a8, &b8, &q8, &r8, 5);
    try eq(i8, &[_]i8{ 2, -3, -3, 2, 5 }, &q8);
    try eq(i8, &[_]i8{ 1, 2, -1, -2, 0 }, &r8);
}

test "divmod array/array over the unsigned integer dtypes" {
    const testing = @import("std").testing;
    const eq = testing.expectEqualSlices;
    const a64 = [_]u64{ 7, 8, 9, 10, 11 };
    const b64 = [_]u64{ 3, 3, 4, 5, 4 };
    var q64: [5]u64 = undefined;
    var r64: [5]u64 = undefined;
    divmod_u64(&a64, &b64, &q64, &r64, 5);
    try eq(u64, &[_]u64{ 2, 2, 2, 2, 2 }, &q64);
    try eq(u64, &[_]u64{ 1, 2, 1, 0, 3 }, &r64);

    const a32 = [_]u32{ 7, 8, 9, 10, 11 };
    const b32 = [_]u32{ 3, 3, 4, 5, 4 };
    var q32: [5]u32 = undefined;
    var r32: [5]u32 = undefined;
    divmod_u32(&a32, &b32, &q32, &r32, 5);
    try eq(u32, &[_]u32{ 2, 2, 2, 2, 2 }, &q32);
    try eq(u32, &[_]u32{ 1, 2, 1, 0, 3 }, &r32);

    const a16 = [_]u16{ 7, 8, 9, 10, 11 };
    const b16 = [_]u16{ 3, 3, 4, 5, 4 };
    var q16: [5]u16 = undefined;
    var r16: [5]u16 = undefined;
    divmod_u16(&a16, &b16, &q16, &r16, 5);
    try eq(u16, &[_]u16{ 2, 2, 2, 2, 2 }, &q16);
    try eq(u16, &[_]u16{ 1, 2, 1, 0, 3 }, &r16);

    const a8 = [_]u8{ 7, 8, 9, 10, 11 };
    const b8 = [_]u8{ 3, 3, 4, 5, 4 };
    var q8: [5]u8 = undefined;
    var r8: [5]u8 = undefined;
    divmod_u8(&a8, &b8, &q8, &r8, 5);
    try eq(u8, &[_]u8{ 2, 2, 2, 2, 2 }, &q8);
    try eq(u8, &[_]u8{ 1, 2, 1, 0, 3 }, &r8);
}

test "divmod_i32 array/array division by zero writes 0/0" {
    const testing = @import("std").testing;
    const a = [_]i32{ 5, -9 };
    const b = [_]i32{ 0, 0 };
    var q: [2]i32 = undefined;
    var r: [2]i32 = undefined;
    divmod_i32(&a, &b, &q, &r, 2);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 0 }, &q);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 0 }, &r);
}
