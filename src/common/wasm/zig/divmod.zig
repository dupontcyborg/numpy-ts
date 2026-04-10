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
