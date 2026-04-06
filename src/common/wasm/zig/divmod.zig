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

/// Fused divmod scalar for i64→f64.
export fn divmod_scalar_i64_f64(a: [*]const i64, out_q: [*]f64, out_r: [*]f64, N: u32, scalar: f64) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const v: f64 = @floatFromInt(a[i]);
        const q = @floor(v / scalar);
        out_q[i] = q;
        out_r[i] = v - q * scalar;
    }
}

/// Fused divmod scalar for u64→f64.
export fn divmod_scalar_u64_f64(a: [*]const u64, out_q: [*]f64, out_r: [*]f64, N: u32, scalar: f64) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const v: f64 = @floatFromInt(a[i]);
        const q = @floor(v / scalar);
        out_q[i] = q;
        out_r[i] = v - q * scalar;
    }
}

/// Fused divmod scalar for i32→f64.
export fn divmod_scalar_i32_f64(a: [*]const i32, out_q: [*]f64, out_r: [*]f64, N: u32, scalar: f64) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const v: f64 = @floatFromInt(a[i]);
        const q = @floor(v / scalar);
        out_q[i] = q;
        out_r[i] = v - q * scalar;
    }
}

/// Fused divmod scalar for u32→f64.
export fn divmod_scalar_u32_f64(a: [*]const u32, out_q: [*]f64, out_r: [*]f64, N: u32, scalar: f64) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const v: f64 = @floatFromInt(a[i]);
        const q = @floor(v / scalar);
        out_q[i] = q;
        out_r[i] = v - q * scalar;
    }
}

/// Fused divmod scalar for i16→f64.
export fn divmod_scalar_i16_f64(a: [*]const i16, out_q: [*]f64, out_r: [*]f64, N: u32, scalar: f64) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const v: f64 = @floatFromInt(a[i]);
        const q = @floor(v / scalar);
        out_q[i] = q;
        out_r[i] = v - q * scalar;
    }
}

/// Fused divmod scalar for u16→f64.
export fn divmod_scalar_u16_f64(a: [*]const u16, out_q: [*]f64, out_r: [*]f64, N: u32, scalar: f64) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const v: f64 = @floatFromInt(a[i]);
        const q = @floor(v / scalar);
        out_q[i] = q;
        out_r[i] = v - q * scalar;
    }
}

/// Fused divmod scalar for i8→f64.
export fn divmod_scalar_i8_f64(a: [*]const i8, out_q: [*]f64, out_r: [*]f64, N: u32, scalar: f64) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const v: f64 = @floatFromInt(a[i]);
        const q = @floor(v / scalar);
        out_q[i] = q;
        out_r[i] = v - q * scalar;
    }
}

/// Fused divmod scalar for u8→f64.
export fn divmod_scalar_u8_f64(a: [*]const u8, out_q: [*]f64, out_r: [*]f64, N: u32, scalar: f64) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const v: f64 = @floatFromInt(a[i]);
        const q = @floor(v / scalar);
        out_q[i] = q;
        out_r[i] = v - q * scalar;
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
