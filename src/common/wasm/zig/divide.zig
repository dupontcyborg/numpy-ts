//! WASM element-wise division kernels for float types.
//!
//! Binary: out[i] = a[i] / b[i]
//! Scalar: out[i] = a[i] / scalar
//! Only float types (integer division promotes to float64 in NumPy).

const simd = @import("simd.zig");

/// Element-wise divide for f64 using 2-wide SIMD: out[i] = a[i] / b[i].
export fn div_f64(a: [*]const f64, b: [*]const f64, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, simd.load2_f64(a, i) / simd.load2_f64(b, i));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] / b[i];
    }
}

/// Element-wise divide by scalar for f64 using 2-wide SIMD: out[i] = a[i] / scalar.
export fn div_scalar_f64(a: [*]const f64, out: [*]f64, N: u32, scalar: f64) void {
    const s: simd.V2f64 = @splat(scalar);
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, simd.load2_f64(a, i) / s);
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] / scalar;
    }
}

/// Element-wise divide for f32 using 4-wide SIMD: out[i] = a[i] / b[i].
export fn div_f32(a: [*]const f32, b: [*]const f32, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_f32(out, i, simd.load4_f32(a, i) / simd.load4_f32(b, i));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] / b[i];
    }
}

/// Element-wise divide by scalar for f32 using 4-wide SIMD: out[i] = a[i] / scalar.
export fn div_scalar_f32(a: [*]const f32, out: [*]f32, N: u32, scalar: f32) void {
    const s: simd.V4f32 = @splat(scalar);
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_f32(out, i, simd.load4_f32(a, i) / s);
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] / scalar;
    }
}

/// --- Tests ---

const testing = @import("std").testing;

test "div_f64 basic" {
    const a = [_]f64{ 10, 20, 30, 40 };
    const b = [_]f64{ 2, 5, 10, 8 };
    var out: [4]f64 = undefined;
    div_f64(&a, &b, &out, 4);
    try testing.expectApproxEqAbs(out[0], 5.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 4.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 5.0, 1e-10);
}

test "div_scalar_f64 basic" {
    const a = [_]f64{ 10, 20, 30, 40 };
    var out: [4]f64 = undefined;
    div_scalar_f64(&a, &out, 4, 5.0);
    try testing.expectApproxEqAbs(out[0], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 4.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 6.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 8.0, 1e-10);
}

test "div_f32 basic" {
    const a = [_]f32{ 10, 20, 30, 40, 50 };
    const b = [_]f32{ 2, 4, 5, 8, 10 };
    var out: [5]f32 = undefined;
    div_f32(&a, &b, &out, 5);
    try testing.expectApproxEqAbs(out[0], 5.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 5.0, 1e-5);
    try testing.expectApproxEqAbs(out[4], 5.0, 1e-5);
}
