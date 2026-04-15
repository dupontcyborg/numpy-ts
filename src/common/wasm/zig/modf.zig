//! WASM element-wise modf kernel: split into integer and fractional parts.
//!
//! int_out[i] = trunc(a[i]), frac_out[i] = a[i] - trunc(a[i])
//! Operates on contiguous 1D buffers of length N.

const simd = @import("simd.zig");

/// Element-wise modf for f64 using 2-wide SIMD.
export fn modf_f64(a: [*]const f64, frac_out: [*]f64, int_out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const v = simd.load2_f64(a, i);
        const t = @trunc(v);
        simd.store2_f64(int_out, i, t);
        simd.store2_f64(frac_out, i, v - t);
    }
    while (i < N) : (i += 1) {
        const v = a[i];
        const t = @trunc(v);
        int_out[i] = t;
        frac_out[i] = v - t;
    }
}

/// Element-wise modf for f32 using 4-wide SIMD.
export fn modf_f32(a: [*]const f32, frac_out: [*]f32, int_out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        const v = simd.load4_f32(a, i);
        const t = @trunc(v);
        simd.store4_f32(int_out, i, t);
        simd.store4_f32(frac_out, i, v - t);
    }
    while (i < N) : (i += 1) {
        const v = a[i];
        const t = @trunc(v);
        int_out[i] = t;
        frac_out[i] = v - t;
    }
}

// --- Tests ---

test "modf_f64 basic" {
    const testing = @import("std").testing;
    const x = [_]f64{ 0.0, 1.5, -2.7, 3.0, 0.25 };
    var frac: [5]f64 = undefined;
    var int: [5]f64 = undefined;
    modf_f64(&x, &frac, &int, 5);
    // 0.0 → (0.0, 0.0)
    try testing.expectApproxEqAbs(int[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(frac[0], 0.0, 1e-10);
    // 1.5 → (1.0, 0.5)
    try testing.expectApproxEqAbs(int[1], 1.0, 1e-10);
    try testing.expectApproxEqAbs(frac[1], 0.5, 1e-10);
    // -2.7 → (-2.0, -0.7)
    try testing.expectApproxEqAbs(int[2], -2.0, 1e-10);
    try testing.expectApproxEqAbs(frac[2], -0.7, 1e-10);
    // 3.0 → (3.0, 0.0)
    try testing.expectApproxEqAbs(int[3], 3.0, 1e-10);
    try testing.expectApproxEqAbs(frac[3], 0.0, 1e-10);
    // 0.25 → (0.0, 0.25)
    try testing.expectApproxEqAbs(int[4], 0.0, 1e-10);
    try testing.expectApproxEqAbs(frac[4], 0.25, 1e-10);
}

test "modf_f32 basic" {
    const testing = @import("std").testing;
    const x = [_]f32{ 0.0, 1.5, -2.7, 3.0, 0.25 };
    var frac: [5]f32 = undefined;
    var int: [5]f32 = undefined;
    modf_f32(&x, &frac, &int, 5);
    try testing.expectApproxEqAbs(int[0], 0.0, 1e-5);
    try testing.expectApproxEqAbs(frac[0], 0.0, 1e-5);
    try testing.expectApproxEqAbs(int[1], 1.0, 1e-5);
    try testing.expectApproxEqAbs(frac[1], 0.5, 1e-5);
    try testing.expectApproxEqAbs(int[2], -2.0, 1e-5);
    try testing.expectApproxEqAbs(frac[2], -0.7, 1e-5);
    try testing.expectApproxEqAbs(int[3], 3.0, 1e-5);
    try testing.expectApproxEqAbs(frac[3], 0.0, 1e-5);
}

test "modf_f64 negative zero" {
    const testing = @import("std").testing;
    const math = @import("std").math;
    const x = [_]f64{-0.0};
    var frac: [1]f64 = undefined;
    var int: [1]f64 = undefined;
    modf_f64(&x, &frac, &int, 1);
    try testing.expect(math.isNegativeZero(int[0]));
    // frac = -0.0 - (-0.0) = +0.0 (IEEE 754 subtraction), not -0.0
    try testing.expectApproxEqAbs(frac[0], 0.0, 1e-10);
}

test "modf_f64 inf" {
    const testing = @import("std").testing;
    const math = @import("std").math;
    const x = [_]f64{ math.inf(f64), -math.inf(f64) };
    var frac: [2]f64 = undefined;
    var int: [2]f64 = undefined;
    modf_f64(&x, &frac, &int, 2);
    try testing.expect(math.isInf(int[0]) and int[0] > 0);
    try testing.expect(math.isInf(int[1]) and int[1] < 0);
}
