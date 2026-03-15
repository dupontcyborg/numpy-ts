//! WASM 1D cross-correlation kernel for float types.
//!
//! Computes the full cross-correlation: out[k] = sum_n(a[n] * v[n - k + vLen - 1])
//! for k = 0..aLen+vLen-2.
//! The JS layer handles mode slicing (full/same/valid), complex types,
//! and dtype conversion (int→float) before calling these kernels.
//!
//! Uses dual SIMD accumulators for instruction-level parallelism.

const simd = @import("simd.zig");

/// Full 1D cross-correlation for f64.
/// out must have length na + nb - 1.
export fn correlate_f64(a: [*]const f64, na: u32, b: [*]const f64, nb: u32, out: [*]f64, outLen: u32) void {
    _ = outLen;
    const n_a = @as(usize, na);
    const n_b = @as(usize, nb);
    const full_len = n_a + n_b - 1;

    for (0..full_len) |k| {
        var acc0: simd.V2f64 = @splat(0.0);
        var acc1: simd.V2f64 = @splat(0.0);

        const j_start = if (k >= n_b - 1) k - (n_b - 1) else 0;
        const j_end = if (k < n_a) k + 1 else n_a;
        // b_off may wrap (usize underflow) but j + b_off is always valid
        const b_off = (n_b - 1) -% k;

        var j = j_start;

        // 4-wide: two V2f64 accumulators
        while (j + 4 <= j_end) : (j += 4) {
            const bi = j +% b_off;
            acc0 += simd.load2_f64(a, j) * simd.load2_f64(b, bi);
            acc1 += simd.load2_f64(a, j + 2) * simd.load2_f64(b, bi + 2);
        }
        // 2-wide remainder
        while (j + 2 <= j_end) : (j += 2) {
            acc0 += simd.load2_f64(a, j) * simd.load2_f64(b, j +% b_off);
        }
        acc0 += acc1;
        var sum: f64 = acc0[0] + acc0[1];
        // Scalar remainder
        while (j < j_end) : (j += 1) {
            sum += a[j] * b[j +% b_off];
        }
        out[k] = sum;
    }
}

/// Full 1D cross-correlation for f32.
/// out must have length na + nb - 1.
export fn correlate_f32(a: [*]const f32, na: u32, b: [*]const f32, nb: u32, out: [*]f32, outLen: u32) void {
    _ = outLen;
    const n_a = @as(usize, na);
    const n_b = @as(usize, nb);
    const full_len = n_a + n_b - 1;

    for (0..full_len) |k| {
        var acc0: simd.V4f32 = @splat(0.0);
        var acc1: simd.V4f32 = @splat(0.0);

        const j_start = if (k >= n_b - 1) k - (n_b - 1) else 0;
        const j_end = if (k < n_a) k + 1 else n_a;
        const b_off = (n_b - 1) -% k;

        var j = j_start;

        // 8-wide: two V4f32 accumulators
        while (j + 8 <= j_end) : (j += 8) {
            const bi = j +% b_off;
            acc0 += simd.load4_f32(a, j) * simd.load4_f32(b, bi);
            acc1 += simd.load4_f32(a, j + 4) * simd.load4_f32(b, bi + 4);
        }
        // 4-wide remainder
        while (j + 4 <= j_end) : (j += 4) {
            acc0 += simd.load4_f32(a, j) * simd.load4_f32(b, j +% b_off);
        }
        acc0 += acc1;
        var sum: f32 = acc0[0] + acc0[1] + acc0[2] + acc0[3];
        // Scalar remainder
        while (j < j_end) : (j += 1) {
            sum += a[j] * b[j +% b_off];
        }
        out[k] = sum;
    }
}

// --- Tests ---

test "correlate_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1, 2, 3 };
    const v = [_]f64{ 0, 1, 0.5 };
    var out: [5]f64 = undefined;
    correlate_f64(&a, 3, &v, 3, &out, 5);
    try testing.expectApproxEqAbs(out[0], 0.5, 1e-10);
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 3.5, 1e-10);
    try testing.expectApproxEqAbs(out[3], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[4], 0.0, 1e-10);
}

test "correlate_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 1, 2, 3 };
    const v = [_]f32{ 0, 1, 0.5 };
    var out: [5]f32 = undefined;
    correlate_f32(&a, 3, &v, 3, &out, 5);
    try testing.expectApproxEqAbs(out[0], 0.5, 1e-5);
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-5);
    try testing.expectApproxEqAbs(out[2], 3.5, 1e-5);
}

test "correlate_f64 autocorrelation" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1, 2, 3 };
    var out: [5]f64 = undefined;
    correlate_f64(&a, 3, &a, 3, &out, 5);
    try testing.expectApproxEqAbs(out[0], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 8.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 14.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 8.0, 1e-10);
    try testing.expectApproxEqAbs(out[4], 3.0, 1e-10);
}

test "correlate_f64 different lengths" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1, 2, 3, 4, 5 };
    const v = [_]f64{ 1, 1 };
    var out: [6]f64 = undefined;
    correlate_f64(&a, 5, &v, 2, &out, 6);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 5.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 7.0, 1e-10);
    try testing.expectApproxEqAbs(out[4], 9.0, 1e-10);
    try testing.expectApproxEqAbs(out[5], 5.0, 1e-10);
}
