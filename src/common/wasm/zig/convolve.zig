//! WASM 1D convolution kernel for float types.
//!
//! Computes full linear convolution: out[k] = sum_n(a[n] * v[k - n])
//! for k = 0..aLen+vLen-2.
//! The JS layer handles mode slicing (full/same/valid), complex types,
//! and dtype conversion (int→float) before calling these kernels.
//!
//! Convolution accesses v in reverse order, preventing contiguous SIMD loads.
//! Kept as scalar inner loop (same approach as wasm-bench).

const simd = @import("simd.zig");

/// Full 1D convolution for f64.
/// out must have length na + nb - 1.
export fn convolve_f64(a: [*]const f64, na: u32, b: [*]const f64, nb: u32, out: [*]f64, outLen: u32) void {
    _ = outLen;
    const n_a = @as(usize, na);
    const n_b = @as(usize, nb);
    const full_len = n_a + n_b - 1;

    for (0..full_len) |k| {
        var sum: f64 = 0.0;
        const j_start = if (k >= n_b - 1) k - (n_b - 1) else 0;
        const j_end = if (k < n_a) k + 1 else n_a;

        var j = j_start;
        while (j < j_end) : (j += 1) {
            sum += a[j] * b[k - j];
        }
        out[k] = sum;
    }
}

/// Full 1D convolution for f32.
/// out must have length na + nb - 1.
export fn convolve_f32(a: [*]const f32, na: u32, b: [*]const f32, nb: u32, out: [*]f32, outLen: u32) void {
    _ = outLen;
    const n_a = @as(usize, na);
    const n_b = @as(usize, nb);
    const full_len = n_a + n_b - 1;

    for (0..full_len) |k| {
        var sum: f32 = 0.0;
        const j_start = if (k >= n_b - 1) k - (n_b - 1) else 0;
        const j_end = if (k < n_a) k + 1 else n_a;

        var j = j_start;
        while (j < j_end) : (j += 1) {
            sum += a[j] * b[k - j];
        }
        out[k] = sum;
    }
}

// --- Tests ---

test "convolve_f64 basic" {
    const testing = @import("std").testing;
    // np.convolve([1, 2, 3], [0, 1, 0.5]) = [0, 1, 2.5, 4, 1.5]
    const a = [_]f64{ 1, 2, 3 };
    const v = [_]f64{ 0, 1, 0.5 };
    var out: [5]f64 = undefined;
    convolve_f64(&a, 3, &v, 3, &out, 5);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 2.5, 1e-10);
    try testing.expectApproxEqAbs(out[3], 4.0, 1e-10);
    try testing.expectApproxEqAbs(out[4], 1.5, 1e-10);
}

test "convolve_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 1, 2, 3 };
    const v = [_]f32{ 0, 1, 0.5 };
    var out: [5]f32 = undefined;
    convolve_f32(&a, 3, &v, 3, &out, 5);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 1.0, 1e-5);
    try testing.expectApproxEqAbs(out[2], 2.5, 1e-5);
    try testing.expectApproxEqAbs(out[3], 4.0, 1e-5);
    try testing.expectApproxEqAbs(out[4], 1.5, 1e-5);
}

test "convolve_f64 identity" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1, 2, 3, 4 };
    const v = [_]f64{1};
    var out: [4]f64 = undefined;
    convolve_f64(&a, 4, &v, 1, &out, 4);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 4.0, 1e-10);
}

test "convolve_f64 different lengths" {
    const testing = @import("std").testing;
    // np.convolve([1, 2, 3, 4, 5], [1, 1]) = [1, 3, 5, 7, 9, 5]
    const a = [_]f64{ 1, 2, 3, 4, 5 };
    const v = [_]f64{ 1, 1 };
    var out: [6]f64 = undefined;
    convolve_f64(&a, 5, &v, 2, &out, 6);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 5.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 7.0, 1e-10);
    try testing.expectApproxEqAbs(out[4], 9.0, 1e-10);
    try testing.expectApproxEqAbs(out[5], 5.0, 1e-10);
}
