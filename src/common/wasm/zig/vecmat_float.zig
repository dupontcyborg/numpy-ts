//! WASM vector-matrix product kernels for floating-point types.
//!
//! Computes y[j] = sum_k x[k] * A[k*N+j] for x[K] and A[K,N].
//! A is accessed column-wise, so we use the i-k-j broadcast pattern:
//! for each k, broadcast x[k] and accumulate x[k]*A[k,:] into y.

const simd = @import("simd.zig");

/// Computes the vector-matrix product of a K-length f64 vector and a KxN f64 matrix.
/// y[j] = sum_k x[k] * A[k*N+j]
export fn vecmat_f64(x: [*]const f64, A: [*]const f64, y: [*]f64, K: u32, N: u32) void {
    // Zero output
    @memset(y[0..@as(usize, N)], 0);
    const n_simd = N & ~@as(u32, 1); // floor to V2f64 (2-wide)
    for (0..K) |k| {
        const x_k: simd.V2f64 = @splat(x[k]);
        const a_row = k * N;
        var j: u32 = 0;
        // Vectorized j loop: 2 f64s per step
        while (j < n_simd) : (j += 2) {
            simd.store2_f64(y, j, simd.mulAdd_f64x2(x_k, simd.load2_f64(A, a_row + j), simd.load2_f64(y, j)));
        }
        // Scalar remainder
        while (j < N) : (j += 1) {
            y[j] += x[k] * A[a_row + j];
        }
    }
}

/// Computes the vector-matrix product of a K-length f32 vector and a KxN f32 matrix.
/// y[j] = sum_k x[k] * A[k*N+j]
export fn vecmat_f32(x: [*]const f32, A: [*]const f32, y: [*]f32, K: u32, N: u32) void {
    // Zero output
    @memset(y[0..@as(usize, N)], 0);
    const n_simd = N & ~@as(u32, 3); // floor to V4f32 (4-wide)
    for (0..K) |k| {
        const x_k: simd.V4f32 = @splat(x[k]);
        const a_row = k * N;
        var j: u32 = 0;
        // Vectorized j loop: 4 f32s per step
        while (j < n_simd) : (j += 4) {
            simd.store4_f32(y, j, simd.mulAdd_f32x4(x_k, simd.load4_f32(A, a_row + j), simd.load4_f32(y, j)));
        }
        // Scalar remainder
        while (j < N) : (j += 1) {
            y[j] += x[k] * A[a_row + j];
        }
    }
}

/// Computes the vector-matrix product of a K-length complex128 vector and a KxN complex128 matrix.
/// x and A are interleaved [re0, im0, re1, im1, ...]
/// y is also interleaved [re0, im0, re1, im1, ...]
export fn vecmat_c128(x: [*]const f64, A: [*]const f64, y: [*]f64, K: u32, N: u32) void {
    // Zero output (2 f64s per complex element)
    @memset(y[0 .. @as(usize, N) * 2], 0);
    // vecmat computes conj(x) @ A, matching NumPy's convention
    for (0..K) |k| {
        const x_re = x[k * 2];
        const x_im = -x[k * 2 + 1]; // conjugate
        const a_row = k * N * 2; // 2 f64s per complex element
        // Scalar loop: conj(x) * A multiply-accumulate
        for (0..N) |j| {
            const idx = j * 2;
            const a_re = A[a_row + idx];
            const a_im = A[a_row + idx + 1];
            // (x_re - x_im*i) * (a_re + a_im*i)
            y[idx] += x_re * a_re - x_im * a_im;
            y[idx + 1] += x_re * a_im + x_im * a_re;
        }
    }
}

/// Computes the vector-matrix product of a K-length complex64 vector and a KxN complex64 matrix.
/// x and A are interleaved [re0, im0, re1, im1, ...]
/// y is also interleaved [re0, im0, re1, im1, ...]
export fn vecmat_c64(x: [*]const f32, A: [*]const f32, y: [*]f32, K: u32, N: u32) void {
    // Zero output (2 f32s per complex element)
    @memset(y[0 .. @as(usize, N) * 2], 0);
    // vecmat computes conj(x) @ A, matching NumPy's convention
    for (0..K) |k| {
        const x_re = x[k * 2];
        const x_im = -x[k * 2 + 1]; // conjugate
        const a_row = k * N * 2; // 2 f32s per complex element
        // Scalar loop: conj(x) * A multiply-accumulate
        for (0..N) |j| {
            const idx = j * 2;
            const a_re = A[a_row + idx];
            const a_im = A[a_row + idx + 1];
            // (x_re - x_im*i) * (a_re + a_im*i)
            y[idx] += x_re * a_re - x_im * a_im;
            y[idx + 1] += x_re * a_im + x_im * a_re;
        }
    }
}

// --- Tests ---

test "vecmat_f64 3 · 3x2 → 2" {
    const testing = @import("std").testing;
    // x = [1,2,3], A = [[1,2],[3,4],[5,6]]
    // y[0] = 1*1+2*3+3*5 = 22, y[1] = 1*2+2*4+3*6 = 28
    const x = [_]f64{ 1, 2, 3 };
    const A = [_]f64{ 1, 2, 3, 4, 5, 6 };
    var y: [2]f64 = undefined;
    vecmat_f64(&x, &A, &y, 3, 2);
    try testing.expectApproxEqAbs(y[0], 22.0, 1e-10);
    try testing.expectApproxEqAbs(y[1], 28.0, 1e-10);
}

test "vecmat_f32 basic" {
    const testing = @import("std").testing;
    const x = [_]f32{ 1, 2 };
    const A = [_]f32{ 1, 0, 0, 0, 1, 0 };
    var y: [3]f32 = undefined;
    vecmat_f32(&x, &A, &y, 2, 3);
    try testing.expectApproxEqAbs(y[0], 1.0, 1e-5);
    try testing.expectApproxEqAbs(y[1], 2.0, 1e-5);
    try testing.expectApproxEqAbs(y[2], 0.0, 1e-5);
}

test "vecmat_c128 basic" {
    const testing = @import("std").testing;
    // x = [(1+2i)], A = [[(3+4i), (5+6i)]], K=1, N=2
    // conj(x) = [(1-2i)]
    // y[0] = (1-2i)*(3+4i) = 3+4i-6i+8 = 11-2i
    // y[1] = (1-2i)*(5+6i) = 5+6i-10i+12 = 17-4i
    const x = [_]f64{ 1, 2 };
    const A = [_]f64{ 3, 4, 5, 6 };
    var y: [4]f64 = undefined;
    vecmat_c128(&x, &A, &y, 1, 2);
    try testing.expectApproxEqAbs(y[0], 11.0, 1e-10);
    try testing.expectApproxEqAbs(y[1], -2.0, 1e-10);
    try testing.expectApproxEqAbs(y[2], 17.0, 1e-10);
    try testing.expectApproxEqAbs(y[3], -4.0, 1e-10);
}

test "vecmat_c64 basic" {
    const testing = @import("std").testing;
    // Same as c128 but f32 — conj(x) @ A
    const x = [_]f32{ 1, 2 };
    const A = [_]f32{ 3, 4, 5, 6 };
    var y: [4]f32 = undefined;
    vecmat_c64(&x, &A, &y, 1, 2);
    try testing.expectApproxEqAbs(y[0], 11.0, 1e-5);
    try testing.expectApproxEqAbs(y[1], -2.0, 1e-5);
    try testing.expectApproxEqAbs(y[2], 17.0, 1e-5);
    try testing.expectApproxEqAbs(y[3], -4.0, 1e-5);
}
