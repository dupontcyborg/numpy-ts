//! WASM matmul kernels for f32, f64, complex64, and complex128.
//!
//! Convention: C = A @ B where A is (M x K), B is (K x N), C is (M x N).
//! All matrices are row-major (C-contiguous).
//! Complex matrices are interleaved [re, im, re, im, ...]; M, N, K are element counts.

const simd = @import("simd.zig");

const TILE_F64 = 64; // Tile size for f64 matmul (tuned for WASM v128)
const TILE_F32 = 128; // Tile size for f32 matmul (tuned for WASM v128)
const F64_CROSSOVER = 256; // ijk→ikj crossover for f64; V2f64 (2-wide)
const F32_CROSSOVER = 512; // ijk→ikj crossover for f32; V4f32 (4-wide)

/// Computes C = A @ B for row-major f64 matrices.
/// A is (M x K), B is (K x N), C is (M x N).
/// Dispatches to ijk (N ≤ F64_CROSSOVER) or ikj (N > F64_CROSSOVER) based on benchmarked
/// crossover point — ijk reduces C-memory traffic but degrades under large N due to B-cache pressure.
export fn matmul_f64(a: [*]const f64, b: [*]const f64, c: [*]f64, M: u32, N: u32, K: u32) void {

    if (N <= F64_CROSSOVER) {
        // For smaller N, the i-j-k order with accumulated SIMD is faster due to better cache reuse of C.
        matmul_f64_ijk(a, b, c, M, N, K);
    } else {
        // For larger N, the i-k-j order with SIMD in the k loop is faster due to better prefetching of B.
        matmul_f64_ikj(a, b, c, M, N, K);
    }
}

/// Computes C = A @ B for row-major f32 matrices.
/// A is (M x K), B is (K x N), C is (M x N).
/// Dispatches to ijk (N ≤ F32_CROSSOVER) or ikj (N > F32_CROSSOVER) based on benchmarked
/// crossover point — ijk reduces C-memory traffic but degrades under large N due to B-cache pressure.
export fn matmul_f32(a: [*]const f32, b: [*]const f32, c: [*]f32, M: u32, N: u32, K: u32) void {

    if (N <= F32_CROSSOVER) {
        // For smaller N, the i-j-k order with accumulated SIMD is faster due to better cache reuse of C.
        matmul_f32_ijk(a, b, c, M, N, K);
    } else {
        // For larger N, the i-k-j order with SIMD in the k loop is faster due to better prefetching of B.
        matmul_f32_ikj(a, b, c, M, N, K);
    }
}

/// Computes C = A @ B for row-major complex128 matrices (interleaved f64: [re, im, re, im, ...]).
/// M, N, K are element counts; the underlying f64 arrays are 2x that size.
/// Each output element:
///   re = sum(A[i,k].re*B[k,j].re - A[i,k].im*B[k,j].im)
///   im = sum(A[i,k].re*B[k,j].im + A[i,k].im*B[k,j].re)
export fn matmul_c128(a: [*]const f64, b: [*]const f64, c: [*]f64, M: u32, N: u32, K: u32) void {

    // TODO: implement i-j-k version with register accumulation; more complex but should give ~35-50% speedup at small N.
    // For now, just use i-k-j
    matmul_c128_ikj(a, b, c, M, N, K);
}

/// Computes C = A @ B for row-major complex64 matrices (interleaved f32: [re, im, re, im, ...]).
/// M, N, K are element counts; the underlying f32 arrays are 2x that size.
/// Each output element:
///   re = sum(A[i,k].re*B[k,j].re - A[i,k].im*B[k,j].im)
///   im = sum(A[i,k].re*B[k,j].im + A[i,k].im*B[k,j].re)
export fn matmul_c64(a: [*]const f32, b: [*]const f32, c: [*]f32, M: u32, N: u32, K: u32) void {

    // TODO: implement i-j-k version with register accumulation; more complex but should give ~35-50% speedup at small N.
    // For now, just use i-k-j
    matmul_c64_ikj(a, b, c, M, N, K);
}

// --- Implementations ---

/// Computes C = A @ B for row-major f64 matrices.
/// Uses i-k-j tiled blocking with SIMD vectorization in the inner k loop.
/// Preferred when N > F64_CROSSOVER — sequential B access degrades gracefully under cache pressure.
fn matmul_f64_ikj(a: [*]const f64, b: [*]const f64, c: [*]f64, M: u32, N: u32, K: u32) void {
    // Zero output
    @memset(c[0..@as(usize, M) * N], 0);

    // Tiled i-k-j loop (BLAS-style blocking)
    // Outer tile loop over i (rows of A and C)
    var ii: usize = 0;
    while (ii < M) : (ii += TILE_F64) {
        const i_end = @min(ii + TILE_F64, M);
    
        // Outer tile loop over k (columns of A, rows of B)
        var kk: usize = 0;
        while (kk < K) : (kk += TILE_F64) {
            const k_end = @min(kk + TILE_F64, K);
    
            // Outer tile loop over j (columns of B and C)
            var jj: usize = 0;
            while (jj < N) : (jj += TILE_F64) {
                const j_end = @min(jj + TILE_F64, N);

                // Inner tile: compute C[ii:i_end, jj:j_end ] += A[ii:i_end, kk:k_end] @ B[kk:k_end, jj:j_end]
                var i: usize = ii;
                while (i < i_end) : (i += 1) {

                    var k: usize = kk;
                    while (k < k_end) : (k += 1) {
                        const a_ik = a[i * K + k];
                        const a_vec: simd.V2f64 = @splat(a_ik);
                        const b_row = k * N;
                        const c_row = i * N;

                        // Vectorized j loop: two v128 (2×f64) per step = 4 f64
                        var j: usize = jj;
                        while (j + 4 <= j_end) : (j += 4) {
                            simd.store2_f64(c, c_row + j, simd.load2_f64(c, c_row + j) + a_vec * simd.load2_f64(b, b_row + j));
                            simd.store2_f64(c, c_row + j + 2, simd.load2_f64(c, c_row + j + 2) + a_vec * simd.load2_f64(b, b_row + j + 2));
                        }
                        // One more v128 if possible
                        while (j + 2 <= j_end) : (j += 2) {
                            simd.store2_f64(c, c_row + j, simd.load2_f64(c, c_row + j) + a_vec * simd.load2_f64(b, b_row + j));
                        }
                        // Scalar remainder
                        while (j < j_end) : (j += 1) {
                            c[c_row + j] += a_ik * b[b_row + j];
                        }
                    }
                }
            }
        }
    }
}

/// Computes C = A @ B for row-major f64 matrices.
/// Uses i-j-k accumulated tiled blocking with SIMD vectorization in the inner k loop.
/// Preferred when N ≤ F64_CROSSOVER — register accumulation gives ~35-50% speedup at small N.
fn matmul_f64_ijk(a: [*]const f64, b: [*]const f64, c: [*]f64, M: u32, N: u32, K: u32) void {
    // Zero output
    @memset(c[0..@as(usize, M) * N], 0);

    // Tiled i-j-k loop (BLAS-style blocking)
    // Outer tile loop over i (rows of A and C)
    var ii: usize = 0;
    while (ii < M) : (ii += TILE_F64) {
        const i_end = @min(ii + TILE_F64, M);
    
        // Outer tile loop over j (columns of B and C)
        var jj: usize = 0;
        while (jj < N) : (jj += TILE_F64) {
            const j_end = @min(jj + TILE_F64, N);

            // Outer tile loop over k (columns of A, rows of B)
            var kk: usize = 0;
            while (kk < K) : (kk += TILE_F64) {
                const k_end = @min(kk + TILE_F64, K);
    
                // Inner tile: compute C[ii:i_end, jj:j_end ] += A[ii:i_end, kk:k_end] @ B[kk:k_end, jj:j_end]
                var i: usize = ii;
                while (i < i_end) : (i += 1) {
                    const c_row = i * N;

                    // Vectorized j: accumulate across k into V2f64 registers, write C once
                    var j: usize = jj;
                    while (j + 4 <= j_end) : (j += 4) {
                        var acc0: simd.V2f64 = @splat(0.0);
                        var acc1: simd.V2f64 = @splat(0.0);
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            const a_vec: simd.V2f64 = @splat(a[i * K + k]);
                            const b_row = k * N;
                            acc0 += a_vec * simd.load2_f64(b, b_row + j);
                            acc1 += a_vec * simd.load2_f64(b, b_row + j + 2);
                        }
                        simd.store2_f64(c, c_row + j,     simd.load2_f64(c, c_row + j)     + acc0);
                        simd.store2_f64(c, c_row + j + 2, simd.load2_f64(c, c_row + j + 2) + acc1);
                    }
                    // One more v128 if possible
                    while (j + 2 <= j_end) : (j += 2) {
                        var acc: simd.V2f64 = @splat(0.0);
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            const a_vec: simd.V2f64 = @splat(a[i * K + k]);
                            acc += a_vec * simd.load2_f64(b, k * N + j);
                        }
                        simd.store2_f64(c, c_row + j, simd.load2_f64(c, c_row + j) + acc);
                    }
                    // Scalar remainder
                    while (j < j_end) : (j += 1) {
                        var acc: f64 = 0.0;
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            acc += a[i * K + k] * b[k * N + j];
                        }
                        c[c_row + j] += acc;
                    }
                }
            }
        }
    }
}

/// Computes C = A @ B for row-major f32 matrices.
/// Uses i-k-j tiled blocking with SIMD vectorization in the inner k loop.
/// Preferred when N > F32_CROSSOVER — sequential B access degrades gracefully under cache pressure.
fn matmul_f32_ikj(a: [*]const f32, b: [*]const f32, c: [*]f32, M: u32, N: u32, K: u32) void {
    // Zero output
    @memset(c[0..@as(usize, M) * N], 0);

    // Tiled i-k-j loop (BLAS-style blocking)
    // Outer tile loop over i (rows of A and C)
    var ii: usize = 0;
    while (ii < M) : (ii += TILE_F32) {
        const i_end = @min(ii + TILE_F32, M);
    
        // Outer tile loop over k (columns of A, rows of B)
        var kk: usize = 0;
        while (kk < K) : (kk += TILE_F32) {
            const k_end = @min(kk + TILE_F32, K);
    
            // Outer tile loop over j (columns of B and C)
            var jj: usize = 0;
            while (jj < N) : (jj += TILE_F32) {
                const j_end = @min(jj + TILE_F32, N);

                // Inner tile: compute C[ii:i_end, jj:j_end ] += A[ii:i_end, kk:k_end] @ B[kk:k_end, jj:j_end]
                var i: usize = ii;
                while (i < i_end) : (i += 1) {

                    var k: usize = kk;
                    while (k < k_end) : (k += 1) {
                        const a_ik = a[i * K + k];
                        const a_vec: simd.V4f32 = @splat(a_ik);
                        const b_row = k * N;
                        const c_row = i * N;

                        // Vectorized j loop: two v128 (4xf32) per step = 8 f32
                        var j: usize = jj;
                        while (j + 8 <= j_end) : (j += 8) {
                            simd.store4_f32(c, c_row + j, simd.load4_f32(c, c_row + j) + a_vec * simd.load4_f32(b, b_row + j));
                            simd.store4_f32(c, c_row + j + 4, simd.load4_f32(c, c_row + j + 4) + a_vec * simd.load4_f32(b, b_row + j + 4));
                        }
                        // One more v128 if possible
                        while (j + 4 <= j_end) : (j += 4) {
                            simd.store4_f32(c, c_row + j, simd.load4_f32(c, c_row + j) + a_vec * simd.load4_f32(b, b_row + j));
                        }
                        // Scalar remainder
                        while (j < j_end) : (j += 1) {
                            c[c_row + j] += a_ik * b[b_row + j];
                        }
                    }
                }
            }
        }
    }
}

/// Computes C = A @ B for row-major f32 matrices.
/// Uses i-j-k accumulated tiled blocking with SIMD vectorization in the inner k loop.
/// Preferred when N ≤ F32_CROSSOVER — register accumulation gives ~35-50% speedup at small N.
fn matmul_f32_ijk(a: [*]const f32, b: [*]const f32, c: [*]f32, M: u32, N: u32, K: u32) void {
    // Zero output
    @memset(c[0..@as(usize, M) * N], 0);

    // Tiled i-j-k loop (BLAS-style blocking)
    // Outer tile loop over i (rows of A and C)
    var ii: usize = 0;
    while (ii < M) : (ii += TILE_F32) {
        const i_end = @min(ii + TILE_F32, M);
    
        // Outer tile loop over j (columns of B and C)
        var jj: usize = 0;
        while (jj < N) : (jj += TILE_F32) {
            const j_end = @min(jj + TILE_F32, N);

            // Outer tile loop over k (columns of A, rows of B)
            var kk: usize = 0;
            while (kk < K) : (kk += TILE_F32) {
                const k_end = @min(kk + TILE_F32, K);
    
                // Inner tile: compute C[ii:i_end, jj:j_end ] += A[ii:i_end, kk:k_end] @ B[kk:k_end, jj:j_end]
                var i: usize = ii;
                while (i < i_end) : (i += 1) {
                    const c_row = i * N;

                    // Vectorized j: accumulate across k into V4f32 registers, write C once
                    var j: usize = jj;
                    while (j + 8 <= j_end) : (j += 8) {
                        var acc0: simd.V4f32 = @splat(0.0);
                        var acc1: simd.V4f32 = @splat(0.0);
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            const a_vec: simd.V4f32 = @splat(a[i * K + k]);
                            const b_row = k * N;
                            acc0 += a_vec * simd.load4_f32(b, b_row + j);
                            acc1 += a_vec * simd.load4_f32(b, b_row + j + 4);
                        }
                        simd.store4_f32(c, c_row + j,     simd.load4_f32(c, c_row + j)     + acc0);
                        simd.store4_f32(c, c_row + j + 4, simd.load4_f32(c, c_row + j + 4) + acc1);
                    }
                    // One more v128 if possible
                    while (j + 4 <= j_end) : (j += 4) {
                        var acc: simd.V4f32 = @splat(0.0);
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            const a_vec: simd.V4f32 = @splat(a[i * K + k]);
                            acc += a_vec * simd.load4_f32(b, k * N + j);
                        }
                        simd.store4_f32(c, c_row + j, simd.load4_f32(c, c_row + j) + acc);
                    }
                    // Scalar remainder
                    while (j < j_end) : (j += 1) {
                        var acc: f32 = 0.0;
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            acc += a[i * K + k] * b[k * N + j];
                        }
                        c[c_row + j] += acc;
                    }
                }
            }
        }
    }
}

/// Computes C = A @ B for row-major complex128 matrices (interleaved f64: [re, im, re, im, ...]).
/// Uses i-k-j tiled blocking with deinterleaved V2f64 SIMD in the inner j loop.
fn matmul_c128_ikj(a: [*]const f64, b: [*]const f64, c: [*]f64, M: u32, N: u32, K: u32) void {
    // Zero output
    @memset(c[0..@as(usize, M) * N * 2], 0);

    // Tiled i-k-j loop (BLAS-style blocking)
    // Outer tile loop over i (rows of A and C)
    var ii: usize = 0;
    while (ii < M) : (ii += TILE_F64) {
        const i_end = @min(ii + TILE_F64, M);

        // Outer tile loop over k (columns of A, rows of B)
        var kk: usize = 0;
        while (kk < K) : (kk += TILE_F64) {
            const k_end = @min(kk + TILE_F64, K);

            // Outer tile loop over j (columns of B and C)
            var jj: usize = 0;
            while (jj < N) : (jj += TILE_F64) {
                const j_end = @min(jj + TILE_F64, N);

                // Inner tile: C[ii:i_end, jj:j_end] += A[ii:i_end, kk:k_end] @ B[kk:k_end, jj:j_end]
                var i: usize = ii;
                while (i < i_end) : (i += 1) {

                    var k: usize = kk;
                    while (k < k_end) : (k += 1) {
                        // Load A[i,k] = (a_re, a_im) — one complex element = one v128
                        const a_base = (i * K + k) * 2;
                        const a_re: simd.V2f64 = @splat(a[a_base]);
                        const a_im: simd.V2f64 = @splat(a[a_base + 1]);

                        const b_row = k * N * 2;
                        const c_row = i * N * 2;

                        // Vectorized j loop: one v128 = one complex element (re, im)
                        // Two complex elements per step = 2 v128 ops
                        var j: usize = jj;
                        while (j + 2 <= j_end) : (j += 2) {
                            const bj = b_row + j * 2;
                            const cj = c_row + j * 2;

                            // Load B[k,j] and B[k,j+1]
                            const b0 = simd.load2_f64(b, bj);      // (b0.re, b0.im)
                            const b1 = simd.load2_f64(b, bj + 2);  // (b1.re, b1.im)

                            // Deinterleave via @shuffle — compiles to i64x2.shuffle on WASM
                            // b_re = (b0.re, b1.re), b_im = (b0.im, b1.im)
                            const b_re = @shuffle(f64, b0, b1, [2]i32{  0, -1 }); // b0[0], b1[0]
                            const b_im = @shuffle(f64, b0, b1, [2]i32{  1, -2 }); // b0[1], b1[1]

                            const c0 = simd.load2_f64(c, cj);
                            const c1 = simd.load2_f64(c, cj + 2);
                            const c_re = @shuffle(f64, c0, c1, [2]i32{  0, -1 }); // c0[0], c1[0]
                            const c_im = @shuffle(f64, c0, c1, [2]i32{  1, -2 }); // c0[1], c1[1]

                            const out_re = c_re + a_re * b_re - a_im * b_im;
                            const out_im = c_im + a_re * b_im + a_im * b_re;

                            // Reinterleave for store — interleave re and im back into (re, im) pairs
                            const out0 = @shuffle(f64, out_re, out_im, [2]i32{  0, -1 }); // (re[0], im[0])
                            const out1 = @shuffle(f64, out_re, out_im, [2]i32{  1, -2 }); // (re[1], im[1])

                            simd.store2_f64(c, cj,     out0);
                            simd.store2_f64(c, cj + 2, out1);
                        }
                        // Scalar remainder
                        while (j < j_end) : (j += 1) {
                            const bj = b_row + j * 2;
                            const cj = c_row + j * 2;
                            const b_re = b[bj];
                            const b_im = b[bj + 1];
                            c[cj]     += a[a_base] * b_re - a[a_base + 1] * b_im;
                            c[cj + 1] += a[a_base] * b_im + a[a_base + 1] * b_re;
                        }
                    }
                }
            }
        }
    }
}

/// Computes C = A @ B for row-major complex64 matrices (interleaved f32: [re, im, re, im, ...]).
/// Uses i-k-j tiled blocking with deinterleaved V4f32 SIMD in the inner j loop.
fn matmul_c64_ikj(a: [*]const f32, b: [*]const f32, c: [*]f32, M: u32, N: u32, K: u32) void {
    // Zero output
    @memset(c[0..@as(usize, M) * N * 2], 0);

    // Tiled i-k-j loop (BLAS-style blocking)
    // Outer tile loop over i (rows of A and C)
    var ii: usize = 0;
    while (ii < M) : (ii += TILE_F32) {
        const i_end = @min(ii + TILE_F32, M);

        // Outer tile loop over k (columns of A, rows of B)
        var kk: usize = 0;
        while (kk < K) : (kk += TILE_F32) {
            const k_end = @min(kk + TILE_F32, K);

            // Outer tile loop over j (columns of B and C)
            var jj: usize = 0;
            while (jj < N) : (jj += TILE_F32) {
                const j_end = @min(jj + TILE_F32, N);

                // Inner tile: C[ii:i_end, jj:j_end] += A[ii:i_end, kk:k_end] @ B[kk:k_end, jj:j_end]
                var i: usize = ii;
                while (i < i_end) : (i += 1) {

                    var k: usize = kk;
                    while (k < k_end) : (k += 1) {
                        // Load A[i,k] = (a_re, a_im) — one complex element = one v128
                        const a_base = (i * K + k) * 2;
                        const a_re: simd.V4f32 = @splat(a[a_base]);
                        const a_im: simd.V4f32 = @splat(a[a_base + 1]);

                        const b_row = k * N * 2;
                        const c_row = i * N * 2;

                        // Vectorized j loop: two v128 = two complex element (re, im)
                        // Four complex elements per step = 2 v128 ops
                        var j: usize = jj;
                        while (j + 4 <= j_end) : (j += 4) {
                            const bj = b_row + j * 2;
                            const cj = c_row + j * 2;

                            // Load 4 complex f32 elements across two v128s
                            const b0 = simd.load4_f32(b, bj);      // (b0.re, b0.im, b1.re, b1.im)
                            const b1 = simd.load4_f32(b, bj + 4);  // (b2.re, b2.im, b3.re, b3.im)

                            // Deinterleave — compiles to f32x4.shuffle on WASM
                            const b_re = @shuffle(f32, b0, b1, [4]i32{  0,  2, -1, -3 }); // b0.re, b1.re, b2.re, b3.re
                            const b_im = @shuffle(f32, b0, b1, [4]i32{  1,  3, -2, -4 }); // b0.im, b1.im, b2.im, b3.im

                            const c0 = simd.load4_f32(c, cj);
                            const c1 = simd.load4_f32(c, cj + 4);
                            const c_re = @shuffle(f32, c0, c1, [4]i32{  0,  2, -1, -3 }); // c0.re, c1.re, c2.re, c3.re
                            const c_im = @shuffle(f32, c0, c1, [4]i32{  1,  3, -2, -4 }); // c0.im, c1.im, c2.im, c3.im

                            const out_re = c_re + a_re * b_re - a_im * b_im;
                            const out_im = c_im + a_re * b_im + a_im * b_re;

                            // Reinterleave for store
                            const out0 = @shuffle(f32, out_re, out_im, [4]i32{  0, -1,  1, -2 }); // (re[0],im[0],re[1],im[1])
                            const out1 = @shuffle(f32, out_re, out_im, [4]i32{  2, -3,  3, -4 }); // (re[2],im[2],re[3],im[3])
                            simd.store4_f32(c, cj,     out0);
                            simd.store4_f32(c, cj + 4, out1);
                        }
                        // Scalar remainder
                        while (j < j_end) : (j += 1) {
                            const bj = b_row + j * 2;
                            const cj = c_row + j * 2;
                            const b_re = b[bj];
                            const b_im = b[bj + 1];
                            c[cj]     += a[a_base] * b_re - a[a_base + 1] * b_im;
                            c[cj + 1] += a[a_base] * b_im + a[a_base + 1] * b_re;
                        }
                    }
                }
            }
        }
    }
}

// --- Tests ---

test "matmul_f64_ikj 2x2" {
    const testing = @import("std").testing;
    // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    // C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    var a = [_]f64{ 1, 2, 3, 4 };
    var b = [_]f64{ 5, 6, 7, 8 };
    var c = [_]f64{ 0, 0, 0, 0 };
    matmul_f64_ikj(&a, &b, &c, 2, 2, 2);
    try testing.expectApproxEqAbs(c[0], 19.0, 1e-10);
    try testing.expectApproxEqAbs(c[1], 22.0, 1e-10);
    try testing.expectApproxEqAbs(c[2], 43.0, 1e-10);
    try testing.expectApproxEqAbs(c[3], 50.0, 1e-10);
}

test "matmul_f64_ikj non-square 2x3x4" {
    const testing = @import("std").testing;
    // A is 2×3, B is 3×4, C is 2×4
    // A = [[1,2,3],[4,5,6]], B = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
    // C[0,0] = 1*1+2*5+3*9 = 38,  C[0,1] = 1*2+2*6+3*10 = 44
    // C[0,2] = 1*3+2*7+3*11 = 50, C[0,3] = 1*4+2*8+3*12 = 56
    // C[1,0] = 4*1+5*5+6*9 = 83,  C[1,1] = 4*2+5*6+6*10 = 98
    // C[1,2] = 4*3+5*7+6*11 = 113, C[1,3] = 4*4+5*8+6*12 = 128
    var a = [_]f64{ 1, 2, 3, 4, 5, 6 };
    var b = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var c = [_]f64{ 0, 0, 0, 0, 0, 0, 0, 0 };
    matmul_f64_ikj(&a, &b, &c, 2, 4, 3);
    try testing.expectApproxEqAbs(c[0],  38.0, 1e-10);
    try testing.expectApproxEqAbs(c[1],  44.0, 1e-10);
    try testing.expectApproxEqAbs(c[2],  50.0, 1e-10);
    try testing.expectApproxEqAbs(c[3],  56.0, 1e-10);
    try testing.expectApproxEqAbs(c[4],  83.0, 1e-10);
    try testing.expectApproxEqAbs(c[5],  98.0, 1e-10);
    try testing.expectApproxEqAbs(c[6], 113.0, 1e-10);
    try testing.expectApproxEqAbs(c[7], 128.0, 1e-10);
}

test "matmul_f64_ikj odd N remainder" {
    const testing = @import("std").testing;
    // N=3 forces scalar remainder path (not divisible by 2 or 4)
    // A = [[1,2],[3,4]], B = [[1,2,3],[4,5,6]]
    // C[0,0]=9, C[0,1]=12, C[0,2]=15
    // C[1,0]=19, C[1,1]=26, C[1,2]=33
    var a = [_]f64{ 1, 2, 3, 4 };
    var b = [_]f64{ 1, 2, 3, 4, 5, 6 };
    var c = [_]f64{ 0, 0, 0, 0, 0, 0 };
    matmul_f64_ikj(&a, &b, &c, 2, 3, 2);
    try testing.expectApproxEqAbs(c[0],  9.0, 1e-10);
    try testing.expectApproxEqAbs(c[1], 12.0, 1e-10);
    try testing.expectApproxEqAbs(c[2], 15.0, 1e-10);
    try testing.expectApproxEqAbs(c[3], 19.0, 1e-10);
    try testing.expectApproxEqAbs(c[4], 26.0, 1e-10);
    try testing.expectApproxEqAbs(c[5], 33.0, 1e-10);
}

test "matmul_f64_ijk 2x2" {
    const testing = @import("std").testing;
    var a = [_]f64{ 1, 2, 3, 4 };
    var b = [_]f64{ 5, 6, 7, 8 };
    var c = [_]f64{ 0, 0, 0, 0 };
    matmul_f64_ijk(&a, &b, &c, 2, 2, 2);
    try testing.expectApproxEqAbs(c[0], 19.0, 1e-10);
    try testing.expectApproxEqAbs(c[1], 22.0, 1e-10);
    try testing.expectApproxEqAbs(c[2], 43.0, 1e-10);
    try testing.expectApproxEqAbs(c[3], 50.0, 1e-10);
}

test "matmul_f64_ijk non-square 2x4x3" {
    const testing = @import("std").testing;
    var a = [_]f64{ 1, 2, 3, 4, 5, 6 };
    var b = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var c = [_]f64{ 0, 0, 0, 0, 0, 0, 0, 0 };
    matmul_f64_ijk(&a, &b, &c, 2, 4, 3);
    try testing.expectApproxEqAbs(c[0],  38.0, 1e-10);
    try testing.expectApproxEqAbs(c[1],  44.0, 1e-10);
    try testing.expectApproxEqAbs(c[2],  50.0, 1e-10);
    try testing.expectApproxEqAbs(c[3],  56.0, 1e-10);
    try testing.expectApproxEqAbs(c[4],  83.0, 1e-10);
    try testing.expectApproxEqAbs(c[5],  98.0, 1e-10);
    try testing.expectApproxEqAbs(c[6], 113.0, 1e-10);
    try testing.expectApproxEqAbs(c[7], 128.0, 1e-10);
}

test "matmul_f64_ijk odd N remainder" {
    const testing = @import("std").testing;
    var a = [_]f64{ 1, 2, 3, 4 };
    var b = [_]f64{ 1, 2, 3, 4, 5, 6 };
    var c = [_]f64{ 0, 0, 0, 0, 0, 0 };
    matmul_f64_ijk(&a, &b, &c, 2, 3, 2);
    try testing.expectApproxEqAbs(c[0],  9.0, 1e-10);
    try testing.expectApproxEqAbs(c[1], 12.0, 1e-10);
    try testing.expectApproxEqAbs(c[2], 15.0, 1e-10);
    try testing.expectApproxEqAbs(c[3], 19.0, 1e-10);
    try testing.expectApproxEqAbs(c[4], 26.0, 1e-10);
    try testing.expectApproxEqAbs(c[5], 33.0, 1e-10);
}

// --- f32 ---

test "matmul_f32_ikj 2x2" {
    const testing = @import("std").testing;
    var a = [_]f32{ 1, 2, 3, 4 };
    var b = [_]f32{ 5, 6, 7, 8 };
    var c = [_]f32{ 0, 0, 0, 0 };
    matmul_f32_ikj(&a, &b, &c, 2, 2, 2);
    try testing.expectApproxEqAbs(c[0], 19.0, 1e-5);
    try testing.expectApproxEqAbs(c[1], 22.0, 1e-5);
    try testing.expectApproxEqAbs(c[2], 43.0, 1e-5);
    try testing.expectApproxEqAbs(c[3], 50.0, 1e-5);
}

test "matmul_f32_ikj non-square 2x4x3" {
    const testing = @import("std").testing;
    var a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var b = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var c = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    matmul_f32_ikj(&a, &b, &c, 2, 4, 3);
    try testing.expectApproxEqAbs(c[0],  38.0, 1e-5);
    try testing.expectApproxEqAbs(c[1],  44.0, 1e-5);
    try testing.expectApproxEqAbs(c[2],  50.0, 1e-5);
    try testing.expectApproxEqAbs(c[3],  56.0, 1e-5);
    try testing.expectApproxEqAbs(c[4],  83.0, 1e-5);
    try testing.expectApproxEqAbs(c[5],  98.0, 1e-5);
    try testing.expectApproxEqAbs(c[6], 113.0, 1e-5);
    try testing.expectApproxEqAbs(c[7], 128.0, 1e-5);
}

test "matmul_f32_ikj odd N remainder" {
    const testing = @import("std").testing;
    var a = [_]f32{ 1, 2, 3, 4 };
    var b = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var c = [_]f32{ 0, 0, 0, 0, 0, 0 };
    matmul_f32_ikj(&a, &b, &c, 2, 3, 2);
    try testing.expectApproxEqAbs(c[0],  9.0, 1e-5);
    try testing.expectApproxEqAbs(c[1], 12.0, 1e-5);
    try testing.expectApproxEqAbs(c[2], 15.0, 1e-5);
    try testing.expectApproxEqAbs(c[3], 19.0, 1e-5);
    try testing.expectApproxEqAbs(c[4], 26.0, 1e-5);
    try testing.expectApproxEqAbs(c[5], 33.0, 1e-5);
}

test "matmul_f32_ijk 2x2" {
    const testing = @import("std").testing;
    var a = [_]f32{ 1, 2, 3, 4 };
    var b = [_]f32{ 5, 6, 7, 8 };
    var c = [_]f32{ 0, 0, 0, 0 };
    matmul_f32_ijk(&a, &b, &c, 2, 2, 2);
    try testing.expectApproxEqAbs(c[0], 19.0, 1e-5);
    try testing.expectApproxEqAbs(c[1], 22.0, 1e-5);
    try testing.expectApproxEqAbs(c[2], 43.0, 1e-5);
    try testing.expectApproxEqAbs(c[3], 50.0, 1e-5);
}

test "matmul_f32_ijk non-square 2x4x3" {
    const testing = @import("std").testing;
    var a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var b = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var c = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    matmul_f32_ijk(&a, &b, &c, 2, 4, 3);
    try testing.expectApproxEqAbs(c[0],  38.0, 1e-5);
    try testing.expectApproxEqAbs(c[1],  44.0, 1e-5);
    try testing.expectApproxEqAbs(c[2],  50.0, 1e-5);
    try testing.expectApproxEqAbs(c[3],  56.0, 1e-5);
    try testing.expectApproxEqAbs(c[4],  83.0, 1e-5);
    try testing.expectApproxEqAbs(c[5],  98.0, 1e-5);
    try testing.expectApproxEqAbs(c[6], 113.0, 1e-5);
    try testing.expectApproxEqAbs(c[7], 128.0, 1e-5);
}

test "matmul_f32_ijk odd N remainder" {
    const testing = @import("std").testing;
    var a = [_]f32{ 1, 2, 3, 4 };
    var b = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var c = [_]f32{ 0, 0, 0, 0, 0, 0 };
    matmul_f32_ijk(&a, &b, &c, 2, 3, 2);
    try testing.expectApproxEqAbs(c[0],  9.0, 1e-5);
    try testing.expectApproxEqAbs(c[1], 12.0, 1e-5);
    try testing.expectApproxEqAbs(c[2], 15.0, 1e-5);
    try testing.expectApproxEqAbs(c[3], 19.0, 1e-5);
    try testing.expectApproxEqAbs(c[4], 26.0, 1e-5);
    try testing.expectApproxEqAbs(c[5], 33.0, 1e-5);
}

// --- c128 ---
// Matrices stored as interleaved [re, im, re, im, ...]
// Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i

test "matmul_c128_ikj 2x2" {
    const testing = @import("std").testing;
    // A = [[1+2i, 3+4i], [5+6i, 7+8i]]
    // B = [[1+1i, 0+1i], [1+0i, 1+1i]]
    // C[0,0] = (1+2i)(1+1i) + (3+4i)(1+0i) = (-1+3i) + (3+4i) = 2+7i
    // C[0,1] = (1+2i)(0+1i) + (3+4i)(1+1i) = (-2+1i) + (-1+7i) = -3+8i
    // C[1,0] = (5+6i)(1+1i) + (7+8i)(1+0i) = (-1+11i) + (7+8i) = 6+19i
    // C[1,1] = (5+6i)(0+1i) + (7+8i)(1+1i) = (-6+5i) + (-1+15i) = -7+20i
    var a = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var b = [_]f64{ 1, 1, 0, 1, 1, 0, 1, 1 };
    var c = [_]f64{ 0, 0, 0, 0, 0, 0, 0, 0 };
    matmul_c128_ikj(&a, &b, &c, 2, 2, 2);
    try testing.expectApproxEqAbs(c[0],  2.0, 1e-10); // C[0,0].re
    try testing.expectApproxEqAbs(c[1],  7.0, 1e-10); // C[0,0].im
    try testing.expectApproxEqAbs(c[2], -3.0, 1e-10); // C[0,1].re
    try testing.expectApproxEqAbs(c[3],  8.0, 1e-10); // C[0,1].im
    try testing.expectApproxEqAbs(c[4],  6.0, 1e-10); // C[1,0].re
    try testing.expectApproxEqAbs(c[5], 19.0, 1e-10); // C[1,0].im
    try testing.expectApproxEqAbs(c[6], -7.0, 1e-10); // C[1,1].re
    try testing.expectApproxEqAbs(c[7], 20.0, 1e-10); // C[1,1].im
}

test "matmul_c128_ikj scalar remainder" {
    const testing = @import("std").testing;
    // N=1 forces scalar remainder path
    // A = [[1+2i]], B = [[3+4i]], C = [[(1+2i)(3+4i)]] = [[-5+10i]]
    var a = [_]f64{ 1, 2 };
    var b = [_]f64{ 3, 4 };
    var c = [_]f64{ 0, 0 };
    matmul_c128_ikj(&a, &b, &c, 1, 1, 1);
    try testing.expectApproxEqAbs(c[0], -5.0, 1e-10);
    try testing.expectApproxEqAbs(c[1], 10.0, 1e-10);
}

test "matmul_c128_ikj non-square 2x3x2" {
    const testing = @import("std").testing;
    // A is 2×2, B is 2×3, C is 2×3 — exercises non-square stride with complex
    // A = [[1+0i, 0+1i], [1+1i, 1-1i]]
    // B = [[1+0i, 0+1i, 1+1i], [1+0i, 1+0i, 0+1i]]
    // C[0,0] = (1)(1)+(0+1i)(1) = 1 + i = 1+1i
    // C[0,1] = (1)(0+1i)+(0+1i)(1) = i+i = 0+2i
    // C[0,2] = (1)(1+1i)+(0+1i)(0+1i) = 1+1i+(-1+0i) = 0+1i
    // C[1,0] = (1+1i)(1)+(1-1i)(1) = 1+1i+1-1i = 2+0i
    // C[1,1] = (1+1i)(0+1i)+(1-1i)(1) = -1+1i+1-1i = 0+0i
    // C[1,2] = (1+1i)(1+1i)+(1-1i)(0+1i) = 2i+(1+1i) = 1+3i
    var a = [_]f64{ 1, 0, 0, 1, 1, 1, 1, -1 };
    var b = [_]f64{ 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1 };
    var c = [_]f64{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    matmul_c128_ikj(&a, &b, &c, 2, 3, 2);
    try testing.expectApproxEqAbs(c[0],  1.0, 1e-10); // C[0,0].re
    try testing.expectApproxEqAbs(c[1],  1.0, 1e-10); // C[0,0].im
    try testing.expectApproxEqAbs(c[2],  0.0, 1e-10); // C[0,1].re
    try testing.expectApproxEqAbs(c[3],  2.0, 1e-10); // C[0,1].im
    try testing.expectApproxEqAbs(c[4],  0.0, 1e-10); // C[0,2].re
    try testing.expectApproxEqAbs(c[5],  1.0, 1e-10); // C[0,2].im
    try testing.expectApproxEqAbs(c[6],  2.0, 1e-10); // C[1,0].re
    try testing.expectApproxEqAbs(c[7],  0.0, 1e-10); // C[1,0].im
    try testing.expectApproxEqAbs(c[8],  0.0, 1e-10); // C[1,1].re
    try testing.expectApproxEqAbs(c[9],  0.0, 1e-10); // C[1,1].im
    try testing.expectApproxEqAbs(c[10], 1.0, 1e-10); // C[1,2].re
    try testing.expectApproxEqAbs(c[11], 3.0, 1e-10); // C[1,2].im
}

// --- c64 ---

test "matmul_c64_ikj 2x2" {
    const testing = @import("std").testing;
    // Same as c128 2x2 test but f32
    var a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var b = [_]f32{ 1, 1, 0, 1, 1, 0, 1, 1 };
    var c = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    matmul_c64_ikj(&a, &b, &c, 2, 2, 2);
    try testing.expectApproxEqAbs(c[0],  2.0, 1e-5);
    try testing.expectApproxEqAbs(c[1],  7.0, 1e-5);
    try testing.expectApproxEqAbs(c[2], -3.0, 1e-5);
    try testing.expectApproxEqAbs(c[3],  8.0, 1e-5);
    try testing.expectApproxEqAbs(c[4],  6.0, 1e-5);
    try testing.expectApproxEqAbs(c[5], 19.0, 1e-5);
    try testing.expectApproxEqAbs(c[6], -7.0, 1e-5);
    try testing.expectApproxEqAbs(c[7], 20.0, 1e-5);
}

test "matmul_c64_ikj scalar remainder" {
    const testing = @import("std").testing;
    // N=1 forces scalar remainder — (1+2i)(3+4i) = -5+10i
    var a = [_]f32{ 1, 2 };
    var b = [_]f32{ 3, 4 };
    var c = [_]f32{ 0, 0 };
    matmul_c64_ikj(&a, &b, &c, 1, 1, 1);
    try testing.expectApproxEqAbs(c[0], -5.0, 1e-5);
    try testing.expectApproxEqAbs(c[1], 10.0, 1e-5);
}

test "matmul_c64_ikj SIMD path N=4" {
    const testing = @import("std").testing;
    // N=4 exercises the full 4-wide SIMD path in c64
    // A = [[1+0i]], B = [[1+0i, 0+1i, 1+1i, 1-1i]]
    // C = [[(1)(1+0i), (1)(0+1i), (1)(1+1i), (1)(1-1i)]]
    //   = [[1+0i, 0+1i, 1+1i, 1-1i]]
    var a = [_]f32{ 1, 0 };
    var b = [_]f32{ 1, 0, 0, 1, 1, 1, 1, -1 };
    var c = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    matmul_c64_ikj(&a, &b, &c, 1, 4, 1);
    try testing.expectApproxEqAbs(c[0],  1.0, 1e-5);
    try testing.expectApproxEqAbs(c[1],  0.0, 1e-5);
    try testing.expectApproxEqAbs(c[2],  0.0, 1e-5);
    try testing.expectApproxEqAbs(c[3],  1.0, 1e-5);
    try testing.expectApproxEqAbs(c[4],  1.0, 1e-5);
    try testing.expectApproxEqAbs(c[5],  1.0, 1e-5);
    try testing.expectApproxEqAbs(c[6],  1.0, 1e-5);
    try testing.expectApproxEqAbs(c[7], -1.0, 1e-5);
}