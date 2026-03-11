//! WASM matmul kernels for f32, f64, complex64, complex128, and integer types.
//!
//! Convention: C = A @ B where A is (M x K), B is (K x N), C is (M x N).
//! All matrices are row-major (C-contiguous).
//! Complex matrices are interleaved [re, im, re, im, ...]; M, N, K are element counts.
//!
//! Uses 4×N register-blocked micro-kernels for real types (f64, f32, i64, i32, i16).
//! Processes 4 rows of i simultaneously — each B load is reused across all 4 rows,

const simd = @import("simd.zig");

const TILE_F64 = 64; // Tile size for f64/i64 matmul (tuned for WASM v128)
const TILE_F32 = 128; // Tile size for f32/i32/i16 matmul (tuned for WASM v128)

/// Computes C = A @ B for row-major f64 matrices.
/// A is (M x K), B is (K x N), C is (M x N).
/// Uses a 4×N register-blocked micro-kernel with 4-wide SIMD for the main loop, then 2-wide and scalar remainders.
export fn matmul_f64(a: [*]const f64, b: [*]const f64, c: [*]f64, M: u32, N: u32, K: u32) void {
    @memset(c[0 .. @as(usize, M) * N], 0);

    var ii: usize = 0;
    while (ii < M) : (ii += TILE_F64) {
        const i_end = @min(ii + TILE_F64, M);
        var jj: usize = 0;
        while (jj < N) : (jj += TILE_F64) {
            const j_end = @min(jj + TILE_F64, N);
            var kk: usize = 0;
            while (kk < K) : (kk += TILE_F64) {
                const k_end = @min(kk + TILE_F64, K);

                // 4-row micro-kernel
                var i: usize = ii;
                while (i + 4 <= i_end) : (i += 4) {
                    var j: usize = jj;
                    while (j + 4 <= j_end) : (j += 4) {
                        var acc00: simd.V2f64 = @splat(0.0);
                        var acc01: simd.V2f64 = @splat(0.0);
                        var acc10: simd.V2f64 = @splat(0.0);
                        var acc11: simd.V2f64 = @splat(0.0);
                        var acc20: simd.V2f64 = @splat(0.0);
                        var acc21: simd.V2f64 = @splat(0.0);
                        var acc30: simd.V2f64 = @splat(0.0);
                        var acc31: simd.V2f64 = @splat(0.0);

                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            const b_row = k * N;
                            const b0 = simd.load2_f64(b, b_row + j);
                            const b1 = simd.load2_f64(b, b_row + j + 2);
                            const a0: simd.V2f64 = @splat(a[(i + 0) * K + k]);
                            acc00 += a0 * b0;
                            acc01 += a0 * b1;
                            const a1: simd.V2f64 = @splat(a[(i + 1) * K + k]);
                            acc10 += a1 * b0;
                            acc11 += a1 * b1;
                            const a2: simd.V2f64 = @splat(a[(i + 2) * K + k]);
                            acc20 += a2 * b0;
                            acc21 += a2 * b1;
                            const a3: simd.V2f64 = @splat(a[(i + 3) * K + k]);
                            acc30 += a3 * b0;
                            acc31 += a3 * b1;
                        }

                        const c0 = (i + 0) * N;
                        simd.store2_f64(c, c0 + j, simd.load2_f64(c, c0 + j) + acc00);
                        simd.store2_f64(c, c0 + j + 2, simd.load2_f64(c, c0 + j + 2) + acc01);
                        const c1 = (i + 1) * N;
                        simd.store2_f64(c, c1 + j, simd.load2_f64(c, c1 + j) + acc10);
                        simd.store2_f64(c, c1 + j + 2, simd.load2_f64(c, c1 + j + 2) + acc11);
                        const c2 = (i + 2) * N;
                        simd.store2_f64(c, c2 + j, simd.load2_f64(c, c2 + j) + acc20);
                        simd.store2_f64(c, c2 + j + 2, simd.load2_f64(c, c2 + j + 2) + acc21);
                        const c3 = (i + 3) * N;
                        simd.store2_f64(c, c3 + j, simd.load2_f64(c, c3 + j) + acc30);
                        simd.store2_f64(c, c3 + j + 2, simd.load2_f64(c, c3 + j + 2) + acc31);
                    }

                    // Remainder columns: 2-wide
                    while (j + 2 <= j_end) : (j += 2) {
                        var r0: simd.V2f64 = @splat(0.0);
                        var r1: simd.V2f64 = @splat(0.0);
                        var r2: simd.V2f64 = @splat(0.0);
                        var r3: simd.V2f64 = @splat(0.0);
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            const bv = simd.load2_f64(b, k * N + j);
                            r0 += @as(simd.V2f64, @splat(a[(i + 0) * K + k])) * bv;
                            r1 += @as(simd.V2f64, @splat(a[(i + 1) * K + k])) * bv;
                            r2 += @as(simd.V2f64, @splat(a[(i + 2) * K + k])) * bv;
                            r3 += @as(simd.V2f64, @splat(a[(i + 3) * K + k])) * bv;
                        }
                        simd.store2_f64(c, (i + 0) * N + j, simd.load2_f64(c, (i + 0) * N + j) + r0);
                        simd.store2_f64(c, (i + 1) * N + j, simd.load2_f64(c, (i + 1) * N + j) + r1);
                        simd.store2_f64(c, (i + 2) * N + j, simd.load2_f64(c, (i + 2) * N + j) + r2);
                        simd.store2_f64(c, (i + 3) * N + j, simd.load2_f64(c, (i + 3) * N + j) + r3);
                    }
                    // Remainder columns: scalar
                    while (j < j_end) : (j += 1) {
                        var s0: f64 = 0;
                        var s1: f64 = 0;
                        var s2: f64 = 0;
                        var s3: f64 = 0;
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            const bv = b[k * N + j];
                            s0 += a[(i + 0) * K + k] * bv;
                            s1 += a[(i + 1) * K + k] * bv;
                            s2 += a[(i + 2) * K + k] * bv;
                            s3 += a[(i + 3) * K + k] * bv;
                        }
                        c[(i + 0) * N + j] += s0;
                        c[(i + 1) * N + j] += s1;
                        c[(i + 2) * N + j] += s2;
                        c[(i + 3) * N + j] += s3;
                    }
                }

                // Remainder rows: 1-row accumulation
                while (i < i_end) : (i += 1) {
                    const c_row = i * N;
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
                        simd.store2_f64(c, c_row + j, simd.load2_f64(c, c_row + j) + acc0);
                        simd.store2_f64(c, c_row + j + 2, simd.load2_f64(c, c_row + j + 2) + acc1);
                    }
                    while (j + 2 <= j_end) : (j += 2) {
                        var acc: simd.V2f64 = @splat(0.0);
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            acc += @as(simd.V2f64, @splat(a[i * K + k])) * simd.load2_f64(b, k * N + j);
                        }
                        simd.store2_f64(c, c_row + j, simd.load2_f64(c, c_row + j) + acc);
                    }
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
/// A is (M x K), B is (K x N), C is (M x N).
/// Uses a 4×N register-blocked micro-kernel with 4-wide SIMD for the main loop, then 2-wide and scalar remainders.
export fn matmul_f32(a: [*]const f32, b: [*]const f32, c: [*]f32, M: u32, N: u32, K: u32) void {
    @memset(c[0 .. @as(usize, M) * N], 0);

    var ii: usize = 0;
    while (ii < M) : (ii += TILE_F32) {
        const i_end = @min(ii + TILE_F32, M);
        var jj: usize = 0;
        while (jj < N) : (jj += TILE_F32) {
            const j_end = @min(jj + TILE_F32, N);
            var kk: usize = 0;
            while (kk < K) : (kk += TILE_F32) {
                const k_end = @min(kk + TILE_F32, K);

                var i: usize = ii;
                while (i + 4 <= i_end) : (i += 4) {
                    var j: usize = jj;
                    while (j + 8 <= j_end) : (j += 8) {
                        var acc00: simd.V4f32 = @splat(0.0);
                        var acc01: simd.V4f32 = @splat(0.0);
                        var acc10: simd.V4f32 = @splat(0.0);
                        var acc11: simd.V4f32 = @splat(0.0);
                        var acc20: simd.V4f32 = @splat(0.0);
                        var acc21: simd.V4f32 = @splat(0.0);
                        var acc30: simd.V4f32 = @splat(0.0);
                        var acc31: simd.V4f32 = @splat(0.0);

                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            const b_row = k * N;
                            const b0 = simd.load4_f32(b, b_row + j);
                            const b1 = simd.load4_f32(b, b_row + j + 4);
                            const a0: simd.V4f32 = @splat(a[(i + 0) * K + k]);
                            acc00 += a0 * b0;
                            acc01 += a0 * b1;
                            const a1: simd.V4f32 = @splat(a[(i + 1) * K + k]);
                            acc10 += a1 * b0;
                            acc11 += a1 * b1;
                            const a2: simd.V4f32 = @splat(a[(i + 2) * K + k]);
                            acc20 += a2 * b0;
                            acc21 += a2 * b1;
                            const a3: simd.V4f32 = @splat(a[(i + 3) * K + k]);
                            acc30 += a3 * b0;
                            acc31 += a3 * b1;
                        }

                        const c0 = (i + 0) * N;
                        simd.store4_f32(c, c0 + j, simd.load4_f32(c, c0 + j) + acc00);
                        simd.store4_f32(c, c0 + j + 4, simd.load4_f32(c, c0 + j + 4) + acc01);
                        const c1 = (i + 1) * N;
                        simd.store4_f32(c, c1 + j, simd.load4_f32(c, c1 + j) + acc10);
                        simd.store4_f32(c, c1 + j + 4, simd.load4_f32(c, c1 + j + 4) + acc11);
                        const c2 = (i + 2) * N;
                        simd.store4_f32(c, c2 + j, simd.load4_f32(c, c2 + j) + acc20);
                        simd.store4_f32(c, c2 + j + 4, simd.load4_f32(c, c2 + j + 4) + acc21);
                        const c3 = (i + 3) * N;
                        simd.store4_f32(c, c3 + j, simd.load4_f32(c, c3 + j) + acc30);
                        simd.store4_f32(c, c3 + j + 4, simd.load4_f32(c, c3 + j + 4) + acc31);
                    }

                    // Remainder columns: 4-wide
                    while (j + 4 <= j_end) : (j += 4) {
                        var r0: simd.V4f32 = @splat(0.0);
                        var r1: simd.V4f32 = @splat(0.0);
                        var r2: simd.V4f32 = @splat(0.0);
                        var r3: simd.V4f32 = @splat(0.0);
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            const bv = simd.load4_f32(b, k * N + j);
                            r0 += @as(simd.V4f32, @splat(a[(i + 0) * K + k])) * bv;
                            r1 += @as(simd.V4f32, @splat(a[(i + 1) * K + k])) * bv;
                            r2 += @as(simd.V4f32, @splat(a[(i + 2) * K + k])) * bv;
                            r3 += @as(simd.V4f32, @splat(a[(i + 3) * K + k])) * bv;
                        }
                        simd.store4_f32(c, (i + 0) * N + j, simd.load4_f32(c, (i + 0) * N + j) + r0);
                        simd.store4_f32(c, (i + 1) * N + j, simd.load4_f32(c, (i + 1) * N + j) + r1);
                        simd.store4_f32(c, (i + 2) * N + j, simd.load4_f32(c, (i + 2) * N + j) + r2);
                        simd.store4_f32(c, (i + 3) * N + j, simd.load4_f32(c, (i + 3) * N + j) + r3);
                    }
                    // Remainder columns: scalar
                    while (j < j_end) : (j += 1) {
                        var s0: f32 = 0;
                        var s1: f32 = 0;
                        var s2: f32 = 0;
                        var s3: f32 = 0;
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            const bv = b[k * N + j];
                            s0 += a[(i + 0) * K + k] * bv;
                            s1 += a[(i + 1) * K + k] * bv;
                            s2 += a[(i + 2) * K + k] * bv;
                            s3 += a[(i + 3) * K + k] * bv;
                        }
                        c[(i + 0) * N + j] += s0;
                        c[(i + 1) * N + j] += s1;
                        c[(i + 2) * N + j] += s2;
                        c[(i + 3) * N + j] += s3;
                    }
                }

                // Remainder rows
                while (i < i_end) : (i += 1) {
                    const c_row = i * N;
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
                        simd.store4_f32(c, c_row + j, simd.load4_f32(c, c_row + j) + acc0);
                        simd.store4_f32(c, c_row + j + 4, simd.load4_f32(c, c_row + j + 4) + acc1);
                    }
                    while (j + 4 <= j_end) : (j += 4) {
                        var acc: simd.V4f32 = @splat(0.0);
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            acc += @as(simd.V4f32, @splat(a[i * K + k])) * simd.load4_f32(b, k * N + j);
                        }
                        simd.store4_f32(c, c_row + j, simd.load4_f32(c, c_row + j) + acc);
                    }
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
/// M, N, K are element counts; the underlying f64 arrays are 2x that size.
/// Register-accumulate strategy: accumulates re/im in SIMD registers over the k-loop,
/// only reading/writing C once per (i,j) per k-tile. Eliminates ~80% of shuffles vs
/// the load-C-every-k approach. Uses i-j-k loop order with 2-row blocking.
export fn matmul_c128(a: [*]const f64, b: [*]const f64, c: [*]f64, M: u32, N: u32, K: u32) void {
    const TILE = TILE_F64 / 2; // Smaller tile for complex (2x physical size)
    @memset(c[0 .. @as(usize, M) * N * 2], 0);

    var ii: usize = 0;
    while (ii < M) : (ii += TILE) {
        const i_end = @min(ii + TILE, M);
        var jj: usize = 0;
        while (jj < N) : (jj += TILE) {
            const j_end = @min(jj + TILE, N);

            // 2-row micro-kernel
            var i: usize = ii;
            while (i + 2 <= i_end) : (i += 2) {
                const c0_row = i * N * 2;
                const c1_row = (i + 1) * N * 2;

                var j: usize = jj;
                while (j + 2 <= j_end) : (j += 2) {
                    // Accumulate in registers over ALL k
                    var acc0_re: simd.V2f64 = @splat(0);
                    var acc0_im: simd.V2f64 = @splat(0);
                    var acc1_re: simd.V2f64 = @splat(0);
                    var acc1_im: simd.V2f64 = @splat(0);

                    for (0..K) |k| {
                        const a0_base = (i * K + k) * 2;
                        const a1_base = ((i + 1) * K + k) * 2;
                        const a0_re: simd.V2f64 = @splat(a[a0_base]);
                        const a0_im: simd.V2f64 = @splat(a[a0_base + 1]);
                        const a1_re: simd.V2f64 = @splat(a[a1_base]);
                        const a1_im: simd.V2f64 = @splat(a[a1_base + 1]);

                        const bj = k * N * 2 + j * 2;
                        const b0 = simd.load2_f64(b, bj);
                        const b1 = simd.load2_f64(b, bj + 2);
                        const b_re = @shuffle(f64, b0, b1, [2]i32{ 0, -1 });
                        const b_im = @shuffle(f64, b0, b1, [2]i32{ 1, -2 });

                        acc0_re += a0_re * b_re - a0_im * b_im;
                        acc0_im += a0_re * b_im + a0_im * b_re;
                        acc1_re += a1_re * b_re - a1_im * b_im;
                        acc1_im += a1_re * b_im + a1_im * b_re;
                    }

                    // Write C once — interleave and store
                    const cj0 = c0_row + j * 2;
                    simd.store2_f64(c, cj0, @shuffle(f64, acc0_re, acc0_im, [2]i32{ 0, -1 }));
                    simd.store2_f64(c, cj0 + 2, @shuffle(f64, acc0_re, acc0_im, [2]i32{ 1, -2 }));
                    const cj1 = c1_row + j * 2;
                    simd.store2_f64(c, cj1, @shuffle(f64, acc1_re, acc1_im, [2]i32{ 0, -1 }));
                    simd.store2_f64(c, cj1 + 2, @shuffle(f64, acc1_re, acc1_im, [2]i32{ 1, -2 }));
                }
                // Scalar remainder for j
                while (j < j_end) : (j += 1) {
                    var s0_re: f64 = 0;
                    var s0_im: f64 = 0;
                    var s1_re: f64 = 0;
                    var s1_im: f64 = 0;
                    for (0..K) |k| {
                        const a0 = (i * K + k) * 2;
                        const a1 = ((i + 1) * K + k) * 2;
                        const bk = (k * N + j) * 2;
                        const br = b[bk];
                        const bi = b[bk + 1];
                        s0_re += a[a0] * br - a[a0 + 1] * bi;
                        s0_im += a[a0] * bi + a[a0 + 1] * br;
                        s1_re += a[a1] * br - a[a1 + 1] * bi;
                        s1_im += a[a1] * bi + a[a1 + 1] * br;
                    }
                    const cj0 = c0_row + j * 2;
                    c[cj0] = s0_re;
                    c[cj0 + 1] = s0_im;
                    const cj1 = c1_row + j * 2;
                    c[cj1] = s1_re;
                    c[cj1 + 1] = s1_im;
                }
            }
            // Remainder row (odd M)
            while (i < i_end) : (i += 1) {
                const c_row = i * N * 2;

                var j: usize = jj;
                while (j + 2 <= j_end) : (j += 2) {
                    var acc_re: simd.V2f64 = @splat(0);
                    var acc_im: simd.V2f64 = @splat(0);

                    for (0..K) |k| {
                        const a_base = (i * K + k) * 2;
                        const a_re: simd.V2f64 = @splat(a[a_base]);
                        const a_im: simd.V2f64 = @splat(a[a_base + 1]);
                        const bj = k * N * 2 + j * 2;
                        const b0 = simd.load2_f64(b, bj);
                        const b1 = simd.load2_f64(b, bj + 2);
                        const b_re = @shuffle(f64, b0, b1, [2]i32{ 0, -1 });
                        const b_im = @shuffle(f64, b0, b1, [2]i32{ 1, -2 });
                        acc_re += a_re * b_re - a_im * b_im;
                        acc_im += a_re * b_im + a_im * b_re;
                    }

                    const cj = c_row + j * 2;
                    simd.store2_f64(c, cj, @shuffle(f64, acc_re, acc_im, [2]i32{ 0, -1 }));
                    simd.store2_f64(c, cj + 2, @shuffle(f64, acc_re, acc_im, [2]i32{ 1, -2 }));
                }
                while (j < j_end) : (j += 1) {
                    var s_re: f64 = 0;
                    var s_im: f64 = 0;
                    for (0..K) |k| {
                        const a0 = (i * K + k) * 2;
                        const bk = (k * N + j) * 2;
                        s_re += a[a0] * b[bk] - a[a0 + 1] * b[bk + 1];
                        s_im += a[a0] * b[bk + 1] + a[a0 + 1] * b[bk];
                    }
                    const cj = c_row + j * 2;
                    c[cj] = s_re;
                    c[cj + 1] = s_im;
                }
            }
        }
    }
}

/// Computes C = A @ B for row-major complex64 matrices (interleaved f32: [re, im, re, im, ...]).
/// M, N, K are element counts; the underlying f32 arrays are 2x that size.
/// Register-accumulate strategy: accumulates re/im in SIMD registers over the k-loop,
/// only reading/writing C once per (i,j) per k-tile. Uses i-j-k loop order with 2-row blocking.
export fn matmul_c64(a: [*]const f32, b: [*]const f32, c: [*]f32, M: u32, N: u32, K: u32) void {
    const TILE = TILE_F32 / 2; // Smaller tile for complex (2x physical size)
    @memset(c[0 .. @as(usize, M) * N * 2], 0);

    var ii: usize = 0;
    while (ii < M) : (ii += TILE) {
        const i_end = @min(ii + TILE, M);
        var jj: usize = 0;
        while (jj < N) : (jj += TILE) {
            const j_end = @min(jj + TILE, N);

            // 2-row micro-kernel
            var i: usize = ii;
            while (i + 2 <= i_end) : (i += 2) {
                const c0_row = i * N * 2;
                const c1_row = (i + 1) * N * 2;

                var j: usize = jj;
                while (j + 4 <= j_end) : (j += 4) {
                    // Accumulate in registers over ALL k
                    var acc0_re: simd.V4f32 = @splat(0);
                    var acc0_im: simd.V4f32 = @splat(0);
                    var acc1_re: simd.V4f32 = @splat(0);
                    var acc1_im: simd.V4f32 = @splat(0);

                    for (0..K) |k| {
                        const a0_base = (i * K + k) * 2;
                        const a1_base = ((i + 1) * K + k) * 2;
                        const a0_re: simd.V4f32 = @splat(a[a0_base]);
                        const a0_im: simd.V4f32 = @splat(a[a0_base + 1]);
                        const a1_re: simd.V4f32 = @splat(a[a1_base]);
                        const a1_im: simd.V4f32 = @splat(a[a1_base + 1]);

                        const bj = k * N * 2 + j * 2;
                        const b0 = simd.load4_f32(b, bj);
                        const b1 = simd.load4_f32(b, bj + 4);
                        const b_re = @shuffle(f32, b0, b1, [4]i32{ 0, 2, -1, -3 });
                        const b_im = @shuffle(f32, b0, b1, [4]i32{ 1, 3, -2, -4 });

                        acc0_re += a0_re * b_re - a0_im * b_im;
                        acc0_im += a0_re * b_im + a0_im * b_re;
                        acc1_re += a1_re * b_re - a1_im * b_im;
                        acc1_im += a1_re * b_im + a1_im * b_re;
                    }

                    // Write C once — interleave and store
                    const cj0 = c0_row + j * 2;
                    simd.store4_f32(c, cj0, @shuffle(f32, acc0_re, acc0_im, [4]i32{ 0, -1, 1, -2 }));
                    simd.store4_f32(c, cj0 + 4, @shuffle(f32, acc0_re, acc0_im, [4]i32{ 2, -3, 3, -4 }));
                    const cj1 = c1_row + j * 2;
                    simd.store4_f32(c, cj1, @shuffle(f32, acc1_re, acc1_im, [4]i32{ 0, -1, 1, -2 }));
                    simd.store4_f32(c, cj1 + 4, @shuffle(f32, acc1_re, acc1_im, [4]i32{ 2, -3, 3, -4 }));
                }
                // Scalar remainder for j
                while (j < j_end) : (j += 1) {
                    var s0_re: f32 = 0;
                    var s0_im: f32 = 0;
                    var s1_re: f32 = 0;
                    var s1_im: f32 = 0;
                    for (0..K) |k| {
                        const a0 = (i * K + k) * 2;
                        const a1 = ((i + 1) * K + k) * 2;
                        const bk = (k * N + j) * 2;
                        const br = b[bk];
                        const bi = b[bk + 1];
                        s0_re += a[a0] * br - a[a0 + 1] * bi;
                        s0_im += a[a0] * bi + a[a0 + 1] * br;
                        s1_re += a[a1] * br - a[a1 + 1] * bi;
                        s1_im += a[a1] * bi + a[a1 + 1] * br;
                    }
                    const cj0 = c0_row + j * 2;
                    c[cj0] = s0_re;
                    c[cj0 + 1] = s0_im;
                    const cj1 = c1_row + j * 2;
                    c[cj1] = s1_re;
                    c[cj1 + 1] = s1_im;
                }
            }
            // Remainder row (odd M)
            while (i < i_end) : (i += 1) {
                const c_row = i * N * 2;

                var j: usize = jj;
                while (j + 4 <= j_end) : (j += 4) {
                    var acc_re: simd.V4f32 = @splat(0);
                    var acc_im: simd.V4f32 = @splat(0);

                    for (0..K) |k| {
                        const a_base = (i * K + k) * 2;
                        const a_re: simd.V4f32 = @splat(a[a_base]);
                        const a_im: simd.V4f32 = @splat(a[a_base + 1]);
                        const bj = k * N * 2 + j * 2;
                        const b0 = simd.load4_f32(b, bj);
                        const b1 = simd.load4_f32(b, bj + 4);
                        const b_re = @shuffle(f32, b0, b1, [4]i32{ 0, 2, -1, -3 });
                        const b_im = @shuffle(f32, b0, b1, [4]i32{ 1, 3, -2, -4 });
                        acc_re += a_re * b_re - a_im * b_im;
                        acc_im += a_re * b_im + a_im * b_re;
                    }

                    const cj = c_row + j * 2;
                    simd.store4_f32(c, cj, @shuffle(f32, acc_re, acc_im, [4]i32{ 0, -1, 1, -2 }));
                    simd.store4_f32(c, cj + 4, @shuffle(f32, acc_re, acc_im, [4]i32{ 2, -3, 3, -4 }));
                }
                while (j < j_end) : (j += 1) {
                    var s_re: f32 = 0;
                    var s_im: f32 = 0;
                    for (0..K) |k| {
                        const a0 = (i * K + k) * 2;
                        const bk = (k * N + j) * 2;
                        s_re += a[a0] * b[bk] - a[a0 + 1] * b[bk + 1];
                        s_im += a[a0] * b[bk + 1] + a[a0 + 1] * b[bk];
                    }
                    const cj = c_row + j * 2;
                    c[cj] = s_re;
                    c[cj + 1] = s_im;
                }
            }
        }
    }
}

/// Computes C = A @ B for row-major i64 matrices with wrapping arithmetic.
/// A is (M x K), B is (K x N), C is (M x N).
/// Handles both signed (i64) and unsigned (u64) — wrapping add/mul produce identical bits.
/// Uses a 4×N register-blocked micro-kernel with 4-wide SIMD for the main loop, then 2-wide and scalar remainders.
export fn matmul_i64(a: [*]const i64, b: [*]const i64, c: [*]i64, M: u32, N: u32, K: u32) void {
    @memset(c[0 .. @as(usize, M) * N], 0);

    var ii: usize = 0;
    while (ii < M) : (ii += TILE_F64) {
        const i_end = @min(ii + TILE_F64, M);
        var jj: usize = 0;
        while (jj < N) : (jj += TILE_F64) {
            const j_end = @min(jj + TILE_F64, N);
            var kk: usize = 0;
            while (kk < K) : (kk += TILE_F64) {
                const k_end = @min(kk + TILE_F64, K);

                var i: usize = ii;
                while (i + 4 <= i_end) : (i += 4) {
                    var j: usize = jj;
                    while (j + 4 <= j_end) : (j += 4) {
                        var acc00: simd.V2i64 = @splat(0);
                        var acc01: simd.V2i64 = @splat(0);
                        var acc10: simd.V2i64 = @splat(0);
                        var acc11: simd.V2i64 = @splat(0);
                        var acc20: simd.V2i64 = @splat(0);
                        var acc21: simd.V2i64 = @splat(0);
                        var acc30: simd.V2i64 = @splat(0);
                        var acc31: simd.V2i64 = @splat(0);

                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            const b_row = k * N;
                            const b0 = simd.load2_i64(b, b_row + j);
                            const b1 = simd.load2_i64(b, b_row + j + 2);
                            const a0: simd.V2i64 = @splat(a[(i + 0) * K + k]);
                            acc00 +%= a0 *% b0;
                            acc01 +%= a0 *% b1;
                            const a1: simd.V2i64 = @splat(a[(i + 1) * K + k]);
                            acc10 +%= a1 *% b0;
                            acc11 +%= a1 *% b1;
                            const a2: simd.V2i64 = @splat(a[(i + 2) * K + k]);
                            acc20 +%= a2 *% b0;
                            acc21 +%= a2 *% b1;
                            const a3: simd.V2i64 = @splat(a[(i + 3) * K + k]);
                            acc30 +%= a3 *% b0;
                            acc31 +%= a3 *% b1;
                        }

                        const c0 = (i + 0) * N;
                        simd.store2_i64(c, c0 + j, simd.load2_i64(c, c0 + j) +% acc00);
                        simd.store2_i64(c, c0 + j + 2, simd.load2_i64(c, c0 + j + 2) +% acc01);
                        const c1 = (i + 1) * N;
                        simd.store2_i64(c, c1 + j, simd.load2_i64(c, c1 + j) +% acc10);
                        simd.store2_i64(c, c1 + j + 2, simd.load2_i64(c, c1 + j + 2) +% acc11);
                        const c2 = (i + 2) * N;
                        simd.store2_i64(c, c2 + j, simd.load2_i64(c, c2 + j) +% acc20);
                        simd.store2_i64(c, c2 + j + 2, simd.load2_i64(c, c2 + j + 2) +% acc21);
                        const c3 = (i + 3) * N;
                        simd.store2_i64(c, c3 + j, simd.load2_i64(c, c3 + j) +% acc30);
                        simd.store2_i64(c, c3 + j + 2, simd.load2_i64(c, c3 + j + 2) +% acc31);
                    }

                    while (j + 2 <= j_end) : (j += 2) {
                        var r0: simd.V2i64 = @splat(0);
                        var r1: simd.V2i64 = @splat(0);
                        var r2: simd.V2i64 = @splat(0);
                        var r3: simd.V2i64 = @splat(0);
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            const bv = simd.load2_i64(b, k * N + j);
                            r0 +%= @as(simd.V2i64, @splat(a[(i + 0) * K + k])) *% bv;
                            r1 +%= @as(simd.V2i64, @splat(a[(i + 1) * K + k])) *% bv;
                            r2 +%= @as(simd.V2i64, @splat(a[(i + 2) * K + k])) *% bv;
                            r3 +%= @as(simd.V2i64, @splat(a[(i + 3) * K + k])) *% bv;
                        }
                        simd.store2_i64(c, (i + 0) * N + j, simd.load2_i64(c, (i + 0) * N + j) +% r0);
                        simd.store2_i64(c, (i + 1) * N + j, simd.load2_i64(c, (i + 1) * N + j) +% r1);
                        simd.store2_i64(c, (i + 2) * N + j, simd.load2_i64(c, (i + 2) * N + j) +% r2);
                        simd.store2_i64(c, (i + 3) * N + j, simd.load2_i64(c, (i + 3) * N + j) +% r3);
                    }
                    while (j < j_end) : (j += 1) {
                        var s0: i64 = 0;
                        var s1: i64 = 0;
                        var s2: i64 = 0;
                        var s3: i64 = 0;
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            const bv = b[k * N + j];
                            s0 +%= a[(i + 0) * K + k] *% bv;
                            s1 +%= a[(i + 1) * K + k] *% bv;
                            s2 +%= a[(i + 2) * K + k] *% bv;
                            s3 +%= a[(i + 3) * K + k] *% bv;
                        }
                        c[(i + 0) * N + j] +%= s0;
                        c[(i + 1) * N + j] +%= s1;
                        c[(i + 2) * N + j] +%= s2;
                        c[(i + 3) * N + j] +%= s3;
                    }
                }

                // Remainder rows
                while (i < i_end) : (i += 1) {
                    const c_row = i * N;
                    var j: usize = jj;
                    while (j + 4 <= j_end) : (j += 4) {
                        var acc0: simd.V2i64 = @splat(0);
                        var acc1: simd.V2i64 = @splat(0);
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            const a_vec: simd.V2i64 = @splat(a[i * K + k]);
                            const b_row = k * N;
                            acc0 +%= a_vec *% simd.load2_i64(b, b_row + j);
                            acc1 +%= a_vec *% simd.load2_i64(b, b_row + j + 2);
                        }
                        simd.store2_i64(c, c_row + j, simd.load2_i64(c, c_row + j) +% acc0);
                        simd.store2_i64(c, c_row + j + 2, simd.load2_i64(c, c_row + j + 2) +% acc1);
                    }
                    while (j + 2 <= j_end) : (j += 2) {
                        var acc: simd.V2i64 = @splat(0);
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            acc +%= @as(simd.V2i64, @splat(a[i * K + k])) *% simd.load2_i64(b, k * N + j);
                        }
                        simd.store2_i64(c, c_row + j, simd.load2_i64(c, c_row + j) +% acc);
                    }
                    while (j < j_end) : (j += 1) {
                        var acc: i64 = 0;
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            acc +%= a[i * K + k] *% b[k * N + j];
                        }
                        c[c_row + j] +%= acc;
                    }
                }
            }
        }
    }
}

/// Computes C = A @ B for row-major i32 matrices with wrapping arithmetic.
/// A is (M x K), B is (K x N), C is (M x N).
/// Handles both signed (i32) and unsigned (u32) — wrapping add/mul produce identical bits.
/// Uses a 4×N register-blocked micro-kernel with 4-wide SIMD for the main loop, then 2-wide and scalar remainders.
export fn matmul_i32(a: [*]const i32, b: [*]const i32, c: [*]i32, M: u32, N: u32, K: u32) void {
    @memset(c[0 .. @as(usize, M) * N], 0);

    var ii: usize = 0;
    while (ii < M) : (ii += TILE_F32) {
        const i_end = @min(ii + TILE_F32, M);
        var jj: usize = 0;
        while (jj < N) : (jj += TILE_F32) {
            const j_end = @min(jj + TILE_F32, N);
            var kk: usize = 0;
            while (kk < K) : (kk += TILE_F32) {
                const k_end = @min(kk + TILE_F32, K);

                var i: usize = ii;
                while (i + 4 <= i_end) : (i += 4) {
                    var j: usize = jj;
                    while (j + 8 <= j_end) : (j += 8) {
                        var acc00: simd.V4i32 = @splat(0);
                        var acc01: simd.V4i32 = @splat(0);
                        var acc10: simd.V4i32 = @splat(0);
                        var acc11: simd.V4i32 = @splat(0);
                        var acc20: simd.V4i32 = @splat(0);
                        var acc21: simd.V4i32 = @splat(0);
                        var acc30: simd.V4i32 = @splat(0);
                        var acc31: simd.V4i32 = @splat(0);

                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            const b_row = k * N;
                            const b0 = simd.load4_i32(b, b_row + j);
                            const b1 = simd.load4_i32(b, b_row + j + 4);
                            const a0: simd.V4i32 = @splat(a[(i + 0) * K + k]);
                            acc00 +%= a0 *% b0;
                            acc01 +%= a0 *% b1;
                            const a1: simd.V4i32 = @splat(a[(i + 1) * K + k]);
                            acc10 +%= a1 *% b0;
                            acc11 +%= a1 *% b1;
                            const a2: simd.V4i32 = @splat(a[(i + 2) * K + k]);
                            acc20 +%= a2 *% b0;
                            acc21 +%= a2 *% b1;
                            const a3: simd.V4i32 = @splat(a[(i + 3) * K + k]);
                            acc30 +%= a3 *% b0;
                            acc31 +%= a3 *% b1;
                        }

                        const c0 = (i + 0) * N;
                        simd.store4_i32(c, c0 + j, simd.load4_i32(c, c0 + j) +% acc00);
                        simd.store4_i32(c, c0 + j + 4, simd.load4_i32(c, c0 + j + 4) +% acc01);
                        const c1 = (i + 1) * N;
                        simd.store4_i32(c, c1 + j, simd.load4_i32(c, c1 + j) +% acc10);
                        simd.store4_i32(c, c1 + j + 4, simd.load4_i32(c, c1 + j + 4) +% acc11);
                        const c2 = (i + 2) * N;
                        simd.store4_i32(c, c2 + j, simd.load4_i32(c, c2 + j) +% acc20);
                        simd.store4_i32(c, c2 + j + 4, simd.load4_i32(c, c2 + j + 4) +% acc21);
                        const c3 = (i + 3) * N;
                        simd.store4_i32(c, c3 + j, simd.load4_i32(c, c3 + j) +% acc30);
                        simd.store4_i32(c, c3 + j + 4, simd.load4_i32(c, c3 + j + 4) +% acc31);
                    }

                    while (j + 4 <= j_end) : (j += 4) {
                        var r0: simd.V4i32 = @splat(0);
                        var r1: simd.V4i32 = @splat(0);
                        var r2: simd.V4i32 = @splat(0);
                        var r3: simd.V4i32 = @splat(0);
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            const bv = simd.load4_i32(b, k * N + j);
                            r0 +%= @as(simd.V4i32, @splat(a[(i + 0) * K + k])) *% bv;
                            r1 +%= @as(simd.V4i32, @splat(a[(i + 1) * K + k])) *% bv;
                            r2 +%= @as(simd.V4i32, @splat(a[(i + 2) * K + k])) *% bv;
                            r3 +%= @as(simd.V4i32, @splat(a[(i + 3) * K + k])) *% bv;
                        }
                        simd.store4_i32(c, (i + 0) * N + j, simd.load4_i32(c, (i + 0) * N + j) +% r0);
                        simd.store4_i32(c, (i + 1) * N + j, simd.load4_i32(c, (i + 1) * N + j) +% r1);
                        simd.store4_i32(c, (i + 2) * N + j, simd.load4_i32(c, (i + 2) * N + j) +% r2);
                        simd.store4_i32(c, (i + 3) * N + j, simd.load4_i32(c, (i + 3) * N + j) +% r3);
                    }
                    while (j < j_end) : (j += 1) {
                        var s0: i32 = 0;
                        var s1: i32 = 0;
                        var s2: i32 = 0;
                        var s3: i32 = 0;
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            const bv = b[k * N + j];
                            s0 +%= a[(i + 0) * K + k] *% bv;
                            s1 +%= a[(i + 1) * K + k] *% bv;
                            s2 +%= a[(i + 2) * K + k] *% bv;
                            s3 +%= a[(i + 3) * K + k] *% bv;
                        }
                        c[(i + 0) * N + j] +%= s0;
                        c[(i + 1) * N + j] +%= s1;
                        c[(i + 2) * N + j] +%= s2;
                        c[(i + 3) * N + j] +%= s3;
                    }
                }

                // Remainder rows
                while (i < i_end) : (i += 1) {
                    const c_row = i * N;
                    var j: usize = jj;
                    while (j + 8 <= j_end) : (j += 8) {
                        var acc0: simd.V4i32 = @splat(0);
                        var acc1: simd.V4i32 = @splat(0);
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            const a_vec: simd.V4i32 = @splat(a[i * K + k]);
                            const b_row = k * N;
                            acc0 +%= a_vec *% simd.load4_i32(b, b_row + j);
                            acc1 +%= a_vec *% simd.load4_i32(b, b_row + j + 4);
                        }
                        simd.store4_i32(c, c_row + j, simd.load4_i32(c, c_row + j) +% acc0);
                        simd.store4_i32(c, c_row + j + 4, simd.load4_i32(c, c_row + j + 4) +% acc1);
                    }
                    while (j + 4 <= j_end) : (j += 4) {
                        var acc: simd.V4i32 = @splat(0);
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            acc +%= @as(simd.V4i32, @splat(a[i * K + k])) *% simd.load4_i32(b, k * N + j);
                        }
                        simd.store4_i32(c, c_row + j, simd.load4_i32(c, c_row + j) +% acc);
                    }
                    while (j < j_end) : (j += 1) {
                        var acc: i32 = 0;
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            acc +%= a[i * K + k] *% b[k * N + j];
                        }
                        c[c_row + j] +%= acc;
                    }
                }
            }
        }
    }
}

/// Computes C = A @ B for row-major i16 matrices with wrapping arithmetic.
/// A is (M x K), B is (K x N), C is (M x N).
/// Handles both signed (i16) and unsigned (u16) — wrapping add/mul produce identical bits.
/// Uses a 4×N register-blocked micro-kernel with 8-wide SIMD for the main loop, then 4-wide and scalar remainders.
export fn matmul_i16(a: [*]const i16, b: [*]const i16, c: [*]i16, M: u32, N: u32, K: u32) void {
    @memset(c[0 .. @as(usize, M) * N], 0);

    var ii: usize = 0;
    while (ii < M) : (ii += TILE_F32) {
        const i_end = @min(ii + TILE_F32, M);
        var jj: usize = 0;
        while (jj < N) : (jj += TILE_F32) {
            const j_end = @min(jj + TILE_F32, N);
            var kk: usize = 0;
            while (kk < K) : (kk += TILE_F32) {
                const k_end = @min(kk + TILE_F32, K);

                var i: usize = ii;
                while (i + 4 <= i_end) : (i += 4) {
                    var j: usize = jj;
                    while (j + 16 <= j_end) : (j += 16) {
                        var acc00: simd.V8i16 = @splat(0);
                        var acc01: simd.V8i16 = @splat(0);
                        var acc10: simd.V8i16 = @splat(0);
                        var acc11: simd.V8i16 = @splat(0);
                        var acc20: simd.V8i16 = @splat(0);
                        var acc21: simd.V8i16 = @splat(0);
                        var acc30: simd.V8i16 = @splat(0);
                        var acc31: simd.V8i16 = @splat(0);

                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            const b_row = k * N;
                            const b0 = simd.load8_i16(b, b_row + j);
                            const b1 = simd.load8_i16(b, b_row + j + 8);
                            const a0: simd.V8i16 = @splat(a[(i + 0) * K + k]);
                            acc00 +%= a0 *% b0;
                            acc01 +%= a0 *% b1;
                            const a1: simd.V8i16 = @splat(a[(i + 1) * K + k]);
                            acc10 +%= a1 *% b0;
                            acc11 +%= a1 *% b1;
                            const a2: simd.V8i16 = @splat(a[(i + 2) * K + k]);
                            acc20 +%= a2 *% b0;
                            acc21 +%= a2 *% b1;
                            const a3: simd.V8i16 = @splat(a[(i + 3) * K + k]);
                            acc30 +%= a3 *% b0;
                            acc31 +%= a3 *% b1;
                        }

                        const c0 = (i + 0) * N;
                        simd.store8_i16(c, c0 + j, simd.load8_i16(c, c0 + j) +% acc00);
                        simd.store8_i16(c, c0 + j + 8, simd.load8_i16(c, c0 + j + 8) +% acc01);
                        const c1 = (i + 1) * N;
                        simd.store8_i16(c, c1 + j, simd.load8_i16(c, c1 + j) +% acc10);
                        simd.store8_i16(c, c1 + j + 8, simd.load8_i16(c, c1 + j + 8) +% acc11);
                        const c2 = (i + 2) * N;
                        simd.store8_i16(c, c2 + j, simd.load8_i16(c, c2 + j) +% acc20);
                        simd.store8_i16(c, c2 + j + 8, simd.load8_i16(c, c2 + j + 8) +% acc21);
                        const c3 = (i + 3) * N;
                        simd.store8_i16(c, c3 + j, simd.load8_i16(c, c3 + j) +% acc30);
                        simd.store8_i16(c, c3 + j + 8, simd.load8_i16(c, c3 + j + 8) +% acc31);
                    }

                    while (j + 8 <= j_end) : (j += 8) {
                        var r0: simd.V8i16 = @splat(0);
                        var r1: simd.V8i16 = @splat(0);
                        var r2: simd.V8i16 = @splat(0);
                        var r3: simd.V8i16 = @splat(0);
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            const bv = simd.load8_i16(b, k * N + j);
                            r0 +%= @as(simd.V8i16, @splat(a[(i + 0) * K + k])) *% bv;
                            r1 +%= @as(simd.V8i16, @splat(a[(i + 1) * K + k])) *% bv;
                            r2 +%= @as(simd.V8i16, @splat(a[(i + 2) * K + k])) *% bv;
                            r3 +%= @as(simd.V8i16, @splat(a[(i + 3) * K + k])) *% bv;
                        }
                        simd.store8_i16(c, (i + 0) * N + j, simd.load8_i16(c, (i + 0) * N + j) +% r0);
                        simd.store8_i16(c, (i + 1) * N + j, simd.load8_i16(c, (i + 1) * N + j) +% r1);
                        simd.store8_i16(c, (i + 2) * N + j, simd.load8_i16(c, (i + 2) * N + j) +% r2);
                        simd.store8_i16(c, (i + 3) * N + j, simd.load8_i16(c, (i + 3) * N + j) +% r3);
                    }
                    while (j < j_end) : (j += 1) {
                        var s0: i16 = 0;
                        var s1: i16 = 0;
                        var s2: i16 = 0;
                        var s3: i16 = 0;
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            const bv = b[k * N + j];
                            s0 +%= a[(i + 0) * K + k] *% bv;
                            s1 +%= a[(i + 1) * K + k] *% bv;
                            s2 +%= a[(i + 2) * K + k] *% bv;
                            s3 +%= a[(i + 3) * K + k] *% bv;
                        }
                        c[(i + 0) * N + j] +%= s0;
                        c[(i + 1) * N + j] +%= s1;
                        c[(i + 2) * N + j] +%= s2;
                        c[(i + 3) * N + j] +%= s3;
                    }
                }

                // Remainder rows
                while (i < i_end) : (i += 1) {
                    const c_row = i * N;
                    var j: usize = jj;
                    while (j + 16 <= j_end) : (j += 16) {
                        var acc0: simd.V8i16 = @splat(0);
                        var acc1: simd.V8i16 = @splat(0);
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            const a_vec: simd.V8i16 = @splat(a[i * K + k]);
                            const b_row = k * N;
                            acc0 +%= a_vec *% simd.load8_i16(b, b_row + j);
                            acc1 +%= a_vec *% simd.load8_i16(b, b_row + j + 8);
                        }
                        simd.store8_i16(c, c_row + j, simd.load8_i16(c, c_row + j) +% acc0);
                        simd.store8_i16(c, c_row + j + 8, simd.load8_i16(c, c_row + j + 8) +% acc1);
                    }
                    while (j + 8 <= j_end) : (j += 8) {
                        var acc: simd.V8i16 = @splat(0);
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            acc +%= @as(simd.V8i16, @splat(a[i * K + k])) *% simd.load8_i16(b, k * N + j);
                        }
                        simd.store8_i16(c, c_row + j, simd.load8_i16(c, c_row + j) +% acc);
                    }
                    while (j < j_end) : (j += 1) {
                        var acc: i16 = 0;
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            acc +%= a[i * K + k] *% b[k * N + j];
                        }
                        c[c_row + j] +%= acc;
                    }
                }
            }
        }
    }
}

/// Computes C = A @ B for row-major i8 matrices with wrapping arithmetic.
/// A is (M x K), B is (K x N), C is (M x N).
/// Handles both signed (i8) and unsigned (u8) — wrapping add/mul produce identical bits.
/// Uses an i-j-j blocked approach with 16-wide SIMD for the main loop, then smaller remainders.
export fn matmul_i8(a: [*]const i8, b: [*]const i8, c: [*]i8, M: u32, N: u32, K: u32) void {
    @memset(c[0 .. @as(usize, M) * N], 0);

    var ii: usize = 0;
    while (ii < M) : (ii += TILE_F32) {
        const i_end = @min(ii + TILE_F32, M);
        var kk: usize = 0;
        while (kk < K) : (kk += TILE_F32) {
            const k_end = @min(kk + TILE_F32, K);
            var jj: usize = 0;
            while (jj < N) : (jj += TILE_F32) {
                const j_end = @min(jj + TILE_F32, N);

                var i: usize = ii;
                while (i < i_end) : (i += 1) {
                    var k: usize = kk;
                    while (k < k_end) : (k += 1) {
                        const a_ik = a[i * K + k];
                        const a_vec: simd.V16i8 = @splat(a_ik);
                        const b_row = k * N;
                        const c_row = i * N;

                        var j: usize = jj;
                        while (j + 32 <= j_end) : (j += 32) {
                            simd.store16_i8(c, c_row + j, simd.muladd_i8x16(simd.load16_i8(c, c_row + j), a_vec, simd.load16_i8(b, b_row + j)));
                            simd.store16_i8(c, c_row + j + 16, simd.muladd_i8x16(simd.load16_i8(c, c_row + j + 16), a_vec, simd.load16_i8(b, b_row + j + 16)));
                        }
                        while (j + 16 <= j_end) : (j += 16) {
                            simd.store16_i8(c, c_row + j, simd.muladd_i8x16(simd.load16_i8(c, c_row + j), a_vec, simd.load16_i8(b, b_row + j)));
                        }
                        while (j < j_end) : (j += 1) {
                            c[c_row + j] +%= a_ik *% b[b_row + j];
                        }
                    }
                }
            }
        }
    }
}

// --- Tests ---

test "matmul_f64 2x2" {
    const testing = @import("std").testing;
    var a = [_]f64{ 1, 2, 3, 4 };
    var b = [_]f64{ 5, 6, 7, 8 };
    var c = [_]f64{ 0, 0, 0, 0 };
    matmul_f64(&a, &b, &c, 2, 2, 2);
    try testing.expectApproxEqAbs(c[0], 19.0, 1e-10);
    try testing.expectApproxEqAbs(c[1], 22.0, 1e-10);
    try testing.expectApproxEqAbs(c[2], 43.0, 1e-10);
    try testing.expectApproxEqAbs(c[3], 50.0, 1e-10);
}

test "matmul_f64 non-square 2x4x3" {
    const testing = @import("std").testing;
    var a = [_]f64{ 1, 2, 3, 4, 5, 6 };
    var b = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var c = [_]f64{ 0, 0, 0, 0, 0, 0, 0, 0 };
    matmul_f64(&a, &b, &c, 2, 4, 3);
    try testing.expectApproxEqAbs(c[0], 38.0, 1e-10);
    try testing.expectApproxEqAbs(c[1], 44.0, 1e-10);
    try testing.expectApproxEqAbs(c[2], 50.0, 1e-10);
    try testing.expectApproxEqAbs(c[3], 56.0, 1e-10);
    try testing.expectApproxEqAbs(c[4], 83.0, 1e-10);
    try testing.expectApproxEqAbs(c[5], 98.0, 1e-10);
    try testing.expectApproxEqAbs(c[6], 113.0, 1e-10);
    try testing.expectApproxEqAbs(c[7], 128.0, 1e-10);
}

test "matmul_f64 odd N remainder" {
    const testing = @import("std").testing;
    var a = [_]f64{ 1, 2, 3, 4 };
    var b = [_]f64{ 1, 2, 3, 4, 5, 6 };
    var c = [_]f64{ 0, 0, 0, 0, 0, 0 };
    matmul_f64(&a, &b, &c, 2, 3, 2);
    try testing.expectApproxEqAbs(c[0], 9.0, 1e-10);
    try testing.expectApproxEqAbs(c[1], 12.0, 1e-10);
    try testing.expectApproxEqAbs(c[2], 15.0, 1e-10);
    try testing.expectApproxEqAbs(c[3], 19.0, 1e-10);
    try testing.expectApproxEqAbs(c[4], 26.0, 1e-10);
    try testing.expectApproxEqAbs(c[5], 33.0, 1e-10);
}

test "matmul_f32 2x2" {
    const testing = @import("std").testing;
    var a = [_]f32{ 1, 2, 3, 4 };
    var b = [_]f32{ 5, 6, 7, 8 };
    var c = [_]f32{ 0, 0, 0, 0 };
    matmul_f32(&a, &b, &c, 2, 2, 2);
    try testing.expectApproxEqAbs(c[0], 19.0, 1e-5);
    try testing.expectApproxEqAbs(c[1], 22.0, 1e-5);
    try testing.expectApproxEqAbs(c[2], 43.0, 1e-5);
    try testing.expectApproxEqAbs(c[3], 50.0, 1e-5);
}

test "matmul_f32 non-square 2x4x3" {
    const testing = @import("std").testing;
    var a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var b = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var c = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    matmul_f32(&a, &b, &c, 2, 4, 3);
    try testing.expectApproxEqAbs(c[0], 38.0, 1e-5);
    try testing.expectApproxEqAbs(c[1], 44.0, 1e-5);
    try testing.expectApproxEqAbs(c[2], 50.0, 1e-5);
    try testing.expectApproxEqAbs(c[3], 56.0, 1e-5);
    try testing.expectApproxEqAbs(c[4], 83.0, 1e-5);
    try testing.expectApproxEqAbs(c[5], 98.0, 1e-5);
    try testing.expectApproxEqAbs(c[6], 113.0, 1e-5);
    try testing.expectApproxEqAbs(c[7], 128.0, 1e-5);
}

test "matmul_f32 odd N remainder" {
    const testing = @import("std").testing;
    var a = [_]f32{ 1, 2, 3, 4 };
    var b = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var c = [_]f32{ 0, 0, 0, 0, 0, 0 };
    matmul_f32(&a, &b, &c, 2, 3, 2);
    try testing.expectApproxEqAbs(c[0], 9.0, 1e-5);
    try testing.expectApproxEqAbs(c[1], 12.0, 1e-5);
    try testing.expectApproxEqAbs(c[2], 15.0, 1e-5);
    try testing.expectApproxEqAbs(c[3], 19.0, 1e-5);
    try testing.expectApproxEqAbs(c[4], 26.0, 1e-5);
    try testing.expectApproxEqAbs(c[5], 33.0, 1e-5);
}

test "matmul_c128 2x2" {
    const testing = @import("std").testing;
    var a = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var b = [_]f64{ 1, 1, 0, 1, 1, 0, 1, 1 };
    var c = [_]f64{ 0, 0, 0, 0, 0, 0, 0, 0 };
    matmul_c128(&a, &b, &c, 2, 2, 2);
    try testing.expectApproxEqAbs(c[0], 2.0, 1e-10);
    try testing.expectApproxEqAbs(c[1], 7.0, 1e-10);
    try testing.expectApproxEqAbs(c[2], -3.0, 1e-10);
    try testing.expectApproxEqAbs(c[3], 8.0, 1e-10);
    try testing.expectApproxEqAbs(c[4], 6.0, 1e-10);
    try testing.expectApproxEqAbs(c[5], 19.0, 1e-10);
    try testing.expectApproxEqAbs(c[6], -7.0, 1e-10);
    try testing.expectApproxEqAbs(c[7], 20.0, 1e-10);
}

test "matmul_c128 scalar remainder" {
    const testing = @import("std").testing;
    var a = [_]f64{ 1, 2 };
    var b = [_]f64{ 3, 4 };
    var c = [_]f64{ 0, 0 };
    matmul_c128(&a, &b, &c, 1, 1, 1);
    try testing.expectApproxEqAbs(c[0], -5.0, 1e-10);
    try testing.expectApproxEqAbs(c[1], 10.0, 1e-10);
}

test "matmul_c128 non-square 2x3x2" {
    const testing = @import("std").testing;
    var a = [_]f64{ 1, 0, 0, 1, 1, 1, 1, -1 };
    var b = [_]f64{ 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1 };
    var c = [_]f64{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    matmul_c128(&a, &b, &c, 2, 3, 2);
    try testing.expectApproxEqAbs(c[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(c[1], 1.0, 1e-10);
    try testing.expectApproxEqAbs(c[2], 0.0, 1e-10);
    try testing.expectApproxEqAbs(c[3], 2.0, 1e-10);
    try testing.expectApproxEqAbs(c[4], 0.0, 1e-10);
    try testing.expectApproxEqAbs(c[5], 1.0, 1e-10);
    try testing.expectApproxEqAbs(c[6], 2.0, 1e-10);
    try testing.expectApproxEqAbs(c[7], 0.0, 1e-10);
    try testing.expectApproxEqAbs(c[8], 0.0, 1e-10);
    try testing.expectApproxEqAbs(c[9], 0.0, 1e-10);
    try testing.expectApproxEqAbs(c[10], 1.0, 1e-10);
    try testing.expectApproxEqAbs(c[11], 3.0, 1e-10);
}

test "matmul_c64 2x2" {
    const testing = @import("std").testing;
    var a = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var b = [_]f32{ 1, 1, 0, 1, 1, 0, 1, 1 };
    var c = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    matmul_c64(&a, &b, &c, 2, 2, 2);
    try testing.expectApproxEqAbs(c[0], 2.0, 1e-5);
    try testing.expectApproxEqAbs(c[1], 7.0, 1e-5);
    try testing.expectApproxEqAbs(c[2], -3.0, 1e-5);
    try testing.expectApproxEqAbs(c[3], 8.0, 1e-5);
    try testing.expectApproxEqAbs(c[4], 6.0, 1e-5);
    try testing.expectApproxEqAbs(c[5], 19.0, 1e-5);
    try testing.expectApproxEqAbs(c[6], -7.0, 1e-5);
    try testing.expectApproxEqAbs(c[7], 20.0, 1e-5);
}

test "matmul_c64 scalar remainder" {
    const testing = @import("std").testing;
    var a = [_]f32{ 1, 2 };
    var b = [_]f32{ 3, 4 };
    var c = [_]f32{ 0, 0 };
    matmul_c64(&a, &b, &c, 1, 1, 1);
    try testing.expectApproxEqAbs(c[0], -5.0, 1e-5);
    try testing.expectApproxEqAbs(c[1], 10.0, 1e-5);
}

test "matmul_c64 SIMD path N=4" {
    const testing = @import("std").testing;
    var a = [_]f32{ 1, 0 };
    var b = [_]f32{ 1, 0, 0, 1, 1, 1, 1, -1 };
    var c = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    matmul_c64(&a, &b, &c, 1, 4, 1);
    try testing.expectApproxEqAbs(c[0], 1.0, 1e-5);
    try testing.expectApproxEqAbs(c[1], 0.0, 1e-5);
    try testing.expectApproxEqAbs(c[2], 0.0, 1e-5);
    try testing.expectApproxEqAbs(c[3], 1.0, 1e-5);
    try testing.expectApproxEqAbs(c[4], 1.0, 1e-5);
    try testing.expectApproxEqAbs(c[5], 1.0, 1e-5);
    try testing.expectApproxEqAbs(c[6], 1.0, 1e-5);
    try testing.expectApproxEqAbs(c[7], -1.0, 1e-5);
}

test "matmul_i64 2x2" {
    const testing = @import("std").testing;
    var a = [_]i64{ 1, 2, 3, 4 };
    var b = [_]i64{ 5, 6, 7, 8 };
    var c = [_]i64{ 0, 0, 0, 0 };
    matmul_i64(&a, &b, &c, 2, 2, 2);
    try testing.expectEqual(c[0], 19);
    try testing.expectEqual(c[1], 22);
    try testing.expectEqual(c[2], 43);
    try testing.expectEqual(c[3], 50);
}

test "matmul_i64 non-square 2x4x3" {
    const testing = @import("std").testing;
    var a = [_]i64{ 1, 2, 3, 4, 5, 6 };
    var b = [_]i64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var c = [_]i64{ 0, 0, 0, 0, 0, 0, 0, 0 };
    matmul_i64(&a, &b, &c, 2, 4, 3);
    try testing.expectEqual(c[0], 38);
    try testing.expectEqual(c[1], 44);
    try testing.expectEqual(c[2], 50);
    try testing.expectEqual(c[3], 56);
    try testing.expectEqual(c[4], 83);
    try testing.expectEqual(c[5], 98);
    try testing.expectEqual(c[6], 113);
    try testing.expectEqual(c[7], 128);
}

test "matmul_i64 odd N remainder" {
    const testing = @import("std").testing;
    var a = [_]i64{ 1, 2, 3, 4 };
    var b = [_]i64{ 1, 2, 3, 4, 5, 6 };
    var c = [_]i64{ 0, 0, 0, 0, 0, 0 };
    matmul_i64(&a, &b, &c, 2, 3, 2);
    try testing.expectEqual(c[0], 9);
    try testing.expectEqual(c[1], 12);
    try testing.expectEqual(c[2], 15);
    try testing.expectEqual(c[3], 19);
    try testing.expectEqual(c[4], 26);
    try testing.expectEqual(c[5], 33);
}

test "matmul_i32 2x2" {
    const testing = @import("std").testing;
    var a = [_]i32{ 1, 2, 3, 4 };
    var b = [_]i32{ 5, 6, 7, 8 };
    var c = [_]i32{ 0, 0, 0, 0 };
    matmul_i32(&a, &b, &c, 2, 2, 2);
    try testing.expectEqual(c[0], 19);
    try testing.expectEqual(c[1], 22);
    try testing.expectEqual(c[2], 43);
    try testing.expectEqual(c[3], 50);
}

test "matmul_i32 non-square 2x4x3" {
    const testing = @import("std").testing;
    var a = [_]i32{ 1, 2, 3, 4, 5, 6 };
    var b = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var c = [_]i32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    matmul_i32(&a, &b, &c, 2, 4, 3);
    try testing.expectEqual(c[0], 38);
    try testing.expectEqual(c[1], 44);
    try testing.expectEqual(c[2], 50);
    try testing.expectEqual(c[3], 56);
    try testing.expectEqual(c[4], 83);
    try testing.expectEqual(c[5], 98);
    try testing.expectEqual(c[6], 113);
    try testing.expectEqual(c[7], 128);
}

test "matmul_i32 odd N remainder" {
    const testing = @import("std").testing;
    var a = [_]i32{ 1, 2, 3, 4 };
    var b = [_]i32{ 1, 2, 3, 4, 5, 6 };
    var c = [_]i32{ 0, 0, 0, 0, 0, 0 };
    matmul_i32(&a, &b, &c, 2, 3, 2);
    try testing.expectEqual(c[0], 9);
    try testing.expectEqual(c[1], 12);
    try testing.expectEqual(c[2], 15);
    try testing.expectEqual(c[3], 19);
    try testing.expectEqual(c[4], 26);
    try testing.expectEqual(c[5], 33);
}

test "matmul_i16 2x2" {
    const testing = @import("std").testing;
    var a = [_]i16{ 1, 2, 3, 4 };
    var b = [_]i16{ 5, 6, 7, 8 };
    var c = [_]i16{ 0, 0, 0, 0 };
    matmul_i16(&a, &b, &c, 2, 2, 2);
    try testing.expectEqual(c[0], 19);
    try testing.expectEqual(c[1], 22);
    try testing.expectEqual(c[2], 43);
    try testing.expectEqual(c[3], 50);
}

test "matmul_i16 overflow wrapping" {
    const testing = @import("std").testing;
    var a = [_]i16{ 200, 200, 200, 200 };
    var c = [_]i16{ 0, 0, 0, 0 };
    matmul_i16(&a, &a, &c, 2, 2, 2);
    try testing.expectEqual(c[0], 14464);
    try testing.expectEqual(c[1], 14464);
    try testing.expectEqual(c[2], 14464);
    try testing.expectEqual(c[3], 14464);
}

test "matmul_i8 2x2" {
    const testing = @import("std").testing;
    var a = [_]i8{ 1, 2, 3, 4 };
    var b = [_]i8{ 5, 6, 7, 8 };
    var c = [_]i8{ 0, 0, 0, 0 };
    matmul_i8(&a, &b, &c, 2, 2, 2);
    try testing.expectEqual(c[0], 19);
    try testing.expectEqual(c[1], 22);
    try testing.expectEqual(c[2], 43);
    try testing.expectEqual(c[3], 50);
}

test "matmul_i8 overflow wrapping" {
    const testing = @import("std").testing;
    var a = [_]i8{ 10, 20, 30, 40 };
    var b = [_]i8{ 5, 6, 7, 8 };
    var c = [_]i8{ 0, 0, 0, 0 };
    matmul_i8(&a, &b, &c, 2, 2, 2);
    try testing.expectEqual(c[0], -66);
    try testing.expectEqual(c[1], -36);
    try testing.expectEqual(c[2], -82);
    try testing.expectEqual(c[3], -12);
}
