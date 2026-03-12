//! WASM matmul benchmark kernels for f64.
//! Includes: ikj (streaming), ijkacc (register-accumulated), and microkernel (4×4 register-blocked).
//! All three are exported so the TS harness can benchmark them head-to-head.

const simd = @import("simd.zig");

const TILE = 64; // Tile size (matches production)

// ─── 1. ikj: streaming i-k-j (production large-N path) ─────────────────────

fn matmul_f64_ikj_impl(a: [*]const f64, b: [*]const f64, c: [*]f64, M: usize, N: usize, K: usize) void {
    @memset(c[0..M * N], 0);

    var ii: usize = 0;
    while (ii < M) : (ii += TILE) {
        const i_end = @min(ii + TILE, M);
        var kk: usize = 0;
        while (kk < K) : (kk += TILE) {
            const k_end = @min(kk + TILE, K);
            var jj: usize = 0;
            while (jj < N) : (jj += TILE) {
                const j_end = @min(jj + TILE, N);

                var i: usize = ii;
                while (i < i_end) : (i += 1) {
                    var k: usize = kk;
                    while (k < k_end) : (k += 1) {
                        const a_ik = a[i * K + k];
                        const a_vec: simd.V2f64 = @splat(a_ik);
                        const b_row = k * N;
                        const c_row = i * N;

                        var j: usize = jj;
                        while (j + 4 <= j_end) : (j += 4) {
                            simd.store2_f64(c, c_row + j, simd.load2_f64(c, c_row + j) + a_vec * simd.load2_f64(b, b_row + j));
                            simd.store2_f64(c, c_row + j + 2, simd.load2_f64(c, c_row + j + 2) + a_vec * simd.load2_f64(b, b_row + j + 2));
                        }
                        while (j + 2 <= j_end) : (j += 2) {
                            simd.store2_f64(c, c_row + j, simd.load2_f64(c, c_row + j) + a_vec * simd.load2_f64(b, b_row + j));
                        }
                        while (j < j_end) : (j += 1) {
                            c[c_row + j] += a_ik * b[b_row + j];
                        }
                    }
                }
            }
        }
    }
}

// ─── 2. ijkacc: i-j-k with register accumulation (production small-N path) ──

fn matmul_f64_ijkacc_impl(a: [*]const f64, b: [*]const f64, c: [*]f64, M: usize, N: usize, K: usize) void {
    @memset(c[0..M * N], 0);

    var ii: usize = 0;
    while (ii < M) : (ii += TILE) {
        const i_end = @min(ii + TILE, M);
        var jj: usize = 0;
        while (jj < N) : (jj += TILE) {
            const j_end = @min(jj + TILE, N);
            var kk: usize = 0;
            while (kk < K) : (kk += TILE) {
                const k_end = @min(kk + TILE, K);

                var i: usize = ii;
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
                            const a_vec: simd.V2f64 = @splat(a[i * K + k]);
                            acc += a_vec * simd.load2_f64(b, k * N + j);
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

// ─── 3. microkernel: 4×4 register-blocked i-j-k ────────────────────────────
// Processes 4 rows of i × 4 columns of j simultaneously.
// 8 V2f64 accumulators (4 rows × 2 V2f64 per row = 4×4 f64 block).
// Each B load is reused across all 4 A rows → 4× fewer B loads vs ijkacc.

fn matmul_f64_micro_impl(a: [*]const f64, b: [*]const f64, c: [*]f64, M: usize, N: usize, K: usize) void {
    @memset(c[0..M * N], 0);

    var ii: usize = 0;
    while (ii < M) : (ii += TILE) {
        const i_end = @min(ii + TILE, M);
        var jj: usize = 0;
        while (jj < N) : (jj += TILE) {
            const j_end = @min(jj + TILE, N);
            var kk: usize = 0;
            while (kk < K) : (kk += TILE) {
                const k_end = @min(kk + TILE, K);

                // Process 4 rows at a time
                var i: usize = ii;
                while (i + 4 <= i_end) : (i += 4) {
                    // Process 4 columns at a time (2 V2f64 = 4 f64)
                    var j: usize = jj;
                    while (j + 4 <= j_end) : (j += 4) {
                        // 8 accumulators: 4 rows × 2 V2f64
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
                            // Load B[k, j:j+4] once — reused for all 4 rows
                            const b_row = k * N;
                            const b0 = simd.load2_f64(b, b_row + j);
                            const b1 = simd.load2_f64(b, b_row + j + 2);

                            // Row 0
                            const a0: simd.V2f64 = @splat(a[(i + 0) * K + k]);
                            acc00 += a0 * b0;
                            acc01 += a0 * b1;

                            // Row 1
                            const a1: simd.V2f64 = @splat(a[(i + 1) * K + k]);
                            acc10 += a1 * b0;
                            acc11 += a1 * b1;

                            // Row 2
                            const a2: simd.V2f64 = @splat(a[(i + 2) * K + k]);
                            acc20 += a2 * b0;
                            acc21 += a2 * b1;

                            // Row 3
                            const a3: simd.V2f64 = @splat(a[(i + 3) * K + k]);
                            acc30 += a3 * b0;
                            acc31 += a3 * b1;
                        }

                        // Write back 4×4 block
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

                    // Remainder columns (j not multiple of 4): fall back to 1-row accumulation
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

                // Remainder rows (i not multiple of 4): fall back to ijkacc-style 1-row
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

// ─── C-ABI exports ──────────────────────────────────────────────────────────

export fn matmul_f64(a_ptr: [*]const f64, b_ptr: [*]const f64, c_ptr: [*]f64, M: u32, N: u32, K: u32) void {
    matmul_f64_ikj_impl(a_ptr, b_ptr, c_ptr, M, N, K);
}

export fn matmul_f64_ijkacc(a_ptr: [*]const f64, b_ptr: [*]const f64, c_ptr: [*]f64, M: u32, N: u32, K: u32) void {
    matmul_f64_ijkacc_impl(a_ptr, b_ptr, c_ptr, M, N, K);
}

export fn matmul_f64_micro(a_ptr: [*]const f64, b_ptr: [*]const f64, c_ptr: [*]f64, M: u32, N: u32, K: u32) void {
    matmul_f64_micro_impl(a_ptr, b_ptr, c_ptr, M, N, K);
}
