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

                // 4×8 micro-kernel (16 accumulators, no FMA)
                var i: usize = ii;
                while (i + 4 <= i_end) : (i += 4) {
                    var j: usize = jj;
                    while (j + 8 <= j_end) : (j += 8) {
                        var c00: simd.V2f64 = @splat(0);
                        var c01: simd.V2f64 = @splat(0);
                        var c02: simd.V2f64 = @splat(0);
                        var c03: simd.V2f64 = @splat(0);
                        var c10: simd.V2f64 = @splat(0);
                        var c11: simd.V2f64 = @splat(0);
                        var c12: simd.V2f64 = @splat(0);
                        var c13: simd.V2f64 = @splat(0);
                        var c20: simd.V2f64 = @splat(0);
                        var c21: simd.V2f64 = @splat(0);
                        var c22: simd.V2f64 = @splat(0);
                        var c23: simd.V2f64 = @splat(0);
                        var c30: simd.V2f64 = @splat(0);
                        var c31: simd.V2f64 = @splat(0);
                        var c32: simd.V2f64 = @splat(0);
                        var c33: simd.V2f64 = @splat(0);

                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            const br = k * N;
                            const b0 = simd.load2_f64(b, br + j);
                            const b1 = simd.load2_f64(b, br + j + 2);
                            const b2 = simd.load2_f64(b, br + j + 4);
                            const b3 = simd.load2_f64(b, br + j + 6);
                            const a0: simd.V2f64 = @splat(a[(i + 0) * K + k]);
                            c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
                            const a1: simd.V2f64 = @splat(a[(i + 1) * K + k]);
                            c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
                            const a2: simd.V2f64 = @splat(a[(i + 2) * K + k]);
                            c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
                            const a3: simd.V2f64 = @splat(a[(i + 3) * K + k]);
                            c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
                        }

                        inline for (0..4) |r| {
                            const off = (i + r) * N + j;
                            const ra = switch (r) {
                                0 => .{ c00, c01, c02, c03 },
                                1 => .{ c10, c11, c12, c13 },
                                2 => .{ c20, c21, c22, c23 },
                                3 => .{ c30, c31, c32, c33 },
                                else => unreachable,
                            };
                            simd.store2_f64(c, off, simd.load2_f64(c, off) + ra[0]);
                            simd.store2_f64(c, off + 2, simd.load2_f64(c, off + 2) + ra[1]);
                            simd.store2_f64(c, off + 4, simd.load2_f64(c, off + 4) + ra[2]);
                            simd.store2_f64(c, off + 6, simd.load2_f64(c, off + 6) + ra[3]);
                        }
                    }

                    // Remainder: 4 cols
                    while (j + 4 <= j_end) : (j += 4) {
                        var r00: simd.V2f64 = @splat(0);
                        var r01: simd.V2f64 = @splat(0);
                        var r10: simd.V2f64 = @splat(0);
                        var r11: simd.V2f64 = @splat(0);
                        var r20: simd.V2f64 = @splat(0);
                        var r21: simd.V2f64 = @splat(0);
                        var r30: simd.V2f64 = @splat(0);
                        var r31: simd.V2f64 = @splat(0);
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            const b0 = simd.load2_f64(b, k * N + j);
                            const b1 = simd.load2_f64(b, k * N + j + 2);
                            const a0: simd.V2f64 = @splat(a[(i + 0) * K + k]);
                            r00 += a0 * b0; r01 += a0 * b1;
                            const a1: simd.V2f64 = @splat(a[(i + 1) * K + k]);
                            r10 += a1 * b0; r11 += a1 * b1;
                            const a2: simd.V2f64 = @splat(a[(i + 2) * K + k]);
                            r20 += a2 * b0; r21 += a2 * b1;
                            const a3: simd.V2f64 = @splat(a[(i + 3) * K + k]);
                            r30 += a3 * b0; r31 += a3 * b1;
                        }
                        inline for (0..4) |r| {
                            const off = (i + r) * N + j;
                            const rv = switch (r) {
                                0 => .{ r00, r01 }, 1 => .{ r10, r11 },
                                2 => .{ r20, r21 }, 3 => .{ r30, r31 },
                                else => unreachable,
                            };
                            simd.store2_f64(c, off, simd.load2_f64(c, off) + rv[0]);
                            simd.store2_f64(c, off + 2, simd.load2_f64(c, off + 2) + rv[1]);
                        }
                    }
                    // Remainder: 2 cols
                    while (j + 2 <= j_end) : (j += 2) {
                        var s0: simd.V2f64 = @splat(0);
                        var s1: simd.V2f64 = @splat(0);
                        var s2: simd.V2f64 = @splat(0);
                        var s3: simd.V2f64 = @splat(0);
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            const bv = simd.load2_f64(b, k * N + j);
                            s0 += @as(simd.V2f64, @splat(a[(i + 0) * K + k])) * bv;
                            s1 += @as(simd.V2f64, @splat(a[(i + 1) * K + k])) * bv;
                            s2 += @as(simd.V2f64, @splat(a[(i + 2) * K + k])) * bv;
                            s3 += @as(simd.V2f64, @splat(a[(i + 3) * K + k])) * bv;
                        }
                        inline for (0..4) |r| {
                            const off = (i + r) * N + j;
                            const sv = switch (r) { 0 => s0, 1 => s1, 2 => s2, 3 => s3, else => unreachable };
                            simd.store2_f64(c, off, simd.load2_f64(c, off) + sv);
                        }
                    }
                    // Remainder: scalar
                    while (j < j_end) : (j += 1) {
                        var sc: [4]f64 = .{0} ** 4;
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            const bv = b[k * N + j];
                            inline for (0..4) |r| sc[r] += a[(i + r) * K + k] * bv;
                        }
                        inline for (0..4) |r| c[(i + r) * N + j] += sc[r];
                    }
                }

                // Remainder rows (1 at a time)
                while (i < i_end) : (i += 1) {
                    const ci = i * N;
                    var j: usize = jj;
                    while (j + 4 <= j_end) : (j += 4) {
                        var a0: simd.V2f64 = @splat(0);
                        var a1: simd.V2f64 = @splat(0);
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            const av: simd.V2f64 = @splat(a[i * K + k]);
                            a0 += av * simd.load2_f64(b, k * N + j);
                            a1 += av * simd.load2_f64(b, k * N + j + 2);
                        }
                        simd.store2_f64(c, ci + j, simd.load2_f64(c, ci + j) + a0);
                        simd.store2_f64(c, ci + j + 2, simd.load2_f64(c, ci + j + 2) + a1);
                    }
                    while (j + 2 <= j_end) : (j += 2) {
                        var acc: simd.V2f64 = @splat(0);
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            acc += @as(simd.V2f64, @splat(a[i * K + k])) * simd.load2_f64(b, k * N + j);
                        }
                        simd.store2_f64(c, ci + j, simd.load2_f64(c, ci + j) + acc);
                    }
                    while (j < j_end) : (j += 1) {
                        var acc: f64 = 0;
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) acc += a[i * K + k] * b[k * N + j];
                        c[ci + j] += acc;
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

                // 4×16 micro-kernel (16 accumulators, V4f32)
                var i: usize = ii;
                while (i + 4 <= i_end) : (i += 4) {
                    var j: usize = jj;
                    while (j + 16 <= j_end) : (j += 16) {
                        var c00: simd.V4f32 = @splat(0); var c01: simd.V4f32 = @splat(0);
                        var c02: simd.V4f32 = @splat(0); var c03: simd.V4f32 = @splat(0);
                        var c10: simd.V4f32 = @splat(0); var c11: simd.V4f32 = @splat(0);
                        var c12: simd.V4f32 = @splat(0); var c13: simd.V4f32 = @splat(0);
                        var c20: simd.V4f32 = @splat(0); var c21: simd.V4f32 = @splat(0);
                        var c22: simd.V4f32 = @splat(0); var c23: simd.V4f32 = @splat(0);
                        var c30: simd.V4f32 = @splat(0); var c31: simd.V4f32 = @splat(0);
                        var c32: simd.V4f32 = @splat(0); var c33: simd.V4f32 = @splat(0);

                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            const br = k * N;
                            const b0 = simd.load4_f32(b, br + j);
                            const b1 = simd.load4_f32(b, br + j + 4);
                            const b2 = simd.load4_f32(b, br + j + 8);
                            const b3 = simd.load4_f32(b, br + j + 12);
                            const a0: simd.V4f32 = @splat(a[(i + 0) * K + k]);
                            c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3;
                            const a1: simd.V4f32 = @splat(a[(i + 1) * K + k]);
                            c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3;
                            const a2: simd.V4f32 = @splat(a[(i + 2) * K + k]);
                            c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3;
                            const a3: simd.V4f32 = @splat(a[(i + 3) * K + k]);
                            c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3;
                        }

                        inline for (0..4) |r| {
                            const off = (i + r) * N + j;
                            const ra = switch (r) {
                                0 => .{ c00, c01, c02, c03 },
                                1 => .{ c10, c11, c12, c13 },
                                2 => .{ c20, c21, c22, c23 },
                                3 => .{ c30, c31, c32, c33 },
                                else => unreachable,
                            };
                            simd.store4_f32(c, off, simd.load4_f32(c, off) + ra[0]);
                            simd.store4_f32(c, off + 4, simd.load4_f32(c, off + 4) + ra[1]);
                            simd.store4_f32(c, off + 8, simd.load4_f32(c, off + 8) + ra[2]);
                            simd.store4_f32(c, off + 12, simd.load4_f32(c, off + 12) + ra[3]);
                        }
                    }

                    // Remainder: 8 cols
                    while (j + 8 <= j_end) : (j += 8) {
                        var r00: simd.V4f32 = @splat(0); var r01: simd.V4f32 = @splat(0);
                        var r10: simd.V4f32 = @splat(0); var r11: simd.V4f32 = @splat(0);
                        var r20: simd.V4f32 = @splat(0); var r21: simd.V4f32 = @splat(0);
                        var r30: simd.V4f32 = @splat(0); var r31: simd.V4f32 = @splat(0);
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            const b0 = simd.load4_f32(b, k * N + j);
                            const b1 = simd.load4_f32(b, k * N + j + 4);
                            const a0: simd.V4f32 = @splat(a[(i + 0) * K + k]);
                            r00 += a0 * b0; r01 += a0 * b1;
                            const a1: simd.V4f32 = @splat(a[(i + 1) * K + k]);
                            r10 += a1 * b0; r11 += a1 * b1;
                            const a2: simd.V4f32 = @splat(a[(i + 2) * K + k]);
                            r20 += a2 * b0; r21 += a2 * b1;
                            const a3: simd.V4f32 = @splat(a[(i + 3) * K + k]);
                            r30 += a3 * b0; r31 += a3 * b1;
                        }
                        inline for (0..4) |r| {
                            const off = (i + r) * N + j;
                            const rv = switch (r) {
                                0 => .{ r00, r01 }, 1 => .{ r10, r11 },
                                2 => .{ r20, r21 }, 3 => .{ r30, r31 },
                                else => unreachable,
                            };
                            simd.store4_f32(c, off, simd.load4_f32(c, off) + rv[0]);
                            simd.store4_f32(c, off + 4, simd.load4_f32(c, off + 4) + rv[1]);
                        }
                    }
                    // Remainder: 4 cols
                    while (j + 4 <= j_end) : (j += 4) {
                        var s0: simd.V4f32 = @splat(0);
                        var s1: simd.V4f32 = @splat(0);
                        var s2: simd.V4f32 = @splat(0);
                        var s3: simd.V4f32 = @splat(0);
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            const bv = simd.load4_f32(b, k * N + j);
                            s0 += @as(simd.V4f32, @splat(a[(i + 0) * K + k])) * bv;
                            s1 += @as(simd.V4f32, @splat(a[(i + 1) * K + k])) * bv;
                            s2 += @as(simd.V4f32, @splat(a[(i + 2) * K + k])) * bv;
                            s3 += @as(simd.V4f32, @splat(a[(i + 3) * K + k])) * bv;
                        }
                        inline for (0..4) |r| {
                            const off = (i + r) * N + j;
                            const sv = switch (r) { 0 => s0, 1 => s1, 2 => s2, 3 => s3, else => unreachable };
                            simd.store4_f32(c, off, simd.load4_f32(c, off) + sv);
                        }
                    }
                    // Remainder: scalar
                    while (j < j_end) : (j += 1) {
                        var sc: [4]f32 = .{0} ** 4;
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            const bv = b[k * N + j];
                            inline for (0..4) |r| sc[r] += a[(i + r) * K + k] * bv;
                        }
                        inline for (0..4) |r| c[(i + r) * N + j] += sc[r];
                    }
                }

                // Remainder rows (1 at a time)
                while (i < i_end) : (i += 1) {
                    const ci = i * N;
                    var j: usize = jj;
                    while (j + 8 <= j_end) : (j += 8) {
                        var a0: simd.V4f32 = @splat(0);
                        var a1: simd.V4f32 = @splat(0);
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            const av: simd.V4f32 = @splat(a[i * K + k]);
                            a0 += av * simd.load4_f32(b, k * N + j);
                            a1 += av * simd.load4_f32(b, k * N + j + 4);
                        }
                        simd.store4_f32(c, ci + j, simd.load4_f32(c, ci + j) + a0);
                        simd.store4_f32(c, ci + j + 4, simd.load4_f32(c, ci + j + 4) + a1);
                    }
                    while (j + 4 <= j_end) : (j += 4) {
                        var acc: simd.V4f32 = @splat(0);
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) {
                            acc += @as(simd.V4f32, @splat(a[i * K + k])) * simd.load4_f32(b, k * N + j);
                        }
                        simd.store4_f32(c, ci + j, simd.load4_f32(c, ci + j) + acc);
                    }
                    while (j < j_end) : (j += 1) {
                        var acc: f32 = 0;
                        var k: usize = kk;
                        while (k < k_end) : (k += 1) acc += a[i * K + k] * b[k * N + j];
                        c[ci + j] += acc;
                    }
                }
            }
        }
    }
}

/// Computes C = A @ B for row-major complex64 matrices (interleaved f32: [re, im, re, im, ...]).
/// C_re = P1 - P2, C_im = P3 - P1 - P2  where:
///   P1 = A_re @ B_re,  P2 = A_im @ B_im,  P3 = (A_re+A_im) @ (B_re+B_im)
///
/// Scratch-optimized layout: after P1, a_re/b_re are dead and overwritten
/// with a_sum/b_sum for P3. Total scratch: 2*M*K + 2*K*N + 3*M*N.
export fn matmul_c64(a: [*]const f32, b: [*]const f32, c: [*]f32, M: u32, N: u32, K: u32, scratch: [*]f32) void {
    const Mu = @as(usize, M);
    const Nu = @as(usize, N);
    const Ku = @as(usize, K);
    const out_count = Mu * Nu;
    const a_count = Mu * Ku;
    const b_count = Ku * Nu;

    // Scratch: a_re/a_sum (aliased), a_im, b_re/b_sum (aliased), b_im, p1, p2, p3
    const a_re = scratch; // later overwritten with a_sum
    const a_im = a_re + a_count;
    const b_re = a_im + a_count; // later overwritten with b_sum
    const b_im = b_re + b_count;
    const p1 = b_im + b_count;
    const p2 = p1 + out_count;
    const p3 = p2 + out_count;

    // Deinterleave A → a_re, a_im
    var i: usize = 0;
    while (i + 4 <= a_count) : (i += 4) {
        const v0 = simd.load4_f32(a, i * 2);
        const v1 = simd.load4_f32(a, i * 2 + 4);
        simd.store4_f32(a_re, i, @shuffle(f32, v0, v1, [4]i32{ 0, 2, -1, -3 }));
        simd.store4_f32(a_im, i, @shuffle(f32, v0, v1, [4]i32{ 1, 3, -2, -4 }));
    }
    while (i < a_count) : (i += 1) {
        a_re[i] = a[i * 2];
        a_im[i] = a[i * 2 + 1];
    }

    // Deinterleave B → b_re, b_im
    i = 0;
    while (i + 4 <= b_count) : (i += 4) {
        const v0 = simd.load4_f32(b, i * 2);
        const v1 = simd.load4_f32(b, i * 2 + 4);
        simd.store4_f32(b_re, i, @shuffle(f32, v0, v1, [4]i32{ 0, 2, -1, -3 }));
        simd.store4_f32(b_im, i, @shuffle(f32, v0, v1, [4]i32{ 1, 3, -2, -4 }));
    }
    while (i < b_count) : (i += 1) {
        b_re[i] = b[i * 2];
        b_im[i] = b[i * 2 + 1];
    }

    // P1 = A_re @ B_re (uses a_re, b_re)
    matmul_f32(a_re, b_re, p1, M, N, K);

    // Overwrite a_re → a_sum = a_re + a_im, b_re → b_sum = b_re + b_im
    // (a_re and b_re are dead after P1)
    i = 0;
    while (i + 4 <= a_count) : (i += 4) {
        simd.store4_f32(a_re, i, simd.load4_f32(a_re, i) + simd.load4_f32(a_im, i));
    }
    while (i < a_count) : (i += 1) {
        a_re[i] += a_im[i];
    }
    i = 0;
    while (i + 4 <= b_count) : (i += 4) {
        simd.store4_f32(b_re, i, simd.load4_f32(b_re, i) + simd.load4_f32(b_im, i));
    }
    while (i < b_count) : (i += 1) {
        b_re[i] += b_im[i];
    }

    // P2 = A_im @ B_im (uses a_im, b_im — still intact)
    matmul_f32(a_im, b_im, p2, M, N, K);

    // P3 = (A_re+A_im) @ (B_re+B_im) = a_sum @ b_sum
    // a_re now holds a_sum, b_re now holds b_sum
    matmul_f32(a_re, b_re, p3, M, N, K);

    // Combine + reinterleave
    // C_re = P1 - P2,  C_im = P3 - P1 - P2
    i = 0;
    while (i + 4 <= out_count) : (i += 4) {
        const v_p1 = simd.load4_f32(p1, i);
        const v_p2 = simd.load4_f32(p2, i);
        const c_re = v_p1 - v_p2;
        const c_im = simd.load4_f32(p3, i) - v_p1 - v_p2;
        simd.store4_f32(c, i * 2, @shuffle(f32, c_re, c_im, [4]i32{ 0, -1, 1, -2 }));
        simd.store4_f32(c, i * 2 + 4, @shuffle(f32, c_re, c_im, [4]i32{ 2, -3, 3, -4 }));
    }
    while (i < out_count) : (i += 1) {
        const v_p1 = p1[i];
        const v_p2 = p2[i];
        c[i * 2] = v_p1 - v_p2;
        c[i * 2 + 1] = p3[i] - v_p1 - v_p2;
    }
}

/// Computes C = A @ B for row-major complex128 matrices (interleaved f64: [re, im, re, im, ...]).
/// Same algorithm as matmul_c64 but for f64.
/// Scratch: 2*M*K + 2*K*N + 3*M*N f64 elements.
export fn matmul_c128(a: [*]const f64, b: [*]const f64, c: [*]f64, M: u32, N: u32, K: u32, scratch: [*]f64) void {
    const Mu = @as(usize, M);
    const Nu = @as(usize, N);
    const Ku = @as(usize, K);
    const out_count = Mu * Nu;
    const a_count = Mu * Ku;
    const b_count = Ku * Nu;

    const a_re = scratch;
    const a_im = a_re + a_count;
    const b_re = a_im + a_count;
    const b_im = b_re + b_count;
    const p1 = b_im + b_count;
    const p2 = p1 + out_count;
    const p3 = p2 + out_count;

    // Deinterleave A
    var i: usize = 0;
    while (i + 2 <= a_count) : (i += 2) {
        const v0 = simd.load2_f64(a, i * 2); // [re0, im0]
        const v1 = simd.load2_f64(a, i * 2 + 2); // [re1, im1]
        simd.store2_f64(a_re, i, @shuffle(f64, v0, v1, [2]i32{ 0, -1 })); // [re0, re1]
        simd.store2_f64(a_im, i, @shuffle(f64, v0, v1, [2]i32{ 1, -2 })); // [im0, im1]
    }
    while (i < a_count) : (i += 1) {
        a_re[i] = a[i * 2];
        a_im[i] = a[i * 2 + 1];
    }

    // Deinterleave B
    i = 0;
    while (i + 2 <= b_count) : (i += 2) {
        const v0 = simd.load2_f64(b, i * 2);
        const v1 = simd.load2_f64(b, i * 2 + 2);
        simd.store2_f64(b_re, i, @shuffle(f64, v0, v1, [2]i32{ 0, -1 }));
        simd.store2_f64(b_im, i, @shuffle(f64, v0, v1, [2]i32{ 1, -2 }));
    }
    while (i < b_count) : (i += 1) {
        b_re[i] = b[i * 2];
        b_im[i] = b[i * 2 + 1];
    }

    // P1 = A_re @ B_re
    matmul_f64(a_re, b_re, p1, M, N, K);

    // Overwrite a_re → a_sum, b_re → b_sum
    i = 0;
    while (i + 2 <= a_count) : (i += 2) {
        simd.store2_f64(a_re, i, simd.load2_f64(a_re, i) + simd.load2_f64(a_im, i));
    }
    while (i < a_count) : (i += 1) {
        a_re[i] += a_im[i];
    }
    i = 0;
    while (i + 2 <= b_count) : (i += 2) {
        simd.store2_f64(b_re, i, simd.load2_f64(b_re, i) + simd.load2_f64(b_im, i));
    }
    while (i < b_count) : (i += 1) {
        b_re[i] += b_im[i];
    }

    // P2 = A_im @ B_im
    matmul_f64(a_im, b_im, p2, M, N, K);

    // P3 = a_sum @ b_sum
    matmul_f64(a_re, b_re, p3, M, N, K);

    // Combine + reinterleave
    i = 0;
    while (i + 2 <= out_count) : (i += 2) {
        const v_p1 = simd.load2_f64(p1, i);
        const v_p2 = simd.load2_f64(p2, i);
        const c_re = v_p1 - v_p2;
        const c_im = simd.load2_f64(p3, i) - v_p1 - v_p2;
        simd.store2_f64(c, i * 2, @shuffle(f64, c_re, c_im, [2]i32{ 0, -1 }));
        simd.store2_f64(c, i * 2 + 2, @shuffle(f64, c_re, c_im, [2]i32{ 1, -2 }));
    }
    while (i < out_count) : (i += 1) {
        const v_p1 = p1[i];
        const v_p2 = p2[i];
        c[i * 2] = v_p1 - v_p2;
        c[i * 2 + 1] = p3[i] - v_p1 - v_p2;
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
    // scratch: 2*2*2 + 2*2*2 + 3*2*2 = 28
    var scratch = [_]f64{0} ** 28;
    matmul_c128(&a, &b, &c, 2, 2, 2, &scratch);
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
    // scratch: 2*1*1 + 2*1*1 + 3*1*1 = 7
    var scratch = [_]f64{0} ** 7;
    matmul_c128(&a, &b, &c, 1, 1, 1, &scratch);
    try testing.expectApproxEqAbs(c[0], -5.0, 1e-10);
    try testing.expectApproxEqAbs(c[1], 10.0, 1e-10);
}

test "matmul_c128 non-square 2x3x2" {
    const testing = @import("std").testing;
    var a = [_]f64{ 1, 0, 0, 1, 1, 1, 1, -1 };
    var b = [_]f64{ 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1 };
    var c = [_]f64{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    // scratch: 2*2*2 + 2*2*3 + 3*2*3 = 38
    var scratch = [_]f64{0} ** 38;
    matmul_c128(&a, &b, &c, 2, 3, 2, &scratch);
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
    var scratch = [_]f32{0} ** 28;
    matmul_c64(&a, &b, &c, 2, 2, 2, &scratch);
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
    var scratch = [_]f32{0} ** 7;
    matmul_c64(&a, &b, &c, 1, 1, 1, &scratch);
    try testing.expectApproxEqAbs(c[0], -5.0, 1e-5);
    try testing.expectApproxEqAbs(c[1], 10.0, 1e-5);
}

test "matmul_c64 SIMD path N=4" {
    const testing = @import("std").testing;
    var a = [_]f32{ 1, 0 };
    var b = [_]f32{ 1, 0, 0, 1, 1, 1, 1, -1 };
    var c = [_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    // scratch: 2*1*1 + 2*1*4 + 3*1*4 = 22
    var scratch = [_]f32{0} ** 22;
    matmul_c64(&a, &b, &c, 1, 4, 1, &scratch);
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

test "matmul_f64 2x3 @ 3x2" {
    const testing = @import("std").testing;
    // A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]]
    // C = [[58,64],[139,154]]
    const a = [_]f64{ 1, 2, 3, 4, 5, 6 };
    const b = [_]f64{ 7, 8, 9, 10, 11, 12 };
    var c: [4]f64 = undefined;
    matmul_f64(&a, &b, &c, 2, 2, 3);
    try testing.expectApproxEqAbs(c[0], 58.0, 1e-10);
    try testing.expectApproxEqAbs(c[1], 64.0, 1e-10);
    try testing.expectApproxEqAbs(c[2], 139.0, 1e-10);
    try testing.expectApproxEqAbs(c[3], 154.0, 1e-10);
}

test "matmul_f64 identity 3x3" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const id = [_]f64{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    var c: [9]f64 = undefined;
    matmul_f64(&a, &id, &c, 3, 3, 3);
    for (0..9) |i| {
        try testing.expectApproxEqAbs(c[i], a[i], 1e-10);
    }
}

test "matmul_f32 2x2 @ 2x2" {
    const testing = @import("std").testing;
    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 5, 6, 7, 8 };
    var c: [4]f32 = undefined;
    matmul_f32(&a, &b, &c, 2, 2, 2);
    try testing.expectApproxEqAbs(c[0], 19.0, 1e-5);
    try testing.expectApproxEqAbs(c[1], 22.0, 1e-5);
    try testing.expectApproxEqAbs(c[2], 43.0, 1e-5);
    try testing.expectApproxEqAbs(c[3], 50.0, 1e-5);
}
