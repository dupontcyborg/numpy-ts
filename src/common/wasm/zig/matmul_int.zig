//! WASM matmul kernels for integer types (i64, i32, i16, i8).
//!
//! Convention: C = A @ B where A is (M x K), B is (K x N), C is (M x N).
//! All matrices are row-major (C-contiguous).
//!
//! Uses 4×N register-blocked micro-kernels for i64, i32, i16.
//! i8 uses a simpler i-k-j blocked approach with 16-wide SIMD.

const simd = @import("simd.zig");

const TILE_I64 = 64; // Tile size for i64 matmul (tuned for WASM v128)
const TILE_I32 = 128; // Tile size for i32/i16/i8 matmul (tuned for WASM v128)

/// Computes C = A @ B for row-major i64 matrices with wrapping arithmetic.
/// A is (M x K), B is (K x N), C is (M x N).
/// Handles both signed (i64) and unsigned (u64) — wrapping add/mul produce identical bits.
/// Uses a 4×N register-blocked micro-kernel with 4-wide SIMD for the main loop, then 2-wide and scalar remainders.
export fn matmul_i64(a: [*]const i64, b: [*]const i64, c: [*]i64, M: u32, N: u32, K: u32) void {
    @memset(c[0 .. @as(usize, M) * N], 0);

    var ii: usize = 0;
    while (ii < M) : (ii += TILE_I64) {
        const i_end = @min(ii + TILE_I64, M);
        var jj: usize = 0;
        while (jj < N) : (jj += TILE_I64) {
            const j_end = @min(jj + TILE_I64, N);
            var kk: usize = 0;
            while (kk < K) : (kk += TILE_I64) {
                const k_end = @min(kk + TILE_I64, K);

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
    while (ii < M) : (ii += TILE_I32) {
        const i_end = @min(ii + TILE_I32, M);
        var jj: usize = 0;
        while (jj < N) : (jj += TILE_I32) {
            const j_end = @min(jj + TILE_I32, N);
            var kk: usize = 0;
            while (kk < K) : (kk += TILE_I32) {
                const k_end = @min(kk + TILE_I32, K);

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
    while (ii < M) : (ii += TILE_I32) {
        const i_end = @min(ii + TILE_I32, M);
        var jj: usize = 0;
        while (jj < N) : (jj += TILE_I32) {
            const j_end = @min(jj + TILE_I32, N);
            var kk: usize = 0;
            while (kk < K) : (kk += TILE_I32) {
                const k_end = @min(kk + TILE_I32, K);

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
    while (ii < M) : (ii += TILE_I32) {
        const i_end = @min(ii + TILE_I32, M);
        var kk: usize = 0;
        while (kk < K) : (kk += TILE_I32) {
            const k_end = @min(kk + TILE_I32, K);
            var jj: usize = 0;
            while (jj < N) : (jj += TILE_I32) {
                const j_end = @min(jj + TILE_I32, N);

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
