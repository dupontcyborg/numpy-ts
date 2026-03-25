//! WASM inner-product kernels for all numeric types.
//!
//! Convention: C = inner(A, B) where A is (M x K), B is (N x K), C is (M x N).
//! C[i,j] = sum_k A[i*K+k] * B[j*K+k]   (dot product of row i of A with row j of B)
//!
//! All matrices are row-major (C-contiguous).
//! Complex matrices are interleaved [re, im, re, im, ...]; M, N, K are element counts.
//!
//! Real types use 4-row blocking: process 4 A rows simultaneously, sharing each
//! B row load across all 4 dot products. Reduces B memory traffic by 4×.

const simd = @import("simd.zig");

/// Computes C = inner(A, B) for row-major f64 arrays.
/// A is (M x K), B is (N x K), C is (M x N).
/// Uses a 4×N register-blocked micro-kernel with 8-wide SIMD for the main loop, then 4-wide and scalar remainders.
export fn inner_f64(a: [*]const f64, b: [*]const f64, c: [*]f64, M: u32, N: u32, K: u32) void {
    // 4-row blocked
    var i: usize = 0;
    while (i + 4 <= M) : (i += 4) {
        for (0..N) |j| {
            const b_row = j * K;
            // 4 rows × 2 V2f64 accumulators = 8 registers
            var a0_0: simd.V2f64 = @splat(0);
            var a0_1: simd.V2f64 = @splat(0);
            var a1_0: simd.V2f64 = @splat(0);
            var a1_1: simd.V2f64 = @splat(0);
            var a2_0: simd.V2f64 = @splat(0);
            var a2_1: simd.V2f64 = @splat(0);
            var a3_0: simd.V2f64 = @splat(0);
            var a3_1: simd.V2f64 = @splat(0);

            var k: usize = 0;
            while (k + 4 <= K) : (k += 4) {
                // Load B[j, k:k+4] once — reused for 4 A rows
                const b0 = simd.load2_f64(b, b_row + k);
                const b1 = simd.load2_f64(b, b_row + k + 2);
                a0_0 += simd.load2_f64(a, (i + 0) * K + k) * b0;
                a0_1 += simd.load2_f64(a, (i + 0) * K + k + 2) * b1;
                a1_0 += simd.load2_f64(a, (i + 1) * K + k) * b0;
                a1_1 += simd.load2_f64(a, (i + 1) * K + k + 2) * b1;
                a2_0 += simd.load2_f64(a, (i + 2) * K + k) * b0;
                a2_1 += simd.load2_f64(a, (i + 2) * K + k + 2) * b1;
                a3_0 += simd.load2_f64(a, (i + 3) * K + k) * b0;
                a3_1 += simd.load2_f64(a, (i + 3) * K + k + 2) * b1;
            }
            // V2f64 remainder
            while (k + 2 <= K) : (k += 2) {
                const bv = simd.load2_f64(b, b_row + k);
                a0_0 += simd.load2_f64(a, (i + 0) * K + k) * bv;
                a1_0 += simd.load2_f64(a, (i + 1) * K + k) * bv;
                a2_0 += simd.load2_f64(a, (i + 2) * K + k) * bv;
                a3_0 += simd.load2_f64(a, (i + 3) * K + k) * bv;
            }
            // Horizontal sum + scalar remainder
            var s0: f64 = a0_0[0] + a0_0[1] + a0_1[0] + a0_1[1];
            var s1: f64 = a1_0[0] + a1_0[1] + a1_1[0] + a1_1[1];
            var s2: f64 = a2_0[0] + a2_0[1] + a2_1[0] + a2_1[1];
            var s3: f64 = a3_0[0] + a3_0[1] + a3_1[0] + a3_1[1];
            while (k < K) : (k += 1) {
                const bv = b[b_row + k];
                s0 += a[(i + 0) * K + k] * bv;
                s1 += a[(i + 1) * K + k] * bv;
                s2 += a[(i + 2) * K + k] * bv;
                s3 += a[(i + 3) * K + k] * bv;
            }
            c[(i + 0) * N + j] = s0;
            c[(i + 1) * N + j] = s1;
            c[(i + 2) * N + j] = s2;
            c[(i + 3) * N + j] = s3;
        }
    }
    // Remainder rows
    while (i < M) : (i += 1) {
        const a_row = i * K;
        for (0..N) |j| {
            const b_row = j * K;
            var acc0: simd.V2f64 = @splat(0);
            var acc1: simd.V2f64 = @splat(0);
            var k: usize = 0;
            while (k + 4 <= K) : (k += 4) {
                acc0 += simd.load2_f64(a, a_row + k) * simd.load2_f64(b, b_row + k);
                acc1 += simd.load2_f64(a, a_row + k + 2) * simd.load2_f64(b, b_row + k + 2);
            }
            while (k + 2 <= K) : (k += 2) {
                acc0 += simd.load2_f64(a, a_row + k) * simd.load2_f64(b, b_row + k);
            }
            var sum: f64 = acc0[0] + acc0[1] + acc1[0] + acc1[1];
            while (k < K) : (k += 1) {
                sum += a[a_row + k] * b[b_row + k];
            }
            c[i * N + j] = sum;
        }
    }
}

/// Computes C = inner(A, B) for row-major f32 arrays.
/// A is (M x K), B is (N x K), C is (M x N).
/// Uses a 4×N register-blocked micro-kernel with 8-wide SIMD for the main loop, then 4-wide and scalar remainders.
export fn inner_f32(a: [*]const f32, b: [*]const f32, c: [*]f32, M: u32, N: u32, K: u32) void {
    var i: usize = 0;
    while (i + 4 <= M) : (i += 4) {
        for (0..N) |j| {
            const b_row = j * K;
            var a0_0: simd.V4f32 = @splat(0);
            var a0_1: simd.V4f32 = @splat(0);
            var a1_0: simd.V4f32 = @splat(0);
            var a1_1: simd.V4f32 = @splat(0);
            var a2_0: simd.V4f32 = @splat(0);
            var a2_1: simd.V4f32 = @splat(0);
            var a3_0: simd.V4f32 = @splat(0);
            var a3_1: simd.V4f32 = @splat(0);

            var k: usize = 0;
            while (k + 8 <= K) : (k += 8) {
                const b0 = simd.load4_f32(b, b_row + k);
                const b1 = simd.load4_f32(b, b_row + k + 4);
                a0_0 += simd.load4_f32(a, (i + 0) * K + k) * b0;
                a0_1 += simd.load4_f32(a, (i + 0) * K + k + 4) * b1;
                a1_0 += simd.load4_f32(a, (i + 1) * K + k) * b0;
                a1_1 += simd.load4_f32(a, (i + 1) * K + k + 4) * b1;
                a2_0 += simd.load4_f32(a, (i + 2) * K + k) * b0;
                a2_1 += simd.load4_f32(a, (i + 2) * K + k + 4) * b1;
                a3_0 += simd.load4_f32(a, (i + 3) * K + k) * b0;
                a3_1 += simd.load4_f32(a, (i + 3) * K + k + 4) * b1;
            }
            while (k + 4 <= K) : (k += 4) {
                const bv = simd.load4_f32(b, b_row + k);
                a0_0 += simd.load4_f32(a, (i + 0) * K + k) * bv;
                a1_0 += simd.load4_f32(a, (i + 1) * K + k) * bv;
                a2_0 += simd.load4_f32(a, (i + 2) * K + k) * bv;
                a3_0 += simd.load4_f32(a, (i + 3) * K + k) * bv;
            }
            var s0: f32 = a0_0[0] + a0_0[1] + a0_0[2] + a0_0[3] + a0_1[0] + a0_1[1] + a0_1[2] + a0_1[3];
            var s1: f32 = a1_0[0] + a1_0[1] + a1_0[2] + a1_0[3] + a1_1[0] + a1_1[1] + a1_1[2] + a1_1[3];
            var s2: f32 = a2_0[0] + a2_0[1] + a2_0[2] + a2_0[3] + a2_1[0] + a2_1[1] + a2_1[2] + a2_1[3];
            var s3: f32 = a3_0[0] + a3_0[1] + a3_0[2] + a3_0[3] + a3_1[0] + a3_1[1] + a3_1[2] + a3_1[3];
            while (k < K) : (k += 1) {
                const bv = b[b_row + k];
                s0 += a[(i + 0) * K + k] * bv;
                s1 += a[(i + 1) * K + k] * bv;
                s2 += a[(i + 2) * K + k] * bv;
                s3 += a[(i + 3) * K + k] * bv;
            }
            c[(i + 0) * N + j] = s0;
            c[(i + 1) * N + j] = s1;
            c[(i + 2) * N + j] = s2;
            c[(i + 3) * N + j] = s3;
        }
    }
    // Remainder rows
    while (i < M) : (i += 1) {
        const a_row = i * K;
        for (0..N) |j| {
            const b_row = j * K;
            var acc0: simd.V4f32 = @splat(0);
            var acc1: simd.V4f32 = @splat(0);
            var k: usize = 0;
            while (k + 8 <= K) : (k += 8) {
                acc0 += simd.load4_f32(a, a_row + k) * simd.load4_f32(b, b_row + k);
                acc1 += simd.load4_f32(a, a_row + k + 4) * simd.load4_f32(b, b_row + k + 4);
            }
            while (k + 4 <= K) : (k += 4) {
                acc0 += simd.load4_f32(a, a_row + k) * simd.load4_f32(b, b_row + k);
            }
            var sum: f32 = acc0[0] + acc0[1] + acc0[2] + acc0[3] + acc1[0] + acc1[1] + acc1[2] + acc1[3];
            while (k < K) : (k += 1) {
                sum += a[a_row + k] * b[b_row + k];
            }
            c[i * N + j] = sum;
        }
    }
}

/// Computes C = inner(A, B) for row-major complex64 arrays (interleaved f32).
/// A is (M x K), B is (N x K), C is (M x N). Scratch: 2*M*K + 2*N*K + 3*M*N.
/// Uses Gauss trick: 3 real inner products instead of 4 complex multiplies.
export fn inner_c64(a: [*]const f32, b: [*]const f32, c: [*]f32, M: u32, N: u32, K: u32, scratch: [*]f32) void {
    const Mu = @as(usize, M);
    const Nu = @as(usize, N);
    const Ku = @as(usize, K);
    const out_count = Mu * Nu;
    const a_count = Mu * Ku;
    const b_count = Nu * Ku;

    const a_re = scratch;
    const a_im = a_re + a_count;
    const b_re = a_im + a_count;
    const b_im = b_re + b_count;
    const p1 = b_im + b_count;
    const p2 = p1 + out_count;
    const p3 = p2 + out_count;

    // Deinterleave A
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

    // Deinterleave B
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

    // P1 = inner(A_re, B_re)
    inner_f32(a_re, b_re, p1, M, N, K);

    // Overwrite a_re → a_sum, b_re → b_sum
    i = 0;
    while (i + 4 <= a_count) : (i += 4) {
        simd.store4_f32(a_re, i, simd.load4_f32(a_re, i) + simd.load4_f32(a_im, i));
    }
    while (i < a_count) : (i += 1) a_re[i] += a_im[i];
    i = 0;
    while (i + 4 <= b_count) : (i += 4) {
        simd.store4_f32(b_re, i, simd.load4_f32(b_re, i) + simd.load4_f32(b_im, i));
    }
    while (i < b_count) : (i += 1) b_re[i] += b_im[i];

    // P2 = inner(A_im, B_im)
    inner_f32(a_im, b_im, p2, M, N, K);

    // P3 = inner(A_sum, B_sum)
    inner_f32(a_re, b_re, p3, M, N, K);

    // Combine + reinterleave: C_re = P1 - P2, C_im = P3 - P1 - P2
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

/// Computes C = inner(A, B) for row-major complex128 arrays (interleaved f64).
/// Same algorithm as inner_c64 but for f64. Scratch: 2*M*K + 2*N*K + 3*M*N.
export fn inner_c128(a: [*]const f64, b: [*]const f64, c: [*]f64, M: u32, N: u32, K: u32, scratch: [*]f64) void {
    const Mu = @as(usize, M);
    const Nu = @as(usize, N);
    const Ku = @as(usize, K);
    const out_count = Mu * Nu;
    const a_count = Mu * Ku;
    const b_count = Nu * Ku;

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
        const v0 = simd.load2_f64(a, i * 2);
        const v1 = simd.load2_f64(a, i * 2 + 2);
        simd.store2_f64(a_re, i, @shuffle(f64, v0, v1, [2]i32{ 0, -1 }));
        simd.store2_f64(a_im, i, @shuffle(f64, v0, v1, [2]i32{ 1, -2 }));
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

    // P1 = inner(A_re, B_re)
    inner_f64(a_re, b_re, p1, M, N, K);

    // Overwrite a_re → a_sum, b_re → b_sum
    i = 0;
    while (i + 2 <= a_count) : (i += 2) {
        simd.store2_f64(a_re, i, simd.load2_f64(a_re, i) + simd.load2_f64(a_im, i));
    }
    while (i < a_count) : (i += 1) a_re[i] += a_im[i];
    i = 0;
    while (i + 2 <= b_count) : (i += 2) {
        simd.store2_f64(b_re, i, simd.load2_f64(b_re, i) + simd.load2_f64(b_im, i));
    }
    while (i < b_count) : (i += 1) b_re[i] += b_im[i];

    // P2 = inner(A_im, B_im)
    inner_f64(a_im, b_im, p2, M, N, K);

    // P3 = inner(A_sum, B_sum)
    inner_f64(a_re, b_re, p3, M, N, K);

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

/// Computes C = inner(A, B) for row-major i64 arrays.
/// A is (M x K), B is (N x K), C is (M x N).
/// Handles both signed (i64) and unsigned (u64) with wrapping arithmetic.
/// Uses a 4×N register-blocked micro-kernel with 4-wide SIMD for the main loop, then smaller remainders.
export fn inner_i64(a: [*]const i64, b: [*]const i64, c: [*]i64, M: u32, N: u32, K: u32) void {
    var i: usize = 0;
    while (i + 4 <= M) : (i += 4) {
        for (0..N) |j| {
            const b_row = j * K;
            var a0_0: simd.V2i64 = @splat(0);
            var a0_1: simd.V2i64 = @splat(0);
            var a1_0: simd.V2i64 = @splat(0);
            var a1_1: simd.V2i64 = @splat(0);
            var a2_0: simd.V2i64 = @splat(0);
            var a2_1: simd.V2i64 = @splat(0);
            var a3_0: simd.V2i64 = @splat(0);
            var a3_1: simd.V2i64 = @splat(0);

            var k: usize = 0;
            while (k + 4 <= K) : (k += 4) {
                const b0 = simd.load2_i64(b, b_row + k);
                const b1 = simd.load2_i64(b, b_row + k + 2);
                a0_0 +%= simd.load2_i64(a, (i + 0) * K + k) *% b0;
                a0_1 +%= simd.load2_i64(a, (i + 0) * K + k + 2) *% b1;
                a1_0 +%= simd.load2_i64(a, (i + 1) * K + k) *% b0;
                a1_1 +%= simd.load2_i64(a, (i + 1) * K + k + 2) *% b1;
                a2_0 +%= simd.load2_i64(a, (i + 2) * K + k) *% b0;
                a2_1 +%= simd.load2_i64(a, (i + 2) * K + k + 2) *% b1;
                a3_0 +%= simd.load2_i64(a, (i + 3) * K + k) *% b0;
                a3_1 +%= simd.load2_i64(a, (i + 3) * K + k + 2) *% b1;
            }
            while (k + 2 <= K) : (k += 2) {
                const bv = simd.load2_i64(b, b_row + k);
                a0_0 +%= simd.load2_i64(a, (i + 0) * K + k) *% bv;
                a1_0 +%= simd.load2_i64(a, (i + 1) * K + k) *% bv;
                a2_0 +%= simd.load2_i64(a, (i + 2) * K + k) *% bv;
                a3_0 +%= simd.load2_i64(a, (i + 3) * K + k) *% bv;
            }
            var s0: i64 = a0_0[0] +% a0_0[1] +% a0_1[0] +% a0_1[1];
            var s1: i64 = a1_0[0] +% a1_0[1] +% a1_1[0] +% a1_1[1];
            var s2: i64 = a2_0[0] +% a2_0[1] +% a2_1[0] +% a2_1[1];
            var s3: i64 = a3_0[0] +% a3_0[1] +% a3_1[0] +% a3_1[1];
            while (k < K) : (k += 1) {
                const bv = b[b_row + k];
                s0 +%= a[(i + 0) * K + k] *% bv;
                s1 +%= a[(i + 1) * K + k] *% bv;
                s2 +%= a[(i + 2) * K + k] *% bv;
                s3 +%= a[(i + 3) * K + k] *% bv;
            }
            c[(i + 0) * N + j] = s0;
            c[(i + 1) * N + j] = s1;
            c[(i + 2) * N + j] = s2;
            c[(i + 3) * N + j] = s3;
        }
    }
    while (i < M) : (i += 1) {
        const a_row = i * K;
        for (0..N) |j| {
            const b_row = j * K;
            var acc: simd.V2i64 = @splat(0);
            var k: usize = 0;
            while (k + 2 <= K) : (k += 2) {
                acc +%= simd.load2_i64(a, a_row + k) *% simd.load2_i64(b, b_row + k);
            }
            var sum: i64 = acc[0] +% acc[1];
            while (k < K) : (k += 1) {
                sum +%= a[a_row + k] *% b[b_row + k];
            }
            c[i * N + j] = sum;
        }
    }
}

/// Computes C = inner(A, B) for row-major i32 arrays.
/// A is (M x K), B is (N x K), C is (M x N).
/// Handles both signed (i32) and unsigned (u32) with wrapping arithmetic.
/// Uses a 4×N register-blocked micro-kernel with 4-wide SIMD for the main loop, then smaller remainders.
export fn inner_i32(a: [*]const i32, b: [*]const i32, c: [*]i32, M: u32, N: u32, K: u32) void {
    var i: usize = 0;
    while (i + 4 <= M) : (i += 4) {
        for (0..N) |j| {
            const b_row = j * K;
            var a0_0: simd.V4i32 = @splat(0);
            var a0_1: simd.V4i32 = @splat(0);
            var a1_0: simd.V4i32 = @splat(0);
            var a1_1: simd.V4i32 = @splat(0);
            var a2_0: simd.V4i32 = @splat(0);
            var a2_1: simd.V4i32 = @splat(0);
            var a3_0: simd.V4i32 = @splat(0);
            var a3_1: simd.V4i32 = @splat(0);

            var k: usize = 0;
            while (k + 8 <= K) : (k += 8) {
                const b0 = simd.load4_i32(b, b_row + k);
                const b1 = simd.load4_i32(b, b_row + k + 4);
                a0_0 +%= simd.load4_i32(a, (i + 0) * K + k) *% b0;
                a0_1 +%= simd.load4_i32(a, (i + 0) * K + k + 4) *% b1;
                a1_0 +%= simd.load4_i32(a, (i + 1) * K + k) *% b0;
                a1_1 +%= simd.load4_i32(a, (i + 1) * K + k + 4) *% b1;
                a2_0 +%= simd.load4_i32(a, (i + 2) * K + k) *% b0;
                a2_1 +%= simd.load4_i32(a, (i + 2) * K + k + 4) *% b1;
                a3_0 +%= simd.load4_i32(a, (i + 3) * K + k) *% b0;
                a3_1 +%= simd.load4_i32(a, (i + 3) * K + k + 4) *% b1;
            }
            while (k + 4 <= K) : (k += 4) {
                const bv = simd.load4_i32(b, b_row + k);
                a0_0 +%= simd.load4_i32(a, (i + 0) * K + k) *% bv;
                a1_0 +%= simd.load4_i32(a, (i + 1) * K + k) *% bv;
                a2_0 +%= simd.load4_i32(a, (i + 2) * K + k) *% bv;
                a3_0 +%= simd.load4_i32(a, (i + 3) * K + k) *% bv;
            }
            var s0: i32 = a0_0[0] +% a0_0[1] +% a0_0[2] +% a0_0[3] +% a0_1[0] +% a0_1[1] +% a0_1[2] +% a0_1[3];
            var s1: i32 = a1_0[0] +% a1_0[1] +% a1_0[2] +% a1_0[3] +% a1_1[0] +% a1_1[1] +% a1_1[2] +% a1_1[3];
            var s2: i32 = a2_0[0] +% a2_0[1] +% a2_0[2] +% a2_0[3] +% a2_1[0] +% a2_1[1] +% a2_1[2] +% a2_1[3];
            var s3: i32 = a3_0[0] +% a3_0[1] +% a3_0[2] +% a3_0[3] +% a3_1[0] +% a3_1[1] +% a3_1[2] +% a3_1[3];
            while (k < K) : (k += 1) {
                const bv = b[b_row + k];
                s0 +%= a[(i + 0) * K + k] *% bv;
                s1 +%= a[(i + 1) * K + k] *% bv;
                s2 +%= a[(i + 2) * K + k] *% bv;
                s3 +%= a[(i + 3) * K + k] *% bv;
            }
            c[(i + 0) * N + j] = s0;
            c[(i + 1) * N + j] = s1;
            c[(i + 2) * N + j] = s2;
            c[(i + 3) * N + j] = s3;
        }
    }
    while (i < M) : (i += 1) {
        const a_row = i * K;
        for (0..N) |j| {
            const b_row = j * K;
            var acc: simd.V4i32 = @splat(0);
            var k: usize = 0;
            while (k + 4 <= K) : (k += 4) {
                acc +%= simd.load4_i32(a, a_row + k) *% simd.load4_i32(b, b_row + k);
            }
            var sum: i32 = acc[0] +% acc[1] +% acc[2] +% acc[3];
            while (k < K) : (k += 1) {
                sum +%= a[a_row + k] *% b[b_row + k];
            }
            c[i * N + j] = sum;
        }
    }
}

/// Computes C = inner(A, B) for row-major i16 arrays.
/// A is (M x K), B is (N x K), C is (M x N).
/// Handles both signed (i16) and unsigned (u16) with wrapping arithmetic.
/// Uses a 4×N register-blocked micro-kernel with 4-wide SIMD for the main loop, then smaller remainders.
export fn inner_i16(a: [*]const i16, b: [*]const i16, c: [*]i16, M: u32, N: u32, K: u32) void {
    var i: usize = 0;
    while (i + 4 <= M) : (i += 4) {
        for (0..N) |j| {
            const b_row = j * K;
            var a0_0: simd.V8i16 = @splat(0);
            var a0_1: simd.V8i16 = @splat(0);
            var a1_0: simd.V8i16 = @splat(0);
            var a1_1: simd.V8i16 = @splat(0);
            var a2_0: simd.V8i16 = @splat(0);
            var a2_1: simd.V8i16 = @splat(0);
            var a3_0: simd.V8i16 = @splat(0);
            var a3_1: simd.V8i16 = @splat(0);

            var k: usize = 0;
            while (k + 16 <= K) : (k += 16) {
                const b0 = simd.load8_i16(b, b_row + k);
                const b1 = simd.load8_i16(b, b_row + k + 8);
                a0_0 +%= simd.load8_i16(a, (i + 0) * K + k) *% b0;
                a0_1 +%= simd.load8_i16(a, (i + 0) * K + k + 8) *% b1;
                a1_0 +%= simd.load8_i16(a, (i + 1) * K + k) *% b0;
                a1_1 +%= simd.load8_i16(a, (i + 1) * K + k + 8) *% b1;
                a2_0 +%= simd.load8_i16(a, (i + 2) * K + k) *% b0;
                a2_1 +%= simd.load8_i16(a, (i + 2) * K + k + 8) *% b1;
                a3_0 +%= simd.load8_i16(a, (i + 3) * K + k) *% b0;
                a3_1 +%= simd.load8_i16(a, (i + 3) * K + k + 8) *% b1;
            }
            while (k + 8 <= K) : (k += 8) {
                const bv = simd.load8_i16(b, b_row + k);
                a0_0 +%= simd.load8_i16(a, (i + 0) * K + k) *% bv;
                a1_0 +%= simd.load8_i16(a, (i + 1) * K + k) *% bv;
                a2_0 +%= simd.load8_i16(a, (i + 2) * K + k) *% bv;
                a3_0 +%= simd.load8_i16(a, (i + 3) * K + k) *% bv;
            }
            var s0: i16 = a0_0[0] +% a0_0[1] +% a0_0[2] +% a0_0[3] +% a0_0[4] +% a0_0[5] +% a0_0[6] +% a0_0[7] +% a0_1[0] +% a0_1[1] +% a0_1[2] +% a0_1[3] +% a0_1[4] +% a0_1[5] +% a0_1[6] +% a0_1[7];
            var s1: i16 = a1_0[0] +% a1_0[1] +% a1_0[2] +% a1_0[3] +% a1_0[4] +% a1_0[5] +% a1_0[6] +% a1_0[7] +% a1_1[0] +% a1_1[1] +% a1_1[2] +% a1_1[3] +% a1_1[4] +% a1_1[5] +% a1_1[6] +% a1_1[7];
            var s2: i16 = a2_0[0] +% a2_0[1] +% a2_0[2] +% a2_0[3] +% a2_0[4] +% a2_0[5] +% a2_0[6] +% a2_0[7] +% a2_1[0] +% a2_1[1] +% a2_1[2] +% a2_1[3] +% a2_1[4] +% a2_1[5] +% a2_1[6] +% a2_1[7];
            var s3: i16 = a3_0[0] +% a3_0[1] +% a3_0[2] +% a3_0[3] +% a3_0[4] +% a3_0[5] +% a3_0[6] +% a3_0[7] +% a3_1[0] +% a3_1[1] +% a3_1[2] +% a3_1[3] +% a3_1[4] +% a3_1[5] +% a3_1[6] +% a3_1[7];
            while (k < K) : (k += 1) {
                const bv = b[b_row + k];
                s0 +%= a[(i + 0) * K + k] *% bv;
                s1 +%= a[(i + 1) * K + k] *% bv;
                s2 +%= a[(i + 2) * K + k] *% bv;
                s3 +%= a[(i + 3) * K + k] *% bv;
            }
            c[(i + 0) * N + j] = s0;
            c[(i + 1) * N + j] = s1;
            c[(i + 2) * N + j] = s2;
            c[(i + 3) * N + j] = s3;
        }
    }
    while (i < M) : (i += 1) {
        const a_row = i * K;
        for (0..N) |j| {
            const b_row = j * K;
            var acc: simd.V8i16 = @splat(0);
            var k: usize = 0;
            while (k + 8 <= K) : (k += 8) {
                acc +%= simd.load8_i16(a, a_row + k) *% simd.load8_i16(b, b_row + k);
            }
            var sum: i16 = acc[0] +% acc[1] +% acc[2] +% acc[3] +% acc[4] +% acc[5] +% acc[6] +% acc[7];
            while (k < K) : (k += 1) {
                sum +%= a[a_row + k] *% b[b_row + k];
            }
            c[i * N + j] = sum;
        }
    }
}

/// Computes C = inner(A, B) for row-major i8 arrays.
/// A is (M x K), B is (N x K), C is (M x N).
/// Handles both signed (i8) and unsigned (u8) with wrapping arithmetic.
/// Uses a 4×N register-blocked micro-kernel with 16-wide SIMD for the main loop, then smaller remainders.
export fn inner_i8(a: [*]const i8, b: [*]const i8, c: [*]i8, M: u32, N: u32, K: u32) void {
    const k_simd = K & ~@as(u32, 15);

    for (0..M) |i| {
        const a_row = i * K;
        for (0..N) |j| {
            const b_row = j * K;
            var acc: simd.V16i8 = @splat(0);

            var k: u32 = 0;
            while (k < k_simd) : (k += 16) {
                acc = simd.muladd_i8x16(acc, simd.load16_i8(a, a_row + k), simd.load16_i8(b, b_row + k));
            }

            var sum: i8 = 0;
            for (0..16) |lane| {
                sum +%= acc[lane];
            }
            while (k < K) : (k += 1) {
                sum +%= a[a_row + k] *% b[b_row + k];
            }

            c[i * N + j] = sum;
        }
    }
}

// --- Tests ---

test "inner_f64 2x3 @ 2x3 → 2x2" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1, 2, 3, 4, 5, 6 };
    const b = [_]f64{ 7, 8, 9, 10, 11, 12 };
    var c: [4]f64 = undefined;
    inner_f64(&a, &b, &c, 2, 2, 3);
    try testing.expectApproxEqAbs(c[0], 50.0, 1e-10);
    try testing.expectApproxEqAbs(c[1], 68.0, 1e-10);
    try testing.expectApproxEqAbs(c[2], 122.0, 1e-10);
    try testing.expectApproxEqAbs(c[3], 167.0, 1e-10);
}

test "inner_f32 3x2 @ 2x2 → 3x2" {
    const testing = @import("std").testing;
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const b = [_]f32{ 1, 0, 0, 1 };
    var c: [6]f32 = undefined;
    inner_f32(&a, &b, &c, 3, 2, 2);
    try testing.expectApproxEqAbs(c[0], 1.0, 1e-5);
    try testing.expectApproxEqAbs(c[1], 2.0, 1e-5);
    try testing.expectApproxEqAbs(c[2], 3.0, 1e-5);
    try testing.expectApproxEqAbs(c[3], 4.0, 1e-5);
    try testing.expectApproxEqAbs(c[4], 5.0, 1e-5);
    try testing.expectApproxEqAbs(c[5], 6.0, 1e-5);
}

test "inner_c128 1x2 @ 1x2 → 1x1" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1, 2, 3, 4 };
    const b = [_]f64{ 5, 6, 7, 8 };
    var c: [2]f64 = undefined;
    // scratch: 2*1*2 + 2*1*2 + 3*1*1 = 11
    var scratch_f64 = [_]f64{0} ** 11;
    inner_c128(&a, &b, &c, 1, 1, 2, &scratch_f64);
    try testing.expectApproxEqAbs(c[0], -18.0, 1e-10);
    try testing.expectApproxEqAbs(c[1], 68.0, 1e-10);
}

test "inner_c64 1x2 @ 1x2 → 1x1" {
    const testing = @import("std").testing;
    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 5, 6, 7, 8 };
    var c: [2]f32 = undefined;
    var scratch_f32 = [_]f32{0} ** 11;
    inner_c64(&a, &b, &c, 1, 1, 2, &scratch_f32);
    try testing.expectApproxEqAbs(c[0], -18.0, 1e-5);
    try testing.expectApproxEqAbs(c[1], 68.0, 1e-5);
}

test "inner_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 1, 2, 3 };
    const b = [_]i64{ 4, 5, 6 };
    var c: [1]i64 = undefined;
    inner_i64(&a, &b, &c, 1, 1, 3);
    try testing.expectEqual(c[0], 32);
}

test "inner_i32 2x3 @ 2x3 → 2x2" {
    const testing = @import("std").testing;
    const a = [_]i32{ 1, 2, 3, 4, 5, 6 };
    const b = [_]i32{ 7, 8, 9, 10, 11, 12 };
    var c: [4]i32 = undefined;
    inner_i32(&a, &b, &c, 2, 2, 3);
    try testing.expectEqual(c[0], 50);
    try testing.expectEqual(c[1], 68);
    try testing.expectEqual(c[2], 122);
    try testing.expectEqual(c[3], 167);
}

test "inner_i16 wrapping" {
    const testing = @import("std").testing;
    const a = [_]i16{ 127, 127, 127, 127 };
    const b = [_]i16{ 127, 127, 127, 127 };
    var c: [1]i16 = undefined;
    inner_i16(&a, &b, &c, 1, 1, 4);
    const expected: i16 = @truncate(@as(i32, 127) * 127 * 4);
    try testing.expectEqual(c[0], expected);
}

test "inner_i8 wrapping" {
    const testing = @import("std").testing;
    const a = [_]i8{ 10, 10, 10, 10 };
    const b = [_]i8{ 10, 10, 10, 10 };
    var c: [1]i8 = undefined;
    inner_i8(&a, &b, &c, 1, 1, 4);
    const expected: i8 = @truncate(@as(i32, 10) * 10 * 4);
    try testing.expectEqual(c[0], expected);
}
