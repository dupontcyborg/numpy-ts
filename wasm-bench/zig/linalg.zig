// Linear algebra WASM kernels: matvec, vecmat, vecdot, outer, kron, cross, norm
//
// Uses native v128 widths: @Vector(2,f64) / @Vector(4,f32)
// Pointer-cast loads/stores for guaranteed v128.load/v128.store opcodes.

const simd = @import("simd.zig");

// ─── matvec: A[m×n] · x[n] → out[m] ────────────────────────────────────────

export fn matvec_f64(a: [*]const f64, x: [*]const f64, out: [*]f64, m: u32, n: u32) void {
    const rows = @as(usize, m);
    const cols = @as(usize, n);
    for (0..rows) |i| {
        var acc0: simd.V2f64 = @splat(0.0);
        var acc1: simd.V2f64 = @splat(0.0);
        const row = a + i * cols;
        var j: usize = 0;
        while (j + 4 <= cols) : (j += 4) {
            acc0 += simd.load2_f64(row, j) * simd.load2_f64(x, j);
            acc1 += simd.load2_f64(row, j + 2) * simd.load2_f64(x, j + 2);
        }
        while (j + 2 <= cols) : (j += 2) {
            acc0 += simd.load2_f64(row, j) * simd.load2_f64(x, j);
        }
        acc0 += acc1;
        var sum: f64 = acc0[0] + acc0[1];
        while (j < cols) : (j += 1) {
            sum += row[j] * x[j];
        }
        out[i] = sum;
    }
}

export fn matvec_f32(a: [*]const f32, x: [*]const f32, out: [*]f32, m: u32, n: u32) void {
    const rows = @as(usize, m);
    const cols = @as(usize, n);
    for (0..rows) |i| {
        var acc0: simd.V4f32 = @splat(0.0);
        var acc1: simd.V4f32 = @splat(0.0);
        const row = a + i * cols;
        var j: usize = 0;
        while (j + 8 <= cols) : (j += 8) {
            acc0 += simd.load4_f32(row, j) * simd.load4_f32(x, j);
            acc1 += simd.load4_f32(row, j + 4) * simd.load4_f32(x, j + 4);
        }
        while (j + 4 <= cols) : (j += 4) {
            acc0 += simd.load4_f32(row, j) * simd.load4_f32(x, j);
        }
        acc0 += acc1;
        var sum: f32 = acc0[0] + acc0[1] + acc0[2] + acc0[3];
        while (j < cols) : (j += 1) {
            sum += row[j] * x[j];
        }
        out[i] = sum;
    }
}

// ─── vecmat: x[m] · A[m×n] → out[n] ────────────────────────────────────────

export fn vecmat_f64(x: [*]const f64, a: [*]const f64, out: [*]f64, m: u32, n: u32) void {
    const rows = @as(usize, m);
    const cols = @as(usize, n);
    // Zero output
    var j: usize = 0;
    while (j + 2 <= cols) : (j += 2) {
        simd.store2_f64(out, j, @splat(0.0));
    }
    while (j < cols) : (j += 1) {
        out[j] = 0.0;
    }
    // Accumulate: out[j] += x[i] * A[i,j]
    for (0..rows) |i| {
        const xi: simd.V2f64 = @splat(x[i]);
        const row = a + i * cols;
        j = 0;
        while (j + 4 <= cols) : (j += 4) {
            simd.store2_f64(out, j, simd.load2_f64(out, j) + xi * simd.load2_f64(row, j));
            simd.store2_f64(out, j + 2, simd.load2_f64(out, j + 2) + xi * simd.load2_f64(row, j + 2));
        }
        while (j + 2 <= cols) : (j += 2) {
            simd.store2_f64(out, j, simd.load2_f64(out, j) + xi * simd.load2_f64(row, j));
        }
        while (j < cols) : (j += 1) {
            out[j] += x[i] * row[j];
        }
    }
}

export fn vecmat_f32(x: [*]const f32, a: [*]const f32, out: [*]f32, m: u32, n: u32) void {
    const rows = @as(usize, m);
    const cols = @as(usize, n);
    var j: usize = 0;
    while (j + 4 <= cols) : (j += 4) {
        simd.store4_f32(out, j, @splat(0.0));
    }
    while (j < cols) : (j += 1) {
        out[j] = 0.0;
    }
    for (0..rows) |i| {
        const xi: simd.V4f32 = @splat(x[i]);
        const row = a + i * cols;
        j = 0;
        while (j + 8 <= cols) : (j += 8) {
            simd.store4_f32(out, j, simd.load4_f32(out, j) + xi * simd.load4_f32(row, j));
            simd.store4_f32(out, j + 4, simd.load4_f32(out, j + 4) + xi * simd.load4_f32(row, j + 4));
        }
        while (j + 4 <= cols) : (j += 4) {
            simd.store4_f32(out, j, simd.load4_f32(out, j) + xi * simd.load4_f32(row, j));
        }
        while (j < cols) : (j += 1) {
            out[j] += x[i] * row[j];
        }
    }
}

// ─── vecdot: batched dot products. a[batch×len], b[batch×len] → out[batch] ──

export fn vecdot_f64(a: [*]const f64, b: [*]const f64, out: [*]f64, nbatch: u32, veclen: u32) void {
    const batch = @as(usize, nbatch);
    const len = @as(usize, veclen);
    for (0..batch) |bi| {
        const off = bi * len;
        var acc0: simd.V2f64 = @splat(0.0);
        var acc1: simd.V2f64 = @splat(0.0);
        var j: usize = 0;
        while (j + 4 <= len) : (j += 4) {
            acc0 += simd.load2_f64(a, off + j) * simd.load2_f64(b, off + j);
            acc1 += simd.load2_f64(a, off + j + 2) * simd.load2_f64(b, off + j + 2);
        }
        while (j + 2 <= len) : (j += 2) {
            acc0 += simd.load2_f64(a, off + j) * simd.load2_f64(b, off + j);
        }
        acc0 += acc1;
        var sum: f64 = acc0[0] + acc0[1];
        while (j < len) : (j += 1) {
            sum += a[off + j] * b[off + j];
        }
        out[bi] = sum;
    }
}

export fn vecdot_f32(a: [*]const f32, b: [*]const f32, out: [*]f32, nbatch: u32, veclen: u32) void {
    const batch = @as(usize, nbatch);
    const len = @as(usize, veclen);
    for (0..batch) |bi| {
        const off = bi * len;
        var acc0: simd.V4f32 = @splat(0.0);
        var acc1: simd.V4f32 = @splat(0.0);
        var j: usize = 0;
        while (j + 8 <= len) : (j += 8) {
            acc0 += simd.load4_f32(a, off + j) * simd.load4_f32(b, off + j);
            acc1 += simd.load4_f32(a, off + j + 4) * simd.load4_f32(b, off + j + 4);
        }
        while (j + 4 <= len) : (j += 4) {
            acc0 += simd.load4_f32(a, off + j) * simd.load4_f32(b, off + j);
        }
        acc0 += acc1;
        var sum: f32 = acc0[0] + acc0[1] + acc0[2] + acc0[3];
        while (j < len) : (j += 1) {
            sum += a[off + j] * b[off + j];
        }
        out[bi] = sum;
    }
}

// ─── outer: a[m] ⊗ b[n] → out[m×n] ────────────────────────────────────────

export fn outer_f64(a: [*]const f64, b: [*]const f64, out: [*]f64, m: u32, n: u32) void {
    const rows = @as(usize, m);
    const cols = @as(usize, n);
    for (0..rows) |i| {
        const ai: simd.V2f64 = @splat(a[i]);
        const row_out = out + i * cols;
        var j: usize = 0;
        while (j + 4 <= cols) : (j += 4) {
            simd.store2_f64(row_out, j, ai * simd.load2_f64(b, j));
            simd.store2_f64(row_out, j + 2, ai * simd.load2_f64(b, j + 2));
        }
        while (j + 2 <= cols) : (j += 2) {
            simd.store2_f64(row_out, j, ai * simd.load2_f64(b, j));
        }
        while (j < cols) : (j += 1) {
            row_out[j] = a[i] * b[j];
        }
    }
}

export fn outer_f32(a: [*]const f32, b: [*]const f32, out: [*]f32, m: u32, n: u32) void {
    const rows = @as(usize, m);
    const cols = @as(usize, n);
    for (0..rows) |i| {
        const ai: simd.V4f32 = @splat(a[i]);
        const row_out = out + i * cols;
        var j: usize = 0;
        while (j + 8 <= cols) : (j += 8) {
            simd.store4_f32(row_out, j, ai * simd.load4_f32(b, j));
            simd.store4_f32(row_out, j + 4, ai * simd.load4_f32(b, j + 4));
        }
        while (j + 4 <= cols) : (j += 4) {
            simd.store4_f32(row_out, j, ai * simd.load4_f32(b, j));
        }
        while (j < cols) : (j += 1) {
            row_out[j] = a[i] * b[j];
        }
    }
}

// ─── kron: Kronecker product A[am×an] ⊗ B[bm×bn] → out[(am*bm)×(an*bn)] ──

export fn kron_f64(
    a: [*]const f64,
    b: [*]const f64,
    out: [*]f64,
    am: u32,
    an: u32,
    bm: u32,
    bn: u32,
) void {
    const a_rows = @as(usize, am);
    const a_cols = @as(usize, an);
    const b_rows = @as(usize, bm);
    const b_cols = @as(usize, bn);
    const out_cols = a_cols * b_cols;

    for (0..a_rows) |ia| {
        for (0..a_cols) |ja| {
            const aij: simd.V2f64 = @splat(a[ia * a_cols + ja]);
            for (0..b_rows) |ib| {
                const out_row = out + (ia * b_rows + ib) * out_cols + ja * b_cols;
                const b_row = b + ib * b_cols;
                var jb: usize = 0;
                while (jb + 2 <= b_cols) : (jb += 2) {
                    simd.store2_f64(out_row, jb, aij * simd.load2_f64(b_row, jb));
                }
                while (jb < b_cols) : (jb += 1) {
                    out_row[jb] = a[ia * a_cols + ja] * b_row[jb];
                }
            }
        }
    }
}

export fn kron_f32(
    a: [*]const f32,
    b: [*]const f32,
    out: [*]f32,
    am: u32,
    an: u32,
    bm: u32,
    bn: u32,
) void {
    const a_rows = @as(usize, am);
    const a_cols = @as(usize, an);
    const b_rows = @as(usize, bm);
    const b_cols = @as(usize, bn);
    const out_cols = a_cols * b_cols;

    for (0..a_rows) |ia| {
        for (0..a_cols) |ja| {
            const aij: simd.V4f32 = @splat(a[ia * a_cols + ja]);
            for (0..b_rows) |ib| {
                const out_row = out + (ia * b_rows + ib) * out_cols + ja * b_cols;
                const b_row = b + ib * b_cols;
                var jb: usize = 0;
                while (jb + 4 <= b_cols) : (jb += 4) {
                    simd.store4_f32(out_row, jb, aij * simd.load4_f32(b_row, jb));
                }
                while (jb < b_cols) : (jb += 1) {
                    out_row[jb] = a[ia * a_cols + ja] * b_row[jb];
                }
            }
        }
    }
}

// ─── cross: cross product of n pairs of 3-vectors ──────────────────────────

export fn cross_f64(a: [*]const f64, b: [*]const f64, out: [*]f64, n: u32) void {
    const count = @as(usize, n);
    for (0..count) |i| {
        const ao = i * 3;
        const bo = i * 3;
        const oo = i * 3;
        out[oo + 0] = a[ao + 1] * b[bo + 2] - a[ao + 2] * b[bo + 1];
        out[oo + 1] = a[ao + 2] * b[bo + 0] - a[ao + 0] * b[bo + 2];
        out[oo + 2] = a[ao + 0] * b[bo + 1] - a[ao + 1] * b[bo + 0];
    }
}

export fn cross_f32(a: [*]const f32, b: [*]const f32, out: [*]f32, n: u32) void {
    const count = @as(usize, n);
    for (0..count) |i| {
        const ao = i * 3;
        const bo = i * 3;
        const oo = i * 3;
        out[oo + 0] = a[ao + 1] * b[bo + 2] - a[ao + 2] * b[bo + 1];
        out[oo + 1] = a[ao + 2] * b[bo + 0] - a[ao + 0] * b[bo + 2];
        out[oo + 2] = a[ao + 0] * b[bo + 1] - a[ao + 1] * b[bo + 0];
    }
}

// ─── norm: L2 norm = sqrt(sum(x²)) ─────────────────────────────────────────

export fn norm_f64(ptr: [*]const f64, n: u32) f64 {
    const len = @as(usize, n);
    var acc0: simd.V2f64 = @splat(0.0);
    var acc1: simd.V2f64 = @splat(0.0);
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        const v0 = simd.load2_f64(ptr, i);
        const v1 = simd.load2_f64(ptr, i + 2);
        acc0 += v0 * v0;
        acc1 += v1 * v1;
    }
    while (i + 2 <= len) : (i += 2) {
        const v = simd.load2_f64(ptr, i);
        acc0 += v * v;
    }
    acc0 += acc1;
    var sum: f64 = acc0[0] + acc0[1];
    while (i < len) : (i += 1) {
        sum += ptr[i] * ptr[i];
    }
    return @sqrt(sum);
}

export fn norm_f32(ptr: [*]const f32, n: u32) f32 {
    const len = @as(usize, n);
    var acc0: simd.V4f32 = @splat(0.0);
    var acc1: simd.V4f32 = @splat(0.0);
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        const v0 = simd.load4_f32(ptr, i);
        const v1 = simd.load4_f32(ptr, i + 4);
        acc0 += v0 * v0;
        acc1 += v1 * v1;
    }
    while (i + 4 <= len) : (i += 4) {
        const v = simd.load4_f32(ptr, i);
        acc0 += v * v;
    }
    acc0 += acc1;
    var sum: f32 = acc0[0] + acc0[1] + acc0[2] + acc0[3];
    while (i < len) : (i += 1) {
        sum += ptr[i] * ptr[i];
    }
    return @sqrt(sum);
}

// ─── Internal tiled matmul (copied from matmul.zig for separate WASM module) ─

const TILE = 48;

fn matmulF64(a: [*]const f64, b: [*]const f64, c: [*]f64, M: usize, N: usize, K: usize) void {
    for (0..M * N) |i| c[i] = 0;
    var ii: usize = 0;
    while (ii < M) : (ii += TILE) {
        const ie = if (ii + TILE < M) ii + TILE else M;
        var kk: usize = 0;
        while (kk < K) : (kk += TILE) {
            const ke = if (kk + TILE < K) kk + TILE else K;
            var jj: usize = 0;
            while (jj < N) : (jj += TILE) {
                const je = if (jj + TILE < N) jj + TILE else N;
                var ri: usize = ii;
                while (ri < ie) : (ri += 1) {
                    var rk: usize = kk;
                    while (rk < ke) : (rk += 1) {
                        const aik = a[ri * K + rk];
                        const av: simd.V2f64 = @splat(aik);
                        const br = rk * N;
                        const cr = ri * N;
                        var j: usize = jj;
                        while (j + 4 <= je) : (j += 4) {
                            simd.store2_f64(c, cr + j, simd.load2_f64(c, cr + j) + av * simd.load2_f64(b, br + j));
                            simd.store2_f64(c, cr + j + 2, simd.load2_f64(c, cr + j + 2) + av * simd.load2_f64(b, br + j + 2));
                        }
                        while (j + 2 <= je) : (j += 2) {
                            simd.store2_f64(c, cr + j, simd.load2_f64(c, cr + j) + av * simd.load2_f64(b, br + j));
                        }
                        while (j < je) : (j += 1) {
                            c[cr + j] += aik * b[br + j];
                        }
                    }
                }
            }
        }
    }
}

fn copyF64(dst: [*]f64, src: [*]const f64, len: usize) void {
    for (0..len) |i| dst[i] = src[i];
}

// ─── matrix_power: out = a^power via binary exponentiation ──────────────────

export fn matrix_power_f64(a: [*]const f64, out: [*]f64, scratch: [*]f64, n: u32, power: u32) void {
    const N = @as(usize, n);
    const nn = N * N;
    const cur = scratch;
    const tmp = scratch + nn;
    var p = @as(usize, power);

    // out = I
    for (0..nn) |idx| out[idx] = 0;
    for (0..N) |di| out[di * N + di] = 1;

    // cur = a
    copyF64(cur, a, nn);

    while (p > 0) {
        if (p & 1 != 0) {
            matmulF64(out, cur, tmp, N, N, N);
            copyF64(out, tmp, nn);
        }
        p >>= 1;
        if (p > 0) {
            matmulF64(cur, cur, tmp, N, N, N);
            copyF64(cur, tmp, nn);
        }
    }
}

// ─── multi_dot3: out = a @ b @ c (3 square matrices) ────────────────────────

export fn multi_dot3_f64(a: [*]const f64, b: [*]const f64, c: [*]const f64, out: [*]f64, tmp: [*]f64, n: u32) void {
    const N = @as(usize, n);
    matmulF64(a, b, tmp, N, N, N);
    matmulF64(tmp, c, out, N, N, N);
}

// ─── qr: Householder QR decomposition ──────────────────────────────────────

export fn qr_f64(a: [*]f64, q: [*]f64, r: [*]f64, tau_out: [*]f64, scratch: [*]f64, m_arg: u32, n_arg: u32) void {
    const M = @as(usize, m_arg);
    const N = @as(usize, n_arg);
    const K = if (M < N) M else N;
    _ = scratch;

    // Householder reflections — modify a in place
    for (0..K) |j| {
        // Compute norm of a[j:,j]
        var norm_sq: f64 = 0;
        for (j..M) |ri| {
            const v = a[ri * N + j];
            norm_sq += v * v;
        }
        var nrm = @sqrt(norm_sq);
        if (nrm == 0) {
            tau_out[j] = 0;
            continue;
        }

        // alpha = -sign(a[j,j]) * norm
        const ajj = a[j * N + j];
        if (ajj >= 0) nrm = -nrm;
        const alpha = nrm;

        // Form Householder vector v in a[j:,j]
        a[j * N + j] -= alpha;
        const v0 = a[j * N + j];

        // tau = 2 / (v^T v)
        var vtv: f64 = v0 * v0;
        for (j + 1..M) |ri| {
            const vi = a[ri * N + j];
            vtv += vi * vi;
        }
        if (vtv == 0) {
            tau_out[j] = 0;
            a[j * N + j] = alpha;
            continue;
        }
        tau_out[j] = 2.0 / vtv;

        // Apply reflection to trailing columns
        for (j + 1..N) |col| {
            var dot: f64 = 0;
            for (j..M) |ri| {
                dot += a[ri * N + j] * a[ri * N + col];
            }
            const factor = tau_out[j] * dot;
            for (j..M) |ri| {
                a[ri * N + col] -= factor * a[ri * N + j];
            }
        }

        // Store alpha on diagonal
        a[j * N + j] = alpha;
    }

    // Extract R: upper triangle of a
    for (0..K) |ri| {
        for (0..N) |ci| {
            r[ri * N + ci] = if (ci >= ri) a[ri * N + ci] else 0;
        }
    }

    // Reconstruct Q: start with I[m×K], apply H_{K-1} ... H_0
    for (0..M * K) |idx| q[idx] = 0;
    for (0..K) |di| q[di * K + di] = 1;

    // Apply reflectors in reverse order
    var jrev: usize = K;
    while (jrev > 0) {
        jrev -= 1;
        const j = jrev;
        if (tau_out[j] == 0) continue;

        // Recover |v[0]| from tau and sub-diagonal elements
        var sub_sq: f64 = 0;
        for (j + 1..M) |ri| {
            const vi = a[ri * N + j];
            sub_sq += vi * vi;
        }
        const vtv2 = 2.0 / tau_out[j];
        const v0sq = vtv2 - sub_sq;
        const v0 = if (v0sq > 0) @sqrt(v0sq) else 0.0;

        // Apply H_j to Q: Q -= tau * v * (v^T * Q)
        for (0..K) |col| {
            var dot: f64 = v0 * q[j * K + col];
            for (j + 1..M) |ri| {
                dot += a[ri * N + j] * q[ri * K + col];
            }
            const factor = tau_out[j] * dot;
            q[j * K + col] -= factor * v0;
            for (j + 1..M) |ri| {
                q[ri * K + col] -= factor * a[ri * N + j];
            }
        }
    }
}

// ─── lstsq: solve Ax=b via QR (overdetermined, m >= n) ─────────────────────

export fn lstsq_f64(a: [*]f64, b: [*]const f64, x: [*]f64, scratch: [*]f64, m_arg: u32, n_arg: u32) void {
    const M = @as(usize, m_arg);
    const N = @as(usize, n_arg);
    const K = if (M < N) M else N;

    // Partition scratch: a_copy[M*N] + Q[M*K] + R[K*N] + tau[K] + QtB[K]
    const a_copy = scratch;
    const q_ptr = a_copy + M * N;
    const r_ptr = q_ptr + M * K;
    const tau_ptr = r_ptr + K * N;
    const qtb_ptr = tau_ptr + K;
    const qr_scratch = qtb_ptr + K;

    // Copy a since qr modifies in place
    copyF64(a_copy, a, M * N);

    // QR decomposition
    qr_f64(a_copy, q_ptr, r_ptr, tau_ptr, qr_scratch, m_arg, n_arg);

    // QtB = Q^T * b
    for (0..K) |ci| {
        var sum: f64 = 0;
        for (0..M) |ri| {
            sum += q_ptr[ri * K + ci] * b[ri];
        }
        qtb_ptr[ci] = sum;
    }

    // Back-substitution: R * x = QtB
    var ii: usize = K;
    while (ii > 0) {
        ii -= 1;
        var sum: f64 = qtb_ptr[ii];
        for (ii + 1..N) |j| {
            sum -= r_ptr[ii * N + j] * x[j];
        }
        const diag = r_ptr[ii * N + ii];
        x[ii] = if (diag != 0) sum / diag else 0;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPLEX MATMUL (c128, c64)
// ═══════════════════════════════════════════════════════════════════════════

// matmul_c128: C[M×N] = A[M×K] × B[K×N], complex f64
// Each element = 2 f64s (re,im). Pointers are to f64 arrays of 2*M*K etc.
export fn matmul_c128(a: [*]const f64, b: [*]const f64, c: [*]f64, M: u32, K: u32, N: u32) void {
    const m = @as(usize, M);
    const k = @as(usize, K);
    const nn = @as(usize, N);

    // Zero output: 2*m*nn f64s
    for (0..m * nn * 2) |i| c[i] = 0;

    // Tiled i-k-j loop with complex multiply-accumulate
    const T = 32;
    var ii: usize = 0;
    while (ii < m) : (ii += T) {
        const ie = if (ii + T < m) ii + T else m;
        var kk: usize = 0;
        while (kk < k) : (kk += T) {
            const ke = if (kk + T < k) kk + T else k;
            var jj: usize = 0;
            while (jj < nn) : (jj += T) {
                const je = if (jj + T < nn) jj + T else nn;
                var ri: usize = ii;
                while (ri < ie) : (ri += 1) {
                    var rk: usize = kk;
                    while (rk < ke) : (rk += 1) {
                        const a_re = a[(ri * k + rk) * 2];
                        const a_im = a[(ri * k + rk) * 2 + 1];
                        var j: usize = jj;
                        while (j < je) : (j += 1) {
                            const b_re = b[(rk * nn + j) * 2];
                            const b_im = b[(rk * nn + j) * 2 + 1];
                            const ci = (ri * nn + j) * 2;
                            c[ci] += a_re * b_re - a_im * b_im;
                            c[ci + 1] += a_re * b_im + a_im * b_re;
                        }
                    }
                }
            }
        }
    }
}

// matmul_c64: C[M×N] = A[M×K] × B[K×N], complex f32
// SIMD: process 2 complex output elements at a time using f32x4
export fn matmul_c64(a: [*]const f32, b: [*]const f32, c: [*]f32, M: u32, K: u32, N: u32) void {
    const m = @as(usize, M);
    const k = @as(usize, K);
    const nn = @as(usize, N);

    for (0..m * nn * 2) |i| c[i] = 0;

    const T = 32;
    var ii: usize = 0;
    while (ii < m) : (ii += T) {
        const ie = if (ii + T < m) ii + T else m;
        var kk: usize = 0;
        while (kk < k) : (kk += T) {
            const ke = if (kk + T < k) kk + T else k;
            var jj: usize = 0;
            while (jj < nn) : (jj += T) {
                const je = if (jj + T < nn) jj + T else nn;
                var ri: usize = ii;
                while (ri < ie) : (ri += 1) {
                    const c_row = c + ri * nn * 2;
                    var rk: usize = kk;
                    while (rk < ke) : (rk += 1) {
                        const a_base = (ri * k + rk) * 2;
                        const a_re = a[a_base];
                        const a_im = a[a_base + 1];
                        const b_row = b + rk * nn * 2;
                        // Splat a_re and a_im across f32x4 for SIMD complex mul
                        const are_v: simd.V4f32 = @splat(a_re);
                        const aim_v: simd.V4f32 = @splat(a_im);
                        const sign: simd.V4f32 = .{ -1.0, 1.0, -1.0, 1.0 };
                        // Process 2 complex outputs at a time (4 f32s)
                        var j: usize = jj;
                        while (j + 2 <= je) : (j += 2) {
                            const bv = simd.load4_f32(b_row, j * 2); // [br0,bi0,br1,bi1]
                            const cv = simd.load4_f32(c_row, j * 2);
                            // a_swap: [ai, ar, ai, ar] pattern for cross-multiply
                            const b_swap: simd.V4f32 = .{ bv[1], bv[0], bv[3], bv[2] };
                            // are * [br0,bi0,br1,bi1] = [are*br0, are*bi0, are*br1, are*bi1]
                            // aim * [bi0,br0,bi1,br1] * [-1,1,-1,1] = [-aim*bi0, aim*br0, -aim*bi1, aim*br1]
                            simd.store4_f32(c_row, j * 2, cv + are_v * bv + aim_v * b_swap * sign);
                        }
                        // Scalar remainder
                        while (j < je) : (j += 1) {
                            const bj = j * 2;
                            const b_re = b_row[bj];
                            const b_im = b_row[bj + 1];
                            const cj = j * 2;
                            c_row[cj] += a_re * b_re - a_im * b_im;
                            c_row[cj + 1] += a_re * b_im + a_im * b_re;
                        }
                    }
                }
            }
        }
    }
}
