//! WASM Householder QR decomposition and least-squares solve.
//!
//! qr_f64: A[m×n] → Q[m×k], R[k×n] where k = min(m,n)
//! lstsq_f64: solve Ax=b via QR for overdetermined systems (m >= n)

/// Householder QR decomposition for f64 matrices.
/// `a` is modified in place (stores R on upper triangle, Householder vectors below).
/// `q` receives Q[m×k], `r` receives R[k×n], `tau_out` receives Householder scalars.
/// `scratch` is unused but reserved for future use.
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
        const v0_val = if (v0sq > 0) @sqrt(v0sq) else 0.0;

        // Apply H_j to Q: Q -= tau * v * (v^T * Q)
        for (0..K) |col| {
            var dot: f64 = v0_val * q[j * K + col];
            for (j + 1..M) |ri| {
                dot += a[ri * N + j] * q[ri * K + col];
            }
            const factor = tau_out[j] * dot;
            q[j * K + col] -= factor * v0_val;
            for (j + 1..M) |ri| {
                q[ri * K + col] -= factor * a[ri * N + j];
            }
        }
    }
}

/// Least-squares solve Ax=b via QR decomposition for f64 matrices.
/// a is modified in place. scratch layout:
///   a_copy[M*N] + Q[M*K] + R[K*N] + tau[K] + QtB[K] + qr_scratch[...]
export fn lstsq_f64(a: [*]f64, b: [*]const f64, x: [*]f64, scratch: [*]f64, m_arg: u32, n_arg: u32) void {
    const M = @as(usize, m_arg);
    const N = @as(usize, n_arg);
    const K = if (M < N) M else N;

    // Partition scratch
    const a_copy = scratch;
    const q_ptr = a_copy + M * N;
    const r_ptr = q_ptr + M * K;
    const tau_ptr = r_ptr + K * N;
    const qtb_ptr = tau_ptr + K;
    const qr_scratch = qtb_ptr + K;

    // Copy a since qr modifies in place
    for (0..M * N) |i| a_copy[i] = a[i];

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

// --- Tests ---

test "qr_f64 2x2" {
    const testing = @import("std").testing;
    // A = [[1, 2], [3, 4]]
    var a = [_]f64{ 1, 2, 3, 4 };
    var q: [4]f64 = undefined;
    var r: [4]f64 = undefined;
    var tau: [2]f64 = undefined;
    var scratch: [16]f64 = undefined;
    qr_f64(&a, &q, &r, &tau, &scratch, 2, 2);

    // Verify Q is orthogonal: Q^T Q ≈ I
    const qtq00 = q[0] * q[0] + q[2] * q[2];
    const qtq01 = q[0] * q[1] + q[2] * q[3];
    const qtq11 = q[1] * q[1] + q[3] * q[3];
    try testing.expectApproxEqAbs(qtq00, 1.0, 1e-10);
    try testing.expectApproxEqAbs(qtq01, 0.0, 1e-10);
    try testing.expectApproxEqAbs(qtq11, 1.0, 1e-10);

    // Verify R is upper triangular
    try testing.expectApproxEqAbs(r[2], 0.0, 1e-10); // R[1,0] = 0

    // Verify QR ≈ A (original)
    const qr00 = q[0] * r[0] + q[1] * r[2];
    const qr01 = q[0] * r[1] + q[1] * r[3];
    const qr10 = q[2] * r[0] + q[3] * r[2];
    const qr11 = q[2] * r[1] + q[3] * r[3];
    try testing.expectApproxEqAbs(qr00, 1.0, 1e-10);
    try testing.expectApproxEqAbs(qr01, 2.0, 1e-10);
    try testing.expectApproxEqAbs(qr10, 3.0, 1e-10);
    try testing.expectApproxEqAbs(qr11, 4.0, 1e-10);
}

test "qr_f64 3x2" {
    const testing = @import("std").testing;
    // A = [[1,2],[3,4],[5,6]]
    var a = [_]f64{ 1, 2, 3, 4, 5, 6 };
    var q: [6]f64 = undefined; // 3x2
    var r: [4]f64 = undefined; // 2x2
    var tau: [2]f64 = undefined;
    var scratch: [32]f64 = undefined;
    qr_f64(&a, &q, &r, &tau, &scratch, 3, 2);

    // Verify Q^T Q ≈ I (2x2)
    const qtq00 = q[0] * q[0] + q[2] * q[2] + q[4] * q[4];
    const qtq01 = q[0] * q[1] + q[2] * q[3] + q[4] * q[5];
    const qtq11 = q[1] * q[1] + q[3] * q[3] + q[5] * q[5];
    try testing.expectApproxEqAbs(qtq00, 1.0, 1e-10);
    try testing.expectApproxEqAbs(qtq01, 0.0, 1e-10);
    try testing.expectApproxEqAbs(qtq11, 1.0, 1e-10);
}

test "lstsq_f64 overdetermined 3x2" {
    const testing = @import("std").testing;
    // A = [[1,1],[1,2],[1,3]], b = [1,2,3]
    // Least squares: x = [0, 1] (exact fit for y=x)
    var a = [_]f64{ 1, 1, 1, 2, 1, 3 };
    const b = [_]f64{ 1, 2, 3 };
    var x: [2]f64 = undefined;
    var scratch: [128]f64 = undefined;
    lstsq_f64(&a, &b, &x, &scratch, 3, 2);
    try testing.expectApproxEqAbs(x[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(x[1], 1.0, 1e-10);
}
