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
    // scratch[0..K] stores v0 values for Q reconstruction
    const v0_store = scratch;

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
            v0_store[j] = 0;
            continue;
        }

        // alpha = -sign(a[j,j]) * norm
        const ajj = a[j * N + j];
        if (ajj >= 0) nrm = -nrm;
        const alpha = nrm;

        // Form Householder vector v in a[j:,j]
        a[j * N + j] -= alpha;
        const v0 = a[j * N + j];
        v0_store[j] = v0;

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

        // Use stored v0 value (exact, preserves sign)
        const v0_val = v0_store[j];

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

test "qr_f64 1x1" {
    const testing = @import("std").testing;
    var a = [_]f64{5.0};
    var q: [1]f64 = undefined;
    var r: [1]f64 = undefined;
    var tau: [1]f64 = undefined;
    var scratch: [8]f64 = undefined;
    qr_f64(&a, &q, &r, &tau, &scratch, 1, 1);
    // Q should be ±1, R should be ±5, Q*R = 5
    try testing.expectApproxEqAbs(q[0] * r[0], 5.0, 1e-10);
    try testing.expectApproxEqAbs(@abs(q[0]), 1.0, 1e-10);
}

test "qr_f64 identity 3x3" {
    const testing = @import("std").testing;
    var a = [_]f64{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    var q: [9]f64 = undefined;
    var r: [9]f64 = undefined;
    var tau: [3]f64 = undefined;
    var scratch: [64]f64 = undefined;
    qr_f64(&a, &q, &r, &tau, &scratch, 3, 3);

    // Q^T Q ≈ I
    for (0..3) |i| {
        for (0..3) |j| {
            var dot: f64 = 0;
            for (0..3) |k| dot += q[k * 3 + i] * q[k * 3 + j];
            const expected: f64 = if (i == j) 1.0 else 0.0;
            try testing.expectApproxEqAbs(dot, expected, 1e-10);
        }
    }

    // R should be ±I (diagonal entries ±1)
    for (0..3) |i| {
        try testing.expectApproxEqAbs(@abs(r[i * 3 + i]), 1.0, 1e-10);
    }
}

test "qr_f64 4x3 tall" {
    const testing = @import("std").testing;
    // A = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
    var a = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const orig = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var q: [12]f64 = undefined; // 4x3
    var r: [9]f64 = undefined; // 3x3
    var tau: [3]f64 = undefined;
    var scratch: [128]f64 = undefined;
    qr_f64(&a, &q, &r, &tau, &scratch, 4, 3);

    // Verify Q^T Q ≈ I (3x3)
    for (0..3) |i| {
        for (0..3) |j| {
            var dot: f64 = 0;
            for (0..4) |k| dot += q[k * 3 + i] * q[k * 3 + j];
            const expected: f64 = if (i == j) 1.0 else 0.0;
            try testing.expectApproxEqAbs(dot, expected, 1e-10);
        }
    }

    // Verify R is upper triangular
    try testing.expectApproxEqAbs(r[3], 0.0, 1e-10); // R[1,0]
    try testing.expectApproxEqAbs(r[6], 0.0, 1e-10); // R[2,0]
    try testing.expectApproxEqAbs(r[7], 0.0, 1e-10); // R[2,1]

    // Verify QR ≈ A
    for (0..4) |i| {
        for (0..3) |j| {
            var val: f64 = 0;
            for (0..3) |k| val += q[i * 3 + k] * r[k * 3 + j];
            try testing.expectApproxEqAbs(val, orig[i * 3 + j], 1e-10);
        }
    }
}

test "qr_f64 2x3 wide" {
    const testing = @import("std").testing;
    // Wide matrix: K = min(2,3) = 2
    var a = [_]f64{ 1, 2, 3, 4, 5, 6 };
    const orig = [_]f64{ 1, 2, 3, 4, 5, 6 };
    var q: [4]f64 = undefined; // 2x2
    var r: [6]f64 = undefined; // 2x3
    var tau: [2]f64 = undefined;
    var scratch: [64]f64 = undefined;
    qr_f64(&a, &q, &r, &tau, &scratch, 2, 3);

    // Q^T Q ≈ I (2x2)
    for (0..2) |i| {
        for (0..2) |j| {
            var dot: f64 = 0;
            for (0..2) |k| dot += q[k * 2 + i] * q[k * 2 + j];
            const expected: f64 = if (i == j) 1.0 else 0.0;
            try testing.expectApproxEqAbs(dot, expected, 1e-10);
        }
    }

    // QR ≈ A
    for (0..2) |i| {
        for (0..3) |j| {
            var val: f64 = 0;
            for (0..2) |k| val += q[i * 2 + k] * r[k * 3 + j];
            try testing.expectApproxEqAbs(val, orig[i * 3 + j], 1e-10);
        }
    }
}

test "lstsq_f64 exact 2x2" {
    const testing = @import("std").testing;
    // A = [[1,0],[0,1]], b = [3,7] → x = [3,7]
    var a = [_]f64{ 1, 0, 0, 1 };
    const b = [_]f64{ 3, 7 };
    var x: [2]f64 = undefined;
    var scratch: [128]f64 = undefined;
    lstsq_f64(&a, &b, &x, &scratch, 2, 2);
    try testing.expectApproxEqAbs(x[0], 3.0, 1e-10);
    try testing.expectApproxEqAbs(x[1], 7.0, 1e-10);
}

test "lstsq_f64 overdetermined 4x2" {
    const testing = @import("std").testing;
    // Fit y = 2x + 1: A = [[1,0],[1,1],[1,2],[1,3]], b = [1,3,5,7]
    var a = [_]f64{ 1, 0, 1, 1, 1, 2, 1, 3 };
    const b = [_]f64{ 1, 3, 5, 7 };
    var x: [2]f64 = undefined;
    var scratch: [256]f64 = undefined;
    lstsq_f64(&a, &b, &x, &scratch, 4, 2);
    try testing.expectApproxEqAbs(x[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(x[1], 2.0, 1e-10);
}

test "lstsq_f64 3x3 exact" {
    const testing = @import("std").testing;
    // A = [[2,1,0],[1,3,1],[0,1,2]], b = [5,10,7] → solve exactly
    var a = [_]f64{ 2, 1, 0, 1, 3, 1, 0, 1, 2 };
    const b = [_]f64{ 5, 10, 7 };
    var x: [3]f64 = undefined;
    var scratch: [256]f64 = undefined;
    lstsq_f64(&a, &b, &x, &scratch, 3, 3);
    // Verify Ax ≈ b
    const orig_a = [_]f64{ 2, 1, 0, 1, 3, 1, 0, 1, 2 };
    for (0..3) |i| {
        var val: f64 = 0;
        for (0..3) |j| val += orig_a[i * 3 + j] * x[j];
        try testing.expectApproxEqAbs(val, b[i], 1e-8);
    }
}
