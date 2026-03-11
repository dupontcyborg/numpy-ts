//! WASM Singular Value Decomposition for real matrices.
//!
//! svd_f64: A[m×n] → U[m×m], S[k], Vt[n×n] where k = min(m,n)
//!
//! Algorithm: One-sided Jacobi SVD — works directly on columns of A.
//! No A^T·A formation, no condition-number squaring.
//! Inner loops are dot products and axpy updates (SIMD-friendly).
//! Uses cyclic sweeps with trig-free Rutishauser rotations.
//!
//! Memory layout (all row-major):
//!   a[m*n]: input (preserved via copy)
//!   u[m*m]: left singular vectors
//!   s[k]:   singular values (descending)
//!   vt[n*n]: right singular vectors transposed
//!   work: scratch (needs m*n + n*n f64s)

/// Full SVD for f64 matrices: A[m×n] → U[m×m] · diag(S) · Vt[n×n].
/// `a` is the input matrix (read-only; copied to work).
/// `u` receives U[m×m], `s` receives singular values[k], `vt` receives V^T[n×n].
/// `work` needs at least (m*n + n*n) f64s.
/// Singular values are sorted in descending order.
export fn svd_f64(a: [*]const f64, u_out: [*]f64, s: [*]f64, vt: [*]f64, work: [*]f64, m_arg: u32, n_arg: u32) void {
    const M = @as(usize, m_arg);
    const N = @as(usize, n_arg);
    const K = if (M < N) M else N;

    // Partition work buffer
    const w = work; // m*n: working copy of A (columns get orthogonalized)
    const v = w + M * N; // n*n: V matrix (accumulates rotations)

    // Copy input into working matrix
    for (0..M * N) |i| w[i] = a[i];

    // Initialize V = I_n
    for (0..N * N) |i| v[i] = 0;
    for (0..N) |i| v[i * N + i] = 1;

    // --- One-sided Jacobi SVD ---
    // Iteratively apply Jacobi rotations to pairs of columns of W
    // until all columns are mutually orthogonal.
    // When converged: σ_j = ||W[:,j]||, U[:,j] = W[:,j]/σ_j, V accumulated.

    const max_sweeps: usize = 30;
    const tol = 1e-14;

    for (0..max_sweeps) |_| {
        // Track max |cos(angle)| between any pair of columns
        var converged = true;

        // Cyclic sweep over all column pairs (p, q)
        for (0..N) |p| {
            for (p + 1..N) |q| {
                // Compute dot products: α = W[:,p]·W[:,p], β = W[:,q]·W[:,q], γ = W[:,p]·W[:,q]
                var alpha: f64 = 0;
                var beta: f64 = 0;
                var gamma: f64 = 0;
                for (0..M) |i| {
                    const wp = w[i * N + p];
                    const wq = w[i * N + q];
                    alpha += wp * wp;
                    beta += wq * wq;
                    gamma += wp * wq;
                }

                // Skip if columns are already orthogonal
                if (@abs(gamma) < tol * @sqrt(alpha * beta)) continue;
                converged = false;

                // Trig-free Jacobi rotation (Rutishauser formula)
                // Compute (c, s) to zero γ
                const zeta = (beta - alpha) / (2.0 * gamma);
                const abs_zeta = @abs(zeta);
                const t = (if (zeta >= 0) @as(f64, 1.0) else @as(f64, -1.0)) / (abs_zeta + @sqrt(1.0 + zeta * zeta));
                const c = 1.0 / @sqrt(1.0 + t * t);
                const sn = t * c;

                // Apply rotation to columns of W: (W[:,p], W[:,q]) *= G(c,s)
                for (0..M) |i| {
                    const wp = w[i * N + p];
                    const wq = w[i * N + q];
                    w[i * N + p] = c * wp - sn * wq;
                    w[i * N + q] = sn * wp + c * wq;
                }

                // Apply rotation to columns of V: (V[:,p], V[:,q]) *= G(c,s)
                for (0..N) |i| {
                    const vp = v[i * N + p];
                    const vq = v[i * N + q];
                    v[i * N + p] = c * vp - sn * vq;
                    v[i * N + q] = sn * vp + c * vq;
                }
            }
        }

        if (converged) break;
    }

    // --- Extract results ---

    // Singular values = column norms of W
    // Sort indices by singular value descending
    var indices: [256]usize = undefined;
    for (0..N) |i| {
        var norm: f64 = 0;
        for (0..M) |r| {
            const val = w[r * N + i];
            norm += val * val;
        }
        s[i] = @sqrt(norm); // temporarily store all N norms in s (s has room for K)
        indices[i] = i;
    }

    // Selection sort by singular value descending
    for (0..N) |i| {
        var max_idx = i;
        var max_val = s[indices[i]];
        for (i + 1..N) |j| {
            const val = s[indices[j]];
            if (val > max_val) {
                max_val = val;
                max_idx = j;
            }
        }
        if (max_idx != i) {
            const tmp = indices[i];
            indices[i] = indices[max_idx];
            indices[max_idx] = tmp;
        }
    }

    // Build V^T: rows are columns of V, sorted by descending singular value
    for (0..N) |i| {
        for (0..N) |j| {
            vt[i * N + j] = v[j * N + indices[i]]; // V^T[i,j] = V[j, sorted_i]
        }
    }

    // Build U: first K columns from normalized W columns (sorted)
    for (0..M * M) |i| u_out[i] = 0;

    for (0..K) |j| {
        const col = indices[j];
        const sigma = s[col];
        if (sigma > 1e-14) {
            for (0..M) |i| {
                u_out[i * M + j] = w[i * N + col] / sigma;
            }
        }
    }

    // Rewrite s with sorted singular values (only K values)
    // Use w as temp storage since we're done with it
    for (0..K) |i| w[i] = s[indices[i]];
    for (0..K) |i| s[i] = w[i];

    // Complete U for remaining columns (m > n case) via Gram-Schmidt
    if (M > K) {
        for (K..M) |j| {
            // Start with standard basis vector e_j
            for (0..M) |i| u_out[i * M + j] = 0;
            u_out[j * M + j] = 1;

            // Orthogonalize against columns 0..j
            for (0..j) |prev| {
                var dot: f64 = 0;
                for (0..M) |i| dot += u_out[i * M + j] * u_out[i * M + prev];
                for (0..M) |i| u_out[i * M + j] -= dot * u_out[i * M + prev];
            }

            // Normalize
            var norm: f64 = 0;
            for (0..M) |i| {
                const val = u_out[i * M + j];
                norm += val * val;
            }
            norm = @sqrt(norm);
            if (norm > 1e-14) {
                for (0..M) |i| u_out[i * M + j] /= norm;
            }
        }
    }
}

// --- Tests ---

test "svd_f64 2x2 identity" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1, 0, 0, 1 };
    var u: [4]f64 = undefined;
    var s: [2]f64 = undefined;
    var vt: [4]f64 = undefined;
    var work: [256]f64 = undefined;
    svd_f64(&a, &u, &s, &vt, &work, 2, 2);

    try testing.expectApproxEqAbs(s[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(s[1], 1.0, 1e-10);
}

test "svd_f64 2x2 diagonal" {
    const testing = @import("std").testing;
    const a = [_]f64{ 3, 0, 0, 5 };
    var u: [4]f64 = undefined;
    var s: [2]f64 = undefined;
    var vt: [4]f64 = undefined;
    var work: [256]f64 = undefined;
    svd_f64(&a, &u, &s, &vt, &work, 2, 2);

    try testing.expectApproxEqAbs(s[0], 5.0, 1e-10);
    try testing.expectApproxEqAbs(s[1], 3.0, 1e-10);
}

test "svd_f64 2x2 reconstruction" {
    const testing = @import("std").testing;
    const orig = [_]f64{ 1, 2, 3, 4 };
    var u: [4]f64 = undefined;
    var s: [2]f64 = undefined;
    var vt: [4]f64 = undefined;
    var work: [256]f64 = undefined;
    svd_f64(&orig, &u, &s, &vt, &work, 2, 2);

    for (0..2) |i| {
        for (0..2) |j| {
            var val: f64 = 0;
            for (0..2) |k| val += u[i * 2 + k] * s[k] * vt[k * 2 + j];
            try testing.expectApproxEqAbs(val, orig[i * 2 + j], 1e-10);
        }
    }
}

test "svd_f64 3x2 reconstruction" {
    const testing = @import("std").testing;
    const orig = [_]f64{ 1, 2, 3, 4, 5, 6 };
    var u: [9]f64 = undefined;
    var s: [2]f64 = undefined;
    var vt: [4]f64 = undefined;
    var work: [256]f64 = undefined;
    svd_f64(&orig, &u, &s, &vt, &work, 3, 2);

    try testing.expect(s[0] >= s[1]);
    try testing.expect(s[1] >= 0);

    for (0..3) |i| {
        for (0..2) |j| {
            var val: f64 = 0;
            for (0..2) |k| val += u[i * 3 + k] * s[k] * vt[k * 2 + j];
            try testing.expectApproxEqAbs(val, orig[i * 2 + j], 1e-8);
        }
    }
}

test "svd_f64 rank-1 matrix" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1, 0, 0, 0 };
    var u: [4]f64 = undefined;
    var s: [2]f64 = undefined;
    var vt: [4]f64 = undefined;
    var work: [256]f64 = undefined;
    svd_f64(&a, &u, &s, &vt, &work, 2, 2);

    try testing.expectApproxEqAbs(s[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(s[1], 0.0, 1e-10);
}

test "svd_f64 4x4 reconstruction" {
    const testing = @import("std").testing;
    const orig = [_]f64{
        2, -1, 0, 0,
        -1, 2, -1, 0,
        0, -1, 2, -1,
        0, 0, -1, 2,
    };
    var u: [16]f64 = undefined;
    var s: [4]f64 = undefined;
    var vt: [16]f64 = undefined;
    var work: [512]f64 = undefined;
    svd_f64(&orig, &u, &s, &vt, &work, 4, 4);

    for (0..4) |i| {
        for (0..4) |j| {
            var val: f64 = 0;
            for (0..4) |k| val += u[i * 4 + k] * s[k] * vt[k * 4 + j];
            try testing.expectApproxEqAbs(val, orig[i * 4 + j], 1e-8);
        }
    }
}

test "svd_f64 5x3 reconstruction" {
    const testing = @import("std").testing;
    const orig = [_]f64{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
        13, 14, 15,
    };
    var u: [25]f64 = undefined;
    var s: [3]f64 = undefined;
    var vt: [9]f64 = undefined;
    var work: [512]f64 = undefined;
    svd_f64(&orig, &u, &s, &vt, &work, 5, 3);

    // Reconstruction using K=3 singular values
    for (0..5) |i| {
        for (0..3) |j| {
            var val: f64 = 0;
            for (0..3) |k| val += u[i * 5 + k] * s[k] * vt[k * 3 + j];
            try testing.expectApproxEqAbs(val, orig[i * 3 + j], 1e-6);
        }
    }
}
