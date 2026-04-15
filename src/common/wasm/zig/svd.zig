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

/// Full SVD for f32 matrices: A[m×n] → U[m×m] · diag(S) · Vt[n×n].
/// Same one-sided Jacobi algorithm as svd_f64 but in single precision throughout.
/// `a` is the input matrix (read-only; copied to work).
/// `u` receives U[m×m], `s` receives singular values[k], `vt` receives V^T[n×n].
/// `work` needs at least (m*n + n*n) f32s.
/// Singular values are sorted in descending order.
export fn svd_f32(a: [*]const f32, u_out: [*]f32, s: [*]f32, vt: [*]f32, work: [*]f32, m_arg: u32, n_arg: u32) void {
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
    const tol: f32 = 1e-6;

    for (0..max_sweeps) |_| {
        // Track max |cos(angle)| between any pair of columns
        var converged = true;

        // Cyclic sweep over all column pairs (p, q)
        for (0..N) |p| {
            for (p + 1..N) |q| {
                // Compute dot products: α = W[:,p]·W[:,p], β = W[:,q]·W[:,q], γ = W[:,p]·W[:,q]
                var alpha: f32 = 0;
                var beta: f32 = 0;
                var gamma: f32 = 0;
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
                const t = (if (zeta >= 0) @as(f32, 1.0) else @as(f32, -1.0)) / (abs_zeta + @sqrt(1.0 + zeta * zeta));
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
        var norm: f32 = 0;
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
        if (sigma > 1e-6) {
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
                var dot: f32 = 0;
                for (0..M) |i| dot += u_out[i * M + j] * u_out[i * M + prev];
                for (0..M) |i| u_out[i * M + j] -= dot * u_out[i * M + prev];
            }

            // Normalize
            var norm: f32 = 0;
            for (0..M) |i| {
                const val = u_out[i * M + j];
                norm += val * val;
            }
            norm = @sqrt(norm);
            if (norm > 1e-6) {
                for (0..M) |i| u_out[i * M + j] /= norm;
            }
        }
    }
}

// --- Tests ---

test "svd_f32 2x2 reconstruction" {
    const testing = @import("std").testing;
    const orig = [_]f32{ 1, 2, 3, 4 };
    var u: [4]f32 = undefined;
    var s: [2]f32 = undefined;
    var vt: [4]f32 = undefined;
    var work: [256]f32 = undefined;
    svd_f32(&orig, &u, &s, &vt, &work, 2, 2);

    for (0..2) |i| {
        for (0..2) |j| {
            var val: f32 = 0;
            for (0..2) |k| val += u[i * 2 + k] * s[k] * vt[k * 2 + j];
            try testing.expectApproxEqAbs(val, orig[i * 2 + j], 1e-4);
        }
    }
}

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
        2,  -1, 0,  0,
        -1, 2,  -1, 0,
        0,  -1, 2,  -1,
        0,  0,  -1, 2,
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
        1,  2,  3,
        4,  5,  6,
        7,  8,  9,
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

test "svd_f64 1x1" {
    const testing = @import("std").testing;
    const a = [_]f64{7.0};
    var u: [1]f64 = undefined;
    var s: [1]f64 = undefined;
    var vt: [1]f64 = undefined;
    var work: [64]f64 = undefined;
    svd_f64(&a, &u, &s, &vt, &work, 1, 1);
    try testing.expectApproxEqAbs(s[0], 7.0, 1e-10);
    try testing.expectApproxEqAbs(@abs(u[0]), 1.0, 1e-10);
    try testing.expectApproxEqAbs(@abs(vt[0]), 1.0, 1e-10);
}

test "svd_f64 2x2 U orthogonality" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1, 2, 3, 4 };
    var u: [4]f64 = undefined;
    var s: [2]f64 = undefined;
    var vt: [4]f64 = undefined;
    var work: [256]f64 = undefined;
    svd_f64(&a, &u, &s, &vt, &work, 2, 2);

    // U^T U ≈ I
    for (0..2) |i| {
        for (0..2) |j| {
            var dot: f64 = 0;
            for (0..2) |k| dot += u[k * 2 + i] * u[k * 2 + j];
            const expected: f64 = if (i == j) 1.0 else 0.0;
            try testing.expectApproxEqAbs(dot, expected, 1e-10);
        }
    }
}

test "svd_f64 2x2 Vt orthogonality" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1, 2, 3, 4 };
    var u: [4]f64 = undefined;
    var s: [2]f64 = undefined;
    var vt: [4]f64 = undefined;
    var work: [256]f64 = undefined;
    svd_f64(&a, &u, &s, &vt, &work, 2, 2);

    // Vt Vt^T ≈ I
    for (0..2) |i| {
        for (0..2) |j| {
            var dot: f64 = 0;
            for (0..2) |k| dot += vt[i * 2 + k] * vt[j * 2 + k];
            const expected: f64 = if (i == j) 1.0 else 0.0;
            try testing.expectApproxEqAbs(dot, expected, 1e-10);
        }
    }
}

test "svd_f64 singular values descending" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 10 };
    var u: [9]f64 = undefined;
    var s: [3]f64 = undefined;
    var vt: [9]f64 = undefined;
    var work: [512]f64 = undefined;
    svd_f64(&a, &u, &s, &vt, &work, 3, 3);
    // Singular values must be in descending order
    try testing.expect(s[0] >= s[1]);
    try testing.expect(s[1] >= s[2]);
    try testing.expect(s[2] >= 0);
}

test "svd_f64 3x3 reconstruction" {
    const testing = @import("std").testing;
    const orig = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 10 };
    var u: [9]f64 = undefined;
    var s: [3]f64 = undefined;
    var vt: [9]f64 = undefined;
    var work: [512]f64 = undefined;
    svd_f64(&orig, &u, &s, &vt, &work, 3, 3);

    for (0..3) |i| {
        for (0..3) |j| {
            var val: f64 = 0;
            for (0..3) |k| val += u[i * 3 + k] * s[k] * vt[k * 3 + j];
            try testing.expectApproxEqAbs(val, orig[i * 3 + j], 1e-8);
        }
    }
}

test "svd_f64 2x3 wide reconstruction" {
    const testing = @import("std").testing;
    const orig = [_]f64{ 1, 2, 3, 4, 5, 6 };
    var u: [4]f64 = undefined;
    var s: [2]f64 = undefined;
    var vt: [9]f64 = undefined;
    var work: [256]f64 = undefined;
    svd_f64(&orig, &u, &s, &vt, &work, 2, 3);

    try testing.expect(s[0] >= s[1]);
    try testing.expect(s[1] >= 0);

    // U·diag(S)·Vt ≈ A
    for (0..2) |i| {
        for (0..3) |j| {
            var val: f64 = 0;
            for (0..2) |k| val += u[i * 2 + k] * s[k] * vt[k * 3 + j];
            try testing.expectApproxEqAbs(val, orig[i * 3 + j], 1e-8);
        }
    }
}

test "svd_f64 symmetric matrix" {
    const testing = @import("std").testing;
    // Symmetric: A = [[2,1],[1,2]], eigenvalues 3 and 1
    const a = [_]f64{ 2, 1, 1, 2 };
    var u: [4]f64 = undefined;
    var s: [2]f64 = undefined;
    var vt: [4]f64 = undefined;
    var work: [256]f64 = undefined;
    svd_f64(&a, &u, &s, &vt, &work, 2, 2);
    // Singular values should be 3 and 1
    try testing.expectApproxEqAbs(s[0], 3.0, 1e-10);
    try testing.expectApproxEqAbs(s[1], 1.0, 1e-10);
}

// ============================================================================
// Golub-Kahan SVD: Householder bidiagonalization + implicit QR iteration
// Much faster than Jacobi for singular values only.
// ============================================================================

/// Compute Givens rotation: [cs sn; -sn cs]^T [f; g] = [r; 0]
inline fn gk_givens(f: f64, g: f64, cs: *f64, sn: *f64, r: *f64) void {
    if (g == 0) {
        cs.* = 1;
        sn.* = 0;
        r.* = f;
    } else if (f == 0) {
        cs.* = 0;
        sn.* = if (g >= 0) 1 else -1;
        r.* = @abs(g);
    } else if (@abs(f) > @abs(g)) {
        const t = g / f;
        const tt = @sqrt(1 + t * t);
        cs.* = 1.0 / tt;
        sn.* = t * cs.*;
        r.* = f * tt;
    } else {
        const t = f / g;
        const tt = @sqrt(1 + t * t);
        sn.* = 1.0 / tt;
        cs.* = t * sn.*;
        r.* = g * tt;
    }
}

/// Householder bidiagonalization: A[m×n] → diag + superdiag (in-place on a).
fn gk_bidiagonalize(a: [*]f64, diag: [*]f64, superdiag: [*]f64, M: usize, N: usize) void {
    const K = if (M < N) M else N;

    for (0..K) |j| {
        // Left Householder: zero a[j+1:, j]
        var norm_sq: f64 = 0;
        for (j..M) |i| {
            const v = a[i * N + j];
            norm_sq += v * v;
        }
        var alpha = @sqrt(norm_sq);
        if (alpha == 0) {
            diag[j] = 0;
        } else {
            if (a[j * N + j] >= 0) alpha = -alpha;
            diag[j] = alpha;
            a[j * N + j] -= alpha;
            var vtv: f64 = 0;
            for (j..M) |i| {
                const vi = a[i * N + j];
                vtv += vi * vi;
            }
            if (vtv != 0) {
                const tau = 2.0 / vtv;
                for (j + 1..N) |col| {
                    var dot: f64 = 0;
                    for (j..M) |i| dot += a[i * N + j] * a[i * N + col];
                    const factor = tau * dot;
                    for (j..M) |i| a[i * N + col] -= factor * a[i * N + j];
                }
            }
        }

        // Right Householder: zero a[j, j+2:]
        if (j + 1 < N) {
            var rnorm_sq: f64 = 0;
            for (j + 1..N) |col| {
                const v = a[j * N + col];
                rnorm_sq += v * v;
            }
            var ralpha = @sqrt(rnorm_sq);
            if (ralpha == 0) {
                if (j < K - 1) superdiag[j] = 0;
            } else {
                if (a[j * N + j + 1] >= 0) ralpha = -ralpha;
                if (j < K - 1) superdiag[j] = ralpha;
                a[j * N + j + 1] -= ralpha;
                var rvtv: f64 = 0;
                for (j + 1..N) |col| {
                    const vc = a[j * N + col];
                    rvtv += vc * vc;
                }
                if (rvtv != 0) {
                    const rtau = 2.0 / rvtv;
                    for (j + 1..M) |row| {
                        var dot: f64 = 0;
                        for (j + 1..N) |col| dot += a[j * N + col] * a[row * N + col];
                        const factor = rtau * dot;
                        for (j + 1..N) |col| a[row * N + col] -= factor * a[j * N + col];
                    }
                }
            }
        }
    }
}

/// Implicit QR step with Wilkinson shift on bidiagonal d[p..end), e[p..end-1).
fn gk_qr_step(d: [*]f64, e: [*]f64, p: usize, end: usize) void {
    const n = end - 1;
    const dn = d[n];
    const dn1 = d[n - 1];
    const en1 = e[n - 1];
    const t11 = dn1 * dn1 + (if (n >= 2) e[n - 2] * e[n - 2] else 0);
    const t22 = dn * dn + en1 * en1;
    const t12 = dn1 * en1;
    const dd = (t11 - t22) * 0.5;
    const sign_dd: f64 = if (dd >= 0) 1 else -1;
    const mu = t22 - t12 * t12 / (dd + sign_dd * @sqrt(dd * dd + t12 * t12));

    var f = d[p] * d[p] - mu;
    var g = d[p] * e[p];

    for (p..n) |k| {
        var cs: f64 = undefined;
        var sn: f64 = undefined;
        var r: f64 = undefined;
        gk_givens(f, g, &cs, &sn, &r);
        if (k > p) e[k - 1] = r;
        f = cs * d[k] + sn * e[k];
        e[k] = -sn * d[k] + cs * e[k];
        g = sn * d[k + 1];
        d[k + 1] *= cs;
        gk_givens(f, g, &cs, &sn, &r);
        d[k] = r;
        f = cs * e[k] + sn * d[k + 1];
        d[k + 1] = -sn * e[k] + cs * d[k + 1];
        if (k + 1 < n) {
            g = sn * e[k + 1];
            e[k + 1] *= cs;
        }
    }
    e[n - 1] = f;
}

/// QR iteration on bidiagonal to extract singular values.
fn gk_bidiag_svd(d: [*]f64, e: [*]f64, n: usize) void {
    const MAX_ITER: usize = 100 * n;
    const eps = 2.2204460492503131e-16;
    var iter: usize = 0;
    var q: usize = 0;

    while (q < n and iter < MAX_ITER) : (iter += 1) {
        // Find converged tail
        q = 0;
        while (q < n - 1) {
            const idx = n - 2 - q;
            if (@abs(e[idx]) <= eps * (@abs(d[idx]) + @abs(d[idx + 1]))) {
                e[idx] = 0;
                q += 1;
            } else break;
        }
        if (q >= n - 1) break;

        // Find start of unreduced block
        var p: usize = n - q - 1;
        while (p > 0) {
            if (@abs(e[p - 1]) <= eps * (@abs(d[p - 1]) + @abs(d[p]))) {
                e[p - 1] = 0;
                break;
            }
            p -= 1;
        }

        const block_end = n - q;
        if (block_end - p < 2) continue;

        gk_qr_step(d, e, p, block_end);
    }

    for (0..n) |i| {
        if (d[i] < 0) d[i] = -d[i];
    }
}

/// Compute singular values of A[m×n] via Golub-Kahan bidiagonalization + QR.
/// `scratch` must hold at least m*n + 3*k f64 values (k = min(m,n)).
/// Output `s[0..k]` contains singular values in descending order.
export fn svd_values_gk_f64(a_in: [*]const f64, s: [*]f64, scratch: [*]f64, m_arg: u32, n_arg: u32) void {
    const M = @as(usize, m_arg);
    const N = @as(usize, n_arg);
    const K = if (M < N) M else N;

    // Copy A into scratch
    const a = scratch;
    for (0..M * N) |i| a[i] = a_in[i];

    const diag = a + M * N;
    const superdiag = diag + K;
    // tau_left/tau_right not needed for values-only — reuse space
    const tau_left = superdiag + K;
    const tau_right = tau_left + K;
    _ = tau_right;

    gk_bidiagonalize(a, diag, superdiag, M, N);
    gk_bidiag_svd(diag, superdiag, K);

    // Sort descending
    const std = @import("std");
    std.mem.sortUnstable(f64, diag[0..K], {}, struct {
        fn cmp(_: void, lhs: f64, rhs: f64) bool {
            return lhs > rhs;
        }
    }.cmp);

    for (0..K) |i| s[i] = diag[i];
}

// --- Tests ---

test "svd_values_gk_f64 diagonal 2x2" {
    const testing = @import("std").testing;
    const a = [_]f64{ 3, 0, 0, 4 };
    var s: [2]f64 = undefined;
    var scratch: [64]f64 = undefined;
    svd_values_gk_f64(&a, &s, &scratch, 2, 2);
    try testing.expectApproxEqAbs(s[0], 4.0, 1e-10);
    try testing.expectApproxEqAbs(s[1], 3.0, 1e-10);
}

test "svd_values_gk_f64 general 2x2" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1, 2, 3, 4 };
    var s: [2]f64 = undefined;
    var scratch: [64]f64 = undefined;
    svd_values_gk_f64(&a, &s, &scratch, 2, 2);
    try testing.expectApproxEqAbs(s[0], 5.4649857042190426, 1e-8);
    try testing.expectApproxEqAbs(s[1], 0.3659661906262574, 1e-8);
}

test "svd_values_gk_f64 identity 3x3" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    var s: [3]f64 = undefined;
    var scratch: [64]f64 = undefined;
    svd_values_gk_f64(&a, &s, &scratch, 3, 3);
    try testing.expectApproxEqAbs(s[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(s[1], 1.0, 1e-10);
    try testing.expectApproxEqAbs(s[2], 1.0, 1e-10);
}

test "svd_values_gk_f64 diagonal 3x3" {
    const testing = @import("std").testing;
    const a = [_]f64{ 5, 0, 0, 0, 3, 0, 0, 0, 1 };
    var s: [3]f64 = undefined;
    var scratch: [64]f64 = undefined;
    svd_values_gk_f64(&a, &s, &scratch, 3, 3);
    try testing.expectApproxEqAbs(s[0], 5.0, 1e-10);
    try testing.expectApproxEqAbs(s[1], 3.0, 1e-10);
    try testing.expectApproxEqAbs(s[2], 1.0, 1e-10);
}

test "svd_values_gk_f64 tall 3x2" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1, 0, 0, 1, 0, 0 };
    var s: [2]f64 = undefined;
    var scratch: [64]f64 = undefined;
    svd_values_gk_f64(&a, &s, &scratch, 3, 2);
    try testing.expectApproxEqAbs(s[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(s[1], 1.0, 1e-10);
}

test "svd_values_gk_f64 matches jacobi" {
    const testing = @import("std").testing;
    // Compare against Jacobi SVD for a non-trivial matrix
    const a = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 10 }; // slightly off-singular
    var s_gk: [3]f64 = undefined;
    var scratch_gk: [128]f64 = undefined;
    svd_values_gk_f64(&a, &s_gk, &scratch_gk, 3, 3);

    // Jacobi reference
    var u: [9]f64 = undefined;
    var s_j: [3]f64 = undefined;
    var vt: [9]f64 = undefined;
    var work: [256]f64 = undefined;
    svd_f64(&a, &u, &s_j, &vt, &work, 3, 3);

    for (0..3) |i| {
        try testing.expectApproxEqAbs(s_gk[i], s_j[i], 1e-8);
    }
}
