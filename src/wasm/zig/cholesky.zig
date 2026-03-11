//! WASM Cholesky decomposition for real matrices.
//!
//! cholesky_f64: A[n×n] → L[n×n] (lower triangular, A = L·L^T)
//! cholesky_f32: A[n×n] → L[n×n] (lower triangular, A = L·L^T)
//!
//! Input A is read-only. Output L is written to a separate buffer.
//! Returns 0 on success, 1 if not positive-definite.

/// Cholesky decomposition for f64 symmetric positive-definite matrices.
/// `a` is the input matrix (row-major, n×n). `out` receives L (lower triangular).
/// Returns 0 on success, 1 if the matrix is not positive-definite.
export fn cholesky_f64(a: [*]const f64, out: [*]f64, n_arg: u32) u32 {
    const N = @as(usize, n_arg);

    // Zero output
    for (0..N * N) |i| out[i] = 0;

    // Standard Cholesky: L[i,j] for j <= i
    for (0..N) |i| {
        for (0..i + 1) |j| {
            var sum: f64 = 0;

            if (i == j) {
                // Diagonal: L[j,j] = sqrt(A[j,j] - sum(L[j,k]^2, k<j))
                for (0..j) |k| {
                    const ljk = out[j * N + k];
                    sum += ljk * ljk;
                }
                const val = a[j * N + j] - sum;
                if (val <= 0) return 1; // Not positive-definite
                out[j * N + j] = @sqrt(val);
            } else {
                // Off-diagonal: L[i,j] = (A[i,j] - sum(L[i,k]*L[j,k], k<j)) / L[j,j]
                for (0..j) |k| {
                    sum += out[i * N + k] * out[j * N + k];
                }
                const ljj = out[j * N + j];
                if (ljj == 0) return 1; // Not positive-definite
                out[i * N + j] = (a[i * N + j] - sum) / ljj;
            }
        }
    }

    return 0;
}

/// Cholesky decomposition for f32 symmetric positive-definite matrices.
/// `a` is the input matrix (row-major, n×n). `out` receives L (lower triangular).
/// Returns 0 on success, 1 if the matrix is not positive-definite.
export fn cholesky_f32(a: [*]const f32, out: [*]f32, n_arg: u32) u32 {
    const N = @as(usize, n_arg);

    // Zero output
    for (0..N * N) |i| out[i] = 0;

    // Standard Cholesky: L[i,j] for j <= i
    for (0..N) |i| {
        for (0..i + 1) |j| {
            var sum: f32 = 0;

            if (i == j) {
                // Diagonal: L[j,j] = sqrt(A[j,j] - sum(L[j,k]^2, k<j))
                for (0..j) |k| {
                    const ljk = out[j * N + k];
                    sum += ljk * ljk;
                }
                const val = a[j * N + j] - sum;
                if (val <= 0) return 1;
                out[j * N + j] = @sqrt(val);
            } else {
                // Off-diagonal: L[i,j] = (A[i,j] - sum(L[i,k]*L[j,k], k<j)) / L[j,j]
                for (0..j) |k| {
                    sum += out[i * N + k] * out[j * N + k];
                }
                const ljj = out[j * N + j];
                if (ljj == 0) return 1;
                out[i * N + j] = (a[i * N + j] - sum) / ljj;
            }
        }
    }

    return 0;
}

// --- Tests ---

test "cholesky_f64 2x2" {
    const testing = @import("std").testing;
    // A = [[4, 2], [2, 5]]  →  L = [[2, 0], [1, 2]]
    const a = [_]f64{ 4, 2, 2, 5 };
    var out: [4]f64 = undefined;
    const rc = cholesky_f64(&a, &out, 2);
    try testing.expectEqual(rc, 0);
    try testing.expectApproxEqAbs(out[0], 2.0, 1e-10); // L[0,0]
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-10); // L[0,1]
    try testing.expectApproxEqAbs(out[2], 1.0, 1e-10); // L[1,0]
    try testing.expectApproxEqAbs(out[3], 2.0, 1e-10); // L[1,1]
}

test "cholesky_f64 3x3" {
    const testing = @import("std").testing;
    // A = [[25, 15, -5], [15, 18, 0], [-5, 0, 11]]
    const a = [_]f64{ 25, 15, -5, 15, 18, 0, -5, 0, 11 };
    var out: [9]f64 = undefined;
    const rc = cholesky_f64(&a, &out, 3);
    try testing.expectEqual(rc, 0);

    // Verify L * L^T ≈ A
    var recon: [9]f64 = undefined;
    for (0..3) |i| {
        for (0..3) |j| {
            var s: f64 = 0;
            for (0..3) |k| {
                s += out[i * 3 + k] * out[j * 3 + k]; // L[i,k] * L^T[k,j] = L[i,k] * L[j,k]
            }
            recon[i * 3 + j] = s;
        }
    }
    for (0..9) |i| {
        try testing.expectApproxEqAbs(recon[i], a[i], 1e-10);
    }
}

test "cholesky_f64 not positive-definite" {
    // A = [[1, 2], [2, 1]] — not positive-definite
    const a = [_]f64{ 1, 2, 2, 1 };
    var out: [4]f64 = undefined;
    const rc = cholesky_f64(&a, &out, 2);
    try @import("std").testing.expectEqual(rc, 1);
}

test "cholesky_f32 2x2" {
    const testing = @import("std").testing;
    const a = [_]f32{ 4, 2, 2, 5 };
    var out: [4]f32 = undefined;
    const rc = cholesky_f32(&a, &out, 2);
    try testing.expectEqual(rc, 0);
    try testing.expectApproxEqAbs(out[0], 2.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-5);
    try testing.expectApproxEqAbs(out[2], 1.0, 1e-5);
    try testing.expectApproxEqAbs(out[3], 2.0, 1e-5);
}
