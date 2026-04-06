//! WASM Cholesky decomposition for real matrices.
//!
//! cholesky_f64: A[n×n] → L[n×n] (lower triangular, A = L·L^T)
//! cholesky_f32: A[n×n] → L[n×n] (lower triangular, A = L·L^T)
//!
//! Input A is read-only. Output L is written to a separate buffer.
//! Returns 0 on success, 1 if not positive-definite.

const simd = @import("simd.zig");

/// Cholesky decomposition for f64 symmetric positive-definite matrices.
/// `a` is the input matrix (row-major, n×n). `out` receives L (lower triangular).
/// Returns 0 on success, 1 if the matrix is not positive-definite.
/// Column-oriented: for each column j, compute diagonal then all off-diagonal entries.
/// Eliminates per-element branching and exposes contiguous memory access.
export fn cholesky_f64(a: [*]const f64, out: [*]f64, n_arg: u32) u32 {
    const N = @as(usize, n_arg);
    for (0..N * N) |i| out[i] = 0;

    for (0..N) |j| {
        // 1. Diagonal: L[j,j] = sqrt(A[j,j] - sum(L[j,k]^2, k<j))
        const row_j = j * N;
        var dacc: simd.V2f64 = @splat(0);
        const j2 = j & ~@as(usize, 1);
        var k: usize = 0;
        while (k < j2) : (k += 2) {
            const v = simd.load2_f64(out, row_j + k);
            dacc += v * v;
        }
        var dsum = dacc[0] + dacc[1];
        while (k < j) : (k += 1) {
            const ljk = out[row_j + k];
            dsum += ljk * ljk;
        }
        const diag_val = a[j * N + j] - dsum;
        if (diag_val <= 0) return 1;
        const ljj = @sqrt(diag_val);
        out[j * N + j] = ljj;
        const inv_ljj = 1.0 / ljj;

        // 2. Off-diagonal: L[i,j] = (A[i,j] - sum(L[i,k]*L[j,k], k<j)) / L[j,j]
        //    for all i > j — no branching, straight loop
        for (j + 1..N) |i| {
            const row_i = i * N;
            var acc: simd.V2f64 = @splat(0);
            k = 0;
            while (k < j2) : (k += 2) {
                acc += simd.load2_f64(out, row_i + k) * simd.load2_f64(out, row_j + k);
            }
            var sum = acc[0] + acc[1];
            while (k < j) : (k += 1) {
                sum += out[row_i + k] * out[row_j + k];
            }
            out[i * N + j] = (a[i * N + j] - sum) * inv_ljj;
        }
    }

    return 0;
}

/// Cholesky decomposition for f32 symmetric positive-definite matrices.
/// `a` is the input matrix (row-major, n×n). `out` receives L (lower triangular).
/// Returns 0 on success, 1 if the matrix is not positive-definite.
export fn cholesky_f32(a: [*]const f32, out: [*]f32, n_arg: u32) u32 {
    const N = @as(usize, n_arg);
    for (0..N * N) |i| out[i] = 0;

    for (0..N) |j| {
        // 1. Diagonal: L[j,j] = sqrt(A[j,j] - sum(L[j,k]^2, k<j))
        const row_j = j * N;
        var dacc: simd.V4f32 = @splat(0);
        const j4 = j & ~@as(usize, 3);
        var k: usize = 0;
        while (k < j4) : (k += 4) {
            const v = simd.load4_f32(out, row_j + k);
            dacc += v * v;
        }
        var dsum = dacc[0] + dacc[1] + dacc[2] + dacc[3];
        while (k < j) : (k += 1) {
            const ljk = out[row_j + k];
            dsum += ljk * ljk;
        }
        const diag_val = a[j * N + j] - dsum;
        if (diag_val <= 0) return 1;
        const ljj = @sqrt(diag_val);
        out[j * N + j] = ljj;
        const inv_ljj = 1.0 / ljj;

        // 2. Off-diagonal: L[i,j] = (A[i,j] - sum(L[i,k]*L[j,k], k<j)) / L[j,j]
        //    for all i > j — no branching, straight loop
        for (j + 1..N) |i| {
            const row_i = i * N;
            var acc: simd.V4f32 = @splat(0);
            k = 0;
            while (k < j4) : (k += 4) {
                acc += simd.load4_f32(out, row_i + k) * simd.load4_f32(out, row_j + k);
            }
            var sum = acc[0] + acc[1] + acc[2] + acc[3];
            while (k < j) : (k += 1) {
                sum += out[row_i + k] * out[row_j + k];
            }
            out[i * N + j] = (a[i * N + j] - sum) * inv_ljj;
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

test "cholesky_f64 1x1" {
    const testing = @import("std").testing;
    const a = [_]f64{9.0};
    var out: [1]f64 = undefined;
    const rc = cholesky_f64(&a, &out, 1);
    try testing.expectEqual(rc, 0);
    try testing.expectApproxEqAbs(out[0], 3.0, 1e-10);
}

test "cholesky_f64 identity 3x3" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    var out: [9]f64 = undefined;
    const rc = cholesky_f64(&a, &out, 3);
    try testing.expectEqual(rc, 0);
    // L should be identity
    for (0..3) |i| {
        for (0..3) |j| {
            const expected: f64 = if (i == j) 1.0 else 0.0;
            try testing.expectApproxEqAbs(out[i * 3 + j], expected, 1e-10);
        }
    }
}

test "cholesky_f64 diagonal 3x3" {
    const testing = @import("std").testing;
    // A = diag(4, 9, 16) → L = diag(2, 3, 4)
    const a = [_]f64{ 4, 0, 0, 0, 9, 0, 0, 0, 16 };
    var out: [9]f64 = undefined;
    const rc = cholesky_f64(&a, &out, 3);
    try testing.expectEqual(rc, 0);
    try testing.expectApproxEqAbs(out[0], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[4], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[8], 4.0, 1e-10);
    // Off-diagonals should be zero
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[5], 0.0, 1e-10);
}

test "cholesky_f64 4x4 reconstruction" {
    const testing = @import("std").testing;
    // A = [[4,12,-16,0],[12,37,-43,0],[-16,-43,98,0],[0,0,0,1]]
    // Known: L[0,0]=2, L[1,0]=6, L[1,1]=1, L[2,0]=-8, L[2,1]=5, L[2,2]=3
    const a = [_]f64{
        4,   12,  -16, 0,
        12,  37,  -43, 0,
        -16, -43, 98,  0,
        0,   0,   0,   1,
    };
    var out: [16]f64 = undefined;
    const rc = cholesky_f64(&a, &out, 4);
    try testing.expectEqual(rc, 0);

    // Verify L*L^T ≈ A
    var recon: [16]f64 = undefined;
    for (0..4) |i| {
        for (0..4) |j| {
            var s: f64 = 0;
            for (0..4) |k| s += out[i * 4 + k] * out[j * 4 + k];
            recon[i * 4 + j] = s;
        }
    }
    for (0..16) |i| {
        try testing.expectApproxEqAbs(recon[i], a[i], 1e-10);
    }
}

test "cholesky_f32 3x3 reconstruction" {
    const testing = @import("std").testing;
    const a = [_]f32{ 25, 15, -5, 15, 18, 0, -5, 0, 11 };
    var out: [9]f32 = undefined;
    const rc = cholesky_f32(&a, &out, 3);
    try testing.expectEqual(rc, 0);

    var recon: [9]f32 = undefined;
    for (0..3) |i| {
        for (0..3) |j| {
            var s: f32 = 0;
            for (0..3) |k| s += out[i * 3 + k] * out[j * 3 + k];
            recon[i * 3 + j] = s;
        }
    }
    for (0..9) |i| {
        try testing.expectApproxEqAbs(recon[i], a[i], 1e-4);
    }
}

test "cholesky_f32 not positive-definite" {
    const a = [_]f32{ 1, 2, 2, 1 };
    var out: [4]f32 = undefined;
    const rc = cholesky_f32(&a, &out, 2);
    try @import("std").testing.expectEqual(rc, 1);
}

test "cholesky_f64 zero diagonal returns error" {
    const a = [_]f64{ 0, 0, 0, 1 };
    var out: [4]f64 = undefined;
    const rc = cholesky_f64(&a, &out, 2);
    try @import("std").testing.expectEqual(rc, 1);
}

test "cholesky_f32 1x1" {
    const testing = @import("std").testing;
    const a = [_]f32{16.0};
    var out: [1]f32 = undefined;
    const rc = cholesky_f32(&a, &out, 1);
    try testing.expectEqual(rc, 0);
    try testing.expectApproxEqAbs(out[0], 4.0, 1e-5);
}
