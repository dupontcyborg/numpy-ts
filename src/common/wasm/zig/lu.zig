//! WASM LU decomposition with partial pivoting.
//!
//! PA = LU factorization for f64 and f32 square matrices.
//! L has unit diagonal (stored below diagonal), U above (stored on+above diagonal).
//! Pivots stored as i32 permutation array.
//!
//! The inner row-update loop uses 2-wide f64 SIMD (multiply-subtract).

const simd = @import("simd.zig");

/// LU factorization with partial pivoting for f64. In-place: a is overwritten with LU.
/// Returns sign of permutation (+1 or -1) via the return value.
/// piv[i] = row index that was swapped into row i.
export fn lu_factor_f64(a: [*]f64, piv: [*]i32, n: u32) i32 {
    const N = @as(usize, n);
    var sign: i32 = 1;

    // Initialize pivot array
    for (0..N) |i| piv[i] = @intCast(i);

    for (0..N) |k| {
        // Find pivot: row with largest |a[i][k]| for i >= k
        var max_val = @abs(a[k * N + k]);
        var max_row: usize = k;
        for (k + 1..N) |i| {
            const v = @abs(a[i * N + k]);
            if (v > max_val) {
                max_val = v;
                max_row = i;
            }
        }

        // Swap rows k and max_row
        if (max_row != k) {
            const row_k = k * N;
            const row_m = max_row * N;
            // SIMD row swap (2-wide f64)
            const n2 = N & ~@as(usize, 1);
            var j: usize = 0;
            while (j < n2) : (j += 2) {
                const vk = simd.load2_f64(a, row_k + j);
                const vm = simd.load2_f64(a, row_m + j);
                simd.store2_f64(a, row_k + j, vm);
                simd.store2_f64(a, row_m + j, vk);
            }
            if (j < N) {
                const tmp = a[row_k + j];
                a[row_k + j] = a[row_m + j];
                a[row_m + j] = tmp;
            }
            // Swap pivots
            const tmp_piv = piv[k];
            piv[k] = piv[max_row];
            piv[max_row] = tmp_piv;
            sign = -sign;
        }

        // Eliminate below pivot
        const pivot = a[k * N + k];
        if (@abs(pivot) > 1e-15) {
            for (k + 1..N) |i| {
                const factor = a[i * N + k] / pivot;
                a[i * N + k] = factor; // store L factor

                // Row update: a[i][j] -= factor * a[k][j] for j > k
                // SIMD: 2-wide f64 multiply-subtract
                const row_i = i * N;
                const row_k2 = k * N;
                const factor_v: simd.V2f64 = @splat(factor);
                const start = k + 1;
                const end = N;
                const n2 = start + ((end - start) & ~@as(usize, 1));
                var jj: usize = start;
                while (jj < n2) : (jj += 2) {
                    const ai = simd.load2_f64(a, row_i + jj);
                    const ak = simd.load2_f64(a, row_k2 + jj);
                    simd.store2_f64(a, row_i + jj, ai - factor_v * ak);
                }
                if (jj < end) {
                    a[row_i + jj] -= factor * a[row_k2 + jj];
                }
            }
        }
    }

    return sign;
}

/// Solve LU @ x = Pb for a single RHS vector. b is overwritten with solution x.
/// lu is the packed LU matrix, piv is the pivot array.
export fn lu_solve_f64(lu: [*]const f64, piv: [*]const i32, b: [*]f64, x: [*]f64, n: u32) void {
    const N = @as(usize, n);

    // Apply permutation: y[i] = b[piv[i]]
    for (0..N) |i| {
        x[i] = b[@intCast(piv[i])];
    }

    // Forward substitution: L @ y = Pb (L has unit diagonal)
    for (1..N) |i| {
        var sum = x[i];
        for (0..i) |j| {
            sum -= lu[i * N + j] * x[j];
        }
        x[i] = sum;
    }

    // Back substitution: U @ x = y
    var i_s: isize = @intCast(N - 1);
    while (i_s >= 0) : (i_s -= 1) {
        const i: usize = @intCast(i_s);
        var sum = x[i];
        for (i + 1..N) |j| {
            sum -= lu[i * N + j] * x[j];
        }
        x[i] = sum / lu[i * N + i];
    }
}

/// Compute full inverse from LU factorization.
/// Row-major access pattern: processes all columns per row to stay cache-friendly.
/// out is stored row-major as the transpose of the solution, then transposed at the end.
export fn lu_inv_f64(lu: [*]const f64, piv: [*]const i32, out: [*]f64, n: u32) void {
    const N = @as(usize, n);

    // Forward substitution: solve L @ Y = P @ I, row by row.
    // out[i][col] = (P@I)[i][col] - sum_{j<i} L[i][j] * out[j][col]
    // (P@I)[i][col] = 1 if piv[i]==col, else 0 → row i of P@I has a single 1 at column piv[i].
    // So out[i][:] = e_{piv[i]} - sum_{j<i} L[i][j] * out[j][:]
    for (0..N) |i| {
        const piv_i = @as(usize, @intCast(piv[i]));
        const row_i = i * N;
        // Initialize row to permuted identity row
        for (0..N) |c| out[row_i + c] = 0.0;
        out[row_i + piv_i] = 1.0;
        // Subtract L[i][j] * out[j][:] for j < i
        for (0..i) |j| {
            const factor = lu[i * N + j];
            const row_j = j * N;
            // SIMD row update
            const n2 = N & ~@as(usize, 1);
            const fv: simd.V2f64 = @splat(factor);
            var c: usize = 0;
            while (c < n2) : (c += 2) {
                const oi = simd.load2_f64(out, row_i + c);
                const oj = simd.load2_f64(out, row_j + c);
                simd.store2_f64(out, row_i + c, oi - fv * oj);
            }
            if (c < N) out[row_i + c] -= factor * out[row_j + c];
        }
    }

    // Back substitution: solve U @ X = Y, row by row from bottom.
    // out[i][:] = (out[i][:] - sum_{j>i} U[i][j] * out[j][:]) / U[i][i]
    var i_s: isize = @intCast(N - 1);
    while (i_s >= 0) : (i_s -= 1) {
        const i: usize = @intCast(i_s);
        const row_i = i * N;
        const diag = lu[i * N + i];
        for (i + 1..N) |j| {
            const factor = lu[i * N + j];
            const row_j = j * N;
            const n2 = N & ~@as(usize, 1);
            const fv: simd.V2f64 = @splat(factor);
            var c: usize = 0;
            while (c < n2) : (c += 2) {
                const oi = simd.load2_f64(out, row_i + c);
                const oj = simd.load2_f64(out, row_j + c);
                simd.store2_f64(out, row_i + c, oi - fv * oj);
            }
            if (c < N) out[row_i + c] -= factor * out[row_j + c];
        }
        // Scale row by 1/diag
        const inv_diag: simd.V2f64 = @splat(1.0 / diag);
        {
            const n2 = N & ~@as(usize, 1);
            var c: usize = 0;
            while (c < n2) : (c += 2) {
                simd.store2_f64(out, row_i + c, simd.load2_f64(out, row_i + c) * inv_diag);
            }
            if (c < N) out[row_i + c] /= diag;
        }
    }
}

/// LU factorization for f32. Same algorithm, f32 precision.
export fn lu_factor_f32(a: [*]f32, piv: [*]i32, n: u32) i32 {
    const N = @as(usize, n);
    var sign: i32 = 1;

    for (0..N) |i| piv[i] = @intCast(i);

    for (0..N) |k| {
        var max_val = @abs(a[k * N + k]);
        var max_row: usize = k;
        for (k + 1..N) |i| {
            const v = @abs(a[i * N + k]);
            if (v > max_val) {
                max_val = v;
                max_row = i;
            }
        }

        if (max_row != k) {
            const row_k = k * N;
            const row_m = max_row * N;
            const n4 = N & ~@as(usize, 3);
            var j: usize = 0;
            while (j < n4) : (j += 4) {
                const vk = simd.load4_f32(a, row_k + j);
                const vm = simd.load4_f32(a, row_m + j);
                simd.store4_f32(a, row_k + j, vm);
                simd.store4_f32(a, row_m + j, vk);
            }
            while (j < N) : (j += 1) {
                const tmp = a[row_k + j];
                a[row_k + j] = a[row_m + j];
                a[row_m + j] = tmp;
            }
            const tmp_piv = piv[k];
            piv[k] = piv[max_row];
            piv[max_row] = tmp_piv;
            sign = -sign;
        }

        const pivot = a[k * N + k];
        if (@abs(pivot) > 1e-7) {
            for (k + 1..N) |i| {
                const factor = a[i * N + k] / pivot;
                a[i * N + k] = factor;
                const row_i = i * N;
                const row_k2 = k * N;
                const factor_v: simd.V4f32 = @splat(factor);
                const start = k + 1;
                const end = N;
                const n4 = start + ((end - start) & ~@as(usize, 3));
                var jj: usize = start;
                while (jj < n4) : (jj += 4) {
                    const ai = simd.load4_f32(a, row_i + jj);
                    const ak = simd.load4_f32(a, row_k2 + jj);
                    simd.store4_f32(a, row_i + jj, ai - factor_v * ak);
                }
                while (jj < end) : (jj += 1) {
                    a[row_i + jj] -= factor * a[row_k2 + jj];
                }
            }
        }
    }
    return sign;
}

/// Solve for f32.
export fn lu_solve_f32(lu: [*]const f32, piv: [*]const i32, b: [*]f32, x: [*]f32, n: u32) void {
    const N = @as(usize, n);
    for (0..N) |i| x[i] = b[@intCast(piv[i])];
    for (1..N) |i| {
        var sum = x[i];
        for (0..i) |j| sum -= lu[i * N + j] * x[j];
        x[i] = sum;
    }
    var i_s: isize = @intCast(N - 1);
    while (i_s >= 0) : (i_s -= 1) {
        const i: usize = @intCast(i_s);
        var sum = x[i];
        for (i + 1..N) |j| sum -= lu[i * N + j] * x[j];
        x[i] = sum / lu[i * N + i];
    }
}

/// Inverse for f32 — row-major cache-friendly access.
export fn lu_inv_f32(lu: [*]const f32, piv: [*]const i32, out: [*]f32, n: u32) void {
    const N = @as(usize, n);

    // Forward substitution row by row
    for (0..N) |i| {
        const piv_i = @as(usize, @intCast(piv[i]));
        const row_i = i * N;
        for (0..N) |c| out[row_i + c] = 0.0;
        out[row_i + piv_i] = 1.0;
        for (0..i) |j| {
            const factor = lu[i * N + j];
            const row_j = j * N;
            const n4 = N & ~@as(usize, 3);
            const fv: simd.V4f32 = @splat(factor);
            var c: usize = 0;
            while (c < n4) : (c += 4) {
                const oi = simd.load4_f32(out, row_i + c);
                const oj = simd.load4_f32(out, row_j + c);
                simd.store4_f32(out, row_i + c, oi - fv * oj);
            }
            while (c < N) : (c += 1) out[row_i + c] -= factor * out[row_j + c];
        }
    }

    // Back substitution row by row from bottom
    var i_s: isize = @intCast(N - 1);
    while (i_s >= 0) : (i_s -= 1) {
        const i: usize = @intCast(i_s);
        const row_i = i * N;
        const diag = lu[i * N + i];
        for (i + 1..N) |j| {
            const factor = lu[i * N + j];
            const row_j = j * N;
            const n4 = N & ~@as(usize, 3);
            const fv: simd.V4f32 = @splat(factor);
            var c: usize = 0;
            while (c < n4) : (c += 4) {
                const oi = simd.load4_f32(out, row_i + c);
                const oj = simd.load4_f32(out, row_j + c);
                simd.store4_f32(out, row_i + c, oi - fv * oj);
            }
            while (c < N) : (c += 1) out[row_i + c] -= factor * out[row_j + c];
        }
        const inv_diag: simd.V4f32 = @splat(1.0 / diag);
        {
            const n4 = N & ~@as(usize, 3);
            var c: usize = 0;
            while (c < n4) : (c += 4) simd.store4_f32(out, row_i + c, simd.load4_f32(out, row_i + c) * inv_diag);
            while (c < N) : (c += 1) out[row_i + c] /= diag;
        }
    }
}

// --- Tests ---

test "lu_factor_f64 identity" {
    const testing = @import("std").testing;
    var a = [_]f64{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    var piv: [3]i32 = undefined;
    const sign = lu_factor_f64(&a, &piv, 3);
    try testing.expectEqual(sign, 1);
    try testing.expectEqual(piv[0], 0);
    try testing.expectEqual(piv[1], 1);
    try testing.expectEqual(piv[2], 2);
}

test "lu_inv_f64 2x2" {
    const testing = @import("std").testing;
    // [[4,7],[2,6]] → inv = [[0.6,-0.7],[-0.2,0.4]]
    var a = [_]f64{ 4, 7, 2, 6 };
    var piv: [2]i32 = undefined;
    _ = lu_factor_f64(&a, &piv, 2);
    var inv_out: [4]f64 = undefined;
    lu_inv_f64(&a, &piv, &inv_out, 2);
    try testing.expectApproxEqAbs(inv_out[0], 0.6, 1e-10);
    try testing.expectApproxEqAbs(inv_out[1], -0.7, 1e-10);
    try testing.expectApproxEqAbs(inv_out[2], -0.2, 1e-10);
    try testing.expectApproxEqAbs(inv_out[3], 0.4, 1e-10);
}

test "lu_solve_f64 basic" {
    const testing = @import("std").testing;
    // [[2,1],[5,3]] @ x = [4,7] → x = [5,-6]... let me compute:
    // 2x + y = 4, 5x + 3y = 7 → x=5, y=-6? No: 2(5)+(-6)=4 ✓, 5(5)+3(-6)=25-18=7 ✓
    var a = [_]f64{ 2, 1, 5, 3 };
    var piv: [2]i32 = undefined;
    _ = lu_factor_f64(&a, &piv, 2);
    var b = [_]f64{ 4, 7 };
    var x: [2]f64 = undefined;
    lu_solve_f64(&a, &piv, &b, &x, 2);
    try testing.expectApproxEqAbs(x[0], 5.0, 1e-10);
    try testing.expectApproxEqAbs(x[1], -6.0, 1e-10);
}

test "lu_factor_f32 identity" {
    const testing = @import("std").testing;
    var a = [_]f32{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    var piv: [3]i32 = undefined;
    const sign = lu_factor_f32(&a, &piv, 3);
    try testing.expectEqual(sign, 1);
    try testing.expectEqual(piv[0], 0);
    try testing.expectEqual(piv[1], 1);
    try testing.expectEqual(piv[2], 2);
}

test "lu_solve_f32 basic" {
    const testing = @import("std").testing;
    // [[2,1],[5,3]] @ x = [4,7] -> x = [5,-6]
    var a = [_]f32{ 2, 1, 5, 3 };
    var piv: [2]i32 = undefined;
    _ = lu_factor_f32(&a, &piv, 2);
    var b = [_]f32{ 4, 7 };
    var x: [2]f32 = undefined;
    lu_solve_f32(&a, &piv, &b, &x, 2);
    try testing.expectApproxEqAbs(x[0], 5.0, 1e-6);
    try testing.expectApproxEqAbs(x[1], -6.0, 1e-6);
}

test "lu_inv_f32 2x2" {
    const testing = @import("std").testing;
    // [[4,7],[2,6]] -> inv = [[0.6,-0.7],[-0.2,0.4]]
    var a = [_]f32{ 4, 7, 2, 6 };
    var piv: [2]i32 = undefined;
    _ = lu_factor_f32(&a, &piv, 2);
    var inv_out: [4]f32 = undefined;
    lu_inv_f32(&a, &piv, &inv_out, 2);
    try testing.expectApproxEqAbs(inv_out[0], 0.6, 1e-6);
    try testing.expectApproxEqAbs(inv_out[1], -0.7, 1e-6);
    try testing.expectApproxEqAbs(inv_out[2], -0.2, 1e-6);
    try testing.expectApproxEqAbs(inv_out[3], 0.4, 1e-6);
}
