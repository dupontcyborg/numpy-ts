//! WASM LU decomposition with partial pivoting: PA = LU factorization for f64,
//! f32 and complex128 square matrices. L has unit diagonal (stored below
//! diagonal), U above (stored on and above diagonal); pivots are stored as an
//! i32 permutation array. Complex matrices arrive as interleaved [re, im]
//! buffers of twice the element count.
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

/// Complex division by Smith's algorithm: dividing through by the larger
/// component keeps re*re + im*im from overflowing or falling subnormal, which
/// the direct conjugate formula does for pivots near the exponent limits.
/// The caller must rule out a zero divisor.
fn cdiv(a_re: f64, a_im: f64, b_re: f64, b_im: f64) [2]f64 {
    if (@abs(b_re) >= @abs(b_im)) {
        const r = b_im / b_re;
        const d = b_re + b_im * r;
        return .{ (a_re + a_im * r) / d, (a_im - a_re * r) / d };
    }
    const r = b_re / b_im;
    const d = b_im + b_re * r;
    return .{ (a_re * r + a_im) / d, (a_im * r - a_re) / d };
}

/// Multiply a complex scalar, splatted into `a_re`/`a_im`, by the complex number
/// a 2-wide f64 load spans. `b_sw` is `b` with its halves swapped, so the cross
/// term lands in the right lanes. Goes through simd.mulAdd_f64x2 rather than
/// @mulAdd, which scalarizes on wasm32 into a soft-float fma call.
inline fn cmul2(a_re: simd.V2f64, a_im: simd.V2f64, b: simd.V2f64, b_sw: simd.V2f64) simd.V2f64 {
    const cross = simd.V2f64{ -1, 1 } * (a_im * b_sw);
    return simd.mulAdd_f64x2(a_re, b, cross);
}

/// LU factorization with partial pivoting for complex128. In-place: `a` is an
/// interleaved [re, im] buffer of 2·n·n f64 values, overwritten with LU.
/// Returns the sign of the permutation; piv[i] is the row swapped into row i.
export fn lu_factor_c128(a: [*]f64, piv: [*]i32, n: u32) i32 {
    const N = @as(usize, n);
    const stride = N * 2;
    var sign: i32 = 1;

    for (0..N) |i| piv[i] = @intCast(i);

    for (0..N) |k| {
        // Pivot on |re| + |im| rather than the true modulus, as LAPACK's izamax
        // does: it orders the candidates the same way and cannot overflow the
        // way re*re + im*im can.
        const kk = k * stride + k * 2;
        var max_val = @abs(a[kk]) + @abs(a[kk + 1]);
        var max_row: usize = k;
        for (k + 1..N) |i| {
            const idx = i * stride + k * 2;
            const v = @abs(a[idx]) + @abs(a[idx + 1]);
            if (v > max_val) {
                max_val = v;
                max_row = i;
            }
        }

        if (max_row != k) {
            const row_k = k * stride;
            const row_m = max_row * stride;
            var j: usize = 0;
            while (j < stride) : (j += 2) {
                const vk = simd.load2_f64(a, row_k + j);
                const vm = simd.load2_f64(a, row_m + j);
                simd.store2_f64(a, row_k + j, vm);
                simd.store2_f64(a, row_m + j, vk);
            }
            const tmp_piv = piv[k];
            piv[k] = piv[max_row];
            piv[max_row] = tmp_piv;
            sign = -sign;
        }

        // max_val is now the pivot's 1-norm, so it doubles as the singularity test.
        if (max_val > 1e-15) {
            const p_re = a[kk];
            const p_im = a[kk + 1];
            for (k + 1..N) |i| {
                const ik = i * stride + k * 2;
                const f = cdiv(a[ik], a[ik + 1], p_re, p_im);
                a[ik] = f[0];
                a[ik + 1] = f[1];

                const f_re: simd.V2f64 = @splat(f[0]);
                const f_im: simd.V2f64 = @splat(f[1]);
                const row_i = i * stride;
                const row_k = k * stride;
                var j: usize = (k + 1) * 2;
                while (j < stride) : (j += 2) {
                    const v = simd.load2_f64(a, row_k + j);
                    const sw = @shuffle(f64, v, undefined, [2]i32{ 1, 0 });
                    const prod = cmul2(f_re, f_im, v, sw);
                    simd.store2_f64(a, row_i + j, simd.load2_f64(a, row_i + j) - prod);
                }
            }
        }
    }

    return sign;
}

/// Solve LU @ x = Pb for a single complex RHS vector. `b` and `x` are
/// interleaved [re, im] buffers of 2·n f64 values; x receives the solution.
export fn lu_solve_c128(lu: [*]const f64, piv: [*]const i32, b: [*]const f64, x: [*]f64, n: u32) void {
    const N = @as(usize, n);
    const stride = N * 2;

    for (0..N) |i| {
        const src = @as(usize, @intCast(piv[i])) * 2;
        x[i * 2] = b[src];
        x[i * 2 + 1] = b[src + 1];
    }

    // Forward substitution: L @ y = Pb, L unit-diagonal.
    for (1..N) |i| {
        var s_re = x[i * 2];
        var s_im = x[i * 2 + 1];
        for (0..i) |j| {
            const l = i * stride + j * 2;
            const l_re = lu[l];
            const l_im = lu[l + 1];
            const x_re = x[j * 2];
            const x_im = x[j * 2 + 1];
            s_re -= l_re * x_re - l_im * x_im;
            s_im -= l_re * x_im + l_im * x_re;
        }
        x[i * 2] = s_re;
        x[i * 2 + 1] = s_im;
    }

    // Back substitution: U @ x = y.
    var i_s: isize = @intCast(N - 1);
    while (i_s >= 0) : (i_s -= 1) {
        const i: usize = @intCast(i_s);
        var s_re = x[i * 2];
        var s_im = x[i * 2 + 1];
        for (i + 1..N) |j| {
            const u = i * stride + j * 2;
            const u_re = lu[u];
            const u_im = lu[u + 1];
            const x_re = x[j * 2];
            const x_im = x[j * 2 + 1];
            s_re -= u_re * x_re - u_im * x_im;
            s_im -= u_re * x_im + u_im * x_re;
        }
        const d = i * stride + i * 2;
        const q = cdiv(s_re, s_im, lu[d], lu[d + 1]);
        x[i * 2] = q[0];
        x[i * 2 + 1] = q[1];
    }
}

/// Compute the full inverse of a complex128 matrix from its LU factorization.
/// `out` is an interleaved [re, im] buffer of 2·n·n f64 values. Row-major
/// throughout: each substitution step updates a whole row at once.
export fn lu_inv_c128(lu: [*]const f64, piv: [*]const i32, out: [*]f64, n: u32) void {
    const N = @as(usize, n);
    const stride = N * 2;

    // Forward substitution: L @ Y = P @ I. Row i of P@I is e_{piv[i]}, so
    // out[i][:] = e_{piv[i]} - sum_{j<i} L[i][j] * out[j][:].
    for (0..N) |i| {
        const row_i = i * stride;
        for (0..stride) |c| out[row_i + c] = 0;
        out[row_i + @as(usize, @intCast(piv[i])) * 2] = 1;
        for (0..i) |j| {
            const l = i * stride + j * 2;
            const f_re: simd.V2f64 = @splat(lu[l]);
            const f_im: simd.V2f64 = @splat(lu[l + 1]);
            const row_j = j * stride;
            var c: usize = 0;
            while (c < stride) : (c += 2) {
                const v = simd.load2_f64(out, row_j + c);
                const sw = @shuffle(f64, v, undefined, [2]i32{ 1, 0 });
                const prod = cmul2(f_re, f_im, v, sw);
                simd.store2_f64(out, row_i + c, simd.load2_f64(out, row_i + c) - prod);
            }
        }
    }

    // Back substitution: U @ X = Y, bottom row first.
    var i_s: isize = @intCast(N - 1);
    while (i_s >= 0) : (i_s -= 1) {
        const i: usize = @intCast(i_s);
        const row_i = i * stride;
        for (i + 1..N) |j| {
            const u = i * stride + j * 2;
            const f_re: simd.V2f64 = @splat(lu[u]);
            const f_im: simd.V2f64 = @splat(lu[u + 1]);
            const row_j = j * stride;
            var c: usize = 0;
            while (c < stride) : (c += 2) {
                const v = simd.load2_f64(out, row_j + c);
                const sw = @shuffle(f64, v, undefined, [2]i32{ 1, 0 });
                const prod = cmul2(f_re, f_im, v, sw);
                simd.store2_f64(out, row_i + c, simd.load2_f64(out, row_i + c) - prod);
            }
        }
        // One reciprocal for the whole row, so the robust division runs once.
        const d = i * stride + i * 2;
        const inv = cdiv(1, 0, lu[d], lu[d + 1]);
        const r_re: simd.V2f64 = @splat(inv[0]);
        const r_im: simd.V2f64 = @splat(inv[1]);
        var c: usize = 0;
        while (c < stride) : (c += 2) {
            const v = simd.load2_f64(out, row_i + c);
            const sw = @shuffle(f64, v, undefined, [2]i32{ 1, 0 });
            simd.store2_f64(out, row_i + c, cmul2(r_re, r_im, v, sw));
        }
    }
}

/// Complex division by Smith's algorithm in f32: dividing through by the larger
/// component keeps re*re + im*im from overflowing or falling subnormal, which
/// the direct conjugate formula does for pivots near the exponent limits — a
/// much narrower range in f32 than in f64. The caller must rule out a zero
/// divisor.
fn cdiv32(a_re: f32, a_im: f32, b_re: f32, b_im: f32) [2]f32 {
    if (@abs(b_re) >= @abs(b_im)) {
        const r = b_im / b_re;
        const d = b_re + b_im * r;
        return .{ (a_re + a_im * r) / d, (a_im - a_re * r) / d };
    }
    const r = b_re / b_im;
    const d = b_im + b_re * r;
    return .{ (a_re * r + a_im) / d, (a_im * r - a_re) / d };
}

/// Multiply a complex scalar, splatted into `a_re`/`a_im`, by the two complex
/// numbers a 4-wide f32 load spans. `b_sw` is `b` with the halves of each pair
/// swapped, so the cross terms land in the right lanes. Goes through
/// simd.mulAdd_f32x4 rather than @mulAdd, which scalarizes on wasm32 into a
/// soft-float fma call.
inline fn cmul4(a_re: simd.V4f32, a_im: simd.V4f32, b: simd.V4f32, b_sw: simd.V4f32) simd.V4f32 {
    const cross = simd.V4f32{ -1, 1, -1, 1 } * (a_im * b_sw);
    return simd.mulAdd_f32x4(a_re, b, cross);
}

/// LU factorization with partial pivoting for complex64. In-place: `a` is an
/// interleaved [re, im] buffer of 2·n·n f32 values, overwritten with LU.
/// Returns the sign of the permutation; piv[i] is the row swapped into row i.
export fn lu_factor_c64(a: [*]f32, piv: [*]i32, n: u32) i32 {
    const N = @as(usize, n);
    const stride = N * 2;
    var sign: i32 = 1;

    for (0..N) |i| piv[i] = @intCast(i);

    for (0..N) |k| {
        // Pivot on |re| + |im| rather than the true modulus, as LAPACK's icamax
        // does: it orders the candidates the same way and cannot overflow the
        // way re*re + im*im can.
        const kk = k * stride + k * 2;
        var max_val = @abs(a[kk]) + @abs(a[kk + 1]);
        var max_row: usize = k;
        for (k + 1..N) |i| {
            const idx = i * stride + k * 2;
            const v = @abs(a[idx]) + @abs(a[idx + 1]);
            if (v > max_val) {
                max_val = v;
                max_row = i;
            }
        }

        if (max_row != k) {
            const row_k = k * stride;
            const row_m = max_row * stride;
            // A V4f32 spans two complex elements, so a row of 2n floats is even
            // but can still leave one complex element past the 4-wide loop.
            const n4 = stride & ~@as(usize, 3);
            var j: usize = 0;
            while (j < n4) : (j += 4) {
                const vk = simd.load4_f32(a, row_k + j);
                const vm = simd.load4_f32(a, row_m + j);
                simd.store4_f32(a, row_k + j, vm);
                simd.store4_f32(a, row_m + j, vk);
            }
            while (j < stride) : (j += 1) {
                const tmp = a[row_k + j];
                a[row_k + j] = a[row_m + j];
                a[row_m + j] = tmp;
            }
            const tmp_piv = piv[k];
            piv[k] = piv[max_row];
            piv[max_row] = tmp_piv;
            sign = -sign;
        }

        // max_val is now the pivot's 1-norm, so it doubles as the singularity
        // test, at the same threshold lu_factor_f32 uses.
        if (max_val > 1e-7) {
            const p_re = a[kk];
            const p_im = a[kk + 1];
            for (k + 1..N) |i| {
                const ik = i * stride + k * 2;
                const f = cdiv32(a[ik], a[ik + 1], p_re, p_im);
                a[ik] = f[0];
                a[ik + 1] = f[1];

                const f_re: simd.V4f32 = @splat(f[0]);
                const f_im: simd.V4f32 = @splat(f[1]);
                const row_i = i * stride;
                const row_k = k * stride;
                const start = (k + 1) * 2;
                const n4 = start + ((stride - start) & ~@as(usize, 3));
                var j: usize = start;
                while (j < n4) : (j += 4) {
                    const v = simd.load4_f32(a, row_k + j);
                    const sw = @shuffle(f32, v, undefined, [4]i32{ 1, 0, 3, 2 });
                    const prod = cmul4(f_re, f_im, v, sw);
                    simd.store4_f32(a, row_i + j, simd.load4_f32(a, row_i + j) - prod);
                }
                while (j < stride) : (j += 2) {
                    const v_re = a[row_k + j];
                    const v_im = a[row_k + j + 1];
                    a[row_i + j] -= f[0] * v_re - f[1] * v_im;
                    a[row_i + j + 1] -= f[0] * v_im + f[1] * v_re;
                }
            }
        }
    }

    return sign;
}

/// Solve LU @ x = Pb for a single complex64 RHS vector. `b` and `x` are
/// interleaved [re, im] buffers of 2·n f32 values; x receives the solution.
export fn lu_solve_c64(lu: [*]const f32, piv: [*]const i32, b: [*]const f32, x: [*]f32, n: u32) void {
    const N = @as(usize, n);
    const stride = N * 2;

    for (0..N) |i| {
        const src = @as(usize, @intCast(piv[i])) * 2;
        x[i * 2] = b[src];
        x[i * 2 + 1] = b[src + 1];
    }

    // Forward substitution: L @ y = Pb, L unit-diagonal.
    for (1..N) |i| {
        var s_re = x[i * 2];
        var s_im = x[i * 2 + 1];
        for (0..i) |j| {
            const l = i * stride + j * 2;
            const l_re = lu[l];
            const l_im = lu[l + 1];
            const x_re = x[j * 2];
            const x_im = x[j * 2 + 1];
            s_re -= l_re * x_re - l_im * x_im;
            s_im -= l_re * x_im + l_im * x_re;
        }
        x[i * 2] = s_re;
        x[i * 2 + 1] = s_im;
    }

    // Back substitution: U @ x = y.
    var i_s: isize = @intCast(N - 1);
    while (i_s >= 0) : (i_s -= 1) {
        const i: usize = @intCast(i_s);
        var s_re = x[i * 2];
        var s_im = x[i * 2 + 1];
        for (i + 1..N) |j| {
            const u = i * stride + j * 2;
            const u_re = lu[u];
            const u_im = lu[u + 1];
            const x_re = x[j * 2];
            const x_im = x[j * 2 + 1];
            s_re -= u_re * x_re - u_im * x_im;
            s_im -= u_re * x_im + u_im * x_re;
        }
        const d = i * stride + i * 2;
        const q = cdiv32(s_re, s_im, lu[d], lu[d + 1]);
        x[i * 2] = q[0];
        x[i * 2 + 1] = q[1];
    }
}

/// Compute the full inverse of a complex64 matrix from its LU factorization.
/// `out` is an interleaved [re, im] buffer of 2·n·n f32 values. Row-major
/// throughout: each substitution step updates a whole row at once.
export fn lu_inv_c64(lu: [*]const f32, piv: [*]const i32, out: [*]f32, n: u32) void {
    const N = @as(usize, n);
    const stride = N * 2;
    // A V4f32 spans two complex elements, so an odd n leaves a trailing complex
    // element for the scalar tail of every row loop below.
    const n4 = stride & ~@as(usize, 3);

    // Forward substitution: L @ Y = P @ I. Row i of P@I is e_{piv[i]}, so
    // out[i][:] = e_{piv[i]} - sum_{j<i} L[i][j] * out[j][:].
    for (0..N) |i| {
        const row_i = i * stride;
        for (0..stride) |c| out[row_i + c] = 0;
        out[row_i + @as(usize, @intCast(piv[i])) * 2] = 1;
        for (0..i) |j| {
            const l = i * stride + j * 2;
            const l_re = lu[l];
            const l_im = lu[l + 1];
            const f_re: simd.V4f32 = @splat(l_re);
            const f_im: simd.V4f32 = @splat(l_im);
            const row_j = j * stride;
            var c: usize = 0;
            while (c < n4) : (c += 4) {
                const v = simd.load4_f32(out, row_j + c);
                const sw = @shuffle(f32, v, undefined, [4]i32{ 1, 0, 3, 2 });
                const prod = cmul4(f_re, f_im, v, sw);
                simd.store4_f32(out, row_i + c, simd.load4_f32(out, row_i + c) - prod);
            }
            while (c < stride) : (c += 2) {
                const v_re = out[row_j + c];
                const v_im = out[row_j + c + 1];
                out[row_i + c] -= l_re * v_re - l_im * v_im;
                out[row_i + c + 1] -= l_re * v_im + l_im * v_re;
            }
        }
    }

    // Back substitution: U @ X = Y, bottom row first.
    var i_s: isize = @intCast(N - 1);
    while (i_s >= 0) : (i_s -= 1) {
        const i: usize = @intCast(i_s);
        const row_i = i * stride;
        for (i + 1..N) |j| {
            const u = i * stride + j * 2;
            const u_re = lu[u];
            const u_im = lu[u + 1];
            const f_re: simd.V4f32 = @splat(u_re);
            const f_im: simd.V4f32 = @splat(u_im);
            const row_j = j * stride;
            var c: usize = 0;
            while (c < n4) : (c += 4) {
                const v = simd.load4_f32(out, row_j + c);
                const sw = @shuffle(f32, v, undefined, [4]i32{ 1, 0, 3, 2 });
                const prod = cmul4(f_re, f_im, v, sw);
                simd.store4_f32(out, row_i + c, simd.load4_f32(out, row_i + c) - prod);
            }
            while (c < stride) : (c += 2) {
                const v_re = out[row_j + c];
                const v_im = out[row_j + c + 1];
                out[row_i + c] -= u_re * v_re - u_im * v_im;
                out[row_i + c + 1] -= u_re * v_im + u_im * v_re;
            }
        }
        // One reciprocal for the whole row, so the robust division runs once.
        const d = i * stride + i * 2;
        const inv = cdiv32(1, 0, lu[d], lu[d + 1]);
        const r_re: simd.V4f32 = @splat(inv[0]);
        const r_im: simd.V4f32 = @splat(inv[1]);
        var c: usize = 0;
        while (c < n4) : (c += 4) {
            const v = simd.load4_f32(out, row_i + c);
            const sw = @shuffle(f32, v, undefined, [4]i32{ 1, 0, 3, 2 });
            simd.store4_f32(out, row_i + c, cmul4(r_re, r_im, v, sw));
        }
        while (c < stride) : (c += 2) {
            const v_re = out[row_i + c];
            const v_im = out[row_i + c + 1];
            out[row_i + c] = inv[0] * v_re - inv[1] * v_im;
            out[row_i + c + 1] = inv[0] * v_im + inv[1] * v_re;
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
    // [[2,1],[5,3]] @ x = [4,7] → x = [5,-6]: 2(5)+(-6)=4, 5(5)+3(-6)=7
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

test "lu_factor_c128 identity" {
    const testing = @import("std").testing;
    var a = [_]f64{ 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 };
    var piv: [3]i32 = undefined;
    const sign = lu_factor_c128(&a, &piv, 3);
    try testing.expectEqual(sign, 1);
    try testing.expectEqual(piv[0], 0);
    try testing.expectEqual(piv[1], 1);
    try testing.expectEqual(piv[2], 2);
}

test "lu_factor_c128 reproduces A" {
    const testing = @import("std").testing;
    // A = [[2+1i, 1-2i], [0.5+0.5i, 3-1i]]
    const orig = [_]f64{ 2, 1, 1, -2, 0.5, 0.5, 3, -1 };
    var a = orig;
    var piv: [2]i32 = undefined;
    _ = lu_factor_c128(&a, &piv, 2);

    // Rebuild P@A from L@U and compare against the permuted original.
    for (0..2) |i| {
        for (0..2) |j| {
            var re: f64 = 0;
            var im: f64 = 0;
            for (0..2) |k| {
                if (k > i) break;
                const l_re = if (k == i) 1.0 else a[i * 4 + k * 2];
                const l_im = if (k == i) 0.0 else a[i * 4 + k * 2 + 1];
                if (k > j) continue;
                const u_re = a[k * 4 + j * 2];
                const u_im = a[k * 4 + j * 2 + 1];
                re += l_re * u_re - l_im * u_im;
                im += l_re * u_im + l_im * u_re;
            }
            const src = @as(usize, @intCast(piv[i]));
            try testing.expectApproxEqAbs(re, orig[src * 4 + j * 2], 1e-12);
            try testing.expectApproxEqAbs(im, orig[src * 4 + j * 2 + 1], 1e-12);
        }
    }
}

test "lu_solve_c128 basic" {
    const testing = @import("std").testing;
    // [[1i, 0], [0, 2]] @ x = [1i, 4] -> x = [1, 2]
    var a = [_]f64{ 0, 1, 0, 0, 0, 0, 2, 0 };
    var piv: [2]i32 = undefined;
    _ = lu_factor_c128(&a, &piv, 2);
    const b = [_]f64{ 0, 1, 4, 0 };
    var x: [4]f64 = undefined;
    lu_solve_c128(&a, &piv, &b, &x, 2);
    try testing.expectApproxEqAbs(x[0], 1.0, 1e-12);
    try testing.expectApproxEqAbs(x[1], 0.0, 1e-12);
    try testing.expectApproxEqAbs(x[2], 2.0, 1e-12);
    try testing.expectApproxEqAbs(x[3], 0.0, 1e-12);
}

test "lu_inv_c128 2x2" {
    const testing = @import("std").testing;
    // A = [[2+1i, 1-2i], [0.5+0.5i, 3-1i]]; check A @ inv == I.
    const orig = [_]f64{ 2, 1, 1, -2, 0.5, 0.5, 3, -1 };
    var a = orig;
    var piv: [2]i32 = undefined;
    _ = lu_factor_c128(&a, &piv, 2);
    var inv_out: [8]f64 = undefined;
    lu_inv_c128(&a, &piv, &inv_out, 2);

    for (0..2) |i| {
        for (0..2) |j| {
            var re: f64 = 0;
            var im: f64 = 0;
            for (0..2) |k| {
                const a_re = orig[i * 4 + k * 2];
                const a_im = orig[i * 4 + k * 2 + 1];
                const b_re = inv_out[k * 4 + j * 2];
                const b_im = inv_out[k * 4 + j * 2 + 1];
                re += a_re * b_re - a_im * b_im;
                im += a_re * b_im + a_im * b_re;
            }
            const want: f64 = if (i == j) 1.0 else 0.0;
            try testing.expectApproxEqAbs(re, want, 1e-12);
            try testing.expectApproxEqAbs(im, 0.0, 1e-12);
        }
    }
}

test "lu_factor_c128 survives a tiny pivot" {
    const testing = @import("std").testing;
    // The direct conjugate division squares the pivot into the subnormals here;
    // Smith's algorithm keeps the factor finite.
    var a = [_]f64{ 1e-170, 1e-170, 1, 0, 1, 0, 1, 0 };
    var piv: [2]i32 = undefined;
    _ = lu_factor_c128(&a, &piv, 2);
    for (a) |v| try testing.expect(!@import("std").math.isNan(v) and !@import("std").math.isInf(v));
}

test "lu_factor_c64 identity" {
    const testing = @import("std").testing;
    var a = [_]f32{ 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0 };
    var piv: [3]i32 = undefined;
    const sign = lu_factor_c64(&a, &piv, 3);
    try testing.expectEqual(sign, 1);
    try testing.expectEqual(piv[0], 0);
    try testing.expectEqual(piv[1], 1);
    try testing.expectEqual(piv[2], 2);
}

test "lu_factor_c64 reproduces A" {
    const testing = @import("std").testing;
    // A = [[2+1i, 1-2i], [0.5+0.5i, 3-1i]]
    const orig = [_]f32{ 2, 1, 1, -2, 0.5, 0.5, 3, -1 };
    var a = orig;
    var piv: [2]i32 = undefined;
    _ = lu_factor_c64(&a, &piv, 2);

    // Rebuild P@A from L@U and compare against the permuted original.
    for (0..2) |i| {
        for (0..2) |j| {
            var re: f32 = 0;
            var im: f32 = 0;
            for (0..2) |k| {
                if (k > i) break;
                const l_re: f32 = if (k == i) 1.0 else a[i * 4 + k * 2];
                const l_im: f32 = if (k == i) 0.0 else a[i * 4 + k * 2 + 1];
                if (k > j) continue;
                const u_re = a[k * 4 + j * 2];
                const u_im = a[k * 4 + j * 2 + 1];
                re += l_re * u_re - l_im * u_im;
                im += l_re * u_im + l_im * u_re;
            }
            const src = @as(usize, @intCast(piv[i]));
            try testing.expectApproxEqAbs(re, orig[src * 4 + j * 2], 1e-6);
            try testing.expectApproxEqAbs(im, orig[src * 4 + j * 2 + 1], 1e-6);
        }
    }
}

test "lu_solve_c64 basic" {
    const testing = @import("std").testing;
    // [[1i, 0], [0, 2]] @ x = [1i, 4] -> x = [1, 2]
    var a = [_]f32{ 0, 1, 0, 0, 0, 0, 2, 0 };
    var piv: [2]i32 = undefined;
    _ = lu_factor_c64(&a, &piv, 2);
    const b = [_]f32{ 0, 1, 4, 0 };
    var x: [4]f32 = undefined;
    lu_solve_c64(&a, &piv, &b, &x, 2);
    try testing.expectApproxEqAbs(x[0], 1.0, 1e-6);
    try testing.expectApproxEqAbs(x[1], 0.0, 1e-6);
    try testing.expectApproxEqAbs(x[2], 2.0, 1e-6);
    try testing.expectApproxEqAbs(x[3], 0.0, 1e-6);
}

test "lu_inv_c64 3x3" {
    const testing = @import("std").testing;
    // A = [[2+1i, 1-2i, 1i], [0.5+0.5i, 3-1i, 1], [1, 2i, 2-1i]]; check A @ inv == I.
    // An odd n is what drives the scalar tail of every row loop in lu_inv_c64.
    const orig = [_]f32{ 2, 1, 1, -2, 0, 1, 0.5, 0.5, 3, -1, 1, 0, 1, 0, 0, 2, 2, -1 };
    var a = orig;
    var piv: [3]i32 = undefined;
    _ = lu_factor_c64(&a, &piv, 3);
    var inv_out: [18]f32 = undefined;
    lu_inv_c64(&a, &piv, &inv_out, 3);

    for (0..3) |i| {
        for (0..3) |j| {
            var re: f32 = 0;
            var im: f32 = 0;
            for (0..3) |k| {
                const a_re = orig[i * 6 + k * 2];
                const a_im = orig[i * 6 + k * 2 + 1];
                const b_re = inv_out[k * 6 + j * 2];
                const b_im = inv_out[k * 6 + j * 2 + 1];
                re += a_re * b_re - a_im * b_im;
                im += a_re * b_im + a_im * b_re;
            }
            const want: f32 = if (i == j) 1.0 else 0.0;
            try testing.expectApproxEqAbs(re, want, 1e-5);
            try testing.expectApproxEqAbs(im, 0.0, 1e-5);
        }
    }
}

test "lu_factor_c64 survives pivots at the exponent limits" {
    const testing = @import("std").testing;
    const math = @import("std").math;
    // The direct conjugate division overflows re*re + im*im to infinity on the
    // first matrix and flushes it to zero on the second; Smith's algorithm
    // keeps both factors finite. f32 reaches both limits at far milder
    // magnitudes than f64 does.
    var big = [_]f32{ 1e25, 1e25, 1, 0, 1e25, 0, 1, 0 };
    var piv: [2]i32 = undefined;
    _ = lu_factor_c64(&big, &piv, 2);
    for (big) |v| try testing.expect(!math.isNan(v) and !math.isInf(v));

    var tiny = [_]f32{ 1e-38, 1e-38, 1, 0, 1, 0, 1, 0 };
    _ = lu_factor_c64(&tiny, &piv, 2);
    for (tiny) |v| try testing.expect(!math.isNan(v) and !math.isInf(v));
}
