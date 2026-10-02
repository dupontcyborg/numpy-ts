//! WASM Householder QR decomposition.
//!
//! qr_f64:  A[m×n] → Q[m×k], R[k×n] where k = min(m,n)
//! qr_c128: the same over the complex numbers, on interleaved [re, im] buffers
//! qr_c64:  the complex form in single precision
//!
//! The complex reflector differs in two places. alpha carries the phase of
//! a[j,j] rather than just its sign, so v = x - alpha·e1 keeps its magnitude;
//! and every inner product conjugates its first factor. Dropping either leaves
//! a Q that is not unitary, which a reconstruction check of Q·R against A will
//! not catch on its own.

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

/// Householder QR for complex128 matrices on interleaved [re, im] buffers.
/// `a` is modified in place and holds the Householder vectors below the
/// diagonal. `q` receives Q[m×k], `r` receives R[k×n], `tau_out` receives the
/// real scalars 2/(v^H v), and `scratch` holds each reflector's leading entry.
export fn qr_c128(a: [*]f64, q: [*]f64, r: [*]f64, tau_out: [*]f64, scratch: [*]f64, m_arg: u32, n_arg: u32) void {
    const M = @as(usize, m_arg);
    const N = @as(usize, n_arg);
    const K = if (M < N) M else N;

    for (0..K * N * 2) |i| r[i] = 0;

    for (0..K) |j| {
        // Scale the column by its largest component before forming the
        // reflector. H = I - 2vv^H/(v^H v) does not depend on the scale of v,
        // but v^H v does: for entries near 1e-162 the squares fall subnormal,
        // 2/(v^H v) overflows to infinity and the reflector becomes NaN.
        var max_abs: f64 = 0;
        for (j..M) |ri| {
            const re = @abs(a[(ri * N + j) * 2]);
            const im = @abs(a[(ri * N + j) * 2 + 1]);
            if (re > max_abs) max_abs = re;
            if (im > max_abs) max_abs = im;
        }
        if (max_abs == 0) {
            tau_out[j] = 0;
            scratch[j * 2] = 0;
            scratch[j * 2 + 1] = 0;
            continue;
        }
        const inv_max = 1.0 / max_abs;
        for (j..M) |ri| {
            a[(ri * N + j) * 2] *= inv_max;
            a[(ri * N + j) * 2 + 1] *= inv_max;
        }

        var norm_sq: f64 = 0;
        for (j..M) |ri| {
            const re = a[(ri * N + j) * 2];
            const im = a[(ri * N + j) * 2 + 1];
            norm_sq += re * re + im * im;
        }
        const nrm = @sqrt(norm_sq);
        if (nrm == 0) {
            tau_out[j] = 0;
            scratch[j * 2] = 0;
            scratch[j * 2 + 1] = 0;
            continue;
        }

        const ajr = a[(j * N + j) * 2];
        const aji = a[(j * N + j) * 2 + 1];
        const aabs = @sqrt(ajr * ajr + aji * aji);
        const pr = if (aabs == 0) 1.0 else ajr / aabs;
        const pi = if (aabs == 0) 0.0 else aji / aabs;

        // alpha = -(a[j,j]/|a[j,j]|)·‖x‖ lands on R's diagonal, at the original
        // scale; v = x - alpha·e1 stays scaled, which the reflector allows.
        r[(j * N + j) * 2] = -pr * nrm * max_abs;
        r[(j * N + j) * 2 + 1] = -pi * nrm * max_abs;
        a[(j * N + j) * 2] = ajr + pr * nrm;
        a[(j * N + j) * 2 + 1] = aji + pi * nrm;
        scratch[j * 2] = a[(j * N + j) * 2];
        scratch[j * 2 + 1] = a[(j * N + j) * 2 + 1];

        var vtv: f64 = 0;
        for (j..M) |ri| {
            const re = a[(ri * N + j) * 2];
            const im = a[(ri * N + j) * 2 + 1];
            vtv += re * re + im * im;
        }
        if (vtv == 0) {
            tau_out[j] = 0;
            continue;
        }
        tau_out[j] = 2.0 / vtv;

        for (j + 1..N) |col| {
            var dr: f64 = 0;
            var di: f64 = 0;
            for (j..M) |ri| {
                const vr = a[(ri * N + j) * 2];
                const vi = a[(ri * N + j) * 2 + 1];
                const cr = a[(ri * N + col) * 2];
                const ci = a[(ri * N + col) * 2 + 1];
                dr += vr * cr + vi * ci;
                di += vr * ci - vi * cr;
            }
            const fr = tau_out[j] * dr;
            const fi = tau_out[j] * di;
            for (j..M) |ri| {
                const vr = a[(ri * N + j) * 2];
                const vi = a[(ri * N + j) * 2 + 1];
                a[(ri * N + col) * 2] -= vr * fr - vi * fi;
                a[(ri * N + col) * 2 + 1] -= vr * fi + vi * fr;
            }
        }
    }

    // R above the diagonal, straight from the reduced a.
    for (0..K) |i| {
        for (i + 1..N) |col| {
            r[(i * N + col) * 2] = a[(i * N + col) * 2];
            r[(i * N + col) * 2 + 1] = a[(i * N + col) * 2 + 1];
        }
    }

    // Q = H_0 H_1 ... H_{K-1} applied to the identity, in reverse.
    for (0..M * K * 2) |i| q[i] = 0;
    const diag = if (M < K) M else K;
    for (0..diag) |i| q[(i * K + i) * 2] = 1;

    var jj: usize = K;
    while (jj > 0) {
        jj -= 1;
        if (tau_out[jj] == 0) continue;
        for (0..K) |col| {
            var dr: f64 = 0;
            var di: f64 = 0;
            for (jj..M) |ri| {
                const vr = if (ri == jj) scratch[jj * 2] else a[(ri * N + jj) * 2];
                const vi = if (ri == jj) scratch[jj * 2 + 1] else a[(ri * N + jj) * 2 + 1];
                const qr_ = q[(ri * K + col) * 2];
                const qi = q[(ri * K + col) * 2 + 1];
                dr += vr * qr_ + vi * qi;
                di += vr * qi - vi * qr_;
            }
            const fr = tau_out[jj] * dr;
            const fi = tau_out[jj] * di;
            for (jj..M) |ri| {
                const vr = if (ri == jj) scratch[jj * 2] else a[(ri * N + jj) * 2];
                const vi = if (ri == jj) scratch[jj * 2 + 1] else a[(ri * N + jj) * 2 + 1];
                q[(ri * K + col) * 2] -= vr * fr - vi * fi;
                q[(ri * K + col) * 2 + 1] -= vr * fi + vi * fr;
            }
        }
    }
}

test "qr_c128 3x2 reconstructs A and gives a unitary Q" {
    const testing = @import("std").testing;
    const M = 3;
    const N = 2;
    const K = 2;
    const src = [_]f64{ 1, -0.5, 2, -0.9, 3, -1.3, 4, -1.7, 5, -2.1, 6, -2.5 };
    var a = src;
    var q: [M * K * 2]f64 = undefined;
    var r: [K * N * 2]f64 = undefined;
    var tau: [K]f64 = undefined;
    var scratch: [K * 2]f64 = undefined;
    qr_c128(&a, &q, &r, &tau, &scratch, M, N);

    // Q·R == A
    for (0..M) |i| {
        for (0..N) |j| {
            var re: f64 = 0;
            var im: f64 = 0;
            for (0..K) |k| {
                const qr_ = q[(i * K + k) * 2];
                const qi = q[(i * K + k) * 2 + 1];
                const rr = r[(k * N + j) * 2];
                const ri = r[(k * N + j) * 2 + 1];
                re += qr_ * rr - qi * ri;
                im += qr_ * ri + qi * rr;
            }
            try testing.expectApproxEqAbs(re, src[(i * N + j) * 2], 1e-12);
            try testing.expectApproxEqAbs(im, src[(i * N + j) * 2 + 1], 1e-12);
        }
    }

    // Q^H·Q == I
    for (0..K) |c1| {
        for (0..K) |c2| {
            var re: f64 = 0;
            var im: f64 = 0;
            for (0..M) |i| {
                const ar = q[(i * K + c1) * 2];
                const ai = q[(i * K + c1) * 2 + 1];
                const br = q[(i * K + c2) * 2];
                const bi = q[(i * K + c2) * 2 + 1];
                re += ar * br + ai * bi;
                im += ar * bi - ai * br;
            }
            const want: f64 = if (c1 == c2) 1.0 else 0.0;
            try testing.expectApproxEqAbs(re, want, 1e-12);
            try testing.expectApproxEqAbs(im, 0.0, 1e-12);
        }
    }

    // R lower triangle is zero
    try testing.expectApproxEqAbs(r[(1 * N + 0) * 2], 0.0, 1e-12);
    try testing.expectApproxEqAbs(r[(1 * N + 0) * 2 + 1], 0.0, 1e-12);
}

test "qr_c128 stays unitary when the leading entry is tiny" {
    const testing = @import("std").testing;
    const M = 2;
    const N = 2;
    const K = 2;
    // Entries whose squares land in the subnormal range: small enough that an
    // unscaled v^H v loses them, large enough that it does not reach exactly
    // zero and take the zero-column early-out instead.
    const t = 1e-160;
    const src = [_]f64{ t, t * 0.5, 1, 0.3, t * 0.25, t, 0.2, 1 };
    var a = src;
    var q: [M * K * 2]f64 = undefined;
    var r: [K * N * 2]f64 = undefined;
    var tau: [K]f64 = undefined;
    var scratch: [K * 2]f64 = undefined;
    qr_c128(&a, &q, &r, &tau, &scratch, M, N);

    for (0..K) |c1| {
        for (0..K) |c2| {
            var re: f64 = 0;
            var im: f64 = 0;
            for (0..M) |i| {
                const ar = q[(i * K + c1) * 2];
                const ai = q[(i * K + c1) * 2 + 1];
                const br = q[(i * K + c2) * 2];
                const bi = q[(i * K + c2) * 2 + 1];
                re += ar * br + ai * bi;
                im += ar * bi - ai * br;
            }
            const want: f64 = if (c1 == c2) 1.0 else 0.0;
            try testing.expectApproxEqAbs(re, want, 1e-12);
            try testing.expectApproxEqAbs(im, 0.0, 1e-12);
        }
    }
}

/// Householder QR for complex64 matrices on interleaved [re, im] buffers.
/// `a` is modified in place and holds the Householder vectors below the
/// diagonal. `q` receives Q[m×k], `r` receives R[k×n], `tau_out` receives the
/// real scalars 2/(v^H v), and `scratch` holds each reflector's leading entry.
export fn qr_c64(a: [*]f32, q: [*]f32, r: [*]f32, tau_out: [*]f32, scratch: [*]f32, m_arg: u32, n_arg: u32) void {
    const M = @as(usize, m_arg);
    const N = @as(usize, n_arg);
    const K = if (M < N) M else N;

    for (0..K * N * 2) |i| r[i] = 0;

    for (0..K) |j| {
        // Scale the column by its largest component before forming the
        // reflector. H = I - 2vv^H/(v^H v) does not depend on the scale of v,
        // but v^H v does: for small entries the squares fall subnormal,
        // 2/(v^H v) overflows to infinity and the reflector becomes NaN. The
        // f32 exponent range is narrow enough that ordinary inputs reach it.
        var max_abs: f32 = 0;
        for (j..M) |ri| {
            const re = @abs(a[(ri * N + j) * 2]);
            const im = @abs(a[(ri * N + j) * 2 + 1]);
            if (re > max_abs) max_abs = re;
            if (im > max_abs) max_abs = im;
        }
        if (max_abs == 0) {
            tau_out[j] = 0;
            scratch[j * 2] = 0;
            scratch[j * 2 + 1] = 0;
            continue;
        }
        const inv_max = 1.0 / max_abs;
        for (j..M) |ri| {
            a[(ri * N + j) * 2] *= inv_max;
            a[(ri * N + j) * 2 + 1] *= inv_max;
        }

        var norm_sq: f32 = 0;
        for (j..M) |ri| {
            const re = a[(ri * N + j) * 2];
            const im = a[(ri * N + j) * 2 + 1];
            norm_sq += re * re + im * im;
        }
        const nrm = @sqrt(norm_sq);
        if (nrm == 0) {
            tau_out[j] = 0;
            scratch[j * 2] = 0;
            scratch[j * 2 + 1] = 0;
            continue;
        }

        const ajr = a[(j * N + j) * 2];
        const aji = a[(j * N + j) * 2 + 1];
        const aabs = @sqrt(ajr * ajr + aji * aji);
        const pr = if (aabs == 0) 1.0 else ajr / aabs;
        const pi = if (aabs == 0) 0.0 else aji / aabs;

        // alpha = -(a[j,j]/|a[j,j]|)·‖x‖ lands on R's diagonal, at the original
        // scale; v = x - alpha·e1 stays scaled, which the reflector allows.
        r[(j * N + j) * 2] = -pr * nrm * max_abs;
        r[(j * N + j) * 2 + 1] = -pi * nrm * max_abs;
        a[(j * N + j) * 2] = ajr + pr * nrm;
        a[(j * N + j) * 2 + 1] = aji + pi * nrm;
        scratch[j * 2] = a[(j * N + j) * 2];
        scratch[j * 2 + 1] = a[(j * N + j) * 2 + 1];

        var vtv: f32 = 0;
        for (j..M) |ri| {
            const re = a[(ri * N + j) * 2];
            const im = a[(ri * N + j) * 2 + 1];
            vtv += re * re + im * im;
        }
        if (vtv == 0) {
            tau_out[j] = 0;
            continue;
        }
        tau_out[j] = 2.0 / vtv;

        for (j + 1..N) |col| {
            var dr: f32 = 0;
            var di: f32 = 0;
            for (j..M) |ri| {
                const vr = a[(ri * N + j) * 2];
                const vi = a[(ri * N + j) * 2 + 1];
                const cr = a[(ri * N + col) * 2];
                const ci = a[(ri * N + col) * 2 + 1];
                dr += vr * cr + vi * ci;
                di += vr * ci - vi * cr;
            }
            const fr = tau_out[j] * dr;
            const fi = tau_out[j] * di;
            for (j..M) |ri| {
                const vr = a[(ri * N + j) * 2];
                const vi = a[(ri * N + j) * 2 + 1];
                a[(ri * N + col) * 2] -= vr * fr - vi * fi;
                a[(ri * N + col) * 2 + 1] -= vr * fi + vi * fr;
            }
        }
    }

    // R above the diagonal, straight from the reduced a.
    for (0..K) |i| {
        for (i + 1..N) |col| {
            r[(i * N + col) * 2] = a[(i * N + col) * 2];
            r[(i * N + col) * 2 + 1] = a[(i * N + col) * 2 + 1];
        }
    }

    // Q = H_0 H_1 ... H_{K-1} applied to the identity, in reverse.
    for (0..M * K * 2) |i| q[i] = 0;
    const diag = if (M < K) M else K;
    for (0..diag) |i| q[(i * K + i) * 2] = 1;

    var jj: usize = K;
    while (jj > 0) {
        jj -= 1;
        if (tau_out[jj] == 0) continue;
        for (0..K) |col| {
            var dr: f32 = 0;
            var di: f32 = 0;
            for (jj..M) |ri| {
                const vr = if (ri == jj) scratch[jj * 2] else a[(ri * N + jj) * 2];
                const vi = if (ri == jj) scratch[jj * 2 + 1] else a[(ri * N + jj) * 2 + 1];
                const qr_ = q[(ri * K + col) * 2];
                const qi = q[(ri * K + col) * 2 + 1];
                dr += vr * qr_ + vi * qi;
                di += vr * qi - vi * qr_;
            }
            const fr = tau_out[jj] * dr;
            const fi = tau_out[jj] * di;
            for (jj..M) |ri| {
                const vr = if (ri == jj) scratch[jj * 2] else a[(ri * N + jj) * 2];
                const vi = if (ri == jj) scratch[jj * 2 + 1] else a[(ri * N + jj) * 2 + 1];
                q[(ri * K + col) * 2] -= vr * fr - vi * fi;
                q[(ri * K + col) * 2 + 1] -= vr * fi + vi * fr;
            }
        }
    }
}

test "qr_c64 3x2 reconstructs A and gives a unitary Q" {
    const testing = @import("std").testing;
    const M = 3;
    const N = 2;
    const K = 2;
    const src = [_]f32{ 1, -0.5, 2, -0.9, 3, -1.3, 4, -1.7, 5, -2.1, 6, -2.5 };
    var a = src;
    var q: [M * K * 2]f32 = undefined;
    var r: [K * N * 2]f32 = undefined;
    var tau: [K]f32 = undefined;
    var scratch: [K * 2]f32 = undefined;
    qr_c64(&a, &q, &r, &tau, &scratch, M, N);

    // Q·R == A
    for (0..M) |i| {
        for (0..N) |j| {
            var re: f32 = 0;
            var im: f32 = 0;
            for (0..K) |k| {
                const qr_ = q[(i * K + k) * 2];
                const qi = q[(i * K + k) * 2 + 1];
                const rr = r[(k * N + j) * 2];
                const ri = r[(k * N + j) * 2 + 1];
                re += qr_ * rr - qi * ri;
                im += qr_ * ri + qi * rr;
            }
            try testing.expectApproxEqAbs(re, src[(i * N + j) * 2], 1e-4);
            try testing.expectApproxEqAbs(im, src[(i * N + j) * 2 + 1], 1e-4);
        }
    }

    // Q^H·Q == I
    for (0..K) |c1| {
        for (0..K) |c2| {
            var re: f32 = 0;
            var im: f32 = 0;
            for (0..M) |i| {
                const ar = q[(i * K + c1) * 2];
                const ai = q[(i * K + c1) * 2 + 1];
                const br = q[(i * K + c2) * 2];
                const bi = q[(i * K + c2) * 2 + 1];
                re += ar * br + ai * bi;
                im += ar * bi - ai * br;
            }
            const want: f32 = if (c1 == c2) 1.0 else 0.0;
            try testing.expectApproxEqAbs(re, want, 1e-5);
            try testing.expectApproxEqAbs(im, 0.0, 1e-5);
        }
    }

    // R lower triangle is zero
    try testing.expectApproxEqAbs(r[(1 * N + 0) * 2], 0.0, 1e-5);
    try testing.expectApproxEqAbs(r[(1 * N + 0) * 2 + 1], 0.0, 1e-5);
}

test "qr_c64 stays unitary when the leading entry is tiny" {
    const testing = @import("std").testing;
    const M = 2;
    const N = 2;
    const K = 2;
    // Entries small enough that their squares fall subnormal in f32, so an
    // unscaled v^H v sends 2/(v^H v) to infinity and the reflector to NaN.
    const t: f32 = 1e-21;
    const src = [_]f32{ t, t * 0.5, 1, 0.3, t * 0.25, t, 0.2, 1 };
    var a = src;
    var q: [M * K * 2]f32 = undefined;
    var r: [K * N * 2]f32 = undefined;
    var tau: [K]f32 = undefined;
    var scratch: [K * 2]f32 = undefined;
    qr_c64(&a, &q, &r, &tau, &scratch, M, N);

    for (0..K) |c1| {
        for (0..K) |c2| {
            var re: f32 = 0;
            var im: f32 = 0;
            for (0..M) |i| {
                const ar = q[(i * K + c1) * 2];
                const ai = q[(i * K + c1) * 2 + 1];
                const br = q[(i * K + c2) * 2];
                const bi = q[(i * K + c2) * 2 + 1];
                re += ar * br + ai * bi;
                im += ar * bi - ai * br;
            }
            const want: f32 = if (c1 == c2) 1.0 else 0.0;
            try testing.expectApproxEqAbs(re, want, 1e-5);
            try testing.expectApproxEqAbs(im, 0.0, 1e-5);
        }
    }
}
