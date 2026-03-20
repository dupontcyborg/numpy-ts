//! WASM FFT kernels: mixed-radix Stockham FFT for factors {2,3,4,5},
//! with Bluestein's fallback for remaining primes.
//! Complex data stored as interleaved [re, im, re, im, ...] f64 pairs.
//!
//! Exports:
//!   fft_c128, ifft_c128         — complex-to-complex 1D FFT (f64 interleaved)
//!   fft_batch_c128, ifft_batch_c128 — batched 1D FFT
//!   fft_c64, ifft_c64           — complex-to-complex 1D FFT (f32, computed in f64)
//!   rfft_f64                    — real-to-complex 1D FFT
//!   irfft_f64                   — complex-to-real 1D inverse FFT
//!   fft_scratch_size            — compute scratch buffer size needed

const std = @import("std");
const math = std.math;

/// Compute scratch buffer size needed for an N-point complex FFT (interleaved f64).
/// For power-of-2 sizes, no scratch is needed. For mixed-radix, need 2*N f64 for ping-pong buffers.
export fn fft_scratch_size(n: u32) u32 {
    return @intCast(scratchSizeF64(@as(usize, n)));
}

/// Computes the FFT for N complex points (interleaved f64).
/// Scratch buffer must hold 2*N f64 for Stockham, or 4*N for Bluestein's.
export fn fft_c128(inp: [*]const f64, out: [*]f64, scratch: [*]f64, n: u32) void {
    fftDispatch(inp, out, scratch, @as(usize, n), false);
}

/// Compute the inverse FFT for N complex points (interleaved f64).
/// Scratch buffer must hold 2*N f64 for Stockham, or 4*N for Bluestein's.
export fn ifft_c128(inp: [*]const f64, out: [*]f64, scratch: [*]f64, n: u32) void {
    fftDispatch(inp, out, scratch, @as(usize, n), true);
}

/// Compute the FFT for N complex points (interleaved f32, computed in f64 for better accuracy).
/// Scratch buffer must hold 4*N f64 for the f64 conversion + Stockham/Bluestein's scratch.
export fn fft_c64(inp: [*]const f32, out: [*]f32, scratch: [*]f64, n: u32) void {
    const N = @as(usize, n);
    const in_f64 = scratch;
    const out_f64 = scratch + 2 * N;
    const fft_scratch = out_f64 + 2 * N;
    for (0..2 * N) |i| in_f64[i] = @as(f64, inp[i]);
    fftDispatch(in_f64, out_f64, fft_scratch, N, false);
    for (0..2 * N) |i| out[i] = @as(f32, @floatCast(out_f64[i]));
}

/// Compute the inverse FFT for N complex points (interleaved f32, computed in f64 for better accuracy).
/// Scratch buffer must hold 4*N f64 for the f64 conversion + Stockham/Bluestein's scratch.
export fn ifft_c64(inp: [*]const f32, out: [*]f32, scratch: [*]f64, n: u32) void {
    const N = @as(usize, n);
    const in_f64 = scratch;
    const out_f64 = scratch + 2 * N;
    const fft_scratch = out_f64 + 2 * N;
    for (0..2 * N) |i| in_f64[i] = @as(f64, inp[i]);
    fftDispatch(in_f64, out_f64, fft_scratch, N, true);
    for (0..2 * N) |i| out[i] = @as(f32, @floatCast(out_f64[i]));
}

/// Real-to-complex FFT: input is N real f64, output is (N/2+1) complex f64 (interleaved).
/// Scratch must hold 4*N f64 for Bluestein's.
export fn rfft_f64(inp: [*]const f64, out: [*]f64, scratch: [*]f64, n: u32) void {
    const N = @as(usize, n);
    const half_n = N / 2 + 1;
    const complex_buf = scratch;
    const fft_scratch = scratch + 2 * N;

    for (0..N) |i| {
        complex_buf[2 * i] = inp[i];
        complex_buf[2 * i + 1] = 0;
    }

    const out_buf = fft_scratch;
    const bs_scratch = fft_scratch + 2 * N;
    fftDispatch(complex_buf, out_buf, bs_scratch, N, false);

    for (0..half_n) |i| {
        out[2 * i] = out_buf[2 * i];
        out[2 * i + 1] = out_buf[2 * i + 1];
    }
}

/// Complex-to-real inverse FFT: input is (N/2+1) complex f64 (interleaved), output is N real f64.
/// Scratch must hold 4*N f64 for Bluestein's.
export fn irfft_f64(inp: [*]const f64, out: [*]f64, scratch: [*]f64, n_half: u32, n_out: u32) void {
    const N = @as(usize, n_out);
    const half_n = @as(usize, n_half);
    const full_spectrum = scratch;
    const fft_scratch = scratch + 2 * N;

    for (0..half_n) |i| {
        full_spectrum[2 * i] = inp[2 * i];
        full_spectrum[2 * i + 1] = inp[2 * i + 1];
    }
    for (1..half_n) |k| {
        if (N - k < half_n) continue;
        full_spectrum[2 * (N - k)] = inp[2 * k];
        full_spectrum[2 * (N - k) + 1] = -inp[2 * k + 1];
    }

    const out_buf = fft_scratch;
    const bs_scratch = fft_scratch + 2 * N;
    fftDispatch(full_spectrum, out_buf, bs_scratch, N, true);

    for (0..N) |i| out[i] = out_buf[2 * i];
}

/// Batched 1D FFT: computes batch independent FFTs of length n, with batch stride of n*2 (complex128).
/// Scratch buffer must hold max FFT scratch size for n, plus 2*n f64 for the input copy if needed.
export fn fft_batch_c128(inp: [*]const f64, out: [*]f64, scratch: [*]f64, n: u32, batch: u32) void {
    const N = @as(usize, n);
    const stride = N * 2;
    for (0..@as(usize, batch)) |b| {
        fftDispatch(inp + b * stride, out + b * stride, scratch, N, false);
    }
}

/// Batched 1D inverse FFT: computes batch independent inverse FFTs of length n, with batch stride of n*2 (complex128).
/// Scratch buffer must hold max FFT scratch size for n, plus 2*n f64 for the input copy if needed.
export fn ifft_batch_c128(inp: [*]const f64, out: [*]f64, scratch: [*]f64, n: u32, batch: u32) void {
    const N = @as(usize, n);
    const stride = N * 2;
    for (0..@as(usize, batch)) |b| {
        fftDispatch(inp + b * stride, out + b * stride, scratch, N, true);
    }
}

/// Main dispatch: choose algorithm based on size and factorization.
fn fftDispatch(input: [*]const f64, output: [*]f64, scratch: [*]f64, N: usize, inverse: bool) void {
    if (N <= 1) {
        if (N == 1) {
            output[0] = input[0];
            output[1] = input[1];
        }
        return;
    }

    if (isPow2(N)) {
        // In-place radix-2: fastest for power-of-2, no scratch needed
        for (0..N * 2) |i| output[i] = input[i];
        fftPow2(output, N, inverse);
    } else if (isFullyFactorable(N)) {
        // Mixed-radix Stockham for composite sizes with only small factors
        stockhamFFT(input, output, scratch, N, inverse);
    } else {
        // Bluestein's for sizes with large prime factors
        bluesteinFft(input, output, N, inverse, scratch);
    }
}

/// Transpose MxN complex matrix (interleaved f64) to NxM.
fn complexTranspose(src: [*]const f64, dst: [*]f64, rows: usize, cols: usize) void {
    for (0..rows) |r| {
        for (0..cols) |c| {
            const si = (r * cols + c) * 2;
            const di = (c * rows + r) * 2;
            dst[di] = src[si];
            dst[di + 1] = src[si + 1];
        }
    }
}

/// 2D forward complex FFT: rows x cols complex128 input → rows x cols complex128 output.
/// Does row FFTs, transpose, column FFTs (as rows), transpose back — all in WASM.
/// scratch must hold: max(scratchSizeF64(cols), scratchSizeF64(rows)) + 2*rows*cols f64s for transpose buffer.
export fn fft2_c128(inp: [*]const f64, out: [*]f64, scratch: [*]f64, rows: u32, cols: u32) void {
    const M = @as(usize, rows);
    const N = @as(usize, cols);
    const total = M * N;

    const fft_scratch_sz = @max(scratchSizeF64(N), scratchSizeF64(M));
    const fft_scratch = scratch;
    const transpose_buf = scratch + fft_scratch_sz;

    // Step 1: FFT each row (length N, M rows) — input→out
    for (0..M) |r| {
        fftDispatch(inp + r * N * 2, out + r * N * 2, fft_scratch, N, false);
    }

    // Step 2: Transpose out (MxN) → transpose_buf (NxM)
    complexTranspose(out, transpose_buf, M, N);

    // Step 3: FFT each row of transposed (length M, N rows)
    for (0..N) |c| {
        fftDispatch(transpose_buf + c * M * 2, out + c * M * 2, fft_scratch, M, false);
    }

    // Step 4: Transpose out (NxM) back → final out (MxN)
    // Need temp: reuse transpose_buf
    for (0..total * 2) |i| transpose_buf[i] = out[i];
    complexTranspose(transpose_buf, out, N, M);
}

/// 2D inverse complex FFT.
export fn ifft2_c128(inp: [*]const f64, out: [*]f64, scratch: [*]f64, rows: u32, cols: u32) void {
    const M = @as(usize, rows);
    const N = @as(usize, cols);
    const total = M * N;

    const fft_scratch_sz = @max(scratchSizeF64(N), scratchSizeF64(M));
    const fft_scratch = scratch;
    const transpose_buf = scratch + fft_scratch_sz;

    for (0..M) |r| {
        fftDispatch(inp + r * N * 2, out + r * N * 2, fft_scratch, N, true);
    }

    complexTranspose(out, transpose_buf, M, N);

    for (0..N) |c| {
        fftDispatch(transpose_buf + c * M * 2, out + c * M * 2, fft_scratch, M, true);
    }

    for (0..total * 2) |i| transpose_buf[i] = out[i];
    complexTranspose(transpose_buf, out, N, M);
}

/// Scratch size for fft2: max FFT scratch + transpose buffer.
export fn fft2_scratch_size(rows: u32, cols: u32) u32 {
    const M = @as(usize, rows);
    const N = @as(usize, cols);
    const fft_scratch_sz = @max(scratchSizeF64(N), scratchSizeF64(M));
    return @intCast(fft_scratch_sz + 2 * M * N);
}

/// 2D real-to-complex FFT: real f64[rows × cols] → complex128[rows × (cols/2+1)].
/// Fuses real→complex conversion, 2D FFT, and truncation into a single WASM call.
/// scratch must hold: 2*rows*cols (complex buffer) + fft2 scratch.
export fn rfft2_f64(inp: [*]const f64, out: [*]f64, scratch: [*]f64, rows: u32, cols: u32) void {
    const M = @as(usize, rows);
    const N = @as(usize, cols);
    const total = M * N;
    const half_n = N / 2 + 1;

    // Pack real → complex into scratch
    const complex_buf = scratch;
    for (0..total) |i| {
        complex_buf[2 * i] = inp[i];
        complex_buf[2 * i + 1] = 0;
    }

    // Run full complex 2D FFT in-place using scratch after complex_buf
    const fft_scratch_sz = @max(scratchSizeF64(N), scratchSizeF64(M));
    const fft_scratch = complex_buf + 2 * total;
    const transpose_buf = fft_scratch + fft_scratch_sz;

    // Temp output for full complex FFT
    const full_out = transpose_buf + 2 * total;

    // Row FFTs
    for (0..M) |r| {
        fftDispatch(complex_buf + r * N * 2, full_out + r * N * 2, fft_scratch, N, false);
    }

    // Transpose + column FFTs + transpose back
    complexTranspose(full_out, transpose_buf, M, N);
    for (0..N) |c| {
        fftDispatch(transpose_buf + c * M * 2, full_out + c * M * 2, fft_scratch, M, false);
    }
    for (0..total * 2) |i| transpose_buf[i] = full_out[i];
    complexTranspose(transpose_buf, full_out, N, M);

    // Truncate: copy only first half_n columns per row to output
    for (0..M) |r| {
        for (0..half_n) |c| {
            const src_idx = (r * N + c) * 2;
            const dst_idx = (r * half_n + c) * 2;
            out[dst_idx] = full_out[src_idx];
            out[dst_idx + 1] = full_out[src_idx + 1];
        }
    }
}

/// Scratch size for rfft2: complex buffer + fft scratch + 2 transpose buffers.
export fn rfft2_scratch_size(rows: u32, cols: u32) u32 {
    const M = @as(usize, rows);
    const N = @as(usize, cols);
    const fft_scratch_sz = @max(scratchSizeF64(N), scratchSizeF64(M));
    // complex_buf(2*M*N) + fft_scratch + transpose_buf(2*M*N) + full_out(2*M*N)
    return @intCast(2 * M * N + fft_scratch_sz + 2 * M * N + 2 * M * N);
}

/// 2D complex-to-real inverse FFT: complex128[rows × (cols/2+1)] → real f64[rows × cols].
/// Fuses Hermitian expansion, 2D inverse FFT, and real extraction.
export fn irfft2_f64(inp: [*]const f64, out: [*]f64, scratch: [*]f64, rows: u32, cols_half: u32, cols_out: u32) void {
    const M = @as(usize, rows);
    const half_n = @as(usize, cols_half);
    const N = @as(usize, cols_out);
    const total = M * N;

    // Expand Hermitian symmetry per row: input[rows × half_n] → full complex[rows × N]
    const full_in = scratch;
    for (0..M) |r| {
        // Copy existing coefficients
        for (0..half_n) |c| {
            const si = (r * half_n + c) * 2;
            const di = (r * N + c) * 2;
            full_in[di] = inp[si];
            full_in[di + 1] = inp[si + 1];
        }
        // Fill conjugate mirror: full[r][N-k] = conj(full[r][k])
        for (1..half_n) |k| {
            if (N - k >= half_n) {
                const si = (r * half_n + k) * 2;
                const di = (r * N + (N - k)) * 2;
                full_in[di] = inp[si];
                full_in[di + 1] = -inp[si + 1];
            }
        }
    }

    // Run full complex 2D inverse FFT
    const fft_scratch_sz = @max(scratchSizeF64(N), scratchSizeF64(M));
    const fft_scratch = full_in + 2 * total;
    const transpose_buf = fft_scratch + fft_scratch_sz;
    const full_out = transpose_buf + 2 * total;

    // Row IFFTs
    for (0..M) |r| {
        fftDispatch(full_in + r * N * 2, full_out + r * N * 2, fft_scratch, N, true);
    }

    // Transpose + column IFFTs + transpose back
    complexTranspose(full_out, transpose_buf, M, N);
    for (0..N) |c| {
        fftDispatch(transpose_buf + c * M * 2, full_out + c * M * 2, fft_scratch, M, true);
    }
    for (0..total * 2) |i| transpose_buf[i] = full_out[i];
    complexTranspose(transpose_buf, full_out, N, M);

    // Extract real parts
    for (0..total) |i| {
        out[i] = full_out[2 * i];
    }
}

/// Scratch size for irfft2.
export fn irfft2_scratch_size(rows: u32, cols_out: u32) u32 {
    const M = @as(usize, rows);
    const N = @as(usize, cols_out);
    const fft_scratch_sz = @max(scratchSizeF64(N), scratchSizeF64(M));
    // full_in(2*M*N) + fft_scratch + transpose_buf(2*M*N) + full_out(2*M*N)
    return @intCast(2 * M * N + fft_scratch_sz + 2 * M * N + 2 * M * N);
}

/// Find the next power of 2 greater than or equal to n, for zero-padding in Bluestein's algorithm.
fn nextPow2(n: usize) usize {
    var v: usize = 1;
    while (v < n) v <<= 1;
    return v;
}

/// Check if n is a power of 2 (only one bit set).
fn isPow2(n: usize) bool {
    return n > 0 and (n & (n - 1)) == 0;
}

const MAX_FACTORS = 32;

/// Factor n into allowed radix factors (4, 2, 3, 5) for Stockham FFT. Returns list of factors and count.
fn factorize(n: usize) struct { factors: [MAX_FACTORS]usize, count: usize } {
    var result: [MAX_FACTORS]usize = undefined;
    var count: usize = 0;
    var r = n;

    // Extract factors in order: 4, 2, 3, 5
    // Factor 4 first for better radix-4 usage
    while (r % 4 == 0 and count < MAX_FACTORS) {
        result[count] = 4;
        count += 1;
        r /= 4;
    }
    while (r % 2 == 0 and count < MAX_FACTORS) {
        result[count] = 2;
        count += 1;
        r /= 2;
    }
    while (r % 3 == 0 and count < MAX_FACTORS) {
        result[count] = 3;
        count += 1;
        r /= 3;
    }
    while (r % 5 == 0 and count < MAX_FACTORS) {
        result[count] = 5;
        count += 1;
        r /= 5;
    }
    // If remainder > 1, it's a prime we can't handle with mixed-radix.
    // Store it as a single factor (will need Bluestein's).
    if (r > 1 and count < MAX_FACTORS) {
        result[count] = r;
        count += 1;
    }

    return .{ .factors = result, .count = count };
}

/// Check if n is fully factorable into allowed radix factors (4, 2, 3, 5).
/// Used to decide if we can use Stockham FFT.
fn isFullyFactorable(n: usize) bool {
    var r = n;
    while (r % 2 == 0) r /= 2;
    while (r % 3 == 0) r /= 3;
    while (r % 5 == 0) r /= 5;
    return r == 1;
}

/// Stockham mixed-radix FFT
/// Out-of-place auto-sort: no bit-reversal needed.
/// Uses ping-pong buffers. Each pass processes one factor.
fn stockhamFFT(input: [*]const f64, output: [*]f64, scratch: [*]f64, N: usize, inverse: bool) void {
    if (N <= 1) {
        if (N == 1) {
            output[0] = input[0];
            output[1] = input[1];
        }
        return;
    }

    const info = factorize(N);
    const factors = info.factors;
    const nFactors = info.count;

    // Ping-pong buffers: buf0 and buf1
    // buf0 starts with input data, output alternates
    const buf0 = scratch;
    const buf1 = scratch + 2 * N;

    // Copy input to buf0
    for (0..N * 2) |i| buf0[i] = input[i];

    var src = buf0;
    var dst = buf1;
    var m: usize = 1; // product of factors processed so far

    for (0..nFactors) |fi| {
        const p = factors[fi];
        const groups = N / (m * p);

        stockhamPass(src, dst, N, p, m, groups, inverse);

        m *= p;
        // Swap buffers
        const tmp = src;
        src = dst;
        dst = tmp;
    }

    // Scale for inverse
    if (inverse) {
        const scale = 1.0 / @as(f64, @floatFromInt(N));
        for (0..N * 2) |i| src[i] *= scale;
    }

    // Copy result to output (src points to final result)
    for (0..N * 2) |i| output[i] = src[i];
}

/// Perform one Stockham pass for factor p. src and dst are interleaved complex buffers.
/// N is total size, m is product of previous factors, groups = N/(m*p).
fn stockhamPass(src: [*]f64, dst: [*]f64, N: usize, p: usize, m: usize, groups: usize, inverse: bool) void {
    const sign: f64 = if (inverse) 1.0 else -1.0;

    switch (p) {
        2 => stockhamRadix2(src, dst, N, m, groups, sign),
        3 => stockhamRadix3(src, dst, N, m, groups, sign),
        4 => stockhamRadix4(src, dst, N, m, groups, sign),
        5 => stockhamRadix5(src, dst, N, m, groups, sign),
        else => {
            // Generic DFT for prime factors — O(p²) per group but only used for large primes
            stockhamGeneric(src, dst, N, p, m, groups, sign);
        },
    }
}

/// Stockham radix-2 pass: processes groups of 2*m elements, with m from previous stages.
/// Each group is split into two halves of m elements, combined with twiddle factors W_N^(k*groups).
fn stockhamRadix2(src: [*]f64, dst: [*]f64, _: usize, m: usize, groups: usize, sign: f64) void {
    const pm = 2 * m;
    for (0..groups) |g| {
        for (0..m) |k| {
            // Twiddle factor: W_N^(k*groups) = exp(sign * 2*pi*i * k*groups / N)
            // Since N = pm * groups, W = exp(sign * 2*pi*i * k / pm)
            const angle = sign * 2.0 * math.pi * @as(f64, @floatFromInt(k)) / @as(f64, @floatFromInt(pm));
            const wr = @cos(angle);
            const wi = @sin(angle);

            const idx0 = (g * m + k) * 2;
            const idx1 = (g * m + k + groups * m) * 2;

            const a0r = src[idx0];
            const a0i = src[idx0 + 1];
            const a1r = src[idx1];
            const a1i = src[idx1 + 1];

            // Twiddle multiply: t = W * a1
            const tr = wr * a1r - wi * a1i;
            const ti = wr * a1i + wi * a1r;

            // Output indices in dst
            const o0 = (g * pm + k) * 2;
            const o1 = (g * pm + k + m) * 2;

            dst[o0] = a0r + tr;
            dst[o0 + 1] = a0i + ti;
            dst[o1] = a0r - tr;
            dst[o1 + 1] = a0i - ti;
        }
    }
}

/// Stockham radix-2 pass: processes groups of 2*m elements, with m from previous stages.
/// Each group is split into two halves of m elements, combined with twiddle factors W_N^(k*groups).
fn stockhamRadix3(src: [*]f64, dst: [*]f64, _: usize, m: usize, groups: usize, sign: f64) void {
    const pm = 3 * m;
    const c1: f64 = -0.5;
    const c2: f64 = sign * 0.86602540378443864676; // sign * sqrt(3)/2

    for (0..groups) |g| {
        for (0..m) |k| {
            const base_angle = sign * 2.0 * math.pi * @as(f64, @floatFromInt(k)) / @as(f64, @floatFromInt(pm));

            const idx0 = (g * m + k) * 2;
            const idx1 = (g * m + k + groups * m) * 2;
            const idx2 = (g * m + k + 2 * groups * m) * 2;

            const a0r = src[idx0];
            const a0i = src[idx0 + 1];
            var a1r = src[idx1];
            var a1i = src[idx1 + 1];
            var a2r = src[idx2];
            var a2i = src[idx2 + 1];

            // Twiddle a1 by W^k
            {
                const wr = @cos(base_angle);
                const wi = @sin(base_angle);
                const tr = wr * a1r - wi * a1i;
                const ti = wr * a1i + wi * a1r;
                a1r = tr;
                a1i = ti;
            }
            // Twiddle a2 by W^(2k)
            {
                const wr = @cos(2.0 * base_angle);
                const wi = @sin(2.0 * base_angle);
                const tr = wr * a2r - wi * a2i;
                const ti = wr * a2i + wi * a2r;
                a2r = tr;
                a2i = ti;
            }

            // Radix-3 butterfly
            const t1r = a1r + a2r;
            const t1i = a1i + a2i;
            const t2r = a1r - a2r;
            const t2i = a1i - a2i;

            const o0 = (g * pm + k) * 2;
            const o1 = (g * pm + k + m) * 2;
            const o2 = (g * pm + k + 2 * m) * 2;

            dst[o0] = a0r + t1r;
            dst[o0 + 1] = a0i + t1i;
            dst[o1] = a0r + c1 * t1r - c2 * t2i;
            dst[o1 + 1] = a0i + c1 * t1i + c2 * t2r;
            dst[o2] = a0r + c1 * t1r + c2 * t2i;
            dst[o2 + 1] = a0i + c1 * t1i - c2 * t2r;
        }
    }
}

/// Stockham radix-2 pass: processes groups of 2*m elements, with m from previous stages.
/// Each group is split into two halves of m elements, combined with twiddle factors W_N^(k*groups).
fn stockhamRadix4(src: [*]f64, dst: [*]f64, _: usize, m: usize, groups: usize, sign: f64) void {
    const pm = 4 * m;

    for (0..groups) |g| {
        for (0..m) |k| {
            const base_angle = sign * 2.0 * math.pi * @as(f64, @floatFromInt(k)) / @as(f64, @floatFromInt(pm));

            const idx0 = (g * m + k) * 2;
            const idx1 = (g * m + k + groups * m) * 2;
            const idx2 = (g * m + k + 2 * groups * m) * 2;
            const idx3 = (g * m + k + 3 * groups * m) * 2;

            const a0r = src[idx0];
            const a0i = src[idx0 + 1];
            var a1r = src[idx1];
            var a1i = src[idx1 + 1];
            var a2r = src[idx2];
            var a2i = src[idx2 + 1];
            var a3r = src[idx3];
            var a3i = src[idx3 + 1];

            // Twiddle a1 by W^k, a2 by W^(2k), a3 by W^(3k)
            {
                const ang1 = base_angle;
                const w1r = @cos(ang1);
                const w1i = @sin(ang1);
                const tr1 = w1r * a1r - w1i * a1i;
                a1i = w1r * a1i + w1i * a1r;
                a1r = tr1;

                const ang2 = 2.0 * base_angle;
                const w2r = @cos(ang2);
                const w2i = @sin(ang2);
                const tr2 = w2r * a2r - w2i * a2i;
                a2i = w2r * a2i + w2i * a2r;
                a2r = tr2;

                const ang3 = 3.0 * base_angle;
                const w3r = @cos(ang3);
                const w3i = @sin(ang3);
                const tr3 = w3r * a3r - w3i * a3i;
                a3i = w3r * a3i + w3i * a3r;
                a3r = tr3;
            }

            // Radix-4 butterfly
            const t0r = a0r + a2r;
            const t0i = a0i + a2i;
            const t1r = a0r - a2r;
            const t1i = a0i - a2i;
            const t2r = a1r + a3r;
            const t2i = a1i + a3i;
            const t3r = a1r - a3r;
            const t3i = a1i - a3i;

            const o0 = (g * pm + k) * 2;
            const o1 = (g * pm + k + m) * 2;
            const o2 = (g * pm + k + 2 * m) * 2;
            const o3 = (g * pm + k + 3 * m) * 2;

            dst[o0] = t0r + t2r;
            dst[o0 + 1] = t0i + t2i;
            // W_4^1 = -i (forward) or +i (inverse): sign=-1 forward, +1 inverse
            dst[o1] = t1r - sign * t3i;
            dst[o1 + 1] = t1i + sign * t3r;
            dst[o2] = t0r - t2r;
            dst[o2 + 1] = t0i - t2i;
            dst[o3] = t1r + sign * t3i;
            dst[o3 + 1] = t1i - sign * t3r;
        }
    }
}

/// Stockham radix-5 pass: processes groups of 5*m elements, with m from previous stages.
/// Each group is split into 5 parts of m elements, combined with twiddle factors W_N^(k*groups).
fn stockhamRadix5(src: [*]f64, dst: [*]f64, _: usize, m: usize, groups: usize, sign: f64) void {
    const pm = 5 * m;
    // DFT-5 constants
    const c1: f64 = 0.30901699437494742; // cos(2π/5)
    const c2: f64 = 0.95105651629515357; // sin(2π/5)
    const c3: f64 = -0.80901699437494742; // cos(4π/5)
    const c4: f64 = 0.58778525229247313; // sin(4π/5)

    for (0..groups) |g| {
        for (0..m) |k| {
            const base_angle = sign * 2.0 * math.pi * @as(f64, @floatFromInt(k)) / @as(f64, @floatFromInt(pm));

            var a: [5][2]f64 = undefined;
            inline for (0..5) |s| {
                const idx = (g * m + k + s * groups * m) * 2;
                a[s][0] = src[idx];
                a[s][1] = src[idx + 1];
            }

            // Twiddle
            inline for (1..5) |s| {
                const ang = @as(f64, @floatFromInt(s)) * base_angle;
                const wr = @cos(ang);
                const wi = @sin(ang);
                const tr = wr * a[s][0] - wi * a[s][1];
                const ti = wr * a[s][1] + wi * a[s][0];
                a[s][0] = tr;
                a[s][1] = ti;
            }

            // Radix-5 butterfly using Winograd-style formulas
            const t1r = a[1][0] + a[4][0];
            const t1i = a[1][1] + a[4][1];
            const t2r = a[2][0] + a[3][0];
            const t2i = a[2][1] + a[3][1];
            const t3r = a[1][0] - a[4][0];
            const t3i = a[1][1] - a[4][1];
            const t4r = a[2][0] - a[3][0];
            const t4i = a[2][1] - a[3][1];
            const t5r = t1r + t2r;
            const t5i = t1i + t2i;

            inline for (0..5) |s| {
                const o = (g * pm + k + s * m) * 2;
                switch (s) {
                    0 => {
                        dst[o] = a[0][0] + t5r;
                        dst[o + 1] = a[0][1] + t5i;
                    },
                    1 => {
                        dst[o] = a[0][0] + c1 * t1r + c3 * t2r - sign * (c2 * t3i + c4 * t4i);
                        dst[o + 1] = a[0][1] + c1 * t1i + c3 * t2i + sign * (c2 * t3r + c4 * t4r);
                    },
                    2 => {
                        dst[o] = a[0][0] + c3 * t1r + c1 * t2r - sign * (c4 * t3i - c2 * t4i);
                        dst[o + 1] = a[0][1] + c3 * t1i + c1 * t2i + sign * (c4 * t3r - c2 * t4r);
                    },
                    3 => {
                        dst[o] = a[0][0] + c3 * t1r + c1 * t2r + sign * (c4 * t3i - c2 * t4i);
                        dst[o + 1] = a[0][1] + c3 * t1i + c1 * t2i - sign * (c4 * t3r - c2 * t4r);
                    },
                    4 => {
                        dst[o] = a[0][0] + c1 * t1r + c3 * t2r + sign * (c2 * t3i + c4 * t4i);
                        dst[o + 1] = a[0][1] + c1 * t1i + c3 * t2i - sign * (c2 * t3r + c4 * t4r);
                    },
                    else => unreachable,
                }
            }
        }
    }
}

/// Generic Stockham pass for prime factors: O(p²) per group, only used for large primes.
/// Computes DFT of size p for each group, with twiddle factors W_N^(k*groups).
fn stockhamGeneric(src: [*]f64, dst: [*]f64, _: usize, p: usize, m: usize, groups: usize, sign: f64) void {
    const pm = p * m;

    for (0..groups) |g| {
        for (0..m) |k| {
            const base_angle = sign * 2.0 * math.pi * @as(f64, @floatFromInt(k)) / @as(f64, @floatFromInt(pm));

            // Gather and twiddle
            var buf: [128][2]f64 = undefined; // max prime factor
            for (0..p) |s| {
                const idx = (g * m + k + s * groups * m) * 2;
                var ar = src[idx];
                var ai = src[idx + 1];
                if (s > 0) {
                    const ang = @as(f64, @floatFromInt(s)) * base_angle;
                    const wr = @cos(ang);
                    const wi = @sin(ang);
                    const tr = wr * ar - wi * ai;
                    const ti = wr * ai + wi * ar;
                    ar = tr;
                    ai = ti;
                }
                buf[s][0] = ar;
                buf[s][1] = ai;
            }

            // DFT of size p (O(p²) — only used for large primes)
            for (0..p) |j| {
                var sumr: f64 = 0;
                var sumi: f64 = 0;
                for (0..p) |s| {
                    const ang = sign * 2.0 * math.pi * @as(f64, @floatFromInt(j * s)) / @as(f64, @floatFromInt(p));
                    const wr = @cos(ang);
                    const wi = @sin(ang);
                    sumr += wr * buf[s][0] - wi * buf[s][1];
                    sumi += wr * buf[s][1] + wi * buf[s][0];
                }
                const o = (g * pm + k + j * m) * 2;
                dst[o] = sumr;
                dst[o + 1] = sumi;
            }
        }
    }
}

/// Cooley-Tukey radix-2 FFT for power-of-two sizes/
/// In-place, with bit-reversal and iterative Danielson-Lanczos steps.
fn fftPow2(data: [*]f64, N: usize, inverse: bool) void {
    var j: usize = 0;
    for (1..N) |i| {
        var bit = N >> 1;
        while (j & bit != 0) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            const ti = i * 2;
            const tj = j * 2;
            const tr = data[ti];
            const timg = data[ti + 1];
            data[ti] = data[tj];
            data[ti + 1] = data[tj + 1];
            data[tj] = tr;
            data[tj + 1] = timg;
        }
    }

    const sign: f64 = if (inverse) 1.0 else -1.0;
    var len: usize = 2;
    while (len <= N) : (len <<= 1) {
        const half = len >> 1;
        const angle = sign * 2.0 * math.pi / @as(f64, @floatFromInt(len));
        const wr_step = @cos(angle);
        const wi_step = @sin(angle);

        var start: usize = 0;
        while (start < N) : (start += len) {
            var wr: f64 = 1.0;
            var wi: f64 = 0.0;
            for (0..half) |k| {
                const u_idx = (start + k) * 2;
                const v_idx = (start + k + half) * 2;
                const ur = data[u_idx];
                const ui = data[u_idx + 1];
                const vr = data[v_idx];
                const vi = data[v_idx + 1];
                const tvr = wr * vr - wi * vi;
                const tvi = wr * vi + wi * vr;
                data[u_idx] = ur + tvr;
                data[u_idx + 1] = ui + tvi;
                data[v_idx] = ur - tvr;
                data[v_idx + 1] = ui - tvi;
                const new_wr = wr * wr_step - wi * wi_step;
                wi = wr * wi_step + wi * wr_step;
                wr = new_wr;
            }
        }
    }

    if (inverse) {
        const scale = 1.0 / @as(f64, @floatFromInt(N));
        for (0..N * 2) |i| data[i] *= scale;
    }
}

/// Bluestein's algorithm: converts arbitrary-size DFT into convolution, then uses FFT for convolution.
/// Requires scratch space for chirp, padded input, and padded chirp (total ~6*N).
fn bluesteinFft(input: [*]const f64, output: [*]f64, N: usize, inverse: bool, scratch: [*]f64) void {
    if (N <= 1) {
        if (N == 1) {
            output[0] = input[0];
            output[1] = input[1];
        }
        return;
    }

    const P = nextPow2(2 * N - 1);
    const chirp = scratch;
    const a_pad = scratch + 2 * P;
    const b_pad = a_pad + 2 * P;

    const sign: f64 = if (inverse) -1.0 else 1.0;

    for (0..N) |k| {
        const angle = sign * math.pi * @as(f64, @floatFromInt(k * k)) / @as(f64, @floatFromInt(N));
        chirp[2 * k] = @cos(angle);
        chirp[2 * k + 1] = @sin(angle);
    }

    for (0..P * 2) |i| a_pad[i] = 0;
    for (0..N) |k| {
        const ir = input[2 * k];
        const ii = input[2 * k + 1];
        const cr = chirp[2 * k];
        const ci = -chirp[2 * k + 1];
        a_pad[2 * k] = ir * cr - ii * ci;
        a_pad[2 * k + 1] = ir * ci + ii * cr;
    }

    for (0..P * 2) |i| b_pad[i] = 0;
    b_pad[0] = chirp[0];
    b_pad[1] = chirp[1];
    for (1..N) |k| {
        b_pad[2 * k] = chirp[2 * k];
        b_pad[2 * k + 1] = chirp[2 * k + 1];
        b_pad[2 * (P - k)] = chirp[2 * k];
        b_pad[2 * (P - k) + 1] = chirp[2 * k + 1];
    }

    fftPow2(a_pad, P, false);
    fftPow2(b_pad, P, false);

    for (0..P) |k| {
        const ar = a_pad[2 * k];
        const ai = a_pad[2 * k + 1];
        const br = b_pad[2 * k];
        const bi = b_pad[2 * k + 1];
        a_pad[2 * k] = ar * br - ai * bi;
        a_pad[2 * k + 1] = ar * bi + ai * br;
    }

    fftPow2(a_pad, P, true);

    for (0..N) |k| {
        const ar = a_pad[2 * k];
        const ai = a_pad[2 * k + 1];
        const cr = chirp[2 * k];
        const ci = -chirp[2 * k + 1];
        output[2 * k] = ar * cr - ai * ci;
        output[2 * k + 1] = ar * ci + ai * cr;
    }

    if (inverse) {
        const scale = 1.0 / @as(f64, @floatFromInt(N));
        for (0..N * 2) |i| output[i] *= scale;
    }
}

/// Scratch size needed for Bluestein's algorithm: 6*N f64s (chirp, padded input, padded chirp).
fn bluesteinScratchF64(N: usize) usize {
    if (N <= 1) return 0;
    return 6 * nextPow2(2 * N - 1);
}

/// Scratch size needed for fftDispatch: depends on algorithm choice.
fn scratchSizeF64(N: usize) usize {
    if (N <= 1) return 0;
    if (isPow2(N)) return 0;
    if (isFullyFactorable(N)) {
        return 4 * N; // two ping-pong buffers of 2*N each
    }
    return bluesteinScratchF64(N);
}

// --- Tests ---

test "fftPow2 forward/inverse roundtrip" {
    const testing = std.testing;
    var data = [_]f64{ 1, 0, 2, 0, 3, 0, 4, 0 };
    var orig: [8]f64 = undefined;
    @memcpy(&orig, &data);

    fftPow2(&data, 4, false);
    try testing.expectApproxEqAbs(data[0], 10.0, 1e-10);
    try testing.expectApproxEqAbs(data[1], 0.0, 1e-10);
    try testing.expectApproxEqAbs(data[2], -2.0, 1e-10);
    try testing.expectApproxEqAbs(data[3], 2.0, 1e-10);

    fftPow2(&data, 4, true);
    for (0..8) |i| {
        try testing.expectApproxEqAbs(data[i], orig[i], 1e-10);
    }
}

test "mixed-radix N=6 (2*3)" {
    const testing = std.testing;
    const inp = [_]f64{ 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0 };
    var out: [12]f64 = undefined;
    var scratch: [24]f64 = undefined; // 4*6 = 24

    fftDispatch(&inp, &out, &scratch, 6, false);

    // DFT([1,2,3,4,5,6]) = [21, -3+5.196i, -3+1.732i, -3, -3-1.732i, -3-5.196i]
    try testing.expectApproxEqAbs(out[0], 21.0, 1e-8);
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-8);
    try testing.expectApproxEqAbs(out[2], -3.0, 1e-8);
    try testing.expectApproxEqAbs(out[3], 5.196152422706632, 1e-8);
    try testing.expectApproxEqAbs(out[4], -3.0, 1e-8);
    try testing.expectApproxEqAbs(out[5], 1.7320508075688772, 1e-8);
    try testing.expectApproxEqAbs(out[6], -3.0, 1e-8);
    try testing.expectApproxEqAbs(out[7], 0.0, 1e-8);

    // Roundtrip
    var inv: [12]f64 = undefined;
    fftDispatch(&out, &inv, &scratch, 6, true);
    for (0..12) |i| {
        try testing.expectApproxEqAbs(inv[i], inp[i], 1e-8);
    }
}

test "mixed-radix N=100 (4*25=4*5*5) roundtrip" {
    const testing = std.testing;
    var inp: [200]f64 = undefined;
    for (0..100) |i| {
        inp[2 * i] = @as(f64, @floatFromInt(i));
        inp[2 * i + 1] = 0;
    }
    var out: [200]f64 = undefined;
    var scratch: [400]f64 = undefined;

    fftDispatch(&inp, &out, &scratch, 100, false);

    var inv: [200]f64 = undefined;
    fftDispatch(&out, &inv, &scratch, 100, true);

    for (0..200) |i| {
        try testing.expectApproxEqAbs(inv[i], inp[i], 1e-6);
    }
}

test "mixed-radix N=1000 (8*125=2^3*5^3) roundtrip" {
    const testing = std.testing;
    var inp: [2000]f64 = undefined;
    for (0..1000) |i| {
        inp[2 * i] = @as(f64, @floatFromInt(i % 17)); // varied input
        inp[2 * i + 1] = 0;
    }
    var out: [2000]f64 = undefined;
    var scratch: [4000]f64 = undefined;

    fftDispatch(&inp, &out, &scratch, 1000, false);

    var inv: [2000]f64 = undefined;
    fftDispatch(&out, &inv, &scratch, 1000, true);

    for (0..2000) |i| {
        try testing.expectApproxEqAbs(inv[i], inp[i], 1e-6);
    }
}

test "fft_c128 roundtrip" {
    const testing = std.testing;
    const inp = [_]f64{ 1, 0, 0, 1, -1, 0, 0, -1 };
    var fwd: [8]f64 = undefined;
    var inv: [8]f64 = undefined;
    var scratch: [1]f64 = undefined; // pow2 size=4, no scratch needed

    fft_c128(&inp, &fwd, &scratch, 4);
    ifft_c128(&fwd, &inv, &scratch, 4);

    for (0..8) |i| {
        try testing.expectApproxEqAbs(inv[i], inp[i], 1e-10);
    }
}
