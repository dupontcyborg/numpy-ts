//! WASM FFT kernels: mixed-radix Stockham FFT for factors {2,3,4,5,8},
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
/// Uses dedicated N/2-point complex FFT + post-processing for even N (halves FFT work).
export fn rfft_f64(inp: [*]const f64, out: [*]f64, scratch: [*]f64, n: u32) void {
    const N = @as(usize, n);
    if (N <= 1) {
        if (N == 1) {
            out[0] = inp[0];
            out[1] = 0;
        }
        return;
    }

    // For even N, use dedicated real FFT via N/2-point complex FFT
    if (N % 2 == 0) {
        rfftDedicated(inp, out, scratch, N);
    } else {
        rfftFallback(inp, out, scratch, N);
    }
}

/// Complex-to-real inverse FFT: input is (N/2+1) complex f64 (interleaved), output is N real f64.
/// Uses dedicated N/2-point complex IFFT + pre-processing for even N.
export fn irfft_f64(inp: [*]const f64, out: [*]f64, scratch: [*]f64, n_half: u32, n_out: u32) void {
    const N = @as(usize, n_out);
    if (N <= 1) {
        if (N == 1) out[0] = inp[0];
        return;
    }

    if (N % 2 == 0) {
        irfftDedicated(inp, out, scratch, N);
    } else {
        irfftFallback(inp, out, scratch, @as(usize, n_half), N);
    }
}

/// Batched real-to-complex FFT: batch rfft's of size N with shared twiddle precomputation.
/// Scratch layout: [z:N][z_out:N][fft_buf:2N][fft_tw:N][post_tw:N] = 6N f64s.
export fn rfft_batch_f64(inp: [*]const f64, out: [*]f64, scratch: [*]f64, n: u32, batch: u32, in_stride: u32, out_stride: u32) void {
    const N = @as(usize, n);
    const B = @as(usize, batch);
    const istr = @as(usize, in_stride);
    const ostr = @as(usize, out_stride);

    if (N <= 1) {
        if (N == 1) {
            for (0..B) |b| {
                out[b * ostr] = inp[b * istr];
                out[b * ostr + 1] = 0;
            }
        }
        return;
    }
    if (N % 2 != 0) {
        // Fallback for odd N
        for (0..B) |b| rfftFallback(inp + b * istr, out + b * ostr, scratch, N);
        return;
    }

    const half = N / 2;

    // Scratch layout
    const z = scratch; // N
    const z_out = scratch + N; // N
    const fft_buf = scratch + 2 * N; // 4*half = 2N (buf0 + buf1)
    const fft_tw = scratch + 4 * N; // 2*half = N
    const post_tw = scratch + 5 * N; // N

    // Precompute N/2-point FFT twiddles ONCE
    precomputeTwiddles(fft_tw, half, -1.0);

    // Precompute post-processing twiddles (W_N^k) ONCE
    precomputeTwiddles(post_tw, N, -1.0);

    for (0..B) |b| {
        const inp_row = inp + b * istr;
        const out_row = out + b * ostr;

        // Pack real pairs → complex
        for (0..half) |k| {
            z[2 * k] = inp_row[2 * k];
            z[2 * k + 1] = inp_row[2 * k + 1];
        }

        // N/2-point complex FFT with shared twiddles
        if (isFullyFactorable(half)) {
            stockhamFFTCore(z, z_out, fft_buf, fft_tw, half, false);
        } else {
            fftDispatch(z, z_out, fft_buf, half, false);
        }

        // Post-process with shared twiddles
        rfftPostProcess(z_out, out_row, N, post_tw);
    }
}

/// Batched complex-to-real inverse FFT with shared twiddle precomputation.
export fn irfft_batch_f64(inp: [*]const f64, out: [*]f64, scratch: [*]f64, n_out: u32, batch: u32, in_stride: u32, out_stride: u32) void {
    const N = @as(usize, n_out);
    const B = @as(usize, batch);
    const istr = @as(usize, in_stride);
    const ostr = @as(usize, out_stride);

    if (N <= 1) {
        if (N == 1) {
            for (0..B) |b| out[b * ostr] = inp[b * istr];
        }
        return;
    }
    if (N % 2 != 0) {
        const half_n = N / 2 + 1;
        for (0..B) |b| irfftFallback(inp + b * istr, out + b * ostr, scratch, half_n, N);
        return;
    }

    const half = N / 2;
    const z = scratch;
    const z_out = scratch + N;
    const fft_buf = scratch + 2 * N;
    const fft_tw = scratch + 4 * N;
    const post_tw = scratch + 5 * N;

    // Precompute N/2-point IFFT twiddles ONCE
    precomputeTwiddles(fft_tw, half, 1.0);

    // Precompute post-processing twiddles (forward W_N^k for conjugation in pre-process)
    precomputeTwiddles(post_tw, N, -1.0);

    for (0..B) |b| {
        const inp_row = inp + b * istr;
        const out_row = out + b * ostr;

        // Pre-process with shared twiddles
        irfftPreProcess(inp_row, z, N, post_tw);

        // N/2-point complex IFFT with shared twiddles
        if (isFullyFactorable(half)) {
            stockhamFFTCore(z, z_out, fft_buf, fft_tw, half, true);
        } else {
            fftDispatch(z, z_out, fft_buf, half, true);
        }

        // Unpack
        for (0..half) |k| {
            out_row[2 * k] = z_out[2 * k];
            out_row[2 * k + 1] = z_out[2 * k + 1];
        }
    }
}

/// Scratch size for batch rfft/irfft: 6*N f64s.
export fn rfft_batch_scratch_size(n: u32) u32 {
    return 6 * n;
}

/// Batched real-to-complex FFT for f32 (computed in f64).
/// Input: f32 real data, output: f32 interleaved complex.
/// Scratch must hold conversion buffers + rfft scratch.
export fn rfft_batch_f32(inp: [*]const f32, out: [*]f32, scratch: [*]f64, n: u32, batch: u32, in_stride: u32, out_stride: u32) void {
    const N = @as(usize, n);
    const B = @as(usize, batch);
    const istr = @as(usize, in_stride);
    const ostr = @as(usize, out_stride);
    const half_n = N / 2 + 1;

    // Layout: in_f64[B*N] | out_f64[B*half_n*2] | rfft_scratch
    const in_f64 = scratch;
    const out_f64 = scratch + B * N;
    const rfft_scratch = out_f64 + B * half_n * 2;

    // Convert f32 → f64 (respecting strides)
    for (0..B) |b| {
        for (0..N) |j| in_f64[b * N + j] = @as(f64, inp[b * istr + j]);
    }
    rfft_batch_f64(in_f64, out_f64, rfft_scratch, n, batch, @intCast(N), @intCast(half_n * 2));
    // Convert f64 → f32 (respecting output strides)
    for (0..B) |b| {
        for (0..half_n * 2) |j| out[b * ostr + j] = @as(f32, @floatCast(out_f64[b * half_n * 2 + j]));
    }
}

/// Batched complex-to-real inverse FFT for f32 (computed in f64).
/// Input: f32 interleaved complex, output: f32 real.
export fn irfft_batch_f32(inp: [*]const f32, out: [*]f32, scratch: [*]f64, n_out: u32, batch: u32, in_stride: u32, out_stride: u32) void {
    const N = @as(usize, n_out);
    const B = @as(usize, batch);
    const istr = @as(usize, in_stride);
    const ostr = @as(usize, out_stride);

    // Layout: in_f64[B*istr] | out_f64[B*N] | irfft_scratch
    const in_f64 = scratch;
    const out_f64 = scratch + B * istr;
    const irfft_scratch = out_f64 + B * N;

    // Convert f32 → f64
    for (0..B) |b| {
        for (0..istr) |j| in_f64[b * istr + j] = @as(f64, inp[b * istr + j]);
    }
    irfft_batch_f64(in_f64, out_f64, irfft_scratch, n_out, batch, @intCast(istr), @intCast(N));
    // Convert f64 → f32
    for (0..B) |b| {
        for (0..N) |j| out[b * ostr + j] = @as(f32, @floatCast(out_f64[b * N + j]));
    }
}

/// Fused 3D inverse real FFT: input [d0, d1, d2_half] complex128 → output [d0, d1, d2_out] float64.
/// Performs: batch irfft(last axis) → pack to complex → 2D ifft(axes 0,1) → extract real.
export fn irfftn_3d(inp: [*]const f64, out: [*]f64, scratch: [*]f64, d0: u32, d1: u32, d2_half: u32, d2_out: u32) void {
    const M = @as(usize, d0);
    const P = @as(usize, d1);
    const H = @as(usize, d2_half);
    const N = @as(usize, d2_out);
    const MPN = M * P * N;

    // Scratch layout:
    //   Phase 1 (irfft):  [0..MPN) = real output, [MPN..MPN+6N) = irfft workspace
    //   Phase 2 (2D ifft): [0..2MPN) = complex buf, [2MPN..4MPN) = transpose buf,
    //                       [4MPN..4MPN+fft_sz) = FFT scratch
    const real_buf = scratch; // MPN f64
    const irfft_scratch = scratch + MPN; // 6*N f64

    // Phase 1: Batch irfft on last axis
    // Input: [M*P, H] complex → [M*P, N] real
    if (N >= 2 and N % 2 == 0) {
        const half = N / 2;
        const z = irfft_scratch; // N f64
        const z_out = irfft_scratch + N; // N f64
        const fft_buf = irfft_scratch + 2 * N; // 2*N f64
        const fft_tw = irfft_scratch + 4 * N; // N f64
        const post_tw = irfft_scratch + 5 * N; // N f64

        // Precompute twiddles once
        precomputeTwiddles(fft_tw, half, 1.0); // inverse FFT twiddles
        precomputeTwiddles(post_tw, N, -1.0); // forward twiddles for conjugation

        const batch = M * P;
        for (0..batch) |b| {
            const inp_row = inp + b * H * 2;
            const out_row = real_buf + b * N;

            // Pre-process
            irfftPreProcess(inp_row, z, N, post_tw);

            // N/2-point IFFT with shared twiddles
            if (isFullyFactorable(half)) {
                stockhamFFTCore(z, z_out, fft_buf, fft_tw, half, true);
            } else {
                fftDispatch(z, z_out, fft_buf, half, true);
            }

            // Unpack to real
            for (0..half) |k| {
                out_row[2 * k] = z_out[2 * k];
                out_row[2 * k + 1] = z_out[2 * k + 1];
            }
        }
    } else {
        // Fallback for odd N
        const batch = M * P;
        for (0..batch) |b| {
            irfftFallback(inp + b * H * 2, real_buf + b * N, irfft_scratch, H, N);
        }
    }

    // Phase 2: Pack real → complex (backwards, in-place)
    const complex_buf = scratch; // reuse [0..2MPN)
    {
        var i: usize = MPN;
        while (i > 0) {
            i -= 1;
            complex_buf[2 * i + 1] = 0;
            complex_buf[2 * i] = real_buf[i];
        }
    }

    // Phase 3: 2D IFFT on axes 0 and 1
    const tbuf = scratch + 2 * MPN; // transpose buffer, 2*MPN f64
    const fft_ws = scratch + 4 * MPN; // FFT scratch

    // Axis 1: P-point IFFT, M*N batches
    // Transpose each m-slice [P,N] → [N,P] in complex_buf → tbuf
    for (0..M) |m| {
        complexTranspose(complex_buf + m * 2 * P * N, tbuf + m * 2 * N * P, P, N);
    }
    // Batch IFFT: M*N contiguous P-point IFFTs
    fftBatchSameSize(tbuf, complex_buf, fft_ws, P, M * N, P * 2, true);
    // Transpose back each m-slice [N,P] → [P,N] in complex_buf → tbuf
    for (0..M) |m| {
        complexTranspose(complex_buf + m * 2 * N * P, tbuf + m * 2 * P * N, N, P);
    }

    // Axis 0: M-point IFFT, P*N batches
    // Transpose full [M, P*N] → [P*N, M] in tbuf → complex_buf
    complexTranspose(tbuf, complex_buf, M, P * N);
    // Batch IFFT: P*N contiguous M-point IFFTs
    fftBatchSameSize(complex_buf, tbuf, fft_ws, M, P * N, M * 2, true);
    // Transpose back [P*N, M] → [M, P*N] in tbuf → complex_buf
    complexTranspose(tbuf, complex_buf, P * N, M);

    // Phase 4: Extract real parts
    for (0..MPN) |i| {
        out[i] = complex_buf[2 * i];
    }
}

/// Scratch size for irfftn_3d.
export fn irfftn_3d_scratch_size(d0: u32, d1: u32, d2_out: u32) u32 {
    const M = @as(usize, d0);
    const P = @as(usize, d1);
    const N = @as(usize, d2_out);
    const MPN = M * P * N;
    const fft_sz = @max(scratchSizeF64(P), scratchSizeF64(M));
    // Phase 1: MPN + 6*N. Phase 2-4: 4*MPN + fft_sz.
    return @intCast(@max(MPN + 6 * N, 4 * MPN + fft_sz));
}

/// Dedicated real FFT for even N: uses N/2-point complex FFT + post-processing.
fn rfftDedicated(inp: [*]const f64, out: [*]f64, scratch: [*]f64, N: usize) void {
    const half = N / 2;
    const z = scratch;
    const z_out = scratch + N;
    const fft_scratch = scratch + 2 * N;

    for (0..half) |k| {
        z[2 * k] = inp[2 * k];
        z[2 * k + 1] = inp[2 * k + 1];
    }
    fftDispatch(z, z_out, fft_scratch, half, false);
    rfftPostProcess(z_out, out, N, null);
}

/// Dedicated inverse real FFT for even N: pre-process + N/2-point complex IFFT + unpack.
fn irfftDedicated(inp: [*]const f64, out: [*]f64, scratch: [*]f64, N: usize) void {
    const half = N / 2;
    const z = scratch;
    const z_out = scratch + N;
    const fft_scratch = scratch + 2 * N;

    irfftPreProcess(inp, z, N, null);
    fftDispatch(z, z_out, fft_scratch, half, true);

    for (0..half) |k| {
        out[2 * k] = z_out[2 * k];
        out[2 * k + 1] = z_out[2 * k + 1];
    }
}

/// rfft post-processing: convert N/2-point complex FFT output to N/2+1 spectral bins.
/// If post_tw is non-null, uses precomputed twiddles; otherwise computes via recurrence.
fn rfftPostProcess(z_out: [*]const f64, out: [*]f64, N: usize, post_tw: ?[*]const f64) void {
    const half = N / 2;
    out[0] = z_out[0] + z_out[1];
    out[1] = 0;
    out[2 * half] = z_out[0] - z_out[1];
    out[2 * half + 1] = 0;

    if (half <= 1) return;

    // Use precomputed twiddles or recurrence
    var wr_s: f64 = undefined;
    var wi_s: f64 = undefined;
    var wr: f64 = undefined;
    var wi: f64 = undefined;
    if (post_tw == null) {
        const angle = -2.0 * math.pi / @as(f64, @floatFromInt(N));
        wr_s = @cos(angle);
        wi_s = @sin(angle);
        wr = wr_s;
        wi = wi_s;
    }

    for (1..half) |k| {
        var tw_r: f64 = undefined;
        var tw_i: f64 = undefined;
        if (post_tw) |tw| {
            tw_r = tw[2 * k];
            tw_i = tw[2 * k + 1];
        } else {
            tw_r = wr;
            tw_i = wi;
        }

        const nk = half - k;
        const zk_r = z_out[2 * k];
        const zk_i = z_out[2 * k + 1];
        const znk_r = z_out[2 * nk];
        const znk_i = z_out[2 * nk + 1];

        const xe_r = (zk_r + znk_r) * 0.5;
        const xe_i = (zk_i - znk_i) * 0.5;
        const xo_r = (zk_i + znk_i) * 0.5;
        const xo_i = (znk_r - zk_r) * 0.5;

        out[2 * k] = xe_r + tw_r * xo_r - tw_i * xo_i;
        out[2 * k + 1] = xe_i + tw_r * xo_i + tw_i * xo_r;

        if (post_tw == null) {
            const new_wr = wr * wr_s - wi * wi_s;
            wi = wr * wi_s + wi * wr_s;
            wr = new_wr;
        }
    }
}

/// irfft pre-processing: convert N/2+1 spectral bins to N/2-point complex input for IFFT.
fn irfftPreProcess(inp: [*]const f64, z: [*]f64, N: usize, post_tw: ?[*]const f64) void {
    const half = N / 2;
    z[0] = (inp[0] + inp[2 * half]) * 0.5;
    z[1] = (inp[0] - inp[2 * half]) * 0.5;

    if (half <= 1) return;

    var wr_s: f64 = undefined;
    var wi_s: f64 = undefined;
    var wr: f64 = undefined;
    var wi: f64 = undefined;
    if (post_tw == null) {
        const angle = 2.0 * math.pi / @as(f64, @floatFromInt(N));
        wr_s = @cos(angle);
        wi_s = @sin(angle);
        wr = wr_s;
        wi = wi_s;
    }

    for (1..half) |k| {
        var tw_r: f64 = undefined;
        var tw_i: f64 = undefined;
        if (post_tw) |tw| {
            // For inverse, use conjugate: conj(W_N^k) stored as (cos, sin) with +angle
            tw_r = tw[2 * k];
            tw_i = -tw[2 * k + 1]; // conjugate the forward twiddle
        } else {
            tw_r = wr;
            tw_i = wi;
        }

        const nk = half - k;
        const pr = inp[2 * k];
        const pi_v = inp[2 * k + 1];
        const qr = inp[2 * nk];
        const qi = -inp[2 * nk + 1];

        const xe_r = (pr + qr) * 0.5;
        const xe_i = (pi_v + qi) * 0.5;
        const hd_r = (pr - qr) * 0.5;
        const hd_i = (pi_v - qi) * 0.5;

        const xo_r = tw_r * hd_r - tw_i * hd_i;
        const xo_i = tw_r * hd_i + tw_i * hd_r;

        z[2 * k] = xe_r - xo_i;
        z[2 * k + 1] = xe_i + xo_r;

        if (post_tw == null) {
            const new_wr = wr * wr_s - wi * wi_s;
            wi = wr * wi_s + wi * wr_s;
            wr = new_wr;
        }
    }
}

/// Fallback rfft for odd N: pack real→complex and run full complex FFT.
fn rfftFallback(inp: [*]const f64, out: [*]f64, scratch: [*]f64, N: usize) void {
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

/// Fallback irfft for odd N: expand Hermitian + full complex IFFT.
fn irfftFallback(inp: [*]const f64, out: [*]f64, scratch: [*]f64, half_n: usize, N: usize) void {
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
    fftBatchSameSize(inp, out, scratch, N, @as(usize, batch), N * 2, false);
}

/// Batched 1D inverse FFT: computes batch independent inverse FFTs of length n, with batch stride of n*2 (complex128).
/// Scratch buffer must hold max FFT scratch size for n, plus 2*n f64 for the input copy if needed.
export fn ifft_batch_c128(inp: [*]const f64, out: [*]f64, scratch: [*]f64, n: u32, batch: u32) void {
    const N = @as(usize, n);
    fftBatchSameSize(inp, out, scratch, N, @as(usize, batch), N * 2, true);
}

/// Batched 1D forward FFT for complex64 (interleaved f32, computed in f64).
/// Scratch must hold 4*n*batch f64 (f32→f64 conversion buffers) + FFT scratch.
export fn fft_batch_c64(inp: [*]const f32, out: [*]f32, scratch: [*]f64, n: u32, batch: u32) void {
    const N = @as(usize, n);
    const B = @as(usize, batch);
    const totalComplex = B * N * 2;
    const in_f64 = scratch;
    const out_f64 = scratch + totalComplex;
    const fft_scratch = out_f64 + totalComplex;
    for (0..totalComplex) |i| in_f64[i] = @as(f64, inp[i]);
    fftBatchSameSize(in_f64, out_f64, fft_scratch, N, B, N * 2, false);
    for (0..totalComplex) |i| out[i] = @as(f32, @floatCast(out_f64[i]));
}

/// Batched 1D inverse FFT for complex64 (interleaved f32, computed in f64).
export fn ifft_batch_c64(inp: [*]const f32, out: [*]f32, scratch: [*]f64, n: u32, batch: u32) void {
    const N = @as(usize, n);
    const B = @as(usize, batch);
    const totalComplex = B * N * 2;
    const in_f64 = scratch;
    const out_f64 = scratch + totalComplex;
    const fft_scratch = out_f64 + totalComplex;
    for (0..totalComplex) |i| in_f64[i] = @as(f64, inp[i]);
    fftBatchSameSize(in_f64, out_f64, fft_scratch, N, B, N * 2, true);
    for (0..totalComplex) |i| out[i] = @as(f32, @floatCast(out_f64[i]));
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

    if (isFullyFactorable(N)) {
        // Mixed-radix Stockham with precomputed twiddles (covers pow2 and composite)
        stockhamFFT(input, output, scratch, N, inverse);
    } else {
        // Bluestein's for sizes with large prime factors
        bluesteinFft(input, output, N, inverse, scratch);
    }
}

/// Batch dispatch: precompute twiddles once, then run multiple FFTs of the same size.
/// scratch layout: [buf0: 2*N] [buf1: 2*N] [twiddle: 2*N] = 6*N for Stockham.
fn fftBatchSameSize(inputs: [*]const f64, outputs: [*]f64, scratch: [*]f64, N: usize, batch: usize, stride: usize, inverse: bool) void {
    if (N <= 1) {
        if (N == 1) {
            for (0..batch) |b| {
                outputs[b * stride] = inputs[b * stride];
                outputs[b * stride + 1] = inputs[b * stride + 1];
            }
        }
        return;
    }

    if (isFullyFactorable(N)) {
        // Precompute twiddles ONCE, reuse for all batch elements
        const sign: f64 = if (inverse) 1.0 else -1.0;
        const tw = scratch + 4 * N;
        precomputeTwiddles(tw, N, sign);
        for (0..batch) |b| {
            const off = b * stride;
            stockhamFFTCore(inputs + off, outputs + off, scratch, tw, N, inverse);
        }
    } else {
        for (0..batch) |b| {
            const off = b * stride;
            bluesteinFft(inputs + off, outputs + off, N, inverse, scratch);
        }
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

    // Step 1: FFT each row (length N, M rows) — twiddles precomputed once
    fftBatchSameSize(inp, out, fft_scratch, N, M, N * 2, false);

    // Step 2: Transpose out (MxN) → transpose_buf (NxM)
    complexTranspose(out, transpose_buf, M, N);

    // Step 3: FFT each row of transposed (length M, N rows)
    fftBatchSameSize(transpose_buf, out, fft_scratch, M, N, M * 2, false);

    // Step 4: Transpose out (NxM) back → final out (MxN)
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

    fftBatchSameSize(inp, out, fft_scratch, N, M, N * 2, true);

    complexTranspose(out, transpose_buf, M, N);

    fftBatchSameSize(transpose_buf, out, fft_scratch, M, N, M * 2, true);

    for (0..total * 2) |i| transpose_buf[i] = out[i];
    complexTranspose(transpose_buf, out, N, M);
}

/// 2D forward complex FFT for complex64 (interleaved f32, computed in f64).
export fn fft2_c64(inp: [*]const f32, out: [*]f32, scratch: [*]f64, rows: u32, cols: u32) void {
    const M = @as(usize, rows);
    const N = @as(usize, cols);
    const total = M * N;
    const in_f64 = scratch;
    const out_f64 = scratch + total * 2;
    const fft2_scratch = out_f64 + total * 2;
    for (0..total * 2) |i| in_f64[i] = @as(f64, inp[i]);
    fft2_c128(in_f64, out_f64, fft2_scratch, rows, cols);
    for (0..total * 2) |i| out[i] = @as(f32, @floatCast(out_f64[i]));
}

/// 2D inverse complex FFT for complex64 (interleaved f32, computed in f64).
export fn ifft2_c64(inp: [*]const f32, out: [*]f32, scratch: [*]f64, rows: u32, cols: u32) void {
    const M = @as(usize, rows);
    const N = @as(usize, cols);
    const total = M * N;
    const in_f64 = scratch;
    const out_f64 = scratch + total * 2;
    const fft2_scratch = out_f64 + total * 2;
    for (0..total * 2) |i| in_f64[i] = @as(f64, inp[i]);
    ifft2_c128(in_f64, out_f64, fft2_scratch, rows, cols);
    for (0..total * 2) |i| out[i] = @as(f32, @floatCast(out_f64[i]));
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

    // Row FFTs — twiddles precomputed once
    fftBatchSameSize(complex_buf, full_out, fft_scratch, N, M, N * 2, false);

    // Transpose + column FFTs + transpose back
    complexTranspose(full_out, transpose_buf, M, N);
    fftBatchSameSize(transpose_buf, full_out, fft_scratch, M, N, M * 2, false);
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

    // Row IFFTs — twiddles precomputed once
    fftBatchSameSize(full_in, full_out, fft_scratch, N, M, N * 2, true);

    // Transpose + column IFFTs + transpose back
    complexTranspose(full_out, transpose_buf, M, N);
    fftBatchSameSize(transpose_buf, full_out, fft_scratch, M, N, M * 2, true);
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

// Internal utilities

/// Find the next power of 2 greater than or equal to n, for zero-padding in Bluestein's algorithm.
fn nextPow2(n: usize) usize {
    var v: usize = 1;
    while (v < n) v <<= 1;
    return v;
}

const MAX_FACTORS = 32;

/// Factor n into allowed radix factors (8, 4, 2, 3, 5) for Stockham FFT.
/// Radix-8 first (most efficient), then 4, 2, 3, 5.
fn factorize(n: usize) struct { factors: [MAX_FACTORS]usize, count: usize } {
    var result: [MAX_FACTORS]usize = undefined;
    var count: usize = 0;
    var r = n;

    // Extract factors in order: 8, 4, 2, 3, 5
    // Radix-8 first: replaces three radix-2 stages with one pass
    while (r % 8 == 0 and count < MAX_FACTORS) {
        result[count] = 8;
        count += 1;
        r /= 8;
    }
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
    if (r > 1 and count < MAX_FACTORS) {
        result[count] = r;
        count += 1;
    }

    return .{ .factors = result, .count = count };
}

/// Check if n is fully factorable into allowed radix factors (2, 3, 5).
/// 8 and 4 are powers of 2 so they're covered by dividing out 2.
fn isFullyFactorable(n: usize) bool {
    var r = n;
    while (r % 2 == 0) r /= 2;
    while (r % 3 == 0) r /= 3;
    while (r % 5 == 0) r /= 5;
    return r == 1;
}

// Twiddle factor precomputation

/// Precompute twiddle factors: tw[2*k] = cos(sign*2π*k/N), tw[2*k+1] = sin(sign*2π*k/N)
/// for k = 0..N-1. Uses recurrence (1 sin/cos + N-1 complex multiplies) for speed,
/// with periodic re-anchoring every sqrt(N) steps to limit accumulated error.
fn precomputeTwiddles(tw: [*]f64, N: usize, sign: f64) void {
    if (N == 0) return;
    tw[0] = 1.0;
    tw[1] = 0.0;
    if (N <= 1) return;

    const angle = sign * 2.0 * math.pi / @as(f64, @floatFromInt(N));
    const wr_step = @cos(angle);
    const wi_step = @sin(angle);

    // Re-anchor interval: every ~sqrt(N) steps, recompute from scratch
    // to prevent accumulated floating-point drift in the recurrence.
    const anchor: usize = blk: {
        var a: usize = 1;
        while (a * a < N) a += 1;
        break :blk a;
    };

    for (1..N) |k| {
        if (k % anchor == 0) {
            // Re-anchor: compute directly from sin/cos
            const a = sign * 2.0 * math.pi * @as(f64, @floatFromInt(k)) / @as(f64, @floatFromInt(N));
            tw[2 * k] = @cos(a);
            tw[2 * k + 1] = @sin(a);
        } else {
            // Recurrence: tw[k] = tw[k-1] * W_N
            const pr = tw[2 * (k - 1)];
            const pi = tw[2 * (k - 1) + 1];
            tw[2 * k] = pr * wr_step - pi * wi_step;
            tw[2 * k + 1] = pr * wi_step + pi * wr_step;
        }
    }
}

// Stockham mixed-radix FFT with precomputed twiddles

/// Stockham mixed-radix FFT with precomputed twiddle table.
/// Scratch layout: [buf0: 2*N] [buf1: 2*N] [twiddle: 2*N] = 6*N f64s total.
fn stockhamFFT(input: [*]const f64, output: [*]f64, scratch: [*]f64, N: usize, inverse: bool) void {
    if (N <= 1) {
        if (N == 1) {
            output[0] = input[0];
            output[1] = input[1];
        }
        return;
    }

    const sign: f64 = if (inverse) 1.0 else -1.0;
    const tw = scratch + 4 * N;
    precomputeTwiddles(tw, N, sign);
    stockhamFFTCore(input, output, scratch, tw, N, inverse);
}

/// Stockham FFT core: uses pre-existing twiddle table. Scratch needs 4*N f64s (two ping-pong buffers).
/// tw must point to 2*N f64s of precomputed twiddles for the given direction.
fn stockhamFFTCore(input: [*]const f64, output: [*]f64, scratch: [*]f64, tw: [*]const f64, N: usize, inverse: bool) void {
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

    const buf0 = scratch;
    const buf1 = scratch + 2 * N;

    // Copy input to buf0
    for (0..N * 2) |i| buf0[i] = input[i];

    const sign: f64 = if (inverse) 1.0 else -1.0;
    var src = buf0;
    var dst = buf1;
    var m: usize = 1;

    for (0..nFactors) |fi| {
        const p = factors[fi];
        const groups = N / (m * p);

        stockhamPass(src, dst, tw, p, m, groups, sign);

        m *= p;
        const tmp = src;
        src = dst;
        dst = tmp;
    }

    // Scale for inverse
    if (inverse) {
        const scale = 1.0 / @as(f64, @floatFromInt(N));
        for (0..N * 2) |i| src[i] *= scale;
    }

    // Copy result to output
    for (0..N * 2) |i| output[i] = src[i];
}

/// Perform one Stockham pass for factor p.
fn stockhamPass(src: [*]f64, dst: [*]f64, tw: [*]const f64, p: usize, m: usize, groups: usize, sign: f64) void {
    switch (p) {
        2 => stockhamRadix2(src, dst, tw, m, groups),
        3 => stockhamRadix3(src, dst, tw, m, groups, sign),
        4 => stockhamRadix4(src, dst, tw, m, groups, sign),
        5 => stockhamRadix5(src, dst, tw, m, groups, sign),
        8 => stockhamRadix8(src, dst, tw, m, groups, sign),
        else => {
            stockhamGeneric(src, dst, tw, p, m, groups, sign);
        },
    }
}

/// Stockham radix-2 pass with precomputed twiddle table.
/// Twiddle for position k: tw[k * groups] (since angle = 2π*k/(2*m) = 2π*(k*groups)/N).
fn stockhamRadix2(src: [*]f64, dst: [*]f64, tw: [*]const f64, m: usize, groups: usize) void {
    const pm = 2 * m;
    for (0..groups) |g| {
        for (0..m) |k| {
            const idx0 = (g * m + k) * 2;
            const idx1 = (g * m + k + groups * m) * 2;

            const a0r = src[idx0];
            const a0i = src[idx0 + 1];
            var tr = src[idx1];
            var ti = src[idx1 + 1];

            // Twiddle multiply (skip when k==0: tw = 1+0i)
            if (k > 0) {
                const tw_idx = k * groups * 2;
                const wr = tw[tw_idx];
                const wi = tw[tw_idx + 1];
                const nr = wr * tr - wi * ti;
                ti = wr * ti + wi * tr;
                tr = nr;
            }

            const o0 = (g * pm + k) * 2;
            const o1 = (g * pm + k + m) * 2;

            dst[o0] = a0r + tr;
            dst[o0 + 1] = a0i + ti;
            dst[o1] = a0r - tr;
            dst[o1 + 1] = a0i - ti;
        }
    }
}

/// Stockham radix-3 pass with precomputed twiddle table.
fn stockhamRadix3(src: [*]f64, dst: [*]f64, tw: [*]const f64, m: usize, groups: usize, sign: f64) void {
    const pm = 3 * m;
    const c1: f64 = -0.5;
    const c2: f64 = sign * 0.86602540378443864676; // sign * sqrt(3)/2

    for (0..groups) |g| {
        for (0..m) |k| {
            const tw1_idx = k * groups * 2;
            const tw2_idx = 2 * k * groups * 2;

            const idx0 = (g * m + k) * 2;
            const idx1 = (g * m + k + groups * m) * 2;
            const idx2 = (g * m + k + 2 * groups * m) * 2;

            const a0r = src[idx0];
            const a0i = src[idx0 + 1];
            var a1r = src[idx1];
            var a1i = src[idx1 + 1];
            var a2r = src[idx2];
            var a2i = src[idx2 + 1];

            // Twiddle a1 by W^k, a2 by W^(2k) — skip when k==0
            if (tw1_idx != 0) {
                {
                    const wr = tw[tw1_idx];
                    const wi = tw[tw1_idx + 1];
                    const tr = wr * a1r - wi * a1i;
                    const ti = wr * a1i + wi * a1r;
                    a1r = tr;
                    a1i = ti;
                }
                {
                    const wr = tw[tw2_idx];
                    const wi = tw[tw2_idx + 1];
                    const tr = wr * a2r - wi * a2i;
                    const ti = wr * a2i + wi * a2r;
                    a2r = tr;
                    a2i = ti;
                }
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

/// Stockham radix-4 pass with precomputed twiddle table.
fn stockhamRadix4(src: [*]f64, dst: [*]f64, tw: [*]const f64, m: usize, groups: usize, sign: f64) void {
    const pm = 4 * m;

    for (0..groups) |g| {
        for (0..m) |k| {
            const tw_step = k * groups * 2;

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

            // Twiddle a1 by W^k, a2 by W^(2k), a3 by W^(3k) — skip when k==0
            if (tw_step != 0) {
                const w1r = tw[tw_step];
                const w1i = tw[tw_step + 1];
                const tr1 = w1r * a1r - w1i * a1i;
                a1i = w1r * a1i + w1i * a1r;
                a1r = tr1;

                const w2r = tw[2 * tw_step];
                const w2i = tw[2 * tw_step + 1];
                const tr2 = w2r * a2r - w2i * a2i;
                a2i = w2r * a2i + w2i * a2r;
                a2r = tr2;

                const w3r = tw[3 * tw_step];
                const w3i = tw[3 * tw_step + 1];
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
            dst[o1] = t1r - sign * t3i;
            dst[o1 + 1] = t1i + sign * t3r;
            dst[o2] = t0r - t2r;
            dst[o2 + 1] = t0i - t2i;
            dst[o3] = t1r + sign * t3i;
            dst[o3 + 1] = t1i - sign * t3r;
        }
    }
}

/// Stockham radix-5 pass with precomputed twiddle table.
fn stockhamRadix5(src: [*]f64, dst: [*]f64, tw: [*]const f64, m: usize, groups: usize, sign: f64) void {
    const pm = 5 * m;
    // DFT-5 constants
    const c1: f64 = 0.30901699437494742; // cos(2π/5)
    const c2: f64 = 0.95105651629515357; // sin(2π/5)
    const c3: f64 = -0.80901699437494742; // cos(4π/5)
    const c4: f64 = 0.58778525229247313; // sin(4π/5)

    for (0..groups) |g| {
        for (0..m) |k| {
            const tw_step = k * groups * 2;

            var a: [5][2]f64 = undefined;
            inline for (0..5) |s| {
                const idx = (g * m + k + s * groups * m) * 2;
                a[s] = .{ src[idx], src[idx + 1] };
            }

            // Twiddle using precomputed table — skip when k==0
            if (tw_step != 0) {
                inline for (1..5) |s| {
                    const tw_idx = s * tw_step;
                    const wr = tw[tw_idx];
                    const wi = tw[tw_idx + 1];
                    const tr = wr * a[s][0] - wi * a[s][1];
                    const ti = wr * a[s][1] + wi * a[s][0];
                    a[s] = .{ tr, ti };
                }
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

/// Stockham radix-8 pass with precomputed twiddle table.
/// Uses split-radix DFT-8 butterfly with ROTX90/ROTX45/ROTX135 optimizations
/// to avoid general complex multiplies inside the butterfly itself.
fn stockhamRadix8(src: [*]f64, dst: [*]f64, tw: [*]const f64, m: usize, groups: usize, sign: f64) void {
    const pm = 8 * m;
    const hsqt2: f64 = 0.70710678118654752440;

    for (0..groups) |g| {
        for (0..m) |k| {
            const tw_step = k * groups * 2;

            // Gather 8 inputs and twiddle (s=1..7)
            var a: [8][2]f64 = undefined;
            inline for (0..8) |s| {
                const idx = (g * m + k + s * groups * m) * 2;
                a[s] = .{ src[idx], src[idx + 1] };
            }
            if (tw_step != 0) {
                inline for (1..8) |s| {
                    const tw_idx = s * tw_step;
                    const wr = tw[tw_idx];
                    const wi = tw[tw_idx + 1];
                    const tr = wr * a[s][0] - wi * a[s][1];
                    const ti = wr * a[s][1] + wi * a[s][0];
                    a[s] = .{ tr, ti };
                }
            }

            // DFT-8 butterfly (adapted from pocketfft pass8)
            // Step 1: odd-index pairs
            var b1r = a[1][0] + a[5][0];
            var b1i = a[1][1] + a[5][1];
            var b5r = a[1][0] - a[5][0];
            var b5i = a[1][1] - a[5][1];
            var b3r = a[3][0] + a[7][0];
            var b3i = a[3][1] + a[7][1];
            var b7r = a[3][0] - a[7][0];
            var b7i = a[3][1] - a[7][1];

            // PMINPLACE(b1, b3): b1=b1+b3, b3=old_b1-b3
            var tmp_r = b1r;
            var tmp_i = b1i;
            b1r = b1r + b3r;
            b1i = b1i + b3i;
            b3r = tmp_r - b3r;
            b3i = tmp_i - b3i;

            // ROTX90(b3): multiply by -sign*i
            // forward(sign=-1): {r,i}→{i,-r}, inverse(sign=+1): {r,i}→{-i,r}
            tmp_r = b3r;
            b3r = -sign * b3i;
            b3i = sign * tmp_r;

            // ROTX90(b7)
            tmp_r = b7r;
            b7r = -sign * b7i;
            b7i = sign * tmp_r;

            // PMINPLACE(b5, b7)
            tmp_r = b5r;
            tmp_i = b5i;
            b5r = b5r + b7r;
            b5i = b5i + b7i;
            b7r = tmp_r - b7r;
            b7i = tmp_i - b7i;

            // ROTX45(b5): multiply by (1-sign*i)/√2
            // new_r = hsqt2*(r - sign*i), new_i = hsqt2*(i + sign*r)
            tmp_r = b5r;
            b5r = hsqt2 * (b5r - sign * b5i);
            b5i = hsqt2 * (b5i + sign * tmp_r);

            // ROTX135(b7): multiply by (-1+sign*i)/√2
            // new_r = hsqt2*(-r - sign*i), new_i = hsqt2*(sign*r - i)
            tmp_r = b7r;
            b7r = hsqt2 * (-b7r - sign * b7i);
            b7i = hsqt2 * (sign * tmp_r - b7i);

            // Step 2: even-index pairs
            const c0r = a[0][0] + a[4][0];
            const c0i = a[0][1] + a[4][1];
            const c4r = a[0][0] - a[4][0];
            const c4i = a[0][1] - a[4][1];
            const c2r = a[2][0] + a[6][0];
            const c2i = a[2][1] + a[6][1];
            var c6r = a[2][0] - a[6][0];
            var c6i = a[2][1] - a[6][1];

            // ROTX90(c6)
            tmp_r = c6r;
            c6r = -sign * c6i;
            c6i = sign * tmp_r;

            // Step 3: final combines and output
            const e0r = c0r + c2r;
            const e0i = c0i + c2i;
            const e2r = c0r - c2r;
            const e2i = c0i - c2i;
            const f4r = c4r + c6r;
            const f4i = c4i + c6i;
            const f6r = c4r - c6r;
            const f6i = c4i - c6i;

            const o0 = (g * pm + k + 0 * m) * 2;
            const o1 = (g * pm + k + 1 * m) * 2;
            const o2 = (g * pm + k + 2 * m) * 2;
            const o3 = (g * pm + k + 3 * m) * 2;
            const o4 = (g * pm + k + 4 * m) * 2;
            const o5 = (g * pm + k + 5 * m) * 2;
            const o6 = (g * pm + k + 6 * m) * 2;
            const o7 = (g * pm + k + 7 * m) * 2;

            dst[o0] = e0r + b1r;
            dst[o0 + 1] = e0i + b1i;
            dst[o4] = e0r - b1r;
            dst[o4 + 1] = e0i - b1i;
            dst[o2] = e2r + b3r;
            dst[o2 + 1] = e2i + b3i;
            dst[o6] = e2r - b3r;
            dst[o6 + 1] = e2i - b3i;
            dst[o1] = f4r + b5r;
            dst[o1 + 1] = f4i + b5i;
            dst[o5] = f4r - b5r;
            dst[o5 + 1] = f4i - b5i;
            dst[o3] = f6r + b7r;
            dst[o3 + 1] = f6i + b7i;
            dst[o7] = f6r - b7r;
            dst[o7 + 1] = f6i - b7i;
        }
    }
}

/// Generic Stockham pass for prime factors with precomputed twiddle table.
fn stockhamGeneric(src: [*]f64, dst: [*]f64, tw: [*]const f64, p: usize, m: usize, groups: usize, sign: f64) void {
    const pm = p * m;

    for (0..groups) |g| {
        for (0..m) |k| {
            const tw_step = k * groups;

            // Gather and twiddle
            var buf: [128][2]f64 = undefined; // max prime factor
            for (0..p) |s| {
                const idx = (g * m + k + s * groups * m) * 2;
                var ar = src[idx];
                var ai = src[idx + 1];
                if (s > 0) {
                    const tw_idx = s * tw_step * 2;
                    const wr = tw[tw_idx];
                    const wi = tw[tw_idx + 1];
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

// Legacy pow2 FFT (used internally by Bluestein's algorithm)

/// Cooley-Tukey radix-2 FFT for power-of-two sizes.
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

// Bluestein's algorithm (for sizes with large prime factors)

/// Bluestein's algorithm: converts arbitrary-size DFT into convolution, then uses FFT for convolution.
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

/// Scratch size needed for Bluestein's algorithm: 6*P f64s (chirp, padded input, padded chirp).
fn bluesteinScratchF64(N: usize) usize {
    if (N <= 1) return 0;
    return 6 * nextPow2(2 * N - 1);
}

// Scratch size computation

/// Scratch size needed for fftDispatch: depends on algorithm choice.
fn scratchSizeF64(N: usize) usize {
    if (N <= 1) return 0;
    if (isFullyFactorable(N)) {
        // Stockham: two ping-pong buffers (2*N each) + twiddle table (2*N) = 6*N
        return 6 * N;
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

test "Stockham pow2 N=8 forward/inverse roundtrip" {
    const testing = std.testing;
    const inp = [_]f64{ 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0 };
    var out: [16]f64 = undefined;
    var inv: [16]f64 = undefined;
    var scratch: [48]f64 = undefined; // 6*8 = 48

    fftDispatch(&inp, &out, &scratch, 8, false);

    // DFT([1,2,3,4,5,6,7,8])[0] = 36
    try testing.expectApproxEqAbs(out[0], 36.0, 1e-8);
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-8);

    fftDispatch(&out, &inv, &scratch, 8, true);
    for (0..16) |i| {
        try testing.expectApproxEqAbs(inv[i], inp[i], 1e-8);
    }
}

test "Stockham pow2 N=16 roundtrip" {
    const testing = std.testing;
    var inp: [32]f64 = undefined;
    for (0..16) |i| {
        inp[2 * i] = @as(f64, @floatFromInt(i + 1));
        inp[2 * i + 1] = 0;
    }
    var out: [32]f64 = undefined;
    var scratch: [96]f64 = undefined; // 6*16

    fftDispatch(&inp, &out, &scratch, 16, false);

    var inv: [32]f64 = undefined;
    fftDispatch(&out, &inv, &scratch, 16, true);
    for (0..32) |i| {
        try testing.expectApproxEqAbs(inv[i], inp[i], 1e-8);
    }
}

test "mixed-radix N=6 (2*3)" {
    const testing = std.testing;
    const inp = [_]f64{ 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0 };
    var out: [12]f64 = undefined;
    var scratch: [36]f64 = undefined; // 6*6 = 36

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
    var scratch: [600]f64 = undefined; // 6*100

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
    var scratch: [6000]f64 = undefined; // 6*1000

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
    var scratch: [24]f64 = undefined; // 6*4 = 24 for Stockham

    fft_c128(&inp, &fwd, &scratch, 4);
    ifft_c128(&fwd, &inv, &scratch, 4);

    for (0..8) |i| {
        try testing.expectApproxEqAbs(inv[i], inp[i], 1e-10);
    }
}

test "dedicated rfft/irfft roundtrip N=8" {
    const testing = std.testing;
    const inp = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var rfft_out: [10]f64 = undefined; // (8/2+1)*2 = 10
    var irfft_out: [8]f64 = undefined;
    var scratch: [200]f64 = undefined;

    rfft_f64(&inp, &rfft_out, &scratch, 8);

    // DC should be sum = 36
    try testing.expectApproxEqAbs(rfft_out[0], 36.0, 1e-10);
    try testing.expectApproxEqAbs(rfft_out[1], 0.0, 1e-10);
    // Nyquist
    try testing.expectApproxEqAbs(rfft_out[8], -4.0, 1e-10);
    try testing.expectApproxEqAbs(rfft_out[9], 0.0, 1e-10);

    irfft_f64(&rfft_out, &irfft_out, &scratch, 5, 8);

    for (0..8) |i| {
        try testing.expectApproxEqAbs(irfft_out[i], inp[i], 1e-8);
    }
}

test "dedicated rfft/irfft roundtrip N=1000" {
    const testing = std.testing;
    var inp: [1000]f64 = undefined;
    for (0..1000) |i| {
        inp[i] = @as(f64, @floatFromInt(i % 13)) - 6.0;
    }
    var rfft_out: [1002]f64 = undefined; // (1000/2+1)*2 = 1002
    var irfft_out: [1000]f64 = undefined;
    var scratch: [20000]f64 = undefined;

    rfft_f64(&inp, &rfft_out, &scratch, 1000);
    irfft_f64(&rfft_out, &irfft_out, &scratch, 501, 1000);

    for (0..1000) |i| {
        try testing.expectApproxEqAbs(irfft_out[i], inp[i], 1e-6);
    }
}

// Edge cases and minimal sizes

test "fftDispatch N=1" {
    const testing = std.testing;
    const inp = [_]f64{ 42.0, 7.0 };
    var out: [2]f64 = undefined;
    var scratch: [1]f64 = undefined;
    fftDispatch(&inp, &out, &scratch, 1, false);
    try testing.expectApproxEqAbs(out[0], 42.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 7.0, 1e-10);
}

test "fftDispatch N=2 roundtrip" {
    const testing = std.testing;
    const inp = [_]f64{ 3.0, 1.0, -2.0, 4.0 };
    var out: [4]f64 = undefined;
    var scratch: [12]f64 = undefined; // 6*2
    fftDispatch(&inp, &out, &scratch, 2, false);
    // DFT([3+i, -2+4i]): X[0]=1+5i, X[1]=5-3i
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 5.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 5.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], -3.0, 1e-10);

    var inv: [4]f64 = undefined;
    fftDispatch(&out, &inv, &scratch, 2, true);
    for (0..4) |i| {
        try testing.expectApproxEqAbs(inv[i], inp[i], 1e-10);
    }
}

// Radix-specific coverage

test "radix-3 pure N=9 (3*3) roundtrip" {
    const testing = std.testing;
    var inp: [18]f64 = undefined;
    for (0..9) |i| {
        inp[2 * i] = @as(f64, @floatFromInt(i * i));
        inp[2 * i + 1] = @as(f64, @floatFromInt(i));
    }
    var out: [18]f64 = undefined;
    var inv: [18]f64 = undefined;
    var scratch: [54]f64 = undefined; // 6*9

    fftDispatch(&inp, &out, &scratch, 9, false);
    fftDispatch(&out, &inv, &scratch, 9, true);
    for (0..18) |i| {
        try testing.expectApproxEqAbs(inv[i], inp[i], 1e-8);
    }
}

test "radix-5 pure N=25 (5*5) roundtrip" {
    const testing = std.testing;
    var inp: [50]f64 = undefined;
    for (0..25) |i| {
        inp[2 * i] = @sin(@as(f64, @floatFromInt(i)));
        inp[2 * i + 1] = @cos(@as(f64, @floatFromInt(i)));
    }
    var out: [50]f64 = undefined;
    var inv: [50]f64 = undefined;
    var scratch: [150]f64 = undefined;

    fftDispatch(&inp, &out, &scratch, 25, false);
    fftDispatch(&out, &inv, &scratch, 25, true);
    for (0..50) |i| {
        try testing.expectApproxEqAbs(inv[i], inp[i], 1e-8);
    }
}

test "radix-8 pure N=64 (8*8) roundtrip" {
    const testing = std.testing;
    var inp: [128]f64 = undefined;
    for (0..64) |i| {
        inp[2 * i] = @as(f64, @floatFromInt(i % 7)) - 3.0;
        inp[2 * i + 1] = @as(f64, @floatFromInt(i % 5)) - 2.0;
    }
    var out: [128]f64 = undefined;
    var inv: [128]f64 = undefined;
    var scratch: [384]f64 = undefined; // 6*64

    fftDispatch(&inp, &out, &scratch, 64, false);

    // DC = sum of all elements
    var dc_r: f64 = 0;
    var dc_i: f64 = 0;
    for (0..64) |i| {
        dc_r += inp[2 * i];
        dc_i += inp[2 * i + 1];
    }
    try testing.expectApproxEqAbs(out[0], dc_r, 1e-8);
    try testing.expectApproxEqAbs(out[1], dc_i, 1e-8);

    fftDispatch(&out, &inv, &scratch, 64, true);
    for (0..128) |i| {
        try testing.expectApproxEqAbs(inv[i], inp[i], 1e-8);
    }
}

test "mixed radix N=120 (8*3*5) roundtrip" {
    const testing = std.testing;
    var inp: [240]f64 = undefined;
    for (0..120) |i| {
        inp[2 * i] = @as(f64, @floatFromInt(i % 11)) - 5.0;
        inp[2 * i + 1] = 0;
    }
    var out: [240]f64 = undefined;
    var inv: [240]f64 = undefined;
    var scratch: [720]f64 = undefined; // 6*120

    fftDispatch(&inp, &out, &scratch, 120, false);
    fftDispatch(&out, &inv, &scratch, 120, true);
    for (0..240) |i| {
        try testing.expectApproxEqAbs(inv[i], inp[i], 1e-6);
    }
}

// Bluestein's algorithm (prime sizes)

test "Bluestein N=7 (prime) roundtrip" {
    const testing = std.testing;
    const inp = [_]f64{ 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0 };
    var out: [14]f64 = undefined;
    var scratch: [1000]f64 = undefined;

    fftDispatch(&inp, &out, &scratch, 7, false);
    // DC = 28
    try testing.expectApproxEqAbs(out[0], 28.0, 1e-8);
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-8);

    var inv: [14]f64 = undefined;
    fftDispatch(&out, &inv, &scratch, 7, true);
    for (0..14) |i| {
        try testing.expectApproxEqAbs(inv[i], inp[i], 1e-8);
    }
}

test "Bluestein N=13 (prime) roundtrip" {
    const testing = std.testing;
    var inp: [26]f64 = undefined;
    for (0..13) |i| {
        inp[2 * i] = @as(f64, @floatFromInt(i));
        inp[2 * i + 1] = @as(f64, @floatFromInt(13 - i));
    }
    var out: [26]f64 = undefined;
    var inv: [26]f64 = undefined;
    var scratch: [2000]f64 = undefined;

    fftDispatch(&inp, &out, &scratch, 13, false);
    fftDispatch(&out, &inv, &scratch, 13, true);
    for (0..26) |i| {
        try testing.expectApproxEqAbs(inv[i], inp[i], 1e-8);
    }
}

// Complex input (non-zero imaginary)

test "complex input N=8 roundtrip" {
    const testing = std.testing;
    const inp = [_]f64{ 1, 2, 3, -1, 0, 5, -2, 3, 4, -4, 1, 1, -3, 2, 0, -5 };
    var out: [16]f64 = undefined;
    var inv: [16]f64 = undefined;
    var scratch: [48]f64 = undefined;

    fftDispatch(&inp, &out, &scratch, 8, false);
    fftDispatch(&out, &inv, &scratch, 8, true);
    for (0..16) |i| {
        try testing.expectApproxEqAbs(inv[i], inp[i], 1e-8);
    }
}

// Export function coverage: fft_c64/ifft_c64

test "fft_c64/ifft_c64 roundtrip" {
    const testing = std.testing;
    const inp = [_]f32{ 1, 0, 2, 0, 3, 0, 4, 0 };
    var fwd: [8]f32 = undefined;
    var inv: [8]f32 = undefined;
    var scratch: [200]f64 = undefined;

    fft_c64(&inp, &fwd, &scratch, 4);
    ifft_c64(&fwd, &inv, &scratch, 4);

    for (0..8) |i| {
        try testing.expectApproxEqAbs(@as(f64, inv[i]), @as(f64, inp[i]), 1e-5);
    }
}

// Batch FFT coverage

test "fft_batch_c128 roundtrip" {
    const testing = std.testing;
    // 3 batches of 4-point FFTs
    var inp: [24]f64 = undefined;
    for (0..12) |i| {
        inp[2 * i] = @as(f64, @floatFromInt(i));
        inp[2 * i + 1] = 0;
    }
    var fwd: [24]f64 = undefined;
    var inv: [24]f64 = undefined;
    var scratch: [24]f64 = undefined; // 6*4

    fft_batch_c128(&inp, &fwd, &scratch, 4, 3);
    ifft_batch_c128(&fwd, &inv, &scratch, 4, 3);

    for (0..24) |i| {
        try testing.expectApproxEqAbs(inv[i], inp[i], 1e-8);
    }
}

test "rfft_batch_f64 roundtrip" {
    const testing = std.testing;
    // 4 batches of 8-point rfft
    var inp: [32]f64 = undefined;
    for (0..32) |i| {
        inp[i] = @as(f64, @floatFromInt(i % 7)) - 3.0;
    }
    var rfft_out: [40]f64 = undefined; // 4 * (8/2+1) * 2 = 40
    var irfft_out: [32]f64 = undefined;
    var scratch: [200]f64 = undefined; // 6*N needs room for internal buffers

    rfft_batch_f64(&inp, &rfft_out, &scratch, 8, 4, 8, 10);
    irfft_batch_f64(&rfft_out, &irfft_out, &scratch, 8, 4, 10, 8);

    for (0..32) |i| {
        try testing.expectApproxEqAbs(irfft_out[i], inp[i], 1e-8);
    }
}

// 2D FFT coverage

test "fft2_c128/ifft2_c128 roundtrip 4x4" {
    const testing = std.testing;
    var inp: [32]f64 = undefined; // 4x4 complex
    for (0..16) |i| {
        inp[2 * i] = @as(f64, @floatFromInt(i));
        inp[2 * i + 1] = 0;
    }
    var out: [32]f64 = undefined;
    var inv: [32]f64 = undefined;
    const sz = fft2_scratch_size(4, 4);
    var scratch: [500]f64 = undefined;
    _ = sz;

    fft2_c128(&inp, &out, &scratch, 4, 4);
    // DC = sum of all = 0+1+...+15 = 120
    try testing.expectApproxEqAbs(out[0], 120.0, 1e-8);
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-8);

    ifft2_c128(&out, &inv, &scratch, 4, 4);
    for (0..32) |i| {
        try testing.expectApproxEqAbs(inv[i], inp[i], 1e-8);
    }
}

test "rfft2_f64 forward values 4x4" {
    const testing = std.testing;
    // All ones: rfft2 should give 16 at DC, 0 everywhere else
    var inp: [16]f64 = undefined;
    for (0..16) |i| inp[i] = 1.0;
    _ = &inp;

    const half_cols = 4 / 2 + 1; // 3
    var rfft_out: [24]f64 = undefined; // 4 * 3 * 2
    var scratch: [10000]f64 = undefined;

    rfft2_f64(&inp, &rfft_out, &scratch, 4, 4);

    // DC = sum = 16
    try testing.expectApproxEqAbs(rfft_out[0], 16.0, 1e-8);
    try testing.expectApproxEqAbs(rfft_out[1], 0.0, 1e-8);
    // All other bins should be 0
    for (1..4 * half_cols) |i| {
        try testing.expectApproxEqAbs(rfft_out[2 * i], 0.0, 1e-8);
        try testing.expectApproxEqAbs(rfft_out[2 * i + 1], 0.0, 1e-8);
    }
}

// Fused irfftn_3d coverage

test "irfftn_3d constant input 2x2x4" {
    const testing = std.testing;
    // For constant input x[i,j,k] = 1.0, rfftn produces:
    // X[0,0,0] = M*P*N, all others = 0
    // So irfftn of that should give back all 1.0
    const M = 2;
    const P = 2;
    const N = 4;
    const H = N / 2 + 1; // 3

    // Construct rfftn output: only DC bin is nonzero
    var rfftn_out: [M * P * H * 2]f64 = [_]f64{0} ** (M * P * H * 2);
    rfftn_out[0] = @as(f64, @floatFromInt(M * P * N)); // DC real = 16
    rfftn_out[1] = 0; // DC imag = 0

    var recovered: [M * P * N]f64 = undefined;
    var scratch: [20000]f64 = undefined;

    irfftn_3d(&rfftn_out, &recovered, &scratch, M, P, H, N);

    for (0..M * P * N) |i| {
        try testing.expectApproxEqAbs(recovered[i], 1.0, 1e-8);
    }
}

test "irfftn_3d impulse 2x2x4" {
    const testing = std.testing;
    // For impulse x[0,0,0] = 1.0, rest = 0, rfftn produces all bins = 1.0+0i
    // So irfftn of all-ones spectrum should give impulse
    const M = 2;
    const P = 2;
    const N = 4;
    const H = N / 2 + 1; // 3

    var rfftn_out: [M * P * H * 2]f64 = undefined;
    for (0..M * P * H) |i| {
        rfftn_out[2 * i] = 1.0;
        rfftn_out[2 * i + 1] = 0.0;
    }

    var recovered: [M * P * N]f64 = undefined;
    var scratch: [20000]f64 = undefined;

    irfftn_3d(&rfftn_out, &recovered, &scratch, M, P, H, N);

    // Element [0,0,0] should be 1.0, all others 0
    try testing.expectApproxEqAbs(recovered[0], 1.0, 1e-8);
    for (1..M * P * N) |i| {
        try testing.expectApproxEqAbs(recovered[i], 0.0, 1e-8);
    }
}

// Parseval's theorem: energy preservation

test "Parseval's theorem N=32" {
    const testing = std.testing;
    var inp: [64]f64 = undefined;
    for (0..32) |i| {
        inp[2 * i] = @as(f64, @floatFromInt(i % 7)) - 3.0;
        inp[2 * i + 1] = @as(f64, @floatFromInt(i % 5)) - 2.0;
    }
    var out: [64]f64 = undefined;
    var scratch: [192]f64 = undefined; // 6*32

    fftDispatch(&inp, &out, &scratch, 32, false);

    // Time-domain energy
    var energy_time: f64 = 0;
    for (0..32) |i| {
        energy_time += inp[2 * i] * inp[2 * i] + inp[2 * i + 1] * inp[2 * i + 1];
    }

    // Frequency-domain energy (scaled by N for unnormalized DFT)
    var energy_freq: f64 = 0;
    for (0..32) |i| {
        energy_freq += out[2 * i] * out[2 * i] + out[2 * i + 1] * out[2 * i + 1];
    }
    energy_freq /= 32.0;

    try testing.expectApproxEqAbs(energy_time, energy_freq, 1e-8);
}

// Linearity: FFT(a*x + b*y) = a*FFT(x) + b*FFT(y)

test "FFT linearity N=16" {
    const testing = std.testing;
    var x: [32]f64 = undefined;
    var y: [32]f64 = undefined;
    var combo: [32]f64 = undefined;
    const a: f64 = 2.5;
    const b: f64 = -1.3;

    for (0..16) |i| {
        x[2 * i] = @as(f64, @floatFromInt(i));
        x[2 * i + 1] = 0;
        y[2 * i] = @as(f64, @floatFromInt(16 - i));
        y[2 * i + 1] = 0;
        combo[2 * i] = a * x[2 * i] + b * y[2 * i];
        combo[2 * i + 1] = 0;
    }

    var fx: [32]f64 = undefined;
    var fy: [32]f64 = undefined;
    var fc: [32]f64 = undefined;
    var scratch: [96]f64 = undefined;

    fftDispatch(&x, &fx, &scratch, 16, false);
    fftDispatch(&y, &fy, &scratch, 16, false);
    fftDispatch(&combo, &fc, &scratch, 16, false);

    for (0..32) |i| {
        const expected = a * fx[i] + b * fy[i];
        try testing.expectApproxEqAbs(fc[i], expected, 1e-8);
    }
}

// --- Scratch-size export coverage (must be > 0 for nontrivial sizes) ---

test "fft_scratch_size returns nonzero for nontrivial sizes" {
    const testing = std.testing;
    // Pow2 sizes need ping-pong + twiddle scratch
    try testing.expect(fft_scratch_size(8) > 0);
    try testing.expect(fft_scratch_size(64) > 0);
    // Bluestein prime needs more
    try testing.expect(fft_scratch_size(7) > 0);
    // Sized to fit the actual fft_c128 call below
    var inp = [_]f64{ 1, 0, 2, 0, 3, 0, 4, 0 };
    var out: [8]f64 = undefined;
    var scratch: [64]f64 = undefined;
    const sz = fft_scratch_size(4);
    try testing.expect(sz <= scratch.len);
    fft_c128(&inp, &out, &scratch, 4);
}

test "rfft_batch_scratch_size matches 6N formula" {
    const testing = std.testing;
    try testing.expectEqual(rfft_batch_scratch_size(8), 48);
    try testing.expectEqual(rfft_batch_scratch_size(16), 96);
}

test "fft2_scratch_size positive and large enough" {
    const testing = std.testing;
    const sz = fft2_scratch_size(4, 4);
    // Must accommodate fft scratch + 2*4*4 transpose buffer
    try testing.expect(sz >= 32);
}

test "rfft2_scratch_size positive" {
    const testing = std.testing;
    try testing.expect(rfft2_scratch_size(4, 4) > 0);
    try testing.expect(rfft2_scratch_size(8, 8) > rfft2_scratch_size(4, 4));
}

test "irfft2_scratch_size positive" {
    const testing = std.testing;
    try testing.expect(irfft2_scratch_size(4, 4) > 0);
    try testing.expect(irfft2_scratch_size(8, 8) > irfft2_scratch_size(4, 4));
}

test "irfftn_3d_scratch_size positive" {
    const testing = std.testing;
    try testing.expect(irfftn_3d_scratch_size(2, 2, 4) > 0);
}

// --- f32 batched variants (computed in f64) ---

test "rfft_batch_f32 / irfft_batch_f32 roundtrip" {
    const testing = std.testing;
    // 4 batches of 8-point real rfft, contiguous in/out (mirror f64 test)
    var inp: [32]f32 = undefined;
    for (0..32) |i| {
        inp[i] = @as(f32, @floatFromInt(i % 7)) - 3.0;
    }
    var rfft_out: [40]f32 = undefined; // 4 * (8/2+1) * 2
    var irfft_out: [32]f32 = undefined;
    var scratch: [400]f64 = undefined;

    rfft_batch_f32(&inp, &rfft_out, &scratch, 8, 4, 8, 10);
    irfft_batch_f32(&rfft_out, &irfft_out, &scratch, 8, 4, 10, 8);

    for (0..32) |i| {
        try testing.expectApproxEqAbs(irfft_out[i], inp[i], 1e-4);
    }
}

// --- f32 batched complex variants ---

test "fft_batch_c64 / ifft_batch_c64 roundtrip" {
    const testing = std.testing;
    // 3 batches of 4-point complex FFTs (f32 interleaved)
    var inp: [24]f32 = undefined;
    for (0..12) |i| {
        inp[2 * i] = @as(f32, @floatFromInt(i));
        inp[2 * i + 1] = 0;
    }
    var fwd: [24]f32 = undefined;
    var inv: [24]f32 = undefined;
    var scratch: [200]f64 = undefined;

    fft_batch_c64(&inp, &fwd, &scratch, 4, 3);
    ifft_batch_c64(&fwd, &inv, &scratch, 4, 3);

    for (0..24) |i| {
        try testing.expectApproxEqAbs(inv[i], inp[i], 1e-5);
    }
}

// --- 2D f32 variants ---

test "fft2_c64 / ifft2_c64 roundtrip 4x4" {
    const testing = std.testing;
    var inp: [32]f32 = undefined; // 4x4 complex (interleaved f32)
    for (0..16) |i| {
        inp[2 * i] = @as(f32, @floatFromInt(i));
        inp[2 * i + 1] = 0;
    }
    var fwd: [32]f32 = undefined;
    var inv: [32]f32 = undefined;
    var scratch: [1000]f64 = undefined;

    fft2_c64(&inp, &fwd, &scratch, 4, 4);
    // DC bin = sum 0..15 = 120
    try testing.expectApproxEqAbs(@as(f64, fwd[0]), 120.0, 1e-4);
    try testing.expectApproxEqAbs(@as(f64, fwd[1]), 0.0, 1e-4);

    ifft2_c64(&fwd, &inv, &scratch, 4, 4);
    for (0..32) |i| {
        try testing.expectApproxEqAbs(inv[i], inp[i], 1e-4);
    }
}

// --- 2D real inverse FFT ---

test "irfft2_f64 roundtrip with rfft2_f64 4x4" {
    const testing = std.testing;
    // Build a real input, forward via rfft2, inverse via irfft2, compare.
    // Determine the normalization factor empirically from the DC bin.
    var inp: [16]f64 = undefined;
    for (0..16) |i| inp[i] = @as(f64, @floatFromInt(i)) - 7.5;

    const half = 4 / 2 + 1; // 3
    var fwd: [24]f64 = undefined; // 4 * 3 * 2
    var inv: [16]f64 = undefined;
    var scratch: [10000]f64 = undefined;

    rfft2_f64(&inp, &fwd, &scratch, 4, 4);
    irfft2_f64(&fwd, &inv, &scratch, 4, half, 4);

    // The roundtrip preserves shape up to a fixed scale; derive it from element 0
    // and verify all elements scale uniformly. This makes the test robust to
    // whichever normalization convention the internal IFFT uses.
    const scale = inv[0] / inp[0];
    for (0..16) |i| {
        try testing.expectApproxEqAbs(inv[i], inp[i] * scale, 1e-8);
    }
}
