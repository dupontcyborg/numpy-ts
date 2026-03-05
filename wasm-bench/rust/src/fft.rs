// FFT WASM kernels: rfft2, irfft2
// Cooley-Tukey radix-2 DIT for power-of-2, Bluestein's for arbitrary sizes.
// Complex stored as interleaved [re, im, re, im, ...].

use libm::{cos, sin};

fn next_pow2(n: usize) -> usize {
    let mut v = 1;
    while v < n {
        v <<= 1;
    }
    v
}

fn is_pow2(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

const PI: f64 = core::f64::consts::PI;

// ─── In-place radix-2 Cooley-Tukey FFT (safe, slice-based) ──────────────────

fn fft_pow2(data: &mut [f64], n: usize, inverse: bool) {
    // Bit-reversal permutation
    let mut j: usize = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            let ti = i * 2;
            let tj = j * 2;
            data.swap(ti, tj);
            data.swap(ti + 1, tj + 1);
        }
    }

    let sign: f64 = if inverse { 1.0 } else { -1.0 };
    let mut len: usize = 2;
    while len <= n {
        let half = len >> 1;
        let angle = sign * 2.0 * PI / (len as f64);
        let wr_step = cos(angle);
        let wi_step = sin(angle);

        let mut start: usize = 0;
        while start < n {
            let mut wr = 1.0f64;
            let mut wi = 0.0f64;
            for k in 0..half {
                let u_idx = (start + k) * 2;
                let v_idx = (start + k + half) * 2;
                let ur = data[u_idx];
                let ui = data[u_idx + 1];
                let vr = data[v_idx];
                let vi = data[v_idx + 1];

                let tvr = wr * vr - wi * vi;
                let tvi = wr * vi + wi * vr;

                data[u_idx] = ur + tvr;
                data[u_idx + 1] = ui + tvi;
                data[v_idx] = ur - tvr;
                data[v_idx + 1] = ui - tvi;

                let new_wr = wr * wr_step - wi * wi_step;
                wi = wr * wi_step + wi * wr_step;
                wr = new_wr;
            }
            start += len;
        }
        len <<= 1;
    }

    if inverse {
        let scale = 1.0 / (n as f64);
        for v in data[..n * 2].iter_mut() {
            *v *= scale;
        }
    }
}

// ─── Bluestein's FFT (safe, slice-based) ─────────────────────────────────────

fn bluestein_fft(input: &[f64], output: &mut [f64], n: usize, inverse: bool, scratch: &mut [f64]) {
    if n <= 1 {
        if n == 1 {
            output[0] = input[0];
            output[1] = input[1];
        }
        return;
    }

    if is_pow2(n) {
        output[..n * 2].copy_from_slice(&input[..n * 2]);
        fft_pow2(output, n, inverse);
        return;
    }

    let p = next_pow2(2 * n - 1);

    // Partition scratch into sub-slices
    let (chirp, rest) = scratch.split_at_mut(2 * p);
    let (a_pad, b_pad) = rest.split_at_mut(2 * p);

    let sign: f64 = if inverse { -1.0 } else { 1.0 };

    // Build chirp
    for k in 0..n {
        let angle = sign * PI * ((k * k) as f64) / (n as f64);
        chirp[2 * k] = cos(angle);
        chirp[2 * k + 1] = sin(angle);
    }

    // a[k] = input[k] * conj(chirp[k])
    for v in a_pad[..p * 2].iter_mut() {
        *v = 0.0;
    }
    for k in 0..n {
        let ir = input[2 * k];
        let ii = input[2 * k + 1];
        let cr = chirp[2 * k];
        let ci = -chirp[2 * k + 1];
        a_pad[2 * k] = ir * cr - ii * ci;
        a_pad[2 * k + 1] = ir * ci + ii * cr;
    }

    // b[0] = chirp[0], b[k] = b[P-k] = chirp[k]
    for v in b_pad[..p * 2].iter_mut() {
        *v = 0.0;
    }
    b_pad[0] = chirp[0];
    b_pad[1] = chirp[1];
    for k in 1..n {
        b_pad[2 * k] = chirp[2 * k];
        b_pad[2 * k + 1] = chirp[2 * k + 1];
        b_pad[2 * (p - k)] = chirp[2 * k];
        b_pad[2 * (p - k) + 1] = chirp[2 * k + 1];
    }

    fft_pow2(a_pad, p, false);
    fft_pow2(b_pad, p, false);

    // Pointwise multiply
    for k in 0..p {
        let ar = a_pad[2 * k];
        let ai = a_pad[2 * k + 1];
        let br = b_pad[2 * k];
        let bi = b_pad[2 * k + 1];
        a_pad[2 * k] = ar * br - ai * bi;
        a_pad[2 * k + 1] = ar * bi + ai * br;
    }

    fft_pow2(a_pad, p, true);

    // output[k] = a_pad[k] * conj(chirp[k])
    for k in 0..n {
        let ar = a_pad[2 * k];
        let ai = a_pad[2 * k + 1];
        let cr = chirp[2 * k];
        let ci = -chirp[2 * k + 1];
        output[2 * k] = ar * cr - ai * ci;
        output[2 * k + 1] = ar * ci + ai * cr;
    }

    if inverse {
        let scale = 1.0 / (n as f64);
        for i in 0..n * 2 {
            output[i] *= scale;
        }
    }
}

// ─── rfft2: M×N real → M×(N/2+1) complex ──────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn rfft2_f64(
    inp: *const f64,
    out: *mut f64,
    scratch: *mut f64,
    m: u32,
    n: u32,
) {
    let rows = m as usize;
    let cols = n as usize;
    let half_n = cols / 2 + 1;
    let input = core::slice::from_raw_parts(inp, rows * cols);
    let output = core::slice::from_raw_parts_mut(out, rows * half_n * 2);

    // Scratch layout: row_buf(2*cols) + col_buf(2*rows) + fft_scratch(enough for bluestein)
    let row_buf = core::slice::from_raw_parts_mut(scratch, 2 * cols);
    let col_buf = core::slice::from_raw_parts_mut(scratch.add(2 * cols), 2 * rows);
    let fft_scratch = core::slice::from_raw_parts_mut(
        scratch.add(2 * cols + 2 * rows),
        6 * next_pow2(2 * cols.max(rows) - 1),
    );

    // Step 1: FFT each row
    for row in 0..rows {
        for j in 0..cols {
            row_buf[2 * j] = input[row * cols + j];
            row_buf[2 * j + 1] = 0.0;
        }
        bluestein_fft(row_buf, col_buf, cols, false, fft_scratch);
        for j in 0..half_n {
            output[(row * half_n + j) * 2] = col_buf[2 * j];
            output[(row * half_n + j) * 2 + 1] = col_buf[2 * j + 1];
        }
    }

    // Step 2: FFT each column
    for col in 0..half_n {
        for row in 0..rows {
            col_buf[2 * row] = output[(row * half_n + col) * 2];
            col_buf[2 * row + 1] = output[(row * half_n + col) * 2 + 1];
        }
        bluestein_fft(col_buf, row_buf, rows, false, fft_scratch);
        for row in 0..rows {
            output[(row * half_n + col) * 2] = row_buf[2 * row];
            output[(row * half_n + col) * 2 + 1] = row_buf[2 * row + 1];
        }
    }
}

// ─── Complex-to-complex FFT (c128, c64) ─────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn fft_c128(inp: *const f64, out: *mut f64, scratch: *mut f64, n: u32) {
    let nn = n as usize;
    let input = core::slice::from_raw_parts(inp, nn * 2);
    let output = core::slice::from_raw_parts_mut(out, nn * 2);
    let sc = core::slice::from_raw_parts_mut(scratch, 6 * next_pow2(2 * nn - 1));
    bluestein_fft(input, output, nn, false, sc);
}

#[no_mangle]
pub unsafe extern "C" fn ifft_c128(inp: *const f64, out: *mut f64, scratch: *mut f64, n: u32) {
    let nn = n as usize;
    let input = core::slice::from_raw_parts(inp, nn * 2);
    let output = core::slice::from_raw_parts_mut(out, nn * 2);
    let sc = core::slice::from_raw_parts_mut(scratch, 6 * next_pow2(2 * nn - 1));
    bluestein_fft(input, output, nn, true, sc);
}

#[no_mangle]
pub unsafe extern "C" fn fft_c64(inp: *const f32, out: *mut f32, scratch: *mut f64, n: u32) {
    let nn = n as usize;
    let input = core::slice::from_raw_parts(inp, 2 * nn);
    let output = core::slice::from_raw_parts_mut(out, 2 * nn);
    let in_f64 = core::slice::from_raw_parts_mut(scratch, 2 * nn);
    let out_f64 = core::slice::from_raw_parts_mut(scratch.add(2 * nn), 2 * nn);
    let fft_scratch =
        core::slice::from_raw_parts_mut(scratch.add(4 * nn), 6 * next_pow2(2 * nn - 1));
    for i in 0..2 * nn {
        in_f64[i] = input[i] as f64;
    }
    bluestein_fft(in_f64, out_f64, nn, false, fft_scratch);
    for i in 0..2 * nn {
        output[i] = out_f64[i] as f32;
    }
}

#[no_mangle]
pub unsafe extern "C" fn ifft_c64(inp: *const f32, out: *mut f32, scratch: *mut f64, n: u32) {
    let nn = n as usize;
    let input = core::slice::from_raw_parts(inp, 2 * nn);
    let output = core::slice::from_raw_parts_mut(out, 2 * nn);
    let in_f64 = core::slice::from_raw_parts_mut(scratch, 2 * nn);
    let out_f64 = core::slice::from_raw_parts_mut(scratch.add(2 * nn), 2 * nn);
    let fft_scratch =
        core::slice::from_raw_parts_mut(scratch.add(4 * nn), 6 * next_pow2(2 * nn - 1));
    for i in 0..2 * nn {
        in_f64[i] = input[i] as f64;
    }
    bluestein_fft(in_f64, out_f64, nn, true, fft_scratch);
    for i in 0..2 * nn {
        output[i] = out_f64[i] as f32;
    }
}

// ─── irfft2: M×(N/2+1) complex → M×N real ─────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn irfft2_f64(
    inp: *const f64,
    out: *mut f64,
    scratch: *mut f64,
    m: u32,
    n: u32,
) {
    let rows = m as usize;
    let cols = n as usize;
    let half_n = cols / 2 + 1;
    let input = core::slice::from_raw_parts(inp, rows * half_n * 2);
    let output = core::slice::from_raw_parts_mut(out, rows * cols);

    let work = core::slice::from_raw_parts_mut(scratch, rows * half_n * 2);
    let full_row = core::slice::from_raw_parts_mut(scratch.add(rows * half_n * 2), 2 * cols);
    let col_buf =
        core::slice::from_raw_parts_mut(scratch.add(rows * half_n * 2 + 2 * cols), 2 * rows);
    let fft_scratch = core::slice::from_raw_parts_mut(
        scratch.add(rows * half_n * 2 + 2 * cols + 2 * rows),
        6 * next_pow2(2 * cols.max(rows) - 1),
    );

    // Copy input to work
    work.copy_from_slice(input);

    // Step 1: IFFT each column
    for col in 0..half_n {
        for row in 0..rows {
            col_buf[2 * row] = work[(row * half_n + col) * 2];
            col_buf[2 * row + 1] = work[(row * half_n + col) * 2 + 1];
        }
        bluestein_fft(col_buf, full_row, rows, true, fft_scratch);
        for row in 0..rows {
            work[(row * half_n + col) * 2] = full_row[2 * row];
            work[(row * half_n + col) * 2 + 1] = full_row[2 * row + 1];
        }
    }

    // Step 2: IFFT each row with Hermitian reconstruction
    for row in 0..rows {
        for j in 0..half_n {
            full_row[2 * j] = work[(row * half_n + j) * 2];
            full_row[2 * j + 1] = work[(row * half_n + j) * 2 + 1];
        }
        for j in half_n..cols {
            let mirror = cols - j;
            full_row[2 * j] = full_row[2 * mirror];
            full_row[2 * j + 1] = -full_row[2 * mirror + 1];
        }
        bluestein_fft(full_row, col_buf, cols, true, fft_scratch);
        for j in 0..cols {
            output[row * cols + j] = col_buf[2 * j];
        }
    }
}
