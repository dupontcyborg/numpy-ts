// 1D cross-correlation and convolution kernels for f32/f64
// SIMD-accelerated inner dot product loop

use crate::simd::{load_f32x4, load_f64x2};
use core::arch::wasm32::*;

fn correlate_f64_inner(a: &[f64], b: &[f64], out: &mut [f64]) {
    let n_a = a.len();
    let n_b = b.len();
    let out_len = n_a + n_b - 1;

    for k in 0..out_len {
        let j_start = if k >= n_b - 1 { k - (n_b - 1) } else { 0 };
        let j_end = if k < n_a { k + 1 } else { n_a };
        let b_off = n_b - 1 - k;

        let mut acc0 = f64x2_splat(0.0);
        let mut acc1 = f64x2_splat(0.0);
        let mut j = j_start;

        while j + 4 <= j_end {
            let bi = j + b_off;
            acc0 = f64x2_add(acc0, f64x2_mul(load_f64x2(a, j), load_f64x2(b, bi)));
            acc1 = f64x2_add(acc1, f64x2_mul(load_f64x2(a, j + 2), load_f64x2(b, bi + 2)));
            j += 4;
        }
        while j + 2 <= j_end {
            acc0 = f64x2_add(acc0, f64x2_mul(load_f64x2(a, j), load_f64x2(b, j + b_off)));
            j += 2;
        }
        acc0 = f64x2_add(acc0, acc1);
        let mut sum = f64x2_extract_lane::<0>(acc0) + f64x2_extract_lane::<1>(acc0);
        while j < j_end {
            sum += a[j] * b[j + b_off];
            j += 1;
        }
        out[k] = sum;
    }
}

fn convolve_f64_inner(a: &[f64], b: &[f64], out: &mut [f64]) {
    // Convolve = correlate with reversed kernel.
    // Reverse b into the output buffer temporarily (it's large enough), then correlate.
    let n_a = a.len();
    let n_b = b.len();
    let out_len = n_a + n_b - 1;

    // Store reversed b at end of output buffer (we need n_b slots, out has out_len >= n_b)
    // Actually, just reverse into a stack-allocated portion or use out as scratch.
    // Since we write to out anyway, use first n_b slots of out as reversed b buffer.
    for i in 0..n_b {
        out[i] = b[n_b - 1 - i];
    }

    // Now correlate a with reversed_b (which is in out[0..n_b])
    // But we're also writing results to out... so we need to be careful.
    // Use a different approach: compute in-place using the correlate logic.
    // The reversed b is b_rev[i] = b[n_b-1-i], so correlate(a, b_rev) at position k:
    //   b_off = n_b - 1 - k (same as correlate), but with b_rev instead of b
    //   b_rev[j + b_off] = b[n_b - 1 - (j + b_off)] = b[n_b - 1 - j - (n_b-1-k)] = b[k - j]
    // This is exactly convolve! So correlate with b_rev = convolve with b.

    // Since we can't easily store reversed b separately, just do scalar with 4x unroll.
    for k in 0..out_len {
        let j_start = if k >= n_b - 1 { k - (n_b - 1) } else { 0 };
        let j_end = if k < n_a { k + 1 } else { n_a };
        let mut sum = 0.0f64;
        let mut j = j_start;
        while j + 4 <= j_end {
            sum += a[j] * b[k - j]
                + a[j + 1] * b[k - j - 1]
                + a[j + 2] * b[k - j - 2]
                + a[j + 3] * b[k - j - 3];
            j += 4;
        }
        while j < j_end {
            sum += a[j] * b[k - j];
            j += 1;
        }
        out[k] = sum;
    }
}

fn correlate_f32_inner(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n_a = a.len();
    let n_b = b.len();
    let out_len = n_a + n_b - 1;

    for k in 0..out_len {
        let j_start = if k >= n_b - 1 { k - (n_b - 1) } else { 0 };
        let j_end = if k < n_a { k + 1 } else { n_a };
        let b_off = n_b - 1 - k;

        let mut acc0 = f32x4_splat(0.0);
        let mut acc1 = f32x4_splat(0.0);
        let mut j = j_start;

        while j + 8 <= j_end {
            let bi = j + b_off;
            acc0 = f32x4_add(acc0, f32x4_mul(load_f32x4(a, j), load_f32x4(b, bi)));
            acc1 = f32x4_add(acc1, f32x4_mul(load_f32x4(a, j + 4), load_f32x4(b, bi + 4)));
            j += 8;
        }
        while j + 4 <= j_end {
            acc0 = f32x4_add(acc0, f32x4_mul(load_f32x4(a, j), load_f32x4(b, j + b_off)));
            j += 4;
        }
        acc0 = f32x4_add(acc0, acc1);
        let mut sum = f32x4_extract_lane::<0>(acc0)
            + f32x4_extract_lane::<1>(acc0)
            + f32x4_extract_lane::<2>(acc0)
            + f32x4_extract_lane::<3>(acc0);
        while j < j_end {
            sum += a[j] * b[j + b_off];
            j += 1;
        }
        out[k] = sum;
    }
}

fn convolve_f32_inner(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n_a = a.len();
    let n_b = b.len();
    let out_len = n_a + n_b - 1;

    for k in 0..out_len {
        let j_start = if k >= n_b - 1 { k - (n_b - 1) } else { 0 };
        let j_end = if k < n_a { k + 1 } else { n_a };
        let mut sum = 0.0f32;
        let mut j = j_start;
        while j + 4 <= j_end {
            sum += a[j] * b[k - j]
                + a[j + 1] * b[k - j - 1]
                + a[j + 2] * b[k - j - 2]
                + a[j + 3] * b[k - j - 3];
            j += 4;
        }
        while j < j_end {
            sum += a[j] * b[k - j];
            j += 1;
        }
        out[k] = sum;
    }
}

// ─── FFI exports ─────────────────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn correlate_f64(
    a: *const f64,
    b: *const f64,
    out: *mut f64,
    na: u32,
    nb: u32,
) {
    let (n_a, n_b) = (na as usize, nb as usize);
    correlate_f64_inner(
        core::slice::from_raw_parts(a, n_a),
        core::slice::from_raw_parts(b, n_b),
        core::slice::from_raw_parts_mut(out, n_a + n_b - 1),
    );
}

#[no_mangle]
pub unsafe extern "C" fn convolve_f64(
    a: *const f64,
    b: *const f64,
    out: *mut f64,
    na: u32,
    nb: u32,
) {
    let (n_a, n_b) = (na as usize, nb as usize);
    convolve_f64_inner(
        core::slice::from_raw_parts(a, n_a),
        core::slice::from_raw_parts(b, n_b),
        core::slice::from_raw_parts_mut(out, n_a + n_b - 1),
    );
}

#[no_mangle]
pub unsafe extern "C" fn correlate_f32(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    na: u32,
    nb: u32,
) {
    let (n_a, n_b) = (na as usize, nb as usize);
    correlate_f32_inner(
        core::slice::from_raw_parts(a, n_a),
        core::slice::from_raw_parts(b, n_b),
        core::slice::from_raw_parts_mut(out, n_a + n_b - 1),
    );
}

#[no_mangle]
pub unsafe extern "C" fn convolve_f32(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    na: u32,
    nb: u32,
) {
    let (n_a, n_b) = (na as usize, nb as usize);
    convolve_f32_inner(
        core::slice::from_raw_parts(a, n_a),
        core::slice::from_raw_parts(b, n_b),
        core::slice::from_raw_parts_mut(out, n_a + n_b - 1),
    );
}
