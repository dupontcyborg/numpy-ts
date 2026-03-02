// 1D cross-correlation and convolution kernels for f32/f64
// SIMD-accelerated inner dot product loop

use core::arch::wasm32::*;

#[no_mangle]
pub unsafe extern "C" fn correlate_f64(
    a: *const f64, b: *const f64, out: *mut f64, na: u32, nb: u32,
) {
    let n_a = na as usize;
    let n_b = nb as usize;
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
            acc0 = f64x2_add(acc0, f64x2_mul(
                v128_load(a.add(j) as *const v128),
                v128_load(b.add(bi) as *const v128),
            ));
            acc1 = f64x2_add(acc1, f64x2_mul(
                v128_load(a.add(j + 2) as *const v128),
                v128_load(b.add(bi + 2) as *const v128),
            ));
            j += 4;
        }
        while j + 2 <= j_end {
            acc0 = f64x2_add(acc0, f64x2_mul(
                v128_load(a.add(j) as *const v128),
                v128_load(b.add(j + b_off) as *const v128),
            ));
            j += 2;
        }
        acc0 = f64x2_add(acc0, acc1);
        let mut sum = f64x2_extract_lane::<0>(acc0) + f64x2_extract_lane::<1>(acc0);
        while j < j_end {
            sum += *a.add(j) * *b.add(j + b_off);
            j += 1;
        }
        *out.add(k) = sum;
    }
}

#[no_mangle]
pub unsafe extern "C" fn convolve_f64(
    a: *const f64, b: *const f64, out: *mut f64, na: u32, nb: u32,
) {
    let n_a = na as usize;
    let n_b = nb as usize;
    let out_len = n_a + n_b - 1;

    for k in 0..out_len {
        let j_start = if k >= n_b - 1 { k - (n_b - 1) } else { 0 };
        let j_end = if k < n_a { k + 1 } else { n_a };
        let mut sum = 0.0f64;
        for j in j_start..j_end {
            sum += *a.add(j) * *b.add(k - j);
        }
        *out.add(k) = sum;
    }
}

#[no_mangle]
pub unsafe extern "C" fn correlate_f32(
    a: *const f32, b: *const f32, out: *mut f32, na: u32, nb: u32,
) {
    let n_a = na as usize;
    let n_b = nb as usize;
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
            acc0 = f32x4_add(acc0, f32x4_mul(
                v128_load(a.add(j) as *const v128),
                v128_load(b.add(bi) as *const v128),
            ));
            acc1 = f32x4_add(acc1, f32x4_mul(
                v128_load(a.add(j + 4) as *const v128),
                v128_load(b.add(bi + 4) as *const v128),
            ));
            j += 8;
        }
        while j + 4 <= j_end {
            acc0 = f32x4_add(acc0, f32x4_mul(
                v128_load(a.add(j) as *const v128),
                v128_load(b.add(j + b_off) as *const v128),
            ));
            j += 4;
        }
        acc0 = f32x4_add(acc0, acc1);
        let mut sum = f32x4_extract_lane::<0>(acc0)
            + f32x4_extract_lane::<1>(acc0)
            + f32x4_extract_lane::<2>(acc0)
            + f32x4_extract_lane::<3>(acc0);
        while j < j_end {
            sum += *a.add(j) * *b.add(j + b_off);
            j += 1;
        }
        *out.add(k) = sum;
    }
}

#[no_mangle]
pub unsafe extern "C" fn convolve_f32(
    a: *const f32, b: *const f32, out: *mut f32, na: u32, nb: u32,
) {
    let n_a = na as usize;
    let n_b = nb as usize;
    let out_len = n_a + n_b - 1;

    for k in 0..out_len {
        let j_start = if k >= n_b - 1 { k - (n_b - 1) } else { 0 };
        let j_end = if k < n_a { k + 1 } else { n_a };
        let mut sum = 0.0f32;
        for j in j_start..j_end {
            sum += *a.add(j) * *b.add(k - j);
        }
        *out.add(k) = sum;
    }
}
