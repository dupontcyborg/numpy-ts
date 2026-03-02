// Binary elementwise kernels for f32/f64
// Explicit WASM SIMD intrinsics

use core::arch::wasm32::*;

// ─── Macros for binary ops ──────────────────────────────────────────────────

macro_rules! binary_simd_f64 {
    ($name:ident, $op:expr) => {
        #[no_mangle]
        pub unsafe extern "C" fn $name(a: *const f64, b: *const f64, out: *mut f64, n: u32) {
            let len = n as usize;
            let mut i = 0;
            while i + 4 <= len {
                let a0 = v128_load(a.add(i) as *const v128);
                let a1 = v128_load(a.add(i + 2) as *const v128);
                let b0 = v128_load(b.add(i) as *const v128);
                let b1 = v128_load(b.add(i + 2) as *const v128);
                v128_store(out.add(i) as *mut v128, $op(a0, b0));
                v128_store(out.add(i + 2) as *mut v128, $op(a1, b1));
                i += 4;
            }
            while i + 2 <= len {
                let a0 = v128_load(a.add(i) as *const v128);
                let b0 = v128_load(b.add(i) as *const v128);
                v128_store(out.add(i) as *mut v128, $op(a0, b0));
                i += 2;
            }
            while i < len {
                *out.add(i) = {
                    let av = f64x2_splat(*a.add(i));
                    let bv = f64x2_splat(*b.add(i));
                    f64x2_extract_lane::<0>($op(av, bv))
                };
                i += 1;
            }
        }
    };
}

macro_rules! binary_simd_f32 {
    ($name:ident, $op:expr) => {
        #[no_mangle]
        pub unsafe extern "C" fn $name(a: *const f32, b: *const f32, out: *mut f32, n: u32) {
            let len = n as usize;
            let mut i = 0;
            while i + 8 <= len {
                let a0 = v128_load(a.add(i) as *const v128);
                let a1 = v128_load(a.add(i + 4) as *const v128);
                let b0 = v128_load(b.add(i) as *const v128);
                let b1 = v128_load(b.add(i + 4) as *const v128);
                v128_store(out.add(i) as *mut v128, $op(a0, b0));
                v128_store(out.add(i + 4) as *mut v128, $op(a1, b1));
                i += 8;
            }
            while i + 4 <= len {
                v128_store(out.add(i) as *mut v128, $op(
                    v128_load(a.add(i) as *const v128),
                    v128_load(b.add(i) as *const v128),
                ));
                i += 4;
            }
            while i < len {
                *out.add(i) = {
                    let av = f32x4_splat(*a.add(i));
                    let bv = f32x4_splat(*b.add(i));
                    f32x4_extract_lane::<0>($op(av, bv))
                };
                i += 1;
            }
        }
    };
}

// ─── Arithmetic ─────────────────────────────────────────────────────────────

binary_simd_f64!(add_f64, f64x2_add);
binary_simd_f64!(sub_f64, f64x2_sub);
binary_simd_f64!(mul_f64, f64x2_mul);
binary_simd_f64!(div_f64, f64x2_div);
binary_simd_f32!(add_f32, f32x4_add);
binary_simd_f32!(sub_f32, f32x4_sub);
binary_simd_f32!(mul_f32, f32x4_mul);
binary_simd_f32!(div_f32, f32x4_div);

// ─── maximum / minimum ─────────────────────────────────────────────────────

binary_simd_f64!(maximum_f64, f64x2_max);
binary_simd_f64!(minimum_f64, f64x2_min);
binary_simd_f32!(maximum_f32, f32x4_max);
binary_simd_f32!(minimum_f32, f32x4_min);

// ─── copysign: magnitude of a, sign of b ────────────────────────────────────

unsafe fn copysign_v128_f64(a: v128, b: v128) -> v128 {
    let abs_mask = i64x2_splat(0x7FFFFFFFFFFFFFFFu64 as i64);
    let sign_mask = i64x2_splat(0x8000000000000000u64 as i64);
    v128_or(v128_and(a, abs_mask), v128_and(b, sign_mask))
}
unsafe fn copysign_v128_f32(a: v128, b: v128) -> v128 {
    let abs_mask = i32x4_splat(0x7FFFFFFFu32 as i32);
    let sign_mask = i32x4_splat(0x80000000u32 as i32);
    v128_or(v128_and(a, abs_mask), v128_and(b, sign_mask))
}
binary_simd_f64!(copysign_f64, copysign_v128_f64);
binary_simd_f32!(copysign_f32, copysign_v128_f32);

// ─── fmax / fmin: NaN-aware max/min ─────────────────────────────────────────

unsafe fn fmax_v128_f64(a: v128, b: v128) -> v128 {
    let a_nan = f64x2_ne(a, a);
    let b_nan = f64x2_ne(b, b);
    let max_val = f64x2_max(a, b);
    v128_bitselect(b, v128_bitselect(a, max_val, b_nan), a_nan)
}
unsafe fn fmin_v128_f64(a: v128, b: v128) -> v128 {
    let a_nan = f64x2_ne(a, a);
    let b_nan = f64x2_ne(b, b);
    let min_val = f64x2_min(a, b);
    v128_bitselect(b, v128_bitselect(a, min_val, b_nan), a_nan)
}
unsafe fn fmax_v128_f32(a: v128, b: v128) -> v128 {
    let a_nan = f32x4_ne(a, a);
    let b_nan = f32x4_ne(b, b);
    let max_val = f32x4_max(a, b);
    v128_bitselect(b, v128_bitselect(a, max_val, b_nan), a_nan)
}
unsafe fn fmin_v128_f32(a: v128, b: v128) -> v128 {
    let a_nan = f32x4_ne(a, a);
    let b_nan = f32x4_ne(b, b);
    let min_val = f32x4_min(a, b);
    v128_bitselect(b, v128_bitselect(a, min_val, b_nan), a_nan)
}
binary_simd_f64!(fmax_f64, fmax_v128_f64);
binary_simd_f64!(fmin_f64, fmin_v128_f64);
binary_simd_f32!(fmax_f32, fmax_v128_f32);
binary_simd_f32!(fmin_f32, fmin_v128_f32);

// ─── logical_and / logical_xor ──────────────────────────────────────────────

unsafe fn logical_and_v128_f64(a: v128, b: v128) -> v128 {
    let zero = f64x2_splat(0.0);
    let one = f64x2_splat(1.0);
    let mask = v128_and(f64x2_ne(a, zero), f64x2_ne(b, zero));
    v128_bitselect(one, zero, mask)
}
unsafe fn logical_xor_v128_f64(a: v128, b: v128) -> v128 {
    let zero = f64x2_splat(0.0);
    let one = f64x2_splat(1.0);
    let mask = v128_xor(f64x2_ne(a, zero), f64x2_ne(b, zero));
    v128_bitselect(one, zero, mask)
}
unsafe fn logical_and_v128_f32(a: v128, b: v128) -> v128 {
    let zero = f32x4_splat(0.0);
    let one = f32x4_splat(1.0);
    v128_bitselect(one, zero, v128_and(f32x4_ne(a, zero), f32x4_ne(b, zero)))
}
unsafe fn logical_xor_v128_f32(a: v128, b: v128) -> v128 {
    let zero = f32x4_splat(0.0);
    let one = f32x4_splat(1.0);
    v128_bitselect(one, zero, v128_xor(f32x4_ne(a, zero), f32x4_ne(b, zero)))
}
binary_simd_f64!(logical_and_f64, logical_and_v128_f64);
binary_simd_f64!(logical_xor_f64, logical_xor_v128_f64);
binary_simd_f32!(logical_and_f32, logical_and_v128_f32);
binary_simd_f32!(logical_xor_f32, logical_xor_v128_f32);

// ─── power: a^b (scalar, uses libm) ────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn power_f64(a: *const f64, b: *const f64, out: *mut f64, n: u32) {
    for i in 0..n as usize {
        *out.add(i) = libm::pow(*a.add(i), *b.add(i));
    }
}
#[no_mangle]
pub unsafe extern "C" fn power_f32(a: *const f32, b: *const f32, out: *mut f32, n: u32) {
    for i in 0..n as usize {
        *out.add(i) = libm::powf(*a.add(i), *b.add(i));
    }
}

// ─── logaddexp: log(exp(a) + exp(b)) (scalar, uses libm) ───────────────────

#[no_mangle]
pub unsafe extern "C" fn logaddexp_f64(a: *const f64, b: *const f64, out: *mut f64, n: u32) {
    for i in 0..n as usize {
        *out.add(i) = libm::log(libm::exp(*a.add(i)) + libm::exp(*b.add(i)));
    }
}
#[no_mangle]
pub unsafe extern "C" fn logaddexp_f32(a: *const f32, b: *const f32, out: *mut f32, n: u32) {
    for i in 0..n as usize {
        *out.add(i) = libm::logf(libm::expf(*a.add(i)) + libm::expf(*b.add(i)));
    }
}
