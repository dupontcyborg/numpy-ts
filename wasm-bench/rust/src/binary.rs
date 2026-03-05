// Binary elementwise kernels for f32/f64
// Explicit WASM SIMD intrinsics

use crate::simd::{
    load_f32x4, load_f64x2, load_i16x8, load_i32x4, load_i8x16, store_f32x4, store_f64x2,
    store_i16x8, store_i32x4, store_i8x16,
};
use core::arch::wasm32::*;

// ─── Macros — safe inner fn + thin unsafe FFI wrapper ───────────────────────

macro_rules! binary_simd_f64 {
    ($name:ident, $op:expr) => {
        #[no_mangle]
        pub unsafe extern "C" fn $name(a: *const f64, b: *const f64, out: *mut f64, n: u32) {
            // Safe inner function that operates on slices, so we can use safe indexing and iterators.
            fn inner(sa: &[f64], sb: &[f64], so: &mut [f64]) {
                let len = sa.len();
                let mut i = 0;
                while i + 4 <= len {
                    let a0 = load_f64x2(sa, i);
                    let a1 = load_f64x2(sa, i + 2);
                    let b0 = load_f64x2(sb, i);
                    let b1 = load_f64x2(sb, i + 2);
                    store_f64x2(so, i, $op(a0, b0));
                    store_f64x2(so, i + 2, $op(a1, b1));
                    i += 4;
                }
                while i + 2 <= len {
                    store_f64x2(so, i, $op(load_f64x2(sa, i), load_f64x2(sb, i)));
                    i += 2;
                }
                while i < len {
                    so[i] = f64x2_extract_lane::<0>($op(f64x2_splat(sa[i]), f64x2_splat(sb[i])));
                    i += 1;
                }
            }

            // Call the inner function with slices; unsafe only for dereferencing & slice creation
            let len = n as usize;
            inner(
                core::slice::from_raw_parts(a, len),
                core::slice::from_raw_parts(b, len),
                core::slice::from_raw_parts_mut(out, len),
            );
        }
    };
}

macro_rules! binary_simd_f32 {
    ($name:ident, $op:expr) => {
        #[no_mangle]
        pub unsafe extern "C" fn $name(a: *const f32, b: *const f32, out: *mut f32, n: u32) {
            // Safe inner function that operates on slices, so we can use safe indexing and iterators.
            fn inner(sa: &[f32], sb: &[f32], so: &mut [f32]) {
                let len = sa.len();
                let mut i = 0;
                while i + 8 <= len {
                    let a0 = load_f32x4(sa, i);
                    let a1 = load_f32x4(sa, i + 4);
                    let b0 = load_f32x4(sb, i);
                    let b1 = load_f32x4(sb, i + 4);
                    store_f32x4(so, i, $op(a0, b0));
                    store_f32x4(so, i + 4, $op(a1, b1));
                    i += 8;
                }
                while i + 4 <= len {
                    store_f32x4(so, i, $op(load_f32x4(sa, i), load_f32x4(sb, i)));
                    i += 4;
                }
                while i < len {
                    so[i] = f32x4_extract_lane::<0>($op(f32x4_splat(sa[i]), f32x4_splat(sb[i])));
                    i += 1;
                }
            }

            // Call the inner function with slices; unsafe only for dereferencing & slice creation
            let len = n as usize;
            inner(
                core::slice::from_raw_parts(a, len),
                core::slice::from_raw_parts(b, len),
                core::slice::from_raw_parts_mut(out, len),
            );
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

fn copysign_v128_f64(a: v128, b: v128) -> v128 {
    let abs_mask = i64x2_splat(0x7FFFFFFFFFFFFFFFu64 as i64);
    let sign_mask = i64x2_splat(0x8000000000000000u64 as i64);
    v128_or(v128_and(a, abs_mask), v128_and(b, sign_mask))
}
fn copysign_v128_f32(a: v128, b: v128) -> v128 {
    let abs_mask = i32x4_splat(0x7FFFFFFFu32 as i32);
    let sign_mask = i32x4_splat(0x80000000u32 as i32);
    v128_or(v128_and(a, abs_mask), v128_and(b, sign_mask))
}
binary_simd_f64!(copysign_f64, copysign_v128_f64);
binary_simd_f32!(copysign_f32, copysign_v128_f32);

// ─── fmax / fmin: NaN-aware max/min ─────────────────────────────────────────

fn fmax_v128_f64(a: v128, b: v128) -> v128 {
    let a_nan = f64x2_ne(a, a);
    let b_nan = f64x2_ne(b, b);
    v128_bitselect(b, v128_bitselect(a, f64x2_max(a, b), b_nan), a_nan)
}
fn fmin_v128_f64(a: v128, b: v128) -> v128 {
    let a_nan = f64x2_ne(a, a);
    let b_nan = f64x2_ne(b, b);
    v128_bitselect(b, v128_bitselect(a, f64x2_min(a, b), b_nan), a_nan)
}
fn fmax_v128_f32(a: v128, b: v128) -> v128 {
    let a_nan = f32x4_ne(a, a);
    let b_nan = f32x4_ne(b, b);
    v128_bitselect(b, v128_bitselect(a, f32x4_max(a, b), b_nan), a_nan)
}
fn fmin_v128_f32(a: v128, b: v128) -> v128 {
    let a_nan = f32x4_ne(a, a);
    let b_nan = f32x4_ne(b, b);
    v128_bitselect(b, v128_bitselect(a, f32x4_min(a, b), b_nan), a_nan)
}
binary_simd_f64!(fmax_f64, fmax_v128_f64);
binary_simd_f64!(fmin_f64, fmin_v128_f64);
binary_simd_f32!(fmax_f32, fmax_v128_f32);
binary_simd_f32!(fmin_f32, fmin_v128_f32);

// ─── mod (floored remainder): a - floor(a/b) * b ─────────────────────────

fn mod_v128_f64(a: v128, b: v128) -> v128 {
    f64x2_sub(a, f64x2_mul(f64x2_floor(f64x2_div(a, b)), b))
}
fn mod_v128_f32(a: v128, b: v128) -> v128 {
    f32x4_sub(a, f32x4_mul(f32x4_floor(f32x4_div(a, b)), b))
}
binary_simd_f64!(mod_f64, mod_v128_f64);
binary_simd_f32!(mod_f32, mod_v128_f32);

// ─── floor_divide: floor(a / b) ─────────────────────────────────────────

fn floor_divide_v128_f64(a: v128, b: v128) -> v128 {
    f64x2_floor(f64x2_div(a, b))
}
fn floor_divide_v128_f32(a: v128, b: v128) -> v128 {
    f32x4_floor(f32x4_div(a, b))
}
binary_simd_f64!(floor_divide_f64, floor_divide_v128_f64);
binary_simd_f32!(floor_divide_f32, floor_divide_v128_f32);

// ─── hypot: sqrt(a² + b²) ──────────────────────────────────────────────

fn hypot_v128_f64(a: v128, b: v128) -> v128 {
    f64x2_sqrt(f64x2_add(f64x2_mul(a, a), f64x2_mul(b, b)))
}
fn hypot_v128_f32(a: v128, b: v128) -> v128 {
    f32x4_sqrt(f32x4_add(f32x4_mul(a, a), f32x4_mul(b, b)))
}
binary_simd_f64!(hypot_f64, hypot_v128_f64);
binary_simd_f32!(hypot_f32, hypot_v128_f32);

// ─── logical_and / logical_xor ──────────────────────────────────────────────

fn logical_and_v128_f64(a: v128, b: v128) -> v128 {
    let zero = f64x2_splat(0.0);
    let one = f64x2_splat(1.0);
    v128_bitselect(one, zero, v128_and(f64x2_ne(a, zero), f64x2_ne(b, zero)))
}
fn logical_xor_v128_f64(a: v128, b: v128) -> v128 {
    let zero = f64x2_splat(0.0);
    let one = f64x2_splat(1.0);
    v128_bitselect(one, zero, v128_xor(f64x2_ne(a, zero), f64x2_ne(b, zero)))
}
fn logical_and_v128_f32(a: v128, b: v128) -> v128 {
    let zero = f32x4_splat(0.0);
    let one = f32x4_splat(1.0);
    v128_bitselect(one, zero, v128_and(f32x4_ne(a, zero), f32x4_ne(b, zero)))
}
fn logical_xor_v128_f32(a: v128, b: v128) -> v128 {
    let zero = f32x4_splat(0.0);
    let one = f32x4_splat(1.0);
    v128_bitselect(one, zero, v128_xor(f32x4_ne(a, zero), f32x4_ne(b, zero)))
}
binary_simd_f64!(logical_and_f64, logical_and_v128_f64);
binary_simd_f64!(logical_xor_f64, logical_xor_v128_f64);
binary_simd_f32!(logical_and_f32, logical_and_v128_f32);
binary_simd_f32!(logical_xor_f32, logical_xor_v128_f32);

// ─── power: a^b (scalar, uses libm) ────────────────────────────────────────

fn power_f64_inner(sa: &[f64], sb: &[f64], so: &mut [f64]) {
    let len = sa.len();
    let mut i = 0;
    while i + 4 <= len {
        so[i] = libm::pow(sa[i], sb[i]);
        so[i + 1] = libm::pow(sa[i + 1], sb[i + 1]);
        so[i + 2] = libm::pow(sa[i + 2], sb[i + 2]);
        so[i + 3] = libm::pow(sa[i + 3], sb[i + 3]);
        i += 4;
    }
    while i < len {
        so[i] = libm::pow(sa[i], sb[i]);
        i += 1;
    }
}
fn power_f32_inner(sa: &[f32], sb: &[f32], so: &mut [f32]) {
    let len = sa.len();
    let mut i = 0;
    while i + 4 <= len {
        so[i] = libm::powf(sa[i], sb[i]);
        so[i + 1] = libm::powf(sa[i + 1], sb[i + 1]);
        so[i + 2] = libm::powf(sa[i + 2], sb[i + 2]);
        so[i + 3] = libm::powf(sa[i + 3], sb[i + 3]);
        i += 4;
    }
    while i < len {
        so[i] = libm::powf(sa[i], sb[i]);
        i += 1;
    }
}

#[no_mangle]
pub unsafe extern "C" fn power_f64(a: *const f64, b: *const f64, out: *mut f64, n: u32) {
    let len = n as usize;
    power_f64_inner(
        core::slice::from_raw_parts(a, len),
        core::slice::from_raw_parts(b, len),
        core::slice::from_raw_parts_mut(out, len),
    );
}
#[no_mangle]
pub unsafe extern "C" fn power_f32(a: *const f32, b: *const f32, out: *mut f32, n: u32) {
    let len = n as usize;
    power_f32_inner(
        core::slice::from_raw_parts(a, len),
        core::slice::from_raw_parts(b, len),
        core::slice::from_raw_parts_mut(out, len),
    );
}

// ─── logaddexp: log(exp(a) + exp(b)) (scalar, uses libm) ───────────────────

fn logaddexp_f64_inner(sa: &[f64], sb: &[f64], so: &mut [f64]) {
    for i in 0..sa.len() {
        so[i] = libm::log(libm::exp(sa[i]) + libm::exp(sb[i]));
    }
}
fn logaddexp_f32_inner(sa: &[f32], sb: &[f32], so: &mut [f32]) {
    for i in 0..sa.len() {
        so[i] = libm::logf(libm::expf(sa[i]) + libm::expf(sb[i]));
    }
}

#[no_mangle]
pub unsafe extern "C" fn logaddexp_f64(a: *const f64, b: *const f64, out: *mut f64, n: u32) {
    let len = n as usize;
    logaddexp_f64_inner(
        core::slice::from_raw_parts(a, len),
        core::slice::from_raw_parts(b, len),
        core::slice::from_raw_parts_mut(out, len),
    );
}
#[no_mangle]
pub unsafe extern "C" fn logaddexp_f32(a: *const f32, b: *const f32, out: *mut f32, n: u32) {
    let len = n as usize;
    logaddexp_f32_inner(
        core::slice::from_raw_parts(a, len),
        core::slice::from_raw_parts(b, len),
        core::slice::from_raw_parts_mut(out, len),
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// INTEGER TYPES (i32, i16, i8)
// ═══════════════════════════════════════════════════════════════════════════

macro_rules! binary_simd_i32 {
    ($name:ident, $op:expr) => {
        #[no_mangle]
        pub unsafe extern "C" fn $name(a: *const i32, b: *const i32, out: *mut i32, n: u32) {
            // Safe inner function that operates on slices, so we can use safe indexing and iterators.
            fn inner(sa: &[i32], sb: &[i32], so: &mut [i32]) {
                let len = sa.len();
                let mut i = 0;
                while i + 8 <= len {
                    store_i32x4(so, i, $op(load_i32x4(sa, i), load_i32x4(sb, i)));
                    store_i32x4(so, i + 4, $op(load_i32x4(sa, i + 4), load_i32x4(sb, i + 4)));
                    i += 8;
                }
                while i + 4 <= len {
                    store_i32x4(so, i, $op(load_i32x4(sa, i), load_i32x4(sb, i)));
                    i += 4;
                }
                while i < len {
                    so[i] = i32x4_extract_lane::<0>($op(i32x4_splat(sa[i]), i32x4_splat(sb[i])));
                    i += 1;
                }
            }

            // Call the inner function with slices; unsafe only for dereferencing & slice creation
            let len = n as usize;
            inner(
                core::slice::from_raw_parts(a, len),
                core::slice::from_raw_parts(b, len),
                core::slice::from_raw_parts_mut(out, len),
            );
        }
    };
}

macro_rules! binary_simd_i16 {
    ($name:ident, $op:expr) => {
        #[no_mangle]
        pub unsafe extern "C" fn $name(a: *const i16, b: *const i16, out: *mut i16, n: u32) {
            // Safe inner function that operates on slices, so we can use safe indexing and iterators.
            fn inner(sa: &[i16], sb: &[i16], so: &mut [i16]) {
                let len = sa.len();
                let mut i = 0;
                while i + 16 <= len {
                    store_i16x8(so, i, $op(load_i16x8(sa, i), load_i16x8(sb, i)));
                    store_i16x8(so, i + 8, $op(load_i16x8(sa, i + 8), load_i16x8(sb, i + 8)));
                    i += 16;
                }
                while i + 8 <= len {
                    store_i16x8(so, i, $op(load_i16x8(sa, i), load_i16x8(sb, i)));
                    i += 8;
                }
                while i < len {
                    so[i] = i16x8_extract_lane::<0>($op(i16x8_splat(sa[i]), i16x8_splat(sb[i])));
                    i += 1;
                }
            }

            // Call the inner function with slices; unsafe only for dereferencing & slice creation
            let len = n as usize;
            inner(
                core::slice::from_raw_parts(a, len),
                core::slice::from_raw_parts(b, len),
                core::slice::from_raw_parts_mut(out, len),
            );
        }
    };
}

macro_rules! binary_simd_i8 {
    ($name:ident, $op:expr) => {
        #[no_mangle]
        pub unsafe extern "C" fn $name(a: *const i8, b: *const i8, out: *mut i8, n: u32) {
            // Safe inner function that operates on slices, so we can use safe indexing and iterators.
            fn inner(sa: &[i8], sb: &[i8], so: &mut [i8]) {
                let len = sa.len();
                let mut i = 0;
                while i + 32 <= len {
                    store_i8x16(so, i, $op(load_i8x16(sa, i), load_i8x16(sb, i)));
                    store_i8x16(
                        so,
                        i + 16,
                        $op(load_i8x16(sa, i + 16), load_i8x16(sb, i + 16)),
                    );
                    i += 32;
                }
                while i + 16 <= len {
                    store_i8x16(so, i, $op(load_i8x16(sa, i), load_i8x16(sb, i)));
                    i += 16;
                }
                while i < len {
                    so[i] = i8x16_extract_lane::<0>($op(i8x16_splat(sa[i]), i8x16_splat(sb[i])));
                    i += 1;
                }
            }

            // Call the inner function with slices; unsafe only for dereferencing & slice creation
            let len = n as usize;
            inner(
                core::slice::from_raw_parts(a, len),
                core::slice::from_raw_parts(b, len),
                core::slice::from_raw_parts_mut(out, len),
            );
        }
    };
}

// i32 ops
binary_simd_i32!(add_i32, i32x4_add);
binary_simd_i32!(sub_i32, i32x4_sub);
binary_simd_i32!(mul_i32, i32x4_mul);
binary_simd_i32!(maximum_i32, i32x4_max);
binary_simd_i32!(minimum_i32, i32x4_min);

// i16 ops
binary_simd_i16!(add_i16, i16x8_add);
binary_simd_i16!(sub_i16, i16x8_sub);
binary_simd_i16!(mul_i16, i16x8_mul);
binary_simd_i16!(maximum_i16, i16x8_max);
binary_simd_i16!(minimum_i16, i16x8_min);

// i8 ops (no i8x16_mul in WASM SIMD)
binary_simd_i8!(add_i8, i8x16_add);
binary_simd_i8!(sub_i8, i8x16_sub);
binary_simd_i8!(maximum_i8, i8x16_max);
binary_simd_i8!(minimum_i8, i8x16_min);

// mul_i8: scalar fallback
fn mul_i8_inner(sa: &[i8], sb: &[i8], so: &mut [i8]) {
    for i in 0..sa.len() {
        so[i] = sa[i].wrapping_mul(sb[i]);
    }
}

#[no_mangle]
pub unsafe extern "C" fn mul_i8(a: *const i8, b: *const i8, out: *mut i8, n: u32) {
    let len = n as usize;
    mul_i8_inner(
        core::slice::from_raw_parts(a, len),
        core::slice::from_raw_parts(b, len),
        core::slice::from_raw_parts_mut(out, len),
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPLEX TYPES (c128, c64)
// ═══════════════════════════════════════════════════════════════════════════

#[no_mangle]
pub unsafe extern "C" fn add_c128(a: *const f64, b: *const f64, out: *mut f64, n: u32) {
    add_f64(a, b, out, n * 2);
}

#[no_mangle]
pub unsafe extern "C" fn add_c64(a: *const f32, b: *const f32, out: *mut f32, n: u32) {
    add_f32(a, b, out, n * 2);
}

fn mul_c128_inner(sa: &[f64], sb: &[f64], so: &mut [f64]) {
    let n = so.len() / 2;
    for i in 0..n {
        let (ar, ai) = (sa[2 * i], sa[2 * i + 1]);
        let (br, bi) = (sb[2 * i], sb[2 * i + 1]);
        so[2 * i] = ar * br - ai * bi;
        so[2 * i + 1] = ar * bi + ai * br;
    }
}

fn mul_c64_inner(sa: &[f32], sb: &[f32], so: &mut [f32]) {
    let n = so.len() / 2;
    for i in 0..n {
        let (ar, ai) = (sa[2 * i], sa[2 * i + 1]);
        let (br, bi) = (sb[2 * i], sb[2 * i + 1]);
        so[2 * i] = ar * br - ai * bi;
        so[2 * i + 1] = ar * bi + ai * br;
    }
}

#[no_mangle]
pub unsafe extern "C" fn mul_c128(a: *const f64, b: *const f64, out: *mut f64, n: u32) {
    let len = n as usize;
    mul_c128_inner(
        core::slice::from_raw_parts(a, len * 2),
        core::slice::from_raw_parts(b, len * 2),
        core::slice::from_raw_parts_mut(out, len * 2),
    );
}

#[no_mangle]
pub unsafe extern "C" fn mul_c64(a: *const f32, b: *const f32, out: *mut f32, n: u32) {
    let len = n as usize;
    mul_c64_inner(
        core::slice::from_raw_parts(a, len * 2),
        core::slice::from_raw_parts(b, len * 2),
        core::slice::from_raw_parts_mut(out, len * 2),
    );
}
