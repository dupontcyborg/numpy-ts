// Reduction kernels: sum and max for f32/f64
// Explicit WASM SIMD — LLVM can't autovectorize loop-carried dependencies
// without fast-math (which Rust doesn't expose)

use core::arch::wasm32::*;

#[no_mangle]
pub unsafe extern "C" fn sum_f64(ptr: *const f64, n: u32) -> f64 {
    let len = n as usize;
    let mut acc0 = f64x2_splat(0.0);
    let mut acc1 = f64x2_splat(0.0);
    let mut i = 0;
    // 4 f64s per iteration (2x v128)
    while i + 4 <= len {
        acc0 = f64x2_add(acc0, v128_load(ptr.add(i) as *const v128));
        acc1 = f64x2_add(acc1, v128_load(ptr.add(i + 2) as *const v128));
        i += 4;
    }
    while i + 2 <= len {
        acc0 = f64x2_add(acc0, v128_load(ptr.add(i) as *const v128));
        i += 2;
    }
    acc0 = f64x2_add(acc0, acc1);
    let mut result = f64x2_extract_lane::<0>(acc0) + f64x2_extract_lane::<1>(acc0);
    while i < len {
        result += *ptr.add(i);
        i += 1;
    }
    result
}

#[no_mangle]
pub unsafe extern "C" fn max_f64(ptr: *const f64, n: u32) -> f64 {
    let len = n as usize;
    if len == 0 {
        return f64::NEG_INFINITY;
    }
    let mut acc = f64x2_splat(*ptr);
    let mut i = 0;
    while i + 2 <= len {
        let v = v128_load(ptr.add(i) as *const v128);
        acc = f64x2_max(acc, v);
        i += 2;
    }
    let a = f64x2_extract_lane::<0>(acc);
    let b = f64x2_extract_lane::<1>(acc);
    let mut result = if a > b { a } else { b };
    while i < len {
        let v = *ptr.add(i);
        if v > result {
            result = v;
        }
        i += 1;
    }
    result
}

#[no_mangle]
pub unsafe extern "C" fn sum_f32(ptr: *const f32, n: u32) -> f32 {
    let len = n as usize;
    let mut acc0 = f32x4_splat(0.0);
    let mut acc1 = f32x4_splat(0.0);
    let mut i = 0;
    // 8 f32s per iteration (2x v128)
    while i + 8 <= len {
        acc0 = f32x4_add(acc0, v128_load(ptr.add(i) as *const v128));
        acc1 = f32x4_add(acc1, v128_load(ptr.add(i + 4) as *const v128));
        i += 8;
    }
    while i + 4 <= len {
        acc0 = f32x4_add(acc0, v128_load(ptr.add(i) as *const v128));
        i += 4;
    }
    acc0 = f32x4_add(acc0, acc1);
    let mut result = f32x4_extract_lane::<0>(acc0)
        + f32x4_extract_lane::<1>(acc0)
        + f32x4_extract_lane::<2>(acc0)
        + f32x4_extract_lane::<3>(acc0);
    while i < len {
        result += *ptr.add(i);
        i += 1;
    }
    result
}

#[no_mangle]
pub unsafe extern "C" fn max_f32(ptr: *const f32, n: u32) -> f32 {
    let len = n as usize;
    if len == 0 {
        return f32::NEG_INFINITY;
    }
    let mut acc = f32x4_splat(*ptr);
    let mut i = 0;
    while i + 4 <= len {
        let v = v128_load(ptr.add(i) as *const v128);
        acc = f32x4_max(acc, v);
        i += 4;
    }
    let mut result = f32x4_extract_lane::<0>(acc);
    let v1 = f32x4_extract_lane::<1>(acc);
    let v2 = f32x4_extract_lane::<2>(acc);
    let v3 = f32x4_extract_lane::<3>(acc);
    if v1 > result { result = v1; }
    if v2 > result { result = v2; }
    if v3 > result { result = v3; }
    while i < len {
        let v = *ptr.add(i);
        if v > result {
            result = v;
        }
        i += 1;
    }
    result
}
