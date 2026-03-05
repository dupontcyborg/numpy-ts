// Reduction kernels for f32/f64
// Explicit WASM SIMD — LLVM can't autovectorize loop-carried dependencies
// without fast-math (which Rust doesn't expose)

use crate::simd::{load_f32x4, load_f64x2, load_i32x4, store_f32x4, store_f64x2};
use core::arch::wasm32::*;

// ─── Safe inner functions ────────────────────────────────────────────────────

fn sum_f64_inner(data: &[f64]) -> f64 {
    let len = data.len();
    let mut acc0 = f64x2_splat(0.0);
    let mut acc1 = f64x2_splat(0.0);
    let mut acc2 = f64x2_splat(0.0);
    let mut acc3 = f64x2_splat(0.0);
    let mut i = 0;
    while i + 8 <= len {
        acc0 = f64x2_add(acc0, load_f64x2(data, i));
        acc1 = f64x2_add(acc1, load_f64x2(data, i + 2));
        acc2 = f64x2_add(acc2, load_f64x2(data, i + 4));
        acc3 = f64x2_add(acc3, load_f64x2(data, i + 6));
        i += 8;
    }
    acc0 = f64x2_add(acc0, acc2);
    acc1 = f64x2_add(acc1, acc3);
    while i + 4 <= len {
        acc0 = f64x2_add(acc0, load_f64x2(data, i));
        acc1 = f64x2_add(acc1, load_f64x2(data, i + 2));
        i += 4;
    }
    while i + 2 <= len {
        acc0 = f64x2_add(acc0, load_f64x2(data, i));
        i += 2;
    }
    acc0 = f64x2_add(acc0, acc1);
    let mut result = f64x2_extract_lane::<0>(acc0) + f64x2_extract_lane::<1>(acc0);
    while i < len {
        result += data[i];
        i += 1;
    }
    result
}

fn max_f64_inner(data: &[f64]) -> f64 {
    let len = data.len();
    if len == 0 {
        return f64::NEG_INFINITY;
    }
    let init = f64x2_splat(data[0]);
    let mut acc0 = init;
    let mut acc1 = init;
    let mut acc2 = init;
    let mut acc3 = init;
    let mut i = 0;
    while i + 8 <= len {
        acc0 = f64x2_max(acc0, load_f64x2(data, i));
        acc1 = f64x2_max(acc1, load_f64x2(data, i + 2));
        acc2 = f64x2_max(acc2, load_f64x2(data, i + 4));
        acc3 = f64x2_max(acc3, load_f64x2(data, i + 6));
        i += 8;
    }
    acc0 = f64x2_max(acc0, acc2);
    acc1 = f64x2_max(acc1, acc3);
    while i + 2 <= len {
        acc0 = f64x2_max(acc0, load_f64x2(data, i));
        i += 2;
    }
    acc0 = f64x2_max(acc0, acc1);
    let a = f64x2_extract_lane::<0>(acc0);
    let b = f64x2_extract_lane::<1>(acc0);
    let mut result = if a > b { a } else { b };
    while i < len {
        let v = data[i];
        if v > result {
            result = v;
        }
        i += 1;
    }
    result
}

fn min_f64_inner(data: &[f64]) -> f64 {
    let len = data.len();
    if len == 0 {
        return f64::INFINITY;
    }
    let init = f64x2_splat(data[0]);
    let mut acc0 = init;
    let mut acc1 = init;
    let mut acc2 = init;
    let mut acc3 = init;
    let mut i = 0;
    while i + 8 <= len {
        acc0 = f64x2_min(acc0, load_f64x2(data, i));
        acc1 = f64x2_min(acc1, load_f64x2(data, i + 2));
        acc2 = f64x2_min(acc2, load_f64x2(data, i + 4));
        acc3 = f64x2_min(acc3, load_f64x2(data, i + 6));
        i += 8;
    }
    acc0 = f64x2_min(acc0, acc2);
    acc1 = f64x2_min(acc1, acc3);
    while i + 2 <= len {
        acc0 = f64x2_min(acc0, load_f64x2(data, i));
        i += 2;
    }
    acc0 = f64x2_min(acc0, acc1);
    let a = f64x2_extract_lane::<0>(acc0);
    let b = f64x2_extract_lane::<1>(acc0);
    let mut result = if a < b { a } else { b };
    while i < len {
        let v = data[i];
        if v < result {
            result = v;
        }
        i += 1;
    }
    result
}

fn prod_f64_inner(data: &[f64]) -> f64 {
    let len = data.len();
    let mut acc0 = f64x2_splat(1.0);
    let mut acc1 = f64x2_splat(1.0);
    let mut i = 0;
    while i + 4 <= len {
        acc0 = f64x2_mul(acc0, load_f64x2(data, i));
        acc1 = f64x2_mul(acc1, load_f64x2(data, i + 2));
        i += 4;
    }
    while i + 2 <= len {
        acc0 = f64x2_mul(acc0, load_f64x2(data, i));
        i += 2;
    }
    acc0 = f64x2_mul(acc0, acc1);
    let mut result = f64x2_extract_lane::<0>(acc0) * f64x2_extract_lane::<1>(acc0);
    while i < len {
        result *= data[i];
        i += 1;
    }
    result
}

// ─── f64 FFI exports ─────────────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn sum_f64(ptr: *const f64, n: u32) -> f64 {
    sum_f64_inner(core::slice::from_raw_parts(ptr, n as usize))
}

#[no_mangle]
pub unsafe extern "C" fn max_f64(ptr: *const f64, n: u32) -> f64 {
    max_f64_inner(core::slice::from_raw_parts(ptr, n as usize))
}

#[no_mangle]
pub unsafe extern "C" fn min_f64(ptr: *const f64, n: u32) -> f64 {
    min_f64_inner(core::slice::from_raw_parts(ptr, n as usize))
}

#[no_mangle]
pub unsafe extern "C" fn prod_f64(ptr: *const f64, n: u32) -> f64 {
    prod_f64_inner(core::slice::from_raw_parts(ptr, n as usize))
}

#[no_mangle]
pub unsafe extern "C" fn mean_f64(ptr: *const f64, n: u32) -> f64 {
    sum_f64_inner(core::slice::from_raw_parts(ptr, n as usize)) / n as f64
}

// ─── f32 safe inner functions ────────────────────────────────────────────────

fn sum_f32_inner(data: &[f32]) -> f32 {
    let len = data.len();
    let mut acc0 = f32x4_splat(0.0);
    let mut acc1 = f32x4_splat(0.0);
    let mut acc2 = f32x4_splat(0.0);
    let mut acc3 = f32x4_splat(0.0);
    let mut i = 0;
    while i + 16 <= len {
        acc0 = f32x4_add(acc0, load_f32x4(data, i));
        acc1 = f32x4_add(acc1, load_f32x4(data, i + 4));
        acc2 = f32x4_add(acc2, load_f32x4(data, i + 8));
        acc3 = f32x4_add(acc3, load_f32x4(data, i + 12));
        i += 16;
    }
    acc0 = f32x4_add(acc0, acc2);
    acc1 = f32x4_add(acc1, acc3);
    while i + 8 <= len {
        acc0 = f32x4_add(acc0, load_f32x4(data, i));
        acc1 = f32x4_add(acc1, load_f32x4(data, i + 4));
        i += 8;
    }
    while i + 4 <= len {
        acc0 = f32x4_add(acc0, load_f32x4(data, i));
        i += 4;
    }
    acc0 = f32x4_add(acc0, acc1);
    let mut result = f32x4_extract_lane::<0>(acc0)
        + f32x4_extract_lane::<1>(acc0)
        + f32x4_extract_lane::<2>(acc0)
        + f32x4_extract_lane::<3>(acc0);
    while i < len {
        result += data[i];
        i += 1;
    }
    result
}

fn max_f32_inner(data: &[f32]) -> f32 {
    let len = data.len();
    if len == 0 {
        return f32::NEG_INFINITY;
    }
    let init = f32x4_splat(data[0]);
    let mut acc0 = init;
    let mut acc1 = init;
    let mut acc2 = init;
    let mut acc3 = init;
    let mut i = 0;
    while i + 16 <= len {
        acc0 = f32x4_max(acc0, load_f32x4(data, i));
        acc1 = f32x4_max(acc1, load_f32x4(data, i + 4));
        acc2 = f32x4_max(acc2, load_f32x4(data, i + 8));
        acc3 = f32x4_max(acc3, load_f32x4(data, i + 12));
        i += 16;
    }
    acc0 = f32x4_max(acc0, acc2);
    acc1 = f32x4_max(acc1, acc3);
    while i + 4 <= len {
        acc0 = f32x4_max(acc0, load_f32x4(data, i));
        i += 4;
    }
    acc0 = f32x4_max(acc0, acc1);
    let mut result = f32x4_extract_lane::<0>(acc0);
    let v1 = f32x4_extract_lane::<1>(acc0);
    let v2 = f32x4_extract_lane::<2>(acc0);
    let v3 = f32x4_extract_lane::<3>(acc0);
    if v1 > result {
        result = v1;
    }
    if v2 > result {
        result = v2;
    }
    if v3 > result {
        result = v3;
    }
    while i < len {
        let v = data[i];
        if v > result {
            result = v;
        }
        i += 1;
    }
    result
}

fn min_f32_inner(data: &[f32]) -> f32 {
    let len = data.len();
    if len == 0 {
        return f32::INFINITY;
    }
    let init = f32x4_splat(data[0]);
    let mut acc0 = init;
    let mut acc1 = init;
    let mut acc2 = init;
    let mut acc3 = init;
    let mut i = 0;
    while i + 16 <= len {
        acc0 = f32x4_min(acc0, load_f32x4(data, i));
        acc1 = f32x4_min(acc1, load_f32x4(data, i + 4));
        acc2 = f32x4_min(acc2, load_f32x4(data, i + 8));
        acc3 = f32x4_min(acc3, load_f32x4(data, i + 12));
        i += 16;
    }
    acc0 = f32x4_min(acc0, acc2);
    acc1 = f32x4_min(acc1, acc3);
    while i + 4 <= len {
        acc0 = f32x4_min(acc0, load_f32x4(data, i));
        i += 4;
    }
    acc0 = f32x4_min(acc0, acc1);
    let mut result = f32x4_extract_lane::<0>(acc0);
    let v1 = f32x4_extract_lane::<1>(acc0);
    let v2 = f32x4_extract_lane::<2>(acc0);
    let v3 = f32x4_extract_lane::<3>(acc0);
    if v1 < result {
        result = v1;
    }
    if v2 < result {
        result = v2;
    }
    if v3 < result {
        result = v3;
    }
    while i < len {
        let v = data[i];
        if v < result {
            result = v;
        }
        i += 1;
    }
    result
}

fn prod_f32_inner(data: &[f32]) -> f32 {
    let len = data.len();
    let mut acc0 = f32x4_splat(1.0);
    let mut acc1 = f32x4_splat(1.0);
    let mut acc2 = f32x4_splat(1.0);
    let mut acc3 = f32x4_splat(1.0);
    let mut i = 0;
    while i + 16 <= len {
        acc0 = f32x4_mul(acc0, load_f32x4(data, i));
        acc1 = f32x4_mul(acc1, load_f32x4(data, i + 4));
        acc2 = f32x4_mul(acc2, load_f32x4(data, i + 8));
        acc3 = f32x4_mul(acc3, load_f32x4(data, i + 12));
        i += 16;
    }
    acc0 = f32x4_mul(acc0, acc2);
    acc1 = f32x4_mul(acc1, acc3);
    while i + 8 <= len {
        acc0 = f32x4_mul(acc0, load_f32x4(data, i));
        acc1 = f32x4_mul(acc1, load_f32x4(data, i + 4));
        i += 8;
    }
    while i + 4 <= len {
        acc0 = f32x4_mul(acc0, load_f32x4(data, i));
        i += 4;
    }
    acc0 = f32x4_mul(acc0, acc1);
    let mut result = f32x4_extract_lane::<0>(acc0)
        * f32x4_extract_lane::<1>(acc0)
        * f32x4_extract_lane::<2>(acc0)
        * f32x4_extract_lane::<3>(acc0);
    while i < len {
        result *= data[i];
        i += 1;
    }
    result
}

// ─── f32 FFI exports ─────────────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn sum_f32(ptr: *const f32, n: u32) -> f32 {
    sum_f32_inner(core::slice::from_raw_parts(ptr, n as usize))
}

#[no_mangle]
pub unsafe extern "C" fn max_f32(ptr: *const f32, n: u32) -> f32 {
    max_f32_inner(core::slice::from_raw_parts(ptr, n as usize))
}

#[no_mangle]
pub unsafe extern "C" fn min_f32(ptr: *const f32, n: u32) -> f32 {
    min_f32_inner(core::slice::from_raw_parts(ptr, n as usize))
}

#[no_mangle]
pub unsafe extern "C" fn prod_f32(ptr: *const f32, n: u32) -> f32 {
    prod_f32_inner(core::slice::from_raw_parts(ptr, n as usize))
}

#[no_mangle]
pub unsafe extern "C" fn mean_f32(ptr: *const f32, n: u32) -> f32 {
    sum_f32_inner(core::slice::from_raw_parts(ptr, n as usize)) / n as f32
}

// ─── nanmax: max ignoring NaN ───────────────────────────────────────────────

fn nanmax_f64_inner(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NEG_INFINITY;
    }
    let mut start = 0;
    while start < data.len() && data[start].is_nan() {
        start += 1;
    }
    if start == data.len() {
        return data[0];
    }
    let mut result = data[start];
    for i in (start + 1)..data.len() {
        let v = data[i];
        if !v.is_nan() && v > result {
            result = v;
        }
    }
    result
}

fn nanmax_f32_inner(data: &[f32]) -> f32 {
    if data.is_empty() {
        return f32::NEG_INFINITY;
    }
    let mut start = 0;
    while start < data.len() && data[start].is_nan() {
        start += 1;
    }
    if start == data.len() {
        return data[0];
    }
    let mut result = data[start];
    for i in (start + 1)..data.len() {
        let v = data[i];
        if !v.is_nan() && v > result {
            result = v;
        }
    }
    result
}

#[no_mangle]
pub unsafe extern "C" fn nanmax_f64(ptr: *const f64, n: u32) -> f64 {
    nanmax_f64_inner(core::slice::from_raw_parts(ptr, n as usize))
}

#[no_mangle]
pub unsafe extern "C" fn nanmax_f32(ptr: *const f32, n: u32) -> f32 {
    nanmax_f32_inner(core::slice::from_raw_parts(ptr, n as usize))
}

// ─── nanmin: min ignoring NaN ───────────────────────────────────────────────

fn nanmin_f64_inner(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::INFINITY;
    }
    let mut start = 0;
    while start < data.len() && data[start].is_nan() {
        start += 1;
    }
    if start == data.len() {
        return data[0];
    }
    let mut result = data[start];
    for i in (start + 1)..data.len() {
        let v = data[i];
        if !v.is_nan() && v < result {
            result = v;
        }
    }
    result
}

fn nanmin_f32_inner(data: &[f32]) -> f32 {
    if data.is_empty() {
        return f32::INFINITY;
    }
    let mut start = 0;
    while start < data.len() && data[start].is_nan() {
        start += 1;
    }
    if start == data.len() {
        return data[0];
    }
    let mut result = data[start];
    for i in (start + 1)..data.len() {
        let v = data[i];
        if !v.is_nan() && v < result {
            result = v;
        }
    }
    result
}

#[no_mangle]
pub unsafe extern "C" fn nanmin_f64(ptr: *const f64, n: u32) -> f64 {
    nanmin_f64_inner(core::slice::from_raw_parts(ptr, n as usize))
}

#[no_mangle]
pub unsafe extern "C" fn nanmin_f32(ptr: *const f32, n: u32) -> f32 {
    nanmin_f32_inner(core::slice::from_raw_parts(ptr, n as usize))
}

// ─── diff: first-order differences ──────────────────────────────────────────

fn diff_f64_inner(input: &[f64], output: &mut [f64]) {
    let len = input.len();
    if len <= 1 {
        return;
    }
    let out_len = len - 1;
    let mut i = 0;
    while i + 2 <= out_len {
        let v0 = load_f64x2(input, i);
        let v1 = load_f64x2(input, i + 1);
        store_f64x2(output, i, f64x2_sub(v1, v0));
        i += 2;
    }
    while i < out_len {
        output[i] = input[i + 1] - input[i];
        i += 1;
    }
}

fn diff_f32_inner(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    if len <= 1 {
        return;
    }
    let out_len = len - 1;
    let mut i = 0;
    while i + 4 <= out_len {
        let v0 = load_f32x4(input, i);
        let v1 = load_f32x4(input, i + 1);
        store_f32x4(output, i, f32x4_sub(v1, v0));
        i += 4;
    }
    while i < out_len {
        output[i] = input[i + 1] - input[i];
        i += 1;
    }
}

#[no_mangle]
pub unsafe extern "C" fn diff_f64(inp: *const f64, out: *mut f64, n: u32) {
    let len = n as usize;
    if len <= 1 {
        return;
    }
    diff_f64_inner(
        core::slice::from_raw_parts(inp, len),
        core::slice::from_raw_parts_mut(out, len - 1),
    );
}

#[no_mangle]
pub unsafe extern "C" fn diff_f32(inp: *const f32, out: *mut f32, n: u32) {
    let len = n as usize;
    if len <= 1 {
        return;
    }
    diff_f32_inner(
        core::slice::from_raw_parts(inp, len),
        core::slice::from_raw_parts_mut(out, len - 1),
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// INTEGER REDUCTIONS (i32, i16, i8)
// ═══════════════════════════════════════════════════════════════════════════

fn sum_i32_inner(data: &[i32]) -> i32 {
    let len = data.len();
    let mut acc0 = i32x4_splat(0);
    let mut acc1 = i32x4_splat(0);
    let mut acc2 = i32x4_splat(0);
    let mut acc3 = i32x4_splat(0);
    let mut i = 0;
    while i + 16 <= len {
        acc0 = i32x4_add(acc0, load_i32x4(data, i));
        acc1 = i32x4_add(acc1, load_i32x4(data, i + 4));
        acc2 = i32x4_add(acc2, load_i32x4(data, i + 8));
        acc3 = i32x4_add(acc3, load_i32x4(data, i + 12));
        i += 16;
    }
    acc0 = i32x4_add(acc0, acc2);
    acc1 = i32x4_add(acc1, acc3);
    while i + 8 <= len {
        acc0 = i32x4_add(acc0, load_i32x4(data, i));
        acc1 = i32x4_add(acc1, load_i32x4(data, i + 4));
        i += 8;
    }
    while i + 4 <= len {
        acc0 = i32x4_add(acc0, load_i32x4(data, i));
        i += 4;
    }
    acc0 = i32x4_add(acc0, acc1);
    let mut result = i32x4_extract_lane::<0>(acc0)
        .wrapping_add(i32x4_extract_lane::<1>(acc0))
        .wrapping_add(i32x4_extract_lane::<2>(acc0))
        .wrapping_add(i32x4_extract_lane::<3>(acc0));
    while i < len {
        result = result.wrapping_add(data[i]);
        i += 1;
    }
    result
}

fn max_i32_inner(data: &[i32]) -> i32 {
    let len = data.len();
    if len == 0 {
        return i32::MIN;
    }
    let init = i32x4_splat(data[0]);
    let mut acc0 = init;
    let mut acc1 = init;
    let mut acc2 = init;
    let mut acc3 = init;
    let mut i = 0;
    while i + 16 <= len {
        acc0 = i32x4_max(acc0, load_i32x4(data, i));
        acc1 = i32x4_max(acc1, load_i32x4(data, i + 4));
        acc2 = i32x4_max(acc2, load_i32x4(data, i + 8));
        acc3 = i32x4_max(acc3, load_i32x4(data, i + 12));
        i += 16;
    }
    acc0 = i32x4_max(acc0, acc2);
    acc1 = i32x4_max(acc1, acc3);
    while i + 4 <= len {
        acc0 = i32x4_max(acc0, load_i32x4(data, i));
        i += 4;
    }
    acc0 = i32x4_max(acc0, acc1);
    let mut result = i32x4_extract_lane::<0>(acc0);
    let v1 = i32x4_extract_lane::<1>(acc0);
    let v2 = i32x4_extract_lane::<2>(acc0);
    let v3 = i32x4_extract_lane::<3>(acc0);
    if v1 > result {
        result = v1;
    }
    if v2 > result {
        result = v2;
    }
    if v3 > result {
        result = v3;
    }
    while i < len {
        let v = data[i];
        if v > result {
            result = v;
        }
        i += 1;
    }
    result
}

fn min_i32_inner(data: &[i32]) -> i32 {
    let len = data.len();
    if len == 0 {
        return i32::MAX;
    }
    let init = i32x4_splat(data[0]);
    let mut acc0 = init;
    let mut acc1 = init;
    let mut acc2 = init;
    let mut acc3 = init;
    let mut i = 0;
    while i + 16 <= len {
        acc0 = i32x4_min(acc0, load_i32x4(data, i));
        acc1 = i32x4_min(acc1, load_i32x4(data, i + 4));
        acc2 = i32x4_min(acc2, load_i32x4(data, i + 8));
        acc3 = i32x4_min(acc3, load_i32x4(data, i + 12));
        i += 16;
    }
    acc0 = i32x4_min(acc0, acc2);
    acc1 = i32x4_min(acc1, acc3);
    while i + 4 <= len {
        acc0 = i32x4_min(acc0, load_i32x4(data, i));
        i += 4;
    }
    acc0 = i32x4_min(acc0, acc1);
    let mut result = i32x4_extract_lane::<0>(acc0);
    let v1 = i32x4_extract_lane::<1>(acc0);
    let v2 = i32x4_extract_lane::<2>(acc0);
    let v3 = i32x4_extract_lane::<3>(acc0);
    if v1 < result {
        result = v1;
    }
    if v2 < result {
        result = v2;
    }
    if v3 < result {
        result = v3;
    }
    while i < len {
        let v = data[i];
        if v < result {
            result = v;
        }
        i += 1;
    }
    result
}

#[no_mangle]
pub unsafe extern "C" fn sum_i32(ptr: *const i32, n: u32) -> i32 {
    sum_i32_inner(core::slice::from_raw_parts(ptr, n as usize))
}

#[no_mangle]
pub unsafe extern "C" fn max_i32(ptr: *const i32, n: u32) -> i32 {
    max_i32_inner(core::slice::from_raw_parts(ptr, n as usize))
}

#[no_mangle]
pub unsafe extern "C" fn min_i32(ptr: *const i32, n: u32) -> i32 {
    min_i32_inner(core::slice::from_raw_parts(ptr, n as usize))
}

// i16 reductions (widen to i32)

fn sum_i16_inner(data: &[i16]) -> i32 {
    let mut result: i32 = 0;
    for &v in data {
        result = result.wrapping_add(v as i32);
    }
    result
}

fn max_i16_inner(data: &[i16]) -> i16 {
    if data.is_empty() {
        return i16::MIN;
    }
    let mut result = data[0];
    for &v in &data[1..] {
        if v > result {
            result = v;
        }
    }
    result
}

fn min_i16_inner(data: &[i16]) -> i16 {
    if data.is_empty() {
        return i16::MAX;
    }
    let mut result = data[0];
    for &v in &data[1..] {
        if v < result {
            result = v;
        }
    }
    result
}

#[no_mangle]
pub unsafe extern "C" fn sum_i16(ptr: *const i16, n: u32) -> i32 {
    sum_i16_inner(core::slice::from_raw_parts(ptr, n as usize))
}

#[no_mangle]
pub unsafe extern "C" fn max_i16(ptr: *const i16, n: u32) -> i16 {
    max_i16_inner(core::slice::from_raw_parts(ptr, n as usize))
}

#[no_mangle]
pub unsafe extern "C" fn min_i16(ptr: *const i16, n: u32) -> i16 {
    min_i16_inner(core::slice::from_raw_parts(ptr, n as usize))
}

// i8 reductions (widen to i32)

fn sum_i8_inner(data: &[i8]) -> i32 {
    let mut result: i32 = 0;
    for &v in data {
        result = result.wrapping_add(v as i32);
    }
    result
}

fn max_i8_inner(data: &[i8]) -> i8 {
    if data.is_empty() {
        return i8::MIN;
    }
    let mut result = data[0];
    for &v in &data[1..] {
        if v > result {
            result = v;
        }
    }
    result
}

fn min_i8_inner(data: &[i8]) -> i8 {
    if data.is_empty() {
        return i8::MAX;
    }
    let mut result = data[0];
    for &v in &data[1..] {
        if v < result {
            result = v;
        }
    }
    result
}

#[no_mangle]
pub unsafe extern "C" fn sum_i8(ptr: *const i8, n: u32) -> i32 {
    sum_i8_inner(core::slice::from_raw_parts(ptr, n as usize))
}

#[no_mangle]
pub unsafe extern "C" fn max_i8(ptr: *const i8, n: u32) -> i8 {
    max_i8_inner(core::slice::from_raw_parts(ptr, n as usize))
}

#[no_mangle]
pub unsafe extern "C" fn min_i8(ptr: *const i8, n: u32) -> i8 {
    min_i8_inner(core::slice::from_raw_parts(ptr, n as usize))
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPLEX REDUCTIONS (c128, c64)
// ═══════════════════════════════════════════════════════════════════════════

fn sum_c128_inner(data: &[f64], out: &mut [f64]) {
    let n = data.len() / 2;
    let mut acc0 = f64x2_splat(0.0);
    let mut acc1 = f64x2_splat(0.0);
    let mut i = 0;
    while i + 4 <= n {
        acc0 = f64x2_add(acc0, load_f64x2(data, i * 2));
        acc0 = f64x2_add(acc0, load_f64x2(data, (i + 1) * 2));
        acc1 = f64x2_add(acc1, load_f64x2(data, (i + 2) * 2));
        acc1 = f64x2_add(acc1, load_f64x2(data, (i + 3) * 2));
        i += 4;
    }
    while i < n {
        acc0 = f64x2_add(acc0, load_f64x2(data, i * 2));
        i += 1;
    }
    acc0 = f64x2_add(acc0, acc1);
    out[0] = f64x2_extract_lane::<0>(acc0);
    out[1] = f64x2_extract_lane::<1>(acc0);
}

fn sum_c64_inner(data: &[f32], out: &mut [f32]) {
    let n = data.len() / 2;
    let mut re_sum: f32 = 0.0;
    let mut im_sum: f32 = 0.0;
    for i in 0..n {
        re_sum += data[2 * i];
        im_sum += data[2 * i + 1];
    }
    out[0] = re_sum;
    out[1] = im_sum;
}

#[no_mangle]
pub unsafe extern "C" fn sum_c128(ptr: *const f64, out: *mut f64, n: u32) {
    let len = n as usize;
    sum_c128_inner(
        core::slice::from_raw_parts(ptr, len * 2),
        core::slice::from_raw_parts_mut(out, 2),
    );
}

#[no_mangle]
pub unsafe extern "C" fn sum_c64(ptr: *const f32, out: *mut f32, n: u32) {
    let len = n as usize;
    sum_c64_inner(
        core::slice::from_raw_parts(ptr, len * 2),
        core::slice::from_raw_parts_mut(out, 2),
    );
}
