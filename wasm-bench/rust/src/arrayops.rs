// Array operation kernels: roll, flip, tile, pad, take, gradient

use crate::simd::{load_f32x4, load_f64x2, store_f32x4, store_f64x2};
use core::arch::wasm32::*;

// ─── SIMD copy/zero helpers (safe, slice-based) ─────────────────────────────

fn simd_copy_f64(dst: &mut [f64], dst_off: usize, src: &[f64], src_off: usize, n: usize) {
    let mut i = 0;
    while i + 4 <= n {
        store_f64x2(dst, dst_off + i, load_f64x2(src, src_off + i));
        store_f64x2(dst, dst_off + i + 2, load_f64x2(src, src_off + i + 2));
        i += 4;
    }
    while i + 2 <= n {
        store_f64x2(dst, dst_off + i, load_f64x2(src, src_off + i));
        i += 2;
    }
    while i < n {
        dst[dst_off + i] = src[src_off + i];
        i += 1;
    }
}

fn simd_copy_f32(dst: &mut [f32], dst_off: usize, src: &[f32], src_off: usize, n: usize) {
    let mut i = 0;
    while i + 8 <= n {
        store_f32x4(dst, dst_off + i, load_f32x4(src, src_off + i));
        store_f32x4(dst, dst_off + i + 4, load_f32x4(src, src_off + i + 4));
        i += 8;
    }
    while i + 4 <= n {
        store_f32x4(dst, dst_off + i, load_f32x4(src, src_off + i));
        i += 4;
    }
    while i < n {
        dst[dst_off + i] = src[src_off + i];
        i += 1;
    }
}

fn simd_zero_f64(dst: &mut [f64], off: usize, n: usize) {
    let zero = f64x2_splat(0.0);
    let mut i = 0;
    while i + 2 <= n {
        store_f64x2(dst, off + i, zero);
        i += 2;
    }
    while i < n {
        dst[off + i] = 0.0;
        i += 1;
    }
}

fn simd_zero_f32(dst: &mut [f32], off: usize, n: usize) {
    let zero = f32x4_splat(0.0);
    let mut i = 0;
    while i + 4 <= n {
        store_f32x4(dst, off + i, zero);
        i += 4;
    }
    while i < n {
        dst[off + i] = 0.0;
        i += 1;
    }
}

// ─── roll: circular shift ───────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn roll_f64(inp: *const f64, out: *mut f64, n: u32, shift: i32) {
    let len = n as usize;
    if len == 0 {
        return;
    }
    let input = core::slice::from_raw_parts(inp, len);
    let output = core::slice::from_raw_parts_mut(out, len);
    let s = ((shift as i64).rem_euclid(len as i64)) as usize;
    if s == 0 {
        simd_copy_f64(output, 0, input, 0, len);
        return;
    }
    simd_copy_f64(output, 0, input, len - s, s);
    simd_copy_f64(output, s, input, 0, len - s);
}

#[no_mangle]
pub unsafe extern "C" fn roll_f32(inp: *const f32, out: *mut f32, n: u32, shift: i32) {
    let len = n as usize;
    if len == 0 {
        return;
    }
    let input = core::slice::from_raw_parts(inp, len);
    let output = core::slice::from_raw_parts_mut(out, len);
    let s = ((shift as i64).rem_euclid(len as i64)) as usize;
    if s == 0 {
        simd_copy_f32(output, 0, input, 0, len);
        return;
    }
    simd_copy_f32(output, 0, input, len - s, s);
    simd_copy_f32(output, s, input, 0, len - s);
}

// ─── flip: reverse array ────────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn flip_f64(inp: *const f64, out: *mut f64, n: u32) {
    let len = n as usize;
    let input = core::slice::from_raw_parts(inp, len);
    let output = core::slice::from_raw_parts_mut(out, len);
    for i in 0..len {
        output[i] = input[len - 1 - i];
    }
}

#[no_mangle]
pub unsafe extern "C" fn flip_f32(inp: *const f32, out: *mut f32, n: u32) {
    let len = n as usize;
    let input = core::slice::from_raw_parts(inp, len);
    let output = core::slice::from_raw_parts_mut(out, len);
    for i in 0..len {
        output[i] = input[len - 1 - i];
    }
}

// ─── tile: repeat array ─────────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn tile_f64(inp: *const f64, out: *mut f64, n: u32, reps: u32) {
    let len = n as usize;
    let input = core::slice::from_raw_parts(inp, len);
    let output = core::slice::from_raw_parts_mut(out, len * reps as usize);
    for rep in 0..reps as usize {
        simd_copy_f64(output, rep * len, input, 0, len);
    }
}

#[no_mangle]
pub unsafe extern "C" fn tile_f32(inp: *const f32, out: *mut f32, n: u32, reps: u32) {
    let len = n as usize;
    let input = core::slice::from_raw_parts(inp, len);
    let output = core::slice::from_raw_parts_mut(out, len * reps as usize);
    for rep in 0..reps as usize {
        simd_copy_f32(output, rep * len, input, 0, len);
    }
}

// ─── pad: zero-pad 2D array ─────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn pad_f64(inp: *const f64, out: *mut f64, rows: u32, cols: u32, pw: u32) {
    let (r, c, p) = (rows as usize, cols as usize, pw as usize);
    let out_cols = c + 2 * p;
    let input = core::slice::from_raw_parts(inp, r * c);
    let output = core::slice::from_raw_parts_mut(out, (r + 2 * p) * out_cols);
    simd_zero_f64(output, 0, (r + 2 * p) * out_cols);
    for i in 0..r {
        simd_copy_f64(output, (i + p) * out_cols + p, input, i * c, c);
    }
}

#[no_mangle]
pub unsafe extern "C" fn pad_f32(inp: *const f32, out: *mut f32, rows: u32, cols: u32, pw: u32) {
    let (r, c, p) = (rows as usize, cols as usize, pw as usize);
    let out_cols = c + 2 * p;
    let input = core::slice::from_raw_parts(inp, r * c);
    let output = core::slice::from_raw_parts_mut(out, (r + 2 * p) * out_cols);
    simd_zero_f32(output, 0, (r + 2 * p) * out_cols);
    for i in 0..r {
        simd_copy_f32(output, (i + p) * out_cols + p, input, i * c, c);
    }
}

// ─── take: gather by index ──────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn take_f64(data: *const f64, indices: *const u32, out: *mut f64, n: u32) {
    let len = n as usize;
    let idx = core::slice::from_raw_parts(indices, len);
    let output = core::slice::from_raw_parts_mut(out, len);
    // data length unknown at FFI boundary — indices index into data
    for i in 0..len {
        output[i] = *data.add(idx[i] as usize);
    }
}

#[no_mangle]
pub unsafe extern "C" fn take_f32(data: *const f32, indices: *const u32, out: *mut f32, n: u32) {
    let len = n as usize;
    let idx = core::slice::from_raw_parts(indices, len);
    let output = core::slice::from_raw_parts_mut(out, len);
    // data length unknown at FFI boundary — indices index into data
    for i in 0..len {
        output[i] = *data.add(idx[i] as usize);
    }
}

// ─── gradient: numerical gradient (central differences) ─────────────────────

fn gradient_f64_inner(input: &[f64], output: &mut [f64]) {
    let len = input.len();
    if len < 2 {
        return;
    }
    output[0] = input[1] - input[0];
    output[len - 1] = input[len - 1] - input[len - 2];
    if len <= 2 {
        return;
    }
    let half = f64x2_splat(0.5);
    let mut i = 1;
    while i + 2 < len {
        let fwd = load_f64x2(input, i + 1);
        let bwd = load_f64x2(input, i - 1);
        store_f64x2(output, i, f64x2_mul(f64x2_sub(fwd, bwd), half));
        i += 2;
    }
    while i < len - 1 {
        output[i] = (input[i + 1] - input[i - 1]) * 0.5;
        i += 1;
    }
}

fn gradient_f32_inner(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    if len < 2 {
        return;
    }
    output[0] = input[1] - input[0];
    output[len - 1] = input[len - 1] - input[len - 2];
    if len <= 2 {
        return;
    }
    let half = f32x4_splat(0.5);
    let mut i = 1;
    while i + 4 < len {
        let fwd = load_f32x4(input, i + 1);
        let bwd = load_f32x4(input, i - 1);
        store_f32x4(output, i, f32x4_mul(f32x4_sub(fwd, bwd), half));
        i += 4;
    }
    while i < len - 1 {
        output[i] = (input[i + 1] - input[i - 1]) * 0.5;
        i += 1;
    }
}

#[no_mangle]
pub unsafe extern "C" fn gradient_f64(inp: *const f64, out: *mut f64, n: u32) {
    let len = n as usize;
    gradient_f64_inner(
        core::slice::from_raw_parts(inp, len),
        core::slice::from_raw_parts_mut(out, len),
    );
}

#[no_mangle]
pub unsafe extern "C" fn gradient_f32(inp: *const f32, out: *mut f32, n: u32) {
    let len = n as usize;
    gradient_f32_inner(
        core::slice::from_raw_parts(inp, len),
        core::slice::from_raw_parts_mut(out, len),
    );
}

// ─── nonzero: return indices of non-zero elements ────────────────────────

#[no_mangle]
pub unsafe extern "C" fn nonzero_f64(ptr: *const f64, out: *mut u32, n: u32) -> u32 {
    let len = n as usize;
    let data = core::slice::from_raw_parts(ptr, len);
    let output = core::slice::from_raw_parts_mut(out, len);
    let mut count: usize = 0;
    for i in 0..len {
        if data[i] != 0.0 {
            output[count] = i as u32;
            count += 1;
        }
    }
    count as u32
}

#[no_mangle]
pub unsafe extern "C" fn nonzero_f32(ptr: *const f32, out: *mut u32, n: u32) -> u32 {
    let len = n as usize;
    let data = core::slice::from_raw_parts(ptr, len);
    let output = core::slice::from_raw_parts_mut(out, len);
    let mut count: usize = 0;
    for i in 0..len {
        if data[i] != 0.0 {
            output[count] = i as u32;
            count += 1;
        }
    }
    count as u32
}
