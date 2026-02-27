// Unary elementwise kernels: sqrt and exp for f32/f64
// sqrt: explicit WASM SIMD (f64x2_sqrt / f32x4_sqrt), double-pumped to match Zig
// exp: libm (no SIMD equivalent)

use core::arch::wasm32::*;

#[no_mangle]
pub unsafe extern "C" fn sqrt_f64(in_ptr: *const f64, out_ptr: *mut f64, n: u32) {
    let len = n as usize;
    let mut i = 0;
    // 4 f64s per iteration (2x f64x2_sqrt)
    while i + 4 <= len {
        let v0 = v128_load(in_ptr.add(i) as *const v128);
        let v1 = v128_load(in_ptr.add(i + 2) as *const v128);
        v128_store(out_ptr.add(i) as *mut v128, f64x2_sqrt(v0));
        v128_store(out_ptr.add(i + 2) as *mut v128, f64x2_sqrt(v1));
        i += 4;
    }
    while i + 2 <= len {
        let v = v128_load(in_ptr.add(i) as *const v128);
        v128_store(out_ptr.add(i) as *mut v128, f64x2_sqrt(v));
        i += 2;
    }
    while i < len {
        *out_ptr.add(i) = libm::sqrt(*in_ptr.add(i));
        i += 1;
    }
}

#[no_mangle]
pub unsafe extern "C" fn exp_f64(in_ptr: *const f64, out_ptr: *mut f64, n: u32) {
    let len = n as usize;
    for i in 0..len {
        *out_ptr.add(i) = libm::exp(*in_ptr.add(i));
    }
}

#[no_mangle]
pub unsafe extern "C" fn sqrt_f32(in_ptr: *const f32, out_ptr: *mut f32, n: u32) {
    let len = n as usize;
    let mut i = 0;
    // 8 f32s per iteration (2x f32x4_sqrt)
    while i + 8 <= len {
        let v0 = v128_load(in_ptr.add(i) as *const v128);
        let v1 = v128_load(in_ptr.add(i + 4) as *const v128);
        v128_store(out_ptr.add(i) as *mut v128, f32x4_sqrt(v0));
        v128_store(out_ptr.add(i + 4) as *mut v128, f32x4_sqrt(v1));
        i += 8;
    }
    while i + 4 <= len {
        let v = v128_load(in_ptr.add(i) as *const v128);
        v128_store(out_ptr.add(i) as *mut v128, f32x4_sqrt(v));
        i += 4;
    }
    while i < len {
        *out_ptr.add(i) = libm::sqrtf(*in_ptr.add(i));
        i += 1;
    }
}

#[no_mangle]
pub unsafe extern "C" fn exp_f32(in_ptr: *const f32, out_ptr: *mut f32, n: u32) {
    let len = n as usize;
    for i in 0..len {
        *out_ptr.add(i) = libm::expf(*in_ptr.add(i));
    }
}
