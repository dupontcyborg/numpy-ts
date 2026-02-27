// Binary elementwise kernels: add and mul for f32/f64
// Explicit WASM SIMD for consistent performance at all sizes

use core::arch::wasm32::*;

#[no_mangle]
pub unsafe extern "C" fn add_f64(a: *const f64, b: *const f64, out: *mut f64, n: u32) {
    let len = n as usize;
    let mut i = 0;
    while i + 4 <= len {
        let va0 = v128_load(a.add(i) as *const v128);
        let vb0 = v128_load(b.add(i) as *const v128);
        let va1 = v128_load(a.add(i + 2) as *const v128);
        let vb1 = v128_load(b.add(i + 2) as *const v128);
        v128_store(out.add(i) as *mut v128, f64x2_add(va0, vb0));
        v128_store(out.add(i + 2) as *mut v128, f64x2_add(va1, vb1));
        i += 4;
    }
    while i + 2 <= len {
        let va = v128_load(a.add(i) as *const v128);
        let vb = v128_load(b.add(i) as *const v128);
        v128_store(out.add(i) as *mut v128, f64x2_add(va, vb));
        i += 2;
    }
    while i < len {
        *out.add(i) = *a.add(i) + *b.add(i);
        i += 1;
    }
}

#[no_mangle]
pub unsafe extern "C" fn mul_f64(a: *const f64, b: *const f64, out: *mut f64, n: u32) {
    let len = n as usize;
    let mut i = 0;
    while i + 4 <= len {
        let va0 = v128_load(a.add(i) as *const v128);
        let vb0 = v128_load(b.add(i) as *const v128);
        let va1 = v128_load(a.add(i + 2) as *const v128);
        let vb1 = v128_load(b.add(i + 2) as *const v128);
        v128_store(out.add(i) as *mut v128, f64x2_mul(va0, vb0));
        v128_store(out.add(i + 2) as *mut v128, f64x2_mul(va1, vb1));
        i += 4;
    }
    while i + 2 <= len {
        let va = v128_load(a.add(i) as *const v128);
        let vb = v128_load(b.add(i) as *const v128);
        v128_store(out.add(i) as *mut v128, f64x2_mul(va, vb));
        i += 2;
    }
    while i < len {
        *out.add(i) = *a.add(i) * *b.add(i);
        i += 1;
    }
}

#[no_mangle]
pub unsafe extern "C" fn add_f32(a: *const f32, b: *const f32, out: *mut f32, n: u32) {
    let len = n as usize;
    let mut i = 0;
    while i + 8 <= len {
        let va0 = v128_load(a.add(i) as *const v128);
        let vb0 = v128_load(b.add(i) as *const v128);
        let va1 = v128_load(a.add(i + 4) as *const v128);
        let vb1 = v128_load(b.add(i + 4) as *const v128);
        v128_store(out.add(i) as *mut v128, f32x4_add(va0, vb0));
        v128_store(out.add(i + 4) as *mut v128, f32x4_add(va1, vb1));
        i += 8;
    }
    while i + 4 <= len {
        let va = v128_load(a.add(i) as *const v128);
        let vb = v128_load(b.add(i) as *const v128);
        v128_store(out.add(i) as *mut v128, f32x4_add(va, vb));
        i += 4;
    }
    while i < len {
        *out.add(i) = *a.add(i) + *b.add(i);
        i += 1;
    }
}

#[no_mangle]
pub unsafe extern "C" fn mul_f32(a: *const f32, b: *const f32, out: *mut f32, n: u32) {
    let len = n as usize;
    let mut i = 0;
    while i + 8 <= len {
        let va0 = v128_load(a.add(i) as *const v128);
        let vb0 = v128_load(b.add(i) as *const v128);
        let va1 = v128_load(a.add(i + 4) as *const v128);
        let vb1 = v128_load(b.add(i + 4) as *const v128);
        v128_store(out.add(i) as *mut v128, f32x4_mul(va0, vb0));
        v128_store(out.add(i + 4) as *mut v128, f32x4_mul(va1, vb1));
        i += 8;
    }
    while i + 4 <= len {
        let va = v128_load(a.add(i) as *const v128);
        let vb = v128_load(b.add(i) as *const v128);
        v128_store(out.add(i) as *mut v128, f32x4_mul(va, vb));
        i += 4;
    }
    while i < len {
        *out.add(i) = *a.add(i) * *b.add(i);
        i += 1;
    }
}
