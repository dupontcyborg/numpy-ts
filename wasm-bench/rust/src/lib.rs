#![no_std]

#[allow(dead_code)]
mod simd;

#[cfg(feature = "kern-arrayops")]
mod arrayops;
#[cfg(feature = "kern-binary")]
mod binary;
#[cfg(feature = "kern-convolve")]
mod convolve;
#[cfg(feature = "kern-fft")]
mod fft;
#[cfg(feature = "kern-linalg")]
mod linalg;
#[cfg(feature = "kern-reduction")]
mod reduction;
#[cfg(feature = "kern-sort")]
mod sort;
#[cfg(feature = "kern-unary")]
mod unary;

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

#[cfg(feature = "kern-matmul")]
const TILE_F64: usize = 48;
#[cfg(feature = "kern-matmul")]
const TILE_F32: usize = 64;

#[cfg(feature = "kern-matmul")]
fn matmul_f64_inner(a: &[f64], b: &[f64], c: &mut [f64], m: usize, n: usize, k: usize) {
    use crate::simd::{load_f64x2, store_f64x2};
    use core::arch::wasm32::*;

    for v in c.iter_mut() {
        *v = 0.0;
    }

    let mut ii = 0;
    while ii < m {
        let i_end = if ii + TILE_F64 < m { ii + TILE_F64 } else { m };
        let mut kk = 0;
        while kk < k {
            let k_end = if kk + TILE_F64 < k { kk + TILE_F64 } else { k };
            let mut jj = 0;
            while jj < n {
                let j_end = if jj + TILE_F64 < n { jj + TILE_F64 } else { n };

                let mut i = ii;
                while i < i_end {
                    let mut ki = kk;
                    while ki < k_end {
                        let a_ik = f64x2_splat(a[i * k + ki]);
                        let c_off = i * n;
                        let b_off = ki * n;
                        let mut j = jj;
                        // SIMD inner loop: 2 f64s at a time
                        while j + 2 <= j_end {
                            let cv = load_f64x2(c, c_off + j);
                            let bv = load_f64x2(b, b_off + j);
                            store_f64x2(c, c_off + j, f64x2_add(cv, f64x2_mul(a_ik, bv)));
                            j += 2;
                        }
                        // Scalar remainder
                        let a_ik_s = a[i * k + ki];
                        while j < j_end {
                            c[i * n + j] += a_ik_s * b[ki * n + j];
                            j += 1;
                        }
                        ki += 1;
                    }
                    i += 1;
                }
                jj += TILE_F64;
            }
            kk += TILE_F64;
        }
        ii += TILE_F64;
    }
}

#[cfg(feature = "kern-matmul")]
#[no_mangle]
pub unsafe extern "C" fn matmul_f64(
    a_ptr: *const f64,
    b_ptr: *const f64,
    c_ptr: *mut f64,
    m: u32,
    n: u32,
    k: u32,
) {
    let (m, n, k) = (m as usize, n as usize, k as usize);
    let a = core::slice::from_raw_parts(a_ptr, m * k);
    let b = core::slice::from_raw_parts(b_ptr, k * n);
    let c = core::slice::from_raw_parts_mut(c_ptr, m * n);
    matmul_f64_inner(a, b, c, m, n, k);
}

#[cfg(feature = "kern-matmul")]
fn matmul_f32_inner(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    use crate::simd::{load_f32x4, store_f32x4};
    use core::arch::wasm32::*;

    for v in c.iter_mut() {
        *v = 0.0;
    }

    let mut ii = 0;
    while ii < m {
        let i_end = if ii + TILE_F32 < m { ii + TILE_F32 } else { m };
        let mut kk = 0;
        while kk < k {
            let k_end = if kk + TILE_F32 < k { kk + TILE_F32 } else { k };
            let mut jj = 0;
            while jj < n {
                let j_end = if jj + TILE_F32 < n { jj + TILE_F32 } else { n };

                let mut i = ii;
                while i < i_end {
                    let mut ki = kk;
                    while ki < k_end {
                        let a_ik = f32x4_splat(a[i * k + ki]);
                        let c_off = i * n;
                        let b_off = ki * n;
                        let mut j = jj;
                        // SIMD inner loop: 4 f32s at a time
                        while j + 4 <= j_end {
                            let cv = load_f32x4(c, c_off + j);
                            let bv = load_f32x4(b, b_off + j);
                            store_f32x4(c, c_off + j, f32x4_add(cv, f32x4_mul(a_ik, bv)));
                            j += 4;
                        }
                        // Scalar remainder
                        let a_ik_s = a[i * k + ki];
                        while j < j_end {
                            c[i * n + j] += a_ik_s * b[ki * n + j];
                            j += 1;
                        }
                        ki += 1;
                    }
                    i += 1;
                }
                jj += TILE_F32;
            }
            kk += TILE_F32;
        }
        ii += TILE_F32;
    }
}

#[cfg(feature = "kern-matmul")]
#[no_mangle]
pub unsafe extern "C" fn matmul_f32(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    m: u32,
    n: u32,
    k: u32,
) {
    let (m, n, k) = (m as usize, n as usize, k as usize);
    let a = core::slice::from_raw_parts(a_ptr, m * k);
    let b = core::slice::from_raw_parts(b_ptr, k * n);
    let c = core::slice::from_raw_parts_mut(c_ptr, m * n);
    matmul_f32_inner(a, b, c, m, n, k);
}
