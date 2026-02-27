#![no_std]

#[cfg(feature = "kern-reduction")]
mod reduction;
#[cfg(feature = "kern-unary")]
mod unary;
#[cfg(feature = "kern-binary")]
mod binary;
#[cfg(feature = "kern-sort")]
mod sort;

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
#[no_mangle]
pub unsafe extern "C" fn matmul_f64(
    a_ptr: *const f64,
    b_ptr: *const f64,
    c_ptr: *mut f64,
    m: u32,
    n: u32,
    k: u32,
) {
    let m = m as usize;
    let n = n as usize;
    let k = k as usize;

    // Zero output
    for i in 0..m * n {
        *c_ptr.add(i) = 0.0;
    }

    // Tiled i-k-j loop
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
                        let a_ik = *a_ptr.add(i * k + ki);
                        let mut j = jj;
                        while j < j_end {
                            let c_idx = i * n + j;
                            let b_val = *b_ptr.add(ki * n + j);
                            *c_ptr.add(c_idx) += a_ik * b_val;
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
pub unsafe extern "C" fn matmul_f32(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    m: u32,
    n: u32,
    k: u32,
) {
    let m = m as usize;
    let n = n as usize;
    let k = k as usize;

    // Zero output
    for i in 0..m * n {
        *c_ptr.add(i) = 0.0;
    }

    // Tiled i-k-j loop
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
                        let a_ik = *a_ptr.add(i * k + ki);
                        let mut j = jj;
                        while j < j_end {
                            let c_idx = i * n + j;
                            let b_val = *b_ptr.add(ki * n + j);
                            *c_ptr.add(c_idx) += a_ik * b_val;
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
