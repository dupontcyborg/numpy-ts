// Linear algebra kernels: matvec, vecmat, vecdot, outer, kron, cross, norm

use crate::simd::{load_f32x4, load_f64x2, store_f32x4, store_f64x2};
use core::arch::wasm32::*;

// ─── matvec: A[m×n] · x[n] → out[m] ────────────────────────────────────────

fn matvec_f64_inner(a: &[f64], x: &[f64], out: &mut [f64], rows: usize, cols: usize) {
    for i in 0..rows {
        let row_off = i * cols;
        let mut acc0 = f64x2_splat(0.0);
        let mut acc1 = f64x2_splat(0.0);
        let mut j = 0;
        while j + 4 <= cols {
            acc0 = f64x2_add(
                acc0,
                f64x2_mul(load_f64x2(a, row_off + j), load_f64x2(x, j)),
            );
            acc1 = f64x2_add(
                acc1,
                f64x2_mul(load_f64x2(a, row_off + j + 2), load_f64x2(x, j + 2)),
            );
            j += 4;
        }
        while j + 2 <= cols {
            acc0 = f64x2_add(
                acc0,
                f64x2_mul(load_f64x2(a, row_off + j), load_f64x2(x, j)),
            );
            j += 2;
        }
        acc0 = f64x2_add(acc0, acc1);
        let mut sum = f64x2_extract_lane::<0>(acc0) + f64x2_extract_lane::<1>(acc0);
        while j < cols {
            sum += a[row_off + j] * x[j];
            j += 1;
        }
        out[i] = sum;
    }
}

fn matvec_f32_inner(a: &[f32], x: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    for i in 0..rows {
        let row_off = i * cols;
        let mut acc0 = f32x4_splat(0.0);
        let mut acc1 = f32x4_splat(0.0);
        let mut j = 0;
        while j + 8 <= cols {
            acc0 = f32x4_add(
                acc0,
                f32x4_mul(load_f32x4(a, row_off + j), load_f32x4(x, j)),
            );
            acc1 = f32x4_add(
                acc1,
                f32x4_mul(load_f32x4(a, row_off + j + 4), load_f32x4(x, j + 4)),
            );
            j += 8;
        }
        while j + 4 <= cols {
            acc0 = f32x4_add(
                acc0,
                f32x4_mul(load_f32x4(a, row_off + j), load_f32x4(x, j)),
            );
            j += 4;
        }
        acc0 = f32x4_add(acc0, acc1);
        let mut sum = f32x4_extract_lane::<0>(acc0)
            + f32x4_extract_lane::<1>(acc0)
            + f32x4_extract_lane::<2>(acc0)
            + f32x4_extract_lane::<3>(acc0);
        while j < cols {
            sum += a[row_off + j] * x[j];
            j += 1;
        }
        out[i] = sum;
    }
}

#[no_mangle]
pub unsafe extern "C" fn matvec_f64(a: *const f64, x: *const f64, out: *mut f64, m: u32, n: u32) {
    let (rows, cols) = (m as usize, n as usize);
    matvec_f64_inner(
        core::slice::from_raw_parts(a, rows * cols),
        core::slice::from_raw_parts(x, cols),
        core::slice::from_raw_parts_mut(out, rows),
        rows,
        cols,
    );
}

#[no_mangle]
pub unsafe extern "C" fn matvec_f32(a: *const f32, x: *const f32, out: *mut f32, m: u32, n: u32) {
    let (rows, cols) = (m as usize, n as usize);
    matvec_f32_inner(
        core::slice::from_raw_parts(a, rows * cols),
        core::slice::from_raw_parts(x, cols),
        core::slice::from_raw_parts_mut(out, rows),
        rows,
        cols,
    );
}

// ─── vecmat: x[m] · A[m×n] → out[n] ────────────────────────────────────────

fn vecmat_f64_inner(x: &[f64], a: &[f64], out: &mut [f64], rows: usize, cols: usize) {
    // Zero output
    let mut j = 0;
    while j + 2 <= cols {
        store_f64x2(out, j, f64x2_splat(0.0));
        j += 2;
    }
    while j < cols {
        out[j] = 0.0;
        j += 1;
    }
    // Accumulate
    for i in 0..rows {
        let xi = f64x2_splat(x[i]);
        let row_off = i * cols;
        j = 0;
        while j + 4 <= cols {
            store_f64x2(
                out,
                j,
                f64x2_add(
                    load_f64x2(out, j),
                    f64x2_mul(xi, load_f64x2(a, row_off + j)),
                ),
            );
            store_f64x2(
                out,
                j + 2,
                f64x2_add(
                    load_f64x2(out, j + 2),
                    f64x2_mul(xi, load_f64x2(a, row_off + j + 2)),
                ),
            );
            j += 4;
        }
        while j + 2 <= cols {
            store_f64x2(
                out,
                j,
                f64x2_add(
                    load_f64x2(out, j),
                    f64x2_mul(xi, load_f64x2(a, row_off + j)),
                ),
            );
            j += 2;
        }
        while j < cols {
            out[j] += x[i] * a[row_off + j];
            j += 1;
        }
    }
}

fn vecmat_f32_inner(x: &[f32], a: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    let mut j = 0;
    while j + 4 <= cols {
        store_f32x4(out, j, f32x4_splat(0.0));
        j += 4;
    }
    while j < cols {
        out[j] = 0.0;
        j += 1;
    }
    for i in 0..rows {
        let xi = f32x4_splat(x[i]);
        let row_off = i * cols;
        j = 0;
        while j + 8 <= cols {
            store_f32x4(
                out,
                j,
                f32x4_add(
                    load_f32x4(out, j),
                    f32x4_mul(xi, load_f32x4(a, row_off + j)),
                ),
            );
            store_f32x4(
                out,
                j + 4,
                f32x4_add(
                    load_f32x4(out, j + 4),
                    f32x4_mul(xi, load_f32x4(a, row_off + j + 4)),
                ),
            );
            j += 8;
        }
        while j + 4 <= cols {
            store_f32x4(
                out,
                j,
                f32x4_add(
                    load_f32x4(out, j),
                    f32x4_mul(xi, load_f32x4(a, row_off + j)),
                ),
            );
            j += 4;
        }
        while j < cols {
            out[j] += x[i] * a[row_off + j];
            j += 1;
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn vecmat_f64(x: *const f64, a: *const f64, out: *mut f64, m: u32, n: u32) {
    let (rows, cols) = (m as usize, n as usize);
    vecmat_f64_inner(
        core::slice::from_raw_parts(x, rows),
        core::slice::from_raw_parts(a, rows * cols),
        core::slice::from_raw_parts_mut(out, cols),
        rows,
        cols,
    );
}

#[no_mangle]
pub unsafe extern "C" fn vecmat_f32(x: *const f32, a: *const f32, out: *mut f32, m: u32, n: u32) {
    let (rows, cols) = (m as usize, n as usize);
    vecmat_f32_inner(
        core::slice::from_raw_parts(x, rows),
        core::slice::from_raw_parts(a, rows * cols),
        core::slice::from_raw_parts_mut(out, cols),
        rows,
        cols,
    );
}

// ─── vecdot: batched dot products ───────────────────────────────────────────

fn vecdot_f64_inner(a: &[f64], b: &[f64], out: &mut [f64], batch: usize, len: usize) {
    for bi in 0..batch {
        let off = bi * len;
        let mut acc0 = f64x2_splat(0.0);
        let mut acc1 = f64x2_splat(0.0);
        let mut j = 0;
        while j + 4 <= len {
            acc0 = f64x2_add(
                acc0,
                f64x2_mul(load_f64x2(a, off + j), load_f64x2(b, off + j)),
            );
            acc1 = f64x2_add(
                acc1,
                f64x2_mul(load_f64x2(a, off + j + 2), load_f64x2(b, off + j + 2)),
            );
            j += 4;
        }
        while j + 2 <= len {
            acc0 = f64x2_add(
                acc0,
                f64x2_mul(load_f64x2(a, off + j), load_f64x2(b, off + j)),
            );
            j += 2;
        }
        acc0 = f64x2_add(acc0, acc1);
        let mut sum = f64x2_extract_lane::<0>(acc0) + f64x2_extract_lane::<1>(acc0);
        while j < len {
            sum += a[off + j] * b[off + j];
            j += 1;
        }
        out[bi] = sum;
    }
}

fn vecdot_f32_inner(a: &[f32], b: &[f32], out: &mut [f32], batch: usize, len: usize) {
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    for bi in 0..batch {
        let a_ptr = unsafe { ap.add(bi * len) };
        let b_ptr = unsafe { bp.add(bi * len) };
        let mut acc0 = f32x4_splat(0.0);
        let mut acc1 = f32x4_splat(0.0);
        let mut acc2 = f32x4_splat(0.0);
        let mut acc3 = f32x4_splat(0.0);
        let mut j = 0;
        while j + 16 <= len {
            unsafe {
                acc0 = f32x4_add(
                    acc0,
                    f32x4_mul(
                        v128_load(a_ptr.add(j) as *const v128),
                        v128_load(b_ptr.add(j) as *const v128),
                    ),
                );
                acc1 = f32x4_add(
                    acc1,
                    f32x4_mul(
                        v128_load(a_ptr.add(j + 4) as *const v128),
                        v128_load(b_ptr.add(j + 4) as *const v128),
                    ),
                );
                acc2 = f32x4_add(
                    acc2,
                    f32x4_mul(
                        v128_load(a_ptr.add(j + 8) as *const v128),
                        v128_load(b_ptr.add(j + 8) as *const v128),
                    ),
                );
                acc3 = f32x4_add(
                    acc3,
                    f32x4_mul(
                        v128_load(a_ptr.add(j + 12) as *const v128),
                        v128_load(b_ptr.add(j + 12) as *const v128),
                    ),
                );
            }
            j += 16;
        }
        acc0 = f32x4_add(acc0, acc2);
        acc1 = f32x4_add(acc1, acc3);
        while j + 8 <= len {
            unsafe {
                acc0 = f32x4_add(
                    acc0,
                    f32x4_mul(
                        v128_load(a_ptr.add(j) as *const v128),
                        v128_load(b_ptr.add(j) as *const v128),
                    ),
                );
                acc1 = f32x4_add(
                    acc1,
                    f32x4_mul(
                        v128_load(a_ptr.add(j + 4) as *const v128),
                        v128_load(b_ptr.add(j + 4) as *const v128),
                    ),
                );
            }
            j += 8;
        }
        while j + 4 <= len {
            unsafe {
                acc0 = f32x4_add(
                    acc0,
                    f32x4_mul(
                        v128_load(a_ptr.add(j) as *const v128),
                        v128_load(b_ptr.add(j) as *const v128),
                    ),
                );
            }
            j += 4;
        }
        acc0 = f32x4_add(acc0, acc1);
        let mut sum = f32x4_extract_lane::<0>(acc0)
            + f32x4_extract_lane::<1>(acc0)
            + f32x4_extract_lane::<2>(acc0)
            + f32x4_extract_lane::<3>(acc0);
        while j < len {
            sum += unsafe { *a_ptr.add(j) * *b_ptr.add(j) };
            j += 1;
        }
        out[bi] = sum;
    }
}

#[no_mangle]
pub unsafe extern "C" fn vecdot_f64(
    a: *const f64,
    b: *const f64,
    out: *mut f64,
    nbatch: u32,
    veclen: u32,
) {
    let (batch, len) = (nbatch as usize, veclen as usize);
    vecdot_f64_inner(
        core::slice::from_raw_parts(a, batch * len),
        core::slice::from_raw_parts(b, batch * len),
        core::slice::from_raw_parts_mut(out, batch),
        batch,
        len,
    );
}

#[no_mangle]
pub unsafe extern "C" fn vecdot_f32(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    nbatch: u32,
    veclen: u32,
) {
    let (batch, len) = (nbatch as usize, veclen as usize);
    vecdot_f32_inner(
        core::slice::from_raw_parts(a, batch * len),
        core::slice::from_raw_parts(b, batch * len),
        core::slice::from_raw_parts_mut(out, batch),
        batch,
        len,
    );
}

// ─── outer: a[m] ⊗ b[n] → out[m×n] ────────────────────────────────────────

fn outer_f64_inner(a: &[f64], b: &[f64], out: &mut [f64], rows: usize, cols: usize) {
    for i in 0..rows {
        let ai = f64x2_splat(a[i]);
        let row_off = i * cols;
        let mut j = 0;
        while j + 4 <= cols {
            store_f64x2(out, row_off + j, f64x2_mul(ai, load_f64x2(b, j)));
            store_f64x2(out, row_off + j + 2, f64x2_mul(ai, load_f64x2(b, j + 2)));
            j += 4;
        }
        while j + 2 <= cols {
            store_f64x2(out, row_off + j, f64x2_mul(ai, load_f64x2(b, j)));
            j += 2;
        }
        while j < cols {
            out[row_off + j] = a[i] * b[j];
            j += 1;
        }
    }
}

fn outer_f32_inner(a: &[f32], b: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    for i in 0..rows {
        let ai = f32x4_splat(a[i]);
        let row_off = i * cols;
        let mut j = 0;
        while j + 8 <= cols {
            store_f32x4(out, row_off + j, f32x4_mul(ai, load_f32x4(b, j)));
            store_f32x4(out, row_off + j + 4, f32x4_mul(ai, load_f32x4(b, j + 4)));
            j += 8;
        }
        while j + 4 <= cols {
            store_f32x4(out, row_off + j, f32x4_mul(ai, load_f32x4(b, j)));
            j += 4;
        }
        while j < cols {
            out[row_off + j] = a[i] * b[j];
            j += 1;
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn outer_f64(a: *const f64, b: *const f64, out: *mut f64, m: u32, n: u32) {
    let (rows, cols) = (m as usize, n as usize);
    outer_f64_inner(
        core::slice::from_raw_parts(a, rows),
        core::slice::from_raw_parts(b, cols),
        core::slice::from_raw_parts_mut(out, rows * cols),
        rows,
        cols,
    );
}

#[no_mangle]
pub unsafe extern "C" fn outer_f32(a: *const f32, b: *const f32, out: *mut f32, m: u32, n: u32) {
    let (rows, cols) = (m as usize, n as usize);
    outer_f32_inner(
        core::slice::from_raw_parts(a, rows),
        core::slice::from_raw_parts(b, cols),
        core::slice::from_raw_parts_mut(out, rows * cols),
        rows,
        cols,
    );
}

// ─── kron: Kronecker product ────────────────────────────────────────────────

fn kron_f64_inner(
    a: &[f64],
    b: &[f64],
    out: &mut [f64],
    ar: usize,
    ac: usize,
    br: usize,
    bc: usize,
) {
    let out_cols = ac * bc;
    for ia in 0..ar {
        for ja in 0..ac {
            let aij = f64x2_splat(a[ia * ac + ja]);
            for ib in 0..br {
                let out_off = (ia * br + ib) * out_cols + ja * bc;
                let b_off = ib * bc;
                let mut jb = 0;
                while jb + 2 <= bc {
                    store_f64x2(out, out_off + jb, f64x2_mul(aij, load_f64x2(b, b_off + jb)));
                    jb += 2;
                }
                while jb < bc {
                    out[out_off + jb] = a[ia * ac + ja] * b[b_off + jb];
                    jb += 1;
                }
            }
        }
    }
}

fn kron_f32_inner(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    ar: usize,
    ac: usize,
    br: usize,
    bc: usize,
) {
    let out_cols = ac * bc;
    for ia in 0..ar {
        for ja in 0..ac {
            let aij = f32x4_splat(a[ia * ac + ja]);
            for ib in 0..br {
                let out_off = (ia * br + ib) * out_cols + ja * bc;
                let b_off = ib * bc;
                let mut jb = 0;
                while jb + 4 <= bc {
                    store_f32x4(out, out_off + jb, f32x4_mul(aij, load_f32x4(b, b_off + jb)));
                    jb += 4;
                }
                while jb < bc {
                    out[out_off + jb] = a[ia * ac + ja] * b[b_off + jb];
                    jb += 1;
                }
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn kron_f64(
    a: *const f64,
    b: *const f64,
    out: *mut f64,
    am: u32,
    an: u32,
    bm: u32,
    bn: u32,
) {
    let (ar, ac, br, bc) = (am as usize, an as usize, bm as usize, bn as usize);
    kron_f64_inner(
        core::slice::from_raw_parts(a, ar * ac),
        core::slice::from_raw_parts(b, br * bc),
        core::slice::from_raw_parts_mut(out, ar * br * ac * bc),
        ar,
        ac,
        br,
        bc,
    );
}

#[no_mangle]
pub unsafe extern "C" fn kron_f32(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    am: u32,
    an: u32,
    bm: u32,
    bn: u32,
) {
    let (ar, ac, br, bc) = (am as usize, an as usize, bm as usize, bn as usize);
    kron_f32_inner(
        core::slice::from_raw_parts(a, ar * ac),
        core::slice::from_raw_parts(b, br * bc),
        core::slice::from_raw_parts_mut(out, ar * br * ac * bc),
        ar,
        ac,
        br,
        bc,
    );
}

// ─── cross: cross product of n pairs of 3-vectors ──────────────────────────

fn cross_f64_inner(a: &[f64], b: &[f64], out: &mut [f64], n: usize) {
    for i in 0..n {
        let (ao, bo, oo) = (i * 3, i * 3, i * 3);
        out[oo] = a[ao + 1] * b[bo + 2] - a[ao + 2] * b[bo + 1];
        out[oo + 1] = a[ao + 2] * b[bo] - a[ao] * b[bo + 2];
        out[oo + 2] = a[ao] * b[bo + 1] - a[ao + 1] * b[bo];
    }
}

fn cross_f32_inner(a: &[f32], b: &[f32], out: &mut [f32], n: usize) {
    for i in 0..n {
        let (ao, bo, oo) = (i * 3, i * 3, i * 3);
        out[oo] = a[ao + 1] * b[bo + 2] - a[ao + 2] * b[bo + 1];
        out[oo + 1] = a[ao + 2] * b[bo] - a[ao] * b[bo + 2];
        out[oo + 2] = a[ao] * b[bo + 1] - a[ao + 1] * b[bo];
    }
}

#[no_mangle]
pub unsafe extern "C" fn cross_f64(a: *const f64, b: *const f64, out: *mut f64, n: u32) {
    let len = n as usize;
    cross_f64_inner(
        core::slice::from_raw_parts(a, len * 3),
        core::slice::from_raw_parts(b, len * 3),
        core::slice::from_raw_parts_mut(out, len * 3),
        len,
    );
}

#[no_mangle]
pub unsafe extern "C" fn cross_f32(a: *const f32, b: *const f32, out: *mut f32, n: u32) {
    let len = n as usize;
    cross_f32_inner(
        core::slice::from_raw_parts(a, len * 3),
        core::slice::from_raw_parts(b, len * 3),
        core::slice::from_raw_parts_mut(out, len * 3),
        len,
    );
}

// ─── norm: L2 norm ──────────────────────────────────────────────────────────

fn norm_f64_inner(data: &[f64]) -> f64 {
    let len = data.len();
    let mut acc0 = f64x2_splat(0.0);
    let mut acc1 = f64x2_splat(0.0);
    let mut acc2 = f64x2_splat(0.0);
    let mut acc3 = f64x2_splat(0.0);
    let mut i = 0;
    while i + 8 <= len {
        let v0 = load_f64x2(data, i);
        let v1 = load_f64x2(data, i + 2);
        let v2 = load_f64x2(data, i + 4);
        let v3 = load_f64x2(data, i + 6);
        acc0 = f64x2_add(acc0, f64x2_mul(v0, v0));
        acc1 = f64x2_add(acc1, f64x2_mul(v1, v1));
        acc2 = f64x2_add(acc2, f64x2_mul(v2, v2));
        acc3 = f64x2_add(acc3, f64x2_mul(v3, v3));
        i += 8;
    }
    acc0 = f64x2_add(acc0, acc2);
    acc1 = f64x2_add(acc1, acc3);
    while i + 4 <= len {
        let v0 = load_f64x2(data, i);
        let v1 = load_f64x2(data, i + 2);
        acc0 = f64x2_add(acc0, f64x2_mul(v0, v0));
        acc1 = f64x2_add(acc1, f64x2_mul(v1, v1));
        i += 4;
    }
    while i + 2 <= len {
        let v = load_f64x2(data, i);
        acc0 = f64x2_add(acc0, f64x2_mul(v, v));
        i += 2;
    }
    acc0 = f64x2_add(acc0, acc1);
    let mut sum = f64x2_extract_lane::<0>(acc0) + f64x2_extract_lane::<1>(acc0);
    while i < len {
        let v = data[i];
        sum += v * v;
        i += 1;
    }
    f64x2_extract_lane::<0>(f64x2_sqrt(f64x2_splat(sum)))
}

fn norm_f32_inner(data: &[f32]) -> f32 {
    let len = data.len();
    let mut acc0 = f32x4_splat(0.0);
    let mut acc1 = f32x4_splat(0.0);
    let mut i = 0;
    while i + 8 <= len {
        let v0 = load_f32x4(data, i);
        let v1 = load_f32x4(data, i + 4);
        acc0 = f32x4_add(acc0, f32x4_mul(v0, v0));
        acc1 = f32x4_add(acc1, f32x4_mul(v1, v1));
        i += 8;
    }
    while i + 4 <= len {
        let v = load_f32x4(data, i);
        acc0 = f32x4_add(acc0, f32x4_mul(v, v));
        i += 4;
    }
    acc0 = f32x4_add(acc0, acc1);
    let mut sum = f32x4_extract_lane::<0>(acc0)
        + f32x4_extract_lane::<1>(acc0)
        + f32x4_extract_lane::<2>(acc0)
        + f32x4_extract_lane::<3>(acc0);
    while i < len {
        let v = data[i];
        sum += v * v;
        i += 1;
    }
    f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(sum)))
}

#[no_mangle]
pub unsafe extern "C" fn norm_f64(ptr: *const f64, n: u32) -> f64 {
    norm_f64_inner(core::slice::from_raw_parts(ptr, n as usize))
}

#[no_mangle]
pub unsafe extern "C" fn norm_f32(ptr: *const f32, n: u32) -> f32 {
    norm_f32_inner(core::slice::from_raw_parts(ptr, n as usize))
}

// ─── Internal tiled matmul (safe, slice-based) ──────────────────────────────

const TILE_INT: usize = 48;

fn matmul_internal_f64(a: &[f64], b: &[f64], c: &mut [f64], m: usize, n: usize, k: usize) {
    for v in c.iter_mut() {
        *v = 0.0;
    }
    let mut ii = 0;
    while ii < m {
        let ie = if ii + TILE_INT < m { ii + TILE_INT } else { m };
        let mut kk = 0;
        while kk < k {
            let ke = if kk + TILE_INT < k { kk + TILE_INT } else { k };
            let mut jj = 0;
            while jj < n {
                let je = if jj + TILE_INT < n { jj + TILE_INT } else { n };
                let mut ri = ii;
                while ri < ie {
                    let mut rk = kk;
                    while rk < ke {
                        let aik = a[ri * k + rk];
                        let mut j = jj;
                        while j < je {
                            c[ri * n + j] += aik * b[rk * n + j];
                            j += 1;
                        }
                        rk += 1;
                    }
                    ri += 1;
                }
                jj += TILE_INT;
            }
            kk += TILE_INT;
        }
        ii += TILE_INT;
    }
}

fn sqrt_f64_scalar(x: f64) -> f64 {
    f64x2_extract_lane::<0>(f64x2_sqrt(f64x2_splat(x)))
}

// ─── matrix_power: out = a^power via binary exponentiation ──────────────────

#[no_mangle]
pub unsafe extern "C" fn matrix_power_f64(
    a: *const f64,
    out: *mut f64,
    scratch: *mut f64,
    n: u32,
    power: u32,
) {
    let nn = n as usize;
    let sz = nn * nn;
    let a_slice = core::slice::from_raw_parts(a, sz);
    let out_slice = core::slice::from_raw_parts_mut(out, sz);
    let cur = core::slice::from_raw_parts_mut(scratch, sz);
    let tmp = core::slice::from_raw_parts_mut(scratch.add(sz), sz);
    let mut p = power as usize;

    for v in out_slice.iter_mut() {
        *v = 0.0;
    }
    for i in 0..nn {
        out_slice[i * nn + i] = 1.0;
    }
    cur.copy_from_slice(a_slice);

    while p > 0 {
        if p & 1 != 0 {
            matmul_internal_f64(out_slice, cur, tmp, nn, nn, nn);
            out_slice.copy_from_slice(tmp);
        }
        p >>= 1;
        if p > 0 {
            matmul_internal_f64(cur, cur, tmp, nn, nn, nn);
            cur.copy_from_slice(tmp);
        }
    }
}

// ─── multi_dot3: out = a @ b @ c ────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn multi_dot3_f64(
    a: *const f64,
    b: *const f64,
    c: *const f64,
    out: *mut f64,
    tmp: *mut f64,
    n: u32,
) {
    let nn = n as usize;
    let sz = nn * nn;
    let sa = core::slice::from_raw_parts(a, sz);
    let sb = core::slice::from_raw_parts(b, sz);
    let sc = core::slice::from_raw_parts(c, sz);
    let so = core::slice::from_raw_parts_mut(out, sz);
    let st = core::slice::from_raw_parts_mut(tmp, sz);
    matmul_internal_f64(sa, sb, st, nn, nn, nn);
    matmul_internal_f64(st, sc, so, nn, nn, nn);
}

// ─── qr: Householder QR decomposition ──────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn qr_f64(
    a_ptr: *mut f64,
    q_ptr: *mut f64,
    r_ptr: *mut f64,
    tau_ptr: *mut f64,
    _scratch: *mut f64,
    m: u32,
    n: u32,
) {
    let rows = m as usize;
    let cols = n as usize;
    let k = if rows < cols { rows } else { cols };
    let a = core::slice::from_raw_parts_mut(a_ptr, rows * cols);
    let q = core::slice::from_raw_parts_mut(q_ptr, rows * k);
    let r = core::slice::from_raw_parts_mut(r_ptr, k * cols);
    let tau = core::slice::from_raw_parts_mut(tau_ptr, k);

    for j in 0..k {
        let mut norm_sq = 0.0f64;
        for i in j..rows {
            let v = a[i * cols + j];
            norm_sq += v * v;
        }
        let mut nrm = sqrt_f64_scalar(norm_sq);
        if nrm == 0.0 {
            tau[j] = 0.0;
            continue;
        }

        let ajj = a[j * cols + j];
        if ajj >= 0.0 {
            nrm = -nrm;
        }
        let alpha = nrm;

        a[j * cols + j] -= alpha;
        let v0 = a[j * cols + j];

        let mut vtv = v0 * v0;
        for i in (j + 1)..rows {
            let vi = a[i * cols + j];
            vtv += vi * vi;
        }
        if vtv == 0.0 {
            tau[j] = 0.0;
            a[j * cols + j] = alpha;
            continue;
        }
        tau[j] = 2.0 / vtv;

        for col in (j + 1)..cols {
            let mut dot = 0.0f64;
            for i in j..rows {
                dot += a[i * cols + j] * a[i * cols + col];
            }
            let factor = tau[j] * dot;
            for i in j..rows {
                a[i * cols + col] -= factor * a[i * cols + j];
            }
        }

        a[j * cols + j] = alpha;
    }

    // Extract R
    for i in 0..k {
        for j in 0..cols {
            r[i * cols + j] = if j >= i { a[i * cols + j] } else { 0.0 };
        }
    }

    // Reconstruct Q
    for v in q.iter_mut() {
        *v = 0.0;
    }
    for i in 0..k {
        q[i * k + i] = 1.0;
    }

    let mut jrev = k;
    while jrev > 0 {
        jrev -= 1;
        let j = jrev;
        if tau[j] == 0.0 {
            continue;
        }

        let mut sub_sq = 0.0f64;
        for i in (j + 1)..rows {
            let vi = a[i * cols + j];
            sub_sq += vi * vi;
        }
        let vtv2 = 2.0 / tau[j];
        let v0sq = vtv2 - sub_sq;
        let v0 = if v0sq > 0.0 {
            sqrt_f64_scalar(v0sq)
        } else {
            0.0
        };

        for col in 0..k {
            let mut dot = v0 * q[j * k + col];
            for i in (j + 1)..rows {
                dot += a[i * cols + j] * q[i * k + col];
            }
            let factor = tau[j] * dot;
            q[j * k + col] -= factor * v0;
            for i in (j + 1)..rows {
                q[i * k + col] -= factor * a[i * cols + j];
            }
        }
    }
}

// ─── lstsq: solve Ax=b via QR ──────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn lstsq_f64(
    a_ptr: *mut f64,
    b_ptr: *const f64,
    x_ptr: *mut f64,
    scratch: *mut f64,
    m: u32,
    n: u32,
) {
    let rows = m as usize;
    let cols = n as usize;
    let k = if rows < cols { rows } else { cols };

    let b = core::slice::from_raw_parts(b_ptr, rows);
    let x = core::slice::from_raw_parts_mut(x_ptr, cols);

    let a_copy = scratch;
    let q_off = a_copy.add(rows * cols);
    let r_off = q_off.add(rows * k);
    let tau_off = r_off.add(k * cols);
    let qtb_off = tau_off.add(k);
    let qr_scratch = qtb_off.add(k);

    let ac = core::slice::from_raw_parts_mut(a_copy, rows * cols);
    let a_src = core::slice::from_raw_parts(a_ptr as *const f64, rows * cols);
    ac.copy_from_slice(a_src);

    qr_f64(a_copy, q_off, r_off, tau_off, qr_scratch, m, n);

    let q = core::slice::from_raw_parts(q_off as *const f64, rows * k);
    let r = core::slice::from_raw_parts(r_off as *const f64, k * cols);
    let qtb = core::slice::from_raw_parts_mut(qtb_off, k);

    for i in 0..k {
        let mut sum = 0.0f64;
        for j in 0..rows {
            sum += q[j * k + i] * b[j];
        }
        qtb[i] = sum;
    }

    let mut ii = k;
    while ii > 0 {
        ii -= 1;
        let mut sum = qtb[ii];
        for j in (ii + 1)..cols {
            sum -= r[ii * cols + j] * x[j];
        }
        let diag = r[ii * cols + ii];
        x[ii] = if diag != 0.0 { sum / diag } else { 0.0 };
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPLEX MATMUL (c128, c64)
// ═══════════════════════════════════════════════════════════════════════════

fn matmul_complex_f64_inner(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    for v in c.iter_mut() {
        *v = 0.0;
    }
    const T: usize = 32;
    let mut ii = 0;
    while ii < m {
        let ie = if ii + T < m { ii + T } else { m };
        let mut kk = 0;
        while kk < k {
            let ke = if kk + T < k { kk + T } else { k };
            let mut jj = 0;
            while jj < n {
                let je = if jj + T < n { jj + T } else { n };
                let mut ri = ii;
                while ri < ie {
                    let mut rk = kk;
                    while rk < ke {
                        let a_re = a[(ri * k + rk) * 2];
                        let a_im = a[(ri * k + rk) * 2 + 1];
                        let mut j = jj;
                        while j < je {
                            let b_re = b[(rk * n + j) * 2];
                            let b_im = b[(rk * n + j) * 2 + 1];
                            let ci = (ri * n + j) * 2;
                            c[ci] += a_re * b_re - a_im * b_im;
                            c[ci + 1] += a_re * b_im + a_im * b_re;
                            j += 1;
                        }
                        rk += 1;
                    }
                    ri += 1;
                }
                jj += T;
            }
            kk += T;
        }
        ii += T;
    }
}

fn matmul_complex_f32_inner(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for v in c.iter_mut() {
        *v = 0.0;
    }
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let cp = c.as_mut_ptr();
    const T: usize = 32;
    let mut ii = 0;
    while ii < m {
        let ie = if ii + T < m { ii + T } else { m };
        let mut kk = 0;
        while kk < k {
            let ke = if kk + T < k { kk + T } else { k };
            let mut jj = 0;
            while jj < n {
                let je = if jj + T < n { jj + T } else { n };
                let mut ri = ii;
                while ri < ie {
                    let c_row = unsafe { cp.add(ri * n * 2) };
                    let mut rk = kk;
                    while rk < ke {
                        let a_base = unsafe { ap.add((ri * k + rk) * 2) };
                        let a_re = unsafe { *a_base };
                        let a_im = unsafe { *a_base.add(1) };
                        let b_row = unsafe { bp.add(rk * n * 2) };
                        let mut j = jj;
                        while j < je {
                            let b_ptr = unsafe { b_row.add(j * 2) };
                            let b_re = unsafe { *b_ptr };
                            let b_im = unsafe { *b_ptr.add(1) };
                            let c_ptr = unsafe { c_row.add(j * 2) };
                            unsafe {
                                *c_ptr += a_re * b_re - a_im * b_im;
                                *c_ptr.add(1) += a_re * b_im + a_im * b_re;
                            }
                            j += 1;
                        }
                        rk += 1;
                    }
                    ri += 1;
                }
                jj += T;
            }
            kk += T;
        }
        ii += T;
    }
}

#[no_mangle]
pub unsafe extern "C" fn matmul_c128(
    a: *const f64,
    b: *const f64,
    c: *mut f64,
    m: u32,
    k: u32,
    n: u32,
) {
    let (m, k, n) = (m as usize, k as usize, n as usize);
    matmul_complex_f64_inner(
        core::slice::from_raw_parts(a, m * k * 2),
        core::slice::from_raw_parts(b, k * n * 2),
        core::slice::from_raw_parts_mut(c, m * n * 2),
        m,
        k,
        n,
    );
}

#[no_mangle]
pub unsafe extern "C" fn matmul_c64(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: u32,
    k: u32,
    n: u32,
) {
    let (m, k, n) = (m as usize, k as usize, n as usize);
    matmul_complex_f32_inner(
        core::slice::from_raw_parts(a, m * k * 2),
        core::slice::from_raw_parts(b, k * n * 2),
        core::slice::from_raw_parts_mut(c, m * n * 2),
        m,
        k,
        n,
    );
}
