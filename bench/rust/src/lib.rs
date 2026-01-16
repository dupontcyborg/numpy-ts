//! WASM Benchmark Operations for numpy-ts
//!
//! Single-threaded and multi-threaded implementations of:
//! - Element-wise operations (add, sin)
//! - Reduction operations (sum)
//! - Matrix operations (matmul)

use std::alloc::{alloc, dealloc, Layout};

#[cfg(feature = "threads")]
use rayon::prelude::*;

// ============================================================================
// Memory Management
// ============================================================================

/// Allocate memory for f64 array
#[no_mangle]
pub extern "C" fn alloc_f64(len: usize) -> *mut f64 {
    let layout = Layout::array::<f64>(len).unwrap();
    unsafe { alloc(layout) as *mut f64 }
}

/// Allocate memory for f32 array
#[no_mangle]
pub extern "C" fn alloc_f32(len: usize) -> *mut f32 {
    let layout = Layout::array::<f32>(len).unwrap();
    unsafe { alloc(layout) as *mut f32 }
}

/// Free f64 array
#[no_mangle]
pub extern "C" fn free_f64(ptr: *mut f64, len: usize) {
    if !ptr.is_null() {
        let layout = Layout::array::<f64>(len).unwrap();
        unsafe { dealloc(ptr as *mut u8, layout) };
    }
}

/// Free f32 array
#[no_mangle]
pub extern "C" fn free_f32(ptr: *mut f32, len: usize) {
    if !ptr.is_null() {
        let layout = Layout::array::<f32>(len).unwrap();
        unsafe { dealloc(ptr as *mut u8, layout) };
    }
}

// ============================================================================
// Element-wise Operations - Single Threaded
// ============================================================================

/// Element-wise addition: c = a + b (f64)
#[no_mangle]
pub extern "C" fn add_f64(a: *const f64, b: *const f64, c: *mut f64, len: usize) {
    unsafe {
        let a = std::slice::from_raw_parts(a, len);
        let b = std::slice::from_raw_parts(b, len);
        let c = std::slice::from_raw_parts_mut(c, len);

        for i in 0..len {
            c[i] = a[i] + b[i];
        }
    }
}

/// Element-wise addition: c = a + b (f32)
#[no_mangle]
pub extern "C" fn add_f32(a: *const f32, b: *const f32, c: *mut f32, len: usize) {
    unsafe {
        let a = std::slice::from_raw_parts(a, len);
        let b = std::slice::from_raw_parts(b, len);
        let c = std::slice::from_raw_parts_mut(c, len);

        for i in 0..len {
            c[i] = a[i] + b[i];
        }
    }
}

/// Element-wise sine: c = sin(a) (f64)
#[no_mangle]
pub extern "C" fn sin_f64(a: *const f64, c: *mut f64, len: usize) {
    unsafe {
        let a = std::slice::from_raw_parts(a, len);
        let c = std::slice::from_raw_parts_mut(c, len);

        for i in 0..len {
            c[i] = a[i].sin();
        }
    }
}

/// Element-wise sine: c = sin(a) (f32)
#[no_mangle]
pub extern "C" fn sin_f32(a: *const f32, c: *mut f32, len: usize) {
    unsafe {
        let a = std::slice::from_raw_parts(a, len);
        let c = std::slice::from_raw_parts_mut(c, len);

        for i in 0..len {
            c[i] = a[i].sin();
        }
    }
}

// ============================================================================
// Reduction Operations - Single Threaded
// ============================================================================

/// Sum reduction (f64)
#[no_mangle]
pub extern "C" fn sum_f64(a: *const f64, len: usize) -> f64 {
    unsafe {
        let a = std::slice::from_raw_parts(a, len);
        a.iter().sum()
    }
}

/// Sum reduction (f32)
#[no_mangle]
pub extern "C" fn sum_f32(a: *const f32, len: usize) -> f32 {
    unsafe {
        let a = std::slice::from_raw_parts(a, len);
        a.iter().sum()
    }
}

// ============================================================================
// Matrix Operations - Single Threaded
// ============================================================================

/// Matrix multiplication: C = A * B (f64)
/// A is MxK, B is KxN, C is MxN (row-major)
#[no_mangle]
pub extern "C" fn matmul_f64(
    a: *const f64,
    b: *const f64,
    c: *mut f64,
    m: usize,
    n: usize,
    k: usize,
) {
    unsafe {
        let a = std::slice::from_raw_parts(a, m * k);
        let b = std::slice::from_raw_parts(b, k * n);
        let c = std::slice::from_raw_parts_mut(c, m * n);

        // Zero output
        c.fill(0.0);

        // Cache-friendly ikj order
        for i in 0..m {
            for kk in 0..k {
                let aik = a[i * k + kk];
                for j in 0..n {
                    c[i * n + j] += aik * b[kk * n + j];
                }
            }
        }
    }
}

/// Matrix multiplication: C = A * B (f32)
#[no_mangle]
pub extern "C" fn matmul_f32(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
) {
    unsafe {
        let a = std::slice::from_raw_parts(a, m * k);
        let b = std::slice::from_raw_parts(b, k * n);
        let c = std::slice::from_raw_parts_mut(c, m * n);

        // Zero output
        c.fill(0.0);

        // Cache-friendly ikj order
        for i in 0..m {
            for kk in 0..k {
                let aik = a[i * k + kk];
                for j in 0..n {
                    c[i * n + j] += aik * b[kk * n + j];
                }
            }
        }
    }
}

// ============================================================================
// Multi-threaded Operations (with rayon feature)
// ============================================================================

#[cfg(feature = "threads")]
mod threaded {
    use super::*;
    use rayon::prelude::*;

    /// Element-wise addition: c = a + b (f64) - multi-threaded
    #[no_mangle]
    pub extern "C" fn add_f64_mt(a: *const f64, b: *const f64, c: *mut f64, len: usize) {
        unsafe {
            let a = std::slice::from_raw_parts(a, len);
            let b = std::slice::from_raw_parts(b, len);
            let c = std::slice::from_raw_parts_mut(c, len);

            c.par_iter_mut()
                .enumerate()
                .for_each(|(i, ci)| {
                    *ci = a[i] + b[i];
                });
        }
    }

    /// Element-wise addition: c = a + b (f32) - multi-threaded
    #[no_mangle]
    pub extern "C" fn add_f32_mt(a: *const f32, b: *const f32, c: *mut f32, len: usize) {
        unsafe {
            let a = std::slice::from_raw_parts(a, len);
            let b = std::slice::from_raw_parts(b, len);
            let c = std::slice::from_raw_parts_mut(c, len);

            c.par_iter_mut()
                .enumerate()
                .for_each(|(i, ci)| {
                    *ci = a[i] + b[i];
                });
        }
    }

    /// Element-wise sine: c = sin(a) (f64) - multi-threaded
    #[no_mangle]
    pub extern "C" fn sin_f64_mt(a: *const f64, c: *mut f64, len: usize) {
        unsafe {
            let a = std::slice::from_raw_parts(a, len);
            let c = std::slice::from_raw_parts_mut(c, len);

            c.par_iter_mut()
                .enumerate()
                .for_each(|(i, ci)| {
                    *ci = a[i].sin();
                });
        }
    }

    /// Element-wise sine: c = sin(a) (f32) - multi-threaded
    #[no_mangle]
    pub extern "C" fn sin_f32_mt(a: *const f32, c: *mut f32, len: usize) {
        unsafe {
            let a = std::slice::from_raw_parts(a, len);
            let c = std::slice::from_raw_parts_mut(c, len);

            c.par_iter_mut()
                .enumerate()
                .for_each(|(i, ci)| {
                    *ci = a[i].sin();
                });
        }
    }

    /// Sum reduction (f64) - multi-threaded
    #[no_mangle]
    pub extern "C" fn sum_f64_mt(a: *const f64, len: usize) -> f64 {
        unsafe {
            let a = std::slice::from_raw_parts(a, len);
            a.par_iter().sum()
        }
    }

    /// Sum reduction (f32) - multi-threaded
    #[no_mangle]
    pub extern "C" fn sum_f32_mt(a: *const f32, len: usize) -> f32 {
        unsafe {
            let a = std::slice::from_raw_parts(a, len);
            a.par_iter().sum()
        }
    }

    /// Matrix multiplication: C = A * B (f64) - multi-threaded
    /// Parallelizes over rows of C
    #[no_mangle]
    pub extern "C" fn matmul_f64_mt(
        a: *const f64,
        b: *const f64,
        c: *mut f64,
        m: usize,
        n: usize,
        k: usize,
    ) {
        unsafe {
            let a = std::slice::from_raw_parts(a, m * k);
            let b = std::slice::from_raw_parts(b, k * n);
            let c = std::slice::from_raw_parts_mut(c, m * n);

            // Parallel over rows
            c.par_chunks_mut(n)
                .enumerate()
                .for_each(|(i, row)| {
                    row.fill(0.0);
                    for kk in 0..k {
                        let aik = a[i * k + kk];
                        for j in 0..n {
                            row[j] += aik * b[kk * n + j];
                        }
                    }
                });
        }
    }

    /// Matrix multiplication: C = A * B (f32) - multi-threaded
    #[no_mangle]
    pub extern "C" fn matmul_f32_mt(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: usize,
        n: usize,
        k: usize,
    ) {
        unsafe {
            let a = std::slice::from_raw_parts(a, m * k);
            let b = std::slice::from_raw_parts(b, k * n);
            let c = std::slice::from_raw_parts_mut(c, m * n);

            // Parallel over rows
            c.par_chunks_mut(n)
                .enumerate()
                .for_each(|(i, row)| {
                    row.fill(0.0);
                    for kk in 0..k {
                        let aik = a[i * k + kk];
                        for j in 0..n {
                            row[j] += aik * b[kk * n + j];
                        }
                    }
                });
        }
    }

    /// Initialize the thread pool with a specific number of threads
    #[no_mangle]
    pub extern "C" fn init_thread_pool(num_threads: usize) {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .ok(); // Ignore error if already initialized
    }
}

// Re-export threaded functions when feature is enabled
#[cfg(feature = "threads")]
pub use threaded::*;
