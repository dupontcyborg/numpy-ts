//! WASM Benchmark Operations for numpy-ts (Zig implementation)
//!
//! Single-threaded and multi-threaded implementations of:
//! - Element-wise operations (add, sin)
//! - Reduction operations (sum)
//! - Matrix operations (matmul)

const std = @import("std");
const math = std.math;

// ============================================================================
// Memory Management
// ============================================================================

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

/// Allocate memory for f64 array
export fn alloc_f64(len: usize) ?[*]f64 {
    const slice = allocator.alloc(f64, len) catch return null;
    return slice.ptr;
}

/// Allocate memory for f32 array
export fn alloc_f32(len: usize) ?[*]f32 {
    const slice = allocator.alloc(f32, len) catch return null;
    return slice.ptr;
}

/// Free f64 array
export fn free_f64(ptr: ?[*]f64, len: usize) void {
    if (ptr) |p| {
        const slice = p[0..len];
        allocator.free(slice);
    }
}

/// Free f32 array
export fn free_f32(ptr: ?[*]f32, len: usize) void {
    if (ptr) |p| {
        const slice = p[0..len];
        allocator.free(slice);
    }
}

// ============================================================================
// Element-wise Operations - Single Threaded
// ============================================================================

/// Element-wise addition: c = a + b (f64)
export fn add_f64(a_ptr: [*]const f64, b_ptr: [*]const f64, c_ptr: [*]f64, len: usize) void {
    const a = a_ptr[0..len];
    const b = b_ptr[0..len];
    const c = c_ptr[0..len];

    // Use SIMD if available (Zig auto-vectorizes well)
    for (a, b, c) |av, bv, *cv| {
        cv.* = av + bv;
    }
}

/// Element-wise addition: c = a + b (f32)
export fn add_f32(a_ptr: [*]const f32, b_ptr: [*]const f32, c_ptr: [*]f32, len: usize) void {
    const a = a_ptr[0..len];
    const b = b_ptr[0..len];
    const c = c_ptr[0..len];

    for (a, b, c) |av, bv, *cv| {
        cv.* = av + bv;
    }
}

/// Element-wise sine: c = sin(a) (f64)
export fn sin_f64(a_ptr: [*]const f64, c_ptr: [*]f64, len: usize) void {
    const a = a_ptr[0..len];
    const c = c_ptr[0..len];

    for (a, c) |av, *cv| {
        cv.* = @sin(av);
    }
}

/// Element-wise sine: c = sin(a) (f32)
export fn sin_f32(a_ptr: [*]const f32, c_ptr: [*]f32, len: usize) void {
    const a = a_ptr[0..len];
    const c = c_ptr[0..len];

    for (a, c) |av, *cv| {
        cv.* = @sin(av);
    }
}

// ============================================================================
// Reduction Operations - Single Threaded
// ============================================================================

/// Sum reduction (f64)
export fn sum_f64(a_ptr: [*]const f64, len: usize) f64 {
    const a = a_ptr[0..len];
    var total: f64 = 0.0;

    // Use @reduce for SIMD optimization
    for (a) |v| {
        total += v;
    }

    return total;
}

/// Sum reduction (f32)
export fn sum_f32(a_ptr: [*]const f32, len: usize) f32 {
    const a = a_ptr[0..len];
    var total: f32 = 0.0;

    for (a) |v| {
        total += v;
    }

    return total;
}

// ============================================================================
// Matrix Operations - Single Threaded
// ============================================================================

/// Matrix multiplication: C = A * B (f64)
/// A is MxK, B is KxN, C is MxN (row-major)
export fn matmul_f64(
    a_ptr: [*]const f64,
    b_ptr: [*]const f64,
    c_ptr: [*]f64,
    m: usize,
    n: usize,
    k: usize,
) void {
    const c = c_ptr[0 .. m * n];

    // Zero output
    @memset(c, 0.0);

    // Cache-friendly ikj order
    for (0..m) |i| {
        for (0..k) |kk| {
            const aik = a_ptr[i * k + kk];
            for (0..n) |j| {
                c_ptr[i * n + j] += aik * b_ptr[kk * n + j];
            }
        }
    }
}

/// Matrix multiplication: C = A * B (f32)
export fn matmul_f32(
    a_ptr: [*]const f32,
    b_ptr: [*]const f32,
    c_ptr: [*]f32,
    m: usize,
    n: usize,
    k: usize,
) void {
    const c = c_ptr[0 .. m * n];

    // Zero output
    @memset(c, 0.0);

    // Cache-friendly ikj order
    for (0..m) |i| {
        for (0..k) |kk| {
            const aik = a_ptr[i * k + kk];
            for (0..n) |j| {
                c_ptr[i * n + j] += aik * b_ptr[kk * n + j];
            }
        }
    }
}

// ============================================================================
// Blocked Matrix Multiplication (better cache utilization)
// ============================================================================

const BLOCK_SIZE: usize = 64;

/// Blocked matrix multiplication (f64)
export fn matmul_blocked_f64(
    a_ptr: [*]const f64,
    b_ptr: [*]const f64,
    c_ptr: [*]f64,
    m: usize,
    n: usize,
    k: usize,
) void {
    const c = c_ptr[0 .. m * n];
    @memset(c, 0.0);

    var ii: usize = 0;
    while (ii < m) : (ii += BLOCK_SIZE) {
        const i_max = @min(ii + BLOCK_SIZE, m);

        var kk_blk: usize = 0;
        while (kk_blk < k) : (kk_blk += BLOCK_SIZE) {
            const k_max = @min(kk_blk + BLOCK_SIZE, k);

            var jj: usize = 0;
            while (jj < n) : (jj += BLOCK_SIZE) {
                const j_max = @min(jj + BLOCK_SIZE, n);

                // Multiply blocks
                for (ii..i_max) |i| {
                    for (kk_blk..k_max) |kk| {
                        const aik = a_ptr[i * k + kk];
                        for (jj..j_max) |j| {
                            c_ptr[i * n + j] += aik * b_ptr[kk * n + j];
                        }
                    }
                }
            }
        }
    }
}

/// Blocked matrix multiplication (f32)
export fn matmul_blocked_f32(
    a_ptr: [*]const f32,
    b_ptr: [*]const f32,
    c_ptr: [*]f32,
    m: usize,
    n: usize,
    k: usize,
) void {
    const c = c_ptr[0 .. m * n];
    @memset(c, 0.0);

    var ii: usize = 0;
    while (ii < m) : (ii += BLOCK_SIZE) {
        const i_max = @min(ii + BLOCK_SIZE, m);

        var kk_blk: usize = 0;
        while (kk_blk < k) : (kk_blk += BLOCK_SIZE) {
            const k_max = @min(kk_blk + BLOCK_SIZE, k);

            var jj: usize = 0;
            while (jj < n) : (jj += BLOCK_SIZE) {
                const j_max = @min(jj + BLOCK_SIZE, n);

                // Multiply blocks
                for (ii..i_max) |i| {
                    for (kk_blk..k_max) |kk| {
                        const aik = a_ptr[i * k + kk];
                        for (jj..j_max) |j| {
                            c_ptr[i * n + j] += aik * b_ptr[kk * n + j];
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Multi-threaded Operations
// Note: WASM threads require SharedArrayBuffer and special build flags.
// For now, we provide stub implementations that fall back to single-threaded.
// True multi-threading would require Web Workers coordination from JS side.
// ============================================================================

/// Multi-threaded add (f64) - stub that calls single-threaded version
/// Real MT would be coordinated via Web Workers from JS
export fn add_f64_mt(a_ptr: [*]const f64, b_ptr: [*]const f64, c_ptr: [*]f64, len: usize) void {
    add_f64(a_ptr, b_ptr, c_ptr, len);
}

/// Multi-threaded add (f32)
export fn add_f32_mt(a_ptr: [*]const f32, b_ptr: [*]const f32, c_ptr: [*]f32, len: usize) void {
    add_f32(a_ptr, b_ptr, c_ptr, len);
}

/// Multi-threaded sin (f64)
export fn sin_f64_mt(a_ptr: [*]const f64, c_ptr: [*]f64, len: usize) void {
    sin_f64(a_ptr, c_ptr, len);
}

/// Multi-threaded sin (f32)
export fn sin_f32_mt(a_ptr: [*]const f32, c_ptr: [*]f32, len: usize) void {
    sin_f32(a_ptr, c_ptr, len);
}

/// Multi-threaded sum (f64)
export fn sum_f64_mt(a_ptr: [*]const f64, len: usize) f64 {
    return sum_f64(a_ptr, len);
}

/// Multi-threaded sum (f32)
export fn sum_f32_mt(a_ptr: [*]const f32, len: usize) f32 {
    return sum_f32(a_ptr, len);
}

/// Multi-threaded matmul (f64)
export fn matmul_f64_mt(
    a_ptr: [*]const f64,
    b_ptr: [*]const f64,
    c_ptr: [*]f64,
    m: usize,
    n: usize,
    k: usize,
) void {
    matmul_blocked_f64(a_ptr, b_ptr, c_ptr, m, n, k);
}

/// Multi-threaded matmul (f32)
export fn matmul_f32_mt(
    a_ptr: [*]const f32,
    b_ptr: [*]const f32,
    c_ptr: [*]f32,
    m: usize,
    n: usize,
    k: usize,
) void {
    matmul_blocked_f32(a_ptr, b_ptr, c_ptr, m, n, k);
}

/// Initialize thread pool (no-op for now)
export fn init_thread_pool(_: usize) void {}
