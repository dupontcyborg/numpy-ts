//! WASM diff kernels: out[i] = a[i+1] - a[i].
//!
//! 1D: flat diff of N elements.
//! 2D: per-row diff along last axis for contiguous [rows x cols] layout.

const simd = @import("simd.zig");

// ---- 1D diff ----

/// 1D diff for f64: out[i] = a[i+1] - a[i], N = output length.
export fn diff_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, simd.load2_f64(a, i + 1) - simd.load2_f64(a, i));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i + 1] - a[i];
    }
}

/// 1D diff for f32.
export fn diff_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_f32(out, i, simd.load4_f32(a, i + 1) - simd.load4_f32(a, i));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i + 1] - a[i];
    }
}

/// 1D diff for i64 (scalar).
export fn diff_i64(a: [*]const i64, out: [*]i64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = a[i + 1] - a[i];
    }
}

/// 1D diff for i32.
export fn diff_i32(a: [*]const i32, out: [*]i32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_i32(out, i, simd.load4_i32(a, i + 1) - simd.load4_i32(a, i));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i + 1] - a[i];
    }
}

/// 1D diff for i16.
export fn diff_i16(a: [*]const i16, out: [*]i16, N: u32) void {
    const n_simd = N & ~@as(u32, 7);
    var i: u32 = 0;
    while (i < n_simd) : (i += 8) {
        simd.store8_i16(out, i, simd.load8_i16(a, i + 1) - simd.load8_i16(a, i));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i + 1] - a[i];
    }
}

/// 1D diff for i8.
export fn diff_i8(a: [*]const i8, out: [*]i8, N: u32) void {
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    while (i < n_simd) : (i += 16) {
        simd.store16_i8(out, i, simd.load16_i8(a, i + 1) - simd.load16_i8(a, i));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i + 1] - a[i];
    }
}

// ---- 2D diff (per-row along last axis) ----

/// 2D diff for f64: per-row diff on [rows x cols] → [rows x (cols-1)].
export fn diff_2d_f64(a: [*]const f64, out: [*]f64, rows: u32, cols: u32) void {
    const out_cols = cols - 1;
    var r: u32 = 0;
    while (r < rows) : (r += 1) {
        const src = a + r * cols;
        const dst = out + r * out_cols;
        const n_simd = out_cols & ~@as(u32, 1);
        var i: u32 = 0;
        while (i < n_simd) : (i += 2) {
            simd.store2_f64(dst, i, simd.load2_f64(src, i + 1) - simd.load2_f64(src, i));
        }
        while (i < out_cols) : (i += 1) {
            dst[i] = src[i + 1] - src[i];
        }
    }
}

/// 2D diff for f32.
export fn diff_2d_f32(a: [*]const f32, out: [*]f32, rows: u32, cols: u32) void {
    const out_cols = cols - 1;
    var r: u32 = 0;
    while (r < rows) : (r += 1) {
        const src = a + r * cols;
        const dst = out + r * out_cols;
        const n_simd = out_cols & ~@as(u32, 3);
        var i: u32 = 0;
        while (i < n_simd) : (i += 4) {
            simd.store4_f32(dst, i, simd.load4_f32(src, i + 1) - simd.load4_f32(src, i));
        }
        while (i < out_cols) : (i += 1) {
            dst[i] = src[i + 1] - src[i];
        }
    }
}

/// 2D diff for i64.
export fn diff_2d_i64(a: [*]const i64, out: [*]i64, rows: u32, cols: u32) void {
    const out_cols = cols - 1;
    var r: u32 = 0;
    while (r < rows) : (r += 1) {
        const src = a + r * cols;
        const dst = out + r * out_cols;
        var i: u32 = 0;
        while (i < out_cols) : (i += 1) {
            dst[i] = src[i + 1] - src[i];
        }
    }
}

/// 2D diff for i32.
export fn diff_2d_i32(a: [*]const i32, out: [*]i32, rows: u32, cols: u32) void {
    const out_cols = cols - 1;
    var r: u32 = 0;
    while (r < rows) : (r += 1) {
        const src = a + r * cols;
        const dst = out + r * out_cols;
        const n_simd = out_cols & ~@as(u32, 3);
        var i: u32 = 0;
        while (i < n_simd) : (i += 4) {
            simd.store4_i32(dst, i, simd.load4_i32(src, i + 1) - simd.load4_i32(src, i));
        }
        while (i < out_cols) : (i += 1) {
            dst[i] = src[i + 1] - src[i];
        }
    }
}

/// 2D diff for i16.
export fn diff_2d_i16(a: [*]const i16, out: [*]i16, rows: u32, cols: u32) void {
    const out_cols = cols - 1;
    var r: u32 = 0;
    while (r < rows) : (r += 1) {
        const src = a + r * cols;
        const dst = out + r * out_cols;
        const n_simd = out_cols & ~@as(u32, 7);
        var i: u32 = 0;
        while (i < n_simd) : (i += 8) {
            simd.store8_i16(dst, i, simd.load8_i16(src, i + 1) - simd.load8_i16(src, i));
        }
        while (i < out_cols) : (i += 1) {
            dst[i] = src[i + 1] - src[i];
        }
    }
}

/// 2D diff for i8.
export fn diff_2d_i8(a: [*]const i8, out: [*]i8, rows: u32, cols: u32) void {
    const out_cols = cols - 1;
    var r: u32 = 0;
    while (r < rows) : (r += 1) {
        const src = a + r * cols;
        const dst = out + r * out_cols;
        const n_simd = out_cols & ~@as(u32, 15);
        var i: u32 = 0;
        while (i < n_simd) : (i += 16) {
            simd.store16_i8(dst, i, simd.load16_i8(src, i + 1) - simd.load16_i8(src, i));
        }
        while (i < out_cols) : (i += 1) {
            dst[i] = src[i + 1] - src[i];
        }
    }
}

// --- Tests ---

test "diff_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1.0, 3.0, 6.0, 10.0, 15.0 };
    var out: [4]f64 = undefined;
    diff_f64(&a, &out, 4);
    try testing.expectApproxEqAbs(out[0], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 4.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 5.0, 1e-10);
}

test "diff_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 0, 1, 4, 9, 16 };
    var out: [4]i32 = undefined;
    diff_i32(&a, &out, 4);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 3);
    try testing.expectEqual(out[2], 5);
    try testing.expectEqual(out[3], 7);
}

test "diff_i8 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{ 0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 121, 123 };
    var out: [17]i8 = undefined;
    diff_i8(&a, &out, 17);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 2);
    try testing.expectEqual(out[2], 3);
    try testing.expectEqual(out[16], 2);
}

test "diff_2d_f64 basic" {
    const testing = @import("std").testing;
    // 2x4 → 2x3
    const a = [_]f64{ 1.0, 3.0, 6.0, 10.0, 0.0, 2.0, 5.0, 9.0 };
    var out: [6]f64 = undefined;
    diff_2d_f64(&a, &out, 2, 4);
    // row 0: [2, 3, 4]
    try testing.expectApproxEqAbs(out[0], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 4.0, 1e-10);
    // row 1: [2, 3, 4]
    try testing.expectApproxEqAbs(out[3], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[4], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[5], 4.0, 1e-10);
}

test "diff_2d_i32 basic" {
    const testing = @import("std").testing;
    // 2x3 → 2x2
    const a = [_]i32{ 1, 4, 9, 10, 20, 30 };
    var out: [4]i32 = undefined;
    diff_2d_i32(&a, &out, 2, 3);
    try testing.expectEqual(out[0], 3);
    try testing.expectEqual(out[1], 5);
    try testing.expectEqual(out[2], 10);
    try testing.expectEqual(out[3], 10);
}
