//! WASM 2D constant-pad kernels for all numeric types.
//!
//! pad_2d: Pad a [rows x cols] matrix with `pad_width` zeros on all sides.
//! Output shape is [rows + 2*pad_width, cols + 2*pad_width].
//! Operates on contiguous row-major buffers. Pad value is always 0.

const simd = @import("simd.zig");

/// 2D zero-pad for f64: pad [rows x cols] with `pw` zeros on all sides.
export fn pad_2d_f64(a: [*]const f64, out: [*]f64, rows: u32, cols: u32, pw: u32) void {
    const out_cols = cols + 2 * pw;
    const out_rows = rows + 2 * pw;
    const out_size = out_rows * out_cols;
    // Zero-fill entire output
    var i: u32 = 0;
    const z: simd.V2f64 = @splat(0.0);
    const n_simd = out_size & ~@as(u32, 1);
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, z);
    }
    while (i < out_size) : (i += 1) {
        out[i] = 0.0;
    }
    // Copy source rows into padded position
    for (0..rows) |r| {
        const src_row = a + r * cols;
        const dst_row = out + (r + pw) * out_cols + pw;
        const cn_simd = cols & ~@as(u32, 1);
        var c: u32 = 0;
        while (c < cn_simd) : (c += 2) {
            simd.store2_f64(dst_row, c, simd.load2_f64(src_row, c));
        }
        while (c < cols) : (c += 1) {
            dst_row[c] = src_row[c];
        }
    }
}

/// 2D zero-pad for f32: pad [rows x cols] with `pw` zeros on all sides.
export fn pad_2d_f32(a: [*]const f32, out: [*]f32, rows: u32, cols: u32, pw: u32) void {
    const out_cols = cols + 2 * pw;
    const out_rows = rows + 2 * pw;
    const out_size = out_rows * out_cols;
    var i: u32 = 0;
    const z: simd.V4f32 = @splat(0.0);
    const n_simd = out_size & ~@as(u32, 3);
    while (i < n_simd) : (i += 4) {
        simd.store4_f32(out, i, z);
    }
    while (i < out_size) : (i += 1) {
        out[i] = 0.0;
    }
    for (0..rows) |r| {
        const src_row = a + r * cols;
        const dst_row = out + (r + pw) * out_cols + pw;
        const cn_simd = cols & ~@as(u32, 3);
        var c: u32 = 0;
        while (c < cn_simd) : (c += 4) {
            simd.store4_f32(dst_row, c, simd.load4_f32(src_row, c));
        }
        while (c < cols) : (c += 1) {
            dst_row[c] = src_row[c];
        }
    }
}

/// 2D zero-pad for i64, scalar loop (no i64x2 in WASM SIMD).
export fn pad_2d_i64(a: [*]const i64, out: [*]i64, rows: u32, cols: u32, pw: u32) void {
    const out_cols = cols + 2 * pw;
    const out_rows = rows + 2 * pw;
    const out_size = out_rows * out_cols;
    var i: u32 = 0;
    while (i < out_size) : (i += 1) {
        out[i] = 0;
    }
    for (0..rows) |r| {
        const src_row = a + r * cols;
        const dst_row = out + (r + pw) * out_cols + pw;
        var c: u32 = 0;
        while (c < cols) : (c += 1) {
            dst_row[c] = src_row[c];
        }
    }
}

/// 2D zero-pad for i32 using 4-wide SIMD.
export fn pad_2d_i32(a: [*]const i32, out: [*]i32, rows: u32, cols: u32, pw: u32) void {
    const out_cols = cols + 2 * pw;
    const out_rows = rows + 2 * pw;
    const out_size = out_rows * out_cols;
    var i: u32 = 0;
    const z: simd.V4i32 = @splat(0);
    const n_simd = out_size & ~@as(u32, 3);
    while (i < n_simd) : (i += 4) {
        simd.store4_i32(out, i, z);
    }
    while (i < out_size) : (i += 1) {
        out[i] = 0;
    }
    for (0..rows) |r| {
        const src_row = a + r * cols;
        const dst_row = out + (r + pw) * out_cols + pw;
        const cn_simd = cols & ~@as(u32, 3);
        var c: u32 = 0;
        while (c < cn_simd) : (c += 4) {
            simd.store4_i32(dst_row, c, simd.load4_i32(src_row, c));
        }
        while (c < cols) : (c += 1) {
            dst_row[c] = src_row[c];
        }
    }
}

/// 2D zero-pad for i16 using 8-wide SIMD.
export fn pad_2d_i16(a: [*]const i16, out: [*]i16, rows: u32, cols: u32, pw: u32) void {
    const out_cols = cols + 2 * pw;
    const out_rows = rows + 2 * pw;
    const out_size = out_rows * out_cols;
    var i: u32 = 0;
    const z: simd.V8i16 = @splat(0);
    const n_simd = out_size & ~@as(u32, 7);
    while (i < n_simd) : (i += 8) {
        simd.store8_i16(out, i, z);
    }
    while (i < out_size) : (i += 1) {
        out[i] = 0;
    }
    for (0..rows) |r| {
        const src_row = a + r * cols;
        const dst_row = out + (r + pw) * out_cols + pw;
        const cn_simd = cols & ~@as(u32, 7);
        var c: u32 = 0;
        while (c < cn_simd) : (c += 8) {
            simd.store8_i16(dst_row, c, simd.load8_i16(src_row, c));
        }
        while (c < cols) : (c += 1) {
            dst_row[c] = src_row[c];
        }
    }
}

/// 2D zero-pad for i8 using 16-wide SIMD.
export fn pad_2d_i8(a: [*]const i8, out: [*]i8, rows: u32, cols: u32, pw: u32) void {
    const out_cols = cols + 2 * pw;
    const out_rows = rows + 2 * pw;
    const out_size = out_rows * out_cols;
    var i: u32 = 0;
    const z: simd.V16i8 = @splat(0);
    const n_simd = out_size & ~@as(u32, 15);
    while (i < n_simd) : (i += 16) {
        simd.store16_i8(out, i, z);
    }
    while (i < out_size) : (i += 1) {
        out[i] = 0;
    }
    for (0..rows) |r| {
        const src_row = a + r * cols;
        const dst_row = out + (r + pw) * out_cols + pw;
        const cn_simd = cols & ~@as(u32, 15);
        var c: u32 = 0;
        while (c < cn_simd) : (c += 16) {
            simd.store16_i8(dst_row, c, simd.load16_i8(src_row, c));
        }
        while (c < cols) : (c += 1) {
            dst_row[c] = src_row[c];
        }
    }
}

// --- Tests ---

test "pad_2d_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1.0, 2.0, 3.0, 4.0 }; // 2x2
    var out: [16]f64 = undefined; // 4x4 (pad_width=1)
    pad_2d_f64(&a, &out, 2, 2, 1);
    // Row 0: all zeros
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 0.0, 1e-10);
    // Row 1: [0, 1, 2, 0]
    try testing.expectApproxEqAbs(out[4], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[5], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[6], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[7], 0.0, 1e-10);
    // Row 2: [0, 3, 4, 0]
    try testing.expectApproxEqAbs(out[8], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[9], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[10], 4.0, 1e-10);
    try testing.expectApproxEqAbs(out[11], 0.0, 1e-10);
    // Row 3: all zeros
    try testing.expectApproxEqAbs(out[12], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[15], 0.0, 1e-10);
}

test "pad_2d_i8 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{ 1, 2, 3, 4, 5, 6 }; // 2x3
    var out: [20]i8 = undefined; // 4x5 (pad_width=1)
    pad_2d_i8(&a, &out, 2, 3, 1);
    // Row 0 (5 elements): all zeros
    for (0..5) |c| {
        try testing.expectEqual(out[c], 0);
    }
    // Row 1: [0, 1, 2, 3, 0]
    try testing.expectEqual(out[5], 0);
    try testing.expectEqual(out[6], 1);
    try testing.expectEqual(out[7], 2);
    try testing.expectEqual(out[8], 3);
    try testing.expectEqual(out[9], 0);
    // Row 2: [0, 4, 5, 6, 0]
    try testing.expectEqual(out[10], 0);
    try testing.expectEqual(out[11], 4);
    try testing.expectEqual(out[12], 5);
    try testing.expectEqual(out[13], 6);
    try testing.expectEqual(out[14], 0);
    // Row 3: all zeros
    for (15..20) |c| {
        try testing.expectEqual(out[c], 0);
    }
}
