//! WASM 2D tile kernel for all numeric types.
//!
//! tile_2d: Tile a [rows x cols] matrix by [rep_rows x rep_cols].
//! Output shape is [rows*rep_rows, cols*rep_cols].
//! Operates on contiguous row-major buffers.

const simd = @import("simd.zig");

/// 2D tile for f64: tile a [rows x cols] matrix by [rep_rows x rep_cols].
export fn tile_2d_f64(a: [*]const f64, out: [*]f64, rows: u32, cols: u32, rep_rows: u32, rep_cols: u32) void {
    const out_cols = cols * rep_cols;
    // Build one tiled row (cols repeated rep_cols times)
    for (0..rows) |r| {
        const src_row = a + r * cols;
        const tiled_row = out + r * out_cols;
        for (0..rep_cols) |rc| {
            const dst = tiled_row + rc * cols;
            const n_simd = cols & ~@as(u32, 1);
            var c: u32 = 0;
            while (c < n_simd) : (c += 2) {
                simd.store2_f64(dst, c, simd.load2_f64(src_row, c));
            }
            while (c < cols) : (c += 1) {
                dst[c] = src_row[c];
            }
        }
    }
    // Replicate the first block of rows for remaining row reps
    const block_size = rows * out_cols;
    for (1..rep_rows) |rr| {
        const dst = out + rr * block_size;
        const n_simd = block_size & ~@as(u32, 1);
        var i: u32 = 0;
        while (i < n_simd) : (i += 2) {
            simd.store2_f64(dst, i, simd.load2_f64(out, i));
        }
        while (i < block_size) : (i += 1) {
            dst[i] = out[i];
        }
    }
}

/// 2D tile for f32: tile a [rows x cols] matrix by [rep_rows x rep_cols].
export fn tile_2d_f32(a: [*]const f32, out: [*]f32, rows: u32, cols: u32, rep_rows: u32, rep_cols: u32) void {
    const out_cols = cols * rep_cols;
    for (0..rows) |r| {
        const src_row = a + r * cols;
        const tiled_row = out + r * out_cols;
        for (0..rep_cols) |rc| {
            const dst = tiled_row + rc * cols;
            const n_simd = cols & ~@as(u32, 3);
            var c: u32 = 0;
            while (c < n_simd) : (c += 4) {
                simd.store4_f32(dst, c, simd.load4_f32(src_row, c));
            }
            while (c < cols) : (c += 1) {
                dst[c] = src_row[c];
            }
        }
    }
    const block_size = rows * out_cols;
    for (1..rep_rows) |rr| {
        const dst = out + rr * block_size;
        const n_simd = block_size & ~@as(u32, 3);
        var i: u32 = 0;
        while (i < n_simd) : (i += 4) {
            simd.store4_f32(dst, i, simd.load4_f32(out, i));
        }
        while (i < block_size) : (i += 1) {
            dst[i] = out[i];
        }
    }
}

/// 2D tile for i64, scalar loop (no i64x2 in WASM SIMD).
export fn tile_2d_i64(a: [*]const i64, out: [*]i64, rows: u32, cols: u32, rep_rows: u32, rep_cols: u32) void {
    const out_cols = cols * rep_cols;
    for (0..rows) |r| {
        const src_row = a + r * cols;
        const tiled_row = out + r * out_cols;
        for (0..rep_cols) |rc| {
            const dst = tiled_row + rc * cols;
            var c: u32 = 0;
            while (c < cols) : (c += 1) {
                dst[c] = src_row[c];
            }
        }
    }
    const block_size = rows * out_cols;
    for (1..rep_rows) |rr| {
        const dst = out + rr * block_size;
        var i: u32 = 0;
        while (i < block_size) : (i += 1) {
            dst[i] = out[i];
        }
    }
}

/// 2D tile for i32 using 4-wide SIMD.
export fn tile_2d_i32(a: [*]const i32, out: [*]i32, rows: u32, cols: u32, rep_rows: u32, rep_cols: u32) void {
    const out_cols = cols * rep_cols;
    for (0..rows) |r| {
        const src_row = a + r * cols;
        const tiled_row = out + r * out_cols;
        for (0..rep_cols) |rc| {
            const dst = tiled_row + rc * cols;
            const n_simd = cols & ~@as(u32, 3);
            var c: u32 = 0;
            while (c < n_simd) : (c += 4) {
                simd.store4_i32(dst, c, simd.load4_i32(src_row, c));
            }
            while (c < cols) : (c += 1) {
                dst[c] = src_row[c];
            }
        }
    }
    const block_size = rows * out_cols;
    for (1..rep_rows) |rr| {
        const dst = out + rr * block_size;
        const n_simd = block_size & ~@as(u32, 3);
        var i: u32 = 0;
        while (i < n_simd) : (i += 4) {
            simd.store4_i32(dst, i, simd.load4_i32(out, i));
        }
        while (i < block_size) : (i += 1) {
            dst[i] = out[i];
        }
    }
}

/// 2D tile for i16 using 8-wide SIMD.
export fn tile_2d_i16(a: [*]const i16, out: [*]i16, rows: u32, cols: u32, rep_rows: u32, rep_cols: u32) void {
    const out_cols = cols * rep_cols;
    for (0..rows) |r| {
        const src_row = a + r * cols;
        const tiled_row = out + r * out_cols;
        for (0..rep_cols) |rc| {
            const dst = tiled_row + rc * cols;
            const n_simd = cols & ~@as(u32, 7);
            var c: u32 = 0;
            while (c < n_simd) : (c += 8) {
                simd.store8_i16(dst, c, simd.load8_i16(src_row, c));
            }
            while (c < cols) : (c += 1) {
                dst[c] = src_row[c];
            }
        }
    }
    const block_size = rows * out_cols;
    for (1..rep_rows) |rr| {
        const dst = out + rr * block_size;
        const n_simd = block_size & ~@as(u32, 7);
        var i: u32 = 0;
        while (i < n_simd) : (i += 8) {
            simd.store8_i16(dst, i, simd.load8_i16(out, i));
        }
        while (i < block_size) : (i += 1) {
            dst[i] = out[i];
        }
    }
}

/// 2D tile for i8 using 16-wide SIMD.
export fn tile_2d_i8(a: [*]const i8, out: [*]i8, rows: u32, cols: u32, rep_rows: u32, rep_cols: u32) void {
    const out_cols = cols * rep_cols;
    for (0..rows) |r| {
        const src_row = a + r * cols;
        const tiled_row = out + r * out_cols;
        for (0..rep_cols) |rc| {
            const dst = tiled_row + rc * cols;
            const n_simd = cols & ~@as(u32, 15);
            var c: u32 = 0;
            while (c < n_simd) : (c += 16) {
                simd.store16_i8(dst, c, simd.load16_i8(src_row, c));
            }
            while (c < cols) : (c += 1) {
                dst[c] = src_row[c];
            }
        }
    }
    const block_size = rows * out_cols;
    for (1..rep_rows) |rr| {
        const dst = out + rr * block_size;
        const n_simd = block_size & ~@as(u32, 15);
        var i: u32 = 0;
        while (i < n_simd) : (i += 16) {
            simd.store16_i8(dst, i, simd.load16_i8(out, i));
        }
        while (i < block_size) : (i += 1) {
            dst[i] = out[i];
        }
    }
}

// --- Tests ---

test "tile_2d_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1.0, 2.0, 3.0, 4.0 }; // 2x2
    var out: [16]f64 = undefined; // 4x4
    tile_2d_f64(&a, &out, 2, 2, 2, 2);
    // Row 0: [1,2,1,2]
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 2.0, 1e-10);
    // Row 1: [3,4,3,4]
    try testing.expectApproxEqAbs(out[4], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[5], 4.0, 1e-10);
    try testing.expectApproxEqAbs(out[6], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[7], 4.0, 1e-10);
    // Row 2: [1,2,1,2] (row rep)
    try testing.expectApproxEqAbs(out[8], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[9], 2.0, 1e-10);
}

test "tile_2d_i8 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{ 1, 2, 3, 4 }; // 2x2
    var out: [16]i8 = undefined; // 4x4
    tile_2d_i8(&a, &out, 2, 2, 2, 2);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 2);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 2);
    try testing.expectEqual(out[4], 3);
    try testing.expectEqual(out[5], 4);
    try testing.expectEqual(out[8], 1);
}
