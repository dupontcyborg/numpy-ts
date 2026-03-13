//! WASM 2D 90-degree rotation kernels for all numeric types.
//!
//! rot90 k=1 (CCW): dst[outRow, outCol] = src[outCol, cols-1-outRow]
//! Output shape is [cols, rows] for k=1.
//! Operates on contiguous row-major 2D buffers.

const simd = @import("simd.zig");

/// 2D rot90 (k=1, CCW) for f64: dst[r,c] = src[c, cols-1-r].
export fn rot90_f64(a: [*]const f64, out: [*]f64, rows: u32, cols: u32) void {
    const out_rows = cols;
    const out_cols = rows;
    for (0..out_rows) |out_row| {
        const dst_offset = out_row * out_cols;
        const src_col = cols - 1 - @as(u32, @intCast(out_row));
        for (0..out_cols) |out_col| {
            (out + dst_offset)[out_col] = (a + out_col * cols)[src_col];
        }
    }
}

/// 2D rot90 (k=1, CCW) for f32: dst[r,c] = src[c, cols-1-r].
export fn rot90_f32(a: [*]const f32, out: [*]f32, rows: u32, cols: u32) void {
    const out_rows = cols;
    const out_cols = rows;
    for (0..out_rows) |out_row| {
        const dst_offset = out_row * out_cols;
        const src_col = cols - 1 - @as(u32, @intCast(out_row));
        for (0..out_cols) |out_col| {
            (out + dst_offset)[out_col] = (a + out_col * cols)[src_col];
        }
    }
}

/// 2D rot90 (k=1, CCW) for i64, scalar loop (no i64x2 in WASM SIMD).
export fn rot90_i64(a: [*]const i64, out: [*]i64, rows: u32, cols: u32) void {
    const out_rows = cols;
    const out_cols = rows;
    for (0..out_rows) |out_row| {
        const dst_offset = out_row * out_cols;
        const src_col = cols - 1 - @as(u32, @intCast(out_row));
        for (0..out_cols) |out_col| {
            (out + dst_offset)[out_col] = (a + out_col * cols)[src_col];
        }
    }
}

/// 2D rot90 (k=1, CCW) for i32: dst[r,c] = src[c, cols-1-r].
export fn rot90_i32(a: [*]const i32, out: [*]i32, rows: u32, cols: u32) void {
    const out_rows = cols;
    const out_cols = rows;
    for (0..out_rows) |out_row| {
        const dst_offset = out_row * out_cols;
        const src_col = cols - 1 - @as(u32, @intCast(out_row));
        for (0..out_cols) |out_col| {
            (out + dst_offset)[out_col] = (a + out_col * cols)[src_col];
        }
    }
}

/// 2D rot90 (k=1, CCW) for i16: dst[r,c] = src[c, cols-1-r].
export fn rot90_i16(a: [*]const i16, out: [*]i16, rows: u32, cols: u32) void {
    const out_rows = cols;
    const out_cols = rows;
    for (0..out_rows) |out_row| {
        const dst_offset = out_row * out_cols;
        const src_col = cols - 1 - @as(u32, @intCast(out_row));
        for (0..out_cols) |out_col| {
            (out + dst_offset)[out_col] = (a + out_col * cols)[src_col];
        }
    }
}

/// 2D rot90 (k=1, CCW) for i8: dst[r,c] = src[c, cols-1-r].
export fn rot90_i8(a: [*]const i8, out: [*]i8, rows: u32, cols: u32) void {
    const out_rows = cols;
    const out_cols = rows;
    for (0..out_rows) |out_row| {
        const dst_offset = out_row * out_cols;
        const src_col = cols - 1 - @as(u32, @intCast(out_row));
        for (0..out_cols) |out_col| {
            (out + dst_offset)[out_col] = (a + out_col * cols)[src_col];
        }
    }
}

// --- Tests ---

test "rot90_f64 basic" {
    const testing = @import("std").testing;
    // [[1,2],[3,4]] -> [[2,4],[1,3]]
    const a = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    var out: [4]f64 = undefined;
    rot90_f64(&a, &out, 2, 2);
    try testing.expectApproxEqAbs(out[0], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 4.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 3.0, 1e-10);
}

test "rot90_i32 3x2" {
    const testing = @import("std").testing;
    // [[1,2],[3,4],[5,6]] -> [[2,4,6],[1,3,5]]
    const a = [_]i32{ 1, 2, 3, 4, 5, 6 };
    var out: [6]i32 = undefined;
    rot90_i32(&a, &out, 3, 2);
    try testing.expectEqual(out[0], 2);
    try testing.expectEqual(out[1], 4);
    try testing.expectEqual(out[2], 6);
    try testing.expectEqual(out[3], 1);
    try testing.expectEqual(out[4], 3);
    try testing.expectEqual(out[5], 5);
}
