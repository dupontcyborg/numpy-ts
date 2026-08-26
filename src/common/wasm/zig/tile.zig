//! WASM 2D tile kernel for all numeric types.
//!
//! tile_2d: Tile a [rows x cols] matrix by [rep_rows x rep_cols].
//! Output shape is [rows*rep_rows, cols*rep_cols].
//! Operates on contiguous row-major buffers.
//!
//! Pure block copying, no per-lane arithmetic, so this is one generic body
//! rather than six per-dtype loops. See finding 1.2 in OPEN-FINDINGS.md.

const bulk_mem = @import("bulk_mem.zig");

/// Build the top band of `rows` rows (each source row repeated `rep_cols`
/// times), then replicate that band `rep_rows - 1` more times. The two phases
/// have very different run lengths, so each picks its own copy strategy. All
/// copies are between non-overlapping regions, so `@memcpy` is safe.
inline fn tileT(
    comptime T: type,
    a: [*]const T,
    out: [*]T,
    rows: u32,
    cols: u32,
    rep_rows: u32,
    rep_cols: u32,
) void {
    if (rows == 0 or cols == 0 or rep_rows == 0 or rep_cols == 0) return;
    const out_cols = cols * rep_cols;

    const row_bulk = bulk_mem.useBulk(T, cols);
    for (0..rows) |r| {
        const src_row = a + r * cols;
        const tiled_row = out + r * out_cols;
        for (0..rep_cols) |rc| {
            const dst = tiled_row + rc * cols;
            if (row_bulk) @memcpy(dst[0..cols], src_row[0..cols]) else bulk_mem.copySmall(T, dst, src_row, cols);
        }
    }

    const block_size = rows * out_cols;
    const block_bulk = bulk_mem.useBulk(T, block_size);
    for (1..rep_rows) |rr| {
        const dst = out + rr * block_size;
        if (block_bulk) @memcpy(dst[0..block_size], out[0..block_size]) else bulk_mem.copySmall(T, dst, out, block_size);
    }
}

/// 2D tile for f64: tile a [rows x cols] matrix by [rep_rows x rep_cols].
export fn tile_2d_f64(a: [*]const f64, out: [*]f64, rows: u32, cols: u32, rep_rows: u32, rep_cols: u32) void {
    tileT(f64, a, out, rows, cols, rep_rows, rep_cols);
}

/// 2D tile for f32.
export fn tile_2d_f32(a: [*]const f32, out: [*]f32, rows: u32, cols: u32, rep_rows: u32, rep_cols: u32) void {
    tileT(f32, a, out, rows, cols, rep_rows, rep_cols);
}

/// 2D tile for i64.
export fn tile_2d_i64(a: [*]const i64, out: [*]i64, rows: u32, cols: u32, rep_rows: u32, rep_cols: u32) void {
    tileT(i64, a, out, rows, cols, rep_rows, rep_cols);
}

/// 2D tile for i32.
export fn tile_2d_i32(a: [*]const i32, out: [*]i32, rows: u32, cols: u32, rep_rows: u32, rep_cols: u32) void {
    tileT(i32, a, out, rows, cols, rep_rows, rep_cols);
}

/// 2D tile for i16.
export fn tile_2d_i16(a: [*]const i16, out: [*]i16, rows: u32, cols: u32, rep_rows: u32, rep_cols: u32) void {
    tileT(i16, a, out, rows, cols, rep_rows, rep_cols);
}

/// 2D tile for i8.
export fn tile_2d_i8(a: [*]const i8, out: [*]i8, rows: u32, cols: u32, rep_rows: u32, rep_cols: u32) void {
    tileT(i8, a, out, rows, cols, rep_rows, rep_cols);
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

test "tile_2d_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 1, 2, 3, 4 }; // 2x2
    var out: [16]f32 = undefined; // 4x4
    tile_2d_f32(&a, &out, 2, 2, 2, 2);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-5);
    try testing.expectApproxEqAbs(out[2], 1.0, 1e-5);
    try testing.expectApproxEqAbs(out[3], 2.0, 1e-5);
}

test "tile_2d_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 1, 2, 3, 4 }; // 2x2
    var out: [16]i32 = undefined; // 4x4
    tile_2d_i32(&a, &out, 2, 2, 2, 2);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 2);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 2);
    try testing.expectEqual(out[4], 3);
}

test "tile_2d_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 10, 20 }; // 1x2
    var out: [12]i64 = undefined; // 3x4
    tile_2d_i64(&a, &out, 1, 2, 3, 2);
    // Row 0: [10, 20, 10, 20]
    try testing.expectEqual(out[0], 10);
    try testing.expectEqual(out[1], 20);
    try testing.expectEqual(out[2], 10);
    try testing.expectEqual(out[3], 20);
    // Row 1 = Row 2 = same
    try testing.expectEqual(out[4], 10);
    try testing.expectEqual(out[8], 10);
}

test "tile_2d_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{ 1, 2 }; // 1x2
    var out: [4]i16 = undefined; // 2x2
    tile_2d_i16(&a, &out, 1, 2, 2, 1);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 2);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 2);
}
