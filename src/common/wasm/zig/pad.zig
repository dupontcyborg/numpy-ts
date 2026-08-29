//! WASM 2D constant-pad kernels for all numeric types.
//!
//! pad_2d: Pad a [rows x cols] matrix with `pad_width` zeros on all sides.
//! Output shape is [rows + 2*pad_width, cols + 2*pad_width].
//! Operates on contiguous row-major buffers. Pad value is always 0.
//!
//! Pure fill + copy, no per-lane arithmetic, so this is one generic body rather
//! than six per-dtype loops. Only the border is zeroed, not the whole output —
//! interior elements are written once, by the row copy.

const bulk_mem = @import("bulk_mem.zig");

/// Zero the border (top band, bottom band, then a `pw`-wide margin either side
/// of each interior row), then drop each source row into place. Margins are 1-2
/// elements wide, so they always use direct stores. `a` and `out` are distinct
/// buffers, so the row copies never overlap.
inline fn padT(comptime T: type, a: [*]const T, out: [*]T, rows: u32, cols: u32, pw: u32) void {
    const out_cols = cols + 2 * pw;
    const out_rows = rows + 2 * pw;
    const out_size = out_rows * out_cols;
    if (out_size == 0) return;

    // No padding: the output is the input.
    if (pw == 0) {
        if (rows * cols != 0) bulk_mem.copyRun(T, out, a, out_size);
        return;
    }

    // Nothing to copy in — the whole output is border.
    if (rows == 0 or cols == 0) {
        bulk_mem.fillZero(T, out, out_size);
        return;
    }

    const band = pw * out_cols;
    bulk_mem.fillZero(T, out, band);
    bulk_mem.fillZero(T, out + (out_size - band), band);

    const row_bulk = bulk_mem.useBulk(T, cols);
    for (0..rows) |r| {
        const dst_row = out + (r + pw) * out_cols;
        var k: u32 = 0;
        while (k < pw) : (k += 1) {
            dst_row[k] = 0;
            dst_row[pw + cols + k] = 0;
        }
        const src_row = a + r * cols;
        const dst = dst_row + pw;
        if (row_bulk) @memcpy(dst[0..cols], src_row[0..cols]) else bulk_mem.copySmall(T, dst, src_row, cols);
    }
}

/// 2D zero-pad for f64: pad [rows x cols] with `pw` zeros on all sides.
export fn pad_2d_f64(a: [*]const f64, out: [*]f64, rows: u32, cols: u32, pw: u32) void {
    padT(f64, a, out, rows, cols, pw);
}

/// 2D zero-pad for f32.
export fn pad_2d_f32(a: [*]const f32, out: [*]f32, rows: u32, cols: u32, pw: u32) void {
    padT(f32, a, out, rows, cols, pw);
}

/// 2D zero-pad for i64.
export fn pad_2d_i64(a: [*]const i64, out: [*]i64, rows: u32, cols: u32, pw: u32) void {
    padT(i64, a, out, rows, cols, pw);
}

/// 2D zero-pad for i32.
export fn pad_2d_i32(a: [*]const i32, out: [*]i32, rows: u32, cols: u32, pw: u32) void {
    padT(i32, a, out, rows, cols, pw);
}

/// 2D zero-pad for i16.
export fn pad_2d_i16(a: [*]const i16, out: [*]i16, rows: u32, cols: u32, pw: u32) void {
    padT(i16, a, out, rows, cols, pw);
}

/// 2D zero-pad for i8.
export fn pad_2d_i8(a: [*]const i8, out: [*]i8, rows: u32, cols: u32, pw: u32) void {
    padT(i8, a, out, rows, cols, pw);
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

test "pad_2d_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 }; // 2x2
    var out: [16]f32 = undefined; // 4x4
    pad_2d_f32(&a, &out, 2, 2, 1);
    // All edges should be zero
    for (0..4) |c| {
        try testing.expectApproxEqAbs(out[c], 0.0, 1e-5); // row 0
    }
    try testing.expectApproxEqAbs(out[5], 1.0, 1e-5);
    try testing.expectApproxEqAbs(out[6], 2.0, 1e-5);
    try testing.expectApproxEqAbs(out[9], 3.0, 1e-5);
    try testing.expectApproxEqAbs(out[10], 4.0, 1e-5);
}

test "pad_2d_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 1, 2, 3, 4 }; // 2x2
    var out: [16]i32 = undefined; // 4x4
    pad_2d_i32(&a, &out, 2, 2, 1);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[5], 1);
    try testing.expectEqual(out[6], 2);
    try testing.expectEqual(out[9], 3);
    try testing.expectEqual(out[10], 4);
    try testing.expectEqual(out[15], 0);
}

test "pad_2d_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 1, 2, 3, 4 }; // 2x2
    var out: [16]i64 = undefined; // 4x4
    pad_2d_i64(&a, &out, 2, 2, 1);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[5], 1);
    try testing.expectEqual(out[6], 2);
    try testing.expectEqual(out[15], 0);
}

test "pad_2d_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{ 1, 2, 3, 4, 5, 6 }; // 2x3
    var out: [20]i16 = undefined; // 4x5
    pad_2d_i16(&a, &out, 2, 3, 1);
    for (0..5) |c| {
        try testing.expectEqual(out[c], 0);
    }
    try testing.expectEqual(out[6], 1);
    try testing.expectEqual(out[7], 2);
    try testing.expectEqual(out[8], 3);
}

test "pad_2d_f64 pad_width=2" {
    const testing = @import("std").testing;
    const a = [_]f64{5.0}; // 1x1
    var out: [25]f64 = undefined; // 5x5
    pad_2d_f64(&a, &out, 1, 1, 2);
    // center should be 5.0
    try testing.expectApproxEqAbs(out[12], 5.0, 1e-10);
    // corners should be 0
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[24], 0.0, 1e-10);
}

test "pad_2d_f64 border-only fill leaves no gaps" {
    const testing = @import("std").testing;
    // The border is filled as three separate pieces (top band, bottom band, side
    // margins), so this checks every output cell rather than a few samples.
    const rows: u32 = 3;
    const cols: u32 = 4;
    const pw: u32 = 2;
    const oc = cols + 2 * pw;
    const orow = rows + 2 * pw;
    var a: [rows * cols]f64 = undefined;
    for (&a, 0..) |*p, i| p.* = @floatFromInt(i + 1);
    var out = [_]f64{-1} ** (oc * orow);
    pad_2d_f64(&a, &out, rows, cols, pw);
    for (0..orow) |r| {
        for (0..oc) |c| {
            const v = out[r * oc + c];
            const inside = r >= pw and r < pw + rows and c >= pw and c < pw + cols;
            if (inside) {
                try testing.expectEqual(a[(r - pw) * cols + (c - pw)], v);
            } else {
                try testing.expectEqual(@as(f64, 0), v);
            }
        }
    }
}

test "pad_2d_i32 pw=0 is a straight copy" {
    const testing = @import("std").testing;
    const a = [_]i32{ 1, 2, 3, 4, 5, 6 };
    var out = [_]i32{-1} ** 6;
    pad_2d_i32(&a, &out, 2, 3, 0);
    try testing.expectEqualSlices(i32, &a, &out);
}

test "pad_2d_i8 zero-sized input is all border" {
    const testing = @import("std").testing;
    const a = [_]i8{};
    var out = [_]i8{-1} ** 9;
    pad_2d_i8(&a, &out, 0, 0, 3);
    // out_rows = out_cols = 6 would be 36; with rows=cols=0 and pw=3 the output
    // is 6x6, so this only checks the first 9 cells are zeroed.
    pad_2d_i8(&a, &out, 0, 3, 0);
    for (out[0..0]) |v| try testing.expectEqual(@as(i8, 0), v);
    var out2 = [_]i8{-1} ** 36;
    pad_2d_i8(&a, &out2, 0, 0, 3);
    for (out2) |v| try testing.expectEqual(@as(i8, 0), v);
}
