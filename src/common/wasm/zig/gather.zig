//! WASM gather/extract kernels for indexing operations.
//!
//! extract: copies elements where condition is nonzero (like np.extract).
//! take_along_axis_2d: gathers elements along axis 0 using index array (2D case).
//! Supported dtypes: f64, f32, i64, u64, i32, u32, i16, u16, i8, u8.

// --- extract: conditional gather ---
// Condition is always i32 (0 or nonzero). Data and output share a dtype.
// Returns the number of elements written to out.

/// Extract f64 elements where cond[i] != 0.
export fn extract_f64(cond: [*]const i32, data: [*]const f64, out: [*]f64, N: u32) u32 {
    var idx: u32 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (cond[i] != 0) {
            out[idx] = data[i];
            idx += 1;
        }
    }
    return idx;
}

/// Extract f32 elements where cond[i] != 0.
export fn extract_f32(cond: [*]const i32, data: [*]const f32, out: [*]f32, N: u32) u32 {
    var idx: u32 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (cond[i] != 0) {
            out[idx] = data[i];
            idx += 1;
        }
    }
    return idx;
}

/// Extract i64 elements where cond[i] != 0.
export fn extract_i64(cond: [*]const i32, data: [*]const i64, out: [*]i64, N: u32) u32 {
    var idx: u32 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (cond[i] != 0) {
            out[idx] = data[i];
            idx += 1;
        }
    }
    return idx;
}

/// Extract u64 elements where cond[i] != 0.
export fn extract_u64(cond: [*]const i32, data: [*]const u64, out: [*]u64, N: u32) u32 {
    var idx: u32 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (cond[i] != 0) {
            out[idx] = data[i];
            idx += 1;
        }
    }
    return idx;
}

/// Extract i32 elements where cond[i] != 0.
export fn extract_i32(cond: [*]const i32, data: [*]const i32, out: [*]i32, N: u32) u32 {
    var idx: u32 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (cond[i] != 0) {
            out[idx] = data[i];
            idx += 1;
        }
    }
    return idx;
}

/// Extract u32 elements where cond[i] != 0.
export fn extract_u32(cond: [*]const i32, data: [*]const u32, out: [*]u32, N: u32) u32 {
    var idx: u32 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (cond[i] != 0) {
            out[idx] = data[i];
            idx += 1;
        }
    }
    return idx;
}

/// Extract i16 elements where cond[i] != 0.
export fn extract_i16(cond: [*]const i32, data: [*]const i16, out: [*]i16, N: u32) u32 {
    var idx: u32 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (cond[i] != 0) {
            out[idx] = data[i];
            idx += 1;
        }
    }
    return idx;
}

/// Extract u16 elements where cond[i] != 0.
export fn extract_u16(cond: [*]const i32, data: [*]const u16, out: [*]u16, N: u32) u32 {
    var idx: u32 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (cond[i] != 0) {
            out[idx] = data[i];
            idx += 1;
        }
    }
    return idx;
}

/// Extract i8 elements where cond[i] != 0.
export fn extract_i8(cond: [*]const i32, data: [*]const i8, out: [*]i8, N: u32) u32 {
    var idx: u32 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (cond[i] != 0) {
            out[idx] = data[i];
            idx += 1;
        }
    }
    return idx;
}

/// Extract u8 elements where cond[i] != 0.
export fn extract_u8(cond: [*]const i32, data: [*]const u8, out: [*]u8, N: u32) u32 {
    var idx: u32 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (cond[i] != 0) {
            out[idx] = data[i];
            idx += 1;
        }
    }
    return idx;
}

// --- take_along_axis for 2D along axis 0 ---
// For a [rows x cols] array and [rows x cols] index array (i32),
// out[r][c] = data[indices[r][c]][c].

/// Gather f64 elements along axis 0 for 2D array.
export fn take_axis0_2d_f64(data: [*]const f64, indices: [*]const i32, out: [*]f64, rows: u32, cols: u32) void {
    var i: u32 = 0;
    const total = rows * cols;
    while (i < total) : (i += 1) {
        const col = i % cols;
        const row_idx: u32 = @intCast(indices[i]);
        out[i] = data[row_idx * cols + col];
    }
}

/// Gather f32 elements along axis 0 for 2D array.
export fn take_axis0_2d_f32(data: [*]const f32, indices: [*]const i32, out: [*]f32, rows: u32, cols: u32) void {
    var i: u32 = 0;
    const total = rows * cols;
    while (i < total) : (i += 1) {
        const col = i % cols;
        const row_idx: u32 = @intCast(indices[i]);
        out[i] = data[row_idx * cols + col];
    }
}

/// Gather i64 elements along axis 0 for 2D array.
export fn take_axis0_2d_i64(data: [*]const i64, indices: [*]const i32, out: [*]i64, rows: u32, cols: u32) void {
    var i: u32 = 0;
    const total = rows * cols;
    while (i < total) : (i += 1) {
        const col = i % cols;
        const row_idx: u32 = @intCast(indices[i]);
        out[i] = data[row_idx * cols + col];
    }
}

/// Gather u64 elements along axis 0 for 2D array.
export fn take_axis0_2d_u64(data: [*]const u64, indices: [*]const i32, out: [*]u64, rows: u32, cols: u32) void {
    var i: u32 = 0;
    const total = rows * cols;
    while (i < total) : (i += 1) {
        const col = i % cols;
        const row_idx: u32 = @intCast(indices[i]);
        out[i] = data[row_idx * cols + col];
    }
}

/// Gather i32 elements along axis 0 for 2D array.
export fn take_axis0_2d_i32(data: [*]const i32, indices: [*]const i32, out: [*]i32, rows: u32, cols: u32) void {
    var i: u32 = 0;
    const total = rows * cols;
    while (i < total) : (i += 1) {
        const col = i % cols;
        const row_idx: u32 = @intCast(indices[i]);
        out[i] = data[row_idx * cols + col];
    }
}

/// Gather u32 elements along axis 0 for 2D array.
export fn take_axis0_2d_u32(data: [*]const u32, indices: [*]const i32, out: [*]u32, rows: u32, cols: u32) void {
    var i: u32 = 0;
    const total = rows * cols;
    while (i < total) : (i += 1) {
        const col = i % cols;
        const row_idx: u32 = @intCast(indices[i]);
        out[i] = data[row_idx * cols + col];
    }
}

/// Gather i16 elements along axis 0 for 2D array.
export fn take_axis0_2d_i16(data: [*]const i16, indices: [*]const i32, out: [*]i16, rows: u32, cols: u32) void {
    var i: u32 = 0;
    const total = rows * cols;
    while (i < total) : (i += 1) {
        const col = i % cols;
        const row_idx: u32 = @intCast(indices[i]);
        out[i] = data[row_idx * cols + col];
    }
}

/// Gather u16 elements along axis 0 for 2D array.
export fn take_axis0_2d_u16(data: [*]const u16, indices: [*]const i32, out: [*]u16, rows: u32, cols: u32) void {
    var i: u32 = 0;
    const total = rows * cols;
    while (i < total) : (i += 1) {
        const col = i % cols;
        const row_idx: u32 = @intCast(indices[i]);
        out[i] = data[row_idx * cols + col];
    }
}

/// Gather i8 elements along axis 0 for 2D array.
export fn take_axis0_2d_i8(data: [*]const i8, indices: [*]const i32, out: [*]i8, rows: u32, cols: u32) void {
    var i: u32 = 0;
    const total = rows * cols;
    while (i < total) : (i += 1) {
        const col = i % cols;
        const row_idx: u32 = @intCast(indices[i]);
        out[i] = data[row_idx * cols + col];
    }
}

/// Gather u8 elements along axis 0 for 2D array.
export fn take_axis0_2d_u8(data: [*]const u8, indices: [*]const i32, out: [*]u8, rows: u32, cols: u32) void {
    var i: u32 = 0;
    const total = rows * cols;
    while (i < total) : (i += 1) {
        const col = i % cols;
        const row_idx: u32 = @intCast(indices[i]);
        out[i] = data[row_idx * cols + col];
    }
}

// --- count_nonzero for condition array ---

/// Count nonzero i32 elements. Used to pre-allocate extract output.
export fn count_nonzero_i32(cond: [*]const i32, N: u32) u32 {
    var count: u32 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (cond[i] != 0) count += 1;
    }
    return count;
}

// --- where: out[i] = cond[i] ? x[i] : y[i] ---

/// Element-wise where for f64: out[i] = cond[i] != 0 ? x[i] : y[i].
export fn where_f64(cond: [*]const i32, x: [*]const f64, y: [*]const f64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (cond[i] != 0) x[i] else y[i];
    }
}

/// Element-wise where for f32.
export fn where_f32(cond: [*]const i32, x: [*]const f32, y: [*]const f32, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (cond[i] != 0) x[i] else y[i];
    }
}

/// Element-wise where for i64.
export fn where_i64(cond: [*]const i32, x: [*]const i64, y: [*]const i64, out: [*]i64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (cond[i] != 0) x[i] else y[i];
    }
}

/// Element-wise where for u64.
export fn where_u64(cond: [*]const i32, x: [*]const u64, y: [*]const u64, out: [*]u64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (cond[i] != 0) x[i] else y[i];
    }
}

/// Element-wise where for i32.
export fn where_i32(cond: [*]const i32, x: [*]const i32, y: [*]const i32, out: [*]i32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (cond[i] != 0) x[i] else y[i];
    }
}

/// Element-wise where for u32.
export fn where_u32(cond: [*]const i32, x: [*]const u32, y: [*]const u32, out: [*]u32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (cond[i] != 0) x[i] else y[i];
    }
}

/// Element-wise where for i16.
export fn where_i16(cond: [*]const i32, x: [*]const i16, y: [*]const i16, out: [*]i16, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (cond[i] != 0) x[i] else y[i];
    }
}

/// Element-wise where for u16.
export fn where_u16(cond: [*]const i32, x: [*]const u16, y: [*]const u16, out: [*]u16, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (cond[i] != 0) x[i] else y[i];
    }
}

/// Element-wise where for i8.
export fn where_i8(cond: [*]const i32, x: [*]const i8, y: [*]const i8, out: [*]i8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (cond[i] != 0) x[i] else y[i];
    }
}

/// Element-wise where for u8.
export fn where_u8(cond: [*]const i32, x: [*]const u8, y: [*]const u8, out: [*]u8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (cond[i] != 0) x[i] else y[i];
    }
}

// --- Tests ---

test "where_f64 basic" {
    const testing = @import("std").testing;
    const cond = [_]i32{ 1, 0, 1, 0 };
    const x = [_]f64{ 10.0, 20.0, 30.0, 40.0 };
    const y = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    var out: [4]f64 = undefined;
    where_f64(&cond, &x, &y, &out, 4);
    try testing.expectApproxEqAbs(out[0], 10.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 30.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 4.0, 1e-10);
}

test "where_u8 basic" {
    const testing = @import("std").testing;
    const cond = [_]i32{ 0, 1, 1, 0 };
    const x = [_]u8{ 10, 20, 30, 40 };
    const y = [_]u8{ 1, 2, 3, 4 };
    var out: [4]u8 = undefined;
    where_u8(&cond, &x, &y, &out, 4);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 20);
    try testing.expectEqual(out[2], 30);
    try testing.expectEqual(out[3], 4);
}

test "extract_f64 basic" {
    const testing = @import("std").testing;
    const cond = [_]i32{ 1, 0, 1, 0, 1 };
    const data = [_]f64{ 10.0, 20.0, 30.0, 40.0, 50.0 };
    var out: [3]f64 = undefined;
    const n = extract_f64(&cond, &data, &out, 5);
    try testing.expectEqual(n, 3);
    try testing.expectApproxEqAbs(out[0], 10.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 30.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 50.0, 1e-10);
}

test "extract_i8 basic" {
    const testing = @import("std").testing;
    const cond = [_]i32{ 0, 1, 0, 1 };
    const data = [_]i8{ 10, 20, 30, 40 };
    var out: [2]i8 = undefined;
    const n = extract_i8(&cond, &data, &out, 4);
    try testing.expectEqual(n, 2);
    try testing.expectEqual(out[0], 20);
    try testing.expectEqual(out[1], 40);
}

test "extract_u8 all true" {
    const testing = @import("std").testing;
    const cond = [_]i32{ 1, 1, 1 };
    const data = [_]u8{ 5, 10, 15 };
    var out: [3]u8 = undefined;
    const n = extract_u8(&cond, &data, &out, 3);
    try testing.expectEqual(n, 3);
    try testing.expectEqual(out[0], 5);
    try testing.expectEqual(out[2], 15);
}

test "take_axis0_2d_f64 basic" {
    const testing = @import("std").testing;
    // data: [[10, 20], [30, 40]]  indices: [[1, 0], [0, 1]]
    // out[0][0] = data[1][0] = 30, out[0][1] = data[0][1] = 20
    // out[1][0] = data[0][0] = 10, out[1][1] = data[1][1] = 40
    const data = [_]f64{ 10.0, 20.0, 30.0, 40.0 };
    const indices = [_]i32{ 1, 0, 0, 1 };
    var out: [4]f64 = undefined;
    take_axis0_2d_f64(&data, &indices, &out, 2, 2);
    try testing.expectApproxEqAbs(out[0], 30.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 20.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 10.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 40.0, 1e-10);
}

test "take_axis0_2d_i8 basic" {
    const testing = @import("std").testing;
    const data = [_]i8{ 1, 2, 3, 4, 5, 6 }; // 3x2
    const indices = [_]i32{ 2, 0, 1, 2, 0, 1 }; // 3x2
    var out: [6]i8 = undefined;
    take_axis0_2d_i8(&data, &indices, &out, 3, 2);
    try testing.expectEqual(out[0], 5); // data[2][0]
    try testing.expectEqual(out[1], 2); // data[0][1]
    try testing.expectEqual(out[2], 3); // data[1][0]
    try testing.expectEqual(out[3], 6); // data[2][1]
}

test "count_nonzero_i32 basic" {
    const testing = @import("std").testing;
    const cond = [_]i32{ 0, 1, 0, 1, 1, 0 };
    try testing.expectEqual(count_nonzero_i32(&cond, 6), 3);
}
