//! WASM indices grid generation kernel.
//!
//! Fills an output buffer with index grids for np.indices().
//! For shape [d0, d1, ...], outputs ndim grids where grid[k][i] = multi_index[k].
//! Supports i32, i64, and f64 output types.

/// Generate index grids for a 2D shape [d0, d1].
/// Output layout: out[0..d0*d1] = row indices, out[d0*d1..2*d0*d1] = col indices.
export fn indices_2d_i32(out: [*]i32, d0: u32, d1: u32) void {
    const gridSize = d0 * d1;

    // Dimension 0: row index = i / d1
    var idx: u32 = 0;
    while (idx < gridSize) : (idx += 1) {
        out[idx] = @intCast(idx / d1);
    }

    // Dimension 1: col index = i % d1
    idx = 0;
    while (idx < gridSize) : (idx += 1) {
        out[gridSize + idx] = @intCast(idx % d1);
    }
}

/// Generate index grids for a 3D shape [d0, d1, d2].
export fn indices_3d_i32(out: [*]i32, d0: u32, d1: u32, d2: u32) void {
    const gridSize = d0 * d1 * d2;
    const stride0 = d1 * d2;

    var idx: u32 = 0;
    while (idx < gridSize) : (idx += 1) {
        out[idx] = @intCast(idx / stride0);
        out[gridSize + idx] = @intCast((idx / d2) % d1);
        out[2 * gridSize + idx] = @intCast(idx % d2);
    }
}

/// Generate index grids for a 2D shape [d0, d1] with i64 output.
export fn indices_2d_i64(out: [*]i64, d0: u32, d1: u32) void {
    const gridSize = d0 * d1;

    var idx: u32 = 0;
    while (idx < gridSize) : (idx += 1) {
        out[idx] = @intCast(idx / d1);
    }

    idx = 0;
    while (idx < gridSize) : (idx += 1) {
        out[gridSize + idx] = @intCast(idx % d1);
    }
}

/// Generate index grids for a 3D shape [d0, d1, d2] with i64 output.
export fn indices_3d_i64(out: [*]i64, d0: u32, d1: u32, d2: u32) void {
    const gridSize = d0 * d1 * d2;
    const stride0 = d1 * d2;

    var idx: u32 = 0;
    while (idx < gridSize) : (idx += 1) {
        out[idx] = @intCast(idx / stride0);
        out[gridSize + idx] = @intCast((idx / d2) % d1);
        out[2 * gridSize + idx] = @intCast(idx % d2);
    }
}

/// Generate index grids for a 2D shape [d0, d1] with f64 output.
export fn indices_2d_f64(out: [*]f64, d0: u32, d1: u32) void {
    const gridSize = d0 * d1;

    var idx: u32 = 0;
    while (idx < gridSize) : (idx += 1) {
        out[idx] = @floatFromInt(idx / d1);
    }

    idx = 0;
    while (idx < gridSize) : (idx += 1) {
        out[gridSize + idx] = @floatFromInt(idx % d1);
    }
}

/// Generate index grids for a 3D shape [d0, d1, d2] with f64 output.
export fn indices_3d_f64(out: [*]f64, d0: u32, d1: u32, d2: u32) void {
    const gridSize = d0 * d1 * d2;
    const stride0 = d1 * d2;

    var idx: u32 = 0;
    while (idx < gridSize) : (idx += 1) {
        out[idx] = @floatFromInt(idx / stride0);
        out[gridSize + idx] = @floatFromInt((idx / d2) % d1);
        out[2 * gridSize + idx] = @floatFromInt(idx % d2);
    }
}

// --- Tests ---

test "indices_2d basic" {
    const testing = @import("std").testing;
    var out: [8]i32 = undefined;
    indices_2d_i32(&out, 2, 2);
    // Row indices: [0, 0, 1, 1]
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 1);
    // Col indices: [0, 1, 0, 1]
    try testing.expectEqual(out[4], 0);
    try testing.expectEqual(out[5], 1);
    try testing.expectEqual(out[6], 0);
    try testing.expectEqual(out[7], 1);
}

test "indices_2d 3x2" {
    const testing = @import("std").testing;
    var out: [12]i32 = undefined;
    indices_2d_i32(&out, 3, 2);
    // Row: [0, 0, 1, 1, 2, 2]
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[4], 2);
    // Col: [0, 1, 0, 1, 0, 1]
    try testing.expectEqual(out[6], 0);
    try testing.expectEqual(out[7], 1);
    try testing.expectEqual(out[10], 0);
    try testing.expectEqual(out[11], 1);
}

test "indices_3d basic" {
    const testing = @import("std").testing;
    var out: [24]i32 = undefined;
    indices_3d_i32(&out, 2, 2, 2);
    // Dim 0: [0,0,0,0, 1,1,1,1]
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[3], 0);
    try testing.expectEqual(out[4], 1);
    try testing.expectEqual(out[7], 1);
    // Dim 1: [0,0,1,1, 0,0,1,1]
    try testing.expectEqual(out[8], 0);
    try testing.expectEqual(out[10], 1);
    // Dim 2: [0,1,0,1, 0,1,0,1]
    try testing.expectEqual(out[16], 0);
    try testing.expectEqual(out[17], 1);
}

test "indices_2d_i64 basic" {
    const testing = @import("std").testing;
    var out: [8]i64 = undefined;
    indices_2d_i64(&out, 2, 2);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 1);
    try testing.expectEqual(out[4], 0);
    try testing.expectEqual(out[5], 1);
    try testing.expectEqual(out[6], 0);
    try testing.expectEqual(out[7], 1);
}

test "indices_3d_i64 basic" {
    const testing = @import("std").testing;
    var out: [24]i64 = undefined;
    indices_3d_i64(&out, 2, 2, 2);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[3], 0);
    try testing.expectEqual(out[4], 1);
    try testing.expectEqual(out[7], 1);
    try testing.expectEqual(out[8], 0);
    try testing.expectEqual(out[10], 1);
    try testing.expectEqual(out[16], 0);
    try testing.expectEqual(out[17], 1);
}

test "indices_2d_f64 basic" {
    const testing = @import("std").testing;
    var out: [8]f64 = undefined;
    indices_2d_f64(&out, 2, 2);
    try testing.expectEqual(out[0], 0.0);
    try testing.expectEqual(out[1], 0.0);
    try testing.expectEqual(out[2], 1.0);
    try testing.expectEqual(out[3], 1.0);
    try testing.expectEqual(out[4], 0.0);
    try testing.expectEqual(out[5], 1.0);
    try testing.expectEqual(out[6], 0.0);
    try testing.expectEqual(out[7], 1.0);
}

test "indices_3d_f64 basic" {
    const testing = @import("std").testing;
    var out: [24]f64 = undefined;
    indices_3d_f64(&out, 2, 2, 2);
    try testing.expectEqual(out[0], 0.0);
    try testing.expectEqual(out[3], 0.0);
    try testing.expectEqual(out[4], 1.0);
    try testing.expectEqual(out[7], 1.0);
    try testing.expectEqual(out[8], 0.0);
    try testing.expectEqual(out[10], 1.0);
    try testing.expectEqual(out[16], 0.0);
    try testing.expectEqual(out[17], 1.0);
}

test "indices_2d_f64 3x2" {
    const testing = @import("std").testing;
    var out: [12]f64 = undefined;
    indices_2d_f64(&out, 3, 2);
    // Row: [0, 0, 1, 1, 2, 2]
    try testing.expectEqual(out[0], 0.0);
    try testing.expectEqual(out[2], 1.0);
    try testing.expectEqual(out[4], 2.0);
    // Col: [0, 1, 0, 1, 0, 1]
    try testing.expectEqual(out[6], 0.0);
    try testing.expectEqual(out[7], 1.0);
    try testing.expectEqual(out[10], 0.0);
    try testing.expectEqual(out[11], 1.0);
}
