/// Unravel flat indices (f64 input) into multi-dimensional indices (f64 output).
/// indices: flat input indices as f64 Numbers, out: output (ndim * N f64 values, row-major by dimension),
/// strides: precomputed strides (ndim i32 values), shape: dimension sizes (ndim i32 values).
/// Computes in i64 internally for exact integer arithmetic.
export fn unravel_index_f64(indices: [*]const f64, out: [*]f64, N: u32, strides: [*]const i32, shape: [*]const i32, ndim: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        var remaining: i64 = @intFromFloat(indices[i]);
        var d: u32 = 0;
        while (d < ndim) : (d += 1) {
            const s: i64 = strides[d];
            const idx = @divTrunc(remaining, s);
            out[d * N + i] = @floatFromInt(@rem(idx, @as(i64, shape[d])));
            remaining = @rem(remaining, s);
        }
    }
}

// --- Tests ---

test "unravel_index_f64 basic 2D" {
    const testing = @import("std").testing;
    const indices = [_]f64{ 0.0, 5.0, 11.0 };
    const strides = [_]i32{ 4, 1 };
    const shape = [_]i32{ 3, 4 };
    var out: [6]f64 = undefined;
    unravel_index_f64(&indices, &out, 3, &strides, &shape, 2);
    try testing.expectEqual(out[0], 0.0); // dim0[0]
    try testing.expectEqual(out[3], 0.0); // dim1[0]
    try testing.expectEqual(out[1], 1.0); // dim0[1]
    try testing.expectEqual(out[4], 1.0); // dim1[1]
    try testing.expectEqual(out[2], 2.0); // dim0[2]
    try testing.expectEqual(out[5], 3.0); // dim1[2]
}

test "unravel_index_f64 3D" {
    const testing = @import("std").testing;
    const indices = [_]f64{ 0.0, 23.0 };
    const strides = [_]i32{ 12, 4, 1 };
    const shape = [_]i32{ 2, 3, 4 };
    var out: [6]f64 = undefined;
    unravel_index_f64(&indices, &out, 2, &strides, &shape, 3);
    try testing.expectEqual(out[0], 0.0);
    try testing.expectEqual(out[2], 0.0);
    try testing.expectEqual(out[4], 0.0);
    try testing.expectEqual(out[1], 1.0);
    try testing.expectEqual(out[3], 2.0);
    try testing.expectEqual(out[5], 3.0);
}
