//! Unravel index implementation for i32/u32 and i64/u64 flat indices, outputting f64 for JS compatibility.
//! Uses i32/i64 arithmetic internally for exact integer calculations, outputs f64 for JS Number compatibility.
//! Also includes a version for f64 input

/// Unravel flat indices into multi-dimensional indices (i32/u32 input, f64 output).
/// Uses i32 arithmetic internally, outputs f64 for JS Number compatibility.
/// Works for both i32 and u32 indices (u32 reinterprets as i32 — safe since indices are non-negative).
export fn unravel_index_i32_f64(indices: [*]const i32, out: [*]f64, N: u32, strides: [*]const i32, shape: [*]const i32, ndim: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        var remaining: i32 = indices[i];
        var d: u32 = 0;
        while (d < ndim) : (d += 1) {
            const s = strides[d];
            const idx = @divTrunc(remaining, s);
            out[d * N + i] = @floatFromInt(@rem(idx, shape[d]));
            remaining = @rem(remaining, s);
        }
    }
}

/// Unravel flat indices into multi-dimensional indices (i64/u64 input, f64 output).
/// Uses i64 arithmetic internally, outputs f64 for JS Number compatibility.
/// Works for both i64 and u64 indices (indices are always non-negative).
export fn unravel_index_i64_f64(indices: [*]const i64, out: [*]f64, N: u32, strides: [*]const i64, shape: [*]const i64, ndim: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        var remaining: i64 = indices[i];
        var d: u32 = 0;
        while (d < ndim) : (d += 1) {
            const s = strides[d];
            const idx = @divTrunc(remaining, s);
            out[d * N + i] = @floatFromInt(@rem(idx, shape[d]));
            remaining = @rem(remaining, s);
        }
    }
}

/// Unravel flat indices (f64 input, f64 output).
/// For non-integer input types that get converted to f64 on the JS side.
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

test "unravel_index_i32_f64 basic 2D" {
    const testing = @import("std").testing;
    const indices = [_]i32{ 0, 5, 11 };
    const strides = [_]i32{ 4, 1 };
    const shape = [_]i32{ 3, 4 };
    var out: [6]f64 = undefined;
    unravel_index_i32_f64(&indices, &out, 3, &strides, &shape, 2);
    try testing.expectEqual(out[0], 0.0);
    try testing.expectEqual(out[3], 0.0);
    try testing.expectEqual(out[1], 1.0);
    try testing.expectEqual(out[4], 1.0);
    try testing.expectEqual(out[2], 2.0);
    try testing.expectEqual(out[5], 3.0);
}

test "unravel_index_i32_f64 3D" {
    const testing = @import("std").testing;
    const indices = [_]i32{ 0, 23 };
    const strides = [_]i32{ 12, 4, 1 };
    const shape = [_]i32{ 2, 3, 4 };
    var out: [6]f64 = undefined;
    unravel_index_i32_f64(&indices, &out, 2, &strides, &shape, 3);
    try testing.expectEqual(out[0], 0.0);
    try testing.expectEqual(out[2], 0.0);
    try testing.expectEqual(out[4], 0.0);
    try testing.expectEqual(out[1], 1.0);
    try testing.expectEqual(out[3], 2.0);
    try testing.expectEqual(out[5], 3.0);
}

test "unravel_index_i64_f64 basic 2D" {
    const testing = @import("std").testing;
    const indices = [_]i64{ 0, 5, 11 };
    const strides = [_]i64{ 4, 1 };
    const shape = [_]i64{ 3, 4 };
    var out: [6]f64 = undefined;
    unravel_index_i64_f64(&indices, &out, 3, &strides, &shape, 2);
    try testing.expectEqual(out[0], 0.0);
    try testing.expectEqual(out[3], 0.0);
    try testing.expectEqual(out[1], 1.0);
    try testing.expectEqual(out[4], 1.0);
    try testing.expectEqual(out[2], 2.0);
    try testing.expectEqual(out[5], 3.0);
}

test "unravel_index_f64 basic 2D" {
    const testing = @import("std").testing;
    const indices = [_]f64{ 0.0, 5.0, 11.0 };
    const strides = [_]i32{ 4, 1 };
    const shape = [_]i32{ 3, 4 };
    var out: [6]f64 = undefined;
    unravel_index_f64(&indices, &out, 3, &strides, &shape, 2);
    try testing.expectEqual(out[0], 0.0);
    try testing.expectEqual(out[3], 0.0);
    try testing.expectEqual(out[1], 1.0);
    try testing.expectEqual(out[4], 1.0);
    try testing.expectEqual(out[2], 2.0);
    try testing.expectEqual(out[5], 3.0);
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
