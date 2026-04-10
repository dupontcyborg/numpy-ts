//! WASM argpartition kernels for all numeric types.
//!
//! Argpartition: fills out[0..N] with indices such that out[kth] is the index
//! of the element that would be at position kth in a sorted array, all indices
//! before kth reference values <= a[out[kth]], and all after reference values
//! >= a[out[kth]].
//!
//! Output is f64 for JS ergonomics. Internally partitions u32 indices in the
//! lower half of the f64 output buffer, then converts in-place via SIMD.
//!
//! Uses quickselect with median-of-three pivot selection.
//! Supported dtypes: f64, f32, i64, u64, i32, u32, i16, u16, i8, u8.
//! Floating-point types treat NaN as greater than all values.

const sc = @import("sorting_common.zig");

/// Helper: partition into u32 indices in the f64 buffer, then convert to f64 in-place.
inline fn argpartitionToF64(comptime T: type, a: [*]const T, out: [*]f64, N: u32, kth: u32) void {
    const out_u32: [*]u32 = @ptrCast(out);
    sc.initIndices(out_u32, N);
    if (N <= 1 or kth >= N) {
        sc.indicesToF64(out_u32, out, N);
        return;
    }
    sc.quickselectIndirect(T, a, out_u32, 0, N - 1, kth);
    sc.indicesToF64(out_u32, out, N);
}

/// Helper: batch slice argpartition into u32, then convert to f64.
inline fn argpartitionSlicesToF64(comptime T: type, a: [*]const T, out: [*]f64, sliceSize: u32, numSlices: u32, kth: u32) void {
    const out_u32: [*]u32 = @ptrCast(out);
    sc.quickselectIndirectSlices(T, a, out_u32, sliceSize, numSlices, kth);
    sc.sliceIndicesToF64(out_u32, out, sliceSize, numSlices);
}

// --- Single-array argpartition (f64 index output) ---

/// Argpartition for f64: indices as f64. NaN sorts to end.
export fn argpartition_f64(a: [*]const f64, out: [*]f64, N: u32, kth: u32) void {
    argpartitionToF64(f64, a, out, N, kth);
}

/// Argpartition for f32: indices as f64. NaN sorts to end.
export fn argpartition_f32(a: [*]const f32, out: [*]f64, N: u32, kth: u32) void {
    argpartitionToF64(f32, a, out, N, kth);
}

/// Argpartition for i64: indices as f64.
export fn argpartition_i64(a: [*]const i64, out: [*]f64, N: u32, kth: u32) void {
    argpartitionToF64(i64, a, out, N, kth);
}

/// Argpartition for u64: indices as f64.
export fn argpartition_u64(a: [*]const u64, out: [*]f64, N: u32, kth: u32) void {
    argpartitionToF64(u64, a, out, N, kth);
}

/// Argpartition for i32: indices as f64.
export fn argpartition_i32(a: [*]const i32, out: [*]f64, N: u32, kth: u32) void {
    argpartitionToF64(i32, a, out, N, kth);
}

/// Argpartition for u32: indices as f64.
export fn argpartition_u32(a: [*]const u32, out: [*]f64, N: u32, kth: u32) void {
    argpartitionToF64(u32, a, out, N, kth);
}

/// Argpartition for i16: indices as f64.
export fn argpartition_i16(a: [*]const i16, out: [*]f64, N: u32, kth: u32) void {
    argpartitionToF64(i16, a, out, N, kth);
}

/// Argpartition for u16: indices as f64.
export fn argpartition_u16(a: [*]const u16, out: [*]f64, N: u32, kth: u32) void {
    argpartitionToF64(u16, a, out, N, kth);
}

/// Argpartition for i8: indices as f64.
export fn argpartition_i8(a: [*]const i8, out: [*]f64, N: u32, kth: u32) void {
    argpartitionToF64(i8, a, out, N, kth);
}

/// Argpartition for u8: indices as f64.
export fn argpartition_u8(a: [*]const u8, out: [*]f64, N: u32, kth: u32) void {
    argpartitionToF64(u8, a, out, N, kth);
}

// --- Batch slice argpartition (f64 index output) ---

/// Argpartition numSlices contiguous f64 slices at kth position. Indices as f64.
export fn argpartition_slices_f64(a: [*]const f64, out: [*]f64, sliceSize: u32, numSlices: u32, kth: u32) void {
    argpartitionSlicesToF64(f64, a, out, sliceSize, numSlices, kth);
}

/// Argpartition numSlices contiguous f32 slices at kth position. Indices as f64.
export fn argpartition_slices_f32(a: [*]const f32, out: [*]f64, sliceSize: u32, numSlices: u32, kth: u32) void {
    argpartitionSlicesToF64(f32, a, out, sliceSize, numSlices, kth);
}

/// Argpartition numSlices contiguous i64 slices at kth position. Indices as f64.
export fn argpartition_slices_i64(a: [*]const i64, out: [*]f64, sliceSize: u32, numSlices: u32, kth: u32) void {
    argpartitionSlicesToF64(i64, a, out, sliceSize, numSlices, kth);
}

/// Argpartition numSlices contiguous u64 slices at kth position. Indices as f64.
export fn argpartition_slices_u64(a: [*]const u64, out: [*]f64, sliceSize: u32, numSlices: u32, kth: u32) void {
    argpartitionSlicesToF64(u64, a, out, sliceSize, numSlices, kth);
}

/// Argpartition numSlices contiguous i32 slices at kth position. Indices as f64.
export fn argpartition_slices_i32(a: [*]const i32, out: [*]f64, sliceSize: u32, numSlices: u32, kth: u32) void {
    argpartitionSlicesToF64(i32, a, out, sliceSize, numSlices, kth);
}

/// Argpartition numSlices contiguous u32 slices at kth position. Indices as f64.
export fn argpartition_slices_u32(a: [*]const u32, out: [*]f64, sliceSize: u32, numSlices: u32, kth: u32) void {
    argpartitionSlicesToF64(u32, a, out, sliceSize, numSlices, kth);
}

/// Argpartition numSlices contiguous i16 slices at kth position. Indices as f64.
export fn argpartition_slices_i16(a: [*]const i16, out: [*]f64, sliceSize: u32, numSlices: u32, kth: u32) void {
    argpartitionSlicesToF64(i16, a, out, sliceSize, numSlices, kth);
}

/// Argpartition numSlices contiguous u16 slices at kth position. Indices as f64.
export fn argpartition_slices_u16(a: [*]const u16, out: [*]f64, sliceSize: u32, numSlices: u32, kth: u32) void {
    argpartitionSlicesToF64(u16, a, out, sliceSize, numSlices, kth);
}

/// Argpartition numSlices contiguous i8 slices at kth position. Indices as f64.
export fn argpartition_slices_i8(a: [*]const i8, out: [*]f64, sliceSize: u32, numSlices: u32, kth: u32) void {
    argpartitionSlicesToF64(i8, a, out, sliceSize, numSlices, kth);
}

/// Argpartition numSlices contiguous u8 slices at kth position. Indices as f64.
export fn argpartition_slices_u8(a: [*]const u8, out: [*]f64, sliceSize: u32, numSlices: u32, kth: u32) void {
    argpartitionSlicesToF64(u8, a, out, sliceSize, numSlices, kth);
}

// --- Tests ---

fn outIdx(out: []const f64, i: usize) usize {
    return @intFromFloat(out[i]);
}

test "argpartition_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 3.0, 1.0, 4.0, 1.5, 2.0 };
    var out: [5]f64 = undefined;
    argpartition_f64(&a, &out, 5, 2);
    try testing.expectEqual(a[outIdx(&out, 2)], 2.0);
    for (0..2) |i| try testing.expect(a[outIdx(&out, i)] <= a[outIdx(&out, 2)]);
    for (3..5) |i| try testing.expect(a[outIdx(&out, i)] >= a[outIdx(&out, 2)]);
}

test "argpartition_f64 kth=0" {
    const testing = @import("std").testing;
    const a = [_]f64{ 5.0, 3.0, 1.0, 4.0, 2.0 };
    var out: [5]f64 = undefined;
    argpartition_f64(&a, &out, 5, 0);
    try testing.expectEqual(a[outIdx(&out, 0)], 1.0);
    for (1..5) |i| try testing.expect(a[outIdx(&out, i)] >= a[outIdx(&out, 0)]);
}

test "argpartition_f64 kth=N-1" {
    const testing = @import("std").testing;
    const a = [_]f64{ 5.0, 3.0, 1.0, 4.0, 2.0 };
    var out: [5]f64 = undefined;
    argpartition_f64(&a, &out, 5, 4);
    try testing.expectEqual(a[outIdx(&out, 4)], 5.0);
    for (0..4) |i| try testing.expect(a[outIdx(&out, i)] <= a[outIdx(&out, 4)]);
}

test "argpartition_f64 NaN handling" {
    const testing = @import("std").testing;
    const nan = @as(f64, @bitCast(@as(u64, 0x7FF8000000000000)));
    const a = [_]f64{ nan, 2.0, 1.0, nan, 3.0 };
    var out: [5]f64 = undefined;
    argpartition_f64(&a, &out, 5, 2);
    try testing.expectEqual(a[outIdx(&out, 2)], 3.0);
    for (0..2) |i| {
        try testing.expect(a[outIdx(&out, i)] <= 3.0);
        try testing.expect(a[outIdx(&out, i)] == a[outIdx(&out, i)]);
    }
    for (3..5) |i| try testing.expect(a[outIdx(&out, i)] != a[outIdx(&out, i)]);
}

test "argpartition_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 10, -5, 3, 7, -2 };
    var out: [5]f64 = undefined;
    argpartition_i32(&a, &out, 5, 2);
    try testing.expectEqual(a[outIdx(&out, 2)], 3);
    for (0..2) |i| try testing.expect(a[outIdx(&out, i)] <= a[outIdx(&out, 2)]);
    for (3..5) |i| try testing.expect(a[outIdx(&out, i)] >= a[outIdx(&out, 2)]);
}

test "argpartition_u8 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{ 50, 10, 40, 20, 30 };
    var out: [5]f64 = undefined;
    argpartition_u8(&a, &out, 5, 2);
    try testing.expectEqual(a[outIdx(&out, 2)], 30);
    for (0..2) |i| try testing.expect(a[outIdx(&out, i)] <= a[outIdx(&out, 2)]);
    for (3..5) |i| try testing.expect(a[outIdx(&out, i)] >= a[outIdx(&out, 2)]);
}

test "argpartition_i32 already sorted" {
    const testing = @import("std").testing;
    const a = [_]i32{ 1, 2, 3, 4, 5 };
    var out: [5]f64 = undefined;
    argpartition_i32(&a, &out, 5, 2);
    try testing.expectEqual(a[outIdx(&out, 2)], 3);
}

test "argpartition_f64 single element" {
    const testing = @import("std").testing;
    const a = [_]f64{42.0};
    var out: [1]f64 = undefined;
    argpartition_f64(&a, &out, 1, 0);
    try testing.expectEqual(out[0], 0.0);
}

test "argpartition_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 3.0, 1.0, 4.0, 1.5, 2.0 };
    var out: [5]f64 = undefined;
    argpartition_f32(&a, &out, 5, 2);
    try testing.expectApproxEqAbs(a[outIdx(&out, 2)], 2.0, 1e-5);
    for (0..2) |i| try testing.expect(a[outIdx(&out, i)] <= a[outIdx(&out, 2)]);
    for (3..5) |i| try testing.expect(a[outIdx(&out, i)] >= a[outIdx(&out, 2)]);
}

test "argpartition_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 50, 10, 40, 20, 30 };
    var out: [5]f64 = undefined;
    argpartition_i64(&a, &out, 5, 2);
    try testing.expectEqual(a[outIdx(&out, 2)], 30);
    for (0..2) |i| try testing.expect(a[outIdx(&out, i)] <= a[outIdx(&out, 2)]);
    for (3..5) |i| try testing.expect(a[outIdx(&out, i)] >= a[outIdx(&out, 2)]);
}

test "argpartition_u64 basic" {
    const testing = @import("std").testing;
    const a = [_]u64{ 50, 10, 40, 20, 30 };
    var out: [5]f64 = undefined;
    argpartition_u64(&a, &out, 5, 2);
    try testing.expectEqual(a[outIdx(&out, 2)], 30);
    for (0..2) |i| try testing.expect(a[outIdx(&out, i)] <= a[outIdx(&out, 2)]);
    for (3..5) |i| try testing.expect(a[outIdx(&out, i)] >= a[outIdx(&out, 2)]);
}

test "argpartition_u32 basic" {
    const testing = @import("std").testing;
    const a = [_]u32{ 50, 10, 40, 20, 30 };
    var out: [5]f64 = undefined;
    argpartition_u32(&a, &out, 5, 2);
    try testing.expectEqual(a[outIdx(&out, 2)], 30);
    for (0..2) |i| try testing.expect(a[outIdx(&out, i)] <= a[outIdx(&out, 2)]);
    for (3..5) |i| try testing.expect(a[outIdx(&out, i)] >= a[outIdx(&out, 2)]);
}

test "argpartition_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{ 50, 10, 40, 20, 30 };
    var out: [5]f64 = undefined;
    argpartition_i16(&a, &out, 5, 2);
    try testing.expectEqual(a[outIdx(&out, 2)], 30);
    for (0..2) |i| try testing.expect(a[outIdx(&out, i)] <= a[outIdx(&out, 2)]);
    for (3..5) |i| try testing.expect(a[outIdx(&out, i)] >= a[outIdx(&out, 2)]);
}

test "argpartition_u16 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{ 500, 100, 400, 200, 300 };
    var out: [5]f64 = undefined;
    argpartition_u16(&a, &out, 5, 2);
    try testing.expectEqual(a[outIdx(&out, 2)], 300);
    for (0..2) |i| try testing.expect(a[outIdx(&out, i)] <= a[outIdx(&out, 2)]);
    for (3..5) |i| try testing.expect(a[outIdx(&out, i)] >= a[outIdx(&out, 2)]);
}

test "argpartition_i8 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{ 50, 10, 40, 20, 30 };
    var out: [5]f64 = undefined;
    argpartition_i8(&a, &out, 5, 2);
    try testing.expectEqual(a[outIdx(&out, 2)], 30);
    for (0..2) |i| try testing.expect(a[outIdx(&out, i)] <= a[outIdx(&out, 2)]);
    for (3..5) |i| try testing.expect(a[outIdx(&out, i)] >= a[outIdx(&out, 2)]);
}

test "argpartition_slices_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 3.0, 1.0, 2.0, 9.0, 7.0, 8.0 };
    var out: [6]f64 = undefined;
    argpartition_slices_f64(&a, &out, 3, 2, 1);
    // out indices are local (0-based within each slice)
    try testing.expectApproxEqAbs(a[0 + outIdx(&out, 1)], 2.0, 1e-10); // kth=1 of [3,1,2]
    try testing.expectApproxEqAbs(a[3 + outIdx(&out, 4)], 8.0, 1e-10); // kth=1 of [9,7,8]
}

test "argpartition_slices_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 3.0, 1.0, 2.0 };
    var out: [3]f64 = undefined;
    argpartition_slices_f32(&a, &out, 3, 1, 1);
    try testing.expectApproxEqAbs(a[outIdx(&out, 1)], 2.0, 1e-5);
}

test "argpartition_slices_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 30, 10, 20 };
    var out: [3]f64 = undefined;
    argpartition_slices_i64(&a, &out, 3, 1, 1);
    try testing.expectEqual(a[outIdx(&out, 1)], 20);
}

test "argpartition_slices_u64 basic" {
    const testing = @import("std").testing;
    const a = [_]u64{ 30, 10, 20 };
    var out: [3]f64 = undefined;
    argpartition_slices_u64(&a, &out, 3, 1, 1);
    try testing.expectEqual(a[outIdx(&out, 1)], 20);
}

test "argpartition_slices_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 30, 10, 20 };
    var out: [3]f64 = undefined;
    argpartition_slices_i32(&a, &out, 3, 1, 1);
    try testing.expectEqual(a[outIdx(&out, 1)], 20);
}

test "argpartition_slices_u32 basic" {
    const testing = @import("std").testing;
    const a = [_]u32{ 30, 10, 20 };
    var out: [3]f64 = undefined;
    argpartition_slices_u32(&a, &out, 3, 1, 1);
    try testing.expectEqual(a[outIdx(&out, 1)], 20);
}

test "argpartition_slices_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{ 30, 10, 20 };
    var out: [3]f64 = undefined;
    argpartition_slices_i16(&a, &out, 3, 1, 1);
    try testing.expectEqual(a[outIdx(&out, 1)], 20);
}

test "argpartition_slices_u16 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{ 300, 100, 200 };
    var out: [3]f64 = undefined;
    argpartition_slices_u16(&a, &out, 3, 1, 1);
    try testing.expectEqual(a[outIdx(&out, 1)], 200);
}

test "argpartition_slices_i8 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{ 30, 10, 20 };
    var out: [3]f64 = undefined;
    argpartition_slices_i8(&a, &out, 3, 1, 1);
    try testing.expectEqual(a[outIdx(&out, 1)], 20);
}

test "argpartition_slices_u8 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{ 200, 50, 100 };
    var out: [3]f64 = undefined;
    argpartition_slices_u8(&a, &out, 3, 1, 1);
    try testing.expectEqual(a[outIdx(&out, 1)], 100);
}
