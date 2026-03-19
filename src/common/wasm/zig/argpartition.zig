//! WASM argpartition kernels for all numeric types.
//!
//! Argpartition: fills out[0..N] with indices such that out[kth] is the index
//! of the element that would be at position kth in a sorted array, all indices
//! before kth reference values <= a[out[kth]], and all after reference values
//! >= a[out[kth]].
//!
//! Uses quickselect with median-of-three pivot selection.
//! Supported dtypes: f64, f32, i64, u64, i32, u32, i16, u16, i8, u8.
//! Floating-point types treat NaN as greater than all values.

const sc = @import("sorting_common.zig");

/// Argpartition for f64. NaN values are treated as greater than all values.
export fn argpartition_f64(a: [*]const f64, out: [*]u32, N: u32, kth: u32) void {
    sc.initIndices(out, N);
    if (N <= 1 or kth >= N) return;
    sc.quickselectIndirect(f64, a, out, 0, N - 1, kth);
}

/// Argpartition for f32. NaN values are treated as greater than all values.
export fn argpartition_f32(a: [*]const f32, out: [*]u32, N: u32, kth: u32) void {
    sc.initIndices(out, N);
    if (N <= 1 or kth >= N) return;
    sc.quickselectIndirect(f32, a, out, 0, N - 1, kth);
}

/// Argpartition for i64.
export fn argpartition_i64(a: [*]const i64, out: [*]u32, N: u32, kth: u32) void {
    sc.initIndices(out, N);
    if (N <= 1 or kth >= N) return;
    sc.quickselectIndirect(i64, a, out, 0, N - 1, kth);
}

/// Argpartition for u64.
export fn argpartition_u64(a: [*]const u64, out: [*]u32, N: u32, kth: u32) void {
    sc.initIndices(out, N);
    if (N <= 1 or kth >= N) return;
    sc.quickselectIndirect(u64, a, out, 0, N - 1, kth);
}

/// Argpartition for i32.
export fn argpartition_i32(a: [*]const i32, out: [*]u32, N: u32, kth: u32) void {
    sc.initIndices(out, N);
    if (N <= 1 or kth >= N) return;
    sc.quickselectIndirect(i32, a, out, 0, N - 1, kth);
}

/// Argpartition for u32.
export fn argpartition_u32(a: [*]const u32, out: [*]u32, N: u32, kth: u32) void {
    sc.initIndices(out, N);
    if (N <= 1 or kth >= N) return;
    sc.quickselectIndirect(u32, a, out, 0, N - 1, kth);
}

/// Argpartition for i16.
export fn argpartition_i16(a: [*]const i16, out: [*]u32, N: u32, kth: u32) void {
    sc.initIndices(out, N);
    if (N <= 1 or kth >= N) return;
    sc.quickselectIndirect(i16, a, out, 0, N - 1, kth);
}

/// Argpartition for u16.
export fn argpartition_u16(a: [*]const u16, out: [*]u32, N: u32, kth: u32) void {
    sc.initIndices(out, N);
    if (N <= 1 or kth >= N) return;
    sc.quickselectIndirect(u16, a, out, 0, N - 1, kth);
}

/// Argpartition for i8.
export fn argpartition_i8(a: [*]const i8, out: [*]u32, N: u32, kth: u32) void {
    sc.initIndices(out, N);
    if (N <= 1 or kth >= N) return;
    sc.quickselectIndirect(i8, a, out, 0, N - 1, kth);
}

/// Argpartition for u8.
export fn argpartition_u8(a: [*]const u8, out: [*]u32, N: u32, kth: u32) void {
    sc.initIndices(out, N);
    if (N <= 1 or kth >= N) return;
    sc.quickselectIndirect(u8, a, out, 0, N - 1, kth);
}

// --- Batch slice argpartition ---

/// Argpartition numSlices contiguous f64 slices at kth position.
export fn argpartition_slices_f64(a: [*]const f64, out: [*]u32, sliceSize: u32, numSlices: u32, kth: u32) void {
    sc.quickselectIndirectSlices(f64, a, out, sliceSize, numSlices, kth);
}

/// Argpartition numSlices contiguous f32 slices at kth position.
export fn argpartition_slices_f32(a: [*]const f32, out: [*]u32, sliceSize: u32, numSlices: u32, kth: u32) void {
    sc.quickselectIndirectSlices(f32, a, out, sliceSize, numSlices, kth);
}

/// Argpartition numSlices contiguous i64 slices at kth position.
export fn argpartition_slices_i64(a: [*]const i64, out: [*]u32, sliceSize: u32, numSlices: u32, kth: u32) void {
    sc.quickselectIndirectSlices(i64, a, out, sliceSize, numSlices, kth);
}

/// Argpartition numSlices contiguous u64 slices at kth position.
export fn argpartition_slices_u64(a: [*]const u64, out: [*]u32, sliceSize: u32, numSlices: u32, kth: u32) void {
    sc.quickselectIndirectSlices(u64, a, out, sliceSize, numSlices, kth);
}

/// Argpartition numSlices contiguous i32 slices at kth position.
export fn argpartition_slices_i32(a: [*]const i32, out: [*]u32, sliceSize: u32, numSlices: u32, kth: u32) void {
    sc.quickselectIndirectSlices(i32, a, out, sliceSize, numSlices, kth);
}

/// Argpartition numSlices contiguous u32 slices at kth position.
export fn argpartition_slices_u32(a: [*]const u32, out: [*]u32, sliceSize: u32, numSlices: u32, kth: u32) void {
    sc.quickselectIndirectSlices(u32, a, out, sliceSize, numSlices, kth);
}

/// Argpartition numSlices contiguous i16 slices at kth position.
export fn argpartition_slices_i16(a: [*]const i16, out: [*]u32, sliceSize: u32, numSlices: u32, kth: u32) void {
    sc.quickselectIndirectSlices(i16, a, out, sliceSize, numSlices, kth);
}

/// Argpartition numSlices contiguous u16 slices at kth position.
export fn argpartition_slices_u16(a: [*]const u16, out: [*]u32, sliceSize: u32, numSlices: u32, kth: u32) void {
    sc.quickselectIndirectSlices(u16, a, out, sliceSize, numSlices, kth);
}

/// Argpartition numSlices contiguous i8 slices at kth position.
export fn argpartition_slices_i8(a: [*]const i8, out: [*]u32, sliceSize: u32, numSlices: u32, kth: u32) void {
    sc.quickselectIndirectSlices(i8, a, out, sliceSize, numSlices, kth);
}

/// Argpartition numSlices contiguous u8 slices at kth position.
export fn argpartition_slices_u8(a: [*]const u8, out: [*]u32, sliceSize: u32, numSlices: u32, kth: u32) void {
    sc.quickselectIndirectSlices(u8, a, out, sliceSize, numSlices, kth);
}

// --- Tests ---

test "argpartition_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 3.0, 1.0, 4.0, 1.5, 2.0 };
    var out: [5]u32 = undefined;
    argpartition_f64(&a, &out, 5, 2);
    try testing.expectEqual(a[out[2]], 2.0);
    for (0..2) |i| try testing.expect(a[out[i]] <= a[out[2]]);
    for (3..5) |i| try testing.expect(a[out[i]] >= a[out[2]]);
}

test "argpartition_f64 kth=0" {
    const testing = @import("std").testing;
    const a = [_]f64{ 5.0, 3.0, 1.0, 4.0, 2.0 };
    var out: [5]u32 = undefined;
    argpartition_f64(&a, &out, 5, 0);
    try testing.expectEqual(a[out[0]], 1.0);
    for (1..5) |i| try testing.expect(a[out[i]] >= a[out[0]]);
}

test "argpartition_f64 kth=N-1" {
    const testing = @import("std").testing;
    const a = [_]f64{ 5.0, 3.0, 1.0, 4.0, 2.0 };
    var out: [5]u32 = undefined;
    argpartition_f64(&a, &out, 5, 4);
    try testing.expectEqual(a[out[4]], 5.0);
    for (0..4) |i| try testing.expect(a[out[i]] <= a[out[4]]);
}

test "argpartition_f64 NaN handling" {
    const testing = @import("std").testing;
    const nan = @as(f64, @bitCast(@as(u64, 0x7FF8000000000000)));
    const a = [_]f64{ nan, 2.0, 1.0, nan, 3.0 };
    var out: [5]u32 = undefined;
    argpartition_f64(&a, &out, 5, 2);
    try testing.expectEqual(a[out[2]], 3.0);
    for (0..2) |i| {
        try testing.expect(a[out[i]] <= 3.0);
        try testing.expect(a[out[i]] == a[out[i]]);
    }
    for (3..5) |i| try testing.expect(a[out[i]] != a[out[i]]);
}

test "argpartition_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 10, -5, 3, 7, -2 };
    var out: [5]u32 = undefined;
    argpartition_i32(&a, &out, 5, 2);
    try testing.expectEqual(a[out[2]], 3);
    for (0..2) |i| try testing.expect(a[out[i]] <= a[out[2]]);
    for (3..5) |i| try testing.expect(a[out[i]] >= a[out[2]]);
}

test "argpartition_u8 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{ 50, 10, 40, 20, 30 };
    var out: [5]u32 = undefined;
    argpartition_u8(&a, &out, 5, 2);
    try testing.expectEqual(a[out[2]], 30);
    for (0..2) |i| try testing.expect(a[out[i]] <= a[out[2]]);
    for (3..5) |i| try testing.expect(a[out[i]] >= a[out[2]]);
}

test "argpartition_i32 already sorted" {
    const testing = @import("std").testing;
    const a = [_]i32{ 1, 2, 3, 4, 5 };
    var out: [5]u32 = undefined;
    argpartition_i32(&a, &out, 5, 2);
    try testing.expectEqual(a[out[2]], 3);
}

test "argpartition_f64 single element" {
    const testing = @import("std").testing;
    const a = [_]f64{42.0};
    var out: [1]u32 = undefined;
    argpartition_f64(&a, &out, 1, 0);
    try testing.expectEqual(out[0], 0);
}
