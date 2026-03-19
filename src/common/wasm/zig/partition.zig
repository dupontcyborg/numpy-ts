//! WASM in-place partition (quickselect) kernels for all numeric types.
//!
//! Partitions an array so that a[kth] contains the value that would be
//! in position kth if the array were sorted. All elements before kth
//! are <= a[kth], all elements after are >= a[kth].
//!
//! Uses quickselect with median-of-three pivot selection.
//! Supported dtypes: f64, f32, i64, u64, i32, u32, i16, u16, i8, u8.

const sc = @import("sorting_common.zig");

/// Partition f64 array in-place. NaN values sort to end.
export fn partition_f64(a: [*]f64, N: u32, kth: u32) void {
    if (N <= 1 or kth >= N) return;
    sc.quickselect(f64, a, 0, N - 1, kth);
}

/// Partition f32 array in-place. NaN values sort to end.
export fn partition_f32(a: [*]f32, N: u32, kth: u32) void {
    if (N <= 1 or kth >= N) return;
    sc.quickselect(f32, a, 0, N - 1, kth);
}

/// Partition i64 array in-place.
export fn partition_i64(a: [*]i64, N: u32, kth: u32) void {
    if (N <= 1 or kth >= N) return;
    sc.quickselect(i64, a, 0, N - 1, kth);
}

/// Partition u64 array in-place.
export fn partition_u64(a: [*]u64, N: u32, kth: u32) void {
    if (N <= 1 or kth >= N) return;
    sc.quickselect(u64, a, 0, N - 1, kth);
}

/// Partition i32 array in-place.
export fn partition_i32(a: [*]i32, N: u32, kth: u32) void {
    if (N <= 1 or kth >= N) return;
    sc.quickselect(i32, a, 0, N - 1, kth);
}

/// Partition u32 array in-place.
export fn partition_u32(a: [*]u32, N: u32, kth: u32) void {
    if (N <= 1 or kth >= N) return;
    sc.quickselect(u32, a, 0, N - 1, kth);
}

/// Partition i16 array in-place.
export fn partition_i16(a: [*]i16, N: u32, kth: u32) void {
    if (N <= 1 or kth >= N) return;
    sc.quickselect(i16, a, 0, N - 1, kth);
}

/// Partition u16 array in-place.
export fn partition_u16(a: [*]u16, N: u32, kth: u32) void {
    if (N <= 1 or kth >= N) return;
    sc.quickselect(u16, a, 0, N - 1, kth);
}

/// Partition i8 array in-place.
export fn partition_i8(a: [*]i8, N: u32, kth: u32) void {
    if (N <= 1 or kth >= N) return;
    sc.quickselect(i8, a, 0, N - 1, kth);
}

/// Partition u8 array in-place.
export fn partition_u8(a: [*]u8, N: u32, kth: u32) void {
    if (N <= 1 or kth >= N) return;
    sc.quickselect(u8, a, 0, N - 1, kth);
}

// --- Batch slice partition ---

/// Partition numSlices contiguous f64 slices at kth position.
export fn partition_slices_f64(a: [*]f64, sliceSize: u32, numSlices: u32, kth: u32) void {
    sc.quickselectSlices(f64, a, sliceSize, numSlices, kth);
}

/// Partition numSlices contiguous f32 slices at kth position.
export fn partition_slices_f32(a: [*]f32, sliceSize: u32, numSlices: u32, kth: u32) void {
    sc.quickselectSlices(f32, a, sliceSize, numSlices, kth);
}

/// Partition numSlices contiguous i64 slices at kth position.
export fn partition_slices_i64(a: [*]i64, sliceSize: u32, numSlices: u32, kth: u32) void {
    sc.quickselectSlices(i64, a, sliceSize, numSlices, kth);
}

/// Partition numSlices contiguous u64 slices at kth position.
export fn partition_slices_u64(a: [*]u64, sliceSize: u32, numSlices: u32, kth: u32) void {
    sc.quickselectSlices(u64, a, sliceSize, numSlices, kth);
}

/// Partition numSlices contiguous i32 slices at kth position.
export fn partition_slices_i32(a: [*]i32, sliceSize: u32, numSlices: u32, kth: u32) void {
    sc.quickselectSlices(i32, a, sliceSize, numSlices, kth);
}

/// Partition numSlices contiguous u32 slices at kth position.
export fn partition_slices_u32(a: [*]u32, sliceSize: u32, numSlices: u32, kth: u32) void {
    sc.quickselectSlices(u32, a, sliceSize, numSlices, kth);
}

/// Partition numSlices contiguous i16 slices at kth position.
export fn partition_slices_i16(a: [*]i16, sliceSize: u32, numSlices: u32, kth: u32) void {
    sc.quickselectSlices(i16, a, sliceSize, numSlices, kth);
}

/// Partition numSlices contiguous u16 slices at kth position.
export fn partition_slices_u16(a: [*]u16, sliceSize: u32, numSlices: u32, kth: u32) void {
    sc.quickselectSlices(u16, a, sliceSize, numSlices, kth);
}

/// Partition numSlices contiguous i8 slices at kth position.
export fn partition_slices_i8(a: [*]i8, sliceSize: u32, numSlices: u32, kth: u32) void {
    sc.quickselectSlices(i8, a, sliceSize, numSlices, kth);
}

/// Partition numSlices contiguous u8 slices at kth position.
export fn partition_slices_u8(a: [*]u8, sliceSize: u32, numSlices: u32, kth: u32) void {
    sc.quickselectSlices(u8, a, sliceSize, numSlices, kth);
}

// --- Tests ---

test "partition_f64 basic" {
    const testing = @import("std").testing;
    var a = [_]f64{ 3.0, 1.0, 4.0, 1.5, 2.0 };
    partition_f64(&a, 5, 2);
    try testing.expectApproxEqAbs(a[2], 2.0, 1e-10);
    for (0..2) |i| try testing.expect(a[i] <= a[2]);
    for (3..5) |i| try testing.expect(a[i] >= a[2]);
}

test "partition_f64 NaN" {
    const testing = @import("std").testing;
    const nan = @as(f64, @bitCast(@as(u64, 0x7FF8000000000000)));
    var a = [_]f64{ nan, 2.0, 1.0, nan, 3.0 };
    partition_f64(&a, 5, 2);
    try testing.expectApproxEqAbs(a[2], 3.0, 1e-10);
}

test "partition_i32 basic" {
    const testing = @import("std").testing;
    var a = [_]i32{ 5, 3, -1, 4, 2 };
    partition_i32(&a, 5, 2);
    try testing.expectEqual(a[2], 3);
    for (0..2) |i| try testing.expect(a[i] <= a[2]);
    for (3..5) |i| try testing.expect(a[i] >= a[2]);
}

test "partition_u8 kth=0" {
    const testing = @import("std").testing;
    var a = [_]u8{ 50, 10, 40, 20, 30 };
    partition_u8(&a, 5, 0);
    try testing.expectEqual(a[0], 10);
}

test "partition_f64 single" {
    const testing = @import("std").testing;
    var a = [_]f64{42.0};
    partition_f64(&a, 1, 0);
    try testing.expectApproxEqAbs(a[0], 42.0, 1e-10);
}
