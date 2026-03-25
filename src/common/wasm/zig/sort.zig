//! WASM in-place sort kernels for all numeric types.
//!
//! Sorts a contiguous 1D buffer in ascending order using introsort
//! (quicksort + heapsort fallback + insertion sort base case).
//! O(N log N) worst-case, in-place, no allocation.
//!
//! Supported dtypes: f64, f32, i64, u64, i32, u32, i16, u16, i8, u8.
//! Floating-point variants treat NaN as greater than all values.

const sc = @import("sorting_common.zig");

/// Sort f64 array in-place. NaN values sort to end.
export fn sort_f64(a: [*]f64, N: u32) void {
    sc.introSort(f64, a, N);
}

/// Sort f32 array in-place. NaN values sort to end.
export fn sort_f32(a: [*]f32, N: u32) void {
    sc.introSort(f32, a, N);
}

/// Sort i64 array in-place.
export fn sort_i64(a: [*]i64, N: u32) void {
    sc.introSort(i64, a, N);
}

/// Sort u64 array in-place.
export fn sort_u64(a: [*]u64, N: u32) void {
    sc.introSort(u64, a, N);
}

/// Sort i32 array in-place. Uses radix sort for N ≥ 16 (≤ 4096), introsort otherwise.
export fn sort_i32(a: [*]i32, N: u32) void {
    if (N >= 256 and N <= 4096) {
        var scratch: [4096]i32 = undefined;
        sc.radixSort(i32, a, N, &scratch);
    } else sc.introSort(i32, a, N);
}

/// Sort u32 array in-place. Uses radix sort for N ≥ 16 (≤ 4096), introsort otherwise.
export fn sort_u32(a: [*]u32, N: u32) void {
    if (N >= 256 and N <= 4096) {
        var scratch: [4096]u32 = undefined;
        sc.radixSort(u32, a, N, &scratch);
    } else sc.introSort(u32, a, N);
}

/// Sort i16 array in-place. Uses radix sort for N ≥ 16 (≤ 4096), introsort otherwise.
export fn sort_i16(a: [*]i16, N: u32) void {
    if (N >= 256 and N <= 4096) {
        var scratch: [4096]i16 = undefined;
        sc.radixSort(i16, a, N, &scratch);
    } else sc.introSort(i16, a, N);
}

/// Sort u16 array in-place. Uses radix sort for N ≥ 16 (≤ 4096), introsort otherwise.
export fn sort_u16(a: [*]u16, N: u32) void {
    if (N >= 256 and N <= 4096) {
        var scratch: [4096]u16 = undefined;
        sc.radixSort(u16, a, N, &scratch);
    } else sc.introSort(u16, a, N);
}

/// Sort i8 array in-place. Uses radix sort for N ≥ 16 (≤ 4096), introsort otherwise.
export fn sort_i8(a: [*]i8, N: u32) void {
    if (N >= 256 and N <= 4096) {
        var scratch: [4096]i8 = undefined;
        sc.radixSort(i8, a, N, &scratch);
    } else sc.introSort(i8, a, N);
}

/// Sort u8 array in-place. Uses radix sort for N ≥ 16 (≤ 4096), introsort otherwise.
export fn sort_u8(a: [*]u8, N: u32) void {
    if (N >= 256 and N <= 4096) {
        var scratch: [4096]u8 = undefined;
        sc.radixSort(u8, a, N, &scratch);
    } else sc.introSort(u8, a, N);
}

// --- Batch slice sort (single WASM call for multi-dim arrays) ---

/// Sort numSlices contiguous f64 slices of sliceSize elements each.
export fn sort_slices_f64(a: [*]f64, sliceSize: u32, numSlices: u32) void {
    sc.sortSlices(f64, a, sliceSize, numSlices);
}

/// Sort numSlices contiguous f32 slices of sliceSize elements each.
export fn sort_slices_f32(a: [*]f32, sliceSize: u32, numSlices: u32) void {
    sc.sortSlices(f32, a, sliceSize, numSlices);
}

/// Sort numSlices contiguous i64 slices of sliceSize elements each.
export fn sort_slices_i64(a: [*]i64, sliceSize: u32, numSlices: u32) void {
    sc.sortSlices(i64, a, sliceSize, numSlices);
}

/// Sort numSlices contiguous u64 slices of sliceSize elements each.
export fn sort_slices_u64(a: [*]u64, sliceSize: u32, numSlices: u32) void {
    sc.sortSlices(u64, a, sliceSize, numSlices);
}

/// Sort numSlices contiguous i32 slices of sliceSize elements each. Uses radix sort.
export fn sort_slices_i32(a: [*]i32, sliceSize: u32, numSlices: u32) void {
    sc.sortSlices(i32, a, sliceSize, numSlices);
}

/// Sort numSlices contiguous u32 slices of sliceSize elements each. Uses radix sort.
export fn sort_slices_u32(a: [*]u32, sliceSize: u32, numSlices: u32) void {
    sc.sortSlices(u32, a, sliceSize, numSlices);
}

/// Sort numSlices contiguous i16 slices of sliceSize elements each. Uses radix sort.
export fn sort_slices_i16(a: [*]i16, sliceSize: u32, numSlices: u32) void {
    sc.sortSlices(i16, a, sliceSize, numSlices);
}

/// Sort numSlices contiguous u16 slices of sliceSize elements each. Uses radix sort.
export fn sort_slices_u16(a: [*]u16, sliceSize: u32, numSlices: u32) void {
    sc.sortSlices(u16, a, sliceSize, numSlices);
}

/// Sort numSlices contiguous i8 slices of sliceSize elements each. Uses radix sort.
export fn sort_slices_i8(a: [*]i8, sliceSize: u32, numSlices: u32) void {
    sc.sortSlices(i8, a, sliceSize, numSlices);
}

/// Sort numSlices contiguous u8 slices of sliceSize elements each. Uses radix sort.
export fn sort_slices_u8(a: [*]u8, sliceSize: u32, numSlices: u32) void {
    sc.sortSlices(u8, a, sliceSize, numSlices);
}

// --- Float16 sort (operates on raw u16 bits with IEEE-754 bit-flip) ---

/// Map float16 bits to a sortable u16: positive floats map to upper half,
/// negative floats map to lower half (inverted), preserving total order.
/// NaN (exponent=0x1F, mantissa!=0) maps to highest values → sorts to end.
inline fn f16ToSortKey(bits: u16) u16 {
    return if (bits >> 15 != 0) ~bits else bits ^ (@as(u16, 1) << 15);
}

inline fn sortKeyToF16(key: u16) u16 {
    return if (key >> 15 != 0) key ^ (@as(u16, 1) << 15) else ~key;
}

fn f16FlipInPlace(a: [*]u16, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) a[i] = f16ToSortKey(a[i]);
}

fn f16UnflipInPlace(a: [*]u16, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) a[i] = sortKeyToF16(a[i]);
}

/// Sort float16 array in-place (data is raw u16 bits). NaN sorts to end.
export fn sort_f16(a: [*]u16, N: u32) void {
    f16FlipInPlace(a, N);
    if (N >= 256 and N <= 4096) {
        var scratch: [4096]u16 = undefined;
        sc.radixSort(u16, a, N, &scratch);
    } else sc.introSort(u16, a, N);
    f16UnflipInPlace(a, N);
}

/// Sort numSlices contiguous float16 slices (data is raw u16 bits).
export fn sort_slices_f16(a: [*]u16, sliceSize: u32, numSlices: u32) void {
    const total = @as(usize, sliceSize) * @as(usize, numSlices);
    f16FlipInPlace(a, @intCast(total));
    sc.sortSlices(u16, a, sliceSize, numSlices);
    f16UnflipInPlace(a, @intCast(total));
}

// --- Complex sort ---

/// Sort complex128 (interleaved f64 pairs) array in-place. N = number of complex elements.
export fn sort_c128(a: [*]f64, N: u32) void {
    sc.complexIntroSort(f64, a, N);
}

/// Sort complex64 (interleaved f32 pairs) array in-place.
export fn sort_c64(a: [*]f32, N: u32) void {
    sc.complexIntroSort(f32, a, N);
}

/// Batch complex128 slice sort.
export fn sort_slices_c128(a: [*]f64, sliceSize: u32, numSlices: u32) void {
    sc.complexSortSlices(f64, a, sliceSize, numSlices);
}

/// Batch complex64 slice sort.
export fn sort_slices_c64(a: [*]f32, sliceSize: u32, numSlices: u32) void {
    sc.complexSortSlices(f32, a, sliceSize, numSlices);
}

// --- Tests ---

test "sort_f64 basic" {
    const testing = @import("std").testing;
    var a = [_]f64{ 3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0 };
    sort_f64(&a, 8);
    try testing.expectApproxEqAbs(a[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(a[1], 1.0, 1e-10);
    try testing.expectApproxEqAbs(a[7], 9.0, 1e-10);
}

test "sort_f64 NaN to end" {
    const testing = @import("std").testing;
    const nan = @as(f64, @bitCast(@as(u64, 0x7FF8000000000000)));
    var a = [_]f64{ nan, 2.0, 1.0, nan, 3.0 };
    sort_f64(&a, 5);
    try testing.expectApproxEqAbs(a[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(a[1], 2.0, 1e-10);
    try testing.expectApproxEqAbs(a[2], 3.0, 1e-10);
    try testing.expect(a[3] != a[3]);
    try testing.expect(a[4] != a[4]);
}

test "sort_i32 basic" {
    const testing = @import("std").testing;
    var a = [_]i32{ 5, 3, -1, 4, 2 };
    sort_i32(&a, 5);
    try testing.expectEqual(a[0], -1);
    try testing.expectEqual(a[4], 5);
}

test "sort_u8 basic" {
    const testing = @import("std").testing;
    var a = [_]u8{ 200, 50, 100, 10, 255 };
    sort_u8(&a, 5);
    try testing.expectEqual(a[0], 10);
    try testing.expectEqual(a[4], 255);
}

test "sort_f64 already sorted" {
    const testing = @import("std").testing;
    var a = [_]f64{ 1.0, 2.0, 3.0 };
    sort_f64(&a, 3);
    try testing.expectApproxEqAbs(a[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(a[2], 3.0, 1e-10);
}

test "sort_i32 reverse sorted" {
    const testing = @import("std").testing;
    var a = [_]i32{ 5, 4, 3, 2, 1 };
    sort_i32(&a, 5);
    try testing.expectEqual(a[0], 1);
    try testing.expectEqual(a[4], 5);
}

test "sort_f64 single element" {
    const testing = @import("std").testing;
    var a = [_]f64{42.0};
    sort_f64(&a, 1);
    try testing.expectApproxEqAbs(a[0], 42.0, 1e-10);
}

test "sort_i32 empty" {
    var a = [_]i32{};
    sort_i32(&a, 0);
}

test "sort_i16 radix sort path" {
    const testing = @import("std").testing;
    // 20 elements to trigger radix sort (threshold=16)
    var a = [_]i16{ 100, -50, 32, -128, 0, 500, -1, 7, 32, 100, -50, 200, 3, -200, 50, 1, -1, 0, 127, -128 };
    sort_i16(&a, 20);
    try testing.expectEqual(a[0], -200);
    try testing.expectEqual(a[1], -128);
    try testing.expectEqual(a[2], -128);
    try testing.expectEqual(a[19], 500);
    // Verify sorted
    var i: usize = 1;
    while (i < 20) : (i += 1) {
        try testing.expect(a[i] >= a[i - 1]);
    }
}

test "sort_u8 radix sort path" {
    const testing = @import("std").testing;
    var a = [_]u8{ 255, 0, 128, 64, 32, 16, 8, 4, 2, 1, 200, 150, 100, 50, 25, 12, 6, 3, 0, 255 };
    sort_u8(&a, 20);
    try testing.expectEqual(a[0], 0);
    try testing.expectEqual(a[1], 0);
    try testing.expectEqual(a[18], 255);
    try testing.expectEqual(a[19], 255);
    var i: usize = 1;
    while (i < 20) : (i += 1) {
        try testing.expect(a[i] >= a[i - 1]);
    }
}

test "sort_f32 radix sort path" {
    const testing = @import("std").testing;
    var a = [_]f32{ 3.14, -2.71, 0.0, 1.0, -1.0, 100.5, -0.5, 42.0, 0.001, -999.0, 5.0, 2.0, -3.0, 7.0, 8.0, -8.0, 0.0, 1.5, -1.5, 99.9 };
    sort_f32(&a, 20);
    try testing.expectApproxEqAbs(a[0], -999.0, 1e-5);
    try testing.expectApproxEqAbs(a[19], 100.5, 1e-5);
    var i: usize = 1;
    while (i < 20) : (i += 1) {
        try testing.expect(a[i] >= a[i - 1]);
    }
}

test "sort_i32 radix sort path" {
    const testing = @import("std").testing;
    var a = [_]i32{ 1000, -500, 0, 2147483647, -2147483648, 42, -42, 100, -100, 7, 3, -3, 999, -999, 50, -50, 1, -1, 0, 0 };
    sort_i32(&a, 20);
    try testing.expectEqual(a[0], -2147483648);
    try testing.expectEqual(a[19], 2147483647);
    var i: usize = 1;
    while (i < 20) : (i += 1) {
        try testing.expect(a[i] >= a[i - 1]);
    }
}
