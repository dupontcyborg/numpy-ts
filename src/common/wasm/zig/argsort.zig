//! WASM argsort kernels for all numeric types.
//!
//! Produces an index array such that out[0] is the index of the smallest
//! element, out[1] the next smallest, etc.  Stable: equal elements preserve
//! original index order (smaller index first).
//!
//! Supported dtypes: f64, f32, i64, u64, i32, u32, i16, u16, i8, u8.
//! Floating-point variants treat NaN as greater than all values.

const sc = @import("sorting_common.zig");

/// Argsort for f64: writes ascending-order indices into out. NaN sorts to end.
export fn argsort_f64(a: [*]const f64, out: [*]u32, N: u32) void {
    sc.initIndices(out, N);
    sc.introSortStable(f64, a, out, N);
}

/// Argsort for f32: writes ascending-order indices into out. NaN sorts to end.
export fn argsort_f32(a: [*]const f32, out: [*]u32, N: u32) void {
    sc.initIndices(out, N);
    sc.introSortStable(f32, a, out, N);
}

/// Argsort for i64.
export fn argsort_i64(a: [*]const i64, out: [*]u32, N: u32) void {
    sc.initIndices(out, N);
    sc.introSortStable(i64, a, out, N);
}

/// Argsort for u64.
export fn argsort_u64(a: [*]const u64, out: [*]u32, N: u32) void {
    sc.initIndices(out, N);
    sc.introSortStable(u64, a, out, N);
}

/// Argsort for i32.
export fn argsort_i32(a: [*]const i32, out: [*]u32, N: u32) void {
    sc.initIndices(out, N);
    sc.introSortStable(i32, a, out, N);
}

/// Argsort for u32.
export fn argsort_u32(a: [*]const u32, out: [*]u32, N: u32) void {
    sc.initIndices(out, N);
    sc.introSortStable(u32, a, out, N);
}

/// Argsort for i16.
export fn argsort_i16(a: [*]const i16, out: [*]u32, N: u32) void {
    sc.initIndices(out, N);
    sc.introSortStable(i16, a, out, N);
}

/// Argsort for u16.
export fn argsort_u16(a: [*]const u16, out: [*]u32, N: u32) void {
    sc.initIndices(out, N);
    sc.introSortStable(u16, a, out, N);
}

/// Argsort for i8.
export fn argsort_i8(a: [*]const i8, out: [*]u32, N: u32) void {
    sc.initIndices(out, N);
    sc.introSortStable(i8, a, out, N);
}

/// Argsort for u8.
export fn argsort_u8(a: [*]const u8, out: [*]u32, N: u32) void {
    sc.initIndices(out, N);
    sc.introSortStable(u8, a, out, N);
}

// --- Batch slice argsort ---

/// Stable argsort of numSlices contiguous f64 slices.
export fn argsort_slices_f64(a: [*]const f64, out: [*]u32, sliceSize: u32, numSlices: u32) void {
    sc.argsortSlices(f64, a, out, sliceSize, numSlices);
}

/// Stable argsort of numSlices contiguous f32 slices.
export fn argsort_slices_f32(a: [*]const f32, out: [*]u32, sliceSize: u32, numSlices: u32) void {
    sc.argsortSlices(f32, a, out, sliceSize, numSlices);
}

/// Stable argsort of numSlices contiguous i64 slices.
export fn argsort_slices_i64(a: [*]const i64, out: [*]u32, sliceSize: u32, numSlices: u32) void {
    sc.argsortSlices(i64, a, out, sliceSize, numSlices);
}

/// Stable argsort of numSlices contiguous u64 slices.
export fn argsort_slices_u64(a: [*]const u64, out: [*]u32, sliceSize: u32, numSlices: u32) void {
    sc.argsortSlices(u64, a, out, sliceSize, numSlices);
}

/// Stable argsort of numSlices contiguous i32 slices.
export fn argsort_slices_i32(a: [*]const i32, out: [*]u32, sliceSize: u32, numSlices: u32) void {
    sc.argsortSlices(i32, a, out, sliceSize, numSlices);
}

/// Stable argsort of numSlices contiguous u32 slices.
export fn argsort_slices_u32(a: [*]const u32, out: [*]u32, sliceSize: u32, numSlices: u32) void {
    sc.argsortSlices(u32, a, out, sliceSize, numSlices);
}

/// Stable argsort of numSlices contiguous i16 slices.
export fn argsort_slices_i16(a: [*]const i16, out: [*]u32, sliceSize: u32, numSlices: u32) void {
    sc.argsortSlices(i16, a, out, sliceSize, numSlices);
}

/// Stable argsort of numSlices contiguous u16 slices.
export fn argsort_slices_u16(a: [*]const u16, out: [*]u32, sliceSize: u32, numSlices: u32) void {
    sc.argsortSlices(u16, a, out, sliceSize, numSlices);
}

/// Stable argsort of numSlices contiguous i8 slices.
export fn argsort_slices_i8(a: [*]const i8, out: [*]u32, sliceSize: u32, numSlices: u32) void {
    sc.argsortSlices(i8, a, out, sliceSize, numSlices);
}

/// Stable argsort of numSlices contiguous u8 slices.
export fn argsort_slices_u8(a: [*]const u8, out: [*]u32, sliceSize: u32, numSlices: u32) void {
    sc.argsortSlices(u8, a, out, sliceSize, numSlices);
}

// --- Complex argsort ---

/// Stable argsort for complex128 (interleaved f64 pairs). NaN sorts to end.
export fn argsort_c128(a: [*]const f64, out: [*]u32, N: u32) void {
    sc.initIndices(out, N);
    sc.complexIntroSortStable(f64, a, out, N);
}

/// Stable argsort for complex64 (interleaved f32 pairs). NaN sorts to end.
export fn argsort_c64(a: [*]const f32, out: [*]u32, N: u32) void {
    sc.initIndices(out, N);
    sc.complexIntroSortStable(f32, a, out, N);
}

/// Stable argsort of numSlices contiguous complex128 slices.
export fn argsort_slices_c128(a: [*]const f64, out: [*]u32, sliceSize: u32, numSlices: u32) void {
    sc.complexArgsortSlices(f64, a, out, sliceSize, numSlices);
}

/// Stable argsort of numSlices contiguous complex64 slices.
export fn argsort_slices_c64(a: [*]const f32, out: [*]u32, sliceSize: u32, numSlices: u32) void {
    sc.complexArgsortSlices(f32, a, out, sliceSize, numSlices);
}

// --- Tests ---

test "argsort_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 3.0, 1.0, 4.0, 1.5, 2.0 };
    var out: [5]u32 = undefined;
    argsort_f64(&a, &out, 5);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 3);
    try testing.expectEqual(out[2], 4);
    try testing.expectEqual(out[3], 0);
    try testing.expectEqual(out[4], 2);
}

test "argsort_f64 duplicates stable" {
    const testing = @import("std").testing;
    const a = [_]f64{ 3.0, 1.0, 3.0, 1.0, 2.0 };
    var out: [5]u32 = undefined;
    argsort_f64(&a, &out, 5);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 3);
    try testing.expectEqual(out[2], 4);
    try testing.expectEqual(out[3], 0);
    try testing.expectEqual(out[4], 2);
}

test "argsort_f64 NaN handling stable" {
    const testing = @import("std").testing;
    const nan = @as(f64, @bitCast(@as(u64, 0x7FF8000000000000)));
    const a = [_]f64{ nan, 2.0, 1.0, nan, 3.0 };
    var out: [5]u32 = undefined;
    argsort_f64(&a, &out, 5);
    try testing.expectEqual(out[0], 2);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 4);
    try testing.expectEqual(out[3], 0);
    try testing.expectEqual(out[4], 3);
}

test "argsort_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 30, -10, 20, 0 };
    var out: [4]u32 = undefined;
    argsort_i32(&a, &out, 4);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 3);
    try testing.expectEqual(out[2], 2);
    try testing.expectEqual(out[3], 0);
}

test "argsort_u8 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{ 200, 50, 100, 10 };
    var out: [4]u32 = undefined;
    argsort_u8(&a, &out, 4);
    try testing.expectEqual(out[0], 3);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 2);
    try testing.expectEqual(out[3], 0);
}

test "argsort_f64 single element" {
    const testing = @import("std").testing;
    const a = [_]f64{42.0};
    var out: [1]u32 = undefined;
    argsort_f64(&a, &out, 1);
    try testing.expectEqual(out[0], 0);
}
