//! WASM argsort kernels for all numeric types.
//!
//! Produces an index array such that out[0] is the index of the smallest
//! element, out[1] the next smallest, etc.  Stable: equal elements preserve
//! original index order (smaller index first).
//!
//! Output is f64 for JS ergonomics. Internally sorts u32 indices in the
//! lower half of the f64 output buffer, then converts in-place via SIMD.
//!
//! Supported dtypes: f64, f32, i64, u64, i32, u32, i16, u16, i8, u8, c128, c64.
//! Floating-point variants treat NaN as greater than all values.

const sc = @import("sorting_common.zig");

/// Helper: sort into u32 indices in the f64 buffer, then convert to f64 in-place.
inline fn argsortToF64(comptime T: type, a: [*]const T, out: [*]f64, N: u32) void {
    const out_u32: [*]u32 = @ptrCast(out);
    sc.initIndices(out_u32, N);
    sc.introSortStable(T, a, out_u32, N);
    sc.indicesToF64(out_u32, out, N);
}

/// Helper: complex sort into u32, then convert to f64.
inline fn complexArgsortToF64(comptime T: type, a: [*]const T, out: [*]f64, N: u32) void {
    const out_u32: [*]u32 = @ptrCast(out);
    sc.initIndices(out_u32, N);
    sc.complexIntroSortStable(T, a, out_u32, N);
    sc.indicesToF64(out_u32, out, N);
}

/// Helper: batch slice argsort into u32, then convert to f64.
inline fn argsortSlicesToF64(comptime T: type, a: [*]const T, out: [*]f64, sliceSize: u32, numSlices: u32) void {
    const out_u32: [*]u32 = @ptrCast(out);
    sc.argsortSlices(T, a, out_u32, sliceSize, numSlices);
    sc.sliceIndicesToF64(out_u32, out, sliceSize, numSlices);
}

/// Helper: batch complex slice argsort into u32, then convert to f64.
inline fn complexArgsortSlicesToF64(comptime T: type, a: [*]const T, out: [*]f64, sliceSize: u32, numSlices: u32) void {
    const out_u32: [*]u32 = @ptrCast(out);
    sc.complexArgsortSlices(T, a, out_u32, sliceSize, numSlices);
    sc.sliceIndicesToF64(out_u32, out, sliceSize, numSlices);
}

// --- Single-array argsort (f64 index output) ---

/// Argsort for f64: ascending-order indices as f64. NaN sorts to end.
export fn argsort_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    argsortToF64(f64, a, out, N);
}

/// Argsort for f32: ascending-order indices as f64. NaN sorts to end.
export fn argsort_f32(a: [*]const f32, out: [*]f64, N: u32) void {
    argsortToF64(f32, a, out, N);
}

/// Argsort for i64: ascending-order indices as f64.
export fn argsort_i64(a: [*]const i64, out: [*]f64, N: u32) void {
    argsortToF64(i64, a, out, N);
}

/// Argsort for u64: ascending-order indices as f64.
export fn argsort_u64(a: [*]const u64, out: [*]f64, N: u32) void {
    argsortToF64(u64, a, out, N);
}

/// Argsort for i32: ascending-order indices as f64.
export fn argsort_i32(a: [*]const i32, out: [*]f64, N: u32) void {
    argsortToF64(i32, a, out, N);
}

/// Argsort for u32: ascending-order indices as f64.
export fn argsort_u32(a: [*]const u32, out: [*]f64, N: u32) void {
    argsortToF64(u32, a, out, N);
}

/// Argsort for i16: ascending-order indices as f64.
export fn argsort_i16(a: [*]const i16, out: [*]f64, N: u32) void {
    argsortToF64(i16, a, out, N);
}

/// Argsort for u16: ascending-order indices as f64.
export fn argsort_u16(a: [*]const u16, out: [*]f64, N: u32) void {
    argsortToF64(u16, a, out, N);
}

/// Argsort for i8: ascending-order indices as f64.
export fn argsort_i8(a: [*]const i8, out: [*]f64, N: u32) void {
    argsortToF64(i8, a, out, N);
}

/// Argsort for u8: ascending-order indices as f64.
export fn argsort_u8(a: [*]const u8, out: [*]f64, N: u32) void {
    argsortToF64(u8, a, out, N);
}

// --- Batch slice argsort (f64 index output) ---

/// Stable argsort of numSlices contiguous f64 slices. Indices as f64.
export fn argsort_slices_f64(a: [*]const f64, out: [*]f64, sliceSize: u32, numSlices: u32) void {
    argsortSlicesToF64(f64, a, out, sliceSize, numSlices);
}

/// Stable argsort of numSlices contiguous f32 slices. Indices as f64.
export fn argsort_slices_f32(a: [*]const f32, out: [*]f64, sliceSize: u32, numSlices: u32) void {
    argsortSlicesToF64(f32, a, out, sliceSize, numSlices);
}

/// Stable argsort of numSlices contiguous i64 slices. Indices as f64.
export fn argsort_slices_i64(a: [*]const i64, out: [*]f64, sliceSize: u32, numSlices: u32) void {
    argsortSlicesToF64(i64, a, out, sliceSize, numSlices);
}

/// Stable argsort of numSlices contiguous u64 slices. Indices as f64.
export fn argsort_slices_u64(a: [*]const u64, out: [*]f64, sliceSize: u32, numSlices: u32) void {
    argsortSlicesToF64(u64, a, out, sliceSize, numSlices);
}

/// Stable argsort of numSlices contiguous i32 slices. Indices as f64.
export fn argsort_slices_i32(a: [*]const i32, out: [*]f64, sliceSize: u32, numSlices: u32) void {
    argsortSlicesToF64(i32, a, out, sliceSize, numSlices);
}

/// Stable argsort of numSlices contiguous u32 slices. Indices as f64.
export fn argsort_slices_u32(a: [*]const u32, out: [*]f64, sliceSize: u32, numSlices: u32) void {
    argsortSlicesToF64(u32, a, out, sliceSize, numSlices);
}

/// Stable argsort of numSlices contiguous i16 slices. Indices as f64.
export fn argsort_slices_i16(a: [*]const i16, out: [*]f64, sliceSize: u32, numSlices: u32) void {
    argsortSlicesToF64(i16, a, out, sliceSize, numSlices);
}

/// Stable argsort of numSlices contiguous u16 slices. Indices as f64.
export fn argsort_slices_u16(a: [*]const u16, out: [*]f64, sliceSize: u32, numSlices: u32) void {
    argsortSlicesToF64(u16, a, out, sliceSize, numSlices);
}

/// Stable argsort of numSlices contiguous i8 slices. Indices as f64.
export fn argsort_slices_i8(a: [*]const i8, out: [*]f64, sliceSize: u32, numSlices: u32) void {
    argsortSlicesToF64(i8, a, out, sliceSize, numSlices);
}

/// Stable argsort of numSlices contiguous u8 slices. Indices as f64.
export fn argsort_slices_u8(a: [*]const u8, out: [*]f64, sliceSize: u32, numSlices: u32) void {
    argsortSlicesToF64(u8, a, out, sliceSize, numSlices);
}

// --- Complex argsort (f64 index output) ---

/// Stable argsort for complex128 (interleaved f64 pairs). Indices as f64.
export fn argsort_c128(a: [*]const f64, out: [*]f64, N: u32) void {
    complexArgsortToF64(f64, a, out, N);
}

/// Stable argsort for complex64 (interleaved f32 pairs). Indices as f64.
export fn argsort_c64(a: [*]const f32, out: [*]f64, N: u32) void {
    complexArgsortToF64(f32, a, out, N);
}

/// Stable argsort of numSlices contiguous complex128 slices. Indices as f64.
export fn argsort_slices_c128(a: [*]const f64, out: [*]f64, sliceSize: u32, numSlices: u32) void {
    complexArgsortSlicesToF64(f64, a, out, sliceSize, numSlices);
}

/// Stable argsort of numSlices contiguous complex64 slices. Indices as f64.
export fn argsort_slices_c64(a: [*]const f32, out: [*]f64, sliceSize: u32, numSlices: u32) void {
    complexArgsortSlicesToF64(f32, a, out, sliceSize, numSlices);
}

// --- Tests ---

test "argsort_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 3.0, 1.0, 4.0, 1.5, 2.0 };
    var out: [5]f64 = undefined;
    argsort_f64(&a, &out, 5);
    try testing.expectEqual(out[0], 1.0);
    try testing.expectEqual(out[1], 3.0);
    try testing.expectEqual(out[2], 4.0);
    try testing.expectEqual(out[3], 0.0);
    try testing.expectEqual(out[4], 2.0);
}

test "argsort_f64 duplicates stable" {
    const testing = @import("std").testing;
    const a = [_]f64{ 3.0, 1.0, 3.0, 1.0, 2.0 };
    var out: [5]f64 = undefined;
    argsort_f64(&a, &out, 5);
    try testing.expectEqual(out[0], 1.0);
    try testing.expectEqual(out[1], 3.0);
    try testing.expectEqual(out[2], 4.0);
    try testing.expectEqual(out[3], 0.0);
    try testing.expectEqual(out[4], 2.0);
}

test "argsort_f64 NaN handling stable" {
    const testing = @import("std").testing;
    const nan = @as(f64, @bitCast(@as(u64, 0x7FF8000000000000)));
    const a = [_]f64{ nan, 2.0, 1.0, nan, 3.0 };
    var out: [5]f64 = undefined;
    argsort_f64(&a, &out, 5);
    try testing.expectEqual(out[0], 2.0);
    try testing.expectEqual(out[1], 1.0);
    try testing.expectEqual(out[2], 4.0);
    try testing.expectEqual(out[3], 0.0);
    try testing.expectEqual(out[4], 3.0);
}

test "argsort_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 30, -10, 20, 0 };
    var out: [4]f64 = undefined;
    argsort_i32(&a, &out, 4);
    try testing.expectEqual(out[0], 1.0);
    try testing.expectEqual(out[1], 3.0);
    try testing.expectEqual(out[2], 2.0);
    try testing.expectEqual(out[3], 0.0);
}

test "argsort_u8 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{ 200, 50, 100, 10 };
    var out: [4]f64 = undefined;
    argsort_u8(&a, &out, 4);
    try testing.expectEqual(out[0], 3.0);
    try testing.expectEqual(out[1], 1.0);
    try testing.expectEqual(out[2], 2.0);
    try testing.expectEqual(out[3], 0.0);
}

test "argsort_f64 single element" {
    const testing = @import("std").testing;
    const a = [_]f64{42.0};
    var out: [1]f64 = undefined;
    argsort_f64(&a, &out, 1);
    try testing.expectEqual(out[0], 0.0);
}

test "argsort_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 3.0, 1.0, 2.0 };
    var out: [3]f64 = undefined;
    argsort_f32(&a, &out, 3);
    try testing.expectEqual(out[0], 1.0);
    try testing.expectEqual(out[1], 2.0);
    try testing.expectEqual(out[2], 0.0);
}

test "argsort_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 100, -50, 0 };
    var out: [3]f64 = undefined;
    argsort_i64(&a, &out, 3);
    try testing.expectEqual(out[0], 1.0);
    try testing.expectEqual(out[1], 2.0);
    try testing.expectEqual(out[2], 0.0);
}

test "argsort_u64 basic" {
    const testing = @import("std").testing;
    const a = [_]u64{ 300, 100, 200 };
    var out: [3]f64 = undefined;
    argsort_u64(&a, &out, 3);
    try testing.expectEqual(out[0], 1.0);
    try testing.expectEqual(out[1], 2.0);
    try testing.expectEqual(out[2], 0.0);
}

test "argsort_u32 basic" {
    const testing = @import("std").testing;
    const a = [_]u32{ 5, 1, 3 };
    var out: [3]f64 = undefined;
    argsort_u32(&a, &out, 3);
    try testing.expectEqual(out[0], 1.0);
    try testing.expectEqual(out[1], 2.0);
    try testing.expectEqual(out[2], 0.0);
}

test "argsort_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{ 10, -20, 0 };
    var out: [3]f64 = undefined;
    argsort_i16(&a, &out, 3);
    try testing.expectEqual(out[0], 1.0);
    try testing.expectEqual(out[1], 2.0);
    try testing.expectEqual(out[2], 0.0);
}

test "argsort_u16 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{ 500, 100, 300 };
    var out: [3]f64 = undefined;
    argsort_u16(&a, &out, 3);
    try testing.expectEqual(out[0], 1.0);
    try testing.expectEqual(out[1], 2.0);
    try testing.expectEqual(out[2], 0.0);
}

test "argsort_i8 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{ 10, -5, 0 };
    var out: [3]f64 = undefined;
    argsort_i8(&a, &out, 3);
    try testing.expectEqual(out[0], 1.0);
    try testing.expectEqual(out[1], 2.0);
    try testing.expectEqual(out[2], 0.0);
}

test "argsort_slices_f64 two slices" {
    const testing = @import("std").testing;
    const a = [_]f64{ 3.0, 1.0, 2.0, 9.0, 7.0, 8.0 };
    var out: [6]f64 = undefined;
    argsort_slices_f64(&a, &out, 3, 2);
    try testing.expectEqual(out[0], 1.0);
    try testing.expectEqual(out[2], 0.0);
    try testing.expectEqual(out[3], 1.0);
    try testing.expectEqual(out[5], 0.0);
}

test "argsort_slices_f32 two slices" {
    const testing = @import("std").testing;
    const a = [_]f32{ 3.0, 1.0, 2.0, 9.0, 7.0, 8.0 };
    var out: [6]f64 = undefined;
    argsort_slices_f32(&a, &out, 3, 2);
    try testing.expectEqual(out[0], 1.0);
    try testing.expectEqual(out[2], 0.0);
    try testing.expectEqual(out[3], 1.0);
    try testing.expectEqual(out[5], 0.0);
}

test "argsort_slices_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 30, 10, 20, -10, -30, -20 };
    var out: [6]f64 = undefined;
    argsort_slices_i64(&a, &out, 3, 2);
    try testing.expectEqual(out[0], 1.0);
    try testing.expectEqual(out[3], 1.0);
}

test "argsort_slices_u64 basic" {
    const testing = @import("std").testing;
    const a = [_]u64{ 30, 10, 20 };
    var out: [3]f64 = undefined;
    argsort_slices_u64(&a, &out, 3, 1);
    try testing.expectEqual(out[0], 1.0);
    try testing.expectEqual(out[2], 0.0);
}

test "argsort_slices_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 30, 10, 20 };
    var out: [3]f64 = undefined;
    argsort_slices_i32(&a, &out, 3, 1);
    try testing.expectEqual(out[0], 1.0);
    try testing.expectEqual(out[2], 0.0);
}

test "argsort_slices_u32 basic" {
    const testing = @import("std").testing;
    const a = [_]u32{ 30, 10, 20 };
    var out: [3]f64 = undefined;
    argsort_slices_u32(&a, &out, 3, 1);
    try testing.expectEqual(out[0], 1.0);
    try testing.expectEqual(out[2], 0.0);
}

test "argsort_slices_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{ 30, 10, 20 };
    var out: [3]f64 = undefined;
    argsort_slices_i16(&a, &out, 3, 1);
    try testing.expectEqual(out[0], 1.0);
    try testing.expectEqual(out[2], 0.0);
}

test "argsort_slices_u16 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{ 300, 100, 200 };
    var out: [3]f64 = undefined;
    argsort_slices_u16(&a, &out, 3, 1);
    try testing.expectEqual(out[0], 1.0);
    try testing.expectEqual(out[2], 0.0);
}

test "argsort_slices_i8 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{ 30, 10, 20 };
    var out: [3]f64 = undefined;
    argsort_slices_i8(&a, &out, 3, 1);
    try testing.expectEqual(out[0], 1.0);
    try testing.expectEqual(out[2], 0.0);
}

test "argsort_slices_u8 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{ 200, 50, 100 };
    var out: [3]f64 = undefined;
    argsort_slices_u8(&a, &out, 3, 1);
    try testing.expectEqual(out[0], 1.0);
    try testing.expectEqual(out[2], 0.0);
}

test "argsort_c128 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 3.0, 0.0, 1.0, 0.0, 2.0, 0.0 };
    var out: [3]f64 = undefined;
    argsort_c128(&a, &out, 3);
    try testing.expectEqual(out[0], 1.0);
    try testing.expectEqual(out[1], 2.0);
    try testing.expectEqual(out[2], 0.0);
}

test "argsort_c64 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 3.0, 0.0, 1.0, 0.0, 2.0, 0.0 };
    var out: [3]f64 = undefined;
    argsort_c64(&a, &out, 3);
    try testing.expectEqual(out[0], 1.0);
    try testing.expectEqual(out[1], 2.0);
    try testing.expectEqual(out[2], 0.0);
}

test "argsort_slices_c128 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 3.0, 0.0, 1.0, 0.0 };
    var out: [2]f64 = undefined;
    argsort_slices_c128(&a, &out, 2, 1);
    try testing.expectEqual(out[0], 1.0);
    try testing.expectEqual(out[1], 0.0);
}

test "argsort_slices_c64 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 3.0, 0.0, 1.0, 0.0 };
    var out: [2]f64 = undefined;
    argsort_slices_c64(&a, &out, 2, 1);
    try testing.expectEqual(out[0], 1.0);
    try testing.expectEqual(out[1], 0.0);
}
