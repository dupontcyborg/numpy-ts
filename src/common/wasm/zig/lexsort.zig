//! WASM lexsort kernels for all numeric types.
//!
//! Lexicographic indirect sort over multiple keys (NumPy convention).
//! keys is a flat buffer of num_keys * N elements where keys[k*N + i] is key k's i-th element.
//! The LAST key is the primary sort key; ties are broken by earlier keys.
//! out receives the sorted indices [0..N) as f64.
//!
//! Output is f64 for JS ergonomics. Internally sorts u32 indices in the
//! lower half of the f64 output buffer, then converts in-place via SIMD.
//!
//! Supported dtypes: f64, f32, i64, u64, i32, u32, i16, u16, i8, u8.
//! Floating-point variants treat NaN as greater than all values.
//! Algorithm: heap sort (in-place, O(N log N) worst-case, stable via index tiebreaker).

const sc = @import("sorting_common.zig");

/// Helper: lexsort into u32 indices in the f64 buffer, then convert to f64 in-place.
inline fn lexsortToF64(comptime T: type, keys: [*]const T, num_keys: u32, N: u32, out: [*]f64) void {
    const out_u32: [*]u32 = @ptrCast(out);
    sc.initIndices(out_u32, N);
    sc.lexHeapSort(T, keys, num_keys, N, out_u32);
    sc.indicesToF64(out_u32, out, N);
}

/// Helper: lexsort with scratch (for i8/u8/i16/u16 radix sort) into u32, then convert to f64.
inline fn lexsortWithScratchToF64(comptime T: type, keys: [*]const T, num_keys: u32, N: u32, out: [*]f64, scratch: [*]u32) void {
    const out_u32: [*]u32 = @ptrCast(out);
    sc.initIndices(out_u32, N);
    sc.lexRadixSort(T, keys, num_keys, N, out_u32, scratch);
    sc.indicesToF64(out_u32, out, N);
}

// --- Lexsort (f64 index output) ---

/// Lexicographic indirect sort for f64 keys. Indices as f64.
export fn lexsort_f64(keys: [*]const f64, num_keys: u32, N: u32, out: [*]f64) void {
    lexsortToF64(f64, keys, num_keys, N, out);
}

/// Lexicographic indirect sort for f32 keys. Indices as f64.
export fn lexsort_f32(keys: [*]const f32, num_keys: u32, N: u32, out: [*]f64) void {
    lexsortToF64(f32, keys, num_keys, N, out);
}

/// Lexicographic indirect sort for i64 keys. Indices as f64.
export fn lexsort_i64(keys: [*]const i64, num_keys: u32, N: u32, out: [*]f64) void {
    lexsortToF64(i64, keys, num_keys, N, out);
}

/// Lexicographic indirect sort for u64 keys. Indices as f64.
export fn lexsort_u64(keys: [*]const u64, num_keys: u32, N: u32, out: [*]f64) void {
    lexsortToF64(u64, keys, num_keys, N, out);
}

/// Lexicographic indirect sort for i32 keys. Indices as f64.
export fn lexsort_i32(keys: [*]const i32, num_keys: u32, N: u32, out: [*]f64) void {
    lexsortToF64(i32, keys, num_keys, N, out);
}

/// Lexicographic indirect sort for u32 keys. Indices as f64.
export fn lexsort_u32(keys: [*]const u32, num_keys: u32, N: u32, out: [*]f64) void {
    lexsortToF64(u32, keys, num_keys, N, out);
}

/// Lexicographic indirect sort for i16 keys. Uses radix sort. Indices as f64.
export fn lexsort_i16(keys: [*]const i16, num_keys: u32, N: u32, out: [*]f64, scratch: [*]u32) void {
    lexsortWithScratchToF64(i16, keys, num_keys, N, out, scratch);
}

/// Lexicographic indirect sort for u16 keys. Uses radix sort. Indices as f64.
export fn lexsort_u16(keys: [*]const u16, num_keys: u32, N: u32, out: [*]f64, scratch: [*]u32) void {
    lexsortWithScratchToF64(u16, keys, num_keys, N, out, scratch);
}

/// Lexicographic indirect sort for i8 keys. Uses radix sort. Indices as f64.
export fn lexsort_i8(keys: [*]const i8, num_keys: u32, N: u32, out: [*]f64, scratch: [*]u32) void {
    lexsortWithScratchToF64(i8, keys, num_keys, N, out, scratch);
}

/// Lexicographic indirect sort for u8 keys. Uses radix sort. Indices as f64.
export fn lexsort_u8(keys: [*]const u8, num_keys: u32, N: u32, out: [*]f64, scratch: [*]u32) void {
    lexsortWithScratchToF64(u8, keys, num_keys, N, out, scratch);
}

// --- Tests ---

test "lexsort_f64 basic 2-key" {
    const testing = @import("std").testing;
    const keys = [_]f64{ 1.0, 2.0, 1.0, 2.0, 9.0, 4.0, 2.0, 3.0 };
    var out: [4]f64 = undefined;
    lexsort_f64(&keys, 2, 4, &out);
    try testing.expectEqual(out[0], 2.0);
    try testing.expectEqual(out[1], 3.0);
    try testing.expectEqual(out[2], 1.0);
    try testing.expectEqual(out[3], 0.0);
}

test "lexsort_i32 ties in primary broken by secondary" {
    const testing = @import("std").testing;
    const keys = [_]i32{ 3, 1, 2, 5, 5, 5 };
    var out: [3]f64 = undefined;
    lexsort_i32(&keys, 2, 3, &out);
    try testing.expectEqual(out[0], 1.0);
    try testing.expectEqual(out[1], 2.0);
    try testing.expectEqual(out[2], 0.0);
}

test "lexsort_i32 single key degenerates to argsort" {
    const testing = @import("std").testing;
    const keys = [_]i32{ 30, 10, 20 };
    var out: [3]f64 = undefined;
    lexsort_i32(&keys, 1, 3, &out);
    try testing.expectEqual(out[0], 1.0);
    try testing.expectEqual(out[1], 2.0);
    try testing.expectEqual(out[2], 0.0);
}

test "lexsort_f64 NaN handling" {
    const testing = @import("std").testing;
    const nan = @as(f64, @bitCast(@as(u64, 0x7FF8000000000000)));
    const keys = [_]f64{ nan, 1.0, 3.0, nan, 2.0 };
    var out: [5]f64 = undefined;
    lexsort_f64(&keys, 1, 5, &out);
    try testing.expectEqual(out[0], 1.0);
    try testing.expectEqual(out[1], 4.0);
    try testing.expectEqual(out[2], 2.0);
    try testing.expectEqual(out[3], 0.0);
    try testing.expectEqual(out[4], 3.0);
}

test "lexsort_i32 single element" {
    const testing = @import("std").testing;
    const keys = [_]i32{ 42, 7 };
    var out: [1]f64 = undefined;
    lexsort_i32(&keys, 2, 1, &out);
    try testing.expectEqual(out[0], 0.0);
}

test "lexsort_i8 radix 2-key" {
    const testing = @import("std").testing;
    // key0 (secondary): [3, 1, 2], key1 (primary): [5, 5, 5] — all tied, sort by key0
    const keys = [_]i8{ 3, 1, 2, 5, 5, 5 };
    var out: [3]f64 = undefined;
    var scratch: [3]u32 = undefined;
    lexsort_i8(&keys, 2, 3, &out, &scratch);
    try testing.expectEqual(out[0], 1.0); // key0=1
    try testing.expectEqual(out[1], 2.0); // key0=2
    try testing.expectEqual(out[2], 0.0); // key0=3
}

test "lexsort_i8 radix negative values" {
    const testing = @import("std").testing;
    const keys = [_]i8{ -5, 3, -1, 0 };
    var out: [4]f64 = undefined;
    var scratch: [4]u32 = undefined;
    lexsort_i8(&keys, 1, 4, &out, &scratch);
    try testing.expectEqual(out[0], 0.0); // -5
    try testing.expectEqual(out[1], 2.0); // -1
    try testing.expectEqual(out[2], 3.0); // 0
    try testing.expectEqual(out[3], 1.0); // 3
}

test "lexsort_i16 radix basic" {
    const testing = @import("std").testing;
    const keys = [_]i16{ 300, -100, 200 };
    var out: [3]f64 = undefined;
    var scratch: [3]u32 = undefined;
    lexsort_i16(&keys, 1, 3, &out, &scratch);
    try testing.expectEqual(out[0], 1.0); // -100
    try testing.expectEqual(out[1], 2.0); // 200
    try testing.expectEqual(out[2], 0.0); // 300
}

test "lexsort_f32 basic" {
    const testing = @import("std").testing;
    const keys = [_]f32{ 3.0, 1.0, 2.0 };
    var out: [3]f64 = undefined;
    lexsort_f32(&keys, 1, 3, &out);
    try testing.expectEqual(out[0], 1.0);
    try testing.expectEqual(out[1], 2.0);
    try testing.expectEqual(out[2], 0.0);
}

test "lexsort_i64 basic" {
    const testing = @import("std").testing;
    const keys = [_]i64{ 30, 10, 20 };
    var out: [3]f64 = undefined;
    lexsort_i64(&keys, 1, 3, &out);
    try testing.expectEqual(out[0], 1.0);
    try testing.expectEqual(out[1], 2.0);
    try testing.expectEqual(out[2], 0.0);
}

test "lexsort_u64 basic" {
    const testing = @import("std").testing;
    const keys = [_]u64{ 30, 10, 20 };
    var out: [3]f64 = undefined;
    lexsort_u64(&keys, 1, 3, &out);
    try testing.expectEqual(out[0], 1.0);
    try testing.expectEqual(out[1], 2.0);
    try testing.expectEqual(out[2], 0.0);
}

test "lexsort_u32 basic" {
    const testing = @import("std").testing;
    const keys = [_]u32{ 30, 10, 20 };
    var out: [3]f64 = undefined;
    lexsort_u32(&keys, 1, 3, &out);
    try testing.expectEqual(out[0], 1.0);
    try testing.expectEqual(out[1], 2.0);
    try testing.expectEqual(out[2], 0.0);
}

test "lexsort_u16 basic" {
    const testing = @import("std").testing;
    const keys = [_]u16{ 300, 100, 200 };
    var out: [3]f64 = undefined;
    var scratch: [3]u32 = undefined;
    lexsort_u16(&keys, 1, 3, &out, &scratch);
    try testing.expectEqual(out[0], 1.0);
    try testing.expectEqual(out[1], 2.0);
    try testing.expectEqual(out[2], 0.0);
}

test "lexsort_u8 basic" {
    const testing = @import("std").testing;
    const keys = [_]u8{ 30, 10, 20 };
    var out: [3]f64 = undefined;
    var scratch: [3]u32 = undefined;
    lexsort_u8(&keys, 1, 3, &out, &scratch);
    try testing.expectEqual(out[0], 1.0);
    try testing.expectEqual(out[1], 2.0);
    try testing.expectEqual(out[2], 0.0);
}
