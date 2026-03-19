//! WASM searchsorted kernels for all numeric types.
//!
//! Binary search for insertion indices into a sorted array.
//! Left variant (bisect_left): first index where sorted[i] >= value.
//! Right variant (bisect_right): first index where sorted[i] > value.
//! Supported dtypes: f64, f32, i64, u64, i32, u32, i16, u16, i8, u8.

/// Binary search (bisect_left) for f64. NaN values map to index N.
export fn searchsorted_left_f64(sorted: [*]const f64, N: u32, values: [*]const f64, out: [*]u32, M: u32) void {
    searchsortedLeftBatch(f64, sorted, N, values, out, M);
}

/// Binary search (bisect_left) for f32. NaN values map to index N.
export fn searchsorted_left_f32(sorted: [*]const f32, N: u32, values: [*]const f32, out: [*]u32, M: u32) void {
    searchsortedLeftBatch(f32, sorted, N, values, out, M);
}

/// Binary search (bisect_left) for i64.
export fn searchsorted_left_i64(sorted: [*]const i64, N: u32, values: [*]const i64, out: [*]u32, M: u32) void {
    searchsortedLeftBatch(i64, sorted, N, values, out, M);
}

/// Binary search (bisect_left) for u64.
export fn searchsorted_left_u64(sorted: [*]const u64, N: u32, values: [*]const u64, out: [*]u32, M: u32) void {
    searchsortedLeftBatch(u64, sorted, N, values, out, M);
}

/// Binary search (bisect_left) for i32.
export fn searchsorted_left_i32(sorted: [*]const i32, N: u32, values: [*]const i32, out: [*]u32, M: u32) void {
    searchsortedLeftBatch(i32, sorted, N, values, out, M);
}

/// Binary search (bisect_left) for u32.
export fn searchsorted_left_u32(sorted: [*]const u32, N: u32, values: [*]const u32, out: [*]u32, M: u32) void {
    searchsortedLeftBatch(u32, sorted, N, values, out, M);
}

/// Binary search (bisect_left) for i16.
export fn searchsorted_left_i16(sorted: [*]const i16, N: u32, values: [*]const i16, out: [*]u32, M: u32) void {
    searchsortedLeftBatch(i16, sorted, N, values, out, M);
}

/// Binary search (bisect_left) for u16.
export fn searchsorted_left_u16(sorted: [*]const u16, N: u32, values: [*]const u16, out: [*]u32, M: u32) void {
    searchsortedLeftBatch(u16, sorted, N, values, out, M);
}

/// Binary search (bisect_left) for i8.
export fn searchsorted_left_i8(sorted: [*]const i8, N: u32, values: [*]const i8, out: [*]u32, M: u32) void {
    searchsortedLeftBatch(i8, sorted, N, values, out, M);
}

/// Binary search (bisect_left) for u8.
export fn searchsorted_left_u8(sorted: [*]const u8, N: u32, values: [*]const u8, out: [*]u32, M: u32) void {
    searchsortedLeftBatch(u8, sorted, N, values, out, M);
}

/// Binary search (bisect_right) for f64. NaN values map to index N.
export fn searchsorted_right_f64(sorted: [*]const f64, N: u32, values: [*]const f64, out: [*]u32, M: u32) void {
    searchsortedRightBatch(f64, sorted, N, values, out, M);
}

/// Binary search (bisect_right) for f32. NaN values map to index N.
export fn searchsorted_right_f32(sorted: [*]const f32, N: u32, values: [*]const f32, out: [*]u32, M: u32) void {
    searchsortedRightBatch(f32, sorted, N, values, out, M);
}

/// Binary search (bisect_right) for i64.
export fn searchsorted_right_i64(sorted: [*]const i64, N: u32, values: [*]const i64, out: [*]u32, M: u32) void {
    searchsortedRightBatch(i64, sorted, N, values, out, M);
}

/// Binary search (bisect_right) for u64.
export fn searchsorted_right_u64(sorted: [*]const u64, N: u32, values: [*]const u64, out: [*]u32, M: u32) void {
    searchsortedRightBatch(u64, sorted, N, values, out, M);
}

/// Binary search (bisect_right) for i32.
export fn searchsorted_right_i32(sorted: [*]const i32, N: u32, values: [*]const i32, out: [*]u32, M: u32) void {
    searchsortedRightBatch(i32, sorted, N, values, out, M);
}

/// Binary search (bisect_right) for u32.
export fn searchsorted_right_u32(sorted: [*]const u32, N: u32, values: [*]const u32, out: [*]u32, M: u32) void {
    searchsortedRightBatch(u32, sorted, N, values, out, M);
}

/// Binary search (bisect_right) for i16.
export fn searchsorted_right_i16(sorted: [*]const i16, N: u32, values: [*]const i16, out: [*]u32, M: u32) void {
    searchsortedRightBatch(i16, sorted, N, values, out, M);
}

/// Binary search (bisect_right) for u16.
export fn searchsorted_right_u16(sorted: [*]const u16, N: u32, values: [*]const u16, out: [*]u32, M: u32) void {
    searchsortedRightBatch(u16, sorted, N, values, out, M);
}

/// Binary search (bisect_right) for i8.
export fn searchsorted_right_i8(sorted: [*]const i8, N: u32, values: [*]const i8, out: [*]u32, M: u32) void {
    searchsortedRightBatch(i8, sorted, N, values, out, M);
}

/// Binary search (bisect_right) for u8.
export fn searchsorted_right_u8(sorted: [*]const u8, N: u32, values: [*]const u8, out: [*]u32, M: u32) void {
    searchsortedRightBatch(u8, sorted, N, values, out, M);
}

/// Returns true if v is NaN (only valid for float types).
fn isNan(comptime T: type, v: T) bool {
    return v != v;
}

/// Binary search for a single value: finds leftmost insertion point.
fn searchsortedLeft(comptime T: type, sorted: [*]const T, N: u32, value: T) u32 {
    if (T == f64 or T == f32) {
        if (isNan(T, value)) return N;
    }
    var lo: u32 = 0;
    var hi: u32 = N;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (sorted[mid] < value) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

/// Binary search for a single value: finds rightmost insertion point.
fn searchsortedRight(comptime T: type, sorted: [*]const T, N: u32, value: T) u32 {
    if (T == f64 or T == f32) {
        if (isNan(T, value)) return N;
    }
    var lo: u32 = 0;
    var hi: u32 = N;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (sorted[mid] <= value) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

/// Batch left search: finds insertion indices for M values.
fn searchsortedLeftBatch(comptime T: type, sorted: [*]const T, N: u32, values: [*]const T, out: [*]u32, M: u32) void {
    var i: u32 = 0;
    while (i < M) : (i += 1) {
        out[i] = searchsortedLeft(T, sorted, N, values[i]);
    }
}

/// Batch right search: finds insertion indices for M values.
fn searchsortedRightBatch(comptime T: type, sorted: [*]const T, N: u32, values: [*]const T, out: [*]u32, M: u32) void {
    var i: u32 = 0;
    while (i < M) : (i += 1) {
        out[i] = searchsortedRight(T, sorted, N, values[i]);
    }
}

// --- Tests ---

test "searchsorted_left_f64 basic" {
    const testing = @import("std").testing;
    const sorted = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const values = [_]f64{ 0.5, 1.0, 2.5, 5.0, 6.0 };
    var out: [5]u32 = undefined;
    searchsorted_left_f64(&sorted, 5, &values, &out, 5);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], 2);
    try testing.expectEqual(out[3], 4);
    try testing.expectEqual(out[4], 5);
}

test "searchsorted_right_f64 basic" {
    const testing = @import("std").testing;
    const sorted = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const values = [_]f64{ 0.5, 1.0, 2.5, 5.0, 6.0 };
    var out: [5]u32 = undefined;
    searchsorted_right_f64(&sorted, 5, &values, &out, 5);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 2);
    try testing.expectEqual(out[3], 5);
    try testing.expectEqual(out[4], 5);
}

test "searchsorted_left_f64 duplicates" {
    const testing = @import("std").testing;
    const sorted = [_]f64{ 1.0, 2.0, 2.0, 2.0, 3.0 };
    const values = [_]f64{2.0};
    var out: [1]u32 = undefined;
    searchsorted_left_f64(&sorted, 5, &values, &out, 1);
    try testing.expectEqual(out[0], 1);
}

test "searchsorted_right_f64 duplicates" {
    const testing = @import("std").testing;
    const sorted = [_]f64{ 1.0, 2.0, 2.0, 2.0, 3.0 };
    const values = [_]f64{2.0};
    var out: [1]u32 = undefined;
    searchsorted_right_f64(&sorted, 5, &values, &out, 1);
    try testing.expectEqual(out[0], 4);
}

test "searchsorted_left_f64 NaN" {
    const testing = @import("std").testing;
    const sorted = [_]f64{ 1.0, 2.0, 3.0 };
    const nan = @as(f64, @bitCast(@as(u64, 0x7FF8000000000000)));
    const values = [_]f64{nan};
    var out: [1]u32 = undefined;
    searchsorted_left_f64(&sorted, 3, &values, &out, 1);
    try testing.expectEqual(out[0], 3);
}

test "searchsorted_left_i32 basic" {
    const testing = @import("std").testing;
    const sorted = [_]i32{ -10, 0, 5, 10, 20 };
    const values = [_]i32{ -20, -10, 3, 10, 25 };
    var out: [5]u32 = undefined;
    searchsorted_left_i32(&sorted, 5, &values, &out, 5);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], 2);
    try testing.expectEqual(out[3], 3);
    try testing.expectEqual(out[4], 5);
}

test "searchsorted_left_u8 basic" {
    const testing = @import("std").testing;
    const sorted = [_]u8{ 10, 20, 30, 40, 50 };
    const values = [_]u8{ 0, 10, 25, 50, 255 };
    var out: [5]u32 = undefined;
    searchsorted_left_u8(&sorted, 5, &values, &out, 5);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], 2);
    try testing.expectEqual(out[3], 4);
    try testing.expectEqual(out[4], 5);
}

test "searchsorted_left_f64 empty sorted array" {
    const testing = @import("std").testing;
    const sorted = [_]f64{};
    const values = [_]f64{ 1.0, 2.0 };
    var out: [2]u32 = undefined;
    searchsorted_left_f64(&sorted, 0, &values, &out, 2);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 0);
}
