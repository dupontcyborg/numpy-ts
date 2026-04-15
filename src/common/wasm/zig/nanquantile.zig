//! WASM nanquantile kernels for float types only.
//!
//! For each dtype: filter out NaN values, sort the remaining, compute
//! quantile via linear interpolation. Returns NaN if all values are NaN.
//!
//! Flat variants: nanquantile_f64 / nanquantile_f32
//! Strided variants: nanquantile_strided_f64 / nanquantile_strided_f32
//!   Operate on [outer, axis_size, inner] layout — for each outer*inner
//!   position, gather axis elements, filter NaN, sort, interpolate.

const sc = @import("sorting_common.zig");

/// Check if a float value is NaN (v != v is true only for NaN).
fn isNan(comptime T: type, v: T) bool {
    return v != v;
}

/// Filter non-NaN values from src into work buffer. Returns count of valid values.
fn filterNaN(comptime T: type, src: [*]const T, N: u32, work: [*]T) u32 {
    var count: u32 = 0;
    for (0..N) |i| {
        const v = src[i];
        if (!isNan(T, v)) {
            work[count] = v;
            count += 1;
        }
    }
    return count;
}

/// Compute quantile from a sorted buffer of `n` valid elements.
fn interpolate(comptime T: type, sorted: [*]const T, n: u32, q: T) T {
    if (n == 0) return @as(T, @bitCast(if (T == f64) @as(u64, 0x7FF8000000000000) else @as(u32, 0x7FC00000)));
    if (n == 1) return sorted[0];
    const nf: T = @floatFromInt(n - 1);
    const idx = q * nf;
    const lower: u32 = @intFromFloat(@floor(idx));
    const upper: u32 = @intFromFloat(@ceil(idx));
    if (lower == upper) return sorted[lower];
    const frac = idx - @as(T, @floatFromInt(lower));
    return sorted[lower] * (1 - frac) + sorted[upper] * frac;
}

/// Nanquantile of f64 array. Returns NaN if all values are NaN.
export fn nanquantile_f64(a: [*]const f64, N: u32, q: f64, work: [*]f64) f64 {
    const valid = filterNaN(f64, a, N, work);
    if (valid == 0) return @as(f64, @bitCast(@as(u64, 0x7FF8000000000000)));
    sc.introSort(f64, work, valid);
    return interpolate(f64, work, valid, q);
}

/// Nanquantile of f32 array. Returns NaN if all values are NaN.
export fn nanquantile_f32(a: [*]const f32, N: u32, q: f32, work: [*]f32) f32 {
    const valid = filterNaN(f32, a, N, work);
    if (valid == 0) return @as(f32, @bitCast(@as(u32, 0x7FC00000)));
    sc.introSort(f32, work, valid);
    return interpolate(f32, work, valid, q);
}

/// Strided nanquantile for f64.
/// Layout: [outer, axis_size, inner]. Output: [outer * inner] f64 values.
/// work must have at least axis_size elements.
export fn nanquantile_strided_f64(a: [*]const f64, out: [*]f64, outer: u32, axis_size: u32, inner: u32, q: f64, work: [*]f64) void {
    const stride = axis_size * inner;
    for (0..outer) |o| {
        for (0..inner) |inn| {
            // Gather axis elements along the reduction dimension, filtering NaN
            var count: u32 = 0;
            const base = o * stride + inn;
            for (0..axis_size) |k| {
                const v = a[base + k * inner];
                if (!isNan(f64, v)) {
                    work[count] = v;
                    count += 1;
                }
            }
            const outIdx = o * inner + inn;
            if (count == 0) {
                // All NaN along this axis slice — output NaN
                out[outIdx] = @as(f64, @bitCast(@as(u64, 0x7FF8000000000000)));
            } else {
                sc.introSort(f64, work, count);
                out[outIdx] = interpolate(f64, work, count, q);
            }
        }
    }
}

/// Strided nanquantile for f32.
/// Layout: [outer, axis_size, inner]. Output: [outer * inner] f64 values.
/// work must have at least axis_size elements.
export fn nanquantile_strided_f32(a: [*]const f32, out: [*]f64, outer: u32, axis_size: u32, inner: u32, q: f32, work: [*]f32) void {
    const stride = axis_size * inner;
    for (0..outer) |o| {
        for (0..inner) |inn| {
            // Gather axis elements along the reduction dimension, filtering NaN
            var count: u32 = 0;
            const base = o * stride + inn;
            for (0..axis_size) |k| {
                const v = a[base + k * inner];
                if (!isNan(f32, v)) {
                    work[count] = v;
                    count += 1;
                }
            }
            const outIdx = o * inner + inn;
            if (count == 0) {
                // All NaN along this axis slice — output NaN
                out[outIdx] = @as(f64, @bitCast(@as(u64, 0x7FF8000000000000)));
            } else {
                sc.introSort(f32, work, count);
                out[outIdx] = interpolate(f32, work, count, @as(f32, q));
            }
        }
    }
}

// --- Tests ---

test "nanquantile_f64 basic" {
    const testing = @import("std").testing;
    const nan = @as(f64, @bitCast(@as(u64, 0x7FF8000000000000)));
    const a = [_]f64{ 1.0, 2.0, nan, 4.0, 5.0 };
    var work: [5]f64 = undefined;
    const result = nanquantile_f64(&a, 5, 0.5, &work);
    try testing.expectApproxEqAbs(result, 3.0, 1e-10);
}

test "nanquantile_f64 all NaN returns NaN" {
    const testing = @import("std").testing;
    const nan = @as(f64, @bitCast(@as(u64, 0x7FF8000000000000)));
    const a = [_]f64{ nan, nan, nan };
    var work: [3]f64 = undefined;
    const result = nanquantile_f64(&a, 3, 0.5, &work);
    try testing.expect(isNan(f64, result));
}

test "nanquantile_f64 no NaN" {
    const testing = @import("std").testing;
    const a = [_]f64{ 5.0, 1.0, 3.0, 2.0, 4.0 };
    var work: [5]f64 = undefined;
    try testing.expectApproxEqAbs(nanquantile_f64(&a, 5, 0.0, &work), 1.0, 1e-10);
    try testing.expectApproxEqAbs(nanquantile_f64(&a, 5, 1.0, &work), 5.0, 1e-10);
    try testing.expectApproxEqAbs(nanquantile_f64(&a, 5, 0.25, &work), 2.0, 1e-10);
}

test "nanquantile_f32 basic" {
    const testing = @import("std").testing;
    const nan = @as(f32, @bitCast(@as(u32, 0x7FC00000)));
    const a = [_]f32{ 1.0, 2.0, nan, 4.0, 5.0 };
    var work: [5]f32 = undefined;
    const result = nanquantile_f32(&a, 5, 0.5, &work);
    try testing.expectApproxEqAbs(result, 3.0, 1e-5);
}

test "nanquantile_strided_f64 basic" {
    const testing = @import("std").testing;
    const nan = @as(f64, @bitCast(@as(u64, 0x7FF8000000000000)));
    // 2x3 array: [[1, nan, 3], [nan, 5, 6]], axis=1 => outer=2, axis_size=3, inner=1
    const a = [_]f64{ 1.0, nan, 3.0, nan, 5.0, 6.0 };
    var out: [2]f64 = undefined;
    var work: [3]f64 = undefined;
    nanquantile_strided_f64(&a, &out, 2, 3, 1, 0.5, &work);
    try testing.expectApproxEqAbs(out[0], 2.0, 1e-10); // median of [1, 3]
    try testing.expectApproxEqAbs(out[1], 5.5, 1e-10); // median of [5, 6]
}

test "nanquantile_strided_f32 basic" {
    const testing = @import("std").testing;
    const nan = @as(f32, @bitCast(@as(u32, 0x7FC00000)));
    // 2x3 array: [[1, nan, 3], [nan, 5, 6]], reduce along axis_size=3
    const a = [_]f32{ 1.0, nan, 3.0, nan, 5.0, 6.0 };
    var out: [2]f64 = undefined;
    var work: [3]f32 = undefined;
    nanquantile_strided_f32(&a, &out, 2, 3, 1, 0.5, &work);
    try testing.expectApproxEqAbs(out[0], 2.0, 1e-5); // median of [1, 3]
    try testing.expectApproxEqAbs(out[1], 5.5, 1e-5); // median of [5, 6]
}

test "nanquantile_strided_f32 all NaN row outputs NaN" {
    const testing = @import("std").testing;
    const nan = @as(f32, @bitCast(@as(u32, 0x7FC00000)));
    // 2x2: first row all NaN, second row valid → out[0] = NaN, out[1] = median
    const a = [_]f32{ nan, nan, 4.0, 8.0 };
    var out: [2]f64 = undefined;
    var work: [2]f32 = undefined;
    nanquantile_strided_f32(&a, &out, 2, 2, 1, 0.5, &work);
    try testing.expect(out[0] != out[0]); // NaN
    try testing.expectApproxEqAbs(out[1], 6.0, 1e-5);
}
