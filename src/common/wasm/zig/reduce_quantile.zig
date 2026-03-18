//! WASM reduction quantile kernel.
//!
//! Computes the q-th quantile of a contiguous f64 array.
//! The TS wrapper converts all dtypes to f64 before calling this kernel.
//! Uses in-place sort then linear interpolation (matching NumPy's default method).

const std = @import("std");

/// Compute the q-th quantile of a contiguous f64 array.
/// The input buffer `a` is modified in-place (sorted).
/// Returns the interpolated quantile value.
export fn reduce_quantile_f64(a: [*]f64, N: u32, q: f64) f64 {
    if (N == 0) return 0;
    if (N == 1) return a[0];

    // Sort in-place using Zig's pdqsort (pattern-defeating quicksort)
    const slice = a[0..@as(usize, N)];
    std.mem.sortUnstable(f64, slice, {}, std.sort.asc(f64));

    // Linear interpolation (NumPy default method='linear')
    const idx = q * @as(f64, @floatFromInt(N - 1));
    const lower = @as(usize, @intFromFloat(@floor(idx)));
    const upper = @as(usize, @intFromFloat(@ceil(idx)));

    if (lower == upper) return a[lower];

    const frac = idx - @as(f64, @floatFromInt(lower));
    return a[lower] * (1.0 - frac) + a[upper] * frac;
}

// --- Tests ---

test "reduce_quantile_f64 median" {
    const testing = std.testing;
    var a = [_]f64{ 5.0, 1.0, 3.0, 2.0, 4.0 };
    try testing.expectApproxEqAbs(reduce_quantile_f64(&a, 5, 0.5), 3.0, 1e-10);
}

test "reduce_quantile_f64 q=0.25" {
    const testing = std.testing;
    var a = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    // q=0.25: idx=1.0 → exactly a[1]=2.0
    try testing.expectApproxEqAbs(reduce_quantile_f64(&a, 5, 0.25), 2.0, 1e-10);
}

test "reduce_quantile_f64 interpolation" {
    const testing = std.testing;
    var a = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    // q=0.5: idx=1.5 → interpolate between 2.0 and 3.0 → 2.5
    try testing.expectApproxEqAbs(reduce_quantile_f64(&a, 4, 0.5), 2.5, 1e-10);
}

test "reduce_quantile_f64 q=0 is min q=1 is max" {
    const testing = std.testing;
    var a = [_]f64{ 3.0, 1.0, 5.0, 2.0, 4.0 };
    try testing.expectApproxEqAbs(reduce_quantile_f64(&a, 5, 0.0), 1.0, 1e-10);
    var b = [_]f64{ 3.0, 1.0, 5.0, 2.0, 4.0 };
    try testing.expectApproxEqAbs(reduce_quantile_f64(&b, 5, 1.0), 5.0, 1e-10);
}

test "reduce_quantile_f64 single element" {
    const testing = std.testing;
    var a = [_]f64{42.0};
    try testing.expectApproxEqAbs(reduce_quantile_f64(&a, 1, 0.5), 42.0, 1e-10);
    try testing.expectApproxEqAbs(reduce_quantile_f64(&a, 1, 0.0), 42.0, 1e-10);
}

test "reduce_quantile_f64 empty returns zero" {
    const testing = std.testing;
    var a = [_]f64{};
    try testing.expectApproxEqAbs(reduce_quantile_f64(&a, 0, 0.5), 0.0, 1e-10);
}

test "reduce_quantile_f64 constant array" {
    const testing = std.testing;
    var a = [_]f64{ 7.0, 7.0, 7.0, 7.0 };
    try testing.expectApproxEqAbs(reduce_quantile_f64(&a, 4, 0.25), 7.0, 1e-10);
    try testing.expectApproxEqAbs(reduce_quantile_f64(&a, 4, 0.75), 7.0, 1e-10);
}

test "reduce_quantile_f64 already sorted" {
    const testing = std.testing;
    var a = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    try testing.expectApproxEqAbs(reduce_quantile_f64(&a, 5, 0.5), 3.0, 1e-10);
}

test "reduce_quantile_f64 negatives" {
    const testing = std.testing;
    var a = [_]f64{ -5.0, -3.0, -1.0 };
    // median of [-5,-3,-1] = -3
    try testing.expectApproxEqAbs(reduce_quantile_f64(&a, 3, 0.5), -3.0, 1e-10);
}

test "reduce_quantile_f64 two elements interpolation" {
    const testing = std.testing;
    // q=0.5: idx=0.5, interpolate between a[0]=1 and a[1]=3 → 2
    var a = [_]f64{ 3.0, 1.0 };
    try testing.expectApproxEqAbs(reduce_quantile_f64(&a, 2, 0.5), 2.0, 1e-10);
}
