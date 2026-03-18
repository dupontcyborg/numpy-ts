//! WASM reduction nanmax kernels for float types only.
//!
//! Reduction: result = max(a[i] for i where !isnan(a[i]))
//! Int/uint types have no NaN — TS routes to regular max.

/// Nanmax f64 array. Returns -inf if all NaN.
export fn reduce_nanmax_f64(a: [*]const f64, N: u32) f64 {
    var result: f64 = -@as(f64, @bitCast(@as(u64, 0x7FF0000000000000))); // -inf
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const v = a[i];
        if (v == v and v > result) result = v;
    }
    return result;
}

/// Nanmax f32 array. Returns -inf if all NaN.
export fn reduce_nanmax_f32(a: [*]const f32, N: u32) f32 {
    var result: f32 = -@as(f32, @bitCast(@as(u32, 0x7F800000))); // -inf
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const v = a[i];
        if (v == v and v > result) result = v;
    }
    return result;
}

// --- Tests ---

test "reduce_nanmax_f64 basic" {
    const testing = @import("std").testing;
    const nan = @as(f64, @bitCast(@as(u64, 0x7FF8000000000000)));
    const a = [_]f64{ 1.0, nan, 5.0, nan, 3.0 };
    try testing.expectApproxEqAbs(reduce_nanmax_f64(&a, 5), 5.0, 1e-10);
}

test "reduce_nanmax_f64 all NaN returns -inf" {
    const testing = @import("std").testing;
    const nan = @as(f64, @bitCast(@as(u64, 0x7FF8000000000000)));
    const a = [_]f64{ nan, nan };
    const result = reduce_nanmax_f64(&a, 2);
    const neg_inf = -@as(f64, @bitCast(@as(u64, 0x7FF0000000000000)));
    try testing.expectEqual(result, neg_inf);
}

test "reduce_nanmax_f64 no NaN same as max" {
    const testing = @import("std").testing;
    const a = [_]f64{ 3.0, 1.0, 5.0, 2.0 };
    try testing.expectApproxEqAbs(reduce_nanmax_f64(&a, 4), 5.0, 1e-10);
}

test "reduce_nanmax_f64 NaN at first and last" {
    const testing = @import("std").testing;
    const nan = @as(f64, @bitCast(@as(u64, 0x7FF8000000000000)));
    const a = [_]f64{ nan, 2.0, 8.0, nan };
    try testing.expectApproxEqAbs(reduce_nanmax_f64(&a, 4), 8.0, 1e-10);
}

test "reduce_nanmax_f64 negatives" {
    const testing = @import("std").testing;
    const nan = @as(f64, @bitCast(@as(u64, 0x7FF8000000000000)));
    const a = [_]f64{ -5.0, nan, -1.0, -3.0 };
    try testing.expectApproxEqAbs(reduce_nanmax_f64(&a, 4), -1.0, 1e-10);
}

test "reduce_nanmax_f32 basic" {
    const testing = @import("std").testing;
    const nan = @as(f32, @bitCast(@as(u32, 0x7FC00000)));
    const a = [_]f32{ 2.0, nan, 7.0, nan };
    try testing.expectApproxEqAbs(reduce_nanmax_f32(&a, 4), 7.0, 1e-5);
}

test "reduce_nanmax_f32 all NaN returns -inf" {
    const testing = @import("std").testing;
    const nan = @as(f32, @bitCast(@as(u32, 0x7FC00000)));
    const a = [_]f32{ nan, nan };
    const result = reduce_nanmax_f32(&a, 2);
    const neg_inf = -@as(f32, @bitCast(@as(u32, 0x7F800000)));
    try testing.expectEqual(result, neg_inf);
}
