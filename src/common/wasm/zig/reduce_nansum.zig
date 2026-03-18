//! WASM reduction nansum kernels for float types only.
//!
//! Reduction: result = sum(a[i] for i where !isnan(a[i]))
//! Int/uint types have no NaN — TS routes to regular sum.

/// Nansum f64 array, scalar (NaN check prevents SIMD vectorization).
export fn reduce_nansum_f64(a: [*]const f64, N: u32) f64 {
    var total: f64 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const v = a[i];
        if (v == v) total += v; // NaN != NaN
    }
    return total;
}

/// Nansum f32 array.
export fn reduce_nansum_f32(a: [*]const f32, N: u32) f32 {
    var total: f32 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const v = a[i];
        if (v == v) total += v;
    }
    return total;
}

// --- Tests ---

test "reduce_nansum_f64 basic" {
    const testing = @import("std").testing;
    const nan = @as(f64, @bitCast(@as(u64, 0x7FF8000000000000)));
    const a = [_]f64{ 1.0, nan, 3.0, nan, 5.0 };
    try testing.expectApproxEqAbs(reduce_nansum_f64(&a, 5), 9.0, 1e-10);
}

test "reduce_nansum_f64 all NaN is zero" {
    const testing = @import("std").testing;
    const nan = @as(f64, @bitCast(@as(u64, 0x7FF8000000000000)));
    const a = [_]f64{ nan, nan, nan };
    try testing.expectApproxEqAbs(reduce_nansum_f64(&a, 3), 0.0, 1e-10);
}

test "reduce_nansum_f64 no NaN same as sum" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1.0, 2.0, 3.0 };
    try testing.expectApproxEqAbs(reduce_nansum_f64(&a, 3), 6.0, 1e-10);
}

test "reduce_nansum_f64 NaN at start and end" {
    const testing = @import("std").testing;
    const nan = @as(f64, @bitCast(@as(u64, 0x7FF8000000000000)));
    const a = [_]f64{ nan, 5.0, 10.0, nan };
    try testing.expectApproxEqAbs(reduce_nansum_f64(&a, 4), 15.0, 1e-10);
}

test "reduce_nansum_f32 basic" {
    const testing = @import("std").testing;
    const nan = @as(f32, @bitCast(@as(u32, 0x7FC00000)));
    const a = [_]f32{ 2.0, nan, 3.0 };
    try testing.expectApproxEqAbs(reduce_nansum_f32(&a, 3), 5.0, 1e-5);
}

test "reduce_nansum_f32 all NaN is zero" {
    const testing = @import("std").testing;
    const nan = @as(f32, @bitCast(@as(u32, 0x7FC00000)));
    const a = [_]f32{ nan, nan };
    try testing.expectApproxEqAbs(reduce_nansum_f32(&a, 2), 0.0, 1e-5);
}

test "reduce_nansum_f64 empty" {
    const testing = @import("std").testing;
    const a = [_]f64{};
    try testing.expectApproxEqAbs(reduce_nansum_f64(&a, 0), 0.0, 1e-10);
}
