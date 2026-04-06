//! WASM reduction nanmax kernels for float types only.
//!
//! Reduction: result = max(a[i] for i where !isnan(a[i]))
//! Int/uint types have no NaN — TS routes to regular max.
//!
//! Uses SIMD + @select for branchless NaN-safe max.

const simd = @import("simd.zig");

const f64_neg_inf: f64 = -@as(f64, @bitCast(@as(u64, 0x7FF0000000000000)));
const f32_neg_inf: f32 = -@as(f32, @bitCast(@as(u32, 0x7F800000)));

/// Nanmax f64 array. Returns -inf if all NaN.
export fn reduce_nanmax_f64(a: [*]const f64, N: u32) f64 {
    if (N == 0) return f64_neg_inf;
    const n_simd = N & ~@as(u32, 1);
    var acc: simd.V2f64 = @splat(f64_neg_inf);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const v = simd.load2_f64(a, i);
        // NaN-safe: acc starts at -inf (never NaN), so v > acc is false when v is NaN.
        acc = @select(f64, v > acc, v, acc);
    }
    var result: f64 = if (acc[0] > acc[1]) acc[0] else acc[1];
    while (i < N) : (i += 1) {
        const v = a[i];
        if (v > result) result = v;
    }
    return result;
}

/// Nanmax f32 array. Returns -inf if all NaN.
export fn reduce_nanmax_f32(a: [*]const f32, N: u32) f32 {
    if (N == 0) return f32_neg_inf;
    const n_simd = N & ~@as(u32, 3);
    var acc: simd.V4f32 = @splat(f32_neg_inf);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        const v = simd.load4_f32(a, i);
        acc = @select(f32, v > acc, v, acc);
    }
    var result: f32 = acc[0];
    inline for (1..4) |lane| {
        if (acc[lane] > result) result = acc[lane];
    }
    while (i < N) : (i += 1) {
        const v = a[i];
        if (v > result) result = v;
    }
    return result;
}

// --- Tests ---

test "reduce_nanmax_f64 basic" {
    const testing = @import("std").testing;
    const nan = @as(f64, @bitCast(@as(u64, 0x7FF8000000000000)));
    const a = [_]f64{ 5.0, nan, 1.0, nan, 3.0 };
    try testing.expectApproxEqAbs(reduce_nanmax_f64(&a, 5), 5.0, 1e-10);
}

test "reduce_nanmax_f64 all NaN returns -inf" {
    const testing = @import("std").testing;
    const nan = @as(f64, @bitCast(@as(u64, 0x7FF8000000000000)));
    const a = [_]f64{ nan, nan };
    const result = reduce_nanmax_f64(&a, 2);
    try testing.expectEqual(result, f64_neg_inf);
}

test "reduce_nanmax_f64 no NaN same as max" {
    const testing = @import("std").testing;
    const a = [_]f64{ 3.0, 1.0, 5.0, 2.0 };
    try testing.expectApproxEqAbs(reduce_nanmax_f64(&a, 4), 5.0, 1e-10);
}

test "reduce_nanmax_f64 negatives" {
    const testing = @import("std").testing;
    const nan = @as(f64, @bitCast(@as(u64, 0x7FF8000000000000)));
    const a = [_]f64{ -1.0, nan, -5.0, -3.0 };
    try testing.expectApproxEqAbs(reduce_nanmax_f64(&a, 4), -1.0, 1e-10);
}

test "reduce_nanmax_f32 basic" {
    const testing = @import("std").testing;
    const nan = @as(f32, @bitCast(@as(u32, 0x7FC00000)));
    const a = [_]f32{ 7.0, nan, 2.0, nan };
    try testing.expectApproxEqAbs(reduce_nanmax_f32(&a, 4), 7.0, 1e-5);
}

test "reduce_nanmax_f32 all NaN returns -inf" {
    const testing = @import("std").testing;
    const nan = @as(f32, @bitCast(@as(u32, 0x7FC00000)));
    const a = [_]f32{ nan, nan };
    const result = reduce_nanmax_f32(&a, 2);
    try testing.expectEqual(result, f32_neg_inf);
}
