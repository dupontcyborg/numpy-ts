//! WASM reduction standard deviation kernels for all numeric types.
//!
//! Reduction: result = sqrt(var(a[0..N])) = sqrt(mean((a - mean(a))^2))
//! Two-pass algorithm: compute mean, then sum of squared diffs, then sqrt.
//! Always returns f64. Unsigned variants needed for correct float conversion.

const std = @import("std");
const math = std.math;
const simd = @import("simd.zig");

/// Computes the standard deviation of f64 elements.
/// Uses 2-wide SIMD for sum and squared diffs.
export fn reduce_std_f64(a: [*]const f64, N: u32) f64 {
    if (N == 0) return 0;
    var sum_acc: simd.V2f64 = .{ 0, 0 };
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        sum_acc += simd.load2_f64(a, i);
    }
    var total: f64 = sum_acc[0] + sum_acc[1];
    while (i < N) : (i += 1) {
        total += a[i];
    }
    const mean = total / @as(f64, @floatFromInt(N));

    const mean_v: simd.V2f64 = @splat(mean);
    var sq_acc: simd.V2f64 = .{ 0, 0 };
    i = 0;
    while (i < n_simd) : (i += 2) {
        const diff = simd.load2_f64(a, i) - mean_v;
        sq_acc += diff * diff;
    }
    var sq_total: f64 = sq_acc[0] + sq_acc[1];
    while (i < N) : (i += 1) {
        const diff = a[i] - mean;
        sq_total += diff * diff;
    }
    return @sqrt(sq_total / @as(f64, @floatFromInt(N)));
}

/// Computes the standard deviation of f32 elements. Promotes to f64 for accuracy.
export fn reduce_std_f32(a: [*]const f32, N: u32) f64 {
    if (N == 0) return 0;
    var total: f64 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) total += @as(f64, a[i]);
    const mean = total / @as(f64, @floatFromInt(N));
    var sq_total: f64 = 0;
    i = 0;
    while (i < N) : (i += 1) {
        const diff = @as(f64, a[i]) - mean;
        sq_total += diff * diff;
    }
    return @sqrt(sq_total / @as(f64, @floatFromInt(N)));
}

/// Computes the standard deviation of i64 elements. Promotes to f64 for accuracy.
export fn reduce_std_i64(a: [*]const i64, N: u32) f64 {
    if (N == 0) return 0;
    var total: f64 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) total += @as(f64, @floatFromInt(a[i]));
    const mean = total / @as(f64, @floatFromInt(N));
    var sq_total: f64 = 0;
    i = 0;
    while (i < N) : (i += 1) {
        const diff = @as(f64, @floatFromInt(a[i])) - mean;
        sq_total += diff * diff;
    }
    return @sqrt(sq_total / @as(f64, @floatFromInt(N)));
}

/// Computes the standard deviation of u64 elements. Promotes to f64 for accuracy.
export fn reduce_std_u64(a: [*]const u64, N: u32) f64 {
    if (N == 0) return 0;
    var total: f64 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) total += @as(f64, @floatFromInt(a[i]));
    const mean = total / @as(f64, @floatFromInt(N));
    var sq_total: f64 = 0;
    i = 0;
    while (i < N) : (i += 1) {
        const diff = @as(f64, @floatFromInt(a[i])) - mean;
        sq_total += diff * diff;
    }
    return @sqrt(sq_total / @as(f64, @floatFromInt(N)));
}

/// Computes the standard deviation of i32 elements. Promotes to f64 for accuracy.
export fn reduce_std_i32(a: [*]const i32, N: u32) f64 {
    if (N == 0) return 0;
    var total: f64 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) total += @as(f64, @floatFromInt(a[i]));
    const mean = total / @as(f64, @floatFromInt(N));
    var sq_total: f64 = 0;
    i = 0;
    while (i < N) : (i += 1) {
        const diff = @as(f64, @floatFromInt(a[i])) - mean;
        sq_total += diff * diff;
    }
    return @sqrt(sq_total / @as(f64, @floatFromInt(N)));
}

/// Computes the standard deviation of u32 elements. Promotes to f64 for accuracy.
export fn reduce_std_u32(a: [*]const u32, N: u32) f64 {
    if (N == 0) return 0;
    var total: f64 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) total += @as(f64, @floatFromInt(a[i]));
    const mean = total / @as(f64, @floatFromInt(N));
    var sq_total: f64 = 0;
    i = 0;
    while (i < N) : (i += 1) {
        const diff = @as(f64, @floatFromInt(a[i])) - mean;
        sq_total += diff * diff;
    }
    return @sqrt(sq_total / @as(f64, @floatFromInt(N)));
}

/// Computes the standard deviation of i16 elements. Promotes to f64 for accuracy.
export fn reduce_std_i16(a: [*]const i16, N: u32) f64 {
    if (N == 0) return 0;
    var total: f64 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) total += @as(f64, @floatFromInt(a[i]));
    const mean = total / @as(f64, @floatFromInt(N));
    var sq_total: f64 = 0;
    i = 0;
    while (i < N) : (i += 1) {
        const diff = @as(f64, @floatFromInt(a[i])) - mean;
        sq_total += diff * diff;
    }
    return @sqrt(sq_total / @as(f64, @floatFromInt(N)));
}

/// Computes the standard deviation of u16 elements. Promotes to f64 for accuracy.
export fn reduce_std_u16(a: [*]const u16, N: u32) f64 {
    if (N == 0) return 0;
    var total: f64 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) total += @as(f64, @floatFromInt(a[i]));
    const mean = total / @as(f64, @floatFromInt(N));
    var sq_total: f64 = 0;
    i = 0;
    while (i < N) : (i += 1) {
        const diff = @as(f64, @floatFromInt(a[i])) - mean;
        sq_total += diff * diff;
    }
    return @sqrt(sq_total / @as(f64, @floatFromInt(N)));
}

/// Computes the standard deviation of i8 elements. Promotes to f64 for accuracy.
export fn reduce_std_i8(a: [*]const i8, N: u32) f64 {
    if (N == 0) return 0;
    var total: f64 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) total += @as(f64, @floatFromInt(a[i]));
    const mean = total / @as(f64, @floatFromInt(N));
    var sq_total: f64 = 0;
    i = 0;
    while (i < N) : (i += 1) {
        const diff = @as(f64, @floatFromInt(a[i])) - mean;
        sq_total += diff * diff;
    }
    return @sqrt(sq_total / @as(f64, @floatFromInt(N)));
}

/// Computes the standard deviation of u8 elements. Promotes to f64 for accuracy.
export fn reduce_std_u8(a: [*]const u8, N: u32) f64 {
    if (N == 0) return 0;
    var total: f64 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) total += @as(f64, @floatFromInt(a[i]));
    const mean = total / @as(f64, @floatFromInt(N));
    var sq_total: f64 = 0;
    i = 0;
    while (i < N) : (i += 1) {
        const diff = @as(f64, @floatFromInt(a[i])) - mean;
        sq_total += diff * diff;
    }
    return @sqrt(sq_total / @as(f64, @floatFromInt(N)));
}

// --- Tests ---

test "reduce_std_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    // std = sqrt(2.0) ≈ 1.4142
    try testing.expectApproxEqAbs(reduce_std_f64(&a, 5), @sqrt(2.0), 1e-10);
}

test "reduce_std_f64 single element is zero" {
    const testing = @import("std").testing;
    const a = [_]f64{42.0};
    try testing.expectApproxEqAbs(reduce_std_f64(&a, 1), 0.0, 1e-10);
}

test "reduce_std_f64 constant array is zero" {
    const testing = @import("std").testing;
    const a = [_]f64{ 3.0, 3.0, 3.0 };
    try testing.expectApproxEqAbs(reduce_std_f64(&a, 3), 0.0, 1e-10);
}

test "reduce_std_f64 empty" {
    const testing = @import("std").testing;
    const a = [_]f64{};
    try testing.expectApproxEqAbs(reduce_std_f64(&a, 0), 0.0, 1e-10);
}

test "reduce_std_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    try testing.expectApproxEqAbs(reduce_std_f32(&a, 5), @sqrt(2.0), 1e-6);
}

test "reduce_std_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 1, 2, 3, 4, 5 };
    try testing.expectApproxEqAbs(reduce_std_i32(&a, 5), @sqrt(2.0), 1e-10);
}

test "reduce_std_i64 negatives" {
    const testing = @import("std").testing;
    // mean=0, std=sqrt(mean(1,1))=1
    const a = [_]i64{ -1, 1 };
    try testing.expectApproxEqAbs(reduce_std_i64(&a, 2), 1.0, 1e-10);
}

test "reduce_std_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{ 10, 20, 30 };
    // mean=20, var=200/3, std=sqrt(200/3)
    try testing.expectApproxEqAbs(reduce_std_i16(&a, 3), @sqrt(200.0 / 3.0), 1e-10);
}

test "reduce_std_i8 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{ 1, 2, 3, 4, 5 };
    try testing.expectApproxEqAbs(reduce_std_i8(&a, 5), @sqrt(2.0), 1e-10);
}

test "reduce_std_u8 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{ 10, 20, 30, 40, 50 };
    try testing.expectApproxEqAbs(reduce_std_u8(&a, 5), @sqrt(200.0), 1e-8);
}

test "reduce_std_u32 constant" {
    const testing = @import("std").testing;
    const a = [_]u32{ 5, 5, 5, 5 };
    try testing.expectApproxEqAbs(reduce_std_u32(&a, 4), 0.0, 1e-10);
}

test "reduce_std_u64 basic" {
    const testing = @import("std").testing;
    // [2, 4] has mean 3, std=1
    const a = [_]u64{ 2, 4 };
    const result = reduce_std_u64(&a, 2);
    try testing.expectApproxEqAbs(result, 1.0, 1e-10);
}

test "reduce_std_u16 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{ 2, 4 };
    const result = reduce_std_u16(&a, 2);
    try testing.expectApproxEqAbs(result, 1.0, 1e-10);
}
