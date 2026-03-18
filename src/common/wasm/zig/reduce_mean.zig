//! WASM reduction mean kernels for all numeric types.
//!
//! Reduction: result = sum(a[0..N]) / N
//! Always returns f64 (matches NumPy behavior).
//! Unsigned variants needed for correct float conversion.

const simd = @import("simd.zig");

/// Returns the mean of f64 elements.
export fn reduce_mean_f64(a: [*]const f64, N: u32) f64 {
    var acc: simd.V2f64 = .{ 0, 0 };
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        acc += simd.load2_f64(a, i);
    }
    var total: f64 = acc[0] + acc[1];
    while (i < N) : (i += 1) {
        total += a[i];
    }
    return total / @as(f64, @floatFromInt(N));
}

/// Returns the mean of f32 elements (upcast to f64 for accumulation).
export fn reduce_mean_f32(a: [*]const f32, N: u32) f64 {
    var total: f64 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        total += @as(f64, a[i]);
    }
    return total / @as(f64, @floatFromInt(N));
}

/// Returns the mean of i64 elements (upcast to f64 for accumulation).
export fn reduce_mean_i64(a: [*]const i64, N: u32) f64 {
    var total: f64 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        total += @as(f64, @floatFromInt(a[i]));
    }
    return total / @as(f64, @floatFromInt(N));
}

/// Returns the mean of u64 elements (upcast to f64 for accumulation).
export fn reduce_mean_u64(a: [*]const u64, N: u32) f64 {
    var total: f64 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        total += @as(f64, @floatFromInt(a[i]));
    }
    return total / @as(f64, @floatFromInt(N));
}

/// Returns the mean of i32 elements (upcast to f64 for accumulation).
export fn reduce_mean_i32(a: [*]const i32, N: u32) f64 {
    var total: f64 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        total += @as(f64, @floatFromInt(a[i]));
    }
    return total / @as(f64, @floatFromInt(N));
}

/// Returns the mean of u32 elements (upcast to f64 for accumulation).
export fn reduce_mean_u32(a: [*]const u32, N: u32) f64 {
    var total: f64 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        total += @as(f64, @floatFromInt(a[i]));
    }
    return total / @as(f64, @floatFromInt(N));
}

/// Returns the mean of i16 elements (upcast to f64 for accumulation).
export fn reduce_mean_i16(a: [*]const i16, N: u32) f64 {
    var total: f64 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        total += @as(f64, @floatFromInt(a[i]));
    }
    return total / @as(f64, @floatFromInt(N));
}

/// Returns the mean of u16 elements (upcast to f64 for accumulation).
export fn reduce_mean_u16(a: [*]const u16, N: u32) f64 {
    var total: f64 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        total += @as(f64, @floatFromInt(a[i]));
    }
    return total / @as(f64, @floatFromInt(N));
}

/// Returns the mean of i8 elements (upcast to f64 for accumulation).
export fn reduce_mean_i8(a: [*]const i8, N: u32) f64 {
    var total: f64 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        total += @as(f64, @floatFromInt(a[i]));
    }
    return total / @as(f64, @floatFromInt(N));
}

/// Returns the mean of u8 elements (upcast to f64 for accumulation).
export fn reduce_mean_u8(a: [*]const u8, N: u32) f64 {
    var total: f64 = 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        total += @as(f64, @floatFromInt(a[i]));
    }
    return total / @as(f64, @floatFromInt(N));
}

/// Returns the mean of f64 elements with strided access (for multi-dimensional reduction).
export fn reduce_mean_strided_f64(a: [*]const f64, out: [*]f64, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    const scale = 1.0 / @as(f64, @floatFromInt(A));
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| out[ob + i] = a[base + i];
        for (1..A) |ax| {
            const src = a + base + ax * I;
            const dst = out + ob;
            const n = I & ~@as(usize, 1);
            var j: usize = 0;
            while (j < n) : (j += 2) simd.store2_f64(dst, j, simd.load2_f64(dst, j) + simd.load2_f64(src, j));
            while (j < I) : (j += 1) dst[j] += src[j];
        }
        for (0..I) |i| out[ob + i] *= scale;
    }
}

/// Returns the mean of f32 elements with strided access (upcast to f64 for accumulation).
export fn reduce_mean_strided_f32(a: [*]const f32, out: [*]f64, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    const scale = 1.0 / @as(f64, @floatFromInt(A));
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| out[ob + i] = @as(f64, a[base + i]);
        for (1..A) |ax| {
            for (0..I) |i| out[ob + i] += @as(f64, a[base + ax * I + i]);
        }
        for (0..I) |i| out[ob + i] *= scale;
    }
}

/// Returns the mean of i64 elements with strided access (upcast to f64 for accumulation).
export fn reduce_mean_strided_i64(a: [*]const i64, out: [*]f64, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    const scale = 1.0 / @as(f64, @floatFromInt(A));
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| out[ob + i] = @as(f64, @floatFromInt(a[base + i]));
        for (1..A) |ax| {
            for (0..I) |i| out[ob + i] += @as(f64, @floatFromInt(a[base + ax * I + i]));
        }
        for (0..I) |i| out[ob + i] *= scale;
    }
}

/// Returns the mean of u64 elements with strided access (upcast to f64 for accumulation).
export fn reduce_mean_strided_u64(a: [*]const u64, out: [*]f64, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    const scale = 1.0 / @as(f64, @floatFromInt(A));
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| out[ob + i] = @as(f64, @floatFromInt(a[base + i]));
        for (1..A) |ax| {
            for (0..I) |i| out[ob + i] += @as(f64, @floatFromInt(a[base + ax * I + i]));
        }
        for (0..I) |i| out[ob + i] *= scale;
    }
}

/// Returns the mean of i32 elements with strided access (upcast to f64 for accumulation).
export fn reduce_mean_strided_i32(a: [*]const i32, out: [*]f64, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    const scale = 1.0 / @as(f64, @floatFromInt(A));
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| out[ob + i] = @as(f64, @floatFromInt(a[base + i]));
        for (1..A) |ax| {
            for (0..I) |i| out[ob + i] += @as(f64, @floatFromInt(a[base + ax * I + i]));
        }
        for (0..I) |i| out[ob + i] *= scale;
    }
}

/// Returns the mean of u32 elements with strided access (upcast to f64 for accumulation).
export fn reduce_mean_strided_u32(a: [*]const u32, out: [*]f64, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    const scale = 1.0 / @as(f64, @floatFromInt(A));
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| out[ob + i] = @as(f64, @floatFromInt(a[base + i]));
        for (1..A) |ax| {
            for (0..I) |i| out[ob + i] += @as(f64, @floatFromInt(a[base + ax * I + i]));
        }
        for (0..I) |i| out[ob + i] *= scale;
    }
}

/// Returns the mean of i16 elements with strided access (upcast to f64 for accumulation).
export fn reduce_mean_strided_i16(a: [*]const i16, out: [*]f64, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    const scale = 1.0 / @as(f64, @floatFromInt(A));
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| out[ob + i] = @as(f64, @floatFromInt(a[base + i]));
        for (1..A) |ax| {
            for (0..I) |i| out[ob + i] += @as(f64, @floatFromInt(a[base + ax * I + i]));
        }
        for (0..I) |i| out[ob + i] *= scale;
    }
}

/// Returns the mean of u16 elements with strided access (upcast to f64 for accumulation).
export fn reduce_mean_strided_u16(a: [*]const u16, out: [*]f64, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    const scale = 1.0 / @as(f64, @floatFromInt(A));
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| out[ob + i] = @as(f64, @floatFromInt(a[base + i]));
        for (1..A) |ax| {
            for (0..I) |i| out[ob + i] += @as(f64, @floatFromInt(a[base + ax * I + i]));
        }
        for (0..I) |i| out[ob + i] *= scale;
    }
}

/// Returns the mean of i8 elements with strided access (upcast to f64 for accumulation).
export fn reduce_mean_strided_i8(a: [*]const i8, out: [*]f64, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    const scale = 1.0 / @as(f64, @floatFromInt(A));
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| out[ob + i] = @as(f64, @floatFromInt(a[base + i]));
        for (1..A) |ax| {
            for (0..I) |i| out[ob + i] += @as(f64, @floatFromInt(a[base + ax * I + i]));
        }
        for (0..I) |i| out[ob + i] *= scale;
    }
}

/// Returns the mean of u8 elements with strided access (upcast to f64 for accumulation).
export fn reduce_mean_strided_u8(a: [*]const u8, out: [*]f64, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    const scale = 1.0 / @as(f64, @floatFromInt(A));
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| out[ob + i] = @as(f64, @floatFromInt(a[base + i]));
        for (1..A) |ax| {
            for (0..I) |i| out[ob + i] += @as(f64, @floatFromInt(a[base + ax * I + i]));
        }
        for (0..I) |i| out[ob + i] *= scale;
    }
}

// --- Tests ---

test "reduce_mean_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    try testing.expectApproxEqAbs(reduce_mean_f64(&a, 5), 3.0, 1e-10);
}

test "reduce_mean_u8 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{ 10, 20, 30, 40, 50 };
    try testing.expectApproxEqAbs(reduce_mean_u8(&a, 5), 30.0, 1e-10);
}

test "reduce_mean_f64 single element" {
    const testing = @import("std").testing;
    const a = [_]f64{42.0};
    try testing.expectApproxEqAbs(reduce_mean_f64(&a, 1), 42.0, 1e-10);
}

test "reduce_mean_f64 negatives" {
    const testing = @import("std").testing;
    const a = [_]f64{ -4.0, 0.0, 4.0 };
    try testing.expectApproxEqAbs(reduce_mean_f64(&a, 3), 0.0, 1e-10);
}

test "reduce_mean_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 2.0, 4.0, 6.0 };
    try testing.expectApproxEqAbs(reduce_mean_f32(&a, 3), 4.0, 1e-6);
}

test "reduce_mean_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 10, 20, 30 };
    try testing.expectApproxEqAbs(reduce_mean_i64(&a, 3), 20.0, 1e-10);
}

test "reduce_mean_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ -6, 0, 6 };
    try testing.expectApproxEqAbs(reduce_mean_i32(&a, 3), 0.0, 1e-10);
}

test "reduce_mean_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{ 1, 3 };
    try testing.expectApproxEqAbs(reduce_mean_i16(&a, 2), 2.0, 1e-10);
}

test "reduce_mean_i8 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{ -10, 10 };
    try testing.expectApproxEqAbs(reduce_mean_i8(&a, 2), 0.0, 1e-10);
}

test "reduce_mean_u64 basic" {
    const testing = @import("std").testing;
    const a = [_]u64{ 100, 200, 300 };
    try testing.expectApproxEqAbs(reduce_mean_u64(&a, 3), 200.0, 1e-10);
}

test "reduce_mean_u32 basic" {
    const testing = @import("std").testing;
    const a = [_]u32{ 0, 10, 20 };
    try testing.expectApproxEqAbs(reduce_mean_u32(&a, 3), 10.0, 1e-10);
}

test "reduce_mean_u16 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{ 4, 8 };
    try testing.expectApproxEqAbs(reduce_mean_u16(&a, 2), 6.0, 1e-10);
}
