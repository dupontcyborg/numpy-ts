//! WASM reduction product kernels for all numeric types.
//!
//! Reduction: result = product(a[0..N])
//! uint types reuse signed kernels (wrapping multiplication gives same bits).

const simd = @import("simd.zig");

/// Returns the product of f64 elements. Uses 2-wide SIMD accumulation.
export fn reduce_prod_f64(a: [*]const f64, N: u32) f64 {
    var acc: simd.V2f64 = .{ 1, 1 };
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        acc *= simd.load2_f64(a, i);
    }
    var total: f64 = acc[0] * acc[1];
    while (i < N) : (i += 1) {
        total *= a[i];
    }
    return total;
}

/// Returns the product of f32 elements. Uses 4-wide SIMD accumulation.
export fn reduce_prod_f32(a: [*]const f32, N: u32) f32 {
    var acc: simd.V4f32 = .{ 1, 1, 1, 1 };
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        acc *= simd.load4_f32(a, i);
    }
    var total: f32 = acc[0] * acc[1] * acc[2] * acc[3];
    while (i < N) : (i += 1) {
        total *= a[i];
    }
    return total;
}

/// Returns the product of i64 elements. Uses 2-wide SIMD accumulation.
/// Note: i64x2.mul is native on x86 but emulated on ARM NEON.
/// Handles both signed (i64) and unsigned (u64) since wrapping multiplication gives same bits.
export fn reduce_prod_i64(a: [*]const i64, N: u32) i64 {
    var acc: simd.V2i64 = .{ 1, 1 };
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        acc *%= simd.load2_i64(a, i);
    }
    var total: i64 = acc[0] *% acc[1];
    while (i < N) : (i += 1) {
        total *%= a[i];
    }
    return total;
}

/// Returns the product of i32 elements. Uses 4-wide SIMD accumulation.
/// Handles both signed (i32) and unsigned (u32) since wrapping multiplication gives same bits.
export fn reduce_prod_i32(a: [*]const i32, N: u32) i32 {
    var acc: simd.V4i32 = .{ 1, 1, 1, 1 };
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        acc *%= simd.load4_i32(a, i);
    }
    var total: i32 = acc[0] *% acc[1] *% acc[2] *% acc[3];
    while (i < N) : (i += 1) {
        total *%= a[i];
    }
    return total;
}

/// Returns the product of i16 elements. Promotes to i64 via 2-wide SIMD accumulation.
export fn reduce_prod_i16(a: [*]const i16, N: u32) i64 {
    var acc: simd.V2i64 = .{ 1, 1 };
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        // Widen i16 → i64 and multiply
        acc *%= simd.V2i64{ @as(i64, a[i]), @as(i64, a[i + 1]) };
    }
    var total: i64 = acc[0] *% acc[1];
    while (i < N) : (i += 1) {
        total *%= @as(i64, a[i]);
    }
    return total;
}

/// Returns the product of u16 elements. Promotes to u64 via 2-wide SIMD accumulation.
export fn reduce_prod_u16(a: [*]const u16, N: u32) u64 {
    var acc: simd.V2u64 = .{ 1, 1 };
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        acc *%= simd.V2u64{ @as(u64, a[i]), @as(u64, a[i + 1]) };
    }
    var total: u64 = acc[0] *% acc[1];
    while (i < N) : (i += 1) {
        total *%= @as(u64, a[i]);
    }
    return total;
}

/// Returns the product of i8 elements. Promotes to i64 via 2-wide SIMD accumulation.
export fn reduce_prod_i8(a: [*]const i8, N: u32) i64 {
    var acc: simd.V2i64 = .{ 1, 1 };
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        acc *%= simd.V2i64{ @as(i64, a[i]), @as(i64, a[i + 1]) };
    }
    var total: i64 = acc[0] *% acc[1];
    while (i < N) : (i += 1) {
        total *%= @as(i64, a[i]);
    }
    return total;
}

/// Returns the product of u8 elements. Promotes to u64 via 2-wide SIMD accumulation.
export fn reduce_prod_u8(a: [*]const u8, N: u32) u64 {
    var acc: simd.V2u64 = .{ 1, 1 };
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        acc *%= simd.V2u64{ @as(u64, a[i]), @as(u64, a[i + 1]) };
    }
    var total: u64 = acc[0] *% acc[1];
    while (i < N) : (i += 1) {
        total *%= @as(u64, a[i]);
    }
    return total;
}

/// Returns the product of u64 elements in a strided layout.
export fn reduce_prod_strided_f64(a: [*]const f64, out: [*]f64, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| out[ob + i] = a[base + i];
        for (1..A) |ax| {
            for (0..I) |i| out[ob + i] *= a[base + ax * I + i];
        }
    }
}

/// Returns the product of f32 elements in a strided layout.
export fn reduce_prod_strided_f32(a: [*]const f32, out: [*]f32, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| out[ob + i] = a[base + i];
        for (1..A) |ax| {
            for (0..I) |i| out[ob + i] *= a[base + ax * I + i];
        }
    }
}

/// Returns the product of i64 elements in a strided layout.
export fn reduce_prod_strided_i64(a: [*]const i64, out: [*]i64, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| out[ob + i] = a[base + i];
        for (1..A) |ax| {
            for (0..I) |i| out[ob + i] *%= a[base + ax * I + i];
        }
    }
}

/// Returns the product of u64 elements in a strided layout.
export fn reduce_prod_strided_u64(a: [*]const u64, out: [*]u64, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| out[ob + i] = a[base + i];
        for (1..A) |ax| {
            for (0..I) |i| out[ob + i] *%= a[base + ax * I + i];
        }
    }
}

/// Returns the product of i32 elements in a strided layout. Uses 4-wide SIMD accumulation.
export fn reduce_prod_strided_i32(a: [*]const i32, out: [*]i32, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| out[ob + i] = a[base + i];
        for (1..A) |ax| {
            for (0..I) |i| out[ob + i] *%= a[base + ax * I + i];
        }
    }
}

/// Returns the product of u32 elements in a strided layout.
export fn reduce_prod_strided_u32(a: [*]const u32, out: [*]u32, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| out[ob + i] = a[base + i];
        for (1..A) |ax| {
            for (0..I) |i| out[ob + i] *%= a[base + ax * I + i];
        }
    }
}

/// Returns the product of i16 elements in a strided layout.
/// Promotes to i64 output to avoid overflow.
export fn reduce_prod_strided_i16(a: [*]const i16, out: [*]i64, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| out[ob + i] = @as(i64, a[base + i]);
        for (1..A) |ax| {
            for (0..I) |i| out[ob + i] *%= @as(i64, a[base + ax * I + i]);
        }
    }
}

/// Returns the product of u16 elements in a strided layout.
/// Promotes to u64 output to avoid overflow.
export fn reduce_prod_strided_u16(a: [*]const u16, out: [*]u64, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| out[ob + i] = @as(u64, a[base + i]);
        for (1..A) |ax| {
            for (0..I) |i| out[ob + i] *%= @as(u64, a[base + ax * I + i]);
        }
    }
}

/// Returns the product of i8 elements in a strided layout.
/// Promotes to i64 output to avoid overflow.
export fn reduce_prod_strided_i8(a: [*]const i8, out: [*]i64, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| out[ob + i] = @as(i64, a[base + i]);
        for (1..A) |ax| {
            for (0..I) |i| out[ob + i] *%= @as(i64, a[base + ax * I + i]);
        }
    }
}

/// Returns to product of u8 elements in a strided layout.
/// Promotes to u64 output to avoid overflow.
export fn reduce_prod_strided_u8(a: [*]const u8, out: [*]u64, outer: u32, axis: u32, inner: u32) void {
    const O = @as(usize, outer);
    const A = @as(usize, axis);
    const I = @as(usize, inner);
    const S = A * I;
    for (0..O) |o| {
        const base = o * S;
        const ob = o * I;
        for (0..I) |i| out[ob + i] = @as(u64, a[base + i]);
        for (1..A) |ax| {
            for (0..I) |i| out[ob + i] *%= @as(u64, a[base + ax * I + i]);
        }
    }
}

// --- Tests ---

test "reduce_prod_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    try testing.expectApproxEqAbs(reduce_prod_f64(&a, 5), 120.0, 1e-10);
}

test "reduce_prod_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 1, 2, 3, 4, 5 };
    try testing.expectEqual(reduce_prod_i32(&a, 5), 120);
}

test "reduce_prod_f64 single and empty" {
    const testing = @import("std").testing;
    const a = [_]f64{7.0};
    try testing.expectApproxEqAbs(reduce_prod_f64(&a, 1), 7.0, 1e-10);
    const b = [_]f64{};
    try testing.expectApproxEqAbs(reduce_prod_f64(&b, 0), 1.0, 1e-10);
}

test "reduce_prod_f64 contains zero" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1.0, 2.0, 0.0, 4.0 };
    try testing.expectApproxEqAbs(reduce_prod_f64(&a, 4), 0.0, 1e-10);
}

test "reduce_prod_f64 negatives" {
    const testing = @import("std").testing;
    const a = [_]f64{ -2.0, 3.0, -4.0 };
    try testing.expectApproxEqAbs(reduce_prod_f64(&a, 3), 24.0, 1e-10);
}

test "reduce_prod_f32 simd boundary" {
    // 4-wide SIMD: N=4, N=5
    const testing = @import("std").testing;
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    try testing.expectApproxEqAbs(reduce_prod_f32(&a, 4), 24.0, 1e-5);
    try testing.expectApproxEqAbs(reduce_prod_f32(&a, 5), 120.0, 1e-5);
}

test "reduce_prod_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 2, 3, 4 };
    try testing.expectEqual(reduce_prod_i64(&a, 3), 24);
}

test "reduce_prod_i64 negatives" {
    const testing = @import("std").testing;
    const a = [_]i64{ -2, -3 };
    try testing.expectEqual(reduce_prod_i64(&a, 2), 6);
}

test "reduce_prod_i16 promotion" {
    // i16 → i64 output to avoid overflow
    const testing = @import("std").testing;
    const a = [_]i16{ 10, 20, 30 };
    try testing.expectEqual(reduce_prod_i16(&a, 3), 6000);
}

test "reduce_prod_i8 promotion" {
    // i8 → i64 output to avoid overflow
    const testing = @import("std").testing;
    const a = [_]i8{ 2, 3, 4, 5 };
    try testing.expectEqual(reduce_prod_i8(&a, 4), 120);
}

test "reduce_prod_u16 promotion" {
    const testing = @import("std").testing;
    const a = [_]u16{ 10, 20, 30 };
    try testing.expectEqual(reduce_prod_u16(&a, 3), 6000);
}

test "reduce_prod_u8 promotion" {
    const testing = @import("std").testing;
    const a = [_]u8{ 2, 3, 4, 5 };
    try testing.expectEqual(reduce_prod_u8(&a, 4), 120);
}
