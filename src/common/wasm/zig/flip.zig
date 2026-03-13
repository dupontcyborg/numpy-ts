//! WASM element-wise array reversal kernels for all numeric types.
//!
//! Unary: out[i] = a[N-1-i]  (full reversal of contiguous buffer)
//! Operates on contiguous 1D buffers of length N.

const simd = @import("simd.zig");

/// Element-wise reversal for f64: out[i] = a[N-1-i].
export fn flip_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = a[N - 1 - i];
    }
}

/// Element-wise reversal for f32: out[i] = a[N-1-i].
export fn flip_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = a[N - 1 - i];
    }
}

/// Element-wise reversal for i64, scalar loop (no i64x2 in WASM SIMD).
export fn flip_i64(a: [*]const i64, out: [*]i64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = a[N - 1 - i];
    }
}

/// Element-wise reversal for i32: out[i] = a[N-1-i].
export fn flip_i32(a: [*]const i32, out: [*]i32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = a[N - 1 - i];
    }
}

/// Element-wise reversal for i16: out[i] = a[N-1-i].
export fn flip_i16(a: [*]const i16, out: [*]i16, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = a[N - 1 - i];
    }
}

/// Element-wise reversal for i8: out[i] = a[N-1-i].
export fn flip_i8(a: [*]const i8, out: [*]i8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = a[N - 1 - i];
    }
}

// --- Tests ---

test "flip_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var out: [5]f64 = undefined;
    flip_f64(&a, &out, 5);
    try testing.expectApproxEqAbs(out[0], 5.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 4.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[4], 1.0, 1e-10);
}

test "flip_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 10, 20, 30, 40, 50 };
    var out: [5]i32 = undefined;
    flip_i32(&a, &out, 5);
    try testing.expectEqual(out[0], 50);
    try testing.expectEqual(out[1], 40);
    try testing.expectEqual(out[2], 30);
    try testing.expectEqual(out[3], 20);
    try testing.expectEqual(out[4], 10);
}

test "flip_i8 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 };
    var out: [17]i8 = undefined;
    flip_i8(&a, &out, 17);
    try testing.expectEqual(out[0], 17);
    try testing.expectEqual(out[16], 1);
    try testing.expectEqual(out[8], 9);
}
