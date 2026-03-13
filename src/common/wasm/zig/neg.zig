//! WASM element-wise negation kernels for all numeric types.
//!
//! Unary: out[i] = -a[i]
//! Operates on contiguous 1D buffers of length N.

const simd = @import("simd.zig");

/// Element-wise negate for f64 using 2-wide SIMD: out[i] = -a[i].
export fn neg_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    const zero: simd.V2f64 = @splat(0);
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, zero - simd.load2_f64(a, i));
    }
    while (i < N) : (i += 1) {
        out[i] = -a[i];
    }
}

/// Element-wise negate for f32 using 4-wide SIMD: out[i] = -a[i].
export fn neg_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    const zero: simd.V4f32 = @splat(0);
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_f32(out, i, zero - simd.load4_f32(a, i));
    }
    while (i < N) : (i += 1) {
        out[i] = -a[i];
    }
}

/// Element-wise negate for complex128 using 2-wide f64 SIMD.
/// N = number of complex elements (each = 2 f64s).
export fn neg_c128(a: [*]const f64, out: [*]f64, N: u32) void {
    const zero: simd.V2f64 = @splat(0);
    const n_f64 = N * 2;
    const n_simd = n_f64 & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, zero - simd.load2_f64(a, i));
    }
    while (i < n_f64) : (i += 1) {
        out[i] = -a[i];
    }
}

/// Element-wise negate for complex64 using 4-wide f32 SIMD.
/// N = number of complex elements (each = 2 f32s).
export fn neg_c64(a: [*]const f32, out: [*]f32, N: u32) void {
    const zero: simd.V4f32 = @splat(0);
    const n_f32 = N * 2;
    const n_simd = n_f32 & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_f32(out, i, zero - simd.load4_f32(a, i));
    }
    while (i < n_f32) : (i += 1) {
        out[i] = -a[i];
    }
}

/// Element-wise negate for i64 using 2-wide SIMD with wrapping arithmetic.
/// Handles both signed (i64) and unsigned (u64).
export fn neg_i64(a: [*]const i64, out: [*]i64, N: u32) void {
    const zero: simd.V2i64 = @splat(0);
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_i64(out, i, zero -% simd.load2_i64(a, i));
    }
    while (i < N) : (i += 1) {
        out[i] = -%a[i];
    }
}

/// Element-wise negate for i32 using 4-wide SIMD with wrapping arithmetic.
/// Handles both signed (i32) and unsigned (u32).
export fn neg_i32(a: [*]const i32, out: [*]i32, N: u32) void {
    const zero: simd.V4i32 = @splat(0);
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_i32(out, i, zero -% simd.load4_i32(a, i));
    }
    while (i < N) : (i += 1) {
        out[i] = -%a[i];
    }
}

/// Element-wise negate for i16 using 8-wide SIMD with wrapping arithmetic.
/// Handles both signed (i16) and unsigned (u16).
export fn neg_i16(a: [*]const i16, out: [*]i16, N: u32) void {
    const zero: simd.V8i16 = @splat(0);
    const n_simd = N & ~@as(u32, 7);
    var i: u32 = 0;
    while (i < n_simd) : (i += 8) {
        simd.store8_i16(out, i, zero -% simd.load8_i16(a, i));
    }
    while (i < N) : (i += 1) {
        out[i] = -%a[i];
    }
}

/// Element-wise negate for i8 using 16-wide SIMD with wrapping arithmetic.
/// Handles both signed (i8) and unsigned (u8).
export fn neg_i8(a: [*]const i8, out: [*]i8, N: u32) void {
    const zero: simd.V16i8 = @splat(0);
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    while (i < n_simd) : (i += 16) {
        simd.store16_i8(out, i, zero -% simd.load16_i8(a, i));
    }
    while (i < N) : (i += 1) {
        out[i] = -%a[i];
    }
}

// --- Tests ---

test "neg_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1, -2, 3 };
    var out: [3]f64 = undefined;
    neg_f64(&a, &out, 3);
    try testing.expectApproxEqAbs(out[0], -1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], -3.0, 1e-10);
}

test "neg_i8 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{ 1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16, 17 };
    var out: [17]i8 = undefined;
    neg_i8(&a, &out, 17);
    try testing.expectEqual(out[0], -1);
    try testing.expectEqual(out[1], 2);
    try testing.expectEqual(out[16], -17);
}

test "neg_c64 basic" {
    const testing = @import("std").testing;
    // -(1+2i) = (-1-2i)
    const a = [_]f32{ 1, 2, 3, 4 };
    var out: [4]f32 = undefined;
    neg_c64(&a, &out, 2);
    try testing.expectApproxEqAbs(out[0], -1.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], -2.0, 1e-5);
    try testing.expectApproxEqAbs(out[2], -3.0, 1e-5);
    try testing.expectApproxEqAbs(out[3], -4.0, 1e-5);
}
