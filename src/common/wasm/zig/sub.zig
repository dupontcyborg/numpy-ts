//! WASM element-wise subtraction kernels for all numeric types.
//!
//! Binary: out[i] = a[i] - b[i]
//! Scalar: out[i] = a[i] - scalar
//! Both operate on contiguous 1D buffers of length N.

const simd = @import("simd.zig");

/// Element-wise subtract for f64 using 2-wide SIMD: out[i] = a[i] - b[i].
export fn sub_f64(a: [*]const f64, b: [*]const f64, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, simd.load2_f64(a, i) - simd.load2_f64(b, i));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] - b[i];
    }
}

/// Element-wise subtract scalar for f64 using 2-wide SIMD: out[i] = a[i] - scalar.
export fn sub_scalar_f64(a: [*]const f64, out: [*]f64, N: u32, scalar: f64) void {
    const s: simd.V2f64 = @splat(scalar);
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, simd.load2_f64(a, i) - s);
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] - scalar;
    }
}

/// Element-wise subtract for f32 using 4-wide SIMD: out[i] = a[i] - b[i].
export fn sub_f32(a: [*]const f32, b: [*]const f32, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_f32(out, i, simd.load4_f32(a, i) - simd.load4_f32(b, i));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] - b[i];
    }
}

/// Element-wise subtract scalar for f32 using 4-wide SIMD: out[i] = a[i] - scalar.
export fn sub_scalar_f32(a: [*]const f32, out: [*]f32, N: u32, scalar: f32) void {
    const s: simd.V4f32 = @splat(scalar);
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_f32(out, i, simd.load4_f32(a, i) - s);
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] - scalar;
    }
}

/// Element-wise complex subtract for complex128 using 2-wide f64 SIMD.
/// N = number of complex elements (each = 2 f64s).
export fn sub_c128(a: [*]const f64, b: [*]const f64, out: [*]f64, N: u32) void {
    const n_f64 = N * 2;
    const n_simd = n_f64 & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, simd.load2_f64(a, i) - simd.load2_f64(b, i));
    }
    while (i < n_f64) : (i += 1) {
        out[i] = a[i] - b[i];
    }
}

/// Subtract real scalar from complex128: out[i] = a[i] - scalar (real part only).
export fn sub_scalar_c128(a: [*]const f64, out: [*]f64, N: u32, scalar: f64) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const idx = i * 2;
        out[idx] = a[idx] - scalar;
        out[idx + 1] = a[idx + 1];
    }
}

/// Element-wise complex subtract for complex64 using 4-wide f32 SIMD.
/// N = number of complex elements (each = 2 f32s).
export fn sub_c64(a: [*]const f32, b: [*]const f32, out: [*]f32, N: u32) void {
    const n_f32 = N * 2;
    const n_simd = n_f32 & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_f32(out, i, simd.load4_f32(a, i) - simd.load4_f32(b, i));
    }
    while (i < n_f32) : (i += 1) {
        out[i] = a[i] - b[i];
    }
}

/// Subtract real scalar from complex64: out[i] = a[i] - scalar (real part only).
export fn sub_scalar_c64(a: [*]const f32, out: [*]f32, N: u32, scalar: f32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const idx = i * 2;
        out[idx] = a[idx] - scalar;
        out[idx + 1] = a[idx + 1];
    }
}

/// Element-wise subtract for i64 using 2-wide SIMD with wrapping arithmetic.
/// Handles both signed (i64) and unsigned (u64).
export fn sub_i64(a: [*]const i64, b: [*]const i64, out: [*]i64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_i64(out, i, simd.load2_i64(a, i) -% simd.load2_i64(b, i));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] -% b[i];
    }
}

/// Element-wise subtract scalar for i64 using 2-wide SIMD with wrapping arithmetic.
/// Handles both signed (i64) and unsigned (u64).
export fn sub_scalar_i64(a: [*]const i64, out: [*]i64, N: u32, scalar: i64) void {
    const s: simd.V2i64 = @splat(scalar);
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_i64(out, i, simd.load2_i64(a, i) -% s);
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] -% scalar;
    }
}

/// Element-wise subtract for i32 using 4-wide SIMD with wrapping arithmetic.
/// Handles both signed (i32) and unsigned (u32).
export fn sub_i32(a: [*]const i32, b: [*]const i32, out: [*]i32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_i32(out, i, simd.load4_i32(a, i) -% simd.load4_i32(b, i));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] -% b[i];
    }
}

/// Element-wise subtract scalar for i32 using 4-wide SIMD with wrapping arithmetic.
/// Handles both signed (i32) and unsigned (u32).
export fn sub_scalar_i32(a: [*]const i32, out: [*]i32, N: u32, scalar: i32) void {
    const s: simd.V4i32 = @splat(scalar);
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_i32(out, i, simd.load4_i32(a, i) -% s);
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] -% scalar;
    }
}

/// Element-wise subtract for i16 using 8-wide SIMD with wrapping arithmetic.
/// Handles both signed (i16) and unsigned (u16).
export fn sub_i16(a: [*]const i16, b: [*]const i16, out: [*]i16, N: u32) void {
    const n_simd = N & ~@as(u32, 7);
    var i: u32 = 0;
    while (i < n_simd) : (i += 8) {
        simd.store8_i16(out, i, simd.load8_i16(a, i) -% simd.load8_i16(b, i));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] -% b[i];
    }
}

/// Element-wise subtract scalar for i16 using 8-wide SIMD with wrapping arithmetic.
/// Handles both signed (i16) and unsigned (u16).
export fn sub_scalar_i16(a: [*]const i16, out: [*]i16, N: u32, scalar: i16) void {
    const s: simd.V8i16 = @splat(scalar);
    const n_simd = N & ~@as(u32, 7);
    var i: u32 = 0;
    while (i < n_simd) : (i += 8) {
        simd.store8_i16(out, i, simd.load8_i16(a, i) -% s);
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] -% scalar;
    }
}

/// Element-wise subtract for i8 using 16-wide SIMD with wrapping arithmetic.
/// Uses widened i16 subtract since WASM SIMD has no i8x16.sub.
/// Handles both signed (i8) and unsigned (u8).
export fn sub_i8(a: [*]const i8, b: [*]const i8, out: [*]i8, N: u32) void {
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    while (i < n_simd) : (i += 16) {
        simd.store16_i8(out, i, simd.load16_i8(a, i) -% simd.load16_i8(b, i));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] -% b[i];
    }
}

/// Element-wise subtract scalar for i8 using 16-wide SIMD with wrapping arithmetic.
/// Uses widened i16 subtract since WASM SIMD has no i8x16.sub.
/// Handles both signed (i8) and unsigned (u8).
export fn sub_scalar_i8(a: [*]const i8, out: [*]i8, N: u32, scalar: i8) void {
    const s: simd.V16i8 = @splat(scalar);
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    while (i < n_simd) : (i += 16) {
        simd.store16_i8(out, i, simd.load16_i8(a, i) -% s);
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] -% scalar;
    }
}

// --- Tests ---

test "sub_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 10, 20, 30 };
    const b = [_]f64{ 1, 2, 3 };
    var out: [3]f64 = undefined;
    sub_f64(&a, &b, &out, 3);
    try testing.expectApproxEqAbs(out[0], 9.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 27.0, 1e-10);
}

test "sub_scalar_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 10, 20, 30, 40, 50 };
    var out: [5]f32 = undefined;
    sub_scalar_f32(&a, &out, 5, 5.0);
    try testing.expectApproxEqAbs(out[0], 5.0, 1e-5);
    try testing.expectApproxEqAbs(out[4], 45.0, 1e-5);
}

test "sub_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{ 100, 200, 300, 400, 500, 600, 700, 800, 900 };
    const b = [_]i16{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    var out: [9]i16 = undefined;
    sub_i16(&a, &b, &out, 9);
    try testing.expectEqual(out[0], 99);
    try testing.expectEqual(out[8], 891);
}
