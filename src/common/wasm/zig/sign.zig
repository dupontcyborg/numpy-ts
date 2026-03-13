//! WASM element-wise sign kernels for all numeric types.
//!
//! Unary: out[i] = sign(a[i])  (returns -1, 0, or 1)
//! Operates on contiguous 1D buffers of length N.
//! Not defined for complex types.

const simd = @import("simd.zig");

/// Element-wise sign for f64 using 2-wide SIMD: returns -1, 0, or 1.
/// Uses two compare+select passes (positive and negative) summed together.
export fn sign_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    const zero: simd.V2f64 = @splat(0);
    const one: simd.V2f64 = @splat(1);
    const neg_one: simd.V2f64 = @splat(-1);
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const v = simd.load2_f64(a, i);
        const pos = @select(f64, v > zero, one, zero);
        const neg = @select(f64, v < zero, neg_one, zero);
        simd.store2_f64(out, i, pos + neg);
    }
    while (i < N) : (i += 1) {
        const v = a[i];
        out[i] = if (v > 0) 1.0 else if (v < 0) -1.0 else 0.0;
    }
}

/// Element-wise sign for f32 using 4-wide SIMD: returns -1, 0, or 1.
/// Uses two compare+select passes (positive and negative) summed together.
export fn sign_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    const zero: simd.V4f32 = @splat(0);
    const one: simd.V4f32 = @splat(1);
    const neg_one: simd.V4f32 = @splat(-1);
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        const v = simd.load4_f32(a, i);
        const pos = @select(f32, v > zero, one, zero);
        const neg = @select(f32, v < zero, neg_one, zero);
        simd.store4_f32(out, i, pos + neg);
    }
    while (i < N) : (i += 1) {
        const v = a[i];
        out[i] = if (v > 0) @as(f32, 1) else if (v < 0) @as(f32, -1) else 0;
    }
}

/// Element-wise sign for i64, scalar loop (no i64x2 compare in WASM SIMD).
export fn sign_i64(a: [*]const i64, out: [*]i64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const v = a[i];
        out[i] = if (v > 0) 1 else if (v < 0) -1 else 0;
    }
}

/// Element-wise sign for i32 using 4-wide SIMD: returns -1, 0, or 1.
export fn sign_i32(a: [*]const i32, out: [*]i32, N: u32) void {
    const zero: simd.V4i32 = @splat(0);
    const one: simd.V4i32 = @splat(1);
    const neg_one: simd.V4i32 = @splat(-1);
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        const v = simd.load4_i32(a, i);
        const pos = @select(i32, v > zero, one, zero);
        const neg = @select(i32, v < zero, neg_one, zero);
        simd.store4_i32(out, i, pos +% neg);
    }
    while (i < N) : (i += 1) {
        const v = a[i];
        out[i] = if (v > 0) 1 else if (v < 0) -1 else 0;
    }
}

/// Element-wise sign for i16 using 8-wide SIMD: returns -1, 0, or 1.
export fn sign_i16(a: [*]const i16, out: [*]i16, N: u32) void {
    const zero: simd.V8i16 = @splat(0);
    const one: simd.V8i16 = @splat(1);
    const neg_one: simd.V8i16 = @splat(-1);
    const n_simd = N & ~@as(u32, 7);
    var i: u32 = 0;
    while (i < n_simd) : (i += 8) {
        const v = simd.load8_i16(a, i);
        const pos = @select(i16, v > zero, one, zero);
        const neg = @select(i16, v < zero, neg_one, zero);
        simd.store8_i16(out, i, pos +% neg);
    }
    while (i < N) : (i += 1) {
        const v = a[i];
        out[i] = if (v > 0) 1 else if (v < 0) -1 else 0;
    }
}

/// Element-wise sign for i8 using 16-wide SIMD: returns -1, 0, or 1.
export fn sign_i8(a: [*]const i8, out: [*]i8, N: u32) void {
    const zero: simd.V16i8 = @splat(0);
    const one: simd.V16i8 = @splat(1);
    const neg_one: simd.V16i8 = @splat(-1);
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    while (i < n_simd) : (i += 16) {
        const v = simd.load16_i8(a, i);
        const pos = @select(i8, v > zero, one, zero);
        const neg = @select(i8, v < zero, neg_one, zero);
        simd.store16_i8(out, i, pos +% neg);
    }
    while (i < N) : (i += 1) {
        const v = a[i];
        out[i] = if (v > 0) 1 else if (v < 0) -1 else 0;
    }
}

// --- Tests ---

test "sign_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ -5, 0, 7 };
    var out: [3]f64 = undefined;
    sign_f64(&a, &out, 3);
    try testing.expectApproxEqAbs(out[0], -1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 1.0, 1e-10);
}

test "sign_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{ -5, 0, 7, -1, 3, 0, -9, 2, 10 };
    var out: [9]i16 = undefined;
    sign_i16(&a, &out, 9);
    try testing.expectEqual(out[0], -1);
    try testing.expectEqual(out[1], 0);
    try testing.expectEqual(out[2], 1);
}

test "sign_i8 large" {
    const testing = @import("std").testing;
    var a: [20]i8 = undefined;
    for (0..20) |idx| {
        a[idx] = if (idx % 3 == 0) -1 else if (idx % 3 == 1) 0 else 1;
    }
    var out: [20]i8 = undefined;
    sign_i8(&a, &out, 20);
    for (0..20) |idx| {
        try testing.expectEqual(out[idx], a[idx]);
    }
}
