//! WASM element-wise minimum kernels for all numeric types.
//!
//! Binary: out[i] = min(a[i], b[i])  (propagates NaN, like np.minimum)
//! Scalar: out[i] = min(a[i], scalar)
//! Both operate on contiguous 1D buffers of length N.

const simd = @import("simd.zig");

/// Element-wise minimum for f64 using 2-wide SIMD: out[i] = min(a[i], b[i]).
export fn min_f64(a: [*]const f64, b: [*]const f64, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, simd.min_f64x2(simd.load2_f64(a, i), simd.load2_f64(b, i)));
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < b[i]) a[i] else b[i];
    }
}

/// Element-wise minimum with scalar for f64 using 2-wide SIMD.
export fn min_scalar_f64(a: [*]const f64, out: [*]f64, N: u32, scalar: f64) void {
    const s: simd.V2f64 = @splat(scalar);
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, simd.min_f64x2(simd.load2_f64(a, i), s));
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < scalar) a[i] else scalar;
    }
}

/// Element-wise minimum for f32 using 4-wide SIMD: out[i] = min(a[i], b[i]).
export fn min_f32(a: [*]const f32, b: [*]const f32, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_f32(out, i, simd.min_f32x4(simd.load4_f32(a, i), simd.load4_f32(b, i)));
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < b[i]) a[i] else b[i];
    }
}

/// Element-wise minimum with scalar for f32 using 4-wide SIMD.
export fn min_scalar_f32(a: [*]const f32, out: [*]f32, N: u32, scalar: f32) void {
    const s: simd.V4f32 = @splat(scalar);
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_f32(out, i, simd.min_f32x4(simd.load4_f32(a, i), s));
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < scalar) a[i] else scalar;
    }
}

/// Element-wise minimum for i64, scalar loop (no i64x2 compare in WASM SIMD).
export fn min_i64(a: [*]const i64, b: [*]const i64, out: [*]i64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < b[i]) a[i] else b[i];
    }
}

/// Element-wise minimum with scalar for i64, scalar loop.
export fn min_scalar_i64(a: [*]const i64, out: [*]i64, N: u32, scalar: i64) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < scalar) a[i] else scalar;
    }
}

/// Element-wise minimum for u64, scalar loop (no u64x2 compare in WASM SIMD).
export fn min_u64(a: [*]const u64, b: [*]const u64, out: [*]u64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < b[i]) a[i] else b[i];
    }
}

/// Element-wise minimum with scalar for u64, scalar loop.
export fn min_scalar_u64(a: [*]const u64, out: [*]u64, N: u32, scalar: u64) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < scalar) a[i] else scalar;
    }
}

/// Element-wise minimum for i32 using 4-wide SIMD with compare+select.
export fn min_i32(a: [*]const i32, b: [*]const i32, out: [*]i32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        const va = simd.load4_i32(a, i);
        const vb = simd.load4_i32(b, i);
        simd.store4_i32(out, i, @select(i32, va < vb, va, vb));
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < b[i]) a[i] else b[i];
    }
}

/// Element-wise minimum with scalar for i32 using 4-wide SIMD.
export fn min_scalar_i32(a: [*]const i32, out: [*]i32, N: u32, scalar: i32) void {
    const s: simd.V4i32 = @splat(scalar);
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        const va = simd.load4_i32(a, i);
        simd.store4_i32(out, i, @select(i32, va < s, va, s));
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < scalar) a[i] else scalar;
    }
}

/// Element-wise minimum for u32 using 4-wide SIMD with unsigned compare+select.
export fn min_u32(a: [*]const u32, b: [*]const u32, out: [*]u32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        const va = simd.load4_u32(a, i);
        const vb = simd.load4_u32(b, i);
        simd.store4_u32(out, i, @select(u32, va < vb, va, vb));
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < b[i]) a[i] else b[i];
    }
}

/// Element-wise minimum with scalar for u32 using 4-wide SIMD.
export fn min_scalar_u32(a: [*]const u32, out: [*]u32, N: u32, scalar: u32) void {
    const s: simd.V4u32 = @splat(scalar);
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        const va = simd.load4_u32(a, i);
        simd.store4_u32(out, i, @select(u32, va < s, va, s));
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < scalar) a[i] else scalar;
    }
}

/// Element-wise minimum for i16 using 8-wide SIMD with compare+select.
export fn min_i16(a: [*]const i16, b: [*]const i16, out: [*]i16, N: u32) void {
    const n_simd = N & ~@as(u32, 7);
    var i: u32 = 0;
    while (i < n_simd) : (i += 8) {
        const va = simd.load8_i16(a, i);
        const vb = simd.load8_i16(b, i);
        simd.store8_i16(out, i, @select(i16, va < vb, va, vb));
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < b[i]) a[i] else b[i];
    }
}

/// Element-wise minimum with scalar for i16 using 8-wide SIMD.
export fn min_scalar_i16(a: [*]const i16, out: [*]i16, N: u32, scalar: i16) void {
    const s: simd.V8i16 = @splat(scalar);
    const n_simd = N & ~@as(u32, 7);
    var i: u32 = 0;
    while (i < n_simd) : (i += 8) {
        const va = simd.load8_i16(a, i);
        simd.store8_i16(out, i, @select(i16, va < s, va, s));
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < scalar) a[i] else scalar;
    }
}

/// Element-wise minimum for u16 using 8-wide SIMD with unsigned compare+select.
export fn min_u16(a: [*]const u16, b: [*]const u16, out: [*]u16, N: u32) void {
    const n_simd = N & ~@as(u32, 7);
    var i: u32 = 0;
    while (i < n_simd) : (i += 8) {
        const va = simd.load8_u16(a, i);
        const vb = simd.load8_u16(b, i);
        simd.store8_u16(out, i, @select(u16, va < vb, va, vb));
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < b[i]) a[i] else b[i];
    }
}

/// Element-wise minimum with scalar for u16 using 8-wide SIMD.
export fn min_scalar_u16(a: [*]const u16, out: [*]u16, N: u32, scalar: u16) void {
    const s: simd.V8u16 = @splat(scalar);
    const n_simd = N & ~@as(u32, 7);
    var i: u32 = 0;
    while (i < n_simd) : (i += 8) {
        const va = simd.load8_u16(a, i);
        simd.store8_u16(out, i, @select(u16, va < s, va, s));
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < scalar) a[i] else scalar;
    }
}

/// Element-wise minimum for i8 using 16-wide SIMD with compare+select.
export fn min_i8(a: [*]const i8, b: [*]const i8, out: [*]i8, N: u32) void {
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    while (i < n_simd) : (i += 16) {
        const va = simd.load16_i8(a, i);
        const vb = simd.load16_i8(b, i);
        simd.store16_i8(out, i, @select(i8, va < vb, va, vb));
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < b[i]) a[i] else b[i];
    }
}

/// Element-wise minimum with scalar for i8 using 16-wide SIMD.
export fn min_scalar_i8(a: [*]const i8, out: [*]i8, N: u32, scalar: i8) void {
    const s: simd.V16i8 = @splat(scalar);
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    while (i < n_simd) : (i += 16) {
        const va = simd.load16_i8(a, i);
        simd.store16_i8(out, i, @select(i8, va < s, va, s));
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < scalar) a[i] else scalar;
    }
}

/// Element-wise minimum for u8 using 16-wide SIMD with unsigned compare+select.
export fn min_u8(a: [*]const u8, b: [*]const u8, out: [*]u8, N: u32) void {
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    while (i < n_simd) : (i += 16) {
        const va = simd.load16_u8(a, i);
        const vb = simd.load16_u8(b, i);
        simd.store16_u8(out, i, @select(u8, va < vb, va, vb));
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < b[i]) a[i] else b[i];
    }
}

/// Element-wise minimum with scalar for u8 using 16-wide SIMD.
export fn min_scalar_u8(a: [*]const u8, out: [*]u8, N: u32, scalar: u8) void {
    const s: simd.V16u8 = @splat(scalar);
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    while (i < n_simd) : (i += 16) {
        const va = simd.load16_u8(a, i);
        simd.store16_u8(out, i, @select(u8, va < s, va, s));
    }
    while (i < N) : (i += 1) {
        out[i] = if (a[i] < scalar) a[i] else scalar;
    }
}

// --- Tests ---

test "min_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1, 5, 3 };
    const b = [_]f64{ 2, 4, 6 };
    var out: [3]f64 = undefined;
    min_f64(&a, &b, &out, 3);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 4.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 3.0, 1e-10);
}

test "min_i8 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{ 1, 5, 3, -2, 7, 0, 8, -1, 9, 10, -5, 4, 6, -3, 2, 11, 12 };
    const b = [_]i8{ 2, 4, 6, -1, 3, 1, 7, 0, 8, 5, -4, 3, 7, -2, 1, 10, 13 };
    var out: [17]i8 = undefined;
    min_i8(&a, &b, &out, 17);
    try testing.expectEqual(out[0], 1);
    try testing.expectEqual(out[1], 4);
    try testing.expectEqual(out[3], -2);
}

test "min_u8 unsigned values above 127" {
    const testing = @import("std").testing;
    const a = [_]u8{ 200, 100, 255, 0, 128, 1, 254, 50, 130, 140, 150, 160, 170, 180, 190, 210, 220 };
    const b = [_]u8{ 100, 200, 128, 1, 255, 0, 50, 254, 140, 130, 160, 150, 180, 170, 210, 190, 230 };
    var out: [17]u8 = undefined;
    min_u8(&a, &b, &out, 17);
    try testing.expectEqual(out[0], 100); // 200 vs 100 → 100
    try testing.expectEqual(out[1], 100); // 100 vs 200 → 100
    try testing.expectEqual(out[2], 128); // 255 vs 128 → 128
    try testing.expectEqual(out[4], 128); // 128 vs 255 → 128
}
