//! WASM element-wise clip (clamp) kernels for all numeric types.
//!
//! Unary: out[i] = clamp(a[i], lo, hi)
//! Operates on contiguous 1D buffers of length N.

const simd = @import("simd.zig");

/// Element-wise clamp for f64 using 2-wide SIMD: out[i] = clamp(a[i], lo, hi).
export fn clip_f64(a: [*]const f64, out: [*]f64, N: u32, lo: f64, hi: f64) void {
    const vlo: simd.V2f64 = @splat(lo);
    const vhi: simd.V2f64 = @splat(hi);
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const v = simd.load2_f64(a, i);
        simd.store2_f64(out, i, simd.min_f64x2(simd.max_f64x2(v, vlo), vhi));
    }
    while (i < N) : (i += 1) {
        const v = a[i];
        out[i] = if (v < lo) lo else if (v > hi) hi else v;
    }
}

/// Element-wise clamp for f32 using 4-wide SIMD: out[i] = clamp(a[i], lo, hi).
export fn clip_f32(a: [*]const f32, out: [*]f32, N: u32, lo: f32, hi: f32) void {
    const vlo: simd.V4f32 = @splat(lo);
    const vhi: simd.V4f32 = @splat(hi);
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        const v = simd.load4_f32(a, i);
        simd.store4_f32(out, i, simd.min_f32x4(simd.max_f32x4(v, vlo), vhi));
    }
    while (i < N) : (i += 1) {
        const v = a[i];
        out[i] = if (v < lo) lo else if (v > hi) hi else v;
    }
}

/// Element-wise clamp for i64, scalar loop (no i64x2 compare in WASM SIMD).
export fn clip_i64(a: [*]const i64, out: [*]i64, N: u32, lo: i64, hi: i64) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        var v = a[i];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        out[i] = v;
    }
}

/// Element-wise clamp for u64, scalar loop (no u64x2 compare in WASM SIMD).
export fn clip_u64(a: [*]const u64, out: [*]u64, N: u32, lo: u64, hi: u64) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        var v = a[i];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        out[i] = v;
    }
}

/// Element-wise clamp for i32 using 4-wide SIMD with compare+select.
export fn clip_i32(a: [*]const i32, out: [*]i32, N: u32, lo: i32, hi: i32) void {
    const vlo: simd.V4i32 = @splat(lo);
    const vhi: simd.V4i32 = @splat(hi);
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        const v = simd.load4_i32(a, i);
        const clamped_lo = @select(i32, v > vlo, v, vlo);
        simd.store4_i32(out, i, @select(i32, clamped_lo < vhi, clamped_lo, vhi));
    }
    while (i < N) : (i += 1) {
        var v = a[i];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        out[i] = v;
    }
}

/// Element-wise clamp for u32 using 4-wide SIMD with unsigned compare+select.
export fn clip_u32(a: [*]const u32, out: [*]u32, N: u32, lo: u32, hi: u32) void {
    const vlo: simd.V4u32 = @splat(lo);
    const vhi: simd.V4u32 = @splat(hi);
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        const v = simd.load4_u32(a, i);
        const clamped_lo = @select(u32, v > vlo, v, vlo);
        simd.store4_u32(out, i, @select(u32, clamped_lo < vhi, clamped_lo, vhi));
    }
    while (i < N) : (i += 1) {
        var v = a[i];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        out[i] = v;
    }
}

/// Element-wise clamp for i16 using 8-wide SIMD with compare+select.
export fn clip_i16(a: [*]const i16, out: [*]i16, N: u32, lo: i16, hi: i16) void {
    const vlo: simd.V8i16 = @splat(lo);
    const vhi: simd.V8i16 = @splat(hi);
    const n_simd = N & ~@as(u32, 7);
    var i: u32 = 0;
    while (i < n_simd) : (i += 8) {
        const v = simd.load8_i16(a, i);
        const clamped_lo = @select(i16, v > vlo, v, vlo);
        simd.store8_i16(out, i, @select(i16, clamped_lo < vhi, clamped_lo, vhi));
    }
    while (i < N) : (i += 1) {
        var v = a[i];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        out[i] = v;
    }
}

/// Element-wise clamp for u16 using 8-wide SIMD with unsigned compare+select.
export fn clip_u16(a: [*]const u16, out: [*]u16, N: u32, lo: u16, hi: u16) void {
    const vlo: simd.V8u16 = @splat(lo);
    const vhi: simd.V8u16 = @splat(hi);
    const n_simd = N & ~@as(u32, 7);
    var i: u32 = 0;
    while (i < n_simd) : (i += 8) {
        const v = simd.load8_u16(a, i);
        const clamped_lo = @select(u16, v > vlo, v, vlo);
        simd.store8_u16(out, i, @select(u16, clamped_lo < vhi, clamped_lo, vhi));
    }
    while (i < N) : (i += 1) {
        var v = a[i];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        out[i] = v;
    }
}

/// Element-wise clamp for i8 using 16-wide SIMD with compare+select.
export fn clip_i8(a: [*]const i8, out: [*]i8, N: u32, lo: i8, hi: i8) void {
    const vlo: simd.V16i8 = @splat(lo);
    const vhi: simd.V16i8 = @splat(hi);
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    while (i < n_simd) : (i += 16) {
        const v = simd.load16_i8(a, i);
        const clamped_lo = @select(i8, v > vlo, v, vlo);
        simd.store16_i8(out, i, @select(i8, clamped_lo < vhi, clamped_lo, vhi));
    }
    while (i < N) : (i += 1) {
        var v = a[i];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        out[i] = v;
    }
}

/// Element-wise clamp for u8 using 16-wide SIMD with unsigned compare+select.
export fn clip_u8(a: [*]const u8, out: [*]u8, N: u32, lo: u8, hi: u8) void {
    const vlo: simd.V16u8 = @splat(lo);
    const vhi: simd.V16u8 = @splat(hi);
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    while (i < n_simd) : (i += 16) {
        const v = simd.load16_u8(a, i);
        const clamped_lo = @select(u8, v > vlo, v, vlo);
        simd.store16_u8(out, i, @select(u8, clamped_lo < vhi, clamped_lo, vhi));
    }
    while (i < N) : (i += 1) {
        var v = a[i];
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        out[i] = v;
    }
}

/// --- Tests ---

const testing = @import("std").testing;

test "clip_f64 basic" {
    const a = [_]f64{ -5, 0, 3, 10, 15 };
    var out: [5]f64 = undefined;
    clip_f64(&a, &out, 5, 0.0, 10.0);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 10.0, 1e-10);
    try testing.expectApproxEqAbs(out[4], 10.0, 1e-10);
}

test "clip_i8 basic" {
    const a = [_]i8{ -100, -50, 0, 50, 100, -10, 10, 20, -20, 30, -30, 40, -40, 50, -50, 60, 70 };
    var out: [17]i8 = undefined;
    clip_i8(&a, &out, 17, -10, 10);
    try testing.expectEqual(out[0], -10);
    try testing.expectEqual(out[2], 0);
    try testing.expectEqual(out[4], 10);
}

test "clip_u8 unsigned values above 127" {
    const a = [_]u8{ 0, 50, 100, 150, 200, 255, 128, 64, 190, 210, 230, 240, 10, 20, 30, 40, 250 };
    var out: [17]u8 = undefined;
    clip_u8(&a, &out, 17, 50, 200);
    try testing.expectEqual(out[0], 50); // 0 → clamped to 50
    try testing.expectEqual(out[1], 50); // 50 → stays 50
    try testing.expectEqual(out[2], 100); // 100 → stays 100
    try testing.expectEqual(out[3], 150); // 150 → stays 150
    try testing.expectEqual(out[4], 200); // 200 → stays 200
    try testing.expectEqual(out[5], 200); // 255 → clamped to 200
}
