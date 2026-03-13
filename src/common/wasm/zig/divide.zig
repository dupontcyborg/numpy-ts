//! WASM element-wise division kernels for float types.
//!
//! Binary: out[i] = a[i] / b[i]
//! Scalar: out[i] = a[i] / scalar
//! Only float types (integer division promotes to float64 in NumPy).

const simd = @import("simd.zig");

/// Element-wise divide for f64 using 2-wide SIMD: out[i] = a[i] / b[i].
export fn div_f64(a: [*]const f64, b: [*]const f64, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, simd.load2_f64(a, i) / simd.load2_f64(b, i));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] / b[i];
    }
}

/// Element-wise divide by scalar for f64 using 2-wide SIMD: out[i] = a[i] / scalar.
export fn div_scalar_f64(a: [*]const f64, out: [*]f64, N: u32, scalar: f64) void {
    const s: simd.V2f64 = @splat(scalar);
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, simd.load2_f64(a, i) / s);
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] / scalar;
    }
}

/// Element-wise divide for f32 using 4-wide SIMD: out[i] = a[i] / b[i].
export fn div_f32(a: [*]const f32, b: [*]const f32, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_f32(out, i, simd.load4_f32(a, i) / simd.load4_f32(b, i));
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] / b[i];
    }
}

/// Element-wise divide by scalar for f32 using 4-wide SIMD: out[i] = a[i] / scalar.
export fn div_scalar_f32(a: [*]const f32, out: [*]f32, N: u32, scalar: f32) void {
    const s: simd.V4f32 = @splat(scalar);
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_f32(out, i, simd.load4_f32(a, i) / s);
    }
    while (i < N) : (i += 1) {
        out[i] = a[i] / scalar;
    }
}

/// i64-to-f64 binary divide: out[i] = f64(a[i]) / f64(b[i]).
export fn div_i64_f64(a: [*]const i64, b: [*]const i64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = @as(f64, @floatFromInt(a[i])) / @as(f64, @floatFromInt(b[i]));
    }
}

/// i64-to-f64 scalar divide: out[i] = f64(a[i]) / scalar.
export fn div_scalar_i64_f64(a: [*]const i64, out: [*]f64, N: u32, scalar: f64) void {
    const s: simd.V2f64 = @splat(scalar);
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const v0: f64 = @floatFromInt(a[i]);
        const v1: f64 = @floatFromInt(a[i + 1]);
        const v = simd.V2f64{ v0, v1 };
        simd.store2_f64(out, i, v / s);
    }
    while (i < N) : (i += 1) {
        out[i] = @as(f64, @floatFromInt(a[i])) / scalar;
    }
}

/// i32-to-f64 binary divide with 4-wide SIMD: out[i] = f64(a[i]) / f64(b[i]).
export fn div_i32_f64(a: [*]const i32, b: [*]const i32, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = @as(f64, @floatFromInt(a[i])) / @as(f64, @floatFromInt(b[i]));
    }
}

/// i32-to-f64 scalar divide: out[i] = f64(a[i]) / scalar.
export fn div_scalar_i32_f64(a: [*]const i32, out: [*]f64, N: u32, scalar: f64) void {
    const s: simd.V2f64 = @splat(scalar);
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const v0: f64 = @floatFromInt(a[i]);
        const v1: f64 = @floatFromInt(a[i + 1]);
        const v = simd.V2f64{ v0, v1 };
        simd.store2_f64(out, i, v / s);
    }
    while (i < N) : (i += 1) {
        out[i] = @as(f64, @floatFromInt(a[i])) / scalar;
    }
}

/// i16-to-f64 binary divide: out[i] = f64(a[i]) / f64(b[i]).
export fn div_i16_f64(a: [*]const i16, b: [*]const i16, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = @as(f64, @floatFromInt(a[i])) / @as(f64, @floatFromInt(b[i]));
    }
}

/// i16-to-f64 scalar divide: out[i] = f64(a[i]) / scalar.
export fn div_scalar_i16_f64(a: [*]const i16, out: [*]f64, N: u32, scalar: f64) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = @as(f64, @floatFromInt(a[i])) / scalar;
    }
}

/// i8-to-f64 binary divide: out[i] = f64(a[i]) / f64(b[i]).
export fn div_i8_f64(a: [*]const i8, b: [*]const i8, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = @as(f64, @floatFromInt(a[i])) / @as(f64, @floatFromInt(b[i]));
    }
}

/// i8-to-f64 scalar divide: out[i] = f64(a[i]) / scalar.
export fn div_scalar_i8_f64(a: [*]const i8, out: [*]f64, N: u32, scalar: f64) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = @as(f64, @floatFromInt(a[i])) / scalar;
    }
}

/// Complex128 binary divide: (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c²+d²).
/// N = number of complex elements (each = 2 f64s).
export fn div_c128(a: [*]const f64, b: [*]const f64, out: [*]f64, N: u32) void {
    var k: u32 = 0;
    while (k + 2 <= N) : (k += 2) {
        const idx = k * 2;
        const a0 = simd.load2_f64(a, idx);
        const a1 = simd.load2_f64(a, idx + 2);
        const b0 = simd.load2_f64(b, idx);
        const b1 = simd.load2_f64(b, idx + 2);
        const a_re = @shuffle(f64, a0, a1, [2]i32{ 0, -1 });
        const a_im = @shuffle(f64, a0, a1, [2]i32{ 1, -2 });
        const b_re = @shuffle(f64, b0, b1, [2]i32{ 0, -1 });
        const b_im = @shuffle(f64, b0, b1, [2]i32{ 1, -2 });
        const denom = b_re * b_re + b_im * b_im;
        const re = (a_re * b_re + a_im * b_im) / denom;
        const im = (a_im * b_re - a_re * b_im) / denom;
        const lo = @shuffle(f64, re, im, [2]i32{ 0, -1 });
        const hi = @shuffle(f64, re, im, [2]i32{ 1, -2 });
        simd.store2_f64(out, idx, lo);
        simd.store2_f64(out, idx + 2, hi);
    }
    while (k < N) : (k += 1) {
        const idx = k * 2;
        const a_re = a[idx];
        const a_im = a[idx + 1];
        const b_re = b[idx];
        const b_im = b[idx + 1];
        const denom = b_re * b_re + b_im * b_im;
        out[idx] = (a_re * b_re + a_im * b_im) / denom;
        out[idx + 1] = (a_im * b_re - a_re * b_im) / denom;
    }
}

/// Complex64 binary divide: (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c²+d²).
/// N = number of complex elements (each = 2 f32s).
export fn div_c64(a: [*]const f32, b: [*]const f32, out: [*]f32, N: u32) void {
    var k: u32 = 0;
    while (k < N) : (k += 1) {
        const idx = k * 2;
        const a_re = a[idx];
        const a_im = a[idx + 1];
        const b_re = b[idx];
        const b_im = b[idx + 1];
        const denom = b_re * b_re + b_im * b_im;
        out[idx] = (a_re * b_re + a_im * b_im) / denom;
        out[idx + 1] = (a_im * b_re - a_re * b_im) / denom;
    }
}

// --- Tests ---

test "div_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 10, 20, 30, 40 };
    const b = [_]f64{ 2, 5, 10, 8 };
    var out: [4]f64 = undefined;
    div_f64(&a, &b, &out, 4);
    try testing.expectApproxEqAbs(out[0], 5.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 4.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 3.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 5.0, 1e-10);
}

test "div_scalar_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 10, 20, 30, 40 };
    var out: [4]f64 = undefined;
    div_scalar_f64(&a, &out, 4, 5.0);
    try testing.expectApproxEqAbs(out[0], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 4.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 6.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 8.0, 1e-10);
}

test "div_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 10, 20, 30, 40, 50 };
    const b = [_]f32{ 2, 4, 5, 8, 10 };
    var out: [5]f32 = undefined;
    div_f32(&a, &b, &out, 5);
    try testing.expectApproxEqAbs(out[0], 5.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 5.0, 1e-5);
    try testing.expectApproxEqAbs(out[4], 5.0, 1e-5);
}

test "div_c128 basic" {
    const testing = @import("std").testing;
    // (3+4i)/(1+2i) = (3+8+i(4-6))/(1+4) = (11-2i)/5 = (2.2-0.4i)
    const a = [_]f64{ 3, 4, 10, 0 };
    const b = [_]f64{ 1, 2, 5, 0 };
    var out: [4]f64 = undefined;
    div_c128(&a, &b, &out, 2);
    try testing.expectApproxEqAbs(out[0], 2.2, 1e-10);
    try testing.expectApproxEqAbs(out[1], -0.4, 1e-10);
    try testing.expectApproxEqAbs(out[2], 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 0.0, 1e-10);
}
