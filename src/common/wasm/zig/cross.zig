//! WASM batched 3-vector cross product kernels for all numeric types.
//!
//! Computes n pairs of 3-vector cross products:
//!   out[i*3+0] = a[i*3+1]*b[i*3+2] - a[i*3+2]*b[i*3+1]
//!   out[i*3+1] = a[i*3+2]*b[i*3+0] - a[i*3+0]*b[i*3+2]
//!   out[i*3+2] = a[i*3+0]*b[i*3+1] - a[i*3+1]*b[i*3+0]

/// Computes out = a×b for n pairs of 3-vectors in f64 arrays.
/// a, b, and out are expected to be laid out as [x0,y0,z0, x1,y1,z1, ...].
export fn cross_f64(a: [*]const f64, b: [*]const f64, out: [*]f64, n: u32) void {
    for (0..@as(usize, n)) |i| {
        const o = i * 3;
        out[o + 0] = a[o + 1] * b[o + 2] - a[o + 2] * b[o + 1];
        out[o + 1] = a[o + 2] * b[o + 0] - a[o + 0] * b[o + 2];
        out[o + 2] = a[o + 0] * b[o + 1] - a[o + 1] * b[o + 0];
    }
}

/// Computes out = a×b for n pairs of 3-vectors in f32 arrays.
/// a, b, and out are expected to be laid out as [x0,y0,z0, x1,y1,z1, ...].
export fn cross_f32(a: [*]const f32, b: [*]const f32, out: [*]f32, n: u32) void {
    for (0..@as(usize, n)) |i| {
        const o = i * 3;
        out[o + 0] = a[o + 1] * b[o + 2] - a[o + 2] * b[o + 1];
        out[o + 1] = a[o + 2] * b[o + 0] - a[o + 0] * b[o + 2];
        out[o + 2] = a[o + 0] * b[o + 1] - a[o + 1] * b[o + 0];
    }
}

/// Computes out = a×b for n pairs of 3-vectors in complex128 arrays.
/// a, b, and out are expected to be laid out as [re0,im0, re1,im1, re2,im2, ...] for each 3
export fn cross_c128(a: [*]const f64, b: [*]const f64, out: [*]f64, n: u32) void {
    for (0..@as(usize, n)) |i| {
        const o = i * 6; // 3 complex elements × 2 f64s each
        // Component 0: a[1]*b[2] - a[2]*b[1]
        cmul_sub_f64(a, o + 2, b, o + 4, a, o + 4, b, o + 2, out, o + 0);
        // Component 1: a[2]*b[0] - a[0]*b[2]
        cmul_sub_f64(a, o + 4, b, o + 0, a, o + 0, b, o + 4, out, o + 2);
        // Component 2: a[0]*b[1] - a[1]*b[0]
        cmul_sub_f64(a, o + 0, b, o + 2, a, o + 2, b, o + 0, out, o + 4);
    }
}

/// Computes out = a×b for n pairs of 3-vectors in complex64 arrays.
/// a, b, and out are expected to be laid out as [re0,im0, re1,im1, re2,im2, ...] for each 3
export fn cross_c64(a: [*]const f32, b: [*]const f32, out: [*]f32, n: u32) void {
    for (0..@as(usize, n)) |i| {
        const o = i * 6; // 3 complex elements × 2 f32s each
        // Component 0: a[1]*b[2] - a[2]*b[1]
        cmul_sub_f32(a, o + 2, b, o + 4, a, o + 4, b, o + 2, out, o + 0);
        // Component 1: a[2]*b[0] - a[0]*b[2]
        cmul_sub_f32(a, o + 4, b, o + 0, a, o + 0, b, o + 4, out, o + 2);
        // Component 2: a[0]*b[1] - a[1]*b[0]
        cmul_sub_f32(a, o + 0, b, o + 2, a, o + 2, b, o + 0, out, o + 4);
    }
}

/// Computes out = a×b for n pairs of 3-vectors in i64 arrays.
/// a, b, and out are expected to be laid out as [x0,y0,z0, x1,y1,z1, ...].
/// Handles both signed (i64) and unsigned (u64).
export fn cross_i64(a: [*]const i64, b: [*]const i64, out: [*]i64, n: u32) void {
    for (0..@as(usize, n)) |i| {
        const o = i * 3;
        out[o + 0] = a[o + 1] *% b[o + 2] -% a[o + 2] *% b[o + 1];
        out[o + 1] = a[o + 2] *% b[o + 0] -% a[o + 0] *% b[o + 2];
        out[o + 2] = a[o + 0] *% b[o + 1] -% a[o + 1] *% b[o + 0];
    }
}

/// Computes out = a×b for n pairs of 3-vectors in i32 arrays.
/// a, b, and out are expected to be laid out as [x0,y0,z0, x1,y1,z1, ...].
/// Handles both signed (i32) and unsigned (u32).
export fn cross_i32(a: [*]const i32, b: [*]const i32, out: [*]i32, n: u32) void {
    for (0..@as(usize, n)) |i| {
        const o = i * 3;
        out[o + 0] = a[o + 1] *% b[o + 2] -% a[o + 2] *% b[o + 1];
        out[o + 1] = a[o + 2] *% b[o + 0] -% a[o + 0] *% b[o + 2];
        out[o + 2] = a[o + 0] *% b[o + 1] -% a[o + 1] *% b[o + 0];
    }
}

/// Computes out = a×b for n pairs of 3-vectors in i16 arrays.
/// a, b, and out are expected to be laid out as [x0,y0,z0, x1,y1,z1, ...].
/// Handles both signed (i16) and unsigned (u16).
export fn cross_i16(a: [*]const i16, b: [*]const i16, out: [*]i16, n: u32) void {
    for (0..@as(usize, n)) |i| {
        const o = i * 3;
        out[o + 0] = a[o + 1] *% b[o + 2] -% a[o + 2] *% b[o + 1];
        out[o + 1] = a[o + 2] *% b[o + 0] -% a[o + 0] *% b[o + 2];
        out[o + 2] = a[o + 0] *% b[o + 1] -% a[o + 1] *% b[o + 0];
    }
}

/// Computes out = a×b for n pairs of 3-vectors in i8 arrays.
/// a, b, and out are expected to be laid out as [x0,y0,z0, x1,y1,z1, ...].
/// Handles both signed (i8) and unsigned (u8).
export fn cross_i8(a: [*]const i8, b: [*]const i8, out: [*]i8, n: u32) void {
    for (0..@as(usize, n)) |i| {
        const o = i * 3;
        out[o + 0] = a[o + 1] *% b[o + 2] -% a[o + 2] *% b[o + 1];
        out[o + 1] = a[o + 2] *% b[o + 0] -% a[o + 0] *% b[o + 2];
        out[o + 2] = a[o + 0] *% b[o + 1] -% a[o + 1] *% b[o + 0];
    }
}

// --- Helpers ---

/// Computes out = x*y - z*w for complex f64 pairs at given byte offsets.
fn cmul_sub_f64(
    x: [*]const f64,
    xi: usize,
    y: [*]const f64,
    yi: usize,
    z: [*]const f64,
    zi: usize,
    w: [*]const f64,
    wi: usize,
    out: [*]f64,
    oi: usize,
) void {
    // x*y
    const xy_re = x[xi] * y[yi] - x[xi + 1] * y[yi + 1];
    const xy_im = x[xi] * y[yi + 1] + x[xi + 1] * y[yi];
    // z*w
    const zw_re = z[zi] * w[wi] - z[zi + 1] * w[wi + 1];
    const zw_im = z[zi] * w[wi + 1] + z[zi + 1] * w[wi];
    out[oi] = xy_re - zw_re;
    out[oi + 1] = xy_im - zw_im;
}

/// Computes out = x*y - z*w for complex f32 pairs at given byte offsets.
fn cmul_sub_f32(
    x: [*]const f32,
    xi: usize,
    y: [*]const f32,
    yi: usize,
    z: [*]const f32,
    zi: usize,
    w: [*]const f32,
    wi: usize,
    out: [*]f32,
    oi: usize,
) void {
    // x*y
    const xy_re = x[xi] * y[yi] - x[xi + 1] * y[yi + 1];
    const xy_im = x[xi] * y[yi + 1] + x[xi + 1] * y[yi];
    // z*w
    const zw_re = z[zi] * w[wi] - z[zi + 1] * w[wi + 1];
    const zw_im = z[zi] * w[wi + 1] + z[zi + 1] * w[wi];
    out[oi] = xy_re - zw_re;
    out[oi + 1] = xy_im - zw_im;
}

// --- Tests ---

test "cross_f64 single pair" {
    const testing = @import("std").testing;
    // [1,0,0] × [0,1,0] = [0,0,1]
    const a = [_]f64{ 1, 0, 0 };
    const b = [_]f64{ 0, 1, 0 };
    var out: [3]f64 = undefined;
    cross_f64(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 1.0, 1e-10);
}

test "cross_f64 batch=2" {
    const testing = @import("std").testing;
    // Pair 0: [1,2,3] × [4,5,6] = [-3,6,-3]
    // Pair 1: [1,0,0] × [0,0,1] = [0,-1,0]
    const a = [_]f64{ 1, 2, 3, 1, 0, 0 };
    const b = [_]f64{ 4, 5, 6, 0, 0, 1 };
    var out: [6]f64 = undefined;
    cross_f64(&a, &b, &out, 2);
    try testing.expectApproxEqAbs(out[0], -3.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 6.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], -3.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[4], -1.0, 1e-10);
    try testing.expectApproxEqAbs(out[5], 0.0, 1e-10);
}

test "cross_i32 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{ 1, 2, 3 };
    const b = [_]i32{ 4, 5, 6 };
    var out: [3]i32 = undefined;
    cross_i32(&a, &b, &out, 1);
    try testing.expectEqual(out[0], -3);
    try testing.expectEqual(out[1], 6);
    try testing.expectEqual(out[2], -3);
}

test "cross_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 1, 0, 0 };
    const b = [_]f32{ 0, 1, 0 };
    var out: [3]f32 = undefined;
    cross_f32(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-5);
    try testing.expectApproxEqAbs(out[2], 1.0, 1e-5);
}

test "cross_c128 basic" {
    const testing = @import("std").testing;
    // a = [(1+0i), (0+0i), (0+0i)], b = [(0+0i), (1+0i), (0+0i)]
    // Cross = [(0+0i), (0+0i), (1+0i)]
    const a = [_]f64{ 1, 0, 0, 0, 0, 0 };
    const b = [_]f64{ 0, 0, 1, 0, 0, 0 };
    var out: [6]f64 = undefined;
    cross_c128(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[4], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[5], 0.0, 1e-10);
}

test "cross_c128 imaginary" {
    const testing = @import("std").testing;
    // a = [(0+1i), (0+0i), (0+0i)], b = [(0+0i), (0+1i), (0+0i)]
    // a[0]*b[1] = (0+1i)*(0+1i) = -1+0i
    // Cross = [(0+0i), (0+0i), (-1+0i)]
    const a = [_]f64{ 0, 1, 0, 0, 0, 0 };
    const b = [_]f64{ 0, 0, 0, 1, 0, 0 };
    var out: [6]f64 = undefined;
    cross_c128(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[4], -1.0, 1e-10);
    try testing.expectApproxEqAbs(out[5], 0.0, 1e-10);
}

test "cross_c64 basic" {
    const testing = @import("std").testing;
    // a = [(1+0i), (0+0i), (0+0i)], b = [(0+0i), (1+0i), (0+0i)]
    // Cross = [(0+0i), (0+0i), (1+0i)]
    const a = [_]f32{ 1, 0, 0, 0, 0, 0 };
    const b = [_]f32{ 0, 0, 1, 0, 0, 0 };
    var out: [6]f32 = undefined;
    cross_c64(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-5);
    try testing.expectApproxEqAbs(out[2], 0.0, 1e-5);
    try testing.expectApproxEqAbs(out[3], 0.0, 1e-5);
    try testing.expectApproxEqAbs(out[4], 1.0, 1e-5);
    try testing.expectApproxEqAbs(out[5], 0.0, 1e-5);
}

test "cross_i64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{ 1, 2, 3 };
    const b = [_]i64{ 4, 5, 6 };
    var out: [3]i64 = undefined;
    cross_i64(&a, &b, &out, 1);
    try testing.expectEqual(out[0], -3);
    try testing.expectEqual(out[1], 6);
    try testing.expectEqual(out[2], -3);
}

test "cross_i16 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{ 1, 2, 3 };
    const b = [_]i16{ 4, 5, 6 };
    var out: [3]i16 = undefined;
    cross_i16(&a, &b, &out, 1);
    try testing.expectEqual(out[0], -3);
    try testing.expectEqual(out[1], 6);
    try testing.expectEqual(out[2], -3);
}

test "cross_i8 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{ 1, 2, 3 };
    const b = [_]i8{ 4, 5, 6 };
    var out: [3]i8 = undefined;
    cross_i8(&a, &b, &out, 1);
    try testing.expectEqual(out[0], -3);
    try testing.expectEqual(out[1], 6);
    try testing.expectEqual(out[2], -3);
}
