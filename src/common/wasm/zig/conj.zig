//! WASM element-wise complex conjugate kernels.
//!
//! Unary: out[re] = a[re], out[im] = -a[im]
//! Operates on contiguous interleaved [re, im, re, im, ...] buffers.
//! N is the number of complex elements (buffer length = N * 2).

const simd = @import("simd.zig");

/// Complex conjugate for complex64 (f32 pairs) using 4-wide f32 SIMD.
/// Multiplies by [1, -1, 1, -1] to negate imaginary lanes.
/// N = number of complex elements (each = 2 f32s).
export fn conj_c64(a: [*]const f32, out: [*]f32, N: u32) void {
    const mask: simd.V4f32 = .{ 1, -1, 1, -1 };
    const n_f32 = N * 2;
    const n_simd = n_f32 & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_f32(out, i, simd.load4_f32(a, i) * mask);
    }
    // Scalar tail: at most 1 complex element (2 floats)
    while (i < n_f32) : (i += 2) {
        out[i] = a[i];
        out[i + 1] = -a[i + 1];
    }
}

/// Complex conjugate for complex128 (f64 pairs) using 2-wide f64 SIMD.
/// Multiplies by [1, -1] to negate the imaginary lane.
/// N = number of complex elements (each = 2 f64s).
export fn conj_c128(a: [*]const f64, out: [*]f64, N: u32) void {
    const mask: simd.V2f64 = .{ 1, -1 };
    const n_f64 = N * 2;
    const n_simd = n_f64 & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, simd.load2_f64(a, i) * mask);
    }
    // No scalar tail needed: N*2 is always even, and SIMD stride is 2.
}

// --- Tests ---

test "conj_c64 basic" {
    const testing = @import("std").testing;
    // conj(1+2i) = 1-2i, conj(3+4i) = 3-4i
    const a = [_]f32{ 1, 2, 3, 4 };
    var out: [4]f32 = undefined;
    conj_c64(&a, &out, 2);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], -2.0, 1e-5);
    try testing.expectApproxEqAbs(out[2], 3.0, 1e-5);
    try testing.expectApproxEqAbs(out[3], -4.0, 1e-5);
}

test "conj_c64 odd count" {
    const testing = @import("std").testing;
    // 3 complex elements = 6 floats, tail after 4-wide SIMD
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var out: [6]f32 = undefined;
    conj_c64(&a, &out, 3);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], -2.0, 1e-5);
    try testing.expectApproxEqAbs(out[4], 5.0, 1e-5);
    try testing.expectApproxEqAbs(out[5], -6.0, 1e-5);
}

test "conj_c128 basic" {
    const testing = @import("std").testing;
    // conj(1+2i) = 1-2i, conj(0+0i) = 0-0i
    const a = [_]f64{ 1, 2, 0, 0 };
    var out: [4]f64 = undefined;
    conj_c128(&a, &out, 2);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], -2.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], 0.0, 1e-10);
}

test "conj_c64 single element" {
    const testing = @import("std").testing;
    const a = [_]f32{ -5, 7 };
    var out: [2]f32 = undefined;
    conj_c64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], -5.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], -7.0, 1e-5);
}

test "conj_c128 single element" {
    const testing = @import("std").testing;
    const a = [_]f64{ -3.14, 2.71 };
    var out: [2]f64 = undefined;
    conj_c128(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], -3.14, 1e-10);
    try testing.expectApproxEqAbs(out[1], -2.71, 1e-10);
}
