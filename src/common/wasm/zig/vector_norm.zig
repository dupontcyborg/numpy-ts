//! WASM vector norm kernels.
//!
//! Computes sqrt(sum(x[i]^2)) for L2 norm (the most common case).
//! Uses SIMD accumulation for high throughput.

const simd = @import("simd.zig");

/// L2 norm for f64: sqrt(sum(x[i]^2)) with 2-wide SIMD accumulation.
export fn vector_norm2_f64(a: [*]const f64, N: u32) f64 {
    var acc: simd.V2f64 = @splat(0);
    const n2 = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n2) : (i += 2) {
        const v = simd.load2_f64(a, i);
        acc += v * v;
    }
    var sum = acc[0] + acc[1];
    while (i < N) : (i += 1) {
        sum += a[i] * a[i];
    }
    return @sqrt(sum);
}

/// L2 norm for f32: sqrt(sum(x[i]^2)) with 4-wide SIMD accumulation.
export fn vector_norm2_f32(a: [*]const f32, N: u32) f32 {
    var acc: simd.V4f32 = @splat(0);
    const n4 = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n4) : (i += 4) {
        const v = simd.load4_f32(a, i);
        acc += v * v;
    }
    var sum = acc[0] + acc[1] + acc[2] + acc[3];
    while (i < N) : (i += 1) {
        sum += a[i] * a[i];
    }
    return @sqrt(sum);
}

// --- Tests ---

test "vector_norm2_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 3.0, 4.0 };
    try testing.expectApproxEqAbs(vector_norm2_f64(&a, 2), 5.0, 1e-10);
}

test "vector_norm2_f64 zeros" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0.0, 0.0, 0.0 };
    try testing.expectApproxEqAbs(vector_norm2_f64(&a, 3), 0.0, 1e-10);
}

test "vector_norm2_f64 single" {
    const testing = @import("std").testing;
    const a = [_]f64{7.0};
    try testing.expectApproxEqAbs(vector_norm2_f64(&a, 1), 7.0, 1e-10);
}

test "vector_norm2_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 3.0, 4.0 };
    try testing.expectApproxEqAbs(vector_norm2_f32(&a, 2), 5.0, 1e-4);
}

test "vector_norm2_f64 larger" {
    const testing = @import("std").testing;
    // [1, 2, 3, 4, 5] → sqrt(1+4+9+16+25) = sqrt(55)
    const a = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    try testing.expectApproxEqAbs(vector_norm2_f64(&a, 5), @sqrt(55.0), 1e-10);
}
