//! WASM conjugate dot product kernels for complex types.
//!
//! Computes out = conj(a) · b = sum_k conj(a[k]) * b[k].
//! For complex128: each element = 2 f64s (re, im).
//! For complex64: each element = 2 f32s (re, im).
//! Real-type vdot is identical to dot — use dot kernels instead.

/// Computes the conjugate dot product of two complex128 vectors of length K.
/// K is the number of complex elements (each = 2 f64s).
/// conj(a) * b = (a_re - a_im*i) * (b_re + b_im*i)
///             = (a_re*b_re + a_im*b_im) + (a_re*b_im - a_im*b_re)*i
export fn vdot_c128(a: [*]const f64, b: [*]const f64, out: [*]f64, K: u32) void {
    var sum_re: f64 = 0;
    var sum_im: f64 = 0;

    // Scalar loop: conjugate complex multiply-accumulate
    for (0..K) |k| {
        const idx = k * 2;
        const a_re = a[idx];
        const a_im = a[idx + 1];
        const b_re = b[idx];
        const b_im = b[idx + 1];
        // conj(a) * b = (a_re - a_im*i) * (b_re + b_im*i)
        sum_re += a_re * b_re + a_im * b_im;
        sum_im += a_re * b_im - a_im * b_re;
    }
    out[0] = sum_re;
    out[1] = sum_im;
}

/// Computes the conjugate dot product of two complex64 vectors of length K.
/// K is the number of complex elements (each = 2 f32s).
/// conj(a) * b = (a_re - a_im*i) * (b_re + b_im*i)
///             = (a_re*b_re + a_im*b_im) + (a_re*b_im - a_im*b_re)*i
export fn vdot_c64(a: [*]const f32, b: [*]const f32, out: [*]f32, K: u32) void {
    var sum_re: f32 = 0;
    var sum_im: f32 = 0;

    // Scalar loop: conjugate complex multiply-accumulate
    for (0..K) |k| {
        const idx = k * 2;
        const a_re = a[idx];
        const a_im = a[idx + 1];
        const b_re = b[idx];
        const b_im = b[idx + 1];
        // conj(a) * b = (a_re - a_im*i) * (b_re + b_im*i)
        sum_re += a_re * b_re + a_im * b_im;
        sum_im += a_re * b_im - a_im * b_re;
    }
    out[0] = sum_re;
    out[1] = sum_im;
}

// --- Tests ---

test "vdot_c128 conjugate" {
    const testing = @import("std").testing;
    // conj(1+2i) * (3+4i) = (1-2i)*(3+4i) = 3+4i-6i+8 = 11-2i
    const a = [_]f64{ 1, 2 };
    const b = [_]f64{ 3, 4 };
    var out: [2]f64 = undefined;
    vdot_c128(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 11.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], -2.0, 1e-10);
}

test "vdot_c128 two elements" {
    const testing = @import("std").testing;
    // conj(1+2i)*(3+4i) + conj(5+6i)*(7+8i)
    // = (11-2i) + (5-6i)*(7+8i) = (11-2i) + (35+40i-42i+48) = (11-2i)+(83-2i) = 94-4i
    const a = [_]f64{ 1, 2, 5, 6 };
    const b = [_]f64{ 3, 4, 7, 8 };
    var out: [2]f64 = undefined;
    vdot_c128(&a, &b, &out, 2);
    try testing.expectApproxEqAbs(out[0], 94.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], -4.0, 1e-10);
}

test "vdot_c64 conjugate" {
    const testing = @import("std").testing;
    // conj(1+2i) * (3+4i) = 11-2i
    const a = [_]f32{ 1, 2 };
    const b = [_]f32{ 3, 4 };
    var out: [2]f32 = undefined;
    vdot_c64(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 11.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], -2.0, 1e-5);
}
