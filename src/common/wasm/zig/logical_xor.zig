//! WASM element-wise logical XOR kernels for all numeric types.
//!
//! Binary: out[i] = (a[i] != 0) ^ (b[i] != 0)
//! Scalar: out[i] = (a[i] != 0) ^ (scalar != 0)
//! Output is always u8 (0 or 1). Operates on contiguous 1D buffers of length N.

const simd = @import("simd.zig");

/// Element-wise logical XOR for f64: out[i] = (a[i] != 0) ^ (b[i] != 0).
export fn logical_xor_f64(a: [*]const f64, b: [*]const f64, out: [*]u8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const ab: u8 = if (a[i] != 0) 1 else 0;
        const bb: u8 = if (b[i] != 0) 1 else 0;
        out[i] = ab ^ bb;
    }
}

/// Element-wise logical XOR scalar for f64: out[i] = (a[i] != 0) ^ (scalar != 0).
export fn logical_xor_scalar_f64(a: [*]const f64, out: [*]u8, N: u32, scalar: f64) void {
    const sb: u8 = if (scalar != 0) 1 else 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const ab: u8 = if (a[i] != 0) 1 else 0;
        out[i] = ab ^ sb;
    }
}

/// Element-wise logical XOR for f32: out[i] = (a[i] != 0) ^ (b[i] != 0).
export fn logical_xor_f32(a: [*]const f32, b: [*]const f32, out: [*]u8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const ab: u8 = if (a[i] != 0) 1 else 0;
        const bb: u8 = if (b[i] != 0) 1 else 0;
        out[i] = ab ^ bb;
    }
}

/// Element-wise logical XOR scalar for f32: out[i] = (a[i] != 0) ^ (scalar != 0).
export fn logical_xor_scalar_f32(a: [*]const f32, out: [*]u8, N: u32, scalar: f32) void {
    const sb: u8 = if (scalar != 0) 1 else 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const ab: u8 = if (a[i] != 0) 1 else 0;
        out[i] = ab ^ sb;
    }
}

/// Element-wise logical XOR for i64, scalar loop (no i64x2 compare in WASM SIMD).
export fn logical_xor_i64(a: [*]const i64, b: [*]const i64, out: [*]u8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const ab: u8 = if (a[i] != 0) 1 else 0;
        const bb: u8 = if (b[i] != 0) 1 else 0;
        out[i] = ab ^ bb;
    }
}

/// Element-wise logical XOR scalar for i64, scalar loop (no i64x2 compare in WASM SIMD).
export fn logical_xor_scalar_i64(a: [*]const i64, out: [*]u8, N: u32, scalar: i64) void {
    const sb: u8 = if (scalar != 0) 1 else 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const ab: u8 = if (a[i] != 0) 1 else 0;
        out[i] = ab ^ sb;
    }
}

/// Element-wise logical XOR for i32: out[i] = (a[i] != 0) ^ (b[i] != 0).
export fn logical_xor_i32(a: [*]const i32, b: [*]const i32, out: [*]u8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const ab: u8 = if (a[i] != 0) 1 else 0;
        const bb: u8 = if (b[i] != 0) 1 else 0;
        out[i] = ab ^ bb;
    }
}

/// Element-wise logical XOR scalar for i32: out[i] = (a[i] != 0) ^ (scalar != 0).
export fn logical_xor_scalar_i32(a: [*]const i32, out: [*]u8, N: u32, scalar: i32) void {
    const sb: u8 = if (scalar != 0) 1 else 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const ab: u8 = if (a[i] != 0) 1 else 0;
        out[i] = ab ^ sb;
    }
}

/// Element-wise logical XOR for i16: out[i] = (a[i] != 0) ^ (b[i] != 0).
export fn logical_xor_i16(a: [*]const i16, b: [*]const i16, out: [*]u8, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const ab: u8 = if (a[i] != 0) 1 else 0;
        const bb: u8 = if (b[i] != 0) 1 else 0;
        out[i] = ab ^ bb;
    }
}

/// Element-wise logical XOR scalar for i16: out[i] = (a[i] != 0) ^ (scalar != 0).
export fn logical_xor_scalar_i16(a: [*]const i16, out: [*]u8, N: u32, scalar: i16) void {
    const sb: u8 = if (scalar != 0) 1 else 0;
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        const ab: u8 = if (a[i] != 0) 1 else 0;
        out[i] = ab ^ sb;
    }
}

/// Element-wise logical XOR for i8 using 16-wide SIMD: out[i] = (a[i] != 0) ^ (b[i] != 0).
/// Input and output are both byte-width, enabling natural 16-wide vectorization.
export fn logical_xor_i8(a: [*]const i8, b: [*]const i8, out: [*]u8, N: u32) void {
    const zero: simd.V16i8 = @splat(0);
    const one: simd.V16u8 = @splat(1);
    const zero_u8: simd.V16u8 = @splat(0);
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    while (i < n_simd) : (i += 16) {
        const va_bool: simd.V16u8 = @select(u8, simd.load16_i8(a, i) != zero, one, zero_u8);
        const vb_bool: simd.V16u8 = @select(u8, simd.load16_i8(b, i) != zero, one, zero_u8);
        simd.store16_u8(out, i, va_bool ^ vb_bool);
    }
    while (i < N) : (i += 1) {
        const ab: u8 = if (a[i] != 0) 1 else 0;
        const bb: u8 = if (b[i] != 0) 1 else 0;
        out[i] = ab ^ bb;
    }
}

/// Element-wise logical XOR scalar for i8 using 16-wide SIMD: out[i] = (a[i] != 0) ^ (scalar != 0).
export fn logical_xor_scalar_i8(a: [*]const i8, out: [*]u8, N: u32, scalar: i8) void {
    const zero: simd.V16i8 = @splat(0);
    const one: simd.V16u8 = @splat(1);
    const zero_u8: simd.V16u8 = @splat(0);
    const scalar_bool: u8 = if (scalar != 0) 1 else 0;
    const n_simd = N & ~@as(u32, 15);
    var i: u32 = 0;
    if (scalar_bool == 0) {
        // XOR with false = identity (toBool)
        while (i < n_simd) : (i += 16) {
            simd.store16_u8(out, i, @select(u8, simd.load16_i8(a, i) != zero, one, zero_u8));
        }
    } else {
        // XOR with true = NOT (toBool)
        while (i < n_simd) : (i += 16) {
            simd.store16_u8(out, i, @select(u8, simd.load16_i8(a, i) != zero, zero_u8, one));
        }
    }
    while (i < N) : (i += 1) {
        const ab: u8 = if (a[i] != 0) 1 else 0;
        out[i] = ab ^ scalar_bool;
    }
}

// --- Tests ---

test "logical_xor_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0.0, 1.5, 0.0, -2.0 };
    const b = [_]f64{ 1.0, 0.0, 0.0, -3.0 };
    var out: [4]u8 = undefined;
    logical_xor_f64(&a, &b, &out, 4);
    try testing.expectEqual(out[0], 1); // F ^ T = T
    try testing.expectEqual(out[1], 1); // T ^ F = T
    try testing.expectEqual(out[2], 0); // F ^ F = F
    try testing.expectEqual(out[3], 0); // T ^ T = F
}

test "logical_xor_scalar_i8 zero scalar" {
    const testing = @import("std").testing;
    const a = [_]i8{ 0, 1, -1, 0, 5 };
    var out: [5]u8 = undefined;
    logical_xor_scalar_i8(&a, &out, 5, 0);
    try testing.expectEqual(out[0], 0);
    try testing.expectEqual(out[1], 1);
    try testing.expectEqual(out[2], 1);
    try testing.expectEqual(out[3], 0);
    try testing.expectEqual(out[4], 1);
}

test "logical_xor_scalar_i8 nonzero scalar" {
    const testing = @import("std").testing;
    const a = [_]i8{ 0, 1, -1, 0, 5 };
    var out: [5]u8 = undefined;
    logical_xor_scalar_i8(&a, &out, 5, 3);
    try testing.expectEqual(out[0], 1); // F ^ T = T
    try testing.expectEqual(out[1], 0); // T ^ T = F
    try testing.expectEqual(out[2], 0); // T ^ T = F
    try testing.expectEqual(out[3], 1); // F ^ T = T
    try testing.expectEqual(out[4], 0); // T ^ T = F
}

test "logical_xor_i8 large SIMD" {
    const testing = @import("std").testing;
    var a: [20]i8 = undefined;
    var b: [20]i8 = undefined;
    for (0..20) |idx| {
        a[idx] = if (idx % 2 == 0) 1 else 0;
        b[idx] = if (idx % 3 == 0) 1 else 0;
    }
    var out: [20]u8 = undefined;
    logical_xor_i8(&a, &b, &out, 20);
    for (0..20) |idx| {
        const a_bool: u8 = if (idx % 2 == 0) 1 else 0;
        const b_bool: u8 = if (idx % 3 == 0) 1 else 0;
        try testing.expectEqual(out[idx], a_bool ^ b_bool);
    }
}
