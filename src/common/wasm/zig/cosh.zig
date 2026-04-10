//! WASM element-wise cosh kernels for float types.
//!
//! Unary: out[i] = cosh(a[i])
//! Operates on contiguous 1D buffers of length N.
//! Only float types — integer inputs handled via JS-side conversion.

const math = @import("std").math;

/// Element-wise cosh for f64: out[i] = cosh(a[i]).
export fn cosh_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = math.cosh(a[i]);
    }
}

/// Element-wise cosh for f32: out[i] = cosh(a[i]).
export fn cosh_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = @floatCast(math.cosh(@as(f64, a[i])));
    }
}

/// Element-wise cosh for i64 → f64 output. Scalar (no i64 SIMD in WASM).
export fn cosh_i64_f64(a: [*]const i64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = math.cosh(@as(f64, @floatFromInt(a[i])));
    }
}

/// Element-wise cosh for u64 → f64 output. Scalar (no u64 SIMD in WASM).
export fn cosh_u64_f64(a: [*]const u64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = math.cosh(@as(f64, @floatFromInt(a[i])));
    }
}

/// Element-wise cosh for i32 → f64 output.
export fn cosh_i32_f64(a: [*]const i32, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = math.cosh(@as(f64, @floatFromInt(a[i])));
}

/// Element-wise cosh for u32 → f64 output.
export fn cosh_u32_f64(a: [*]const u32, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = math.cosh(@as(f64, @floatFromInt(a[i])));
}

/// Element-wise cosh for i16 → f32 output.
export fn cosh_i16_f32(a: [*]const i16, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = @floatCast(math.cosh(@as(f64, @floatFromInt(a[i]))));
}

/// Element-wise cosh for u16 → f32 output.
export fn cosh_u16_f32(a: [*]const u16, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = @floatCast(math.cosh(@as(f64, @floatFromInt(a[i]))));
}

/// Element-wise cosh for i8 → f32 output.
export fn cosh_i8_f32(a: [*]const i8, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = @floatCast(math.cosh(@as(f64, @floatFromInt(a[i]))));
}

/// Element-wise cosh for u8 → f32 output.
export fn cosh_u8_f32(a: [*]const u8, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = @floatCast(math.cosh(@as(f64, @floatFromInt(a[i]))));
}

// --- Tests ---

test "cosh_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0.0, 1.0 };
    var out: [2]f64 = undefined;
    cosh_f64(&a, &out, 2);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 1.5430806348152437, 1e-10);
}

test "cosh_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 0.0, 1.0 };
    var out: [2]f32 = undefined;
    cosh_f32(&a, &out, 2);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 1.5431, 1e-4);
}

test "cosh_i64_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{0};
    var out: [1]f64 = undefined;
    cosh_i64_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
}

test "cosh_u64_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]u64{0};
    var out: [1]f64 = undefined;
    cosh_u64_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
}

test "cosh_i32_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{0};
    var out: [1]f64 = undefined;
    cosh_i32_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
}

test "cosh_u32_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]u32{0};
    var out: [1]f64 = undefined;
    cosh_u32_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-10);
}

test "cosh_i16_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{0};
    var out: [1]f32 = undefined;
    cosh_i16_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
}

test "cosh_u16_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{0};
    var out: [1]f32 = undefined;
    cosh_u16_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
}

test "cosh_i8_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{0};
    var out: [1]f32 = undefined;
    cosh_i8_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
}

test "cosh_u8_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{0};
    var out: [1]f32 = undefined;
    cosh_u8_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 1.0, 1e-5);
}
