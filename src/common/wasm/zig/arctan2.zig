//! WASM element-wise arctan2 kernels for float types.
//!
//! Binary: out[i] = atan2(a[i], b[i])
//! Operates on contiguous 1D buffers of length N.
//! Only float types — integer inputs are promoted to float64 in JS.
//! WASM SIMD has no native atan2 instruction, so we use scalar loops.

const math = @import("std").math;

/// Element-wise atan2 for f64: out[i] = atan2(a[i], b[i]).
/// No SIMD — WASM has no f64x2.atan2 instruction.
export fn arctan2_f64(a: [*]const f64, b: [*]const f64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = math.atan2(a[i], b[i]);
    }
}

/// Element-wise atan2 for f32: out[i] = atan2(a[i], b[i]).
/// No SIMD — WASM has no f32x4.atan2 instruction.
export fn arctan2_f32(a: [*]const f32, b: [*]const f32, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = @floatCast(math.atan2(@as(f64, a[i]), @as(f64, b[i])));
    }
}

// --- Integer-to-f64 variants (avoid JS BigInt→Number conversion) ---

export fn arctan2_i64_f64(a: [*]const i64, b: [*]const i64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = math.atan2(@as(f64, @floatFromInt(a[i])), @as(f64, @floatFromInt(b[i])));
    }
}

export fn arctan2_u64_f64(a: [*]const u64, b: [*]const u64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = math.atan2(@as(f64, @floatFromInt(a[i])), @as(f64, @floatFromInt(b[i])));
    }
}

export fn arctan2_i32_f64(a: [*]const i32, b: [*]const i32, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = math.atan2(@as(f64, @floatFromInt(a[i])), @as(f64, @floatFromInt(b[i])));
    }
}

export fn arctan2_u32_f64(a: [*]const u32, b: [*]const u32, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = math.atan2(@as(f64, @floatFromInt(a[i])), @as(f64, @floatFromInt(b[i])));
    }
}

export fn arctan2_i16_f32(a: [*]const i16, b: [*]const i16, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = @floatCast(math.atan2(@as(f64, @floatFromInt(a[i])), @as(f64, @floatFromInt(b[i]))));
    }
}

export fn arctan2_u16_f32(a: [*]const u16, b: [*]const u16, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = @floatCast(math.atan2(@as(f64, @floatFromInt(a[i])), @as(f64, @floatFromInt(b[i]))));
    }
}

export fn arctan2_i8_f32(a: [*]const i8, b: [*]const i8, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = @floatCast(math.atan2(@as(f64, @floatFromInt(a[i])), @as(f64, @floatFromInt(b[i]))));
    }
}

export fn arctan2_u8_f32(a: [*]const u8, b: [*]const u8, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        out[i] = @floatCast(math.atan2(@as(f64, @floatFromInt(a[i])), @as(f64, @floatFromInt(b[i]))));
    }
}

// --- Tests ---

test "arctan2_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1.0, 0.0, -1.0, 0.0 };
    const b = [_]f64{ 0.0, 1.0, 0.0, -1.0 };
    var out: [4]f64 = undefined;
    arctan2_f64(&a, &b, &out, 4);
    try testing.expectApproxEqAbs(out[0], math.pi / 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], -math.pi / 2.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], math.pi, 1e-10);
}

test "arctan2_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 1.0, 0.0, -1.0 };
    const b = [_]f32{ 0.0, 1.0, 0.0 };
    var out: [3]f32 = undefined;
    arctan2_f32(&a, &b, &out, 3);
    try testing.expectApproxEqAbs(out[0], math.pi / 2.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 0.0, 1e-5);
    try testing.expectApproxEqAbs(out[2], -math.pi / 2.0, 1e-5);
}

test "arctan2_f64 negative values" {
    const testing = @import("std").testing;
    const a = [_]f64{ -1.0, 1.0 };
    const b = [_]f64{ -1.0, 1.0 };
    var out: [2]f64 = undefined;
    arctan2_f64(&a, &b, &out, 2);
    try testing.expectApproxEqAbs(out[0], -3.0 * math.pi / 4.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], math.pi / 4.0, 1e-10);
}

test "arctan2_i64_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{0};
    const b = [_]i64{1};
    var out: [1]f64 = undefined;
    arctan2_i64_f64(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
}

test "arctan2_u64_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]u64{0};
    const b = [_]u64{1};
    var out: [1]f64 = undefined;
    arctan2_u64_f64(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
}

test "arctan2_i32_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{0};
    const b = [_]i32{1};
    var out: [1]f64 = undefined;
    arctan2_i32_f64(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
}

test "arctan2_u32_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]u32{0};
    const b = [_]u32{1};
    var out: [1]f64 = undefined;
    arctan2_u32_f64(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
}

test "arctan2_i16_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{0};
    const b = [_]i16{1};
    var out: [1]f32 = undefined;
    arctan2_i16_f32(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
}

test "arctan2_u16_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{0};
    const b = [_]u16{1};
    var out: [1]f32 = undefined;
    arctan2_u16_f32(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
}

test "arctan2_i8_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{0};
    const b = [_]i8{1};
    var out: [1]f32 = undefined;
    arctan2_i8_f32(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
}

test "arctan2_u8_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{0};
    const b = [_]u8{1};
    var out: [1]f32 = undefined;
    arctan2_u8_f32(&a, &b, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
}
