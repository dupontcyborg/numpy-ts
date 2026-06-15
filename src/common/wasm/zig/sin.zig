//! WASM element-wise sine kernels for float types.
//!
//! Unary: out[i] = sin(a[i])
//! Operates on contiguous 1D buffers of length N.
//! Only float types — integer inputs are promoted to float64 in JS.
//!
//! No sin opcode exists on any ISA; it's a software polynomial. The f64/f32 fast
//! paths use the shared SIMD Cephes core in trig.zig (Cody-Waite reduction +
//! quadrant select), processing a 2-wide f64 lane per step instead of a scalar
//! libm call. f32 is computed through the f64 core then narrowed — matching the
//! previous scalar path's accuracy for the large arguments the benchmarks feed.

const math = @import("std").math;
const simd = @import("simd.zig");
const trig = @import("trig.zig");

/// Element-wise sin for f64 using 2-wide SIMD: out[i] = sin(a[i]).
export fn sin_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, trig.sinv_f64(simd.load2_f64(a, i)));
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = .{ a[i], a[i] };
        out[i] = trig.sinv_f64(v)[0];
    }
}

/// Element-wise sin for f32 via the 2-wide f64 core, then narrowed.
export fn sin_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const v: simd.V2f64 = .{ a[i], a[i + 1] };
        const r = trig.sinv_f64(v);
        out[i] = @floatCast(r[0]);
        out[i + 1] = @floatCast(r[1]);
    }
    while (i < N) : (i += 1) {
        out[i] = @floatCast(math.sin(@as(f64, a[i])));
    }
}

// --- Integer-to-f64 variants (avoid JS conversion loop) ---

/// Element-wise sin for i64 → f64 output: out[i] = sin(float(a[i])).
export fn sin_i64_f64(a: [*]const i64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = math.sin(@as(f64, @floatFromInt(a[i])));
}

/// Element-wise sin for u64 → f64 output: out[i] = sin(float(a[i])).
export fn sin_u64_f64(a: [*]const u64, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = math.sin(@as(f64, @floatFromInt(a[i])));
}

/// Element-wise sin for i32 → f64 output: out[i] = sin(float(a[i])).
export fn sin_i32_f64(a: [*]const i32, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = math.sin(@as(f64, @floatFromInt(a[i])));
}

/// Element-wise sin for u32 → f64 output: out[i] = sin(float(a[i])).
export fn sin_u32_f64(a: [*]const u32, out: [*]f64, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = math.sin(@as(f64, @floatFromInt(a[i])));
}

/// Element-wise sin for i16 → f32 output: out[i] = sin(float(a[i])).
export fn sin_i16_f32(a: [*]const i16, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = @floatCast(math.sin(@as(f64, @floatFromInt(a[i]))));
}

/// Element-wise sin for u16 → f32 output: out[i] = sin(float(a[i])).
export fn sin_u16_f32(a: [*]const u16, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = @floatCast(math.sin(@as(f64, @floatFromInt(a[i]))));
}

/// Element-wise sin for i8 → f32 output: out[i] = sin(float(a[i])).
export fn sin_i8_f32(a: [*]const i8, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = @floatCast(math.sin(@as(f64, @floatFromInt(a[i]))));
}

/// Element-wise sin for u8 → f32 output: out[i] = sin(float(a[i])).
export fn sin_u8_f32(a: [*]const u8, out: [*]f32, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) out[i] = @floatCast(math.sin(@as(f64, @floatFromInt(a[i]))));
}

// --- Tests ---

test "sin_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0 };
    var out: [4]f64 = undefined;
    sin_f64(&a, &out, 4);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[1], 1.0, 1e-10);
    try testing.expectApproxEqAbs(out[2], 0.0, 1e-10);
    try testing.expectApproxEqAbs(out[3], -1.0, 1e-10);
}

test "sin_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]f32{ 0, math.pi / 2.0, math.pi };
    var out: [3]f32 = undefined;
    sin_f32(&a, &out, 3);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
    try testing.expectApproxEqAbs(out[1], 1.0, 1e-5);
    try testing.expectApproxEqAbs(out[2], 0.0, 1e-5);
}

test "sin_i64_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]i64{0};
    var out: [1]f64 = undefined;
    sin_i64_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
}

test "sin_u64_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]u64{0};
    var out: [1]f64 = undefined;
    sin_u64_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
}

test "sin_i32_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]i32{0};
    var out: [1]f64 = undefined;
    sin_i32_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
}

test "sin_u32_f64 basic" {
    const testing = @import("std").testing;
    const a = [_]u32{0};
    var out: [1]f64 = undefined;
    sin_u32_f64(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-10);
}

test "sin_i16_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]i16{0};
    var out: [1]f32 = undefined;
    sin_i16_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
}

test "sin_u16_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]u16{0};
    var out: [1]f32 = undefined;
    sin_u16_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
}

test "sin_i8_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]i8{0};
    var out: [1]f32 = undefined;
    sin_i8_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
}

test "sin_u8_f32 basic" {
    const testing = @import("std").testing;
    const a = [_]u8{0};
    var out: [1]f32 = undefined;
    sin_u8_f32(&a, &out, 1);
    try testing.expectApproxEqAbs(out[0], 0.0, 1e-5);
}
