//! WASM element-wise logarithm kernels (natural / base-2 / base-10) for
//! float / int / complex types.
//!
//! Real f64 uses the shared Cephes core (transcend.logv_f64); the f32 core is
//! local. Domain handling (x<0→NaN, x==0→-inf, +inf→+inf, NaN→NaN) and the
//! base rescale (log2/log10) are applied per-op here. Complex uses
//! log_b(a+bi) = (½·ln(a²+b²) + i·atan2(b,a)) / ln b.

const math = @import("std").math;
const simd = @import("simd.zig");
const t = @import("transcend.zig");

const LOG2E_F64: f64 = 1.4426950408889634073599; // 1/ln2
const LOG10E_F64: f64 = 0.43429448190325182765; // 1/ln10

/// Natural-log core (f64) + scale, with NumPy-faithful domain handling.
inline fn logScaled_f64(x: simd.V2f64, scale: f64) simd.V2f64 {
    var result = t.logv_f64(x) * @as(simd.V2f64, @splat(scale));
    const zero: simd.V2f64 = @splat(0.0);
    result = @select(f64, x < zero, @as(simd.V2f64, @splat(math.nan(f64))), result);
    result = @select(f64, x == zero, @as(simd.V2f64, @splat(-math.inf(f64))), result);
    result = @select(f64, x == @as(simd.V2f64, @splat(math.inf(f64))), @as(simd.V2f64, @splat(math.inf(f64))), result);
    result = @select(f64, x != x, x, result);
    return result;
}

fn logBase_f64(a: [*]const f64, out: [*]f64, N: u32, comptime scale: f64) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, logScaled_f64(simd.load2_f64(a, i), scale));
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = .{ a[i], a[i] };
        out[i] = logScaled_f64(v, scale)[0];
    }
}

export fn log_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    logBase_f64(a, out, N, 1.0);
}
export fn log2_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    logBase_f64(a, out, N, LOG2E_F64);
}
export fn log10_f64(a: [*]const f64, out: [*]f64, N: u32) void {
    logBase_f64(a, out, N, LOG10E_F64);
}

// --- f32 core (local; complex routes through transcend's f64 core) ---

const SQRTH_F32: f32 = 0.707106781186547524;
const LN2_HI_F32: f32 = 0.693359375;
const LN2_LO_F32: f32 = -2.12194440e-4;
const LOG2E_F32: f32 = 1.44269504088896341;
const LOG10E_F32: f32 = 0.43429448190325182765;

inline fn logv_f32(x: simd.V4f32) simd.V4f32 {
    const bits: simd.V4u32 = @bitCast(x);
    const raw_e: simd.V4u32 = (bits >> @as(simd.V4u32, @splat(23))) & @as(simd.V4u32, @splat(0xff));
    const ei: simd.V4i32 = @as(simd.V4i32, @bitCast(raw_e)) -% @as(simd.V4i32, @splat(126));
    const mant_bits = (bits & @as(simd.V4u32, @splat(0x007fffff))) |
        @as(simd.V4u32, @splat(0x3f000000));
    var m: simd.V4f32 = @bitCast(mant_bits);
    var e: simd.V4f32 = @floatFromInt(ei);

    const lo = m < @as(simd.V4f32, @splat(SQRTH_F32));
    e = @select(f32, lo, e - @as(simd.V4f32, @splat(1.0)), e);
    m = @select(f32, lo, m + m - @as(simd.V4f32, @splat(1.0)), m - @as(simd.V4f32, @splat(1.0)));

    const z = m * m;
    var p: simd.V4f32 = @splat(7.0376836292e-2);
    p = simd.mulAdd_f32x4(p, m, @splat(-1.1514610310e-1));
    p = simd.mulAdd_f32x4(p, m, @splat(1.1676998740e-1));
    p = simd.mulAdd_f32x4(p, m, @splat(-1.2420140846e-1));
    p = simd.mulAdd_f32x4(p, m, @splat(1.4249322787e-1));
    p = simd.mulAdd_f32x4(p, m, @splat(-1.6668057665e-1));
    p = simd.mulAdd_f32x4(p, m, @splat(2.0000714765e-1));
    p = simd.mulAdd_f32x4(p, m, @splat(-2.4999993993e-1));
    p = simd.mulAdd_f32x4(p, m, @splat(3.3333331174e-1));

    var y = p * m * z;
    y = simd.mulAdd_f32x4(e, @splat(LN2_LO_F32), y);
    y = simd.nmulAdd_f32x4(@splat(0.5), z, y);
    const result = m + y;
    return simd.mulAdd_f32x4(e, @splat(LN2_HI_F32), result);
}

inline fn logScaled_f32(x: simd.V4f32, scale: f32) simd.V4f32 {
    var result = logv_f32(x) * @as(simd.V4f32, @splat(scale));
    const zero: simd.V4f32 = @splat(0.0);
    result = @select(f32, x < zero, @as(simd.V4f32, @splat(math.nan(f32))), result);
    result = @select(f32, x == zero, @as(simd.V4f32, @splat(-math.inf(f32))), result);
    result = @select(f32, x == @as(simd.V4f32, @splat(math.inf(f32))), @as(simd.V4f32, @splat(math.inf(f32))), result);
    result = @select(f32, x != x, x, result);
    return result;
}

fn logBase_f32(a: [*]const f32, out: [*]f32, N: u32, comptime scale: f32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        simd.store4_f32(out, i, logScaled_f32(simd.load4_f32(a, i), scale));
    }
    while (i < N) : (i += 1) {
        const v: simd.V4f32 = .{ a[i], a[i], a[i], a[i] };
        out[i] = logScaled_f32(v, scale)[0];
    }
}

export fn log_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    logBase_f32(a, out, N, 1.0);
}
export fn log2_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    logBase_f32(a, out, N, LOG2E_F32);
}
export fn log10_f32(a: [*]const f32, out: [*]f32, N: u32) void {
    logBase_f32(a, out, N, LOG10E_F32);
}

// --- Integer inputs (i8/u8→f32→f16 in JS, i16/u16→f32, i32+/u32+→f64) ---

inline fn logIntBase_f32(comptime I: type, a: [*]const I, out: [*]f32, N: u32, comptime scale: f32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        const vi = @as(*align(1) const @Vector(4, I), @ptrCast(a + i)).*;
        const xf: simd.V4f32 = @floatFromInt(vi);
        simd.store4_f32(out, i, logScaled_f32(xf, scale));
    }
    while (i < N) : (i += 1) {
        const v: simd.V4f32 = @splat(@as(f32, @floatFromInt(a[i])));
        out[i] = logScaled_f32(v, scale)[0];
    }
}

inline fn logIntBase_f64(comptime I: type, a: [*]const I, out: [*]f64, N: u32, comptime scale: f64) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const vi = @as(*align(1) const @Vector(2, I), @ptrCast(a + i)).*;
        const xf: simd.V2f64 = @floatFromInt(vi);
        simd.store2_f64(out, i, logScaled_f64(xf, scale));
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = @splat(@as(f64, @floatFromInt(a[i])));
        out[i] = logScaled_f64(v, scale)[0];
    }
}

export fn log_i8_f32(a: [*]const i8, out: [*]f32, N: u32) void {
    logIntBase_f32(i8, a, out, N, 1.0);
}
export fn log_u8_f32(a: [*]const u8, out: [*]f32, N: u32) void {
    logIntBase_f32(u8, a, out, N, 1.0);
}
export fn log_i16_f32(a: [*]const i16, out: [*]f32, N: u32) void {
    logIntBase_f32(i16, a, out, N, 1.0);
}
export fn log_u16_f32(a: [*]const u16, out: [*]f32, N: u32) void {
    logIntBase_f32(u16, a, out, N, 1.0);
}
export fn log_i32_f64(a: [*]const i32, out: [*]f64, N: u32) void {
    logIntBase_f64(i32, a, out, N, 1.0);
}
export fn log_u32_f64(a: [*]const u32, out: [*]f64, N: u32) void {
    logIntBase_f64(u32, a, out, N, 1.0);
}
export fn log_i64_f64(a: [*]const i64, out: [*]f64, N: u32) void {
    logIntBase_f64(i64, a, out, N, 1.0);
}
export fn log_u64_f64(a: [*]const u64, out: [*]f64, N: u32) void {
    logIntBase_f64(u64, a, out, N, 1.0);
}

export fn log2_i8_f32(a: [*]const i8, out: [*]f32, N: u32) void {
    logIntBase_f32(i8, a, out, N, LOG2E_F32);
}
export fn log2_u8_f32(a: [*]const u8, out: [*]f32, N: u32) void {
    logIntBase_f32(u8, a, out, N, LOG2E_F32);
}
export fn log2_i16_f32(a: [*]const i16, out: [*]f32, N: u32) void {
    logIntBase_f32(i16, a, out, N, LOG2E_F32);
}
export fn log2_u16_f32(a: [*]const u16, out: [*]f32, N: u32) void {
    logIntBase_f32(u16, a, out, N, LOG2E_F32);
}
export fn log2_i32_f64(a: [*]const i32, out: [*]f64, N: u32) void {
    logIntBase_f64(i32, a, out, N, LOG2E_F64);
}
export fn log2_u32_f64(a: [*]const u32, out: [*]f64, N: u32) void {
    logIntBase_f64(u32, a, out, N, LOG2E_F64);
}
export fn log2_i64_f64(a: [*]const i64, out: [*]f64, N: u32) void {
    logIntBase_f64(i64, a, out, N, LOG2E_F64);
}
export fn log2_u64_f64(a: [*]const u64, out: [*]f64, N: u32) void {
    logIntBase_f64(u64, a, out, N, LOG2E_F64);
}

export fn log10_i8_f32(a: [*]const i8, out: [*]f32, N: u32) void {
    logIntBase_f32(i8, a, out, N, LOG10E_F32);
}
export fn log10_u8_f32(a: [*]const u8, out: [*]f32, N: u32) void {
    logIntBase_f32(u8, a, out, N, LOG10E_F32);
}
export fn log10_i16_f32(a: [*]const i16, out: [*]f32, N: u32) void {
    logIntBase_f32(i16, a, out, N, LOG10E_F32);
}
export fn log10_u16_f32(a: [*]const u16, out: [*]f32, N: u32) void {
    logIntBase_f32(u16, a, out, N, LOG10E_F32);
}
export fn log10_i32_f64(a: [*]const i32, out: [*]f64, N: u32) void {
    logIntBase_f64(i32, a, out, N, LOG10E_F64);
}
export fn log10_u32_f64(a: [*]const u32, out: [*]f64, N: u32) void {
    logIntBase_f64(u32, a, out, N, LOG10E_F64);
}
export fn log10_i64_f64(a: [*]const i64, out: [*]f64, N: u32) void {
    logIntBase_f64(i64, a, out, N, LOG10E_F64);
}
export fn log10_u64_f64(a: [*]const u64, out: [*]f64, N: u32) void {
    logIntBase_f64(u64, a, out, N, LOG10E_F64);
}

// --- Complex: log_b(a+bi) = (½·ln(a²+b²) + i·atan2(b,a)) / ln b ---
inline fn logImpl(re: simd.V2f64, im: simd.V2f64, out_re: *simd.V2f64, out_im: *simd.V2f64, comptime scale: f64) void {
    const mag2 = re * re + im * im;
    var lr = t.logv_f64(mag2) * @as(simd.V2f64, @splat(0.5 * scale));
    const zero: simd.V2f64 = @splat(0.0);
    lr = @select(f64, mag2 == zero, @as(simd.V2f64, @splat(-math.inf(f64))), lr);
    out_re.* = lr;
    out_im.* = t.atan2v_f64(im, re) * @as(simd.V2f64, @splat(scale));
}

inline fn opLog(re: simd.V2f64, im: simd.V2f64, out_re: *simd.V2f64, out_im: *simd.V2f64) void {
    logImpl(re, im, out_re, out_im, 1.0);
}
inline fn opLog2(re: simd.V2f64, im: simd.V2f64, out_re: *simd.V2f64, out_im: *simd.V2f64) void {
    logImpl(re, im, out_re, out_im, LOG2E_F64);
}
inline fn opLog10(re: simd.V2f64, im: simd.V2f64, out_re: *simd.V2f64, out_im: *simd.V2f64) void {
    logImpl(re, im, out_re, out_im, LOG10E_F64);
}

export fn log_c128(a: [*]const f64, out: [*]f64, N: u32) void {
    t.cdrive_c128(a, out, N, opLog);
}
export fn log_c64(a: [*]const f32, out: [*]f32, N: u32) void {
    t.cdrive_c64(a, out, N, opLog);
}
export fn log2_c128(a: [*]const f64, out: [*]f64, N: u32) void {
    t.cdrive_c128(a, out, N, opLog2);
}
export fn log2_c64(a: [*]const f32, out: [*]f32, N: u32) void {
    t.cdrive_c64(a, out, N, opLog2);
}
export fn log10_c128(a: [*]const f64, out: [*]f64, N: u32) void {
    t.cdrive_c128(a, out, N, opLog10);
}
export fn log10_c64(a: [*]const f32, out: [*]f32, N: u32) void {
    t.cdrive_c64(a, out, N, opLog10);
}

// --- Tests ---

test "log_f64 matches std.math.log" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1.0, 2.0, 0.5, 10.0, 100.0, 0.001, 1234.5, 1e-8 };
    var out: [8]f64 = undefined;
    log_f64(&a, &out, 8);
    for (a, 0..) |x, i| {
        try testing.expectApproxEqRel(out[i], @log(x), 1e-12);
    }
}

test "log_f64 domain edges" {
    const testing = @import("std").testing;
    const a = [_]f64{ 0.0, -1.0 };
    var out: [2]f64 = undefined;
    log_f64(&a, &out, 2);
    try testing.expect(out[0] == -math.inf(f64));
    try testing.expect(math.isNan(out[1]));
}

test "log_i64_f64 / log2_i64_f64 integer inputs" {
    const testing = @import("std").testing;
    const a = [_]i64{ 8, 64 };
    var o: [2]f64 = undefined;
    log_i64_f64(&a, &o, 2);
    try testing.expectApproxEqRel(o[0], @log(8.0), 1e-12);
    var o2: [2]f64 = undefined;
    log2_i64_f64(&a, &o2, 2);
    try testing.expectApproxEqAbs(o2[0], 3.0, 1e-12);
    try testing.expectApproxEqAbs(o2[1], 6.0, 1e-12);
}

test "log_c128 matches (ln|z|, atan2)" {
    const std = @import("std");
    const testing = std.testing;
    const a = [_]f64{ 3.0, 4.0, -1.0, 1.0 };
    var out: [4]f64 = undefined;
    log_c128(&a, &out, 2);
    try testing.expectApproxEqRel(out[0], @log(5.0), 1e-12);
    try testing.expectApproxEqRel(out[1], std.math.atan2(@as(f64, 4.0), @as(f64, 3.0)), 1e-12);
    try testing.expectApproxEqRel(out[3], std.math.atan2(@as(f64, 1.0), @as(f64, -1.0)), 1e-12);
}

test "log2_f64 / log10_f64 match reference" {
    const testing = @import("std").testing;
    const a = [_]f64{ 1.0, 2.0, 8.0, 10.0, 100.0, 0.5, 1234.5, 1e-8 };
    var o2: [8]f64 = undefined;
    var o10: [8]f64 = undefined;
    log2_f64(&a, &o2, 8);
    log10_f64(&a, &o10, 8);
    for (a, 0..) |x, i| {
        try testing.expectApproxEqRel(o2[i], @log2(x), 1e-12);
        try testing.expectApproxEqRel(o10[i], @log10(x), 1e-12);
    }
}

test "log_f32 / log2_f32 / log10_f32 match reference" {
    const testing = @import("std").testing;
    const a = [_]f32{ 1.0, 2.0, 8.0, 10.0, 100.0, 0.5, 1234.5, 7.0 };
    var ol: [8]f32 = undefined;
    var o2: [8]f32 = undefined;
    var o10: [8]f32 = undefined;
    log_f32(&a, &ol, 8);
    log2_f32(&a, &o2, 8);
    log10_f32(&a, &o10, 8);
    for (a, 0..) |x, i| {
        try testing.expectApproxEqAbs(ol[i], @log(x), 1e-5);
        try testing.expectApproxEqAbs(o2[i], @log2(x), 1e-5);
        try testing.expectApproxEqAbs(o10[i], @log10(x), 1e-5);
    }
}

test "log integer inputs (f32-widening)" {
    const testing = @import("std").testing;
    const i8a = [_]i8{ 1, 2, 8, 100 };
    const u8a = [_]u8{ 1, 2, 8, 200 };
    const i16a = [_]i16{ 1, 2, 8, 1000 };
    const u16a = [_]u16{ 1, 2, 8, 5000 };
    var o: [4]f32 = undefined;

    log_i8_f32(&i8a, &o, 4);
    for (i8a, 0..) |x, i| try testing.expectApproxEqAbs(o[i], @log(@as(f32, @floatFromInt(x))), 1e-5);
    log2_i8_f32(&i8a, &o, 4);
    for (i8a, 0..) |x, i| try testing.expectApproxEqAbs(o[i], @log2(@as(f32, @floatFromInt(x))), 1e-5);
    log10_i8_f32(&i8a, &o, 4);
    for (i8a, 0..) |x, i| try testing.expectApproxEqAbs(o[i], @log10(@as(f32, @floatFromInt(x))), 1e-5);

    log_u8_f32(&u8a, &o, 4);
    for (u8a, 0..) |x, i| try testing.expectApproxEqAbs(o[i], @log(@as(f32, @floatFromInt(x))), 1e-5);
    log2_u8_f32(&u8a, &o, 4);
    for (u8a, 0..) |x, i| try testing.expectApproxEqAbs(o[i], @log2(@as(f32, @floatFromInt(x))), 1e-5);
    log10_u8_f32(&u8a, &o, 4);
    for (u8a, 0..) |x, i| try testing.expectApproxEqAbs(o[i], @log10(@as(f32, @floatFromInt(x))), 1e-5);

    log_i16_f32(&i16a, &o, 4);
    for (i16a, 0..) |x, i| try testing.expectApproxEqAbs(o[i], @log(@as(f32, @floatFromInt(x))), 1e-4);
    log2_i16_f32(&i16a, &o, 4);
    for (i16a, 0..) |x, i| try testing.expectApproxEqAbs(o[i], @log2(@as(f32, @floatFromInt(x))), 1e-4);
    log10_i16_f32(&i16a, &o, 4);
    for (i16a, 0..) |x, i| try testing.expectApproxEqAbs(o[i], @log10(@as(f32, @floatFromInt(x))), 1e-4);

    log_u16_f32(&u16a, &o, 4);
    for (u16a, 0..) |x, i| try testing.expectApproxEqAbs(o[i], @log(@as(f32, @floatFromInt(x))), 1e-3);
    log2_u16_f32(&u16a, &o, 4);
    for (u16a, 0..) |x, i| try testing.expectApproxEqAbs(o[i], @log2(@as(f32, @floatFromInt(x))), 1e-3);
    log10_u16_f32(&u16a, &o, 4);
    for (u16a, 0..) |x, i| try testing.expectApproxEqAbs(o[i], @log10(@as(f32, @floatFromInt(x))), 1e-3);
}

test "log integer inputs (f64-widening)" {
    const testing = @import("std").testing;
    const i32a = [_]i32{ 1, 8, 100, 100000 };
    const u32a = [_]u32{ 1, 8, 100, 100000 };
    const i64a = [_]i64{ 1, 8, 100, 100000 };
    const u64a = [_]u64{ 1, 8, 100, 100000 };
    var o: [4]f64 = undefined;

    log_i32_f64(&i32a, &o, 4);
    for (i32a, 0..) |x, i| try testing.expectApproxEqRel(o[i], @log(@as(f64, @floatFromInt(x))), 1e-12);
    log2_i32_f64(&i32a, &o, 4);
    for (i32a, 0..) |x, i| try testing.expectApproxEqRel(o[i], @log2(@as(f64, @floatFromInt(x))), 1e-12);
    log10_i32_f64(&i32a, &o, 4);
    for (i32a, 0..) |x, i| try testing.expectApproxEqRel(o[i], @log10(@as(f64, @floatFromInt(x))), 1e-12);

    log_u32_f64(&u32a, &o, 4);
    for (u32a, 0..) |x, i| try testing.expectApproxEqRel(o[i], @log(@as(f64, @floatFromInt(x))), 1e-12);
    log2_u32_f64(&u32a, &o, 4);
    for (u32a, 0..) |x, i| try testing.expectApproxEqRel(o[i], @log2(@as(f64, @floatFromInt(x))), 1e-12);
    log10_u32_f64(&u32a, &o, 4);
    for (u32a, 0..) |x, i| try testing.expectApproxEqRel(o[i], @log10(@as(f64, @floatFromInt(x))), 1e-12);

    log_u64_f64(&u64a, &o, 4);
    for (u64a, 0..) |x, i| try testing.expectApproxEqRel(o[i], @log(@as(f64, @floatFromInt(x))), 1e-12);
    log2_u64_f64(&u64a, &o, 4);
    for (u64a, 0..) |x, i| try testing.expectApproxEqRel(o[i], @log2(@as(f64, @floatFromInt(x))), 1e-12);
    log10_i64_f64(&i64a, &o, 4);
    for (i64a, 0..) |x, i| try testing.expectApproxEqRel(o[i], @log10(@as(f64, @floatFromInt(x))), 1e-12);
    log10_u64_f64(&u64a, &o, 4);
    for (u64a, 0..) |x, i| try testing.expectApproxEqRel(o[i], @log10(@as(f64, @floatFromInt(x))), 1e-12);
}

test "complex log2/log10 (c128 & c64)" {
    const std = @import("std");
    const testing = std.testing;
    // z0 = 3+4i (|z|=5), z1 = -1+1i
    const a = [_]f64{ 3.0, 4.0, -1.0, 1.0 };
    var o2: [4]f64 = undefined;
    var o10: [4]f64 = undefined;
    log2_c128(&a, &o2, 2);
    log10_c128(&a, &o10, 2);
    try testing.expectApproxEqRel(o2[0], @log2(5.0), 1e-12);
    try testing.expectApproxEqRel(o2[1], std.math.atan2(@as(f64, 4.0), @as(f64, 3.0)) * LOG2E_F64, 1e-12);
    try testing.expectApproxEqRel(o10[0], @log10(5.0), 1e-12);
    try testing.expectApproxEqRel(o10[1], std.math.atan2(@as(f64, 4.0), @as(f64, 3.0)) * LOG10E_F64, 1e-12);

    const a32 = [_]f32{ 3.0, 4.0, -1.0, 1.0 };
    var c1: [4]f32 = undefined;
    var c2: [4]f32 = undefined;
    var c10: [4]f32 = undefined;
    log_c64(&a32, &c1, 2);
    log2_c64(&a32, &c2, 2);
    log10_c64(&a32, &c10, 2);
    try testing.expectApproxEqAbs(c1[0], @log(@as(f32, 5.0)), 1e-5);
    try testing.expectApproxEqAbs(c1[1], std.math.atan2(@as(f32, 4.0), @as(f32, 3.0)), 1e-5);
    try testing.expectApproxEqAbs(c2[0], @log2(@as(f32, 5.0)), 1e-5);
    try testing.expectApproxEqAbs(c10[0], @log10(@as(f32, 5.0)), 1e-5);
}
