//! WASM element-wise sinc: sinc(x) = sin(πx)/(πx), with sinc(0) = 1.
//!
//! Reuses the shared SIMD sine core (`transcend.sinv_f64`); the only extra work
//! is one multiply by π, one divide, and an x==0 → 1 select. Float paths are
//! 2-wide f64 (f32 widens through the f64 core). Integer inputs widen to f64
//! lanes (NumPy promotes int→float64 for sinc); i8/u8/i16/u16 emit f32.

const simd = @import("simd.zig");
const t = @import("transcend.zig");

const PI: f64 = 3.141592653589793;

/// sinc of a 2-wide f64 lane.
inline fn sincv_f64(x: simd.V2f64) simd.V2f64 {
    const pix = x * @as(simd.V2f64, @splat(PI));
    const r = t.sinv_f64(pix) / pix;
    return @select(f64, x == @as(simd.V2f64, @splat(0.0)), @as(simd.V2f64, @splat(1.0)), r);
}

export fn sinc_f64(x: [*]const f64, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        simd.store2_f64(out, i, sincv_f64(simd.load2_f64(x, i)));
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = .{ x[i], x[i] };
        out[i] = sincv_f64(v)[0];
    }
}

export fn sinc_f32(x: [*]const f32, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const v: simd.V2f64 = .{ x[i], x[i + 1] };
        const r = sincv_f64(v);
        out[i] = @floatCast(r[0]);
        out[i + 1] = @floatCast(r[1]);
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = .{ x[i], x[i] };
        out[i] = @floatCast(sincv_f64(v)[0]);
    }
}

inline fn sincIntToF64(comptime T: type, x: [*]const T, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const v: simd.V2f64 = .{ @floatFromInt(x[i]), @floatFromInt(x[i + 1]) };
        simd.store2_f64(out, i, sincv_f64(v));
    }
    while (i < N) : (i += 1) {
        const v: simd.V2f64 = @splat(@floatFromInt(x[i]));
        out[i] = sincv_f64(v)[0];
    }
}

// NumPy promotes ALL integer dtypes to float64 for sinc.
export fn sinc_i64(x: [*]const i64, out: [*]f64, N: u32) void {
    sincIntToF64(i64, x, out, N);
}
export fn sinc_u64(x: [*]const u64, out: [*]f64, N: u32) void {
    sincIntToF64(u64, x, out, N);
}
export fn sinc_i32(x: [*]const i32, out: [*]f64, N: u32) void {
    sincIntToF64(i32, x, out, N);
}
export fn sinc_u32(x: [*]const u32, out: [*]f64, N: u32) void {
    sincIntToF64(u32, x, out, N);
}
export fn sinc_i16(x: [*]const i16, out: [*]f64, N: u32) void {
    sincIntToF64(i16, x, out, N);
}
export fn sinc_u16(x: [*]const u16, out: [*]f64, N: u32) void {
    sincIntToF64(u16, x, out, N);
}
export fn sinc_i8(x: [*]const i8, out: [*]f64, N: u32) void {
    sincIntToF64(i8, x, out, N);
}
export fn sinc_u8(x: [*]const u8, out: [*]f64, N: u32) void {
    sincIntToF64(u8, x, out, N);
}

// --- Tests ---

test "sinc_f64 zero and known points" {
    const t2 = @import("std").testing;
    const a = [_]f64{ 0.0, 0.5, 1.0, 2.0 };
    var o: [4]f64 = undefined;
    sinc_f64(&a, &o, 4);
    try t2.expectApproxEqAbs(o[0], 1.0, 1e-12); // sinc(0)=1
    try t2.expectApproxEqAbs(o[1], 0.6366197723675814, 1e-12); // sin(pi/2)/(pi/2)=2/pi
    try t2.expectApproxEqAbs(o[2], 0.0, 1e-12); // sin(pi)/pi = 0
    try t2.expectApproxEqAbs(o[3], 0.0, 1e-12);
}

test "sinc_f32 basic" {
    const t2 = @import("std").testing;
    const a = [_]f32{ 0.0, 1.0 };
    var o: [2]f32 = undefined;
    sinc_f32(&a, &o, 2);
    try t2.expectApproxEqAbs(o[0], 1.0, 1e-6);
    try t2.expectApproxEqAbs(o[1], 0.0, 1e-6);
}

test "sinc_i32 widens to f64" {
    const t2 = @import("std").testing;
    const a = [_]i32{ 0, 1, 2 };
    var o: [3]f64 = undefined;
    sinc_i32(&a, &o, 3);
    try t2.expectApproxEqAbs(o[0], 1.0, 1e-12);
    try t2.expectApproxEqAbs(o[1], 0.0, 1e-12);
    try t2.expectApproxEqAbs(o[2], 0.0, 1e-12);
}

test "sinc int variants widen to f64" {
    const t2 = @import("std").testing;
    // sinc(0)=1, sinc(nonzero int)=sin(pi*n)/(pi*n)=0.
    {
        const a = [_]i64{ 0, 1, 2 };
        var o: [3]f64 = undefined;
        sinc_i64(&a, &o, 3);
        try t2.expectApproxEqAbs(o[0], 1.0, 1e-12);
        try t2.expectApproxEqAbs(o[1], 0.0, 1e-12);
        try t2.expectApproxEqAbs(o[2], 0.0, 1e-12);
    }
    {
        const a = [_]u64{ 0, 1, 2 };
        var o: [3]f64 = undefined;
        sinc_u64(&a, &o, 3);
        try t2.expectApproxEqAbs(o[0], 1.0, 1e-12);
        try t2.expectApproxEqAbs(o[1], 0.0, 1e-12);
        try t2.expectApproxEqAbs(o[2], 0.0, 1e-12);
    }
    {
        const a = [_]u32{ 0, 1, 2 };
        var o: [3]f64 = undefined;
        sinc_u32(&a, &o, 3);
        try t2.expectApproxEqAbs(o[0], 1.0, 1e-12);
        try t2.expectApproxEqAbs(o[1], 0.0, 1e-12);
        try t2.expectApproxEqAbs(o[2], 0.0, 1e-12);
    }
    {
        const a = [_]i16{ 0, 1, 2 };
        var o: [3]f64 = undefined;
        sinc_i16(&a, &o, 3);
        try t2.expectApproxEqAbs(o[0], 1.0, 1e-12);
        try t2.expectApproxEqAbs(o[1], 0.0, 1e-12);
        try t2.expectApproxEqAbs(o[2], 0.0, 1e-12);
    }
    {
        const a = [_]u16{ 0, 1, 2 };
        var o: [3]f64 = undefined;
        sinc_u16(&a, &o, 3);
        try t2.expectApproxEqAbs(o[0], 1.0, 1e-12);
        try t2.expectApproxEqAbs(o[1], 0.0, 1e-12);
        try t2.expectApproxEqAbs(o[2], 0.0, 1e-12);
    }
    {
        const a = [_]i8{ 0, 1, 2 };
        var o: [3]f64 = undefined;
        sinc_i8(&a, &o, 3);
        try t2.expectApproxEqAbs(o[0], 1.0, 1e-12);
        try t2.expectApproxEqAbs(o[1], 0.0, 1e-12);
        try t2.expectApproxEqAbs(o[2], 0.0, 1e-12);
    }
    {
        const a = [_]u8{ 0, 1, 2 };
        var o: [3]f64 = undefined;
        sinc_u8(&a, &o, 3);
        try t2.expectApproxEqAbs(o[0], 1.0, 1e-12);
        try t2.expectApproxEqAbs(o[1], 0.0, 1e-12);
        try t2.expectApproxEqAbs(o[2], 0.0, 1e-12);
    }
}
