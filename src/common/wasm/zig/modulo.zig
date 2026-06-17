//! WASM scalar-divisor kernels for the modulo family: mod (floor remainder),
//! floor_divide, and fmod (truncated remainder). Same-dtype in/out.
//!
//!   mod      : r = a − floor(a/s)·s   (NumPy: sign of divisor)
//!   floordiv : q = floor(a/s)
//!   fmod     : r = a − trunc(a/s)·s   (C fmod: sign of dividend)
//!
//! Float paths are SIMD (2-wide f64, 4-wide f32). Integer paths are scalar
//! (WASM has no SIMD integer divide) but still beat JS — especially i64/u64,
//! where the JS fallback pays BigInt costs. Division by zero writes 0, matching
//! the existing divmod kernel and NumPy's integer behavior.

const simd = @import("simd.zig");

const Op = enum { mod_, floordiv, fmod_ };

// --- Float (SIMD) ---

inline fn modF64(comptime op: Op, a: [*]const f64, out: [*]f64, N: u32, scalar: f64) void {
    const s: simd.V2f64 = @splat(scalar);
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const v = simd.load2_f64(a, i);
        const q = if (op == .fmod_) @trunc(v / s) else @floor(v / s);
        const r = if (op == .floordiv) q else simd.nmulAdd_f64x2(q, s, v); // v − q·s
        simd.store2_f64(out, i, r);
    }
    while (i < N) : (i += 1) {
        const q = if (op == .fmod_) @trunc(a[i] / scalar) else @floor(a[i] / scalar);
        out[i] = if (op == .floordiv) q else a[i] - q * scalar;
    }
}

inline fn modF32(comptime op: Op, a: [*]const f32, out: [*]f32, N: u32, scalar: f32) void {
    const s: simd.V4f32 = @splat(scalar);
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        const v = simd.load4_f32(a, i);
        const q = if (op == .fmod_) @trunc(v / s) else @floor(v / s);
        const r = if (op == .floordiv) q else simd.nmulAdd_f32x4(q, s, v);
        simd.store4_f32(out, i, r);
    }
    while (i < N) : (i += 1) {
        const q = if (op == .fmod_) @trunc(a[i] / scalar) else @floor(a[i] / scalar);
        out[i] = if (op == .floordiv) q else a[i] - q * scalar;
    }
}

export fn mod_scalar_f64(a: [*]const f64, out: [*]f64, N: u32, s: f64) void {
    modF64(.mod_, a, out, N, s);
}
export fn floordiv_scalar_f64(a: [*]const f64, out: [*]f64, N: u32, s: f64) void {
    modF64(.floordiv, a, out, N, s);
}
export fn fmod_scalar_f64(a: [*]const f64, out: [*]f64, N: u32, s: f64) void {
    modF64(.fmod_, a, out, N, s);
}
export fn mod_scalar_f32(a: [*]const f32, out: [*]f32, N: u32, s: f32) void {
    modF32(.mod_, a, out, N, s);
}
export fn floordiv_scalar_f32(a: [*]const f32, out: [*]f32, N: u32, s: f32) void {
    modF32(.floordiv, a, out, N, s);
}
export fn fmod_scalar_f32(a: [*]const f32, out: [*]f32, N: u32, s: f32) void {
    modF32(.fmod_, a, out, N, s);
}

// --- Integer (scalar; same dtype in/out) ---

inline fn modInt(comptime T: type, comptime op: Op, a: [*]const T, out: [*]T, N: u32, scalar: T) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (scalar == 0) {
            out[i] = 0;
            continue;
        }
        out[i] = switch (op) {
            .mod_ => @mod(a[i], scalar), // floor modulo (sign of divisor)
            .floordiv => @divFloor(a[i], scalar),
            .fmod_ => @rem(a[i], scalar), // truncated (sign of dividend)
        };
    }
}

export fn mod_scalar_i64(a: [*]const i64, out: [*]i64, N: u32, s: i64) void {
    modInt(i64, .mod_, a, out, N, s);
}
export fn floordiv_scalar_i64(a: [*]const i64, out: [*]i64, N: u32, s: i64) void {
    modInt(i64, .floordiv, a, out, N, s);
}
export fn fmod_scalar_i64(a: [*]const i64, out: [*]i64, N: u32, s: i64) void {
    modInt(i64, .fmod_, a, out, N, s);
}
export fn mod_scalar_u64(a: [*]const u64, out: [*]u64, N: u32, s: u64) void {
    modInt(u64, .mod_, a, out, N, s);
}
export fn floordiv_scalar_u64(a: [*]const u64, out: [*]u64, N: u32, s: u64) void {
    modInt(u64, .floordiv, a, out, N, s);
}
export fn fmod_scalar_u64(a: [*]const u64, out: [*]u64, N: u32, s: u64) void {
    modInt(u64, .fmod_, a, out, N, s);
}
export fn mod_scalar_i32(a: [*]const i32, out: [*]i32, N: u32, s: i32) void {
    modInt(i32, .mod_, a, out, N, s);
}
export fn floordiv_scalar_i32(a: [*]const i32, out: [*]i32, N: u32, s: i32) void {
    modInt(i32, .floordiv, a, out, N, s);
}
export fn fmod_scalar_i32(a: [*]const i32, out: [*]i32, N: u32, s: i32) void {
    modInt(i32, .fmod_, a, out, N, s);
}
export fn mod_scalar_u32(a: [*]const u32, out: [*]u32, N: u32, s: u32) void {
    modInt(u32, .mod_, a, out, N, s);
}
export fn floordiv_scalar_u32(a: [*]const u32, out: [*]u32, N: u32, s: u32) void {
    modInt(u32, .floordiv, a, out, N, s);
}
export fn fmod_scalar_u32(a: [*]const u32, out: [*]u32, N: u32, s: u32) void {
    modInt(u32, .fmod_, a, out, N, s);
}
export fn mod_scalar_i16(a: [*]const i16, out: [*]i16, N: u32, s: i16) void {
    modInt(i16, .mod_, a, out, N, s);
}
export fn floordiv_scalar_i16(a: [*]const i16, out: [*]i16, N: u32, s: i16) void {
    modInt(i16, .floordiv, a, out, N, s);
}
export fn fmod_scalar_i16(a: [*]const i16, out: [*]i16, N: u32, s: i16) void {
    modInt(i16, .fmod_, a, out, N, s);
}
export fn mod_scalar_u16(a: [*]const u16, out: [*]u16, N: u32, s: u16) void {
    modInt(u16, .mod_, a, out, N, s);
}
export fn floordiv_scalar_u16(a: [*]const u16, out: [*]u16, N: u32, s: u16) void {
    modInt(u16, .floordiv, a, out, N, s);
}
export fn fmod_scalar_u16(a: [*]const u16, out: [*]u16, N: u32, s: u16) void {
    modInt(u16, .fmod_, a, out, N, s);
}
export fn mod_scalar_i8(a: [*]const i8, out: [*]i8, N: u32, s: i8) void {
    modInt(i8, .mod_, a, out, N, s);
}
export fn floordiv_scalar_i8(a: [*]const i8, out: [*]i8, N: u32, s: i8) void {
    modInt(i8, .floordiv, a, out, N, s);
}
export fn fmod_scalar_i8(a: [*]const i8, out: [*]i8, N: u32, s: i8) void {
    modInt(i8, .fmod_, a, out, N, s);
}
export fn mod_scalar_u8(a: [*]const u8, out: [*]u8, N: u32, s: u8) void {
    modInt(u8, .mod_, a, out, N, s);
}
export fn floordiv_scalar_u8(a: [*]const u8, out: [*]u8, N: u32, s: u8) void {
    modInt(u8, .floordiv, a, out, N, s);
}
export fn fmod_scalar_u8(a: [*]const u8, out: [*]u8, N: u32, s: u8) void {
    modInt(u8, .fmod_, a, out, N, s);
}

// --- Tests ---

test "mod_scalar_f64 floor modulo sign" {
    const t = @import("std").testing;
    const a = [_]f64{ 7, -7, 7, -7 };
    var o: [4]f64 = undefined;
    mod_scalar_f64(&a, &o, 4, 3);
    // np.mod(7,3)=1, np.mod(-7,3)=2, np.mod(7,-3)=-2, np.mod(-7,-3)=-1
    try t.expectApproxEqAbs(o[0], 1, 1e-12);
    try t.expectApproxEqAbs(o[1], 2, 1e-12);
    mod_scalar_f64(&a, &o, 4, -3);
    try t.expectApproxEqAbs(o[0], -2, 1e-12);
    try t.expectApproxEqAbs(o[1], -1, 1e-12);
}

test "floordiv_scalar_f64 / fmod_scalar_f64" {
    const t = @import("std").testing;
    const a = [_]f64{ 7, -7, 8, 9 };
    var o: [4]f64 = undefined;
    floordiv_scalar_f64(&a, &o, 4, 3);
    try t.expectApproxEqAbs(o[0], 2, 1e-12); // floor(7/3)
    try t.expectApproxEqAbs(o[1], -3, 1e-12); // floor(-7/3)
    fmod_scalar_f64(&a, &o, 4, 3);
    try t.expectApproxEqAbs(o[0], 1, 1e-12); // fmod(7,3)
    try t.expectApproxEqAbs(o[1], -1, 1e-12); // fmod(-7,3) sign of dividend
}

test "mod_scalar_f32 basic" {
    const t = @import("std").testing;
    const a = [_]f32{ 5, 6, 7, 8, 9 };
    var o: [5]f32 = undefined;
    mod_scalar_f32(&a, &o, 5, 4);
    try t.expectApproxEqAbs(o[0], 1, 1e-5);
    try t.expectApproxEqAbs(o[4], 1, 1e-5);
}

test "modInt i32 mod/floordiv/fmod" {
    const t = @import("std").testing;
    const a = [_]i32{ 7, -7 };
    var o: [2]i32 = undefined;
    modInt(i32, .mod_, &a, &o, 2, 3);
    try t.expectEqual(o[0], 1);
    try t.expectEqual(o[1], 2); // floor modulo
    modInt(i32, .fmod_, &a, &o, 2, 3);
    try t.expectEqual(o[0], 1);
    try t.expectEqual(o[1], -1); // truncated
    modInt(i32, .floordiv, &a, &o, 2, 3);
    try t.expectEqual(o[0], 2);
    try t.expectEqual(o[1], -3);
}

test "modInt div by zero -> 0" {
    const t = @import("std").testing;
    const a = [_]i32{ 5, 9 };
    var o: [2]i32 = undefined;
    modInt(i32, .mod_, &a, &o, 2, 0);
    try t.expectEqual(o[0], 0);
    try t.expectEqual(o[1], 0);
}
