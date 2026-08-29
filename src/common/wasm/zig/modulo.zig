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

// --- Array / array (same dtype, same shape) ---

inline fn modF64Arr(comptime op: Op, a: [*]const f64, b: [*]const f64, out: [*]f64, N: u32) void {
    const n_simd = N & ~@as(u32, 1);
    var i: u32 = 0;
    while (i < n_simd) : (i += 2) {
        const v = simd.load2_f64(a, i);
        const s = simd.load2_f64(b, i);
        const q = if (op == .fmod_) @trunc(v / s) else @floor(v / s);
        const r = if (op == .floordiv) q else simd.nmulAdd_f64x2(q, s, v); // v − q·s
        simd.store2_f64(out, i, r);
    }
    while (i < N) : (i += 1) {
        const q = if (op == .fmod_) @trunc(a[i] / b[i]) else @floor(a[i] / b[i]);
        out[i] = if (op == .floordiv) q else a[i] - q * b[i];
    }
}

inline fn modF32Arr(comptime op: Op, a: [*]const f32, b: [*]const f32, out: [*]f32, N: u32) void {
    const n_simd = N & ~@as(u32, 3);
    var i: u32 = 0;
    while (i < n_simd) : (i += 4) {
        const v = simd.load4_f32(a, i);
        const s = simd.load4_f32(b, i);
        const q = if (op == .fmod_) @trunc(v / s) else @floor(v / s);
        const r = if (op == .floordiv) q else simd.nmulAdd_f32x4(q, s, v);
        simd.store4_f32(out, i, r);
    }
    while (i < N) : (i += 1) {
        const q = if (op == .fmod_) @trunc(a[i] / b[i]) else @floor(a[i] / b[i]);
        out[i] = if (op == .floordiv) q else a[i] - q * b[i];
    }
}

inline fn modIntArr(comptime T: type, comptime op: Op, a: [*]const T, b: [*]const T, out: [*]T, N: u32) void {
    var i: u32 = 0;
    while (i < N) : (i += 1) {
        if (b[i] == 0) {
            out[i] = 0; // matches the scalar path and NumPy integer behaviour
            continue;
        }
        out[i] = switch (op) {
            .mod_ => @mod(a[i], b[i]),
            .floordiv => @divFloor(a[i], b[i]),
            .fmod_ => @rem(a[i], b[i]),
        };
    }
}

export fn mod_f64(a: [*]const f64, b: [*]const f64, out: [*]f64, N: u32) void {
    modF64Arr(.mod_, a, b, out, N);
}
export fn floordiv_f64(a: [*]const f64, b: [*]const f64, out: [*]f64, N: u32) void {
    modF64Arr(.floordiv, a, b, out, N);
}
export fn fmod_f64(a: [*]const f64, b: [*]const f64, out: [*]f64, N: u32) void {
    modF64Arr(.fmod_, a, b, out, N);
}
export fn mod_f32(a: [*]const f32, b: [*]const f32, out: [*]f32, N: u32) void {
    modF32Arr(.mod_, a, b, out, N);
}
export fn floordiv_f32(a: [*]const f32, b: [*]const f32, out: [*]f32, N: u32) void {
    modF32Arr(.floordiv, a, b, out, N);
}
export fn fmod_f32(a: [*]const f32, b: [*]const f32, out: [*]f32, N: u32) void {
    modF32Arr(.fmod_, a, b, out, N);
}
export fn mod_i64(a: [*]const i64, b: [*]const i64, out: [*]i64, N: u32) void {
    modIntArr(i64, .mod_, a, b, out, N);
}
export fn floordiv_i64(a: [*]const i64, b: [*]const i64, out: [*]i64, N: u32) void {
    modIntArr(i64, .floordiv, a, b, out, N);
}
export fn fmod_i64(a: [*]const i64, b: [*]const i64, out: [*]i64, N: u32) void {
    modIntArr(i64, .fmod_, a, b, out, N);
}
export fn mod_u64(a: [*]const u64, b: [*]const u64, out: [*]u64, N: u32) void {
    modIntArr(u64, .mod_, a, b, out, N);
}
export fn floordiv_u64(a: [*]const u64, b: [*]const u64, out: [*]u64, N: u32) void {
    modIntArr(u64, .floordiv, a, b, out, N);
}
export fn fmod_u64(a: [*]const u64, b: [*]const u64, out: [*]u64, N: u32) void {
    modIntArr(u64, .fmod_, a, b, out, N);
}
export fn mod_i32(a: [*]const i32, b: [*]const i32, out: [*]i32, N: u32) void {
    modIntArr(i32, .mod_, a, b, out, N);
}
export fn floordiv_i32(a: [*]const i32, b: [*]const i32, out: [*]i32, N: u32) void {
    modIntArr(i32, .floordiv, a, b, out, N);
}
export fn fmod_i32(a: [*]const i32, b: [*]const i32, out: [*]i32, N: u32) void {
    modIntArr(i32, .fmod_, a, b, out, N);
}
export fn mod_u32(a: [*]const u32, b: [*]const u32, out: [*]u32, N: u32) void {
    modIntArr(u32, .mod_, a, b, out, N);
}
export fn floordiv_u32(a: [*]const u32, b: [*]const u32, out: [*]u32, N: u32) void {
    modIntArr(u32, .floordiv, a, b, out, N);
}
export fn fmod_u32(a: [*]const u32, b: [*]const u32, out: [*]u32, N: u32) void {
    modIntArr(u32, .fmod_, a, b, out, N);
}
export fn mod_i16(a: [*]const i16, b: [*]const i16, out: [*]i16, N: u32) void {
    modIntArr(i16, .mod_, a, b, out, N);
}
export fn floordiv_i16(a: [*]const i16, b: [*]const i16, out: [*]i16, N: u32) void {
    modIntArr(i16, .floordiv, a, b, out, N);
}
export fn fmod_i16(a: [*]const i16, b: [*]const i16, out: [*]i16, N: u32) void {
    modIntArr(i16, .fmod_, a, b, out, N);
}
export fn mod_u16(a: [*]const u16, b: [*]const u16, out: [*]u16, N: u32) void {
    modIntArr(u16, .mod_, a, b, out, N);
}
export fn floordiv_u16(a: [*]const u16, b: [*]const u16, out: [*]u16, N: u32) void {
    modIntArr(u16, .floordiv, a, b, out, N);
}
export fn fmod_u16(a: [*]const u16, b: [*]const u16, out: [*]u16, N: u32) void {
    modIntArr(u16, .fmod_, a, b, out, N);
}
export fn mod_i8(a: [*]const i8, b: [*]const i8, out: [*]i8, N: u32) void {
    modIntArr(i8, .mod_, a, b, out, N);
}
export fn floordiv_i8(a: [*]const i8, b: [*]const i8, out: [*]i8, N: u32) void {
    modIntArr(i8, .floordiv, a, b, out, N);
}
export fn fmod_i8(a: [*]const i8, b: [*]const i8, out: [*]i8, N: u32) void {
    modIntArr(i8, .fmod_, a, b, out, N);
}
export fn mod_u8(a: [*]const u8, b: [*]const u8, out: [*]u8, N: u32) void {
    modIntArr(u8, .mod_, a, b, out, N);
}
export fn floordiv_u8(a: [*]const u8, b: [*]const u8, out: [*]u8, N: u32) void {
    modIntArr(u8, .floordiv, a, b, out, N);
}
export fn fmod_u8(a: [*]const u8, b: [*]const u8, out: [*]u8, N: u32) void {
    modIntArr(u8, .fmod_, a, b, out, N);
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

test "floordiv_scalar_f32 / fmod_scalar_f32" {
    const t = @import("std").testing;
    const a = [_]f32{ 7, -7, 8, 9, 10 };
    var o: [5]f32 = undefined;
    floordiv_scalar_f32(&a, &o, 5, 3);
    try t.expectApproxEqAbs(o[0], 2, 1e-5); // floor(7/3)
    try t.expectApproxEqAbs(o[1], -3, 1e-5); // floor(-7/3)
    fmod_scalar_f32(&a, &o, 5, 3);
    try t.expectApproxEqAbs(o[0], 1, 1e-5); // fmod(7,3)
    try t.expectApproxEqAbs(o[1], -1, 1e-5); // fmod(-7,3) sign of dividend
}

// --- Signed integer scalar variants ---
// a=[7,-7], s=3: mod=[1,2], fmod=[1,-1], floordiv=[2,-3]

test "mod/floordiv/fmod scalar i64" {
    const t = @import("std").testing;
    const a = [_]i64{ 7, -7 };
    var o: [2]i64 = undefined;
    mod_scalar_i64(&a, &o, 2, 3);
    try t.expectEqual(o[0], 1);
    try t.expectEqual(o[1], 2);
    fmod_scalar_i64(&a, &o, 2, 3);
    try t.expectEqual(o[0], 1);
    try t.expectEqual(o[1], -1);
    floordiv_scalar_i64(&a, &o, 2, 3);
    try t.expectEqual(o[0], 2);
    try t.expectEqual(o[1], -3);
}

test "mod/floordiv/fmod scalar i32" {
    const t = @import("std").testing;
    const a = [_]i32{ 7, -7 };
    var o: [2]i32 = undefined;
    mod_scalar_i32(&a, &o, 2, 3);
    try t.expectEqual(o[0], 1);
    try t.expectEqual(o[1], 2);
    fmod_scalar_i32(&a, &o, 2, 3);
    try t.expectEqual(o[0], 1);
    try t.expectEqual(o[1], -1);
    floordiv_scalar_i32(&a, &o, 2, 3);
    try t.expectEqual(o[0], 2);
    try t.expectEqual(o[1], -3);
}

test "mod/floordiv/fmod scalar i16" {
    const t = @import("std").testing;
    const a = [_]i16{ 7, -7 };
    var o: [2]i16 = undefined;
    mod_scalar_i16(&a, &o, 2, 3);
    try t.expectEqual(o[0], 1);
    try t.expectEqual(o[1], 2);
    fmod_scalar_i16(&a, &o, 2, 3);
    try t.expectEqual(o[0], 1);
    try t.expectEqual(o[1], -1);
    floordiv_scalar_i16(&a, &o, 2, 3);
    try t.expectEqual(o[0], 2);
    try t.expectEqual(o[1], -3);
}

test "mod/floordiv/fmod scalar i8" {
    const t = @import("std").testing;
    const a = [_]i8{ 7, -7 };
    var o: [2]i8 = undefined;
    mod_scalar_i8(&a, &o, 2, 3);
    try t.expectEqual(o[0], 1);
    try t.expectEqual(o[1], 2);
    fmod_scalar_i8(&a, &o, 2, 3);
    try t.expectEqual(o[0], 1);
    try t.expectEqual(o[1], -1);
    floordiv_scalar_i8(&a, &o, 2, 3);
    try t.expectEqual(o[0], 2);
    try t.expectEqual(o[1], -3);
}

// --- Unsigned integer scalar variants ---
// a=[7,8], s=3: mod=[1,2], fmod=[1,2], floordiv=[2,2]

test "mod/floordiv/fmod scalar u64" {
    const t = @import("std").testing;
    const a = [_]u64{ 7, 8 };
    var o: [2]u64 = undefined;
    mod_scalar_u64(&a, &o, 2, 3);
    try t.expectEqual(o[0], 1);
    try t.expectEqual(o[1], 2);
    fmod_scalar_u64(&a, &o, 2, 3);
    try t.expectEqual(o[0], 1);
    try t.expectEqual(o[1], 2);
    floordiv_scalar_u64(&a, &o, 2, 3);
    try t.expectEqual(o[0], 2);
    try t.expectEqual(o[1], 2);
}

test "mod/floordiv/fmod scalar u32" {
    const t = @import("std").testing;
    const a = [_]u32{ 7, 8 };
    var o: [2]u32 = undefined;
    mod_scalar_u32(&a, &o, 2, 3);
    try t.expectEqual(o[0], 1);
    try t.expectEqual(o[1], 2);
    fmod_scalar_u32(&a, &o, 2, 3);
    try t.expectEqual(o[0], 1);
    try t.expectEqual(o[1], 2);
    floordiv_scalar_u32(&a, &o, 2, 3);
    try t.expectEqual(o[0], 2);
    try t.expectEqual(o[1], 2);
}

test "mod/floordiv/fmod scalar u16" {
    const t = @import("std").testing;
    const a = [_]u16{ 7, 8 };
    var o: [2]u16 = undefined;
    mod_scalar_u16(&a, &o, 2, 3);
    try t.expectEqual(o[0], 1);
    try t.expectEqual(o[1], 2);
    fmod_scalar_u16(&a, &o, 2, 3);
    try t.expectEqual(o[0], 1);
    try t.expectEqual(o[1], 2);
    floordiv_scalar_u16(&a, &o, 2, 3);
    try t.expectEqual(o[0], 2);
    try t.expectEqual(o[1], 2);
}

test "mod/floordiv/fmod scalar u8" {
    const t = @import("std").testing;
    const a = [_]u8{ 7, 8 };
    var o: [2]u8 = undefined;
    mod_scalar_u8(&a, &o, 2, 3);
    try t.expectEqual(o[0], 1);
    try t.expectEqual(o[1], 2);
    fmod_scalar_u8(&a, &o, 2, 3);
    try t.expectEqual(o[0], 1);
    try t.expectEqual(o[1], 2);
    floordiv_scalar_u8(&a, &o, 2, 3);
    try t.expectEqual(o[0], 2);
    try t.expectEqual(o[1], 2);
}

// --- Array / array variants ---
// Signed integers use a=[7,-7,8,-8,5], b=[3,3,-3,-3,1]; unsigned use
// a=[7,8,9,10,11], b=[3,3,4,5,4]. Expected values match NumPy 2.3.1.

test "mod/floordiv/fmod arrays f64 with a scalar tail" {
    const t = @import("std").testing;
    const a = [_]f64{ 7, -7, 7.5, -7.5, 3.0 }; // odd length -> tail
    const b = [_]f64{ 3, 3, 2, 2, 1 };
    var o: [5]f64 = undefined;
    mod_f64(&a, &b, &o, 5);
    try t.expectEqualSlices(f64, &[_]f64{ 1, 2, 1.5, 0.5, 0 }, &o);
    floordiv_f64(&a, &b, &o, 5);
    try t.expectEqualSlices(f64, &[_]f64{ 2, -3, 3, -4, 3 }, &o);
    fmod_f64(&a, &b, &o, 5);
    try t.expectEqualSlices(f64, &[_]f64{ 1, -1, 1.5, -1.5, 0 }, &o);
}

test "mod/floordiv/fmod arrays f32 with a scalar tail" {
    const t = @import("std").testing;
    const a = [_]f32{ 7, -7, 7.5, -7.5, 3.0 };
    const b = [_]f32{ 3, 3, 2, 2, 1 };
    var o: [5]f32 = undefined;
    mod_f32(&a, &b, &o, 5);
    try t.expectEqualSlices(f32, &[_]f32{ 1, 2, 1.5, 0.5, 0 }, &o);
    floordiv_f32(&a, &b, &o, 5);
    try t.expectEqualSlices(f32, &[_]f32{ 2, -3, 3, -4, 3 }, &o);
    fmod_f32(&a, &b, &o, 5);
    try t.expectEqualSlices(f32, &[_]f32{ 1, -1, 1.5, -1.5, 0 }, &o);
}

test "mod/floordiv/fmod arrays i64" {
    const t = @import("std").testing;
    const a = [_]i64{ 7, -7, 8, -8, 5 };
    const b = [_]i64{ 3, 3, -3, -3, 1 };
    var o: [5]i64 = undefined;
    mod_i64(&a, &b, &o, 5);
    try t.expectEqualSlices(i64, &[_]i64{ 1, 2, -1, -2, 0 }, &o);
    floordiv_i64(&a, &b, &o, 5);
    try t.expectEqualSlices(i64, &[_]i64{ 2, -3, -3, 2, 5 }, &o);
    fmod_i64(&a, &b, &o, 5);
    try t.expectEqualSlices(i64, &[_]i64{ 1, -1, 2, -2, 0 }, &o);
}

test "mod/floordiv/fmod arrays i32" {
    const t = @import("std").testing;
    const a = [_]i32{ 7, -7, 8, -8, 5 };
    const b = [_]i32{ 3, 3, -3, -3, 1 };
    var o: [5]i32 = undefined;
    mod_i32(&a, &b, &o, 5);
    try t.expectEqualSlices(i32, &[_]i32{ 1, 2, -1, -2, 0 }, &o);
    floordiv_i32(&a, &b, &o, 5);
    try t.expectEqualSlices(i32, &[_]i32{ 2, -3, -3, 2, 5 }, &o);
    fmod_i32(&a, &b, &o, 5);
    try t.expectEqualSlices(i32, &[_]i32{ 1, -1, 2, -2, 0 }, &o);
}

test "mod/floordiv/fmod arrays i16" {
    const t = @import("std").testing;
    const a = [_]i16{ 7, -7, 8, -8, 5 };
    const b = [_]i16{ 3, 3, -3, -3, 1 };
    var o: [5]i16 = undefined;
    mod_i16(&a, &b, &o, 5);
    try t.expectEqualSlices(i16, &[_]i16{ 1, 2, -1, -2, 0 }, &o);
    floordiv_i16(&a, &b, &o, 5);
    try t.expectEqualSlices(i16, &[_]i16{ 2, -3, -3, 2, 5 }, &o);
    fmod_i16(&a, &b, &o, 5);
    try t.expectEqualSlices(i16, &[_]i16{ 1, -1, 2, -2, 0 }, &o);
}

test "mod/floordiv/fmod arrays i8" {
    const t = @import("std").testing;
    const a = [_]i8{ 7, -7, 8, -8, 5 };
    const b = [_]i8{ 3, 3, -3, -3, 1 };
    var o: [5]i8 = undefined;
    mod_i8(&a, &b, &o, 5);
    try t.expectEqualSlices(i8, &[_]i8{ 1, 2, -1, -2, 0 }, &o);
    floordiv_i8(&a, &b, &o, 5);
    try t.expectEqualSlices(i8, &[_]i8{ 2, -3, -3, 2, 5 }, &o);
    fmod_i8(&a, &b, &o, 5);
    try t.expectEqualSlices(i8, &[_]i8{ 1, -1, 2, -2, 0 }, &o);
}

test "mod/floordiv/fmod arrays u64" {
    const t = @import("std").testing;
    const a = [_]u64{ 7, 8, 9, 10, 11 };
    const b = [_]u64{ 3, 3, 4, 5, 4 };
    var o: [5]u64 = undefined;
    mod_u64(&a, &b, &o, 5);
    try t.expectEqualSlices(u64, &[_]u64{ 1, 2, 1, 0, 3 }, &o);
    floordiv_u64(&a, &b, &o, 5);
    try t.expectEqualSlices(u64, &[_]u64{ 2, 2, 2, 2, 2 }, &o);
    fmod_u64(&a, &b, &o, 5);
    try t.expectEqualSlices(u64, &[_]u64{ 1, 2, 1, 0, 3 }, &o);
}

test "mod/floordiv/fmod arrays u32" {
    const t = @import("std").testing;
    const a = [_]u32{ 7, 8, 9, 10, 11 };
    const b = [_]u32{ 3, 3, 4, 5, 4 };
    var o: [5]u32 = undefined;
    mod_u32(&a, &b, &o, 5);
    try t.expectEqualSlices(u32, &[_]u32{ 1, 2, 1, 0, 3 }, &o);
    floordiv_u32(&a, &b, &o, 5);
    try t.expectEqualSlices(u32, &[_]u32{ 2, 2, 2, 2, 2 }, &o);
    fmod_u32(&a, &b, &o, 5);
    try t.expectEqualSlices(u32, &[_]u32{ 1, 2, 1, 0, 3 }, &o);
}

test "mod/floordiv/fmod arrays u16" {
    const t = @import("std").testing;
    const a = [_]u16{ 7, 8, 9, 10, 11 };
    const b = [_]u16{ 3, 3, 4, 5, 4 };
    var o: [5]u16 = undefined;
    mod_u16(&a, &b, &o, 5);
    try t.expectEqualSlices(u16, &[_]u16{ 1, 2, 1, 0, 3 }, &o);
    floordiv_u16(&a, &b, &o, 5);
    try t.expectEqualSlices(u16, &[_]u16{ 2, 2, 2, 2, 2 }, &o);
    fmod_u16(&a, &b, &o, 5);
    try t.expectEqualSlices(u16, &[_]u16{ 1, 2, 1, 0, 3 }, &o);
}

test "mod/floordiv/fmod arrays u8" {
    const t = @import("std").testing;
    const a = [_]u8{ 7, 8, 9, 10, 11 };
    const b = [_]u8{ 3, 3, 4, 5, 4 };
    var o: [5]u8 = undefined;
    mod_u8(&a, &b, &o, 5);
    try t.expectEqualSlices(u8, &[_]u8{ 1, 2, 1, 0, 3 }, &o);
    floordiv_u8(&a, &b, &o, 5);
    try t.expectEqualSlices(u8, &[_]u8{ 2, 2, 2, 2, 2 }, &o);
    fmod_u8(&a, &b, &o, 5);
    try t.expectEqualSlices(u8, &[_]u8{ 1, 2, 1, 0, 3 }, &o);
}

test "integer array division by zero writes 0 per element" {
    const t = @import("std").testing;
    const a = [_]i32{ 5, 9, -3 };
    const b = [_]i32{ 0, 2, 0 };
    var o: [3]i32 = undefined;
    mod_i32(&a, &b, &o, 3);
    try t.expectEqualSlices(i32, &[_]i32{ 0, 1, 0 }, &o);
    floordiv_i32(&a, &b, &o, 3);
    try t.expectEqualSlices(i32, &[_]i32{ 0, 4, 0 }, &o);
    fmod_i32(&a, &b, &o, 3);
    try t.expectEqualSlices(i32, &[_]i32{ 0, 1, 0 }, &o);
}
