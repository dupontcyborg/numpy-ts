// Shared SIMD vector types and load/store helpers for WASM v128.
//
// Each kernel file imports this: const simd = @import("simd.zig");
// Then uses simd.V2f64, simd.load2_f64, etc.

// ─── Vector types ────────────────────────────────────────────────────────

pub const V2f64 = @Vector(2, f64);
pub const V4f32 = @Vector(4, f32);
pub const V2u64 = @Vector(2, u64);
pub const V4u32 = @Vector(4, u32);
pub const V4i32 = @Vector(4, i32);
pub const V8i16 = @Vector(8, i16);
pub const V16i8 = @Vector(16, i8);

// ─── f64 (2-wide) ───────────────────────────────────────────────────────

pub inline fn load2_f64(ptr: [*]const f64, i: usize) V2f64 {
    return @as(*align(1) const V2f64, @ptrCast(ptr + i)).*;
}
pub inline fn store2_f64(ptr: [*]f64, i: usize, v: V2f64) void {
    @as(*align(1) V2f64, @ptrCast(ptr + i)).* = v;
}

// ─── f32 (4-wide) ───────────────────────────────────────────────────────

pub inline fn load4_f32(ptr: [*]const f32, i: usize) V4f32 {
    return @as(*align(1) const V4f32, @ptrCast(ptr + i)).*;
}
pub inline fn store4_f32(ptr: [*]f32, i: usize, v: V4f32) void {
    @as(*align(1) V4f32, @ptrCast(ptr + i)).* = v;
}

// ─── i32 (4-wide) ───────────────────────────────────────────────────────

pub inline fn load4_i32(ptr: [*]const i32, i: usize) V4i32 {
    return @as(*align(1) const V4i32, @ptrCast(ptr + i)).*;
}
pub inline fn store4_i32(ptr: [*]i32, i: usize, v: V4i32) void {
    @as(*align(1) V4i32, @ptrCast(ptr + i)).* = v;
}

// ─── i16 (8-wide) ───────────────────────────────────────────────────────

pub inline fn load8_i16(ptr: [*]const i16, i: usize) V8i16 {
    return @as(*align(1) const V8i16, @ptrCast(ptr + i)).*;
}
pub inline fn store8_i16(ptr: [*]i16, i: usize, v: V8i16) void {
    @as(*align(1) V8i16, @ptrCast(ptr + i)).* = v;
}

// ─── i8 (16-wide) ───────────────────────────────────────────────────────

pub inline fn load16_i8(ptr: [*]const i8, i: usize) V16i8 {
    return @as(*align(1) const V16i8, @ptrCast(ptr + i)).*;
}
pub inline fn store16_i8(ptr: [*]i8, i: usize, v: V16i8) void {
    @as(*align(1) V16i8, @ptrCast(ptr + i)).* = v;
}

// ─── WASM SIMD intrinsics (bypass Zig's strict @max/@min semantics) ─────
// Zig's @max/@min follow IEEE 754-2019 (llvm.maximum/llvm.minimum) which LLVM
// scalarizes on WASM because f64x2.max has different NaN payload semantics.
// These call the WASM-specific LLVM intrinsics that emit native f64x2.max etc.

// Use @select with vector comparison to avoid @max/@min scalarization.
// @max/@min use IEEE 754-2019 semantics (llvm.maximum) which gets scalarized
// on WASM because f64x2.max has different NaN behavior. @select compiles to
// f64x2.gt + v128.bitselect (2 SIMD ops) instead of per-lane scalar calls.
// NaN behavior doesn't matter here since callers wrap with their own NaN guards.

pub inline fn max_f64x2(a: V2f64, b: V2f64) V2f64 {
    return @select(f64, a > b, a, b);
}
pub inline fn min_f64x2(a: V2f64, b: V2f64) V2f64 {
    return @select(f64, a < b, a, b);
}
pub inline fn max_f32x4(a: V4f32, b: V4f32) V4f32 {
    return @select(f32, a > b, a, b);
}
pub inline fn min_f32x4(a: V4f32, b: V4f32) V4f32 {
    return @select(f32, a < b, a, b);
}
