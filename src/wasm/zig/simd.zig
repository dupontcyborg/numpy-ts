//! Shared SIMD vector types and load/store helpers for WASM v128.

// --- Vector types ---

pub const V2f64 = @Vector(2, f64);
pub const V4f32 = @Vector(4, f32);
pub const V2i64 = @Vector(2, i64);
pub const V4i32 = @Vector(4, i32);
pub const V8i16 = @Vector(8, i16);
pub const V16i8 = @Vector(16, i8);
pub const V2u64 = @Vector(2, u64);
pub const V4u32 = @Vector(4, u32);
pub const V8u16 = @Vector(8, u16);
pub const V16u8 = @Vector(16, u8);

// --- f64 (2-wide) ---

/// Returns a V2f64 (2-wide f64) loaded from an unaligned memory address.
/// Uses an align(1) cast to support unaligned access.
pub inline fn load2_f64(ptr: [*]const f64, i: usize) V2f64 {
    return @as(*align(1) const V2f64, @ptrCast(ptr + i)).*;
}

/// Stores a V2f64 (2-wide f64) to an unaligned memory address.
/// Uses an align(1) cast to support unaligned access.
pub inline fn store2_f64(ptr: [*]f64, i: usize, v: V2f64) void {
    @as(*align(1) V2f64, @ptrCast(ptr + i)).* = v;
}

// --- f32 (4-wide) ---

/// Returns a V4f32 (4-wide f32) loaded from an unaligned memory address.
/// Uses an align(1) cast to support unaligned access.
pub inline fn load4_f32(ptr: [*]const f32, i: usize) V4f32 {
    return @as(*align(1) const V4f32, @ptrCast(ptr + i)).*;
}

/// Stores a V4f32 (4-wide f32) to an unaligned memory address.
/// Uses an align(1) cast to support unaligned access.
pub inline fn store4_f32(ptr: [*]f32, i: usize, v: V4f32) void {
    @as(*align(1) V4f32, @ptrCast(ptr + i)).* = v;
}

// --- i64 (2-wide) ---

/// Returns a V2i64 (2-wide i64) loaded from an unaligned memory address.
/// Uses an align(1) cast to support unaligned access.
pub inline fn load2_i64(ptr: [*]const i64, i: usize) V2i64 {
    return @as(*align(1) const V2i64, @ptrCast(ptr + i)).*;
}

/// Stores a V2i64 (2-wide i64) to an unaligned memory address.
/// Uses an align(1) cast to support unaligned access.
pub inline fn store2_i64(ptr: [*]i64, i: usize, v: V2i64) void {
    @as(*align(1) V2i64, @ptrCast(ptr + i)).* = v;
}

// --- i32 (4-wide) ---

/// Returns a V4i32 (4-wide i32) loaded from an unaligned memory address.
/// Uses an align(1) cast to support unaligned access.
pub inline fn load4_i32(ptr: [*]const i32, i: usize) V4i32 {
    return @as(*align(1) const V4i32, @ptrCast(ptr + i)).*;
}

/// Stores a V4i32 (4-wide i32) to an unaligned memory address.
/// Uses an align(1) cast to support unaligned access.
pub inline fn store4_i32(ptr: [*]i32, i: usize, v: V4i32) void {
    @as(*align(1) V4i32, @ptrCast(ptr + i)).* = v;
}

// --- i16 (8-wide) ---

/// Returns a V8i16 (8-wide i16) loaded from an unaligned memory address.
/// Uses an align(1) cast to support unaligned access.
pub inline fn load8_i16(ptr: [*]const i16, i: usize) V8i16 {
    return @as(*align(1) const V8i16, @ptrCast(ptr + i)).*;
}

/// Stores a V8i16 (8-wide i16) to an unaligned memory address.
/// Uses an align(1) cast to support unaligned access.
pub inline fn store8_i16(ptr: [*]i16, i: usize, v: V8i16) void {
    @as(*align(1) V8i16, @ptrCast(ptr + i)).* = v;
}

// --- i8 (16-wide) ---

/// Returns a V16i8 (16-wide i8) loaded from an unaligned memory address.
/// Uses an align(1) cast to support unaligned access.
pub inline fn load16_i8(ptr: [*]const i8, i: usize) V16i8 {
    return @as(*align(1) const V16i8, @ptrCast(ptr + i)).*;
}

/// Stores a V16i8 (16-wide i8) to an unaligned memory address.
/// Uses an align(1) cast to support unaligned access.
pub inline fn store16_i8(ptr: [*]i8, i: usize, v: V16i8) void {
    @as(*align(1) V16i8, @ptrCast(ptr + i)).* = v;
}

// --- u64 (2-wide) ---

/// Returns a V2u64 (2-wide u64) loaded from an unaligned memory address.
/// Uses an align(1) cast to support unaligned access.
pub inline fn load2_u64(ptr: [*]const u64, i: usize) V2u64 {
    return @as(*align(1) const V2u64, @ptrCast(ptr + i)).*;
}

/// Stores a V2u64 (2-wide u64) to an unaligned memory address.
/// Uses an align(1) cast to support unaligned access.
pub inline fn store2_u64(ptr: [*]u64, i: usize, v: V2u64) void {
    @as(*align(1) V2u64, @ptrCast(ptr + i)).* = v;
}

// --- u32 (4-wide) ---

/// Returns a V4u32 (4-wide u32) loaded from an unaligned memory address.
/// Uses an align(1) cast to support unaligned access.
pub inline fn load4_u32(ptr: [*]const u32, i: usize) V4u32 {
    return @as(*align(1) const V4u32, @ptrCast(ptr + i)).*;
}

/// Stores a V4u32 (4-wide u32) to an unaligned memory address.
/// Uses an align(1) cast to support unaligned access.
pub inline fn store4_u32(ptr: [*]u32, i: usize, v: V4u32) void {
    @as(*align(1) V4u32, @ptrCast(ptr + i)).* = v;
}

// --- u16 (8-wide) ---

/// Returns a V8u16 (8-wide u16) loaded from an unaligned memory address.
/// Uses an align(1) cast to support unaligned access.
pub inline fn load8_u16(ptr: [*]const u16, i: usize) V8u16 {
    return @as(*align(1) const V8u16, @ptrCast(ptr + i)).*;
}

/// Stores a V8u16 (8-wide u16) to an unaligned memory address.
/// Uses an align(1) cast to support unaligned access.
pub inline fn store8_u16(ptr: [*]u16, i: usize, v: V8u16) void {
    @as(*align(1) V8u16, @ptrCast(ptr + i)).* = v;
}

// --- u8 (16-wide) ---

/// Returns a V16u8 (16-wide u8) loaded from an unaligned memory address.
/// Uses an align(1) cast to support unaligned access.
pub inline fn load16_u8(ptr: [*]const u8, i: usize) V16u8 {
    return @as(*align(1) const V16u8, @ptrCast(ptr + i)).*;
}

/// Stores a V16u8 (16-wide u8) to an unaligned memory address.
/// Uses an align(1) cast to support unaligned access.
pub inline fn store16_u8(ptr: [*]u8, i: usize, v: V16u8) void {
    @as(*align(1) V16u8, @ptrCast(ptr + i)).* = v;
}

// --- WASM SIMD min/max ---

/// Returns the element-wise max of two V2f64 vectors.
/// Uses @select to prevent LLVM from scalarizing on WASM targets.
/// `@max` uses IEEE 754-2019 semantics which LLVM lowers to scalar calls;
/// `@select` compiles to `f64x2.gt + v128.bitselect` (2 SIMD ops).
pub inline fn max_f64x2(a: V2f64, b: V2f64) V2f64 {
    return @select(f64, a > b, a, b);
}

/// Returns the element-wise min of two V2f64 vectors.
/// Uses @select to prevent LLVM from scalarizing on WASM targets.
/// `@min` uses IEEE 754-2019 semantics which LLVM lowers to scalar calls;
/// `@select` compiles to `f64x2.lt + v128.bitselect` (2 SIMD ops).
pub inline fn min_f64x2(a: V2f64, b: V2f64) V2f64 {
    return @select(f64, a < b, a, b);
}

/// Returns the element-wise max of two V4f32 vectors.
/// Uses @select to prevent LLVM from scalarizing on WASM targets.
/// `@max` uses IEEE 754-2019 semantics which LLVM lowers to scalar calls;
/// `@select` compiles to `f32x4.gt + v128.bitselect` (2 SIMD ops).
pub inline fn max_f32x4(a: V4f32, b: V4f32) V4f32 {
    return @select(f32, a > b, a, b);
}

/// Returns the element-wise min of two V4f32 vectors.
/// Uses @select to prevent LLVM from scalarizing on WASM targets.
/// `@min` uses IEEE 754-2019 semantics which LLVM lowers to scalar calls;
/// `@select` compiles to `f32x4.lt + v128.bitselect` (2 SIMD ops).
pub inline fn min_f32x4(a: V4f32, b: V4f32) V4f32 {
    return @select(f32, a < b, a, b);
}