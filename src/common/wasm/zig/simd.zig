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

// --- i8 extended multiply ---
// WASM SIMD has i16x8.extmul_low_i8x16_s / extmul_high_i8x16_s (base SIMD128):
// sign-extend half of two V16i8 inputs and multiply → V8i16 in one instruction.
// We express this as shuffle-to-extract-half → @intCast (sign-extend) → multiply,
// which LLVM pattern-matches to the native extmul instructions.

const lo_half = @Vector(8, i32){ 0, 1, 2, 3, 4, 5, 6, 7 };
const hi_half = @Vector(8, i32){ 8, 9, 10, 11, 12, 13, 14, 15 };

/// Extended multiply of the low 8 lanes of two V16i8 → V8i16.
/// Compiles to i16x8.extmul_low_i8x16_s (1 SIMD op).
pub inline fn extmul_low_i8x16_s(a: V16i8, b: V16i8) V8i16 {
    const a_lo: V8i16 = @intCast(@shuffle(i8, a, undefined, lo_half));
    const b_lo: V8i16 = @intCast(@shuffle(i8, b, undefined, lo_half));
    return a_lo *% b_lo;
}

/// Extended multiply of the high 8 lanes of two V16i8 → V8i16.
/// Compiles to i16x8.extmul_high_i8x16_s (1 SIMD op).
pub inline fn extmul_high_i8x16_s(a: V16i8, b: V16i8) V8i16 {
    const a_hi: V8i16 = @intCast(@shuffle(i8, a, undefined, hi_half));
    const b_hi: V8i16 = @intCast(@shuffle(i8, b, undefined, hi_half));
    return a_hi *% b_hi;
}

// --- i8 widen-multiply-narrow ---
// For element-wise i8 multiply-add where the result must stay in i8.
// NOTE: @intCast + @truncate causes LLVM to fully scalarize (extract_lane + i32.mul × 16).
// The @bitCast approach below stays in u8/i16 space and LLVM keeps it vectorized
// (2× i16x8.mul + 1× i8x16.shuffle to narrow).

/// Computes c +% (a *% b) element-wise for V16i8 using widened i16x8 multiplies.
/// WASM SIMD has i16x8.mul but no i8x16.mul.
/// Operates entirely in i16/u8 space via @bitCast to prevent LLVM from
/// recognizing and re-scalarizing the i8 multiply pattern.
pub inline fn muladd_i8x16(c_vec: V16i8, a_vec: V16i8, b_vec: V16i8) V16i8 {
    // Reinterpret as raw bytes to operate below Zig's type system
    const a_bytes: V16u8 = @bitCast(a_vec);
    const b_bytes: V16u8 = @bitCast(b_vec);
    const c_bytes: V16u8 = @bitCast(c_vec);

    // Widen to i16 by interleaving with zero bytes (zero-extend, not sign-extend —
    // wrapping mul produces the same low byte regardless of sign interpretation)
    const zero: V16u8 = @splat(0);

    // Low 8 elements: interleave bytes 0..7 with zeros → 8×u16
    const lo_even = @Vector(16, i32){ 0, -1, 1, -1, 2, -1, 3, -1, 4, -1, 5, -1, 6, -1, 7, -1 };
    const a_lo: V8i16 = @bitCast(@shuffle(u8, a_bytes, zero, lo_even));
    const b_lo: V8i16 = @bitCast(@shuffle(u8, b_bytes, zero, lo_even));
    const c_lo: V8i16 = @bitCast(@shuffle(u8, c_bytes, zero, lo_even));

    // High 8 elements: interleave bytes 8..15 with zeros → 8×u16
    const hi_even = @Vector(16, i32){ 8, -1, 9, -1, 10, -1, 11, -1, 12, -1, 13, -1, 14, -1, 15, -1 };
    const a_hi: V8i16 = @bitCast(@shuffle(u8, a_bytes, zero, hi_even));
    const b_hi: V8i16 = @bitCast(@shuffle(u8, b_bytes, zero, hi_even));
    const c_hi: V8i16 = @bitCast(@shuffle(u8, c_bytes, zero, hi_even));

    // Multiply-add in i16 (uses native i16x8.mul + i16x8.add)
    const r_lo = c_lo +% (a_lo *% b_lo);
    const r_hi = c_hi +% (a_hi *% b_hi);

    // Narrow back: pick low bytes from each i16 result (little-endian byte 0,2,4,...)
    const r_lo_bytes: V16u8 = @bitCast(r_lo);
    const r_hi_bytes: V16u8 = @bitCast(r_hi);
    const narrow = @Vector(16, i32){ 0, 2, 4, 6, 8, 10, 12, 14, -1, -3, -5, -7, -9, -11, -13, -15 };
    return @bitCast(@shuffle(u8, r_lo_bytes, r_hi_bytes, narrow));
}

/// Computes a *% b element-wise for V16i8 (no accumulator).
/// Uses the same @bitCast widen-multiply-narrow approach as muladd_i8x16,
/// but skips the c addition. Avoids the LLVM scalarization that occurs when
/// muladd_i8x16 is called with c=zero (LLVM constant-folds the zero add,
/// recognizes i8*i8, and scalarizes to 16× extract+mul+replace).
pub inline fn mul_i8x16(a_vec: V16i8, b_vec: V16i8) V16i8 {
    const a_bytes: V16u8 = @bitCast(a_vec);
    const b_bytes: V16u8 = @bitCast(b_vec);
    const zero: V16u8 = @splat(0);

    const lo_even = @Vector(16, i32){ 0, -1, 1, -1, 2, -1, 3, -1, 4, -1, 5, -1, 6, -1, 7, -1 };
    const a_lo: V8i16 = @bitCast(@shuffle(u8, a_bytes, zero, lo_even));
    const b_lo: V8i16 = @bitCast(@shuffle(u8, b_bytes, zero, lo_even));

    const hi_even = @Vector(16, i32){ 8, -1, 9, -1, 10, -1, 11, -1, 12, -1, 13, -1, 14, -1, 15, -1 };
    const a_hi: V8i16 = @bitCast(@shuffle(u8, a_bytes, zero, hi_even));
    const b_hi: V8i16 = @bitCast(@shuffle(u8, b_bytes, zero, hi_even));

    const r_lo = a_lo *% b_lo;
    const r_hi = a_hi *% b_hi;

    const r_lo_bytes: V16u8 = @bitCast(r_lo);
    const r_hi_bytes: V16u8 = @bitCast(r_hi);
    const narrow = @Vector(16, i32){ 0, 2, 4, 6, 8, 10, 12, 14, -1, -3, -5, -7, -9, -11, -13, -15 };
    return @bitCast(@shuffle(u8, r_lo_bytes, r_hi_bytes, narrow));
}

// --- i32x4.dot_i16x8_s ---
// Base SIMD128 instruction: pairwise multiply i16 pairs and sum into i32.
// out[i] = a[2i] * b[2i] + a[2i+1] * b[2i+1]
// Replaces i16x8.mul + pairwise i16x8.add with a single instruction.
// On wasm32: LLVM intrinsic → i32x4.dot_i16x8_s (1 op).
// On native (tests): generic fallback via widen + multiply + pairwise add.

const is_wasm32 = @import("builtin").cpu.arch == .wasm32;

const wasm_simd128 = if (is_wasm32) struct {
    extern fn @"llvm.wasm.dot"(V8i16, V8i16) V4i32;
} else struct {};

/// Pairwise dot product of two V8i16 → V4i32.
/// wasm32: i32x4.dot_i16x8_s (1 SIMD op). Native: widen + mul + pairwise add.
pub inline fn dot_i16x8_s(a: V8i16, b: V8i16) V4i32 {
    if (is_wasm32) return wasm_simd128.@"llvm.wasm.dot"(a, b);
    // Fallback for non-wasm targets (native tests)
    const a_i32: @Vector(8, i32) = @intCast(a);
    const b_i32: @Vector(8, i32) = @intCast(b);
    const products = a_i32 *% b_i32;
    const even = @shuffle(i32, products, undefined, @Vector(4, i32){ 0, 2, 4, 6 });
    const odd = @shuffle(i32, products, undefined, @Vector(4, i32){ 1, 3, 5, 7 });
    return even +% odd;
}

/// Widening dot product of two V16i8 → V4i32.
/// Sign-extends i8→i16 via extmul, then uses i32x4.dot_i16x8_s for pairwise accumulation.
/// Processes all 16 i8 lanes into 4 i32 partial sums (2 dot instructions).
pub inline fn dot_i8x16_to_i32x4(a: V16i8, b: V16i8) V4i32 {
    const a_lo: V8i16 = @intCast(@shuffle(i8, a, undefined, lo_half));
    const b_lo: V8i16 = @intCast(@shuffle(i8, b, undefined, lo_half));
    const a_hi: V8i16 = @intCast(@shuffle(i8, a, undefined, hi_half));
    const b_hi: V8i16 = @intCast(@shuffle(i8, b, undefined, hi_half));
    return dot_i16x8_s(a_lo, b_lo) +% dot_i16x8_s(a_hi, b_hi);
}

// --- Fused multiply-add ---
// Baseline (simd128 only): a * b + c = f64x2.mul + f64x2.add (2 SIMD ops, no regression).
// Relaxed (+relaxed_simd): LLVM intrinsic → f64x2.relaxed_madd (1 SIMD op, true FMA).
// NOTE: Zig's @mulAdd scalarizes on wasm32 (extracts lanes → scalar fma → reassemble).
// We must use the LLVM intrinsic directly for the relaxed path.

const has_relaxed_simd = blk: {
    const builtin = @import("builtin");
    break :blk builtin.cpu.arch == .wasm32 and
        @import("std").Target.wasm.featureSetHas(builtin.cpu.features, .relaxed_simd);
};

const relaxed_wasm = if (has_relaxed_simd) struct {
    extern fn @"llvm.wasm.relaxed.madd.v2f64"(V2f64, V2f64, V2f64) V2f64;
    extern fn @"llvm.wasm.relaxed.madd.v4f32"(V4f32, V4f32, V4f32) V4f32;
    extern fn @"llvm.wasm.relaxed.nmadd.v2f64"(V2f64, V2f64, V2f64) V2f64;
    extern fn @"llvm.wasm.relaxed.nmadd.v4f32"(V4f32, V4f32, V4f32) V4f32;
} else struct {};

/// Fused multiply-add for V2f64: a * b + c.
/// Relaxed: f64x2.relaxed_madd (1 op). Baseline: f64x2.mul + f64x2.add (2 ops).
pub inline fn mulAdd_f64x2(a: V2f64, b: V2f64, c: V2f64) V2f64 {
    if (has_relaxed_simd) return relaxed_wasm.@"llvm.wasm.relaxed.madd.v2f64"(a, b, c);
    return a * b + c;
}

/// Fused multiply-add for V4f32: a * b + c.
/// Relaxed: f32x4.relaxed_madd (1 op). Baseline: f32x4.mul + f32x4.add (2 ops).
pub inline fn mulAdd_f32x4(a: V4f32, b: V4f32, c: V4f32) V4f32 {
    if (has_relaxed_simd) return relaxed_wasm.@"llvm.wasm.relaxed.madd.v4f32"(a, b, c);
    return a * b + c;
}

/// Negated fused multiply-add for V2f64: -(a * b) + c = c - a * b.
/// Relaxed: f64x2.relaxed_nmadd (1 op). Baseline: f64x2.mul + f64x2.sub (2 ops).
pub inline fn nmulAdd_f64x2(a: V2f64, b: V2f64, c: V2f64) V2f64 {
    if (has_relaxed_simd) return relaxed_wasm.@"llvm.wasm.relaxed.nmadd.v2f64"(a, b, c);
    return c - a * b;
}

/// Negated fused multiply-add for V4f32: -(a * b) + c = c - a * b.
/// Relaxed: f32x4.relaxed_nmadd (1 op). Baseline: f32x4.mul + f32x4.sub (2 ops).
pub inline fn nmulAdd_f32x4(a: V4f32, b: V4f32, c: V4f32) V4f32 {
    if (has_relaxed_simd) return relaxed_wasm.@"llvm.wasm.relaxed.nmadd.v4f32"(a, b, c);
    return c - a * b;
}

// --- WASM SIMD min/max ---
// LLVM pattern-matches @select(f32, a < b, a, b) directly to f32x4.pmin (1 SIMD op).
// f32x4.pmin and f32x4.relaxed_min are both single instructions with identical
// throughput — the only difference is NaN handling. No relaxed variant needed.
// NOTE: Do NOT change to @min/@max — LLVM scalarizes those to scalar fmin/fmax calls.

/// Returns the element-wise max of two V2f64 vectors.
/// Compiles to f64x2.pmax (1 SIMD op).
pub inline fn max_f64x2(a: V2f64, b: V2f64) V2f64 {
    return @select(f64, a > b, a, b);
}

/// Returns the element-wise min of two V2f64 vectors.
/// Compiles to f64x2.pmin (1 SIMD op).
pub inline fn min_f64x2(a: V2f64, b: V2f64) V2f64 {
    return @select(f64, a < b, a, b);
}

/// Returns the element-wise max of two V4f32 vectors.
/// Compiles to f32x4.pmax (1 SIMD op).
pub inline fn max_f32x4(a: V4f32, b: V4f32) V4f32 {
    return @select(f32, a > b, a, b);
}

/// Returns the element-wise min of two V4f32 vectors.
/// Compiles to f32x4.pmin (1 SIMD op).
pub inline fn min_f32x4(a: V4f32, b: V4f32) V4f32 {
    return @select(f32, a < b, a, b);
}

// --- Integer SIMD min/max ---
// @select(T, a < b, a, b) pattern-matches to native WASM SIMD min/max instructions.
// i32x4.min_s, i32x4.max_s, i16x8.min_s, etc. — all base SIMD128.
// i64x2 has signed compare (i64x2.lt_s) but no unsigned compare.
// For u64: use sign-bit flip (XOR 0x8000000000000000) to convert to signed domain.
// NOTE: Do NOT change to @min/@max — LLVM scalarizes those.

pub inline fn max_i64x2(a: V2i64, b: V2i64) V2i64 {
    return @select(i64, a > b, a, b);
}
pub inline fn min_i64x2(a: V2i64, b: V2i64) V2i64 {
    return @select(i64, a < b, a, b);
}

const SIGN_FLIP_64: V2i64 = @splat(@bitCast(@as(u64, 0x8000000000000000)));

/// Element-wise max for V2u64 via sign-flip + signed compare.
/// WASM has i64x2.gt_s but no unsigned variant.
pub inline fn max_u64x2(a: V2u64, b: V2u64) V2u64 {
    const sa: V2i64 = @bitCast(a);
    const sb: V2i64 = @bitCast(b);
    const fa = sa +% SIGN_FLIP_64;
    const fb = sb +% SIGN_FLIP_64;
    return @select(u64, fa > fb, a, b);
}

/// Element-wise min for V2u64 via sign-flip + signed compare.
pub inline fn min_u64x2(a: V2u64, b: V2u64) V2u64 {
    const sa: V2i64 = @bitCast(a);
    const sb: V2i64 = @bitCast(b);
    const fa = sa +% SIGN_FLIP_64;
    const fb = sb +% SIGN_FLIP_64;
    return @select(u64, fa < fb, a, b);
}

pub inline fn max_i32x4(a: V4i32, b: V4i32) V4i32 {
    return @select(i32, a > b, a, b);
}
pub inline fn min_i32x4(a: V4i32, b: V4i32) V4i32 {
    return @select(i32, a < b, a, b);
}
pub inline fn max_u32x4(a: V4u32, b: V4u32) V4u32 {
    return @select(u32, a > b, a, b);
}
pub inline fn min_u32x4(a: V4u32, b: V4u32) V4u32 {
    return @select(u32, a < b, a, b);
}
pub inline fn max_i16x8(a: V8i16, b: V8i16) V8i16 {
    return @select(i16, a > b, a, b);
}
pub inline fn min_i16x8(a: V8i16, b: V8i16) V8i16 {
    return @select(i16, a < b, a, b);
}
pub inline fn max_u16x8(a: V8u16, b: V8u16) V8u16 {
    return @select(u16, a > b, a, b);
}
pub inline fn min_u16x8(a: V8u16, b: V8u16) V8u16 {
    return @select(u16, a < b, a, b);
}
pub inline fn max_i8x16(a: V16i8, b: V16i8) V16i8 {
    return @select(i8, a > b, a, b);
}
pub inline fn min_i8x16(a: V16i8, b: V16i8) V16i8 {
    return @select(i8, a < b, a, b);
}
pub inline fn max_u8x16(a: V16u8, b: V16u8) V16u8 {
    return @select(u8, a > b, a, b);
}
pub inline fn min_u8x16(a: V16u8, b: V16u8) V16u8 {
    return @select(u8, a < b, a, b);
}
