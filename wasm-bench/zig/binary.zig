// WASM binary elementwise kernels for f32/f64 with SIMD
//
// Uses native v128 widths: @Vector(2,f64) / @Vector(4,f32)
// Pointer-cast loads/stores to guarantee v128.load/v128.store opcodes.

const simd = @import("simd.zig");

// ─── Generic helpers ───────────────────────────────────────────────────────

fn binaryV2_f64(a: [*]const f64, b: [*]const f64, out: [*]f64, n: u32, comptime op: fn (simd.V2f64, simd.V2f64) simd.V2f64) void {
    const len = @as(usize, n);
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        simd.store2_f64(out, i, op(simd.load2_f64(a, i), simd.load2_f64(b, i)));
        simd.store2_f64(out, i + 2, op(simd.load2_f64(a, i + 2), simd.load2_f64(b, i + 2)));
    }
    while (i + 2 <= len) : (i += 2) {
        simd.store2_f64(out, i, op(simd.load2_f64(a, i), simd.load2_f64(b, i)));
    }
    while (i < len) : (i += 1) {
        const va: simd.V2f64 = .{ a[i], 0 };
        const vb: simd.V2f64 = .{ b[i], 0 };
        out[i] = op(va, vb)[0];
    }
}

fn binaryV4_f32(a: [*]const f32, b: [*]const f32, out: [*]f32, n: u32, comptime op: fn (simd.V4f32, simd.V4f32) simd.V4f32) void {
    const len = @as(usize, n);
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        simd.store4_f32(out, i, op(simd.load4_f32(a, i), simd.load4_f32(b, i)));
        simd.store4_f32(out, i + 4, op(simd.load4_f32(a, i + 4), simd.load4_f32(b, i + 4)));
    }
    while (i + 4 <= len) : (i += 4) {
        simd.store4_f32(out, i, op(simd.load4_f32(a, i), simd.load4_f32(b, i)));
    }
    while (i < len) : (i += 1) {
        const va: simd.V4f32 = .{ a[i], 0, 0, 0 };
        const vb: simd.V4f32 = .{ b[i], 0, 0, 0 };
        out[i] = op(va, vb)[0];
    }
}

// ─── Arithmetic op implementations ─────────────────────────────────────────

fn addOp_f64(a: simd.V2f64, b: simd.V2f64) simd.V2f64 {
    return a + b;
}
fn subOp_f64(a: simd.V2f64, b: simd.V2f64) simd.V2f64 {
    return a - b;
}
fn mulOp_f64(a: simd.V2f64, b: simd.V2f64) simd.V2f64 {
    return a * b;
}
fn divOp_f64(a: simd.V2f64, b: simd.V2f64) simd.V2f64 {
    return a / b;
}

fn addOp_f32(a: simd.V4f32, b: simd.V4f32) simd.V4f32 {
    return a + b;
}
fn subOp_f32(a: simd.V4f32, b: simd.V4f32) simd.V4f32 {
    return a - b;
}
fn mulOp_f32(a: simd.V4f32, b: simd.V4f32) simd.V4f32 {
    return a * b;
}
fn divOp_f32(a: simd.V4f32, b: simd.V4f32) simd.V4f32 {
    return a / b;
}

// ─── copysign: magnitude of a, sign of b ────────────────────────────────────

fn copysignOp_f64(a: simd.V2f64, b: simd.V2f64) simd.V2f64 {
    const abs_mask: simd.V2u64 = @splat(0x7FFFFFFFFFFFFFFF);
    const sign_mask: simd.V2u64 = @splat(0x8000000000000000);
    const a_bits: simd.V2u64 = @bitCast(a);
    const b_bits: simd.V2u64 = @bitCast(b);
    return @bitCast((a_bits & abs_mask) | (b_bits & sign_mask));
}
fn copysignOp_f32(a: simd.V4f32, b: simd.V4f32) simd.V4f32 {
    const abs_mask: simd.V4u32 = @splat(0x7FFFFFFF);
    const sign_mask: simd.V4u32 = @splat(0x80000000);
    const a_bits: simd.V4u32 = @bitCast(a);
    const b_bits: simd.V4u32 = @bitCast(b);
    return @bitCast((a_bits & abs_mask) | (b_bits & sign_mask));
}

// ─── power: a^b = exp(b * log(a)), requires positive a ──────────────────────

fn powerOp_f64(a: simd.V2f64, b: simd.V2f64) simd.V2f64 {
    return @exp(b * @log(a));
}
fn powerOp_f32(a: simd.V4f32, b: simd.V4f32) simd.V4f32 {
    return @exp(b * @log(a));
}

// ─── maximum / minimum: propagates NaN (matches NumPy) ──────────────────────

// maximum/minimum propagate NaN: if either input is NaN, result is NaN.
// Use simd.max/min (which use @select to avoid scalarization) + NaN propagation.
fn maximumOp_f64(a: simd.V2f64, b: simd.V2f64) simd.V2f64 {
    const any_nan = (a != a) | (b != b);
    const nan_vec: simd.V2f64 = @splat(@as(f64, @bitCast(@as(u64, 0x7FF8000000000000))));
    return @select(f64, any_nan, nan_vec, simd.max_f64x2(a, b));
}
fn minimumOp_f64(a: simd.V2f64, b: simd.V2f64) simd.V2f64 {
    const any_nan = (a != a) | (b != b);
    const nan_vec: simd.V2f64 = @splat(@as(f64, @bitCast(@as(u64, 0x7FF8000000000000))));
    return @select(f64, any_nan, nan_vec, simd.min_f64x2(a, b));
}
fn maximumOp_f32(a: simd.V4f32, b: simd.V4f32) simd.V4f32 {
    const any_nan = (a != a) | (b != b);
    const nan_vec: simd.V4f32 = @splat(@as(f32, @bitCast(@as(u32, 0x7FC00000))));
    return @select(f32, any_nan, nan_vec, simd.max_f32x4(a, b));
}
fn minimumOp_f32(a: simd.V4f32, b: simd.V4f32) simd.V4f32 {
    const any_nan = (a != a) | (b != b);
    const nan_vec: simd.V4f32 = @splat(@as(f32, @bitCast(@as(u32, 0x7FC00000))));
    return @select(f32, any_nan, nan_vec, simd.min_f32x4(a, b));
}

// ─── fmax / fmin: ignores NaN (returns the non-NaN value) ───────────────────

fn fmaxOp_f64(a: simd.V2f64, b: simd.V2f64) simd.V2f64 {
    const a_nan = a != a;
    const b_nan = b != b;
    return @select(f64, a_nan, b, @select(f64, b_nan, a, simd.max_f64x2(a, b)));
}
fn fminOp_f64(a: simd.V2f64, b: simd.V2f64) simd.V2f64 {
    const a_nan = a != a;
    const b_nan = b != b;
    return @select(f64, a_nan, b, @select(f64, b_nan, a, simd.min_f64x2(a, b)));
}
fn fmaxOp_f32(a: simd.V4f32, b: simd.V4f32) simd.V4f32 {
    const a_nan = a != a;
    const b_nan = b != b;
    return @select(f32, a_nan, b, @select(f32, b_nan, a, simd.max_f32x4(a, b)));
}
fn fminOp_f32(a: simd.V4f32, b: simd.V4f32) simd.V4f32 {
    const a_nan = a != a;
    const b_nan = b != b;
    return @select(f32, a_nan, b, @select(f32, b_nan, a, simd.min_f32x4(a, b)));
}

// ─── logaddexp: log(exp(a) + exp(b)) ────────────────────────────────────────

fn logaddexpOp_f64(a: simd.V2f64, b: simd.V2f64) simd.V2f64 {
    return @log(@exp(a) + @exp(b));
}
fn logaddexpOp_f32(a: simd.V4f32, b: simd.V4f32) simd.V4f32 {
    return @log(@exp(a) + @exp(b));
}

// Scalar logaddexp for better WASM performance (builtins scalarize poorly)
fn logaddexpScalar_f64(av: f64, bv: f64) f64 {
    return @log(@exp(av) + @exp(bv));
}
fn logaddexpScalar_f32(av: f32, bv: f32) f32 {
    return @log(@exp(av) + @exp(bv));
}

// ─── logical_and: (a != 0 && b != 0) ? 1.0 : 0.0 ──────────────────────────

fn logicalAndOp_f64(a: simd.V2f64, b: simd.V2f64) simd.V2f64 {
    const zero: simd.V2f64 = @splat(0.0);
    const one: simd.V2f64 = @splat(1.0);
    return @select(f64, a != zero, @select(f64, b != zero, one, zero), zero);
}
fn logicalAndOp_f32(a: simd.V4f32, b: simd.V4f32) simd.V4f32 {
    const zero: simd.V4f32 = @splat(0.0);
    const one: simd.V4f32 = @splat(1.0);
    return @select(f32, a != zero, @select(f32, b != zero, one, zero), zero);
}

// ─── logical_xor: (a != 0) != (b != 0) ? 1.0 : 0.0 ────────────────────────

fn logicalXorOp_f64(a: simd.V2f64, b: simd.V2f64) simd.V2f64 {
    const zero: simd.V2f64 = @splat(0.0);
    const one: simd.V2f64 = @splat(1.0);
    // XOR the raw comparison masks directly (avoids intermediate float conversion)
    const a_nz = a != zero;
    const b_nz = b != zero;
    return @select(f64, a_nz != b_nz, one, zero);
}
fn logicalXorOp_f32(a: simd.V4f32, b: simd.V4f32) simd.V4f32 {
    const zero: simd.V4f32 = @splat(0.0);
    const one: simd.V4f32 = @splat(1.0);
    const a_nz = a != zero;
    const b_nz = b != zero;
    return @select(f32, a_nz != b_nz, one, zero);
}

// ─── mod (floored remainder): a - floor(a/b) * b ────────────────────────

fn modOp_f64(a: simd.V2f64, b: simd.V2f64) simd.V2f64 {
    return a - @floor(a / b) * b;
}
fn modOp_f32(a: simd.V4f32, b: simd.V4f32) simd.V4f32 {
    return a - @floor(a / b) * b;
}

// ─── floor_divide: floor(a / b) ─────────────────────────────────────────

fn floorDivOp_f64(a: simd.V2f64, b: simd.V2f64) simd.V2f64 {
    return @floor(a / b);
}
fn floorDivOp_f32(a: simd.V4f32, b: simd.V4f32) simd.V4f32 {
    return @floor(a / b);
}

// ─── hypot: sqrt(a² + b²) ──────────────────────────────────────────────

fn hypotOp_f64(a: simd.V2f64, b: simd.V2f64) simd.V2f64 {
    return @sqrt(a * a + b * b);
}
fn hypotOp_f32(a: simd.V4f32, b: simd.V4f32) simd.V4f32 {
    return @sqrt(a * a + b * b);
}

// ═══════════════════════════════════════════════════════════════════════════
// EXPORTS
// ═══════════════════════════════════════════════════════════════════════════

// ─── f64 arithmetic ──────
export fn add_f64(a: [*]const f64, b: [*]const f64, o: [*]f64, n: u32) void {
    binaryV2_f64(a, b, o, n, addOp_f64);
}
export fn sub_f64(a: [*]const f64, b: [*]const f64, o: [*]f64, n: u32) void {
    binaryV2_f64(a, b, o, n, subOp_f64);
}
export fn mul_f64(a: [*]const f64, b: [*]const f64, o: [*]f64, n: u32) void {
    binaryV2_f64(a, b, o, n, mulOp_f64);
}
export fn div_f64(a: [*]const f64, b: [*]const f64, o: [*]f64, n: u32) void {
    binaryV2_f64(a, b, o, n, divOp_f64);
}

// ─── f32 arithmetic ──────
export fn add_f32(a: [*]const f32, b: [*]const f32, o: [*]f32, n: u32) void {
    binaryV4_f32(a, b, o, n, addOp_f32);
}
export fn sub_f32(a: [*]const f32, b: [*]const f32, o: [*]f32, n: u32) void {
    binaryV4_f32(a, b, o, n, subOp_f32);
}
export fn mul_f32(a: [*]const f32, b: [*]const f32, o: [*]f32, n: u32) void {
    binaryV4_f32(a, b, o, n, mulOp_f32);
}
export fn div_f32(a: [*]const f32, b: [*]const f32, o: [*]f32, n: u32) void {
    binaryV4_f32(a, b, o, n, divOp_f32);
}

// ─── f64 new ops ─────────
export fn copysign_f64(a: [*]const f64, b: [*]const f64, o: [*]f64, n: u32) void {
    binaryV2_f64(a, b, o, n, copysignOp_f64);
}
export fn power_f64(a: [*]const f64, b: [*]const f64, o: [*]f64, n: u32) void {
    binaryV2_f64(a, b, o, n, powerOp_f64);
}
export fn maximum_f64(a: [*]const f64, b: [*]const f64, o: [*]f64, n: u32) void {
    binaryV2_f64(a, b, o, n, maximumOp_f64);
}
export fn minimum_f64(a: [*]const f64, b: [*]const f64, o: [*]f64, n: u32) void {
    binaryV2_f64(a, b, o, n, minimumOp_f64);
}
export fn fmax_f64(a: [*]const f64, b: [*]const f64, o: [*]f64, n: u32) void {
    binaryV2_f64(a, b, o, n, fmaxOp_f64);
}
export fn fmin_f64(a: [*]const f64, b: [*]const f64, o: [*]f64, n: u32) void {
    binaryV2_f64(a, b, o, n, fminOp_f64);
}
export fn logaddexp_f64(a: [*]const f64, b: [*]const f64, o: [*]f64, n: u32) void {
    const len = @as(usize, n);
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        o[i] = logaddexpScalar_f64(a[i], b[i]);
        o[i + 1] = logaddexpScalar_f64(a[i + 1], b[i + 1]);
        o[i + 2] = logaddexpScalar_f64(a[i + 2], b[i + 2]);
        o[i + 3] = logaddexpScalar_f64(a[i + 3], b[i + 3]);
    }
    while (i < len) : (i += 1) {
        o[i] = logaddexpScalar_f64(a[i], b[i]);
    }
}
export fn logical_and_f64(a: [*]const f64, b: [*]const f64, o: [*]f64, n: u32) void {
    binaryV2_f64(a, b, o, n, logicalAndOp_f64);
}
export fn logical_xor_f64(a: [*]const f64, b: [*]const f64, o: [*]f64, n: u32) void {
    binaryV2_f64(a, b, o, n, logicalXorOp_f64);
}

// ─── f32 new ops ─────────
export fn copysign_f32(a: [*]const f32, b: [*]const f32, o: [*]f32, n: u32) void {
    binaryV4_f32(a, b, o, n, copysignOp_f32);
}
export fn power_f32(a: [*]const f32, b: [*]const f32, o: [*]f32, n: u32) void {
    binaryV4_f32(a, b, o, n, powerOp_f32);
}
export fn maximum_f32(a: [*]const f32, b: [*]const f32, o: [*]f32, n: u32) void {
    binaryV4_f32(a, b, o, n, maximumOp_f32);
}
export fn minimum_f32(a: [*]const f32, b: [*]const f32, o: [*]f32, n: u32) void {
    binaryV4_f32(a, b, o, n, minimumOp_f32);
}
export fn fmax_f32(a: [*]const f32, b: [*]const f32, o: [*]f32, n: u32) void {
    binaryV4_f32(a, b, o, n, fmaxOp_f32);
}
export fn fmin_f32(a: [*]const f32, b: [*]const f32, o: [*]f32, n: u32) void {
    binaryV4_f32(a, b, o, n, fminOp_f32);
}
export fn logaddexp_f32(a: [*]const f32, b: [*]const f32, o: [*]f32, n: u32) void {
    const len = @as(usize, n);
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        o[i] = logaddexpScalar_f32(a[i], b[i]);
        o[i + 1] = logaddexpScalar_f32(a[i + 1], b[i + 1]);
        o[i + 2] = logaddexpScalar_f32(a[i + 2], b[i + 2]);
        o[i + 3] = logaddexpScalar_f32(a[i + 3], b[i + 3]);
    }
    while (i < len) : (i += 1) {
        o[i] = logaddexpScalar_f32(a[i], b[i]);
    }
}
export fn logical_and_f32(a: [*]const f32, b: [*]const f32, o: [*]f32, n: u32) void {
    binaryV4_f32(a, b, o, n, logicalAndOp_f32);
}
export fn logical_xor_f32(a: [*]const f32, b: [*]const f32, o: [*]f32, n: u32) void {
    binaryV4_f32(a, b, o, n, logicalXorOp_f32);
}

// ─── f64 new batch 2 ────
export fn mod_f64(a: [*]const f64, b: [*]const f64, o: [*]f64, n: u32) void {
    binaryV2_f64(a, b, o, n, modOp_f64);
}
export fn floor_divide_f64(a: [*]const f64, b: [*]const f64, o: [*]f64, n: u32) void {
    binaryV2_f64(a, b, o, n, floorDivOp_f64);
}
export fn hypot_f64(a: [*]const f64, b: [*]const f64, o: [*]f64, n: u32) void {
    binaryV2_f64(a, b, o, n, hypotOp_f64);
}

// ─── f32 new batch 2 ────
export fn mod_f32(a: [*]const f32, b: [*]const f32, o: [*]f32, n: u32) void {
    binaryV4_f32(a, b, o, n, modOp_f32);
}
export fn floor_divide_f32(a: [*]const f32, b: [*]const f32, o: [*]f32, n: u32) void {
    binaryV4_f32(a, b, o, n, floorDivOp_f32);
}
export fn hypot_f32(a: [*]const f32, b: [*]const f32, o: [*]f32, n: u32) void {
    binaryV4_f32(a, b, o, n, hypotOp_f32);
}

// ═══════════════════════════════════════════════════════════════════════════
// INTEGER TYPES (i32, i16, i8) — wrapping arithmetic, SIMD
// ═══════════════════════════════════════════════════════════════════════════

// ─── Generic integer binary loop drivers ─────────────────────────────────

fn binaryV4_i32(a: [*]const i32, b: [*]const i32, out: [*]i32, n: u32, comptime op: fn (simd.V4i32, simd.V4i32) simd.V4i32) void {
    const len = @as(usize, n);
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        simd.store4_i32(out, i, op(simd.load4_i32(a, i), simd.load4_i32(b, i)));
        simd.store4_i32(out, i + 4, op(simd.load4_i32(a, i + 4), simd.load4_i32(b, i + 4)));
    }
    while (i + 4 <= len) : (i += 4) {
        simd.store4_i32(out, i, op(simd.load4_i32(a, i), simd.load4_i32(b, i)));
    }
    while (i < len) : (i += 1) {
        const va: simd.V4i32 = .{ a[i], 0, 0, 0 };
        const vb: simd.V4i32 = .{ b[i], 0, 0, 0 };
        out[i] = op(va, vb)[0];
    }
}

fn binaryV8_i16(a: [*]const i16, b: [*]const i16, out: [*]i16, n: u32, comptime op: fn (simd.V8i16, simd.V8i16) simd.V8i16) void {
    const len = @as(usize, n);
    var i: usize = 0;
    while (i + 16 <= len) : (i += 16) {
        simd.store8_i16(out, i, op(simd.load8_i16(a, i), simd.load8_i16(b, i)));
        simd.store8_i16(out, i + 8, op(simd.load8_i16(a, i + 8), simd.load8_i16(b, i + 8)));
    }
    while (i + 8 <= len) : (i += 8) {
        simd.store8_i16(out, i, op(simd.load8_i16(a, i), simd.load8_i16(b, i)));
    }
    while (i < len) : (i += 1) {
        const va: simd.V8i16 = .{ a[i], 0, 0, 0, 0, 0, 0, 0 };
        const vb: simd.V8i16 = .{ b[i], 0, 0, 0, 0, 0, 0, 0 };
        out[i] = op(va, vb)[0];
    }
}

fn binaryV16_i8(a: [*]const i8, b: [*]const i8, out: [*]i8, n: u32, comptime op: fn (simd.V16i8, simd.V16i8) simd.V16i8) void {
    const len = @as(usize, n);
    var i: usize = 0;
    while (i + 32 <= len) : (i += 32) {
        simd.store16_i8(out, i, op(simd.load16_i8(a, i), simd.load16_i8(b, i)));
        simd.store16_i8(out, i + 16, op(simd.load16_i8(a, i + 16), simd.load16_i8(b, i + 16)));
    }
    while (i + 16 <= len) : (i += 16) {
        simd.store16_i8(out, i, op(simd.load16_i8(a, i), simd.load16_i8(b, i)));
    }
    while (i < len) : (i += 1) {
        const va: simd.V16i8 = .{ a[i], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        const vb: simd.V16i8 = .{ b[i], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        out[i] = op(va, vb)[0];
    }
}

// ─── Integer op implementations (wrapping) ───────────────────────────────

fn addOp_i32(a: simd.V4i32, b: simd.V4i32) simd.V4i32 {
    return a +% b;
}
fn subOp_i32(a: simd.V4i32, b: simd.V4i32) simd.V4i32 {
    return a -% b;
}
fn mulOp_i32(a: simd.V4i32, b: simd.V4i32) simd.V4i32 {
    return a *% b;
}
fn maxOp_i32(a: simd.V4i32, b: simd.V4i32) simd.V4i32 {
    return @max(a, b);
}
fn minOp_i32(a: simd.V4i32, b: simd.V4i32) simd.V4i32 {
    return @min(a, b);
}

fn addOp_i16(a: simd.V8i16, b: simd.V8i16) simd.V8i16 {
    return a +% b;
}
fn subOp_i16(a: simd.V8i16, b: simd.V8i16) simd.V8i16 {
    return a -% b;
}
fn mulOp_i16(a: simd.V8i16, b: simd.V8i16) simd.V8i16 {
    return a *% b;
}
fn maxOp_i16(a: simd.V8i16, b: simd.V8i16) simd.V8i16 {
    return @max(a, b);
}
fn minOp_i16(a: simd.V8i16, b: simd.V8i16) simd.V8i16 {
    return @min(a, b);
}

fn addOp_i8(a: simd.V16i8, b: simd.V16i8) simd.V16i8 {
    return a +% b;
}
fn subOp_i8(a: simd.V16i8, b: simd.V16i8) simd.V16i8 {
    return a -% b;
}
// i8 mul: no WASM SIMD i8x16_mul, but Zig wrapping mul on vectors should emit scalar or widened code
fn mulOp_i8(a: simd.V16i8, b: simd.V16i8) simd.V16i8 {
    return a *% b;
}
fn maxOp_i8(a: simd.V16i8, b: simd.V16i8) simd.V16i8 {
    return @max(a, b);
}
fn minOp_i8(a: simd.V16i8, b: simd.V16i8) simd.V16i8 {
    return @min(a, b);
}

// ─── i32 exports ─────────
export fn add_i32(a: [*]const i32, b: [*]const i32, o: [*]i32, n: u32) void {
    binaryV4_i32(a, b, o, n, addOp_i32);
}
export fn sub_i32(a: [*]const i32, b: [*]const i32, o: [*]i32, n: u32) void {
    binaryV4_i32(a, b, o, n, subOp_i32);
}
export fn mul_i32(a: [*]const i32, b: [*]const i32, o: [*]i32, n: u32) void {
    binaryV4_i32(a, b, o, n, mulOp_i32);
}
export fn maximum_i32(a: [*]const i32, b: [*]const i32, o: [*]i32, n: u32) void {
    binaryV4_i32(a, b, o, n, maxOp_i32);
}
export fn minimum_i32(a: [*]const i32, b: [*]const i32, o: [*]i32, n: u32) void {
    binaryV4_i32(a, b, o, n, minOp_i32);
}

// ─── i16 exports ─────────
export fn add_i16(a: [*]const i16, b: [*]const i16, o: [*]i16, n: u32) void {
    binaryV8_i16(a, b, o, n, addOp_i16);
}
export fn sub_i16(a: [*]const i16, b: [*]const i16, o: [*]i16, n: u32) void {
    binaryV8_i16(a, b, o, n, subOp_i16);
}
export fn mul_i16(a: [*]const i16, b: [*]const i16, o: [*]i16, n: u32) void {
    binaryV8_i16(a, b, o, n, mulOp_i16);
}
export fn maximum_i16(a: [*]const i16, b: [*]const i16, o: [*]i16, n: u32) void {
    binaryV8_i16(a, b, o, n, maxOp_i16);
}
export fn minimum_i16(a: [*]const i16, b: [*]const i16, o: [*]i16, n: u32) void {
    binaryV8_i16(a, b, o, n, minOp_i16);
}

// ─── i8 exports ──────────
export fn add_i8(a: [*]const i8, b: [*]const i8, o: [*]i8, n: u32) void {
    binaryV16_i8(a, b, o, n, addOp_i8);
}
export fn sub_i8(a: [*]const i8, b: [*]const i8, o: [*]i8, n: u32) void {
    binaryV16_i8(a, b, o, n, subOp_i8);
}
export fn mul_i8(a: [*]const i8, b: [*]const i8, o: [*]i8, n: u32) void {
    // Scalar fallback — no WASM SIMD i8x16_mul
    const len = @as(usize, n);
    var i: usize = 0;
    while (i < len) : (i += 1) {
        o[i] = a[i] *% b[i];
    }
}
export fn maximum_i8(a: [*]const i8, b: [*]const i8, o: [*]i8, n: u32) void {
    binaryV16_i8(a, b, o, n, maxOp_i8);
}
export fn minimum_i8(a: [*]const i8, b: [*]const i8, o: [*]i8, n: u32) void {
    binaryV16_i8(a, b, o, n, minOp_i8);
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPLEX TYPES (c128 = interleaved f64, c64 = interleaved f32)
// ═══════════════════════════════════════════════════════════════════════════

// ─── add_c128: component-wise f64 add on 2N elements ─────────────────────
export fn add_c128(a: [*]const f64, b: [*]const f64, o: [*]f64, n: u32) void {
    // n = number of complex elements, data has 2*n f64s
    binaryV2_f64(a, b, o, n * 2, addOp_f64);
}

// ─── add_c64: component-wise f32 add on 2N elements ──────────────────────
export fn add_c64(a: [*]const f32, b: [*]const f32, o: [*]f32, n: u32) void {
    binaryV4_f32(a, b, o, n * 2, addOp_f32);
}

// ─── mul_c128: (ac-bd, ad+bc) per complex pair — scalar ──────────────────
export fn mul_c128(a: [*]const f64, b: [*]const f64, o: [*]f64, n: u32) void {
    const len = @as(usize, n);
    var i: usize = 0;
    while (i < len) : (i += 1) {
        const ar = a[2 * i];
        const ai = a[2 * i + 1];
        const br = b[2 * i];
        const bi = b[2 * i + 1];
        o[2 * i] = ar * br - ai * bi;
        o[2 * i + 1] = ar * bi + ai * br;
    }
}

// ─── mul_c64: complex mul with SIMD shuffle for 2 complex per v128 ───────
export fn mul_c64(a: [*]const f32, b: [*]const f32, o: [*]f32, n: u32) void {
    const len = @as(usize, n);
    var i: usize = 0;
    // Process 2 complex numbers at a time using V4f32
    while (i + 2 <= len) : (i += 2) {
        const idx = i * 2;
        const av = simd.load4_f32(a, idx); // [ar0, ai0, ar1, ai1]
        const bv = simd.load4_f32(b, idx); // [br0, bi0, br1, bi1]
        // Shuffle to get [ai0, ar0, ai1, ar1]
        const a_swap: simd.V4f32 = .{ av[1], av[0], av[3], av[2] };
        // [br0, br0, br1, br1]
        const b_re: simd.V4f32 = .{ bv[0], bv[0], bv[2], bv[2] };
        // [bi0, bi0, bi1, bi1]
        const b_im: simd.V4f32 = .{ bv[1], bv[1], bv[3], bv[3] };
        // av * b_re = [ar*br, ai*br, ar*br, ai*br]
        // a_swap * b_im = [ai*bi, ar*bi, ai*bi, ar*bi]
        const t1 = av * b_re;
        const t2 = a_swap * b_im;
        // result = [ar*br - ai*bi, ai*br + ar*bi, ...]
        const sign: simd.V4f32 = .{ -1.0, 1.0, -1.0, 1.0 };
        simd.store4_f32(o, idx, t1 + t2 * sign);
    }
    // Scalar remainder
    while (i < len) : (i += 1) {
        const ar = a[2 * i];
        const ai = a[2 * i + 1];
        const br = b[2 * i];
        const bi = b[2 * i + 1];
        o[2 * i] = ar * br - ai * bi;
        o[2 * i + 1] = ar * bi + ai * br;
    }
}
