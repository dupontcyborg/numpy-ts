// WASM binary elementwise kernels for f32/f64 with SIMD
//
// Uses native v128 widths: @Vector(2,f64) / @Vector(4,f32)
// Pointer-cast loads/stores to guarantee v128.load/v128.store opcodes.

const V2f64 = @Vector(2, f64);
const V4f32 = @Vector(4, f32);
const V2u64 = @Vector(2, u64);
const V4u32 = @Vector(4, u32);

inline fn load2_f64(ptr: [*]const f64, i: usize) V2f64 {
    return @as(*align(1) const V2f64, @ptrCast(ptr + i)).*;
}
inline fn store2_f64(ptr: [*]f64, i: usize, v: V2f64) void {
    @as(*align(1) V2f64, @ptrCast(ptr + i)).* = v;
}
inline fn load4_f32(ptr: [*]const f32, i: usize) V4f32 {
    return @as(*align(1) const V4f32, @ptrCast(ptr + i)).*;
}
inline fn store4_f32(ptr: [*]f32, i: usize, v: V4f32) void {
    @as(*align(1) V4f32, @ptrCast(ptr + i)).* = v;
}

// ─── Generic helpers ───────────────────────────────────────────────────────

fn binaryV2_f64(a: [*]const f64, b: [*]const f64, out: [*]f64, n: u32, comptime op: fn (V2f64, V2f64) V2f64) void {
    const len = @as(usize, n);
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        store2_f64(out, i, op(load2_f64(a, i), load2_f64(b, i)));
        store2_f64(out, i + 2, op(load2_f64(a, i + 2), load2_f64(b, i + 2)));
    }
    while (i + 2 <= len) : (i += 2) {
        store2_f64(out, i, op(load2_f64(a, i), load2_f64(b, i)));
    }
    while (i < len) : (i += 1) {
        const va: V2f64 = .{ a[i], 0 };
        const vb: V2f64 = .{ b[i], 0 };
        out[i] = op(va, vb)[0];
    }
}

fn binaryV4_f32(a: [*]const f32, b: [*]const f32, out: [*]f32, n: u32, comptime op: fn (V4f32, V4f32) V4f32) void {
    const len = @as(usize, n);
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        store4_f32(out, i, op(load4_f32(a, i), load4_f32(b, i)));
        store4_f32(out, i + 4, op(load4_f32(a, i + 4), load4_f32(b, i + 4)));
    }
    while (i + 4 <= len) : (i += 4) {
        store4_f32(out, i, op(load4_f32(a, i), load4_f32(b, i)));
    }
    while (i < len) : (i += 1) {
        const va: V4f32 = .{ a[i], 0, 0, 0 };
        const vb: V4f32 = .{ b[i], 0, 0, 0 };
        out[i] = op(va, vb)[0];
    }
}

// ─── Arithmetic op implementations ─────────────────────────────────────────

fn addOp_f64(a: V2f64, b: V2f64) V2f64 { return a + b; }
fn subOp_f64(a: V2f64, b: V2f64) V2f64 { return a - b; }
fn mulOp_f64(a: V2f64, b: V2f64) V2f64 { return a * b; }
fn divOp_f64(a: V2f64, b: V2f64) V2f64 { return a / b; }

fn addOp_f32(a: V4f32, b: V4f32) V4f32 { return a + b; }
fn subOp_f32(a: V4f32, b: V4f32) V4f32 { return a - b; }
fn mulOp_f32(a: V4f32, b: V4f32) V4f32 { return a * b; }
fn divOp_f32(a: V4f32, b: V4f32) V4f32 { return a / b; }

// ─── copysign: magnitude of a, sign of b ────────────────────────────────────

fn copysignOp_f64(a: V2f64, b: V2f64) V2f64 {
    const abs_mask: V2u64 = @splat(0x7FFFFFFFFFFFFFFF);
    const sign_mask: V2u64 = @splat(0x8000000000000000);
    const a_bits: V2u64 = @bitCast(a);
    const b_bits: V2u64 = @bitCast(b);
    return @bitCast((a_bits & abs_mask) | (b_bits & sign_mask));
}
fn copysignOp_f32(a: V4f32, b: V4f32) V4f32 {
    const abs_mask: V4u32 = @splat(0x7FFFFFFF);
    const sign_mask: V4u32 = @splat(0x80000000);
    const a_bits: V4u32 = @bitCast(a);
    const b_bits: V4u32 = @bitCast(b);
    return @bitCast((a_bits & abs_mask) | (b_bits & sign_mask));
}

// ─── power: a^b = exp(b * log(a)), requires positive a ──────────────────────

fn powerOp_f64(a: V2f64, b: V2f64) V2f64 { return @exp(b * @log(a)); }
fn powerOp_f32(a: V4f32, b: V4f32) V4f32 { return @exp(b * @log(a)); }

// ─── maximum / minimum: propagates NaN (matches NumPy) ──────────────────────

fn maximumOp_f64(a: V2f64, b: V2f64) V2f64 { return @max(a, b); }
fn minimumOp_f64(a: V2f64, b: V2f64) V2f64 { return @min(a, b); }
fn maximumOp_f32(a: V4f32, b: V4f32) V4f32 { return @max(a, b); }
fn minimumOp_f32(a: V4f32, b: V4f32) V4f32 { return @min(a, b); }

// ─── fmax / fmin: ignores NaN (returns the non-NaN value) ───────────────────

fn fmaxOp_f64(a: V2f64, b: V2f64) V2f64 {
    const a_nan = a != a;
    const b_nan = b != b;
    return @select(f64, a_nan, b, @select(f64, b_nan, a, @max(a, b)));
}
fn fminOp_f64(a: V2f64, b: V2f64) V2f64 {
    const a_nan = a != a;
    const b_nan = b != b;
    return @select(f64, a_nan, b, @select(f64, b_nan, a, @min(a, b)));
}
fn fmaxOp_f32(a: V4f32, b: V4f32) V4f32 {
    const a_nan = a != a;
    const b_nan = b != b;
    return @select(f32, a_nan, b, @select(f32, b_nan, a, @max(a, b)));
}
fn fminOp_f32(a: V4f32, b: V4f32) V4f32 {
    const a_nan = a != a;
    const b_nan = b != b;
    return @select(f32, a_nan, b, @select(f32, b_nan, a, @min(a, b)));
}

// ─── logaddexp: log(exp(a) + exp(b)) ────────────────────────────────────────

fn logaddexpOp_f64(a: V2f64, b: V2f64) V2f64 { return @log(@exp(a) + @exp(b)); }
fn logaddexpOp_f32(a: V4f32, b: V4f32) V4f32 { return @log(@exp(a) + @exp(b)); }

// ─── logical_and: (a != 0 && b != 0) ? 1.0 : 0.0 ──────────────────────────

fn logicalAndOp_f64(a: V2f64, b: V2f64) V2f64 {
    const zero: V2f64 = @splat(0.0);
    const one: V2f64 = @splat(1.0);
    return @select(f64, a != zero, @select(f64, b != zero, one, zero), zero);
}
fn logicalAndOp_f32(a: V4f32, b: V4f32) V4f32 {
    const zero: V4f32 = @splat(0.0);
    const one: V4f32 = @splat(1.0);
    return @select(f32, a != zero, @select(f32, b != zero, one, zero), zero);
}

// ─── logical_xor: (a != 0) != (b != 0) ? 1.0 : 0.0 ────────────────────────

fn logicalXorOp_f64(a: V2f64, b: V2f64) V2f64 {
    const zero: V2f64 = @splat(0.0);
    const one: V2f64 = @splat(1.0);
    const a_bool = @select(f64, a != zero, one, zero);
    const b_bool = @select(f64, b != zero, one, zero);
    return @select(f64, a_bool != b_bool, one, zero);
}
fn logicalXorOp_f32(a: V4f32, b: V4f32) V4f32 {
    const zero: V4f32 = @splat(0.0);
    const one: V4f32 = @splat(1.0);
    const a_bool = @select(f32, a != zero, one, zero);
    const b_bool = @select(f32, b != zero, one, zero);
    return @select(f32, a_bool != b_bool, one, zero);
}

// ═══════════════════════════════════════════════════════════════════════════
// EXPORTS
// ═══════════════════════════════════════════════════════════════════════════

// ─── f64 arithmetic ──────
export fn add_f64(a: [*]const f64, b: [*]const f64, o: [*]f64, n: u32) void { binaryV2_f64(a, b, o, n, addOp_f64); }
export fn sub_f64(a: [*]const f64, b: [*]const f64, o: [*]f64, n: u32) void { binaryV2_f64(a, b, o, n, subOp_f64); }
export fn mul_f64(a: [*]const f64, b: [*]const f64, o: [*]f64, n: u32) void { binaryV2_f64(a, b, o, n, mulOp_f64); }
export fn div_f64(a: [*]const f64, b: [*]const f64, o: [*]f64, n: u32) void { binaryV2_f64(a, b, o, n, divOp_f64); }

// ─── f32 arithmetic ──────
export fn add_f32(a: [*]const f32, b: [*]const f32, o: [*]f32, n: u32) void { binaryV4_f32(a, b, o, n, addOp_f32); }
export fn sub_f32(a: [*]const f32, b: [*]const f32, o: [*]f32, n: u32) void { binaryV4_f32(a, b, o, n, subOp_f32); }
export fn mul_f32(a: [*]const f32, b: [*]const f32, o: [*]f32, n: u32) void { binaryV4_f32(a, b, o, n, mulOp_f32); }
export fn div_f32(a: [*]const f32, b: [*]const f32, o: [*]f32, n: u32) void { binaryV4_f32(a, b, o, n, divOp_f32); }

// ─── f64 new ops ─────────
export fn copysign_f64(a: [*]const f64, b: [*]const f64, o: [*]f64, n: u32) void { binaryV2_f64(a, b, o, n, copysignOp_f64); }
export fn power_f64(a: [*]const f64, b: [*]const f64, o: [*]f64, n: u32) void { binaryV2_f64(a, b, o, n, powerOp_f64); }
export fn maximum_f64(a: [*]const f64, b: [*]const f64, o: [*]f64, n: u32) void { binaryV2_f64(a, b, o, n, maximumOp_f64); }
export fn minimum_f64(a: [*]const f64, b: [*]const f64, o: [*]f64, n: u32) void { binaryV2_f64(a, b, o, n, minimumOp_f64); }
export fn fmax_f64(a: [*]const f64, b: [*]const f64, o: [*]f64, n: u32) void { binaryV2_f64(a, b, o, n, fmaxOp_f64); }
export fn fmin_f64(a: [*]const f64, b: [*]const f64, o: [*]f64, n: u32) void { binaryV2_f64(a, b, o, n, fminOp_f64); }
export fn logaddexp_f64(a: [*]const f64, b: [*]const f64, o: [*]f64, n: u32) void { binaryV2_f64(a, b, o, n, logaddexpOp_f64); }
export fn logical_and_f64(a: [*]const f64, b: [*]const f64, o: [*]f64, n: u32) void { binaryV2_f64(a, b, o, n, logicalAndOp_f64); }
export fn logical_xor_f64(a: [*]const f64, b: [*]const f64, o: [*]f64, n: u32) void { binaryV2_f64(a, b, o, n, logicalXorOp_f64); }

// ─── f32 new ops ─────────
export fn copysign_f32(a: [*]const f32, b: [*]const f32, o: [*]f32, n: u32) void { binaryV4_f32(a, b, o, n, copysignOp_f32); }
export fn power_f32(a: [*]const f32, b: [*]const f32, o: [*]f32, n: u32) void { binaryV4_f32(a, b, o, n, powerOp_f32); }
export fn maximum_f32(a: [*]const f32, b: [*]const f32, o: [*]f32, n: u32) void { binaryV4_f32(a, b, o, n, maximumOp_f32); }
export fn minimum_f32(a: [*]const f32, b: [*]const f32, o: [*]f32, n: u32) void { binaryV4_f32(a, b, o, n, minimumOp_f32); }
export fn fmax_f32(a: [*]const f32, b: [*]const f32, o: [*]f32, n: u32) void { binaryV4_f32(a, b, o, n, fmaxOp_f32); }
export fn fmin_f32(a: [*]const f32, b: [*]const f32, o: [*]f32, n: u32) void { binaryV4_f32(a, b, o, n, fminOp_f32); }
export fn logaddexp_f32(a: [*]const f32, b: [*]const f32, o: [*]f32, n: u32) void { binaryV4_f32(a, b, o, n, logaddexpOp_f32); }
export fn logical_and_f32(a: [*]const f32, b: [*]const f32, o: [*]f32, n: u32) void { binaryV4_f32(a, b, o, n, logicalAndOp_f32); }
export fn logical_xor_f32(a: [*]const f32, b: [*]const f32, o: [*]f32, n: u32) void { binaryV4_f32(a, b, o, n, logicalXorOp_f32); }
