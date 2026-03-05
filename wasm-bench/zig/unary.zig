// WASM unary elementwise kernels for f32/f64 with SIMD
//
// Uses native v128 widths: @Vector(2,f64) / @Vector(4,f32)
// Two v128 loads/stores per iteration for throughput.
// Pointer-cast loads/stores to guarantee v128.load/v128.store opcodes.
//
// SIMD-native: sqrt, abs, neg, ceil, floor (map to WASM opcodes)
// Libm-style: exp, log, sin, cos (Zig builtins → LLVM intrinsics)

const simd = @import("simd.zig");

// ─── Generic helpers ───────────────────────────────────────────────────────

fn unaryV2_f64(in_ptr: [*]const f64, out_ptr: [*]f64, n: u32, comptime op: fn (simd.V2f64) simd.V2f64) void {
    const len = @as(usize, n);
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        simd.store2_f64(out_ptr, i, op(simd.load2_f64(in_ptr, i)));
        simd.store2_f64(out_ptr, i + 2, op(simd.load2_f64(in_ptr, i + 2)));
    }
    while (i + 2 <= len) : (i += 2) {
        simd.store2_f64(out_ptr, i, op(simd.load2_f64(in_ptr, i)));
    }
    while (i < len) : (i += 1) {
        const v: simd.V2f64 = .{ in_ptr[i], 0 };
        out_ptr[i] = op(v)[0];
    }
}

fn unaryV4_f32(in_ptr: [*]const f32, out_ptr: [*]f32, n: u32, comptime op: fn (simd.V4f32) simd.V4f32) void {
    const len = @as(usize, n);
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        simd.store4_f32(out_ptr, i, op(simd.load4_f32(in_ptr, i)));
        simd.store4_f32(out_ptr, i + 4, op(simd.load4_f32(in_ptr, i + 4)));
    }
    while (i + 4 <= len) : (i += 4) {
        simd.store4_f32(out_ptr, i, op(simd.load4_f32(in_ptr, i)));
    }
    while (i < len) : (i += 1) {
        const v: simd.V4f32 = .{ in_ptr[i], 0, 0, 0 };
        out_ptr[i] = op(v)[0];
    }
}

// ─── Scalar-loop driver for transcendentals ──────────────────────────────
// Zig's vector builtins (@exp, @log, @sin, @cos, @exp2) scalarize poorly
// in the self-hosted backend for WASM. Using explicit scalar loops with
// scalar builtins produces much better codegen (matching Rust's libm approach).

fn unaryScalar_f64(in_ptr: [*]const f64, out_ptr: [*]f64, n: u32, comptime op: fn (f64) f64) void {
    const len = @as(usize, n);
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        out_ptr[i] = op(in_ptr[i]);
        out_ptr[i + 1] = op(in_ptr[i + 1]);
        out_ptr[i + 2] = op(in_ptr[i + 2]);
        out_ptr[i + 3] = op(in_ptr[i + 3]);
    }
    while (i < len) : (i += 1) {
        out_ptr[i] = op(in_ptr[i]);
    }
}

fn unaryScalar_f32(in_ptr: [*]const f32, out_ptr: [*]f32, n: u32, comptime op: fn (f32) f32) void {
    const len = @as(usize, n);
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        out_ptr[i] = op(in_ptr[i]);
        out_ptr[i + 1] = op(in_ptr[i + 1]);
        out_ptr[i + 2] = op(in_ptr[i + 2]);
        out_ptr[i + 3] = op(in_ptr[i + 3]);
    }
    while (i < len) : (i += 1) {
        out_ptr[i] = op(in_ptr[i]);
    }
}

// ─── Op implementations ────────────────────────────────────────────────────

// SIMD-native ops (map to single WASM opcodes — keep as vector ops)
fn sqrtOp_f64(v: simd.V2f64) simd.V2f64 {
    return @sqrt(v);
}
fn absOp_f64(v: simd.V2f64) simd.V2f64 {
    return @abs(v);
}
fn negOp_f64(v: simd.V2f64) simd.V2f64 {
    return -v;
}
fn ceilOp_f64(v: simd.V2f64) simd.V2f64 {
    return @ceil(v);
}
fn floorOp_f64(v: simd.V2f64) simd.V2f64 {
    return @floor(v);
}
fn signbitOp_f64(v: simd.V2f64) simd.V2f64 {
    const sign_mask: simd.V2u64 = @splat(0x8000000000000000);
    const one: simd.V2f64 = @splat(1.0);
    const zero: simd.V2f64 = @splat(0.0);
    const has_sign = (@as(simd.V2u64, @bitCast(v)) & sign_mask) != @as(simd.V2u64, @splat(0));
    return @select(f64, has_sign, one, zero);
}

fn sqrtOp_f32(v: simd.V4f32) simd.V4f32 {
    return @sqrt(v);
}
fn absOp_f32(v: simd.V4f32) simd.V4f32 {
    return @abs(v);
}
fn negOp_f32(v: simd.V4f32) simd.V4f32 {
    return -v;
}
fn ceilOp_f32(v: simd.V4f32) simd.V4f32 {
    return @ceil(v);
}
fn floorOp_f32(v: simd.V4f32) simd.V4f32 {
    return @floor(v);
}

fn signbitOp_f32(v: simd.V4f32) simd.V4f32 {
    const sign_mask: simd.V4u32 = @splat(0x80000000);
    const one: simd.V4f32 = @splat(1.0);
    const zero: simd.V4f32 = @splat(0.0);
    const has_sign = (@as(simd.V4u32, @bitCast(v)) & sign_mask) != @as(simd.V4u32, @splat(0));
    return @select(f32, has_sign, one, zero);
}

// Transcendental scalar ops (f64)
fn scalarExp_f64(x: f64) f64 {
    return @exp(x);
}
fn scalarLog_f64(x: f64) f64 {
    return @log(x);
}
fn scalarSin_f64(x: f64) f64 {
    return @sin(x);
}
fn scalarCos_f64(x: f64) f64 {
    return @cos(x);
}
fn scalarExp2_f64(x: f64) f64 {
    return @exp2(x);
}
fn scalarTan_f64(x: f64) f64 {
    return @tan(x);
}
fn scalarSinh_f64(x: f64) f64 {
    const e = @exp(x);
    return (e - 1.0 / e) * 0.5;
}
fn scalarCosh_f64(x: f64) f64 {
    const e = @exp(@abs(x));
    return (e + 1.0 / e) * 0.5;
}
fn scalarTanh_f64(x: f64) f64 {
    const e2 = @exp(2.0 * x);
    return (e2 - 1.0) / (e2 + 1.0);
}

// Transcendental scalar ops (f32)
fn scalarExp_f32(x: f32) f32 {
    return @exp(x);
}
fn scalarLog_f32(x: f32) f32 {
    return @log(x);
}
fn scalarSin_f32(x: f32) f32 {
    return @sin(x);
}
fn scalarCos_f32(x: f32) f32 {
    return @cos(x);
}
fn scalarExp2_f32(x: f32) f32 {
    return @exp2(x);
}
fn scalarTan_f32(x: f32) f32 {
    return @tan(x);
}
fn scalarSinh_f32(x: f32) f32 {
    const e = @exp(x);
    return (e - 1.0 / e) * 0.5;
}
fn scalarCosh_f32(x: f32) f32 {
    const e = @exp(@abs(x));
    return (e + 1.0 / e) * 0.5;
}
fn scalarTanh_f32(x: f32) f32 {
    const e2 = @exp(2.0 * x);
    return (e2 - 1.0) / (e2 + 1.0);
}

// ─── f64 exports ───────────────────────────────────────────────────────────

// SIMD-native ops (use vector driver)
export fn sqrt_f64(i: [*]const f64, o: [*]f64, n: u32) void {
    unaryV2_f64(i, o, n, sqrtOp_f64);
}
export fn abs_f64(i: [*]const f64, o: [*]f64, n: u32) void {
    unaryV2_f64(i, o, n, absOp_f64);
}
export fn neg_f64(i: [*]const f64, o: [*]f64, n: u32) void {
    unaryV2_f64(i, o, n, negOp_f64);
}
export fn ceil_f64(i: [*]const f64, o: [*]f64, n: u32) void {
    unaryV2_f64(i, o, n, ceilOp_f64);
}
export fn floor_f64(i: [*]const f64, o: [*]f64, n: u32) void {
    unaryV2_f64(i, o, n, floorOp_f64);
}
export fn signbit_f64(i: [*]const f64, o: [*]f64, n: u32) void {
    unaryV2_f64(i, o, n, signbitOp_f64);
}

// Transcendental ops (use scalar driver to avoid poor vector scalarization)
export fn exp_f64(i: [*]const f64, o: [*]f64, n: u32) void {
    unaryScalar_f64(i, o, n, scalarExp_f64);
}
export fn log_f64(i: [*]const f64, o: [*]f64, n: u32) void {
    unaryScalar_f64(i, o, n, scalarLog_f64);
}
export fn sin_f64(i: [*]const f64, o: [*]f64, n: u32) void {
    unaryScalar_f64(i, o, n, scalarSin_f64);
}
export fn cos_f64(i: [*]const f64, o: [*]f64, n: u32) void {
    unaryScalar_f64(i, o, n, scalarCos_f64);
}
export fn tan_f64(i: [*]const f64, o: [*]f64, n: u32) void {
    unaryScalar_f64(i, o, n, scalarTan_f64);
}
export fn exp2_f64(i: [*]const f64, o: [*]f64, n: u32) void {
    unaryScalar_f64(i, o, n, scalarExp2_f64);
}
export fn sinh_f64(i: [*]const f64, o: [*]f64, n: u32) void {
    unaryScalar_f64(i, o, n, scalarSinh_f64);
}
export fn cosh_f64(i: [*]const f64, o: [*]f64, n: u32) void {
    unaryScalar_f64(i, o, n, scalarCosh_f64);
}
export fn tanh_f64(i: [*]const f64, o: [*]f64, n: u32) void {
    unaryScalar_f64(i, o, n, scalarTanh_f64);
}

// ─── f32 exports ───────────────────────────────────────────────────────────

// SIMD-native ops (use vector driver)
export fn sqrt_f32(i: [*]const f32, o: [*]f32, n: u32) void {
    unaryV4_f32(i, o, n, sqrtOp_f32);
}
export fn abs_f32(i: [*]const f32, o: [*]f32, n: u32) void {
    unaryV4_f32(i, o, n, absOp_f32);
}
export fn neg_f32(i: [*]const f32, o: [*]f32, n: u32) void {
    unaryV4_f32(i, o, n, negOp_f32);
}
export fn ceil_f32(i: [*]const f32, o: [*]f32, n: u32) void {
    unaryV4_f32(i, o, n, ceilOp_f32);
}
export fn floor_f32(i: [*]const f32, o: [*]f32, n: u32) void {
    unaryV4_f32(i, o, n, floorOp_f32);
}
export fn signbit_f32(i: [*]const f32, o: [*]f32, n: u32) void {
    unaryV4_f32(i, o, n, signbitOp_f32);
}

// Transcendental ops (use scalar driver)
export fn exp_f32(i: [*]const f32, o: [*]f32, n: u32) void {
    unaryScalar_f32(i, o, n, scalarExp_f32);
}
export fn log_f32(i: [*]const f32, o: [*]f32, n: u32) void {
    unaryScalar_f32(i, o, n, scalarLog_f32);
}
export fn sin_f32(i: [*]const f32, o: [*]f32, n: u32) void {
    unaryScalar_f32(i, o, n, scalarSin_f32);
}
export fn cos_f32(i: [*]const f32, o: [*]f32, n: u32) void {
    unaryScalar_f32(i, o, n, scalarCos_f32);
}
export fn tan_f32(i: [*]const f32, o: [*]f32, n: u32) void {
    unaryScalar_f32(i, o, n, scalarTan_f32);
}
export fn exp2_f32(i: [*]const f32, o: [*]f32, n: u32) void {
    unaryScalar_f32(i, o, n, scalarExp2_f32);
}
export fn sinh_f32(i: [*]const f32, o: [*]f32, n: u32) void {
    unaryScalar_f32(i, o, n, scalarSinh_f32);
}
export fn cosh_f32(i: [*]const f32, o: [*]f32, n: u32) void {
    unaryScalar_f32(i, o, n, scalarCosh_f32);
}
export fn tanh_f32(i: [*]const f32, o: [*]f32, n: u32) void {
    unaryScalar_f32(i, o, n, scalarTanh_f32);
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPLEX UNARY OPS (c128, c64)
// ═══════════════════════════════════════════════════════════════════════════

// abs_c128: |z| = sqrt(re²+im²). Input: 2n f64s, output: n f64s
export fn abs_c128(inp: [*]const f64, out: [*]f64, n: u32) void {
    const len = @as(usize, n);
    var idx: usize = 0;
    while (idx < len) : (idx += 1) {
        const re = inp[2 * idx];
        const im = inp[2 * idx + 1];
        out[idx] = @sqrt(re * re + im * im);
    }
}

// abs_c64: |z| = sqrt(re²+im²). Input: 2n f32s, output: n f32s
export fn abs_c64(inp: [*]const f32, out: [*]f32, n: u32) void {
    const len = @as(usize, n);
    var idx: usize = 0;
    // Process 4 complex numbers at a time
    while (idx + 4 <= len) : (idx += 4) {
        const v0 = simd.load4_f32(inp, 2 * idx); // [re0, im0, re1, im1]
        const v1 = simd.load4_f32(inp, 2 * idx + 4); // [re2, im2, re3, im3]
        const sq0 = v0 * v0;
        const sq1 = v1 * v1;
        // Sum adjacent pairs for magnitudes
        const mag0 = sq0[0] + sq0[1];
        const mag1 = sq0[2] + sq0[3];
        const mag2 = sq1[0] + sq1[1];
        const mag3 = sq1[2] + sq1[3];
        const mags: simd.V4f32 = .{ mag0, mag1, mag2, mag3 };
        simd.store4_f32(out, idx, @sqrt(mags));
    }
    while (idx < len) : (idx += 1) {
        const re = inp[2 * idx];
        const im = inp[2 * idx + 1];
        out[idx] = @sqrt(re * re + im * im);
    }
}

// exp_c128: exp(a+bi) = exp(a)*(cos(b)+i*sin(b)). Input/output: 2n f64s
export fn exp_c128(inp: [*]const f64, out: [*]f64, n: u32) void {
    const len = @as(usize, n);
    var idx: usize = 0;
    while (idx < len) : (idx += 1) {
        const re = inp[2 * idx];
        const im = inp[2 * idx + 1];
        const ea = @exp(re);
        out[2 * idx] = ea * @cos(im);
        out[2 * idx + 1] = ea * @sin(im);
    }
}

// exp_c64: exp(a+bi) = exp(a)*(cos(b)+i*sin(b)). Input/output: 2n f32s
export fn exp_c64(inp: [*]const f32, out: [*]f32, n: u32) void {
    const len = @as(usize, n);
    var idx: usize = 0;
    while (idx < len) : (idx += 1) {
        const re = inp[2 * idx];
        const im = inp[2 * idx + 1];
        const ea = @exp(re);
        out[2 * idx] = ea * @cos(im);
        out[2 * idx + 1] = ea * @sin(im);
    }
}
