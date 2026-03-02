// WASM unary elementwise kernels for f32/f64 with SIMD
//
// Uses native v128 widths: @Vector(2,f64) / @Vector(4,f32)
// Two v128 loads/stores per iteration for throughput.
// Pointer-cast loads/stores to guarantee v128.load/v128.store opcodes.
//
// SIMD-native: sqrt, abs, neg, ceil, floor (map to WASM opcodes)
// Libm-style: exp, log, sin, cos (Zig builtins → LLVM intrinsics)

const V2f64 = @Vector(2, f64);
const V4f32 = @Vector(4, f32);

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

fn unaryV2_f64(in_ptr: [*]const f64, out_ptr: [*]f64, n: u32, comptime op: fn (V2f64) V2f64) void {
    const len = @as(usize, n);
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        store2_f64(out_ptr, i, op(load2_f64(in_ptr, i)));
        store2_f64(out_ptr, i + 2, op(load2_f64(in_ptr, i + 2)));
    }
    while (i + 2 <= len) : (i += 2) {
        store2_f64(out_ptr, i, op(load2_f64(in_ptr, i)));
    }
    while (i < len) : (i += 1) {
        const v: V2f64 = .{ in_ptr[i], 0 };
        out_ptr[i] = op(v)[0];
    }
}

fn unaryV4_f32(in_ptr: [*]const f32, out_ptr: [*]f32, n: u32, comptime op: fn (V4f32) V4f32) void {
    const len = @as(usize, n);
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        store4_f32(out_ptr, i, op(load4_f32(in_ptr, i)));
        store4_f32(out_ptr, i + 4, op(load4_f32(in_ptr, i + 4)));
    }
    while (i + 4 <= len) : (i += 4) {
        store4_f32(out_ptr, i, op(load4_f32(in_ptr, i)));
    }
    while (i < len) : (i += 1) {
        const v: V4f32 = .{ in_ptr[i], 0, 0, 0 };
        out_ptr[i] = op(v)[0];
    }
}

// ─── Op implementations ────────────────────────────────────────────────────

fn sqrtOp_f64(v: V2f64) V2f64 { return @sqrt(v); }
fn expOp_f64(v: V2f64) V2f64 { return @exp(v); }
fn logOp_f64(v: V2f64) V2f64 { return @log(v); }
fn sinOp_f64(v: V2f64) V2f64 { return @sin(v); }
fn cosOp_f64(v: V2f64) V2f64 { return @cos(v); }
fn absOp_f64(v: V2f64) V2f64 { return @abs(v); }
fn negOp_f64(v: V2f64) V2f64 { return -v; }
fn ceilOp_f64(v: V2f64) V2f64 { return @ceil(v); }
fn floorOp_f64(v: V2f64) V2f64 { return @floor(v); }

fn sqrtOp_f32(v: V4f32) V4f32 { return @sqrt(v); }
fn expOp_f32(v: V4f32) V4f32 { return @exp(v); }
fn logOp_f32(v: V4f32) V4f32 { return @log(v); }
fn sinOp_f32(v: V4f32) V4f32 { return @sin(v); }
fn cosOp_f32(v: V4f32) V4f32 { return @cos(v); }
fn absOp_f32(v: V4f32) V4f32 { return @abs(v); }
fn negOp_f32(v: V4f32) V4f32 { return -v; }
fn ceilOp_f32(v: V4f32) V4f32 { return @ceil(v); }
fn floorOp_f32(v: V4f32) V4f32 { return @floor(v); }

// ─── f64 exports ───────────────────────────────────────────────────────────

export fn sqrt_f64(i: [*]const f64, o: [*]f64, n: u32) void { unaryV2_f64(i, o, n, sqrtOp_f64); }
export fn exp_f64(i: [*]const f64, o: [*]f64, n: u32) void { unaryV2_f64(i, o, n, expOp_f64); }
export fn log_f64(i: [*]const f64, o: [*]f64, n: u32) void { unaryV2_f64(i, o, n, logOp_f64); }
export fn sin_f64(i: [*]const f64, o: [*]f64, n: u32) void { unaryV2_f64(i, o, n, sinOp_f64); }
export fn cos_f64(i: [*]const f64, o: [*]f64, n: u32) void { unaryV2_f64(i, o, n, cosOp_f64); }
export fn abs_f64(i: [*]const f64, o: [*]f64, n: u32) void { unaryV2_f64(i, o, n, absOp_f64); }
export fn neg_f64(i: [*]const f64, o: [*]f64, n: u32) void { unaryV2_f64(i, o, n, negOp_f64); }
export fn ceil_f64(i: [*]const f64, o: [*]f64, n: u32) void { unaryV2_f64(i, o, n, ceilOp_f64); }
export fn floor_f64(i: [*]const f64, o: [*]f64, n: u32) void { unaryV2_f64(i, o, n, floorOp_f64); }

// ─── f32 exports ───────────────────────────────────────────────────────────

export fn sqrt_f32(i: [*]const f32, o: [*]f32, n: u32) void { unaryV4_f32(i, o, n, sqrtOp_f32); }
export fn exp_f32(i: [*]const f32, o: [*]f32, n: u32) void { unaryV4_f32(i, o, n, expOp_f32); }
export fn log_f32(i: [*]const f32, o: [*]f32, n: u32) void { unaryV4_f32(i, o, n, logOp_f32); }
export fn sin_f32(i: [*]const f32, o: [*]f32, n: u32) void { unaryV4_f32(i, o, n, sinOp_f32); }
export fn cos_f32(i: [*]const f32, o: [*]f32, n: u32) void { unaryV4_f32(i, o, n, cosOp_f32); }
export fn abs_f32(i: [*]const f32, o: [*]f32, n: u32) void { unaryV4_f32(i, o, n, absOp_f32); }
export fn neg_f32(i: [*]const f32, o: [*]f32, n: u32) void { unaryV4_f32(i, o, n, negOp_f32); }
export fn ceil_f32(i: [*]const f32, o: [*]f32, n: u32) void { unaryV4_f32(i, o, n, ceilOp_f32); }
export fn floor_f32(i: [*]const f32, o: [*]f32, n: u32) void { unaryV4_f32(i, o, n, floorOp_f32); }
