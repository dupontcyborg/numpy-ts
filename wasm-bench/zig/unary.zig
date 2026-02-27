// WASM unary elementwise kernels: sqrt and exp for f32/f64 with SIMD

// ─── f64 unary ──────────────────────────────────────────────────────────────

export fn sqrt_f64(in_ptr: [*]const f64, out_ptr: [*]f64, n: u32) void {
    @setFloatMode(.optimized);
    const len = @as(usize, n);
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        const v: @Vector(4, f64) = .{ in_ptr[i], in_ptr[i + 1], in_ptr[i + 2], in_ptr[i + 3] };
        const r = @sqrt(v);
        out_ptr[i] = r[0];
        out_ptr[i + 1] = r[1];
        out_ptr[i + 2] = r[2];
        out_ptr[i + 3] = r[3];
    }
    while (i < len) : (i += 1) {
        out_ptr[i] = @sqrt(@as(@Vector(1, f64), .{in_ptr[i]}))[0];
    }
}

export fn exp_f64(in_ptr: [*]const f64, out_ptr: [*]f64, n: u32) void {
    @setFloatMode(.optimized);
    const len = @as(usize, n);
    var i: usize = 0;
    while (i + 4 <= len) : (i += 4) {
        const v: @Vector(4, f64) = .{ in_ptr[i], in_ptr[i + 1], in_ptr[i + 2], in_ptr[i + 3] };
        const r = @exp(v);
        out_ptr[i] = r[0];
        out_ptr[i + 1] = r[1];
        out_ptr[i + 2] = r[2];
        out_ptr[i + 3] = r[3];
    }
    while (i < len) : (i += 1) {
        out_ptr[i] = @exp(@as(@Vector(1, f64), .{in_ptr[i]}))[0];
    }
}

// ─── f32 unary ──────────────────────────────────────────────────────────────

export fn sqrt_f32(in_ptr: [*]const f32, out_ptr: [*]f32, n: u32) void {
    @setFloatMode(.optimized);
    const len = @as(usize, n);
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        const v: @Vector(8, f32) = .{
            in_ptr[i],     in_ptr[i + 1], in_ptr[i + 2], in_ptr[i + 3],
            in_ptr[i + 4], in_ptr[i + 5], in_ptr[i + 6], in_ptr[i + 7],
        };
        const r = @sqrt(v);
        out_ptr[i] = r[0];
        out_ptr[i + 1] = r[1];
        out_ptr[i + 2] = r[2];
        out_ptr[i + 3] = r[3];
        out_ptr[i + 4] = r[4];
        out_ptr[i + 5] = r[5];
        out_ptr[i + 6] = r[6];
        out_ptr[i + 7] = r[7];
    }
    while (i < len) : (i += 1) {
        out_ptr[i] = @sqrt(@as(@Vector(1, f32), .{in_ptr[i]}))[0];
    }
}

export fn exp_f32(in_ptr: [*]const f32, out_ptr: [*]f32, n: u32) void {
    @setFloatMode(.optimized);
    const len = @as(usize, n);
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        const v: @Vector(8, f32) = .{
            in_ptr[i],     in_ptr[i + 1], in_ptr[i + 2], in_ptr[i + 3],
            in_ptr[i + 4], in_ptr[i + 5], in_ptr[i + 6], in_ptr[i + 7],
        };
        const r = @exp(v);
        out_ptr[i] = r[0];
        out_ptr[i + 1] = r[1];
        out_ptr[i + 2] = r[2];
        out_ptr[i + 3] = r[3];
        out_ptr[i + 4] = r[4];
        out_ptr[i + 5] = r[5];
        out_ptr[i + 6] = r[6];
        out_ptr[i + 7] = r[7];
    }
    while (i < len) : (i += 1) {
        out_ptr[i] = @exp(@as(@Vector(1, f32), .{in_ptr[i]}))[0];
    }
}
