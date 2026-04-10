//! WASM RNG kernels matching NumPy's random number generation exactly.
//!
//! Implements MT19937, PCG64 (XSL-RR), SeedSequence, and Ziggurat distributions.
//! State is stored in Zig globals (WASM data segment), persisting across calls.

const std = @import("std");
const math = std.math;
const zt = @import("ziggurat_tables.zig");

// ============================================================================
// MT19937 (Mersenne Twister) Engine
// ============================================================================

// -- Exports --

/// Initialize MT19937 with a 32-bit seed.
/// Populates the 624-element state array using the same linear recurrence as
/// NumPy's `mt19937_seed` (multiplier 1812433253).
export fn mt19937_init(seed_val: u32) void {
    mt_initialized = true;
    var s: u32 = seed_val;
    var pos: u32 = 0;
    while (pos < MT_N) : (pos += 1) {
        mt_key[pos] = s;
        s = @as(u32, @truncate(@as(u64, 1812433253) *% @as(u64, s ^ (s >> 30)) +% (pos + 1)));
    }
    mt_pos = MT_N;
}

/// Generate a single random u32 with tempering.
/// Equivalent to NumPy's `mt19937_next` — draws from the MT state, twisting
/// when the 624-word buffer is exhausted.
export fn mt19937_genrand() u32 {
    return mt19937_next();
}

/// Generate a float64 in [0, 1) with 53-bit precision.
/// Combines two u32 draws (27 + 26 bits) exactly as NumPy's `mt19937_next_double`.
export fn mt19937_random_f64() f64 {
    const a: i32 = @bitCast(mt19937_next() >> 5); // 27 bits
    const b: i32 = @bitCast(mt19937_next() >> 6); // 26 bits
    return (@as(f64, @floatFromInt(a)) * 67108864.0 + @as(f64, @floatFromInt(b))) / 9007199254740992.0;
}

/// Copy the full 624-element MT19937 state into `out` and return the current
/// position index. Used for state serialization / checkpointing.
export fn mt19937_get_state(out: [*]u32) u32 {
    for (0..MT_N) |i| {
        out[i] = mt_key[i];
    }
    return mt_pos;
}

/// Restore MT19937 state from an external 624-element array and position index.
/// Counterpart to `mt19937_get_state` for deserialization.
export fn mt19937_set_state(state: [*]const u32, index: u32) void {
    for (0..MT_N) |i| {
        mt_key[i] = state[i];
    }
    mt_pos = index;
}

// -- Constants & State --

const MT_N: u32 = 624;
const MT_M: u32 = 397;
const MATRIX_A: u32 = 0x9908b0df;
const UPPER_MASK: u32 = 0x80000000;
const LOWER_MASK: u32 = 0x7fffffff;

var mt_key: [MT_N]u32 = [_]u32{0} ** MT_N;
var mt_pos: u32 = MT_N + 1;
var mt_initialized: bool = false;

// -- Internal --

/// Twist: regenerate the next 624 values in the state array.
fn mt19937_gen() void {
    var i: u32 = 0;
    while (i < MT_N - MT_M) : (i += 1) {
        const y = (mt_key[i] & UPPER_MASK) | (mt_key[i + 1] & LOWER_MASK);
        mt_key[i] = mt_key[i + MT_M] ^ (y >> 1) ^ ((0 -% (y & 1)) & MATRIX_A);
    }
    while (i < MT_N - 1) : (i += 1) {
        const y = (mt_key[i] & UPPER_MASK) | (mt_key[i + 1] & LOWER_MASK);
        mt_key[i] = mt_key[i + MT_M - MT_N] ^ (y >> 1) ^ ((0 -% (y & 1)) & MATRIX_A);
    }
    const y = (mt_key[MT_N - 1] & UPPER_MASK) | (mt_key[0] & LOWER_MASK);
    mt_key[MT_N - 1] = mt_key[MT_M - 1] ^ (y >> 1) ^ ((0 -% (y & 1)) & MATRIX_A);
    mt_pos = 0;
}

/// Draw and temper the next u32 from the MT state.
/// Auto-seeds with 5489 if never initialized (matches NumPy legacy default).
fn mt19937_next() u32 {
    if (mt_pos >= MT_N) {
        if (!mt_initialized) {
            mt19937_init(5489);
        }
        mt19937_gen();
    }
    var y = mt_key[mt_pos];
    mt_pos += 1;

    // Tempering
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= (y >> 18);
    return y;
}

// ============================================================================
// SeedSequence (NumPy's seed expansion algorithm)
// ============================================================================

// -- Exports --

/// Expand a single u32 seed into `n_words` output u32s using NumPy's
/// SeedSequence algorithm. The result in `out_ptr` is suitable for
/// initializing bit generators like PCG64 (which needs 8 words).
///
/// Implements the three-phase algorithm: initial hash mixing into a 4-word
/// pool, cross-mixing, then output generation with a separate multiplier.
export fn seed_sequence(seed: u32, out_ptr: [*]u32, n_words: u32) void {
    var mixer: [SS_POOL_SIZE]u32 = undefined;
    var hash_const: u32 = SS_INIT_A;

    // Phase 1: initial hash mixing
    mixer[0] = ss_hashmix(seed, &hash_const);
    for (1..SS_POOL_SIZE) |i| {
        mixer[i] = ss_hashmix(0, &hash_const);
    }

    // Phase 2: cross-mixing
    for (0..SS_POOL_SIZE) |i_src| {
        for (0..SS_POOL_SIZE) |i_dst| {
            if (i_src != i_dst) {
                const hashed = ss_hashmix(mixer[i_src], &hash_const);
                mixer[i_dst] = ss_mix(mixer[i_dst], hashed);
            }
        }
    }

    // Phase 3: generate output using MULT_B
    var hc: u32 = SS_INIT_B;
    for (0..n_words) |i| {
        const data_val = mixer[i % SS_POOL_SIZE];
        var value = data_val ^ hc;
        hc = hc *% SS_MULT_B;
        value = value *% hc;
        value = value ^ (value >> SS_XSHIFT);
        out_ptr[i] = value;
    }
}

// -- Constants --

const SS_MULT_A: u32 = 0x931e8875;
const SS_MULT_B: u32 = 0x58f38ded;
const SS_INIT_A: u32 = 0x43b0d7e5;
const SS_INIT_B: u32 = 0x8b51f9dd;
const SS_MIX_MULT_L: u32 = 0xca01f9dd;
const SS_MIX_MULT_R: u32 = 0x4973f715;
const SS_XSHIFT: u5 = 16;
const SS_POOL_SIZE: u32 = 4;

// -- Internal --

fn ss_hashmix(value: u32, hash_const: *u32) u32 {
    var v = value ^ hash_const.*;
    hash_const.* = hash_const.* *% SS_MULT_A;
    v = v *% hash_const.*;
    v = v ^ (v >> SS_XSHIFT);
    return v;
}

fn ss_mix(x: u32, y: u32) u32 {
    var result = (SS_MIX_MULT_L *% x) -% (SS_MIX_MULT_R *% y);
    result = result ^ (result >> SS_XSHIFT);
    return result;
}

// ============================================================================
// PCG64 (XSL-RR 128/64 variant)
// ============================================================================

// -- Exports --

/// Initialize PCG64 from SeedSequence output (8 u32 words).
/// Keeps all 64-bit combining in Zig to avoid JS number precision loss.
/// Word layout matches NumPy: [s0_lo, s0_hi, s1_lo, s1_hi, s2_lo, s2_hi, s3_lo, s3_hi]
/// where initState = (s0 << 64) | s1, initSeq = (s2 << 64) | s3.
export fn pcg64_init_from_ss(words: [*]const u32) void {
    const s0 = combine_u32_to_u64(words[0], words[1]);
    const s1 = combine_u32_to_u64(words[2], words[3]);
    const s2 = combine_u32_to_u64(words[4], words[5]);
    const s3 = combine_u32_to_u64(words[6], words[7]);

    const init_state: u128 = (@as(u128, s0) << 64) | @as(u128, s1);
    const init_seq: u128 = (@as(u128, s2) << 64) | @as(u128, s3);

    pcg64_init_raw(init_state, init_seq);
}

/// Initialize PCG64 with pre-computed 128-bit state and increment, passed as
/// two u64 pairs. Performs the 2-bump seeded initialization matching
/// NumPy's `pcg_setseq_128_srandom_r`.
fn pcg64_init(state_lo: u64, state_hi: u64, inc_lo: u64, inc_hi: u64) void {
    const init_state: u128 = (@as(u128, state_hi) << 64) | @as(u128, state_lo);
    const init_seq: u128 = (@as(u128, inc_hi) << 64) | @as(u128, inc_lo);
    pcg64_init_raw(init_state, init_seq);
}

/// Advance the PCG64 state and return the next u64 (XSL-RR output function).
/// Matches NumPy's `pcg64_random_r`.
fn pcg64_step() u64 {
    pcg64_step_internal();
    return pcg64_output(pcg_state);
}

/// Generate a float64 in [0, 1) with 53-bit precision from PCG64.
/// Drops the low 11 bits of the u64 and divides by 2^53.
export fn pcg64_random_f64() f64 {
    return @as(f64, @floatFromInt(pcg64_step() >> 11)) / 9007199254740992.0;
}

/// Return a random integer in [off, off + rng] inclusive.
/// Dispatches to 32-bit or 64-bit Lemire rejection depending on range size.
/// Matches NumPy's `random_bounded_uint64` (use_masked=false).
export fn pcg64_bounded_uint64(off: u64, rng: u64) u64 {
    if (rng == 0) {
        return off;
    } else if (rng <= 0xFFFFFFFF) {
        if (rng == 0xFFFFFFFF) {
            return off + @as(u64, pcg64_next_uint32());
        }
        return off + @as(u64, bounded_lemire_uint32(@truncate(rng)));
    } else {
        return off + bounded_lemire_uint64(rng);
    }
}

/// Fill `out[0..n]` with bounded random i64 in [off, off + rng] using PCG64 Lemire rejection.
/// Eliminates per-element JS↔WASM overhead for Generator.integers().
export fn fill_bounded_uint64_pcg(out: [*]i64, n: u32, off: u64, rng: u64) void {
    for (0..n) |i| {
        out[i] = @bitCast(pcg64_bounded_uint64(off, rng));
    }
}

/// Serialize PCG64 state to 6 u64 values:
/// [state_lo, state_hi, inc_lo, inc_hi, has_uint32, cached_uinteger].
export fn pcg64_get_state(out: [*]u64) void {
    out[0] = @truncate(pcg_state);
    out[1] = @truncate(pcg_state >> 64);
    out[2] = @truncate(pcg_inc);
    out[3] = @truncate(pcg_inc >> 64);
    out[4] = if (pcg_has_uint32) 1 else 0;
    out[5] = @as(u64, pcg_uinteger);
}

/// Set PCG64 state directly from u64 pairs (no initialization bumps).
/// Use this to restore a previously serialized state mid-stream.
fn pcg64_set_state(state_lo: u64, state_hi: u64, inc_lo: u64, inc_hi: u64) void {
    pcg_state = (@as(u128, state_hi) << 64) | @as(u128, state_lo);
    pcg_inc = (@as(u128, inc_hi) << 64) | @as(u128, inc_lo);
}

/// Set PCG64 state from a pointer to 6 u64 values (avoids JS BigInt precision loss).
/// Layout: [state_lo, state_hi, inc_lo, inc_hi, has_uint32, cached_uinteger].
export fn pcg64_set_state_ptr(data: [*]const u64) void {
    pcg_state = (@as(u128, data[1]) << 64) | @as(u128, data[0]);
    pcg_inc = (@as(u128, data[3]) << 64) | @as(u128, data[2]);
    pcg_has_uint32 = data[4] != 0;
    pcg_uinteger = @truncate(data[5]);
}

// -- Constants & State --

const PCG64_MULT: u128 = (@as(u128, 2549297995355413924) << 64) | 4865540595714422341;

var pcg_state: u128 = 0;
var pcg_inc: u128 = 0;
var pcg_has_uint32: bool = false;
var pcg_uinteger: u32 = 0;

// -- Internal --

/// Single LCG step: state = state * MULT + inc.
fn pcg64_step_internal() void {
    pcg_state = pcg_state *% PCG64_MULT +% pcg_inc;
}

/// XSL-RR output permutation: XOR high and low halves, then rotate.
fn pcg64_output(state: u128) u64 {
    const hi: u64 = @truncate(state >> 64);
    const lo: u64 = @truncate(state);
    const xored = hi ^ lo;
    const rot: u6 = @truncate(state >> 122);
    return math.rotr(u64, xored, rot);
}

/// Combine two u32 values into a u64 (little-endian: lo + hi<<32).
fn combine_u32_to_u64(lo: u32, hi: u32) u64 {
    return @as(u64, lo) | (@as(u64, hi) << 32);
}

/// 2-bump seeded initialization matching NumPy's pcg64 seeding.
/// inc = (initseq << 1) | 1, then two LCG steps with state addition between.
fn pcg64_init_raw(init_state: u128, init_seq: u128) void {
    pcg_inc = (init_seq << 1) | 1;
    pcg_state = 0;
    pcg_has_uint32 = false;
    pcg_uinteger = 0;

    pcg64_step_internal();
    pcg_state +%= init_state;
    pcg64_step_internal();
}

/// Buffered u32 from PCG64: returns lower 32 bits first, caches upper 32 bits.
/// Matches NumPy's `pcg64_next32`.
fn pcg64_next_uint32() u32 {
    if (pcg_has_uint32) {
        pcg_has_uint32 = false;
        return pcg_uinteger;
    }
    const next = pcg64_step();
    pcg_has_uint32 = true;
    pcg_uinteger = @truncate(next >> 32);
    return @truncate(next);
}

/// Lemire's nearly-divisionless bounded u32 rejection sampling.
/// `rng` is the inclusive max (range - 1). Must not be 0xFFFFFFFF.
fn bounded_lemire_uint32(rng: u32) u32 {
    const rng_excl: u64 = @as(u64, rng) + 1;

    var m: u64 = @as(u64, pcg64_next_uint32()) * rng_excl;
    var leftover: u32 = @truncate(m);

    if (leftover < @as(u32, @truncate(rng_excl))) {
        const threshold: u32 = (0 -% @as(u32, @truncate(rng_excl))) % @as(u32, @truncate(rng_excl));
        while (leftover < threshold) {
            m = @as(u64, pcg64_next_uint32()) * rng_excl;
            leftover = @truncate(m);
        }
    }

    return @truncate(m >> 32);
}

/// Lemire's bounded u64 rejection for ranges > 2^32.
fn bounded_lemire_uint64(rng: u64) u64 {
    const rng_excl: u64 = rng +% 1;

    var x: u64 = pcg64_step();
    var m: u128 = @as(u128, x) * @as(u128, rng_excl);
    var leftover: u64 = @truncate(m);

    if (leftover < rng_excl) {
        const threshold: u64 = (0 -% rng_excl) % rng_excl;
        while (leftover < threshold) {
            x = pcg64_step();
            m = @as(u128, x) * @as(u128, rng_excl);
            leftover = @truncate(m);
        }
    }

    return @truncate(m >> 64);
}

// ============================================================================
// Ziggurat Distributions (Standard Normal & Exponential)
// ============================================================================

// -- Exports --

/// Draw a single standard normal variate using the Ziggurat method with PCG64.
/// Matches NumPy's `random_standard_normal` from the Generator API.
export fn standard_normal_pcg() f64 {
    return standard_normal_impl(.pcg);
}

/// Draw a single standard exponential variate using the Ziggurat method with PCG64.
/// Matches NumPy's `random_standard_exponential` from the Generator API.
export fn standard_exponential_pcg() f64 {
    return standard_exponential_impl(.pcg);
}

/// Fill `out[0..n]` with standard normal variates (PCG64 + Ziggurat).
export fn fill_standard_normal_pcg(out: [*]f64, n: u32) void {
    for (0..n) |i| {
        out[i] = standard_normal_impl(.pcg);
    }
}

/// Fill `out[0..n]` with standard exponential variates (PCG64 + Ziggurat).
export fn fill_standard_exponential_pcg(out: [*]f64, n: u32) void {
    for (0..n) |i| {
        out[i] = standard_exponential_impl(.pcg);
    }
}

// -- Internal --

/// Selects which engine to use for the generic distribution implementations.
const GenType = enum(u8) { mt = 0, pcg = 1 };

fn next_u64(gen: GenType) u64 {
    return switch (gen) {
        .mt => (@as(u64, mt19937_next()) << 32) | mt19937_next(),
        .pcg => pcg64_step(),
    };
}

fn next_double(gen: GenType) f64 {
    return @as(f64, @floatFromInt(next_u64(gen) >> 11)) / 9007199254740992.0;
}

/// Ziggurat algorithm for standard normal. 99.3% of draws take the fast path
/// (single compare against the rectangle boundary). Tail sampling uses the
/// standard rejection method with exponential proposals.
fn standard_normal_impl(gen: GenType) f64 {
    while (true) {
        const r = next_u64(gen);
        const idx: u8 = @truncate(r);
        const rr = r >> 8;
        const sign: u1 = @truncate(rr);
        const rabs: u64 = (rr >> 1) & 0x000fffffffffffff;
        var x: f64 = @as(f64, @floatFromInt(rabs)) * zt.wi_double[idx];
        if (sign != 0) x = -x;

        if (rabs < zt.ki_double[idx]) {
            return x; // 99.3% fast path
        }
        if (idx == 0) {
            // Tail sampling
            while (true) {
                const xx = -zt.ziggurat_nor_inv_r * math.log1p(-next_double(gen));
                const yy = -math.log1p(-next_double(gen));
                if (yy + yy > xx * xx) {
                    return if ((rabs >> 8) & 0x1 != 0)
                        -(zt.ziggurat_nor_r + xx)
                    else
                        zt.ziggurat_nor_r + xx;
                }
            }
        } else {
            if ((zt.fi_double[idx - 1] - zt.fi_double[idx]) * next_double(gen) + zt.fi_double[idx] < @exp(-0.5 * x * x)) {
                return x;
            }
        }
    }
}

/// Ziggurat algorithm for standard exponential. 98.9% fast path.
fn standard_exponential_impl(gen: GenType) f64 {
    while (true) {
        var ri = next_u64(gen);
        ri >>= 3;
        const idx: u8 = @truncate(ri);
        ri >>= 8;
        const x: f64 = @as(f64, @floatFromInt(ri)) * zt.we_double[idx];
        if (ri < zt.ke_double[idx]) {
            return x; // 98.9% fast path
        }
        if (idx == 0) {
            return zt.ziggurat_exp_r - math.log1p(-next_double(gen));
        } else {
            if ((zt.fe_double[idx - 1] - zt.fe_double[idx]) * next_double(gen) + zt.fe_double[idx] < @exp(-x)) {
                return x;
            }
        }
    }
}

// ============================================================================
// Legacy Distributions (MT19937-based, matching NumPy's RandomState API)
// ============================================================================

// -- Exports --

/// Draw a single standard normal using the polar (Box-Muller) method.
/// Matches NumPy's `legacy_gauss` / `rk_gauss` — produces values in pairs and
/// caches the second value for the next call.
export fn legacy_gauss() f64 {
    return legacy_gauss_impl();
}

/// Draw a single standard exponential: -log(1 - U).
/// Matches NumPy's legacy `rk_standard_exponential`.
export fn legacy_standard_exponential() f64 {
    return legacy_standard_exponential_impl();
}

/// Reset the legacy Gauss cache. Must be called when re-seeding MT19937 to
/// avoid stale cached values contaminating the new stream.
export fn legacy_gauss_reset() void {
    legacy_has_gauss = false;
    legacy_gauss_cached = 0.0;
}

/// Fill `out[0..n]` with legacy standard normal variates (MT19937 + polar method).
export fn fill_legacy_gauss(out: [*]f64, n: u32) void {
    for (0..n) |i| {
        out[i] = legacy_gauss_impl();
    }
}

/// Fill `out[0..n]` with legacy standard exponential variates (MT19937).
export fn fill_legacy_standard_exponential(out: [*]f64, n: u32) void {
    for (0..n) |i| {
        out[i] = legacy_standard_exponential_impl();
    }
}

/// Draw a single legacy standard gamma variate.
/// Marsaglia & Tsang for shape >= 1, rejection for shape < 1.
export fn legacy_standard_gamma(shape: f64) f64 {
    return legacy_standard_gamma_impl(shape);
}

/// Fill `out[0..n]` with legacy standard gamma variates with given shape.
export fn fill_legacy_standard_gamma(out: [*]f64, n: u32, shape: f64) void {
    for (0..n) |i| {
        out[i] = legacy_standard_gamma_impl(shape);
    }
}

/// Fill `out[0..n]` with legacy chi-square variates: 2 * Gamma(df/2).
export fn fill_legacy_chisquare(out: [*]f64, n: u32, df: f64) void {
    const half_df = df / 2.0;
    for (0..n) |i| {
        out[i] = 2.0 * legacy_standard_gamma_impl(half_df);
    }
}

// -- State --

var legacy_has_gauss: bool = false;
var legacy_gauss_cached: f64 = 0.0;

// -- Internal --

/// Polar (Box-Muller) method with caching for legacy standard normal.
fn legacy_gauss_impl() f64 {
    if (legacy_has_gauss) {
        const temp = legacy_gauss_cached;
        legacy_gauss_cached = 0.0;
        legacy_has_gauss = false;
        return temp;
    }

    var x1: f64 = undefined;
    var x2: f64 = undefined;
    var r2: f64 = undefined;
    while (true) {
        x1 = 2.0 * mt19937_random_f64() - 1.0;
        x2 = 2.0 * mt19937_random_f64() - 1.0;
        r2 = x1 * x1 + x2 * x2;
        if (r2 < 1.0 and r2 != 0.0) break;
    }

    const f = @sqrt(-2.0 * @log(r2) / r2);
    legacy_gauss_cached = f * x1;
    legacy_has_gauss = true;
    return f * x2;
}

/// Legacy standard exponential: -log(1 - U).
fn legacy_standard_exponential_impl() f64 {
    return -@log(1.0 - mt19937_random_f64());
}

/// Legacy standard gamma using Marsaglia & Tsang (shape >= 1) or rejection (shape < 1).
fn legacy_standard_gamma_impl(shape: f64) f64 {
    if (shape == 1.0) {
        return legacy_standard_exponential_impl();
    } else if (shape == 0.0) {
        return 0.0;
    } else if (shape < 1.0) {
        while (true) {
            const U = mt19937_random_f64();
            const V = legacy_standard_exponential_impl();
            if (U <= 1.0 - shape) {
                const X = math.pow(f64, U, 1.0 / shape);
                if (X <= V) return X;
            } else {
                const Y = -@log((1.0 - U) / shape);
                const X = math.pow(f64, 1.0 - shape + shape * Y, 1.0 / shape);
                if (X <= V + Y) return X;
            }
        }
    } else {
        const b = shape - 1.0 / 3.0;
        const c = 1.0 / @sqrt(9.0 * b);
        while (true) {
            var X: f64 = undefined;
            var V: f64 = undefined;
            while (true) {
                X = legacy_gauss_impl();
                V = 1.0 + c * X;
                if (V > 0.0) break;
            }
            V = V * V * V;
            const U = mt19937_random_f64();
            if (U < 1.0 - 0.0331 * (X * X) * (X * X)) return b * V;
            if (@log(U) < 0.5 * X * X + b * (1.0 - V + @log(V))) return b * V;
        }
    }
}

/// Legacy chi-square: 2 * Gamma(df/2).
fn legacy_chisquare_impl(df: f64) f64 {
    return 2.0 * legacy_standard_gamma_impl(df / 2.0);
}

// ============================================================================
// Bulk Uniform Fill
// ============================================================================

/// Fill `out[0..n]` with uniform f64 in [0, 1) using MT19937.
export fn fill_uniform_f64_mt(out: [*]f64, n: u32) void {
    for (0..n) |i| {
        out[i] = mt19937_random_f64();
    }
}

/// Fill `out[0..n]` with uniform f64 in [0, 1) using PCG64.
export fn fill_uniform_f64_pcg(out: [*]f64, n: u32) void {
    for (0..n) |i| {
        out[i] = pcg64_random_f64();
    }
}

// ============================================================================
// Bounded Integer Generation
// ============================================================================

// -- Exports --

/// Fill `out[0..n]` with bounded random u32 in [0, max] inclusive using MT19937.
/// Uses bitmask rejection sampling matching NumPy's `rk_interval`.
export fn fill_rk_interval(out: [*]u32, n: u32, max: u32) void {
    for (0..n) |i| {
        out[i] = rk_interval(max);
    }
}

/// Fill `out[0..n]` with bounded random u8 in [off, off + rng_val] using MT19937.
/// Extracts 4 bytes from each u32 draw for efficiency. Matches NumPy's legacy
/// buffered bounded masked uint8 for randint with int8/uint8 dtype.
export fn fill_randint_u8(out: [*]u8, n: u32, rng_val: u8, off: u8) void {
    if (rng_val == 0) {
        for (0..n) |i| {
            out[i] = off;
        }
        return;
    }
    const mask: u8 = blk: {
        var m: u8 = rng_val;
        m |= m >> 1;
        m |= m >> 2;
        m |= m >> 4;
        break :blk m;
    };

    var buf: u32 = 0;
    var bcnt: u32 = 0;

    for (0..n) |i| {
        while (true) {
            if (bcnt == 0) {
                buf = mt19937_next();
                bcnt = 3;
            } else {
                buf >>= 8;
                bcnt -= 1;
            }
            const val: u8 = @truncate(buf);
            if ((val & mask) <= rng_val) {
                out[i] = off +% (val & mask);
                break;
            }
        }
    }
}

/// Fill `out[0..n]` with bounded random u16 in [off, off + rng_val] using MT19937.
/// Extracts 2 uint16 values from each u32 draw. Matches NumPy's legacy buffered
/// bounded masked uint16 for randint with int16/uint16 dtype.
export fn fill_randint_u16(out: [*]u16, n: u32, rng_val: u16, off: u16) void {
    if (rng_val == 0) {
        for (0..n) |i| {
            out[i] = off;
        }
        return;
    }
    const mask: u16 = blk: {
        var m: u16 = rng_val;
        m |= m >> 1;
        m |= m >> 2;
        m |= m >> 4;
        m |= m >> 8;
        break :blk m;
    };

    var buf: u32 = 0;
    var bcnt: u32 = 0;

    for (0..n) |i| {
        while (true) {
            if (bcnt == 0) {
                buf = mt19937_next();
                bcnt = 1;
            } else {
                buf >>= 16;
                bcnt -= 1;
            }
            const val: u16 = @truncate(buf);
            if ((val & mask) <= rng_val) {
                out[i] = off +% (val & mask);
                break;
            }
        }
    }
}

/// Fill `out[0..n]` with bounded random i64 in [low, low + max] using MT19937.
/// Uses bitmask rejection (same as `fill_rk_interval`) then adds the `low` offset.
/// Eliminates the JS BigInt conversion loop for randint.
export fn fill_randint_i64(out: [*]i64, n: u32, max: u32, low: i64) void {
    for (0..n) |i| {
        out[i] = @as(i64, rk_interval(max)) + low;
    }
}

// ============================================================================
// Continuous Distributions (all MT19937 legacy unless noted)
// ============================================================================

/// Fill with Pareto variates: exp(Exp(1)/a) - 1.
export fn fill_pareto(out: [*]f64, n: u32, a: f64) void {
    for (0..n) |i| {
        out[i] = @exp(legacy_standard_exponential_impl() / a) - 1.0;
    }
}

/// Fill with power-distribution variates: (1 - exp(-Exp(1)))^(1/a).
export fn fill_power(out: [*]f64, n: u32, a: f64) void {
    for (0..n) |i| {
        out[i] = @exp(@log(1.0 - @exp(-legacy_standard_exponential_impl())) / a);
    }
}

/// Fill with Weibull variates: Exp(1)^(1/a).
export fn fill_weibull(out: [*]f64, n: u32, a: f64) void {
    for (0..n) |i| {
        if (a == 0.0) {
            out[i] = 0.0;
        } else {
            out[i] = @exp(@log(legacy_standard_exponential_impl()) / a);
        }
    }
}

/// Fill with logistic variates: loc + scale * log(U / (1 - U)).
export fn fill_logistic(out: [*]f64, n: u32, loc: f64, scale: f64) void {
    for (0..n) |i| {
        while (true) {
            const U = mt19937_random_f64();
            if (U > 0.0) {
                out[i] = loc + scale * @log(U / (1.0 - U));
                break;
            }
        }
    }
}

/// Fill with Gumbel (extreme value type I) variates: loc - scale * log(-log(1-U)).
export fn fill_gumbel(out: [*]f64, n: u32, loc: f64, scale: f64) void {
    for (0..n) |i| {
        while (true) {
            const U = 1.0 - mt19937_random_f64();
            if (U < 1.0) {
                out[i] = loc - scale * @log(-@log(U));
                break;
            }
        }
    }
}

/// Fill with Laplace variates: loc +/- scale * log(2U) depending on sign of U - 0.5.
export fn fill_laplace(out: [*]f64, n: u32, loc: f64, scale: f64) void {
    for (0..n) |i| {
        while (true) {
            const U = mt19937_random_f64();
            if (U == 0.0) continue;
            if (U >= 0.5) {
                out[i] = loc - scale * @log(2.0 - U - U);
            } else {
                out[i] = loc + scale * @log(U + U);
            }
            break;
        }
    }
}

/// Fill with Rayleigh variates: scale * sqrt(2 * Exp(1)).
export fn fill_rayleigh(out: [*]f64, n: u32, scale: f64) void {
    for (0..n) |i| {
        out[i] = scale * @sqrt(2.0 * legacy_standard_exponential_impl());
    }
}

/// Fill with triangular distribution variates over [left, right] with given mode.
/// Uses the inverse-CDF method with a uniform draw.
export fn fill_triangular(out: [*]f64, n: u32, left: f64, mode: f64, right: f64) void {
    const base = right - left;
    const leftbase = mode - left;
    const ratio = leftbase / base;
    const leftprod = leftbase * base;
    const rightprod = (right - mode) * base;
    for (0..n) |i| {
        const U = mt19937_random_f64();
        if (U <= ratio) {
            out[i] = left + @sqrt(U * leftprod);
        } else {
            out[i] = right - @sqrt((1.0 - U) * rightprod);
        }
    }
}

/// Fill with standard Cauchy variates: ratio of two independent normals.
export fn fill_standard_cauchy(out: [*]f64, n: u32) void {
    for (0..n) |i| {
        out[i] = legacy_gauss_impl() / legacy_gauss_impl();
    }
}

/// Fill with lognormal variates: exp(mean + sigma * N(0,1)).
export fn fill_lognormal(out: [*]f64, n: u32, mean: f64, sigma: f64) void {
    for (0..n) |i| {
        out[i] = @exp(mean + sigma * legacy_gauss_impl());
    }
}

/// Fill with Wald (inverse Gaussian) variates using the Micheal/Schucany/Haas method.
export fn fill_wald(out: [*]f64, n: u32, mean: f64, scale: f64) void {
    const mu_2l = mean / (2.0 * scale);
    for (0..n) |i| {
        const g = legacy_gauss_impl();
        const Y = mean * g * g;
        const X = mean + mu_2l * (Y - @sqrt(4.0 * scale * Y + Y * Y));
        const U = mt19937_random_f64();
        if (U <= mean / (mean + X)) {
            out[i] = X;
        } else {
            out[i] = mean * mean / X;
        }
    }
}

/// Fill with Student's t variates: N(0,1) * sqrt(df / Gamma(df/2)) / sqrt(Gamma(df/2)).
export fn fill_standard_t(out: [*]f64, n: u32, df: f64) void {
    for (0..n) |i| {
        const num = legacy_gauss_impl();
        const denom = legacy_standard_gamma_impl(df / 2.0);
        out[i] = @sqrt(df / 2.0) * num / @sqrt(denom);
    }
}

/// Fill with Beta variates. Uses Johnk's algorithm when both a,b <= 1,
/// otherwise the Gamma ratio method.
export fn fill_beta(out: [*]f64, n: u32, a: f64, b: f64) void {
    for (0..n) |i| {
        if (a <= 1.0 and b <= 1.0) {
            // Johnk's algorithm
            while (true) {
                const U = mt19937_random_f64();
                const V = mt19937_random_f64();
                const X = math.pow(f64, U, 1.0 / a);
                const Y = math.pow(f64, V, 1.0 / b);
                if (X + Y <= 1.0) {
                    if (X + Y > 0.0) {
                        out[i] = X / (X + Y);
                    } else {
                        // Log-space fallback
                        const logX = @log(U) / a;
                        const logY = @log(V) / b;
                        const logM = if (logX > logY) logX else logY;
                        const lX = logX - logM;
                        const lY = logY - logM;
                        out[i] = @exp(lX - @log(@exp(lX) + @exp(lY)));
                    }
                    break;
                }
            }
        } else {
            const Ga = legacy_standard_gamma_impl(a);
            const Gb = legacy_standard_gamma_impl(b);
            out[i] = Ga / (Ga + Gb);
        }
    }
}

/// Fill with F-distribution variates: (ChiSq(dfnum)/dfnum) / (ChiSq(dfden)/dfden).
export fn fill_f(out: [*]f64, n: u32, dfnum: f64, dfden: f64) void {
    for (0..n) |i| {
        out[i] = (legacy_chisquare_impl(dfnum) * dfden) / (legacy_chisquare_impl(dfden) * dfnum);
    }
}

/// Fill with noncentral chi-square variates.
/// Uses the decomposition: ChiSq(df-1) + (N(0,1) + sqrt(nonc))^2 when df > 1,
/// or ChiSq(df + 2*Poisson(nonc/2)) otherwise.
export fn fill_noncentral_chisquare(out: [*]f64, n: u32, df: f64, nonc: f64) void {
    for (0..n) |i| {
        out[i] = legacy_noncentral_chisquare_impl(df, nonc);
    }
}

/// Fill with noncentral F variates: (NoncentralChiSq(dfnum,nonc)/dfnum) / (ChiSq(dfden)/dfden).
export fn fill_noncentral_f(out: [*]f64, n: u32, dfnum: f64, dfden: f64, nonc: f64) void {
    for (0..n) |i| {
        const t = legacy_noncentral_chisquare_impl(dfnum, nonc) * dfden;
        out[i] = t / (legacy_chisquare_impl(dfden) * dfnum);
    }
}

/// Fill with von Mises (circular normal) variates with mean `mu` and
/// concentration `kappa`. For very small kappa, falls back to uniform on (-pi, pi).
export fn fill_vonmises(out: [*]f64, n: u32, mu: f64, kappa: f64) void {
    for (0..n) |i| {
        if (kappa < 1e-8) {
            out[i] = math.pi * (2.0 * mt19937_random_f64() - 1.0);
            continue;
        }

        var s: f64 = undefined;
        if (kappa < 1e-5) {
            s = 1.0 / kappa + kappa;
        } else {
            const r = 1.0 + @sqrt(1.0 + 4.0 * kappa * kappa);
            const rho = (r - @sqrt(2.0 * r)) / (2.0 * kappa);
            s = (1.0 + rho * rho) / (2.0 * rho);
        }

        var W: f64 = undefined;
        while (true) {
            const U = mt19937_random_f64();
            const Z = @cos(math.pi * U);
            W = (1.0 + s * Z) / (s + Z);
            const Y = kappa * (s - W);
            const V = mt19937_random_f64();
            if ((Y * (2.0 - Y) - V >= 0.0) or (@log(Y / V) + 1.0 - Y >= 0.0)) {
                break;
            }
        }

        const U2 = mt19937_random_f64();
        var result = math.acos(W);
        if (U2 < 0.5) result = -result;
        result += mu;
        const neg = result < 0.0;
        var mod_val = @abs(result);
        mod_val = @mod(mod_val + math.pi, 2.0 * math.pi) - math.pi;
        if (neg) mod_val = -mod_val;
        out[i] = mod_val;
    }
}

// -- Internal (continuous) --

fn legacy_noncentral_chisquare_impl(df: f64, nonc: f64) f64 {
    if (nonc == 0.0) {
        return legacy_chisquare_impl(df);
    }
    if (1.0 < df) {
        const Chi2 = legacy_chisquare_impl(df - 1.0);
        const nn = legacy_gauss_impl() + @sqrt(nonc);
        return Chi2 + nn * nn;
    } else {
        const i = random_poisson_impl(nonc / 2.0);
        return legacy_chisquare_impl(df + 2.0 * @as(f64, @floatFromInt(i)));
    }
}

// ============================================================================
// Discrete Distributions (all MT19937 legacy)
// ============================================================================

// -- Exports --

/// Fill with Poisson variates. Uses multiplication method for lam < 10,
/// PTRS (transformed rejection with squeeze) for lam >= 10.
export fn fill_poisson(out: [*]i64, n: u32, lam: f64) void {
    for (0..n) |i| {
        out[i] = random_poisson_impl(lam);
    }
}

/// Fill with binomial variates. Dispatches to inversion (n*p <= 30) or
/// BTPE algorithm (n*p > 30) after folding p > 0.5 to 1-p.
export fn fill_binomial(out: [*]i64, n: u32, trials: i64, p: f64) void {
    for (0..n) |i| {
        out[i] = legacy_random_binomial_original(p, trials);
    }
}

/// Fill with negative binomial variates: Poisson(Gamma(n, (1-p)/p)).
export fn fill_negative_binomial(out: [*]i64, n: u32, nn: f64, p: f64) void {
    for (0..n) |i| {
        const Y = legacy_standard_gamma_impl(nn) * ((1.0 - p) / p);
        out[i] = random_poisson_impl(Y);
    }
}

/// Fill with geometric variates. Uses search for p >= 1/3, inversion otherwise.
export fn fill_geometric(out: [*]i64, n: u32, p: f64) void {
    for (0..n) |i| {
        if (p >= 0.333333333333333333333333) {
            out[i] = random_geometric_search_impl(p);
        } else {
            out[i] = legacy_geometric_inversion_impl(p);
        }
    }
}

/// Fill with hypergeometric variates. Uses the simple HYP algorithm for
/// sample <= 10, or the ratio-of-uniforms HRUA algorithm for larger samples.
export fn fill_hypergeometric(out: [*]i64, n: u32, good: i64, bad: i64, sample: i64) void {
    for (0..n) |i| {
        if (sample > 10) {
            out[i] = random_hypergeometric_hrua(good, bad, sample);
        } else if (sample > 0) {
            out[i] = random_hypergeometric_hyp(good, bad, sample);
        } else {
            out[i] = 0;
        }
    }
}

/// Fill with logarithmic series variates using the rejection method.
export fn fill_logseries(out: [*]i64, n: u32, p: f64) void {
    const r = math.log1p(-p);
    for (0..n) |i| {
        while (true) {
            const V = mt19937_random_f64();
            if (V >= p) {
                out[i] = 1;
                break;
            }
            const U = mt19937_random_f64();
            const q = -math.expm1(r * U);
            if (V <= q * q) {
                const result: i64 = @intFromFloat(@floor(1.0 + @log(V) / @log(q)));
                if (result < 1 or V == 0.0) continue;
                out[i] = result;
                break;
            }
            if (V >= q) {
                out[i] = 1;
                break;
            }
            out[i] = 2;
            break;
        }
    }
}

/// Fill with Zipf (zeta) variates using rejection.
export fn fill_zipf(out: [*]i64, n: u32, a: f64) void {
    const am1 = a - 1.0;
    const b = math.pow(f64, 2.0, am1);
    for (0..n) |i| {
        while (true) {
            const U = 1.0 - mt19937_random_f64();
            const V = mt19937_random_f64();
            const X = @floor(math.pow(f64, U, -1.0 / am1));
            if (X > 9007199254740992.0 or X < 1.0) continue;
            const T = math.pow(f64, 1.0 + 1.0 / X, am1);
            if (V * X * (T - 1.0) / (b - 1.0) <= T / b) {
                out[i] = @intFromFloat(X);
                break;
            }
        }
    }
}

// -- Internal (discrete) --

/// Poisson dispatch: PTRS for lam >= 10, multiplication method for small lam.
fn random_poisson_impl(lam: f64) i64 {
    if (lam >= 10.0) {
        return random_poisson_ptrs_impl(lam);
    } else if (lam == 0.0) {
        return 0;
    } else {
        return random_poisson_mult_impl(lam);
    }
}

/// Poisson via multiplication of uniform draws (small lambda).
fn random_poisson_mult_impl(lam: f64) i64 {
    const enlam = @exp(-lam);
    var X: i64 = 0;
    var prod: f64 = 1.0;
    while (true) {
        const U = mt19937_random_f64();
        prod *= U;
        if (prod > enlam) {
            X += 1;
        } else {
            return X;
        }
    }
}

const LS2PI: f64 = 0.91893853320467267;
const TWELFTH: f64 = 0.083333333333333333333333;

/// Log-gamma function used by Poisson PTRS and hypergeometric HRUA.
fn random_loggam(x: f64) f64 {
    const a = [10]f64{
        8.333333333333333e-02, -2.777777777777778e-03,
        7.936507936507937e-04, -5.952380952380952e-04,
        8.417508417508418e-04, -1.917526917526918e-03,
        6.410256410256410e-03, -2.955065359477124e-02,
        1.796443723688307e-01, -1.39243221690590e+00,
    };
    var n_steps: i64 = 0;
    if (x == 1.0 or x == 2.0) {
        return 0.0;
    } else if (x < 7.0) {
        n_steps = @intFromFloat(7.0 - x);
    }
    var x0 = x + @as(f64, @floatFromInt(n_steps));
    const x2 = (1.0 / x0) * (1.0 / x0);
    const lg2pi: f64 = 1.8378770664093453e+00;
    var gl0 = a[9];
    var k: i32 = 8;
    while (k >= 0) : (k -= 1) {
        gl0 *= x2;
        gl0 += a[@intCast(@as(u32, @bitCast(k)))];
    }
    var gl = gl0 / x0 + 0.5 * lg2pi + (x0 - 0.5) * @log(x0) - x0;
    if (x < 7.0) {
        var ki: i64 = 1;
        while (ki <= n_steps) : (ki += 1) {
            gl -= @log(x0 - 1.0);
            x0 -= 1.0;
        }
    }
    return gl;
}

/// Poisson PTRS: transformed rejection with squeeze for lam >= 10.
fn random_poisson_ptrs_impl(lam: f64) i64 {
    const slam = @sqrt(lam);
    const loglam = @log(lam);
    const b = 0.931 + 2.53 * slam;
    const a_val = -0.059 + 0.02483 * b;
    const invalpha = 1.1239 + 1.1328 / (b - 3.4);
    const vr = 0.9277 - 3.6224 / (b - 2.0);

    while (true) {
        const U = mt19937_random_f64() - 0.5;
        const V = mt19937_random_f64();
        const us = 0.5 - @abs(U);
        const k: i64 = @intFromFloat(@floor((2.0 * a_val / us + b) * U + lam + 0.43));
        if (us >= 0.07 and V <= vr) {
            return k;
        }
        if (k < 0 or (us < 0.013 and V > us)) {
            continue;
        }
        if (@log(V) + @log(invalpha) - @log(a_val / (us * us) + b) <=
            -lam + @as(f64, @floatFromInt(k)) * loglam - random_loggam(@as(f64, @floatFromInt(k)) + 1.0))
        {
            return k;
        }
    }
}

/// Binomial dispatch: fold p > 0.5, then choose inversion or BTPE.
fn legacy_random_binomial_original(p: f64, nn: i64) i64 {
    if (p <= 0.5) {
        if (p * @as(f64, @floatFromInt(nn)) <= 30.0) {
            return legacy_random_binomial_inversion(nn, p);
        } else {
            return random_binomial_btpe_impl(nn, p);
        }
    } else {
        const q = 1.0 - p;
        if (q * @as(f64, @floatFromInt(nn)) <= 30.0) {
            return nn - legacy_random_binomial_inversion(nn, q);
        } else {
            return nn - random_binomial_btpe_impl(nn, q);
        }
    }
}

const BinomialState = struct {
    has_binomial: bool = false,
    nsave: i64 = 0,
    psave: f64 = 0.0,
    r: f64 = 0.0,
    q: f64 = 0.0,
    fm: f64 = 0.0,
    m: i64 = 0,
    p1: f64 = 0.0,
    xm: f64 = 0.0,
    xl: f64 = 0.0,
    xr: f64 = 0.0,
    c: f64 = 0.0,
    laml: f64 = 0.0,
    lamr: f64 = 0.0,
    p2: f64 = 0.0,
    p3: f64 = 0.0,
    p4: f64 = 0.0,
};

var binomial_state: BinomialState = .{};

/// Binomial inversion method for small n*p.
fn legacy_random_binomial_inversion(nn: i64, p: f64) i64 {
    const n_f = @as(f64, @floatFromInt(nn));
    var q: f64 = undefined;
    var qn: f64 = undefined;
    var np: f64 = undefined;
    var bound_f: f64 = undefined;

    if (!binomial_state.has_binomial or binomial_state.nsave != nn or binomial_state.psave != p) {
        binomial_state.nsave = nn;
        binomial_state.psave = p;
        binomial_state.has_binomial = true;
        q = 1.0 - p;
        binomial_state.q = q;
        qn = @exp(n_f * @log(q));
        binomial_state.r = qn;
        np = n_f * p;
        binomial_state.c = np;
        bound_f = @min(n_f, np + 10.0 * @sqrt(np * q + 1.0));
        binomial_state.m = @intFromFloat(bound_f);
    } else {
        q = binomial_state.q;
        qn = binomial_state.r;
        np = binomial_state.c;
    }
    const bound = binomial_state.m;

    var X: i64 = 0;
    var px: f64 = qn;
    var U: f64 = mt19937_random_f64();
    while (U > px) {
        X += 1;
        if (X > bound) {
            X = 0;
            px = qn;
            U = mt19937_random_f64();
        } else {
            U -= px;
            px = (@as(f64, @floatFromInt(nn - X + 1)) * p * px) / (@as(f64, @floatFromInt(X)) * q);
        }
    }
    return X;
}

/// BTPE algorithm for binomial (Kachitvichyanukul & Schmeiser) for large n*p.
fn random_binomial_btpe_impl(nn: i64, p: f64) i64 {
    const n_f = @as(f64, @floatFromInt(nn));
    var r: f64 = undefined;
    var q: f64 = undefined;
    var fm: f64 = undefined;
    var p1: f64 = undefined;
    var xm: f64 = undefined;
    var xl: f64 = undefined;
    var xr: f64 = undefined;
    var c: f64 = undefined;
    var laml: f64 = undefined;
    var lamr: f64 = undefined;
    var p2: f64 = undefined;
    var p3: f64 = undefined;
    var p4: f64 = undefined;
    var m: i64 = undefined;

    if (!binomial_state.has_binomial or binomial_state.nsave != nn or binomial_state.psave != p) {
        binomial_state.nsave = nn;
        binomial_state.psave = p;
        binomial_state.has_binomial = true;
        r = @min(p, 1.0 - p);
        binomial_state.r = r;
        q = 1.0 - r;
        binomial_state.q = q;
        fm = n_f * r + r;
        binomial_state.fm = fm;
        m = @intFromFloat(@floor(fm));
        binomial_state.m = m;
        const m_f = @as(f64, @floatFromInt(m));
        p1 = @floor(2.195 * @sqrt(n_f * r * q) - 4.6 * q) + 0.5;
        binomial_state.p1 = p1;
        xm = m_f + 0.5;
        binomial_state.xm = xm;
        xl = xm - p1;
        binomial_state.xl = xl;
        xr = xm + p1;
        binomial_state.xr = xr;
        c = 0.134 + 20.5 / (15.3 + m_f);
        binomial_state.c = c;
        var a_val = (fm - xl) / (fm - xl * r);
        laml = a_val * (1.0 + a_val / 2.0);
        binomial_state.laml = laml;
        a_val = (xr - fm) / (xr * q);
        lamr = a_val * (1.0 + a_val / 2.0);
        binomial_state.lamr = lamr;
        p2 = p1 * (1.0 + 2.0 * c);
        binomial_state.p2 = p2;
        p3 = p2 + c / laml;
        binomial_state.p3 = p3;
        p4 = p3 + c / lamr;
        binomial_state.p4 = p4;
    } else {
        r = binomial_state.r;
        q = binomial_state.q;
        fm = binomial_state.fm;
        m = binomial_state.m;
        p1 = binomial_state.p1;
        xm = binomial_state.xm;
        xl = binomial_state.xl;
        xr = binomial_state.xr;
        c = binomial_state.c;
        laml = binomial_state.laml;
        lamr = binomial_state.lamr;
        p2 = binomial_state.p2;
        p3 = binomial_state.p3;
        p4 = binomial_state.p4;
    }

    const m_f = @as(f64, @floatFromInt(m));
    const nrq = n_f * r * q;

    while (true) {
        const u = mt19937_random_f64() * p4;
        var v = mt19937_random_f64();
        var y: i64 = undefined;

        if (u <= p1) {
            y = @intFromFloat(@floor(xm - p1 * v + u));
        } else if (u <= p2) {
            const x = xl + (u - p1) / c;
            v = v * c + 1.0 - @abs(m_f - x + 0.5) / p1;
            if (v > 1.0) continue;
            y = @intFromFloat(@floor(x));
        } else if (u <= p3) {
            y = @intFromFloat(@floor(xl + @log(v) / laml));
            if (y < 0 or v == 0.0) continue;
            v = v * (u - p2) * laml;
        } else {
            y = @intFromFloat(@floor(xr - @log(v) / lamr));
            if (y > nn or v == 0.0) continue;
            v = v * (u - p3) * lamr;
        }

        const k_val: i64 = if (y > m) y - m else m - y;
        const k_f = @as(f64, @floatFromInt(k_val));

        if (k_val > 20 and @as(f64, @floatFromInt(k_val)) < (nrq / 2.0 - 1.0)) {
            const rho = (k_f / nrq) * ((k_f * (k_f / 3.0 + 0.625) + 0.16666666666666666) / nrq + 0.5);
            const t = -k_f * k_f / (2.0 * nrq);
            const A = @log(v);
            if (A < t - rho) {
                // accept
            } else if (A > t + rho) {
                continue;
            } else {
                // Full Stirling check
                const y_f = @as(f64, @floatFromInt(y));
                const x1 = y_f + 1.0;
                const f1 = m_f + 1.0;
                const z = n_f + 1.0 - m_f;
                const w = n_f - y_f + 1.0;
                const x2 = x1 * x1;
                const f2 = f1 * f1;
                const z2 = z * z;
                const w2 = w * w;
                if (A > (xm * @log(f1 / x1) + (n_f - m_f + 0.5) * @log(z / w) +
                    (y_f - m_f) * @log(w * r / (x1 * q)) +
                    (13680.0 - (462.0 - (132.0 - (99.0 - 140.0 / f2) / f2) / f2) / f2) / f1 / 166320.0 +
                    (13680.0 - (462.0 - (132.0 - (99.0 - 140.0 / z2) / z2) / z2) / z2) / z / 166320.0 +
                    (13680.0 - (462.0 - (132.0 - (99.0 - 140.0 / x2) / x2) / x2) / x2) / x1 / 166320.0 +
                    (13680.0 - (462.0 - (132.0 - (99.0 - 140.0 / w2) / w2) / w2) / w2) / w / 166320.0))
                {
                    continue;
                }
            }
        } else {
            const s = r / q;
            const a_val2 = s * (n_f + 1.0);
            var F: f64 = 1.0;
            if (m < y) {
                var ii = m + 1;
                while (ii <= y) : (ii += 1) {
                    F *= (a_val2 / @as(f64, @floatFromInt(ii)) - s);
                }
            } else if (m > y) {
                var ii = y + 1;
                while (ii <= m) : (ii += 1) {
                    F /= (a_val2 / @as(f64, @floatFromInt(ii)) - s);
                }
            }
            if (v > F) continue;
        }

        if (p > 0.5) {
            return nn - y;
        }
        return y;
    }
}

/// Geometric via sequential search (for p >= 1/3).
fn random_geometric_search_impl(p: f64) i64 {
    var X: i64 = 1;
    const q = 1.0 - p;
    var sum: f64 = p;
    var prod: f64 = p;
    const U = mt19937_random_f64();
    while (U > sum) {
        prod *= q;
        sum += prod;
        X += 1;
    }
    return X;
}

/// Geometric via inversion of the CDF (for p < 1/3).
fn legacy_geometric_inversion_impl(p: f64) i64 {
    const z = @ceil(math.log1p(-mt19937_random_f64()) / @log(1.0 - p));
    if (z >= 9.223372036854776e+18) {
        return math.maxInt(i64);
    }
    return @intFromFloat(z);
}

/// Hypergeometric HYP algorithm (simple, for small samples).
fn random_hypergeometric_hyp(good: i64, bad: i64, sample: i64) i64 {
    const d1 = bad + good - sample;
    const d2_init: f64 = @floatFromInt(@min(bad, good));

    var y = d2_init;
    var k = sample;
    while (y > 0.0) {
        const u = mt19937_random_f64();
        y -= @as(f64, @floatFromInt(@as(i64, @intFromFloat(@floor(u + y / @as(f64, @floatFromInt(d1 + k)))))));
        k -= 1;
        if (k == 0) break;
    }
    const z: i64 = @intFromFloat(d2_init - y);
    if (good > bad) {
        return sample - z;
    }
    return z;
}

const D1_HRUA: f64 = 1.7155277699214135;
const D2_HRUA: f64 = 0.8989161620588988;

/// Hypergeometric HRUA (ratio-of-uniforms) for larger samples.
fn random_hypergeometric_hrua(good: i64, bad: i64, sample: i64) i64 {
    const mingoodbad = @min(good, bad);
    const popsize = good + bad;
    const maxgoodbad = @max(good, bad);
    const m = @min(sample, popsize - sample);

    const mingoodbad_f: f64 = @floatFromInt(mingoodbad);
    const popsize_f: f64 = @floatFromInt(popsize);
    const maxgoodbad_f: f64 = @floatFromInt(maxgoodbad);
    const m_f: f64 = @floatFromInt(m);

    const d4 = mingoodbad_f / popsize_f;
    const d5 = 1.0 - d4;
    const d6 = m_f * d4 + 0.5;
    const d7 = @sqrt(@as(f64, @floatFromInt(popsize - m)) * @as(f64, @floatFromInt(sample)) * d4 * d5 / @as(f64, @floatFromInt(popsize - 1)) + 0.5);
    const d8 = D1_HRUA * d7 + D2_HRUA;
    const d9: i64 = @intFromFloat(@floor((m_f + 1.0) * (mingoodbad_f + 1.0) / (popsize_f + 2.0)));
    const d9_f: f64 = @floatFromInt(d9);
    const d10 = random_loggam(d9_f + 1.0) + random_loggam(mingoodbad_f - d9_f + 1.0) +
        random_loggam(m_f - d9_f + 1.0) + random_loggam(maxgoodbad_f - m_f + d9_f + 1.0);
    const d11 = @min(@min(m_f, mingoodbad_f) + 1.0, @floor(d6 + 16.0 * d7));

    while (true) {
        const X = mt19937_random_f64();
        const Y = mt19937_random_f64();
        const W = d6 + d8 * (Y - 0.5) / X;

        if (W < 0.0 or W >= d11) continue;

        const Z: i64 = @intFromFloat(@floor(W));
        const Z_f: f64 = @floatFromInt(Z);
        const T = d10 - (random_loggam(Z_f + 1.0) + random_loggam(mingoodbad_f - Z_f + 1.0) +
            random_loggam(m_f - Z_f + 1.0) + random_loggam(maxgoodbad_f - m_f + Z_f + 1.0));

        if ((X * (4.0 - X) - 3.0) <= T) {
            var result = Z;
            if (good > bad) result = m - result;
            if (m < sample) result = good - result;
            return result;
        }
        if (X * (X - T) >= 1.0) continue;
        if (2.0 * @log(X) <= T) {
            var result = Z;
            if (good > bad) result = m - result;
            if (m < sample) result = good - result;
            return result;
        }
    }
}

// ============================================================================
// Utility
// ============================================================================

/// Fisher-Yates shuffle of an f64 arange [0, n) in-place.
/// Writes the shuffled result to `out`. Matches NumPy's `permutation(n)`.
/// Uses `rk_interval` (bitmask rejection) for index selection, matching NumPy exactly.
export fn fill_permutation(out: [*]f64, n: u32) void {
    // Initialize arange
    for (0..n) |i| {
        out[i] = @floatFromInt(i);
    }
    // Fisher-Yates shuffle using rk_interval (bounded rejection with bitmask)
    if (n <= 1) return;
    var i: u32 = n - 1;
    while (i > 0) : (i -= 1) {
        const j = rk_interval(i);
        const tmp = out[i];
        out[i] = out[j];
        out[j] = tmp;
    }
}

/// Fisher-Yates shuffle of an i64 arange [0, n) using PCG64.
/// Uses `random_interval` (buffered uint32 bitmask rejection), matching
/// NumPy's `Generator.permutation(n)` / `_shuffle_raw` exactly.
export fn fill_permutation_pcg(out: [*]i64, n: u32) void {
    for (0..n) |i| {
        out[i] = @intCast(i);
    }
    if (n <= 1) return;
    var i: u32 = n - 1;
    while (i > 0) : (i -= 1) {
        const j = pcg_random_interval(i);
        const tmp = out[i];
        out[i] = out[j];
        out[j] = tmp;
    }
}

/// Bounded random u32 in [0, max] using PCG64's buffered uint32 with bitmask rejection.
/// Matches NumPy's `random_interval` for the PCG64 bit generator exactly.
fn pcg_random_interval(max: u32) u32 {
    if (max == 0) return 0;
    var mask: u32 = max;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    var value: u32 = pcg64_next_uint32() & mask;
    while (value > max) {
        value = pcg64_next_uint32() & mask;
    }
    return value;
}

/// Draw a single bounded random u32 in [0, max] using MT19937 bitmask rejection.
/// Matches NumPy's `rk_interval` exactly.
fn rk_interval(max: u32) u32 {
    if (max == 0) return 0;
    var mask: u32 = max;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    var value: u32 = mt19937_next() & mask;
    while (value > max) {
        value = mt19937_next() & mask;
    }
    return value;
}

// --- Tests ---

const testing = std.testing;

/// Helper: compare f64 with tolerance (14 decimal digits, matching NumPy validation suite).
fn expectCloseF64(actual: f64, expected: f64) !void {
    const tol = 1e-14;
    if (@abs(actual - expected) > tol * @max(@abs(expected), 1.0)) {
        std.debug.print("expected {d}, got {d}\n", .{ expected, actual });
        return error.TestExpectedApproxEql;
    }
}

// MT19937 Engine

test "MT19937 seed=42: first 5 raw u32 match NumPy" {
    mt19937_init(42);
    try testing.expectEqual(@as(u32, 1608637542), mt19937_next());
    try testing.expectEqual(@as(u32, 3421126067), mt19937_next());
    try testing.expectEqual(@as(u32, 4083286876), mt19937_next());
    try testing.expectEqual(@as(u32, 787846414), mt19937_next());
    try testing.expectEqual(@as(u32, 3143890026), mt19937_next());
}

test "MT19937 seed=0: first 3 raw u32 match NumPy" {
    mt19937_init(0);
    try testing.expectEqual(@as(u32, 2357136044), mt19937_next());
    try testing.expectEqual(@as(u32, 2546248239), mt19937_next());
    try testing.expectEqual(@as(u32, 3071714933), mt19937_next());
}

test "MT19937 seed=42: first 5 f64 match NumPy random()" {
    mt19937_init(42);
    const expected = [5]f64{
        0.3745401188473625,
        0.9507143064099162,
        0.7319939418114051,
        0.5986584841970366,
        0.15601864044243652,
    };
    for (expected) |e| {
        try expectCloseF64(mt19937_random_f64(), e);
    }
}

test "MT19937 state save/restore produces identical stream" {
    mt19937_init(42);
    // Advance 100 steps
    for (0..100) |_| _ = mt19937_next();

    // Save state
    var saved_key: [MT_N]u32 = undefined;
    for (0..MT_N) |i| saved_key[i] = mt_key[i];
    const saved_pos = mt_pos;

    // Draw 5 values
    var ref: [5]u32 = undefined;
    for (0..5) |i| ref[i] = mt19937_next();

    // Restore and verify identical
    for (0..MT_N) |i| mt_key[i] = saved_key[i];
    mt_pos = saved_pos;
    for (0..5) |i| {
        try testing.expectEqual(ref[i], mt19937_next());
    }
}

// SeedSequence

test "SeedSequence(42) produces 8 words matching NumPy" {
    var out: [8]u32 = undefined;
    seed_sequence(42, &out, 8);
    const expected = [8]u32{
        3444837047, 2669555309, 2046530742, 3581440988,
        1691623607, 2099784219, 1184028159, 862288241,
    };
    try testing.expectEqualSlices(u32, &expected, &out);
}

test "SeedSequence(0) is deterministic and non-trivial" {
    var out1: [8]u32 = undefined;
    var out2: [8]u32 = undefined;
    seed_sequence(0, &out1, 8);
    seed_sequence(0, &out2, 8);
    try testing.expectEqualSlices(u32, &out1, &out2);
    // At least one word should be nonzero
    var any_nonzero = false;
    for (out1) |w| {
        if (w != 0) any_nonzero = true;
    }
    try testing.expect(any_nonzero);
}

// PCG64 Engine

test "PCG64 seed=42: first 5 f64 match NumPy default_rng(42).random()" {
    var words: [8]u32 = undefined;
    seed_sequence(42, &words, 8);
    pcg64_init_from_ss(&words);
    const expected = [5]f64{
        0.7739560485559633,
        0.4388784397520523,
        0.8585979199113825,
        0.6973680290593639,
        0.09417734788764953,
    };
    for (expected) |e| {
        try expectCloseF64(pcg64_random_f64(), e);
    }
}

test "PCG64 seed=0: first 3 f64 match NumPy default_rng(0).random()" {
    var words: [8]u32 = undefined;
    seed_sequence(0, &words, 8);
    pcg64_init_from_ss(&words);
    const expected = [3]f64{
        0.6369616873214543,
        0.2697867137638703,
        0.04097352393619469,
    };
    for (expected) |e| {
        try expectCloseF64(pcg64_random_f64(), e);
    }
}

test "PCG64 state save/restore produces identical stream" {
    var words: [8]u32 = undefined;
    seed_sequence(42, &words, 8);
    pcg64_init_from_ss(&words);
    // Advance 50 steps
    for (0..50) |_| _ = pcg64_step();

    // Save state
    var state_buf: [6]u64 = undefined;
    pcg64_get_state(&state_buf);

    // Draw 5 values
    var ref: [5]u64 = undefined;
    for (0..5) |i| ref[i] = pcg64_step();

    // Restore and verify identical
    pcg64_set_state_ptr(&state_buf);
    for (0..5) |i| {
        try testing.expectEqual(ref[i], pcg64_step());
    }
}

test "PCG64 bounded_uint64 stays in range" {
    var words: [8]u32 = undefined;
    seed_sequence(42, &words, 8);
    pcg64_init_from_ss(&words);
    // Draw 1000 bounded values and check range
    for (0..1000) |_| {
        const v = pcg64_bounded_uint64(10, 89); // [10, 99]
        try testing.expect(v >= 10);
        try testing.expect(v <= 99);
    }
}

// Ziggurat Distributions (PCG64)

test "PCG64 standard_normal seed=42: first 3 match NumPy" {
    var words: [8]u32 = undefined;
    seed_sequence(42, &words, 8);
    pcg64_init_from_ss(&words);
    const expected = [3]f64{
        0.30471707975443135,
        -1.0399841062404955,
        0.7504511958064572,
    };
    for (expected) |e| {
        try expectCloseF64(standard_normal_impl(.pcg), e);
    }
}

test "PCG64 standard_exponential seed=42: first 3 match NumPy" {
    var words: [8]u32 = undefined;
    seed_sequence(42, &words, 8);
    pcg64_init_from_ss(&words);
    const expected = [3]f64{
        2.4042086039659947,
        2.3361896558244535,
        2.384760999874255,
    };
    for (expected) |e| {
        try expectCloseF64(standard_exponential_impl(.pcg), e);
    }
}

// Legacy Distributions (MT19937)

test "Legacy gauss seed=42: first 5 match NumPy randn()" {
    mt19937_init(42);
    legacy_gauss_reset();
    const expected = [5]f64{
        0.4967141530112327,
        -0.13826430117118466,
        0.6476885381006925,
        1.5230298564080254,
        -0.23415337472333597,
    };
    for (expected) |e| {
        try expectCloseF64(legacy_gauss_impl(), e);
    }
}

test "Legacy standard_exponential seed=42: first 3 match NumPy" {
    mt19937_init(42);
    const expected = [3]f64{
        0.4692680899768591,
        3.010121430917521,
        1.3167456935454493,
    };
    for (expected) |e| {
        try expectCloseF64(legacy_standard_exponential_impl(), e);
    }
}

test "Legacy standard_gamma seed=42, shape=2.0: first 3 match NumPy" {
    mt19937_init(42);
    legacy_gauss_reset();
    const expected = [3]f64{
        2.3936793898692366,
        1.4944647302155876,
        1.3822835843709536,
    };
    for (expected) |e| {
        try expectCloseF64(legacy_standard_gamma_impl(2.0), e);
    }
}

test "Legacy gauss_reset clears cached value" {
    mt19937_init(42);
    legacy_gauss_reset();
    _ = legacy_gauss_impl(); // produces pair, caches one
    try testing.expect(legacy_has_gauss);
    legacy_gauss_reset();
    try testing.expect(!legacy_has_gauss);
    try testing.expectEqual(@as(f64, 0.0), legacy_gauss_cached);
}

// Bulk Fill — Bounded Integers

test "fill_rk_interval seed=42, max=9: first 10 match NumPy randint(0,10)" {
    mt19937_init(42);
    var out: [10]u32 = undefined;
    fill_rk_interval(&out, 10, 9);
    const expected = [10]u32{ 6, 3, 7, 4, 6, 9, 2, 6, 7, 4 };
    try testing.expectEqualSlices(u32, &expected, &out);
}

test "fill_rk_interval max=0 produces all zeros" {
    mt19937_init(0);
    var out: [5]u32 = undefined;
    fill_rk_interval(&out, 5, 0);
    for (out) |v| try testing.expectEqual(@as(u32, 0), v);
}

test "fill_randint_u8 stays in range" {
    mt19937_init(42);
    var out: [100]u8 = undefined;
    fill_randint_u8(&out, 100, 49, 10); // [10, 59]
    for (out) |v| {
        try testing.expect(v >= 10);
        try testing.expect(v <= 59);
    }
}

test "fill_randint_u16 stays in range" {
    mt19937_init(42);
    var out: [100]u16 = undefined;
    fill_randint_u16(&out, 100, 999, 100); // [100, 1099]
    for (out) |v| {
        try testing.expect(v >= 100);
        try testing.expect(v <= 1099);
    }
}

test "fill_randint_i64 stays in range" {
    mt19937_init(42);
    var out: [100]i64 = undefined;
    fill_randint_i64(&out, 100, 49, -20); // [-20, 29]
    for (out) |v| {
        try testing.expect(v >= -20);
        try testing.expect(v <= 29);
    }
}

// Bulk Fill — Uniform

test "fill_uniform_f64_mt seed=42 matches sequential mt19937_random_f64" {
    mt19937_init(42);
    var bulk: [10]f64 = undefined;
    fill_uniform_f64_mt(&bulk, 10);

    mt19937_init(42);
    for (0..10) |i| {
        try expectCloseF64(bulk[i], mt19937_random_f64());
    }
}

test "fill_uniform_f64_pcg seed=42 matches sequential pcg64_random_f64" {
    var words: [8]u32 = undefined;
    seed_sequence(42, &words, 8);
    pcg64_init_from_ss(&words);
    var bulk: [10]f64 = undefined;
    fill_uniform_f64_pcg(&bulk, 10);

    seed_sequence(42, &words, 8);
    pcg64_init_from_ss(&words);
    for (0..10) |i| {
        try expectCloseF64(bulk[i], pcg64_random_f64());
    }
}

// Continuous Distributions — range & basic sanity

test "fill_pareto produces positive values" {
    mt19937_init(42);
    var out: [100]f64 = undefined;
    fill_pareto(&out, 100, 2.0);
    for (out) |v| try testing.expect(v >= 0.0);
}

test "fill_weibull produces non-negative values" {
    mt19937_init(42);
    var out: [100]f64 = undefined;
    fill_weibull(&out, 100, 1.5);
    for (out) |v| try testing.expect(v >= 0.0);
}

test "fill_rayleigh produces non-negative values" {
    mt19937_init(42);
    var out: [100]f64 = undefined;
    fill_rayleigh(&out, 100, 2.0);
    for (out) |v| try testing.expect(v >= 0.0);
}

test "fill_beta produces values in (0, 1)" {
    mt19937_init(42);
    legacy_gauss_reset();
    var out: [200]f64 = undefined;
    fill_beta(&out, 200, 2.0, 5.0);
    for (out) |v| {
        try testing.expect(v > 0.0);
        try testing.expect(v < 1.0);
    }
}

test "fill_power produces values in (0, 1)" {
    mt19937_init(42);
    var out: [100]f64 = undefined;
    fill_power(&out, 100, 3.0);
    for (out) |v| {
        try testing.expect(v > 0.0);
        try testing.expect(v < 1.0);
    }
}

test "fill_triangular produces values in [left, right]" {
    mt19937_init(42);
    var out: [200]f64 = undefined;
    fill_triangular(&out, 200, 1.0, 5.0, 10.0);
    for (out) |v| {
        try testing.expect(v >= 1.0);
        try testing.expect(v <= 10.0);
    }
}

test "fill_standard_t has mean near zero" {
    mt19937_init(42);
    legacy_gauss_reset();
    var out: [5000]f64 = undefined;
    fill_standard_t(&out, 5000, 10.0);
    var sum: f64 = 0;
    for (out) |v| sum += v;
    const mean = sum / 5000.0;
    try testing.expect(@abs(mean) < 0.1);
}

test "fill_f produces positive values" {
    mt19937_init(42);
    legacy_gauss_reset();
    var out: [100]f64 = undefined;
    fill_f(&out, 100, 5.0, 10.0);
    for (out) |v| try testing.expect(v > 0.0);
}

// Discrete Distributions — range & basic sanity

test "fill_poisson produces non-negative values" {
    mt19937_init(42);
    var out: [200]i64 = undefined;
    fill_poisson(&out, 200, 5.0);
    for (out) |v| try testing.expect(v >= 0);
}

test "fill_poisson lam=0 produces all zeros" {
    mt19937_init(42);
    var out: [10]i64 = undefined;
    fill_poisson(&out, 10, 0.0);
    for (out) |v| try testing.expectEqual(@as(i64, 0), v);
}

test "fill_binomial stays in [0, trials]" {
    mt19937_init(42);
    var out: [200]i64 = undefined;
    fill_binomial(&out, 200, 20, 0.3);
    for (out) |v| {
        try testing.expect(v >= 0);
        try testing.expect(v <= 20);
    }
}

test "fill_geometric produces values >= 1" {
    mt19937_init(42);
    var out: [200]i64 = undefined;
    fill_geometric(&out, 200, 0.5);
    for (out) |v| try testing.expect(v >= 1);
}

test "fill_negative_binomial produces non-negative values" {
    mt19937_init(42);
    legacy_gauss_reset();
    var out: [200]i64 = undefined;
    fill_negative_binomial(&out, 200, 5.0, 0.5);
    for (out) |v| try testing.expect(v >= 0);
}

test "fill_hypergeometric stays in [0, min(ngood, nsample)]" {
    mt19937_init(42);
    var out: [200]i64 = undefined;
    fill_hypergeometric(&out, 200, 30, 70, 20);
    for (out) |v| {
        try testing.expect(v >= 0);
        try testing.expect(v <= 20);
    }
}

test "fill_logseries produces values >= 1" {
    mt19937_init(42);
    var out: [200]i64 = undefined;
    fill_logseries(&out, 200, 0.6);
    for (out) |v| try testing.expect(v >= 1);
}

test "fill_zipf produces values >= 1" {
    mt19937_init(42);
    var out: [200]i64 = undefined;
    fill_zipf(&out, 200, 2.0);
    for (out) |v| try testing.expect(v >= 1);
}

// Utility

test "fill_permutation seed=42, n=6 matches NumPy" {
    mt19937_init(42);
    var out: [6]f64 = undefined;
    fill_permutation(&out, 6);
    const expected = [6]f64{ 0.0, 1.0, 5.0, 2.0, 4.0, 3.0 };
    try testing.expectEqualSlices(f64, &expected, &out);
}

test "fill_permutation is a valid permutation" {
    mt19937_init(123);
    const n = 10;
    var out: [n]f64 = undefined;
    fill_permutation(&out, n);
    // Every integer 0..9 should appear exactly once
    var seen: [n]bool = [_]bool{false} ** n;
    for (out) |v| {
        const idx: usize = @intFromFloat(v);
        try testing.expect(!seen[idx]);
        seen[idx] = true;
    }
    for (seen) |s| try testing.expect(s);
}

test "fill_permutation_pcg seed=42, n=10 matches NumPy Generator.permutation(10)" {
    var words: [8]u32 = undefined;
    seed_sequence(42, &words, 8);
    pcg64_init_from_ss(&words);
    var out: [10]i64 = undefined;
    fill_permutation_pcg(&out, 10);
    const expected = [10]i64{ 5, 6, 0, 7, 3, 2, 4, 9, 1, 8 };
    try testing.expectEqualSlices(i64, &expected, &out);
}

test "fill_permutation_pcg seed=42, n=20 matches NumPy Generator.permutation(20)" {
    var words: [8]u32 = undefined;
    seed_sequence(42, &words, 8);
    pcg64_init_from_ss(&words);
    var out: [20]i64 = undefined;
    fill_permutation_pcg(&out, 20);
    const expected = [20]i64{ 15, 9, 14, 7, 12, 10, 6, 19, 3, 0, 16, 5, 11, 18, 2, 4, 17, 1, 13, 8 };
    try testing.expectEqualSlices(i64, &expected, &out);
}

test "fill_permutation_pcg is a valid permutation" {
    var words: [8]u32 = undefined;
    seed_sequence(123, &words, 8);
    pcg64_init_from_ss(&words);
    const n = 15;
    var out: [n]i64 = undefined;
    fill_permutation_pcg(&out, n);
    var seen: [n]bool = [_]bool{false} ** n;
    for (out) |v| {
        const idx: usize = @intCast(v);
        try testing.expect(!seen[idx]);
        seen[idx] = true;
    }
    for (seen) |s| try testing.expect(s);
}

test "mt19937_genrand produces u32" {
    mt19937_init(42);
    const v = mt19937_genrand();
    try testing.expect(v >= 0);
}

test "mt19937_get_state and mt19937_set_state round-trip" {
    mt19937_init(99);
    var state: [624]u32 = undefined;
    const idx = mt19937_get_state(&state);
    mt19937_init(1);
    mt19937_set_state(&state, idx);
    mt19937_init(99);
    _ = mt19937_get_state(&state);
    try testing.expect(idx < 625);
}

test "pcg64_init produces valid output" {
    pcg64_init(12345, 67890, 1, 1);
    const v = pcg64_random_f64();
    try testing.expect(v >= 0.0 and v < 1.0);
}

test "fill_bounded_uint64_pcg stays in range" {
    pcg64_init(42, 0, 1, 1);
    var out: [5]i64 = undefined;
    fill_bounded_uint64_pcg(&out, 5, 0, 9);
    for (out) |v| {
        try testing.expect(v >= 0 and v <= 9);
    }
}

test "pcg64_set_state restores state" {
    pcg64_init(42, 0, 1, 1);
    var state: [6]u64 = undefined;
    pcg64_get_state(&state);
    const v1 = pcg64_random_f64();
    pcg64_set_state(state[0], state[1], state[2], state[3]);
    const v2 = pcg64_random_f64();
    try testing.expectEqual(v1, v2);
}

test "standard_normal_pcg is in plausible range" {
    pcg64_init(42, 0, 1, 1);
    const v = standard_normal_pcg();
    try testing.expect(v > -10.0 and v < 10.0);
}

test "standard_exponential_pcg is positive" {
    pcg64_init(42, 0, 1, 1);
    const v = standard_exponential_pcg();
    try testing.expect(v > 0.0);
}

test "fill_standard_normal_pcg values in plausible range" {
    pcg64_init(42, 0, 1, 1);
    var out: [10]f64 = undefined;
    fill_standard_normal_pcg(&out, 10);
    for (out) |v| {
        try testing.expect(v > -10.0 and v < 10.0);
    }
}

test "fill_standard_exponential_pcg values are positive" {
    pcg64_init(42, 0, 1, 1);
    var out: [10]f64 = undefined;
    fill_standard_exponential_pcg(&out, 10);
    for (out) |v| {
        try testing.expect(v > 0.0);
    }
}

test "legacy_gauss is in plausible range" {
    mt19937_init(42);
    const v = legacy_gauss();
    try testing.expect(v > -10.0 and v < 10.0);
}

test "legacy_standard_exponential is positive" {
    mt19937_init(42);
    const v = legacy_standard_exponential();
    try testing.expect(v > 0.0);
}

test "fill_legacy_gauss values in plausible range" {
    mt19937_init(42);
    var out: [10]f64 = undefined;
    fill_legacy_gauss(&out, 10);
    for (out) |v| {
        try testing.expect(v > -10.0 and v < 10.0);
    }
}

test "fill_legacy_standard_exponential values are positive" {
    mt19937_init(42);
    var out: [10]f64 = undefined;
    fill_legacy_standard_exponential(&out, 10);
    for (out) |v| {
        try testing.expect(v > 0.0);
    }
}

test "legacy_standard_gamma shape=2 is positive" {
    mt19937_init(42);
    const v = legacy_standard_gamma(2.0);
    try testing.expect(v > 0.0);
}

test "fill_legacy_standard_gamma values are positive" {
    mt19937_init(42);
    var out: [10]f64 = undefined;
    fill_legacy_standard_gamma(&out, 10, 2.0);
    for (out) |v| {
        try testing.expect(v > 0.0);
    }
}

test "fill_legacy_chisquare values are positive" {
    mt19937_init(42);
    var out: [10]f64 = undefined;
    fill_legacy_chisquare(&out, 10, 3.0);
    for (out) |v| {
        try testing.expect(v > 0.0);
    }
}

test "fill_logistic values in plausible range" {
    pcg64_init(42, 0, 1, 1);
    var out: [10]f64 = undefined;
    fill_logistic(&out, 10, 0.0, 1.0);
    for (out) |v| {
        try testing.expect(v > -50.0 and v < 50.0);
    }
}

test "fill_gumbel values in plausible range" {
    pcg64_init(42, 0, 1, 1);
    var out: [10]f64 = undefined;
    fill_gumbel(&out, 10, 0.0, 1.0);
    for (out) |v| {
        try testing.expect(v > -50.0 and v < 100.0);
    }
}

test "fill_laplace values in plausible range" {
    pcg64_init(42, 0, 1, 1);
    var out: [10]f64 = undefined;
    fill_laplace(&out, 10, 0.0, 1.0);
    for (out) |v| {
        try testing.expect(v > -50.0 and v < 50.0);
    }
}

test "fill_standard_cauchy values are finite" {
    pcg64_init(42, 0, 1, 1);
    var out: [10]f64 = undefined;
    fill_standard_cauchy(&out, 10);
    for (out) |v| {
        try testing.expect(!std.math.isNan(v));
    }
}

test "fill_lognormal values are positive" {
    pcg64_init(42, 0, 1, 1);
    var out: [10]f64 = undefined;
    fill_lognormal(&out, 10, 0.0, 1.0);
    for (out) |v| {
        try testing.expect(v > 0.0);
    }
}

test "fill_wald values are positive" {
    pcg64_init(42, 0, 1, 1);
    var out: [10]f64 = undefined;
    fill_wald(&out, 10, 1.0, 1.0);
    for (out) |v| {
        try testing.expect(v > 0.0);
    }
}

test "fill_noncentral_chisquare values are positive" {
    pcg64_init(42, 0, 1, 1);
    var out: [10]f64 = undefined;
    fill_noncentral_chisquare(&out, 10, 3.0, 1.0);
    for (out) |v| {
        try testing.expect(v >= 0.0);
    }
}

test "fill_noncentral_f values are positive" {
    pcg64_init(42, 0, 1, 1);
    var out: [10]f64 = undefined;
    fill_noncentral_f(&out, 10, 2.0, 3.0, 1.0);
    for (out) |v| {
        try testing.expect(v >= 0.0);
    }
}

test "fill_vonmises values in [-pi, pi]" {
    pcg64_init(42, 0, 1, 1);
    var out: [10]f64 = undefined;
    fill_vonmises(&out, 10, 0.0, 1.0);
    for (out) |v| {
        try testing.expect(v >= -std.math.pi and v <= std.math.pi);
    }
}
