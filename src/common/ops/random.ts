/**
 * Random number generation module
 *
 * Implements NumPy-compatible random functions:
 * - Legacy API (np.random.seed, random, rand, uniform, etc.) uses MT19937
 * - Modern API (np.random.default_rng) uses PCG64 with SeedSequence
 *
 * Both implementations produce identical output to NumPy when seeded.
 */

import { ArrayStorage } from '../storage';
import { type DType, isBigIntDType } from '../dtype';

// ============================================================================
// MT19937 (Mersenne Twister) - Used for legacy np.random.* functions
// Produces identical output to NumPy's np.random.seed() based functions
// ============================================================================

const MT_N = 624;
const MT_M = 397;
const MT_MATRIX_A = 0x9908b0df;
const MT_UPPER_MASK = 0x80000000;
const MT_LOWER_MASK = 0x7fffffff;

interface MT19937State {
  mt: Uint32Array;
  mti: number;
}

let _mtState: MT19937State = {
  mt: new Uint32Array(MT_N),
  mti: MT_N + 1,
};

/**
 * Initialize MT19937 with a seed
 */
function mtInit(seed: number): void {
  const mt = _mtState.mt;
  mt[0] = seed >>> 0;
  for (let i = 1; i < MT_N; i++) {
    const s = mt[i - 1]! ^ (mt[i - 1]! >>> 30);
    mt[i] = (Math.imul(1812433253, s) + i) >>> 0;
  }
  _mtState.mti = MT_N;
}

/**
 * Generate a random 32-bit integer using MT19937
 */
function mtRandom(): number {
  const mt = _mtState.mt;
  let y: number;
  const mag01 = [0, MT_MATRIX_A];

  if (_mtState.mti >= MT_N) {
    let kk: number;

    if (_mtState.mti === MT_N + 1) {
      mtInit(5489);
    }

    for (kk = 0; kk < MT_N - MT_M; kk++) {
      y = (mt[kk]! & MT_UPPER_MASK) | (mt[kk + 1]! & MT_LOWER_MASK);
      mt[kk] = mt[kk + MT_M]! ^ (y >>> 1) ^ mag01[y & 1]!;
    }
    for (; kk < MT_N - 1; kk++) {
      y = (mt[kk]! & MT_UPPER_MASK) | (mt[kk + 1]! & MT_LOWER_MASK);
      mt[kk] = mt[kk + (MT_M - MT_N)]! ^ (y >>> 1) ^ mag01[y & 1]!;
    }
    y = (mt[MT_N - 1]! & MT_UPPER_MASK) | (mt[0]! & MT_LOWER_MASK);
    mt[MT_N - 1] = mt[MT_M - 1]! ^ (y >>> 1) ^ mag01[y & 1]!;

    _mtState.mti = 0;
  }

  y = mt[_mtState.mti++]!;

  // Tempering
  y ^= y >>> 11;
  y ^= (y << 7) & 0x9d2c5680;
  y ^= (y << 15) & 0xefc60000;
  y ^= y >>> 18;

  return y >>> 0;
}

/**
 * Generate a random float in [0, 1) using MT19937 (53-bit precision)
 * Matches NumPy's random_standard_uniform exactly
 */
function mtRandomFloat53(): number {
  const a = mtRandom() >>> 5; // 27 bits
  const b = mtRandom() >>> 6; // 26 bits
  return (a * 67108864.0 + b) / 9007199254740992.0;
}

// ============================================================================
// SeedSequence - NumPy's seed expansion algorithm
// ============================================================================

const SS_MULT_A = 0x931e8875;
const SS_MULT_B = 0x58f38ded;
const SS_INIT_A = 0x43b0d7e5;
const SS_INIT_B = 0x8b51f9dd;
const SS_MIX_MULT_L = 0xca01f9dd;
const SS_MIX_MULT_R = 0x4973f715;
const SS_XSHIFT = 16;
const SS_POOL_SIZE = 4;

function u32(x: number): number {
  return x >>> 0;
}

/**
 * SeedSequence hashmix function - uses MULT_A
 */
function ssHashmix(value: number, hashConstRef: { val: number }): number {
  value = u32(u32(value) ^ hashConstRef.val);
  hashConstRef.val = u32(Math.imul(hashConstRef.val, SS_MULT_A));
  value = u32(Math.imul(value, hashConstRef.val));
  value = u32(value ^ (value >>> SS_XSHIFT));
  return value;
}

/**
 * SeedSequence mix function
 */
function ssMix(x: number, y: number): number {
  let result = u32(u32(Math.imul(SS_MIX_MULT_L, u32(x))) - u32(Math.imul(SS_MIX_MULT_R, u32(y))));
  result = u32(result ^ (result >>> SS_XSHIFT));
  return result;
}

/**
 * Create SeedSequence pool from entropy
 * Includes Phase 1 (initial mixing) and Phase 2 (cross-mixing)
 */
function seedSequence(seed: number): number[] {
  const mixer: number[] = [0, 0, 0, 0];
  const entropy = [seed >>> 0];
  const hashConst = { val: SS_INIT_A };

  // Phase 1: Initial hash mixing
  for (let i = 0; i < SS_POOL_SIZE; i++) {
    if (i < entropy.length) {
      mixer[i] = ssHashmix(entropy[i]!, hashConst);
    } else {
      mixer[i] = ssHashmix(0, hashConst);
    }
  }

  // Phase 2: Cross-mixing (ensures late bits affect early bits)
  for (let iSrc = 0; iSrc < SS_POOL_SIZE; iSrc++) {
    for (let iDst = 0; iDst < SS_POOL_SIZE; iDst++) {
      if (iSrc !== iDst) {
        const hashed = ssHashmix(mixer[iSrc]!, hashConst);
        mixer[iDst] = ssMix(mixer[iDst]!, hashed);
      }
    }
  }

  return mixer;
}

/**
 * Generate state uint32 array from pool
 * Note: Uses MULT_B (not MULT_A) for the inline hashmix operation
 */
function ssGenerateState(pool: number[], nWords: number): number[] {
  const result: number[] = [];
  let hashConst = SS_INIT_B;

  for (let i = 0; i < nWords; i++) {
    const dataVal = pool[i % SS_POOL_SIZE]!;
    // Inline hashmix-like operation using MULT_B
    let value = u32(dataVal ^ hashConst);
    hashConst = u32(Math.imul(hashConst, SS_MULT_B));
    value = u32(Math.imul(value, hashConst));
    value = u32(value ^ (value >>> SS_XSHIFT));
    result.push(value);
  }

  return result;
}

// ============================================================================
// PCG64 (XSL-RR variant) - Used for Generator class (default_rng)
// Uses 128-bit state and ADVANCE-THEN-OUTPUT order like NumPy
// ============================================================================

interface PCG64State {
  state: bigint;
  inc: bigint;
}

// PCG64 128-bit multiplier (from PCG paper)
const PCG64_MULT_LO = BigInt('4865540595714422341');
const PCG64_MULT_HI = BigInt('2549297995355413924');
const PCG64_MULT = (PCG64_MULT_HI << BigInt(64)) | PCG64_MULT_LO;
const MASK_64 = BigInt('0xffffffffffffffff');
const MASK_128 = (BigInt(1) << BigInt(128)) - BigInt(1);

/**
 * PCG64 XSL-RR output function
 */
function pcg64Output(state: bigint): bigint {
  const hi = state >> BigInt(64);
  const lo = state & MASK_64;
  const xored = (hi ^ lo) & MASK_64;
  const rot = Number(state >> BigInt(122));
  // 64-bit rotate right
  const rotated = ((xored >> BigInt(rot)) | (xored << BigInt(64 - rot))) & MASK_64;
  return rotated;
}

/**
 * PCG64 state advance function
 */
function pcg64Advance(state: bigint, inc: bigint): bigint {
  return (state * PCG64_MULT + inc) & MASK_128;
}

/**
 * Initialize PCG64 with SeedSequence (matches NumPy's default_rng exactly)
 */
function pcg64Init(seed: number): PCG64State {
  const pool = seedSequence(seed);
  const stateU32 = ssGenerateState(pool, 8);

  // Combine uint32 pairs into uint64 (little-endian)
  const s0 = BigInt(stateU32[0]!) | (BigInt(stateU32[1]!) << BigInt(32));
  const s1 = BigInt(stateU32[2]!) | (BigInt(stateU32[3]!) << BigInt(32));
  const s2 = BigInt(stateU32[4]!) | (BigInt(stateU32[5]!) << BigInt(32));
  const s3 = BigInt(stateU32[6]!) | (BigInt(stateU32[7]!) << BigInt(32));

  // Combine into 128-bit values: (high << 64) | low
  const initState = (s0 << BigInt(64)) | s1;
  let initInc = ((s2 << BigInt(64)) | s3) << BigInt(1);
  initInc = (initInc | BigInt(1)) & MASK_128; // Must be odd

  // PCG setseq initialization: 2 bumps
  let state = BigInt(0);
  state = pcg64Advance(state, initInc); // bump 1
  state = (state + initState) & MASK_128;
  state = pcg64Advance(state, initInc); // bump 2

  return { state, inc: initInc };
}

/**
 * PCG64 step: ADVANCE then OUTPUT (NumPy order)
 */
function pcg64Step(pcgState: PCG64State): bigint {
  // First advance
  pcgState.state = pcg64Advance(pcgState.state, pcgState.inc);
  // Then output from new state
  return pcg64Output(pcgState.state);
}

/**
 * Generate random float in [0, 1) from PCG64 (53-bit precision)
 */
function pcg64RandomFloat(pcgState: PCG64State): number {
  const u64 = pcg64Step(pcgState);
  const shifted = u64 >> BigInt(11);
  return Number(shifted) / 9007199254740992.0;
}

// ============================================================================
// Generator Class - Modern API using PCG64
// ============================================================================

/**
 * Random number generator class (matches NumPy's Generator from default_rng)
 */
export class Generator {
  private _pcgState: PCG64State;

  constructor(seedValue?: number) {
    if (seedValue !== undefined) {
      this._pcgState = pcg64Init(seedValue);
    } else {
      const randomSeed = Math.floor(Math.random() * 0x100000000);
      this._pcgState = pcg64Init(randomSeed);
    }
  }

  private _randomFloat(): number {
    return pcg64RandomFloat(this._pcgState);
  }

  random(size?: number | number[]): ArrayStorage | number {
    if (size === undefined) {
      return this._randomFloat();
    }
    const shape = Array.isArray(size) ? size : [size];
    const totalSize = shape.reduce((a, b) => a * b, 1);
    const result = ArrayStorage.zeros(shape, 'float64');
    const data = result.data as Float64Array;
    for (let i = 0; i < totalSize; i++) {
      data[i] = this._randomFloat();
    }
    return result;
  }

  integers(low: number, high?: number, size?: number | number[]): ArrayStorage | number {
    if (high === undefined) {
      high = low;
      low = 0;
    }
    if (size === undefined) {
      return Math.floor(this._randomFloat() * (high - low)) + low;
    }
    const shape = Array.isArray(size) ? size : [size];
    const totalSize = shape.reduce((a, b) => a * b, 1);
    const result = ArrayStorage.zeros(shape, 'int64');
    const data = result.data as BigInt64Array;
    const range = high - low;
    for (let i = 0; i < totalSize; i++) {
      data[i] = BigInt(Math.floor(this._randomFloat() * range) + low);
    }
    return result;
  }

  standard_normal(size?: number | number[]): ArrayStorage | number {
    if (size === undefined) {
      return boxMullerTransform(this._randomFloat.bind(this));
    }
    const shape = Array.isArray(size) ? size : [size];
    const totalSize = shape.reduce((a, b) => a * b, 1);
    const result = ArrayStorage.zeros(shape, 'float64');
    const data = result.data as Float64Array;
    for (let i = 0; i < totalSize; i += 2) {
      const [z0, z1] = boxMullerTransformPair(this._randomFloat.bind(this));
      data[i] = z0;
      if (i + 1 < totalSize) {
        data[i + 1] = z1;
      }
    }
    return result;
  }

  normal(loc: number = 0, scale: number = 1, size?: number | number[]): ArrayStorage | number {
    if (size === undefined) {
      return boxMullerTransform(this._randomFloat.bind(this)) * scale + loc;
    }
    const shape = Array.isArray(size) ? size : [size];
    const totalSize = shape.reduce((a, b) => a * b, 1);
    const result = ArrayStorage.zeros(shape, 'float64');
    const data = result.data as Float64Array;
    for (let i = 0; i < totalSize; i += 2) {
      const [z0, z1] = boxMullerTransformPair(this._randomFloat.bind(this));
      data[i] = z0 * scale + loc;
      if (i + 1 < totalSize) {
        data[i + 1] = z1 * scale + loc;
      }
    }
    return result;
  }

  uniform(low: number = 0, high: number = 1, size?: number | number[]): ArrayStorage | number {
    if (size === undefined) {
      return this._randomFloat() * (high - low) + low;
    }
    const shape = Array.isArray(size) ? size : [size];
    const totalSize = shape.reduce((a, b) => a * b, 1);
    const result = ArrayStorage.zeros(shape, 'float64');
    const data = result.data as Float64Array;
    const range = high - low;
    for (let i = 0; i < totalSize; i++) {
      data[i] = this._randomFloat() * range + low;
    }
    return result;
  }

  choice(
    a: number | ArrayStorage,
    size?: number | number[],
    replace: boolean = true,
    p?: ArrayStorage | number[]
  ): ArrayStorage | number {
    return choiceImpl(a, size, replace, p, this._randomFloat.bind(this));
  }

  permutation(x: number | ArrayStorage): ArrayStorage {
    return permutationImpl(x, this._randomFloat.bind(this));
  }

  shuffle(x: ArrayStorage): void {
    shuffleImpl(x, this._randomFloat.bind(this));
  }

  exponential(scale: number = 1, size?: number | number[]): ArrayStorage | number {
    if (size === undefined) {
      return -Math.log(1 - this._randomFloat()) * scale;
    }
    const shape = Array.isArray(size) ? size : [size];
    const totalSize = shape.reduce((a, b) => a * b, 1);
    const result = ArrayStorage.zeros(shape, 'float64');
    const data = result.data as Float64Array;
    for (let i = 0; i < totalSize; i++) {
      data[i] = -Math.log(1 - this._randomFloat()) * scale;
    }
    return result;
  }

  poisson(lam: number = 1, size?: number | number[]): ArrayStorage | number {
    if (size === undefined) {
      return poissonSample(lam, this._randomFloat.bind(this));
    }
    const shape = Array.isArray(size) ? size : [size];
    const totalSize = shape.reduce((a, b) => a * b, 1);
    const result = ArrayStorage.zeros(shape, 'int64');
    const data = result.data as BigInt64Array;
    for (let i = 0; i < totalSize; i++) {
      data[i] = BigInt(poissonSample(lam, this._randomFloat.bind(this)));
    }
    return result;
  }

  binomial(n: number, p: number, size?: number | number[]): ArrayStorage | number {
    if (size === undefined) {
      return binomialSample(n, p, this._randomFloat.bind(this));
    }
    const shape = Array.isArray(size) ? size : [size];
    const totalSize = shape.reduce((a, b) => a * b, 1);
    const result = ArrayStorage.zeros(shape, 'int64');
    const data = result.data as BigInt64Array;
    for (let i = 0; i < totalSize; i++) {
      data[i] = BigInt(binomialSample(n, p, this._randomFloat.bind(this)));
    }
    return result;
  }
}

/**
 * Create a new Generator instance (matches np.random.default_rng)
 * @param seedValue - Optional seed value
 */
export function default_rng(seedValue?: number): Generator {
  return new Generator(seedValue);
}

// ============================================================================
// Legacy API State Management (MT19937)
// ============================================================================

/**
 * Seed the random number generator (legacy API)
 * @param seedValue - Seed value (integer)
 */
export function seed(seedValue?: number | null): void {
  if (seedValue === undefined || seedValue === null) {
    seedValue = Math.floor(Date.now() ^ (Math.random() * 0x100000000));
  }
  mtInit(seedValue >>> 0);
}

/**
 * Get the internal state of the random number generator
 * @returns State object that can be used with set_state
 */
export function get_state(): { mt: number[]; mti: number } {
  return {
    mt: Array.from(_mtState.mt),
    mti: _mtState.mti,
  };
}

/**
 * Set the internal state of the random number generator
 * @param state - State object from get_state
 */
export function set_state(state: { mt: number[]; mti: number }): void {
  _mtState.mt = new Uint32Array(state.mt);
  _mtState.mti = state.mti;
}

// ============================================================================
// Helper functions for random distributions
// ============================================================================

/**
 * Box-Muller transform for generating standard normal random numbers
 */
function boxMullerTransform(rng: () => number): number {
  let u1: number, u2: number;
  do {
    u1 = rng();
    u2 = rng();
  } while (u1 === 0);
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/**
 * Box-Muller transform returning both values
 */
function boxMullerTransformPair(rng: () => number): [number, number] {
  let u1: number, u2: number;
  do {
    u1 = rng();
    u2 = rng();
  } while (u1 === 0);
  const r = Math.sqrt(-2 * Math.log(u1));
  const theta = 2 * Math.PI * u2;
  return [r * Math.cos(theta), r * Math.sin(theta)];
}

/**
 * Poisson random sample using inverse transform sampling
 */
function poissonSample(lam: number, rng: () => number): number {
  if (lam < 30) {
    const L = Math.exp(-lam);
    let k = 0;
    let p = 1;
    do {
      k++;
      p *= rng();
    } while (p > L);
    return k - 1;
  } else {
    const z = boxMullerTransform(rng);
    return Math.max(0, Math.round(lam + Math.sqrt(lam) * z));
  }
}

/**
 * Binomial random sample
 */
function binomialSample(n: number, p: number, rng: () => number): number {
  if (n * p < 10 && n * (1 - p) < 10) {
    let successes = 0;
    for (let i = 0; i < n; i++) {
      if (rng() < p) successes++;
    }
    return successes;
  } else {
    const mean = n * p;
    const std = Math.sqrt(n * p * (1 - p));
    const z = boxMullerTransform(rng);
    return Math.max(0, Math.min(n, Math.round(mean + std * z)));
  }
}

// ============================================================================
// Legacy API Top-level functions (using MT19937)
// ============================================================================

/**
 * Generate random floats in the half-open interval [0.0, 1.0)
 * @param size - Output shape. If not provided, returns a single float.
 */
export function random(size?: number | number[]): ArrayStorage | number {
  if (size === undefined) {
    return mtRandomFloat53();
  }
  const shape = Array.isArray(size) ? size : [size];
  const totalSize = shape.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shape, 'float64');
  const data = result.data as Float64Array;
  for (let i = 0; i < totalSize; i++) {
    data[i] = mtRandomFloat53();
  }
  return result;
}

/**
 * Random values in a given shape (alias for random with shape)
 * @param d0, d1, ..., dn - The dimensions of the returned array
 */
export function rand(...shape: number[]): ArrayStorage | number {
  if (shape.length === 0) {
    return mtRandomFloat53();
  }
  const totalSize = shape.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shape, 'float64');
  const data = result.data as Float64Array;
  for (let i = 0; i < totalSize; i++) {
    data[i] = mtRandomFloat53();
  }
  return result;
}

/**
 * Return random floats from standard normal distribution
 * @param d0, d1, ..., dn - The dimensions of the returned array
 */
export function randn(...shape: number[]): ArrayStorage | number {
  if (shape.length === 0) {
    return boxMullerTransform(mtRandomFloat53);
  }
  const totalSize = shape.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shape, 'float64');
  const data = result.data as Float64Array;
  for (let i = 0; i < totalSize; i += 2) {
    const [z0, z1] = boxMullerTransformPair(mtRandomFloat53);
    data[i] = z0;
    if (i + 1 < totalSize) {
      data[i + 1] = z1;
    }
  }
  return result;
}

/**
 * Return random integers from low (inclusive) to high (exclusive)
 * @param low - Lowest integer (or high if high is not provided)
 * @param high - One above the highest integer (optional)
 * @param size - Output shape
 * @param dtype - Output dtype (default 'int64')
 */
export function randint(
  low: number,
  high?: number | null,
  size?: number | number[],
  dtype: DType = 'int64'
): ArrayStorage | number {
  if (high === undefined || high === null) {
    high = low;
    low = 0;
  }
  const range = high - low;

  if (size === undefined) {
    return Math.floor(mtRandomFloat53() * range) + low;
  }

  const shape = Array.isArray(size) ? size : [size];
  const totalSize = shape.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shape, dtype);
  const data = result.data;

  if (isBigIntDType(dtype)) {
    const bigData = data as BigInt64Array | BigUint64Array;
    for (let i = 0; i < totalSize; i++) {
      bigData[i] = BigInt(Math.floor(mtRandomFloat53() * range) + low);
    }
  } else {
    const numData = data as Exclude<typeof data, BigInt64Array | BigUint64Array>;
    for (let i = 0; i < totalSize; i++) {
      numData[i] = Math.floor(mtRandomFloat53() * range) + low;
    }
  }
  return result;
}

/**
 * Draw samples from a uniform distribution
 * @param low - Lower boundary (default 0)
 * @param high - Upper boundary (default 1)
 * @param size - Output shape
 */
export function uniform(
  low: number = 0,
  high: number = 1,
  size?: number | number[]
): ArrayStorage | number {
  if (size === undefined) {
    return mtRandomFloat53() * (high - low) + low;
  }
  const shape = Array.isArray(size) ? size : [size];
  const totalSize = shape.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shape, 'float64');
  const data = result.data as Float64Array;
  const range = high - low;
  for (let i = 0; i < totalSize; i++) {
    data[i] = mtRandomFloat53() * range + low;
  }
  return result;
}

/**
 * Draw samples from a normal (Gaussian) distribution
 * @param loc - Mean of the distribution (default 0)
 * @param scale - Standard deviation (default 1)
 * @param size - Output shape
 */
export function normal(
  loc: number = 0,
  scale: number = 1,
  size?: number | number[]
): ArrayStorage | number {
  if (size === undefined) {
    return boxMullerTransform(mtRandomFloat53) * scale + loc;
  }
  const shape = Array.isArray(size) ? size : [size];
  const totalSize = shape.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shape, 'float64');
  const data = result.data as Float64Array;
  for (let i = 0; i < totalSize; i += 2) {
    const [z0, z1] = boxMullerTransformPair(mtRandomFloat53);
    data[i] = z0 * scale + loc;
    if (i + 1 < totalSize) {
      data[i + 1] = z1 * scale + loc;
    }
  }
  return result;
}

/**
 * Draw samples from a standard normal distribution (mean=0, std=1)
 * @param size - Output shape
 */
export function standard_normal(size?: number | number[]): ArrayStorage | number {
  if (size === undefined) {
    return boxMullerTransform(mtRandomFloat53);
  }
  const shape = Array.isArray(size) ? size : [size];
  const totalSize = shape.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shape, 'float64');
  const data = result.data as Float64Array;
  for (let i = 0; i < totalSize; i += 2) {
    const [z0, z1] = boxMullerTransformPair(mtRandomFloat53);
    data[i] = z0;
    if (i + 1 < totalSize) {
      data[i + 1] = z1;
    }
  }
  return result;
}

/**
 * Draw samples from an exponential distribution
 * @param scale - The scale parameter (beta = 1/lambda) (default 1)
 * @param size - Output shape
 */
export function exponential(scale: number = 1, size?: number | number[]): ArrayStorage | number {
  if (size === undefined) {
    return -Math.log(1 - mtRandomFloat53()) * scale;
  }
  const shape = Array.isArray(size) ? size : [size];
  const totalSize = shape.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shape, 'float64');
  const data = result.data as Float64Array;
  for (let i = 0; i < totalSize; i++) {
    data[i] = -Math.log(1 - mtRandomFloat53()) * scale;
  }
  return result;
}

/**
 * Draw samples from a Poisson distribution
 * @param lam - Expected number of events (lambda) (default 1)
 * @param size - Output shape
 */
export function poisson(lam: number = 1, size?: number | number[]): ArrayStorage | number {
  if (size === undefined) {
    return poissonSample(lam, mtRandomFloat53);
  }
  const shape = Array.isArray(size) ? size : [size];
  const totalSize = shape.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shape, 'int64');
  const data = result.data as BigInt64Array;
  for (let i = 0; i < totalSize; i++) {
    data[i] = BigInt(poissonSample(lam, mtRandomFloat53));
  }
  return result;
}

/**
 * Draw samples from a binomial distribution
 * @param n - Number of trials
 * @param p - Probability of success
 * @param size - Output shape
 */
export function binomial(n: number, p: number, size?: number | number[]): ArrayStorage | number {
  if (size === undefined) {
    return binomialSample(n, p, mtRandomFloat53);
  }
  const shape = Array.isArray(size) ? size : [size];
  const totalSize = shape.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shape, 'int64');
  const data = result.data as BigInt64Array;
  for (let i = 0; i < totalSize; i++) {
    data[i] = BigInt(binomialSample(n, p, mtRandomFloat53));
  }
  return result;
}

/**
 * Implementation of choice
 */
function choiceImpl(
  a: number | ArrayStorage,
  size?: number | number[],
  replace: boolean = true,
  p?: ArrayStorage | number[],
  rng: () => number = mtRandomFloat53
): ArrayStorage | number {
  let population: number[];
  if (typeof a === 'number') {
    population = Array.from({ length: a }, (_, i) => i);
  } else {
    const totalSize = a.size;
    population = [];
    for (let i = 0; i < totalSize; i++) {
      population.push(Number(a.iget(i)));
    }
  }

  const n = population.length;
  if (n === 0) {
    throw new Error('cannot take a sample from an empty sequence');
  }

  let probabilities: number[] | undefined;
  if (p !== undefined) {
    if (Array.isArray(p)) {
      probabilities = p;
    } else {
      const pSize = p.size;
      probabilities = [];
      for (let i = 0; i < pSize; i++) {
        probabilities.push(Number(p.iget(i)));
      }
    }
    if (probabilities.length !== n) {
      throw new Error('p and a must have the same size');
    }
    const sum = probabilities.reduce((a, b) => a + b, 0);
    if (Math.abs(sum - 1) > 1e-10) {
      probabilities = probabilities.map((x) => x / sum);
    }
  }

  if (size === undefined) {
    if (probabilities) {
      const r = rng();
      let cumsum = 0;
      for (let i = 0; i < n; i++) {
        cumsum += probabilities[i]!;
        if (r < cumsum) {
          return population[i]!;
        }
      }
      return population[n - 1]!;
    }
    return population[Math.floor(rng() * n)]!;
  }

  const shape = Array.isArray(size) ? size : [size];
  const totalSize = shape.reduce((a, b) => a * b, 1);

  if (!replace && totalSize > n) {
    throw new Error('cannot take a larger sample than population when replace=false');
  }

  const result = ArrayStorage.zeros(shape, 'float64');
  const data = result.data as Float64Array;

  if (replace) {
    if (probabilities) {
      const cumProbs = new Array(n);
      cumProbs[0] = probabilities[0]!;
      for (let i = 1; i < n; i++) {
        cumProbs[i] = cumProbs[i - 1] + probabilities[i]!;
      }

      for (let i = 0; i < totalSize; i++) {
        const r = rng();
        let idx = 0;
        while (idx < n - 1 && r >= cumProbs[idx]) {
          idx++;
        }
        data[i] = population[idx]!;
      }
    } else {
      for (let i = 0; i < totalSize; i++) {
        data[i] = population[Math.floor(rng() * n)]!;
      }
    }
  } else {
    const available = [...population];
    const availableProbs = probabilities ? [...probabilities] : undefined;

    for (let i = 0; i < totalSize; i++) {
      let idx: number;
      if (availableProbs) {
        const sum = availableProbs.reduce((a, b) => a + b, 0);
        const r = rng() * sum;
        let cumsum = 0;
        idx = 0;
        for (let j = 0; j < available.length; j++) {
          cumsum += availableProbs[j]!;
          if (r < cumsum) {
            idx = j;
            break;
          }
        }
        if (idx === 0 && r >= cumsum) {
          idx = available.length - 1;
        }
      } else {
        idx = Math.floor(rng() * available.length);
      }

      data[i] = available[idx]!;
      available.splice(idx, 1);
      if (availableProbs) {
        availableProbs.splice(idx, 1);
      }
    }
  }

  return result;
}

/**
 * Generate a random sample from a given 1-D array
 * @param a - Input array or int. If int, samples from arange(a)
 * @param size - Output shape
 * @param replace - Whether sampling with replacement (default true)
 * @param p - Probabilities associated with each entry
 */
export function choice(
  a: number | ArrayStorage,
  size?: number | number[],
  replace: boolean = true,
  p?: ArrayStorage | number[]
): ArrayStorage | number {
  return choiceImpl(a, size, replace, p, mtRandomFloat53);
}

/**
 * Implementation of permutation
 */
function permutationImpl(
  x: number | ArrayStorage,
  rng: () => number = mtRandomFloat53
): ArrayStorage {
  let arr: ArrayStorage;
  if (typeof x === 'number') {
    const data = new Float64Array(x);
    for (let i = 0; i < x; i++) {
      data[i] = i;
    }
    arr = ArrayStorage.fromData(data, [x], 'float64');
  } else {
    arr = x.copy();
  }

  const n = arr.size;
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const temp = arr.iget(i);
    arr.iset(i, arr.iget(j));
    arr.iset(j, temp);
  }

  return arr;
}

/**
 * Randomly permute a sequence, or return a permuted range
 * @param x - Input array or int. If int, returns permutation of arange(x)
 */
export function permutation(x: number | ArrayStorage): ArrayStorage {
  return permutationImpl(x, mtRandomFloat53);
}

/**
 * Implementation of shuffle (in-place)
 */
function shuffleImpl(x: ArrayStorage, rng: () => number = mtRandomFloat53): void {
  const n = x.size;
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const temp = x.iget(i);
    x.iset(i, x.iget(j));
    x.iset(j, temp);
  }
}

/**
 * Modify a sequence in-place by shuffling its contents
 * @param x - Array to be shuffled
 */
export function shuffle(x: ArrayStorage): void {
  shuffleImpl(x, mtRandomFloat53);
}

// ============================================================================
// Simple Aliases (for NumPy compatibility)
// ============================================================================

/**
 * Return random floats in the half-open interval [0.0, 1.0)
 * Alias for random()
 * @param size - Output shape
 */
export function random_sample(size?: number | number[]): ArrayStorage | number {
  return random(size);
}

/**
 * Return random floats in the half-open interval [0.0, 1.0)
 * Alias for random()
 * @param size - Output shape
 */
export function ranf(size?: number | number[]): ArrayStorage | number {
  return random(size);
}

/**
 * Return random floats in the half-open interval [0.0, 1.0)
 * Alias for random()
 * @param size - Output shape
 */
export function sample(size?: number | number[]): ArrayStorage | number {
  return random(size);
}

/**
 * Return random integers between low and high, inclusive (DEPRECATED)
 * @deprecated Use randint instead
 * @param low - Lowest integer
 * @param high - Highest integer (inclusive, unlike randint)
 * @param size - Output shape
 */
export function random_integers(
  low: number,
  high?: number,
  size?: number | number[]
): ArrayStorage | number {
  if (high === undefined) {
    high = low;
    low = 1;
  }
  // random_integers is inclusive on both ends, so add 1 to high for randint
  return randint(low, high + 1, size);
}

// ============================================================================
// Infrastructure functions
// ============================================================================

/**
 * Return random bytes
 * @param length - Number of bytes to return
 */
export function bytes(length: number): Uint8Array {
  const result = new Uint8Array(length);
  for (let i = 0; i < length; i++) {
    result[i] = mtRandom() & 0xff;
  }
  return result;
}

// Bit generator interface for compatibility
interface BitGenerator {
  name: string;
  state: object;
}

let _currentBitGenerator: BitGenerator = {
  name: 'MT19937',
  state: _mtState,
};

/**
 * Get the current bit generator
 * @returns The current bit generator object
 */
export function get_bit_generator(): BitGenerator {
  return _currentBitGenerator;
}

/**
 * Set the bit generator
 * @param bitgen - The bit generator to use
 */
export function set_bit_generator(bitgen: BitGenerator): void {
  _currentBitGenerator = bitgen;
}

// ============================================================================
// Standard distribution functions
// ============================================================================

/**
 * Draw samples from the standard exponential distribution (scale=1)
 * @param size - Output shape
 */
export function standard_exponential(size?: number | number[]): ArrayStorage | number {
  return exponential(1, size);
}

/**
 * Draw samples from a standard gamma distribution
 * Uses Marsaglia and Tsang's method
 * @param shape - Shape parameter (alpha, must be > 0)
 * @param size - Output shape
 */
export function standard_gamma(shape: number, size?: number | number[]): ArrayStorage | number {
  if (shape <= 0) {
    throw new Error('shape must be positive');
  }
  return gamma(shape, 1, size);
}

/**
 * Draw samples from a standard Cauchy distribution
 * @param size - Output shape
 */
export function standard_cauchy(size?: number | number[]): ArrayStorage | number {
  if (size === undefined) {
    return Math.tan(Math.PI * (mtRandomFloat53() - 0.5));
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shapeArr, 'float64');
  const data = result.data as Float64Array;
  for (let i = 0; i < totalSize; i++) {
    data[i] = Math.tan(Math.PI * (mtRandomFloat53() - 0.5));
  }
  return result;
}

/**
 * Draw samples from a standard Student's t distribution with df degrees of freedom
 * @param df - Degrees of freedom (must be > 0)
 * @param size - Output shape
 */
export function standard_t(df: number, size?: number | number[]): ArrayStorage | number {
  if (df <= 0) {
    throw new Error('df must be positive');
  }

  const generateSample = (rng: () => number): number => {
    // t = N(0,1) / sqrt(chi2(df) / df)
    const z = boxMullerTransform(rng);
    const chi2 = gammaSample(df / 2, 2, rng);
    return z / Math.sqrt(chi2 / df);
  };

  if (size === undefined) {
    return generateSample(mtRandomFloat53);
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shapeArr, 'float64');
  const data = result.data as Float64Array;
  for (let i = 0; i < totalSize; i++) {
    data[i] = generateSample(mtRandomFloat53);
  }
  return result;
}

// ============================================================================
// Gamma distribution (needed for many other distributions)
// ============================================================================

/**
 * Generate a single gamma sample using Marsaglia and Tsang's method
 */
function gammaSample(shape: number, scale: number, rng: () => number): number {
  if (shape < 1) {
    // For shape < 1, use Ahrens-Dieter method
    const u = rng();
    return gammaSample(1 + shape, scale, rng) * Math.pow(u, 1 / shape);
  }

  // Marsaglia and Tsang's method for shape >= 1
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);

  while (true) {
    let x: number;
    let v: number;

    do {
      x = boxMullerTransform(rng);
      v = 1 + c * x;
    } while (v <= 0);

    v = v * v * v;
    const u = rng();
    const x2 = x * x;

    if (u < 1 - 0.0331 * x2 * x2) {
      return d * v * scale;
    }

    if (Math.log(u) < 0.5 * x2 + d * (1 - v + Math.log(v))) {
      return d * v * scale;
    }
  }
}

/**
 * Draw samples from a Gamma distribution
 * @param shape - Shape parameter (k, alpha) (must be > 0)
 * @param scale - Scale parameter (theta) (default 1.0)
 * @param size - Output shape
 */
export function gamma(
  shape: number,
  scale: number = 1,
  size?: number | number[]
): ArrayStorage | number {
  if (shape <= 0) {
    throw new Error('shape must be positive');
  }
  if (scale <= 0) {
    throw new Error('scale must be positive');
  }

  if (size === undefined) {
    return gammaSample(shape, scale, mtRandomFloat53);
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shapeArr, 'float64');
  const data = result.data as Float64Array;
  for (let i = 0; i < totalSize; i++) {
    data[i] = gammaSample(shape, scale, mtRandomFloat53);
  }
  return result;
}

// ============================================================================
// Two-parameter continuous distributions
// ============================================================================

/**
 * Draw samples from a Beta distribution
 * @param a - Alpha parameter (must be > 0)
 * @param b - Beta parameter (must be > 0)
 * @param size - Output shape
 */
export function beta(a: number, b: number, size?: number | number[]): ArrayStorage | number {
  if (a <= 0 || b <= 0) {
    throw new Error('a and b must be positive');
  }

  const generateSample = (rng: () => number): number => {
    const x = gammaSample(a, 1, rng);
    const y = gammaSample(b, 1, rng);
    return x / (x + y);
  };

  if (size === undefined) {
    return generateSample(mtRandomFloat53);
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shapeArr, 'float64');
  const data = result.data as Float64Array;
  for (let i = 0; i < totalSize; i++) {
    data[i] = generateSample(mtRandomFloat53);
  }
  return result;
}

/**
 * Draw samples from a Laplace (double exponential) distribution
 * @param loc - Location parameter (default 0)
 * @param scale - Scale parameter (default 1)
 * @param size - Output shape
 */
export function laplace(
  loc: number = 0,
  scale: number = 1,
  size?: number | number[]
): ArrayStorage | number {
  if (scale <= 0) {
    throw new Error('scale must be positive');
  }

  const generateSample = (rng: () => number): number => {
    const u = rng() - 0.5;
    return loc - scale * Math.sign(u) * Math.log(1 - 2 * Math.abs(u));
  };

  if (size === undefined) {
    return generateSample(mtRandomFloat53);
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shapeArr, 'float64');
  const data = result.data as Float64Array;
  for (let i = 0; i < totalSize; i++) {
    data[i] = generateSample(mtRandomFloat53);
  }
  return result;
}

/**
 * Draw samples from a logistic distribution
 * @param loc - Location parameter (default 0)
 * @param scale - Scale parameter (default 1)
 * @param size - Output shape
 */
export function logistic(
  loc: number = 0,
  scale: number = 1,
  size?: number | number[]
): ArrayStorage | number {
  if (scale <= 0) {
    throw new Error('scale must be positive');
  }

  const generateSample = (rng: () => number): number => {
    const u = rng();
    return loc + scale * Math.log(u / (1 - u));
  };

  if (size === undefined) {
    return generateSample(mtRandomFloat53);
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shapeArr, 'float64');
  const data = result.data as Float64Array;
  for (let i = 0; i < totalSize; i++) {
    data[i] = generateSample(mtRandomFloat53);
  }
  return result;
}

/**
 * Draw samples from a log-normal distribution
 * @param mean - Mean of the underlying normal distribution (default 0)
 * @param sigma - Standard deviation of the underlying normal (default 1)
 * @param size - Output shape
 */
export function lognormal(
  mean: number = 0,
  sigma: number = 1,
  size?: number | number[]
): ArrayStorage | number {
  if (sigma <= 0) {
    throw new Error('sigma must be positive');
  }

  const generateSample = (rng: () => number): number => {
    return Math.exp(mean + sigma * boxMullerTransform(rng));
  };

  if (size === undefined) {
    return generateSample(mtRandomFloat53);
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shapeArr, 'float64');
  const data = result.data as Float64Array;
  for (let i = 0; i < totalSize; i++) {
    data[i] = generateSample(mtRandomFloat53);
  }
  return result;
}

/**
 * Draw samples from a Gumbel distribution
 * @param loc - Location parameter (default 0)
 * @param scale - Scale parameter (default 1)
 * @param size - Output shape
 */
export function gumbel(
  loc: number = 0,
  scale: number = 1,
  size?: number | number[]
): ArrayStorage | number {
  if (scale <= 0) {
    throw new Error('scale must be positive');
  }

  const generateSample = (rng: () => number): number => {
    const u = rng();
    return loc - scale * Math.log(-Math.log(u));
  };

  if (size === undefined) {
    return generateSample(mtRandomFloat53);
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shapeArr, 'float64');
  const data = result.data as Float64Array;
  for (let i = 0; i < totalSize; i++) {
    data[i] = generateSample(mtRandomFloat53);
  }
  return result;
}

/**
 * Draw samples from a Pareto II (Lomax) distribution
 * @param a - Shape parameter (must be > 0)
 * @param size - Output shape
 */
export function pareto(a: number, size?: number | number[]): ArrayStorage | number {
  if (a <= 0) {
    throw new Error('a must be positive');
  }

  const generateSample = (rng: () => number): number => {
    const u = rng();
    return Math.pow(1 - u, -1 / a) - 1;
  };

  if (size === undefined) {
    return generateSample(mtRandomFloat53);
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shapeArr, 'float64');
  const data = result.data as Float64Array;
  for (let i = 0; i < totalSize; i++) {
    data[i] = generateSample(mtRandomFloat53);
  }
  return result;
}

/**
 * Draw samples from a power distribution with positive exponent a-1
 * @param a - Shape parameter (must be > 0)
 * @param size - Output shape
 */
export function power(a: number, size?: number | number[]): ArrayStorage | number {
  if (a <= 0) {
    throw new Error('a must be positive');
  }

  const generateSample = (rng: () => number): number => {
    const u = rng();
    return Math.pow(u, 1 / a);
  };

  if (size === undefined) {
    return generateSample(mtRandomFloat53);
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shapeArr, 'float64');
  const data = result.data as Float64Array;
  for (let i = 0; i < totalSize; i++) {
    data[i] = generateSample(mtRandomFloat53);
  }
  return result;
}

/**
 * Draw samples from a Rayleigh distribution
 * @param scale - Scale parameter (default 1)
 * @param size - Output shape
 */
export function rayleigh(scale: number = 1, size?: number | number[]): ArrayStorage | number {
  if (scale <= 0) {
    throw new Error('scale must be positive');
  }

  const generateSample = (rng: () => number): number => {
    const u = rng();
    return scale * Math.sqrt(-2 * Math.log(u));
  };

  if (size === undefined) {
    return generateSample(mtRandomFloat53);
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shapeArr, 'float64');
  const data = result.data as Float64Array;
  for (let i = 0; i < totalSize; i++) {
    data[i] = generateSample(mtRandomFloat53);
  }
  return result;
}

/**
 * Draw samples from a triangular distribution
 * @param left - Lower limit
 * @param mode - Mode (peak)
 * @param right - Upper limit
 * @param size - Output shape
 */
export function triangular(
  left: number,
  mode: number,
  right: number,
  size?: number | number[]
): ArrayStorage | number {
  if (left > mode || mode > right || left === right) {
    throw new Error('must have left <= mode <= right and left < right');
  }

  const generateSample = (rng: () => number): number => {
    const u = rng();
    const fc = (mode - left) / (right - left);
    if (u < fc) {
      return left + Math.sqrt(u * (right - left) * (mode - left));
    } else {
      return right - Math.sqrt((1 - u) * (right - left) * (right - mode));
    }
  };

  if (size === undefined) {
    return generateSample(mtRandomFloat53);
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shapeArr, 'float64');
  const data = result.data as Float64Array;
  for (let i = 0; i < totalSize; i++) {
    data[i] = generateSample(mtRandomFloat53);
  }
  return result;
}

/**
 * Draw samples from a Wald (inverse Gaussian) distribution
 * @param mean - Mean of distribution (must be > 0)
 * @param scale - Scale parameter (must be > 0)
 * @param size - Output shape
 */
export function wald(mean: number, scale: number, size?: number | number[]): ArrayStorage | number {
  if (mean <= 0) {
    throw new Error('mean must be positive');
  }
  if (scale <= 0) {
    throw new Error('scale must be positive');
  }

  const generateSample = (rng: () => number): number => {
    const v = boxMullerTransform(rng);
    const y = v * v;
    const mu_2l = mean / (2 * scale);
    const x = mean + mu_2l * (mean * y - Math.sqrt(4 * mean * scale * y + mean * mean * y * y));
    const u = rng();
    if (u <= mean / (mean + x)) {
      return x;
    } else {
      return (mean * mean) / x;
    }
  };

  if (size === undefined) {
    return generateSample(mtRandomFloat53);
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shapeArr, 'float64');
  const data = result.data as Float64Array;
  for (let i = 0; i < totalSize; i++) {
    data[i] = generateSample(mtRandomFloat53);
  }
  return result;
}

/**
 * Draw samples from a Weibull distribution
 * @param a - Shape parameter (must be > 0)
 * @param size - Output shape
 */
export function weibull(a: number, size?: number | number[]): ArrayStorage | number {
  if (a <= 0) {
    throw new Error('a must be positive');
  }

  const generateSample = (rng: () => number): number => {
    const u = rng();
    return Math.pow(-Math.log(1 - u), 1 / a);
  };

  if (size === undefined) {
    return generateSample(mtRandomFloat53);
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shapeArr, 'float64');
  const data = result.data as Float64Array;
  for (let i = 0; i < totalSize; i++) {
    data[i] = generateSample(mtRandomFloat53);
  }
  return result;
}

// ============================================================================
// Chi-square family of distributions
// ============================================================================

/**
 * Draw samples from a chi-square distribution
 * @param df - Degrees of freedom (must be > 0)
 * @param size - Output shape
 */
export function chisquare(df: number, size?: number | number[]): ArrayStorage | number {
  if (df <= 0) {
    throw new Error('df must be positive');
  }
  // chi-square(df) = gamma(df/2, 2)
  return gamma(df / 2, 2, size);
}

/**
 * Draw samples from a noncentral chi-square distribution
 * @param df - Degrees of freedom (must be > 0)
 * @param nonc - Non-centrality parameter (must be >= 0)
 * @param size - Output shape
 */
export function noncentral_chisquare(
  df: number,
  nonc: number,
  size?: number | number[]
): ArrayStorage | number {
  if (df <= 0) {
    throw new Error('df must be positive');
  }
  if (nonc < 0) {
    throw new Error('nonc must be non-negative');
  }

  const generateSample = (rng: () => number): number => {
    // Use Poisson mixture representation
    if (nonc === 0) {
      return gammaSample(df / 2, 2, rng);
    }
    const i = poissonSample(nonc / 2, rng);
    return gammaSample(df / 2 + i, 2, rng);
  };

  if (size === undefined) {
    return generateSample(mtRandomFloat53);
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shapeArr, 'float64');
  const data = result.data as Float64Array;
  for (let i = 0; i < totalSize; i++) {
    data[i] = generateSample(mtRandomFloat53);
  }
  return result;
}

/**
 * Draw samples from an F distribution
 * @param dfnum - Degrees of freedom in numerator (must be > 0)
 * @param dfden - Degrees of freedom in denominator (must be > 0)
 * @param size - Output shape
 */
export function f(dfnum: number, dfden: number, size?: number | number[]): ArrayStorage | number {
  if (dfnum <= 0) {
    throw new Error('dfnum must be positive');
  }
  if (dfden <= 0) {
    throw new Error('dfden must be positive');
  }

  const generateSample = (rng: () => number): number => {
    const chi1 = gammaSample(dfnum / 2, 2, rng);
    const chi2 = gammaSample(dfden / 2, 2, rng);
    return chi1 / dfnum / (chi2 / dfden);
  };

  if (size === undefined) {
    return generateSample(mtRandomFloat53);
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shapeArr, 'float64');
  const data = result.data as Float64Array;
  for (let i = 0; i < totalSize; i++) {
    data[i] = generateSample(mtRandomFloat53);
  }
  return result;
}

/**
 * Draw samples from a noncentral F distribution
 * @param dfnum - Degrees of freedom in numerator (must be > 0)
 * @param dfden - Degrees of freedom in denominator (must be > 0)
 * @param nonc - Non-centrality parameter (must be >= 0)
 * @param size - Output shape
 */
export function noncentral_f(
  dfnum: number,
  dfden: number,
  nonc: number,
  size?: number | number[]
): ArrayStorage | number {
  if (dfnum <= 0) {
    throw new Error('dfnum must be positive');
  }
  if (dfden <= 0) {
    throw new Error('dfden must be positive');
  }
  if (nonc < 0) {
    throw new Error('nonc must be non-negative');
  }

  const generateSample = (rng: () => number): number => {
    // Noncentral F = (noncentral chi-square(dfnum, nonc) / dfnum) / (chi-square(dfden) / dfden)
    const nc_chi2 =
      nonc === 0
        ? gammaSample(dfnum / 2, 2, rng)
        : gammaSample(dfnum / 2 + poissonSample(nonc / 2, rng), 2, rng);
    const chi2 = gammaSample(dfden / 2, 2, rng);
    return nc_chi2 / dfnum / (chi2 / dfden);
  };

  if (size === undefined) {
    return generateSample(mtRandomFloat53);
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shapeArr, 'float64');
  const data = result.data as Float64Array;
  for (let i = 0; i < totalSize; i++) {
    data[i] = generateSample(mtRandomFloat53);
  }
  return result;
}

// ============================================================================
// Integer distributions
// ============================================================================

/**
 * Draw samples from a geometric distribution
 * @param p - Probability of success (0 < p <= 1)
 * @param size - Output shape
 */
export function geometric(p: number, size?: number | number[]): ArrayStorage | number {
  if (p <= 0 || p > 1) {
    throw new Error('p must be in (0, 1]');
  }

  const generateSample = (rng: () => number): number => {
    if (p === 1) return 1;
    const u = rng();
    return Math.floor(Math.log(u) / Math.log(1 - p)) + 1;
  };

  if (size === undefined) {
    return generateSample(mtRandomFloat53);
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shapeArr, 'int64');
  const data = result.data as BigInt64Array;
  for (let i = 0; i < totalSize; i++) {
    data[i] = BigInt(generateSample(mtRandomFloat53));
  }
  return result;
}

/**
 * Draw samples from a hypergeometric distribution
 * @param ngood - Number of good elements in population
 * @param nbad - Number of bad elements in population
 * @param nsample - Number of items to sample
 * @param size - Output shape
 */
export function hypergeometric(
  ngood: number,
  nbad: number,
  nsample: number,
  size?: number | number[]
): ArrayStorage | number {
  if (ngood < 0) throw new Error('ngood must be non-negative');
  if (nbad < 0) throw new Error('nbad must be non-negative');
  if (nsample < 0) throw new Error('nsample must be non-negative');
  if (nsample > ngood + nbad) throw new Error('nsample must be <= ngood + nbad');

  const generateSample = (rng: () => number): number => {
    // Simple simulation
    let good = ngood;
    let bad = nbad;
    let drawnGood = 0;
    let remaining = nsample;

    while (remaining > 0) {
      const total = good + bad;
      if (total === 0) break;
      const u = rng();
      if (u < good / total) {
        drawnGood++;
        good--;
      } else {
        bad--;
      }
      remaining--;
    }
    return drawnGood;
  };

  if (size === undefined) {
    return generateSample(mtRandomFloat53);
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shapeArr, 'int64');
  const data = result.data as BigInt64Array;
  for (let i = 0; i < totalSize; i++) {
    data[i] = BigInt(generateSample(mtRandomFloat53));
  }
  return result;
}

/**
 * Draw samples from a logarithmic series distribution
 * @param p - Shape parameter (0 < p < 1)
 * @param size - Output shape
 */
export function logseries(p: number, size?: number | number[]): ArrayStorage | number {
  if (p <= 0 || p >= 1) {
    throw new Error('p must be in (0, 1)');
  }

  const r = Math.log(1 - p);

  const generateSample = (rng: () => number): number => {
    // Kemp's algorithm
    const u = rng();
    const v = rng();
    const q = 1 - Math.exp(r * u);
    if (v >= q) return 1;
    if (v === 0) return 1; // Avoid log(0)
    const logV = Math.log(v);
    const logQ = Math.log(q);
    if (logV >= logQ) return 1;
    if (logV >= 2 * logQ) return 2;
    return Math.floor(1 + logV / logQ);
  };

  if (size === undefined) {
    return generateSample(mtRandomFloat53);
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shapeArr, 'int64');
  const data = result.data as BigInt64Array;
  for (let i = 0; i < totalSize; i++) {
    data[i] = BigInt(generateSample(mtRandomFloat53));
  }
  return result;
}

/**
 * Draw samples from a negative binomial distribution
 * @param n - Number of successes (must be > 0)
 * @param p - Probability of success (0 < p <= 1)
 * @param size - Output shape
 */
export function negative_binomial(
  n: number,
  p: number,
  size?: number | number[]
): ArrayStorage | number {
  if (n <= 0) {
    throw new Error('n must be positive');
  }
  if (p <= 0 || p > 1) {
    throw new Error('p must be in (0, 1]');
  }

  const generateSample = (rng: () => number): number => {
    // Use Poisson-gamma mixture: NegBin(n, p) = Poisson(Gamma(n, (1-p)/p))
    if (p === 1) return 0;
    const y = gammaSample(n, (1 - p) / p, rng);
    return poissonSample(y, rng);
  };

  if (size === undefined) {
    return generateSample(mtRandomFloat53);
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shapeArr, 'int64');
  const data = result.data as BigInt64Array;
  for (let i = 0; i < totalSize; i++) {
    data[i] = BigInt(generateSample(mtRandomFloat53));
  }
  return result;
}

/**
 * Draw samples from a Zipf distribution
 * @param a - Distribution parameter (must be > 1)
 * @param size - Output shape
 */
export function zipf(a: number, size?: number | number[]): ArrayStorage | number {
  if (a <= 1) {
    throw new Error('a must be > 1');
  }

  // Rejection method based on devroye
  const am1 = a - 1;
  const b = Math.pow(2, am1);

  const generateSample = (rng: () => number): number => {
    while (true) {
      const u = 1 - rng();
      const v = rng();
      const x = Math.floor(Math.pow(u, -1 / am1));
      const t = Math.pow(1 + 1 / x, am1);
      if ((v * x * (t - 1)) / (b - 1) <= t / b) {
        return x;
      }
    }
  };

  if (size === undefined) {
    return generateSample(mtRandomFloat53);
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shapeArr, 'int64');
  const data = result.data as BigInt64Array;
  for (let i = 0; i < totalSize; i++) {
    data[i] = BigInt(generateSample(mtRandomFloat53));
  }
  return result;
}

// ============================================================================
// Multivariate distributions
// ============================================================================

/**
 * Draw samples from a multinomial distribution
 * @param n - Number of experiments
 * @param pvals - Probabilities of each category (must sum to 1)
 * @param size - Output shape (number of experiments to run)
 */
export function multinomial(
  n: number,
  pvals: number[] | ArrayStorage,
  size?: number | number[]
): ArrayStorage {
  const probs = Array.isArray(pvals)
    ? pvals
    : Array.from({ length: pvals.size }, (_, i) => Number(pvals.iget(i)));

  const k = probs.length;
  if (k === 0) {
    throw new Error('pvals must have at least one element');
  }

  // Normalize probabilities
  const sum = probs.reduce((a, b) => a + b, 0);
  const normalizedProbs = probs.map((p) => p / sum);

  const generateSample = (rng: () => number): number[] => {
    const result = new Array(k).fill(0);
    let remaining = n;
    let pRemaining = 1.0;

    for (let i = 0; i < k - 1 && remaining > 0; i++) {
      const p = normalizedProbs[i]! / pRemaining;
      const x = binomialSample(remaining, Math.min(1, Math.max(0, p)), rng);
      result[i] = x;
      remaining -= x;
      pRemaining -= normalizedProbs[i]!;
    }
    result[k - 1] = remaining;
    return result;
  };

  if (size === undefined) {
    const sample = generateSample(mtRandomFloat53);
    const result = ArrayStorage.zeros([k], 'int64');
    const data = result.data as BigInt64Array;
    for (let i = 0; i < k; i++) {
      data[i] = BigInt(sample[i]!);
    }
    return result;
  }

  const shapeArr = Array.isArray(size) ? size : [size];
  const numSamples = shapeArr.reduce((a, b) => a * b, 1);
  const outShape = [...shapeArr, k];
  const result = ArrayStorage.zeros(outShape, 'int64');
  const data = result.data as BigInt64Array;

  for (let i = 0; i < numSamples; i++) {
    const sample = generateSample(mtRandomFloat53);
    for (let j = 0; j < k; j++) {
      data[i * k + j] = BigInt(sample[j]!);
    }
  }

  return result;
}

/**
 * Draw samples from a multivariate normal distribution
 * @param mean - Mean of the distribution (1-D array of length N)
 * @param cov - Covariance matrix (N x N array)
 * @param size - Number of samples to draw
 * @param check_valid - Check validity of covariance matrix (default 'warn')
 * @param tol - Tolerance for checking positive semi-definiteness
 */
export function multivariate_normal(
  mean: number[] | ArrayStorage,
  cov: number[][] | ArrayStorage,
  size?: number | number[],
  check_valid: 'warn' | 'raise' | 'ignore' = 'warn',
  tol: number = 1e-8
): ArrayStorage {
  const meanArr = Array.isArray(mean)
    ? mean
    : Array.from({ length: mean.size }, (_, i) => Number(mean.iget(i)));

  const n = meanArr.length;

  // Convert cov to 2D array
  let covArr: number[][];
  if (Array.isArray(cov)) {
    covArr = cov;
  } else {
    covArr = [];
    for (let i = 0; i < n; i++) {
      covArr.push([]);
      for (let j = 0; j < n; j++) {
        covArr[i]!.push(Number(cov.iget(i * n + j)));
      }
    }
  }

  // Simple Cholesky decomposition
  const L: number[][] = Array(n)
    .fill(0)
    .map(() => Array(n).fill(0) as number[]);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = covArr[i]![j]!;
      for (let k = 0; k < j; k++) {
        sum -= L[i]![k]! * L[j]![k]!;
      }
      if (i === j) {
        if (sum < -tol) {
          if (check_valid === 'raise') {
            throw new Error('covariance matrix is not positive semi-definite');
          } else if (check_valid === 'warn') {
            console.warn('covariance matrix is not positive semi-definite');
          }
          sum = 0;
        }
        L[i]![j] = Math.sqrt(Math.max(0, sum));
      } else {
        L[i]![j] = L[j]![j]! !== 0 ? sum / L[j]![j]! : 0;
      }
    }
  }

  const generateSample = (rng: () => number): number[] => {
    // Generate standard normal samples
    const z: number[] = [];
    for (let i = 0; i < n; i++) {
      z.push(boxMullerTransform(rng));
    }

    // Transform: x = mean + L * z
    const result: number[] = [];
    for (let i = 0; i < n; i++) {
      let val = meanArr[i]!;
      for (let j = 0; j <= i; j++) {
        val += L[i]![j]! * z[j]!;
      }
      result.push(val);
    }
    return result;
  };

  if (size === undefined) {
    const sample = generateSample(mtRandomFloat53);
    const result = ArrayStorage.zeros([n], 'float64');
    const data = result.data as Float64Array;
    for (let i = 0; i < n; i++) {
      data[i] = sample[i]!;
    }
    return result;
  }

  const shapeArr = Array.isArray(size) ? size : [size];
  const numSamples = shapeArr.reduce((a, b) => a * b, 1);
  const outShape = [...shapeArr, n];
  const result = ArrayStorage.zeros(outShape, 'float64');
  const data = result.data as Float64Array;

  for (let i = 0; i < numSamples; i++) {
    const sample = generateSample(mtRandomFloat53);
    for (let j = 0; j < n; j++) {
      data[i * n + j] = sample[j]!;
    }
  }

  return result;
}

/**
 * Draw samples from a Dirichlet distribution
 * @param alpha - Concentration parameters (must all be > 0)
 * @param size - Number of samples to draw
 */
export function dirichlet(alpha: number[] | ArrayStorage, size?: number | number[]): ArrayStorage {
  const alphaArr = Array.isArray(alpha)
    ? alpha
    : Array.from({ length: alpha.size }, (_, i) => Number(alpha.iget(i)));

  const k = alphaArr.length;
  if (k < 2) {
    throw new Error('alpha must have at least 2 elements');
  }

  for (const a of alphaArr) {
    if (a <= 0) {
      throw new Error('all alpha values must be positive');
    }
  }

  const generateSample = (rng: () => number): number[] => {
    // Generate gamma samples and normalize
    const gammas: number[] = [];
    let sum = 0;
    for (let i = 0; i < k; i++) {
      const g = gammaSample(alphaArr[i]!, 1, rng);
      gammas.push(g);
      sum += g;
    }
    return gammas.map((g) => g / sum);
  };

  if (size === undefined) {
    const sample = generateSample(mtRandomFloat53);
    const result = ArrayStorage.zeros([k], 'float64');
    const data = result.data as Float64Array;
    for (let i = 0; i < k; i++) {
      data[i] = sample[i]!;
    }
    return result;
  }

  const shapeArr = Array.isArray(size) ? size : [size];
  const numSamples = shapeArr.reduce((a, b) => a * b, 1);
  const outShape = [...shapeArr, k];
  const result = ArrayStorage.zeros(outShape, 'float64');
  const data = result.data as Float64Array;

  for (let i = 0; i < numSamples; i++) {
    const sample = generateSample(mtRandomFloat53);
    for (let j = 0; j < k; j++) {
      data[i * k + j] = sample[j]!;
    }
  }

  return result;
}

/**
 * Draw samples from a von Mises distribution
 * @param mu - Mode (center) of the distribution (in radians)
 * @param kappa - Concentration parameter (must be >= 0)
 * @param size - Output shape
 */
export function vonmises(
  mu: number,
  kappa: number,
  size?: number | number[]
): ArrayStorage | number {
  if (kappa < 0) {
    throw new Error('kappa must be non-negative');
  }

  const generateSample = (rng: () => number): number => {
    if (kappa === 0) {
      // Uniform on circle
      return 2 * Math.PI * rng() - Math.PI;
    }

    // Best-Fisher algorithm for von Mises
    const a = 1 + Math.sqrt(1 + 4 * kappa * kappa);
    const b = (a - Math.sqrt(2 * a)) / (2 * kappa);
    const r = (1 + b * b) / (2 * b);

    while (true) {
      const u1 = rng();
      const z = Math.cos(Math.PI * u1);
      const f = (1 + r * z) / (r + z);
      const c = kappa * (r - f);
      const u2 = rng();

      if (c * (2 - c) > u2 || Math.log(c / u2) + 1 - c >= 0) {
        const u3 = rng();
        const theta = u3 > 0.5 ? Math.acos(f) : -Math.acos(f);
        return ((theta + mu + Math.PI) % (2 * Math.PI)) - Math.PI;
      }
    }
  };

  if (size === undefined) {
    return generateSample(mtRandomFloat53);
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shapeArr, 'float64');
  const data = result.data as Float64Array;
  for (let i = 0; i < totalSize; i++) {
    data[i] = generateSample(mtRandomFloat53);
  }
  return result;
}
