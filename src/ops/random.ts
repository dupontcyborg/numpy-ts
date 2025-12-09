/**
 * Random number generation module
 *
 * Implements NumPy-compatible random functions using a seedable PRNG.
 * Uses a PCG (Permuted Congruential Generator) algorithm for high-quality
 * random number generation with seedability.
 */

import { ArrayStorage } from '../core/storage';
import { type DType, isBigIntDType } from '../core/dtype';

/**
 * Internal state for the random number generator
 * Using PCG32 algorithm (Permuted Congruential Generator)
 */
interface RandomState {
  state: bigint;
  inc: bigint;
}

// Global RNG state
let _rngState: RandomState = {
  state: BigInt('0x853c49e6748fea9b'),
  inc: BigInt('0xda3e39cb94b95bdb'),
};

// PCG32 constants
const PCG_MULT = BigInt('6364136223846793005');
const MASK_32 = BigInt(0xffffffff);
const MASK_64 = BigInt('0xffffffffffffffff');

/**
 * PCG32 random number generator step
 * Returns a 32-bit unsigned integer
 */
function pcg32(): number {
  const oldState = _rngState.state;

  // Advance internal state
  _rngState.state = (oldState * PCG_MULT + _rngState.inc) & MASK_64;

  // Calculate output function (XSH RR)
  const xorShifted = Number((((oldState >> BigInt(18)) ^ oldState) >> BigInt(27)) & MASK_32);
  const rot = Number(oldState >> BigInt(59));
  const negRot = -rot & 31;
  return ((xorShifted >>> rot) | (xorShifted << negRot)) >>> 0;
}

/**
 * Generate a random float in [0, 1)
 */
function randomFloat(): number {
  return pcg32() / 4294967296; // 2^32
}

/**
 * Generate a random float using higher precision (53 bits)
 * Uses two 32-bit numbers to get better precision for doubles
 */
function randomFloat53(): number {
  const a = pcg32() >>> 5; // 27 bits
  const b = pcg32() >>> 6; // 26 bits
  return (a * 67108864 + b) / 9007199254740992; // 2^53
}

/**
 * Seed the random number generator
 * @param seedValue - Seed value (integer)
 */
export function seed(seedValue?: number | null): void {
  if (seedValue === undefined || seedValue === null) {
    // Use current time and some entropy for non-deterministic seeding
    seedValue = Date.now() ^ (Math.random() * 0x100000000);
  }

  // Initialize state using the seed
  const s = BigInt(seedValue >>> 0);
  _rngState.state = BigInt(0);
  _rngState.inc = (s << BigInt(1)) | BigInt(1);

  // Warm up the generator
  pcg32();
  _rngState.state = _rngState.state + s;
  pcg32();
}

/**
 * Get the internal state of the random number generator
 * @returns State object that can be used with set_state
 */
export function get_state(): { state: string; inc: string } {
  return {
    state: _rngState.state.toString(),
    inc: _rngState.inc.toString(),
  };
}

/**
 * Set the internal state of the random number generator
 * @param state - State object from get_state
 */
export function set_state(state: { state: string; inc: string }): void {
  _rngState.state = BigInt(state.state);
  _rngState.inc = BigInt(state.inc);
}

/**
 * Random number generator class (similar to NumPy's Generator)
 */
export class Generator {
  private _state: RandomState;

  constructor(seedValue?: number) {
    if (seedValue !== undefined) {
      const s = BigInt(seedValue >>> 0);
      this._state = {
        state: BigInt(0),
        inc: (s << BigInt(1)) | BigInt(1),
      };
      this._pcg32();
      this._state.state = this._state.state + s;
      this._pcg32();
    } else {
      // Copy global state
      this._state = { ..._rngState };
    }
  }

  private _pcg32(): number {
    const oldState = this._state.state;
    this._state.state = (oldState * PCG_MULT + this._state.inc) & MASK_64;
    const xorShifted = Number((((oldState >> BigInt(18)) ^ oldState) >> BigInt(27)) & MASK_32);
    const rot = Number(oldState >> BigInt(59));
    return ((xorShifted >>> rot) | (xorShifted << (-rot & 31))) >>> 0;
  }

  private _randomFloat(): number {
    return this._pcg32() / 4294967296;
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
 * Create a new Generator instance (similar to np.random.default_rng)
 * @param seedValue - Optional seed value
 */
export function default_rng(seedValue?: number): Generator {
  return new Generator(seedValue);
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
    // Direct method for small lambda
    const L = Math.exp(-lam);
    let k = 0;
    let p = 1;
    do {
      k++;
      p *= rng();
    } while (p > L);
    return k - 1;
  } else {
    // Use normal approximation for large lambda
    const z = boxMullerTransform(rng);
    return Math.max(0, Math.round(lam + Math.sqrt(lam) * z));
  }
}

/**
 * Binomial random sample
 */
function binomialSample(n: number, p: number, rng: () => number): number {
  if (n * p < 10 && n * (1 - p) < 10) {
    // Direct method for small n*p
    let successes = 0;
    for (let i = 0; i < n; i++) {
      if (rng() < p) successes++;
    }
    return successes;
  } else {
    // Normal approximation for larger n*p
    const mean = n * p;
    const std = Math.sqrt(n * p * (1 - p));
    const z = boxMullerTransform(rng);
    return Math.max(0, Math.min(n, Math.round(mean + std * z)));
  }
}

// ============================================================================
// Top-level functions that use the global RNG state
// ============================================================================

/**
 * Generate random floats in the half-open interval [0.0, 1.0)
 * @param size - Output shape. If not provided, returns a single float.
 */
export function random(size?: number | number[]): ArrayStorage | number {
  if (size === undefined) {
    return randomFloat53();
  }
  const shape = Array.isArray(size) ? size : [size];
  const totalSize = shape.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shape, 'float64');
  const data = result.data as Float64Array;
  for (let i = 0; i < totalSize; i++) {
    data[i] = randomFloat53();
  }
  return result;
}

/**
 * Random values in a given shape (alias for random with shape)
 * @param d0, d1, ..., dn - The dimensions of the returned array
 */
export function rand(...shape: number[]): ArrayStorage | number {
  if (shape.length === 0) {
    return randomFloat53();
  }
  const totalSize = shape.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shape, 'float64');
  const data = result.data as Float64Array;
  for (let i = 0; i < totalSize; i++) {
    data[i] = randomFloat53();
  }
  return result;
}

/**
 * Return random floats from standard normal distribution
 * @param d0, d1, ..., dn - The dimensions of the returned array
 */
export function randn(...shape: number[]): ArrayStorage | number {
  if (shape.length === 0) {
    return boxMullerTransform(randomFloat);
  }
  const totalSize = shape.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shape, 'float64');
  const data = result.data as Float64Array;
  for (let i = 0; i < totalSize; i += 2) {
    const [z0, z1] = boxMullerTransformPair(randomFloat);
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
    return Math.floor(randomFloat() * range) + low;
  }

  const shape = Array.isArray(size) ? size : [size];
  const totalSize = shape.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shape, dtype);
  const data = result.data;

  if (isBigIntDType(dtype)) {
    const bigData = data as BigInt64Array | BigUint64Array;
    for (let i = 0; i < totalSize; i++) {
      bigData[i] = BigInt(Math.floor(randomFloat() * range) + low);
    }
  } else {
    const numData = data as Exclude<typeof data, BigInt64Array | BigUint64Array>;
    for (let i = 0; i < totalSize; i++) {
      numData[i] = Math.floor(randomFloat() * range) + low;
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
    return randomFloat53() * (high - low) + low;
  }
  const shape = Array.isArray(size) ? size : [size];
  const totalSize = shape.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shape, 'float64');
  const data = result.data as Float64Array;
  const range = high - low;
  for (let i = 0; i < totalSize; i++) {
    data[i] = randomFloat53() * range + low;
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
    return boxMullerTransform(randomFloat) * scale + loc;
  }
  const shape = Array.isArray(size) ? size : [size];
  const totalSize = shape.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shape, 'float64');
  const data = result.data as Float64Array;
  for (let i = 0; i < totalSize; i += 2) {
    const [z0, z1] = boxMullerTransformPair(randomFloat);
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
    return boxMullerTransform(randomFloat);
  }
  const shape = Array.isArray(size) ? size : [size];
  const totalSize = shape.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shape, 'float64');
  const data = result.data as Float64Array;
  for (let i = 0; i < totalSize; i += 2) {
    const [z0, z1] = boxMullerTransformPair(randomFloat);
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
    return -Math.log(1 - randomFloat53()) * scale;
  }
  const shape = Array.isArray(size) ? size : [size];
  const totalSize = shape.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shape, 'float64');
  const data = result.data as Float64Array;
  for (let i = 0; i < totalSize; i++) {
    data[i] = -Math.log(1 - randomFloat53()) * scale;
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
    return poissonSample(lam, randomFloat);
  }
  const shape = Array.isArray(size) ? size : [size];
  const totalSize = shape.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shape, 'int64');
  const data = result.data as BigInt64Array;
  for (let i = 0; i < totalSize; i++) {
    data[i] = BigInt(poissonSample(lam, randomFloat));
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
    return binomialSample(n, p, randomFloat);
  }
  const shape = Array.isArray(size) ? size : [size];
  const totalSize = shape.reduce((a, b) => a * b, 1);
  const result = ArrayStorage.zeros(shape, 'int64');
  const data = result.data as BigInt64Array;
  for (let i = 0; i < totalSize; i++) {
    data[i] = BigInt(binomialSample(n, p, randomFloat));
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
  rng: () => number = randomFloat
): ArrayStorage | number {
  // Determine the population
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

  // Handle probabilities
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
    // Normalize probabilities
    const sum = probabilities.reduce((a, b) => a + b, 0);
    if (Math.abs(sum - 1) > 1e-10) {
      probabilities = probabilities.map((x) => x / sum);
    }
  }

  // Single sample
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

  // Multiple samples
  const shape = Array.isArray(size) ? size : [size];
  const totalSize = shape.reduce((a, b) => a * b, 1);

  if (!replace && totalSize > n) {
    throw new Error('cannot take a larger sample than population when replace=false');
  }

  const result = ArrayStorage.zeros(shape, 'float64');
  const data = result.data as Float64Array;

  if (replace) {
    if (probabilities) {
      // Compute cumulative probabilities for efficient sampling
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
    // Without replacement - Fisher-Yates shuffle approach
    const available = [...population];
    const availableProbs = probabilities ? [...probabilities] : undefined;

    for (let i = 0; i < totalSize; i++) {
      let idx: number;
      if (availableProbs) {
        // Normalize remaining probabilities
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
  return choiceImpl(a, size, replace, p, randomFloat);
}

/**
 * Implementation of permutation
 */
function permutationImpl(x: number | ArrayStorage, rng: () => number = randomFloat): ArrayStorage {
  let arr: ArrayStorage;
  if (typeof x === 'number') {
    // Create arange(x) and permute it
    const data = new Float64Array(x);
    for (let i = 0; i < x; i++) {
      data[i] = i;
    }
    arr = ArrayStorage.fromData(data, [x], 'float64');
  } else {
    // Copy the array
    arr = x.copy();
  }

  // Fisher-Yates shuffle
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
  return permutationImpl(x, randomFloat);
}

/**
 * Implementation of shuffle (in-place)
 */
function shuffleImpl(x: ArrayStorage, rng: () => number = randomFloat): void {
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
  shuffleImpl(x, randomFloat);
}
