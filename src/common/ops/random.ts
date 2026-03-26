/**
 * Random number generation module
 *
 * Implements NumPy-compatible random functions:
 * - Legacy API (np.random.seed, random, rand, uniform, etc.) uses MT19937
 * - Modern API (np.random.default_rng) uses PCG64 with SeedSequence
 *
 * Core RNG and distribution algorithms run in WASM (Zig) for:
 * - Exact NumPy output matching (Ziggurat for Generator, polar for legacy)
 * - Native u128 for PCG64 (vs slow JS BigInt)
 */

import { ArrayStorage } from '../storage';
import { svd } from '../ops/linalg';
import { type DType, isBigIntDType } from '../dtype';
import {
  initMT19937,
  mt19937Uint32,
  mt19937Float64,
  getMT19937State,
  setMT19937State,
  initPCG64FromSeed,
  pcg64Float64,
  pcg64SaveState,
  pcg64RestoreState,
  pcg64BoundedUint64,
  standardNormalPCG,
  standardExponentialPCG,
  legacyGauss,
  legacyGaussReset,
  legacyStandardExponential,
  fillUniformF64MT,
  fillUniformF64PCG,
  fillStandardNormalPCG,
  fillStandardExponentialPCG,
  fillLegacyGauss,
  fillLegacyStandardExponential,
  fillRkInterval,
  wasmLegacyStandardGamma,
  fillLegacyStandardGamma,
  fillLegacyChisquare,
  fillPareto,
  fillPower,
  fillWeibull,
  fillLogistic,
  fillGumbel,
  fillLaplace,
  fillRayleigh,
  fillTriangular,
  fillStandardCauchy,
  fillLognormal,
  fillWald,
  fillStandardT,
  fillBeta,
  fillF,
  fillNoncentralChisquare,
  fillNoncentralF,
  fillGeometric,
  fillPoisson,
  fillBinomial,
  fillNegativeBinomial,
  fillHypergeometric,
  fillLogseries,
  fillZipf,
  fillVonmises,
  fillRandintI64,
  fillRandintU8,
  fillRandintU16,
  fillPermutation,
  fillPermutationPCG,
  fillBoundedUint64PCG,
} from '../wasm/rng';

// ============================================================================
// Generator Class - Modern API using PCG64 + Ziggurat (via WASM)
// ============================================================================

/**
 * Random number generator class (matches NumPy's Generator from default_rng).
 * Each instance maintains its own PCG64 state snapshot, swapped into WASM on each call.
 */
export class Generator {
  private _state: BigUint64Array;

  constructor(seedValue?: number) {
    const s = seedValue !== undefined ? seedValue : Math.floor(Math.random() * 0x100000000);
    initPCG64FromSeed(s);
    this._state = pcg64SaveState();
  }

  /** Load this instance's state into WASM, run fn, snapshot state back. */
  private _withState<T>(fn: () => T): T {
    pcg64RestoreState(this._state);
    const result = fn();
    this._state = pcg64SaveState();
    return result;
  }

  random(size?: number | number[]): ArrayStorage | number {
    return this._withState(() => {
      if (size === undefined) {
        return pcg64Float64();
      }
      const shape = Array.isArray(size) ? size : [size];
      const totalSize = shape.reduce((a, b) => a * b, 1);
      const data = fillUniformF64PCG(totalSize);
      return ArrayStorage.fromData(new Float64Array(data), shape, 'float64');
    });
  }

  integers(low: number, high?: number, size?: number | number[]): ArrayStorage | number {
    return this._withState(() => {
      if (high === undefined) {
        high = low;
        low = 0;
      }
      const rng = high - low - 1; // inclusive max offset
      if (size === undefined) {
        return Number(pcg64BoundedUint64(low, rng));
      }
      const shape = Array.isArray(size) ? size : [size];
      const totalSize = shape.reduce((a, b) => a * b, 1);
      const data = fillBoundedUint64PCG(totalSize, low, rng);
      return ArrayStorage.fromData(new BigInt64Array(data), shape, 'int64');
    });
  }

  standard_normal(size?: number | number[]): ArrayStorage | number {
    return this._withState(() => {
      if (size === undefined) {
        return standardNormalPCG();
      }
      const shape = Array.isArray(size) ? size : [size];
      const totalSize = shape.reduce((a, b) => a * b, 1);
      const data = fillStandardNormalPCG(totalSize);
      return ArrayStorage.fromData(new Float64Array(data), shape, 'float64');
    });
  }

  normal(loc: number = 0, scale: number = 1, size?: number | number[]): ArrayStorage | number {
    return this._withState(() => {
      if (size === undefined) {
        return standardNormalPCG() * scale + loc;
      }
      const shape = Array.isArray(size) ? size : [size];
      const totalSize = shape.reduce((a, b) => a * b, 1);
      const rawData = fillStandardNormalPCG(totalSize);
      const data = new Float64Array(totalSize);
      for (let i = 0; i < totalSize; i++) {
        data[i] = rawData[i]! * scale + loc;
      }
      return ArrayStorage.fromData(data, shape, 'float64');
    });
  }

  uniform(low: number = 0, high: number = 1, size?: number | number[]): ArrayStorage | number {
    return this._withState(() => {
      if (size === undefined) {
        return pcg64Float64() * (high - low) + low;
      }
      const shape = Array.isArray(size) ? size : [size];
      const totalSize = shape.reduce((a, b) => a * b, 1);
      const rawData = fillUniformF64PCG(totalSize);
      const data = new Float64Array(totalSize);
      const range = high - low;
      for (let i = 0; i < totalSize; i++) {
        data[i] = rawData[i]! * range + low;
      }
      return ArrayStorage.fromData(data, shape, 'float64');
    });
  }

  choice(
    a: number | ArrayStorage,
    size?: number | number[],
    replace: boolean = true,
    p?: ArrayStorage | number[]
  ): ArrayStorage | number {
    return this._withState(() => choiceImpl(a, size, replace, p, pcg64Float64, true));
  }

  permutation(x: number | ArrayStorage): ArrayStorage {
    return this._withState(() => permutationImpl(x, pcg64Float64, true));
  }

  shuffle(x: ArrayStorage): void {
    this._withState(() => {
      shuffleImpl(x, pcg64Float64, true);
    });
  }

  exponential(scale: number = 1, size?: number | number[]): ArrayStorage | number {
    return this._withState(() => {
      if (size === undefined) {
        return standardExponentialPCG() * scale;
      }
      const shape = Array.isArray(size) ? size : [size];
      const totalSize = shape.reduce((a, b) => a * b, 1);
      const rawData = fillStandardExponentialPCG(totalSize);
      const data = new Float64Array(totalSize);
      for (let i = 0; i < totalSize; i++) {
        data[i] = rawData[i]! * scale;
      }
      return ArrayStorage.fromData(data, shape, 'float64');
    });
  }

  poisson(lam: number = 1, size?: number | number[]): ArrayStorage | number {
    return this._withState(() => {
      if (size === undefined) {
        return poissonSample(lam, pcg64Float64);
      }
      const shape = Array.isArray(size) ? size : [size];
      const totalSize = shape.reduce((a, b) => a * b, 1);
      const result = ArrayStorage.zeros(shape, 'int64');
      const data = result.data as BigInt64Array;
      for (let i = 0; i < totalSize; i++) {
        data[i] = BigInt(poissonSample(lam, pcg64Float64));
      }
      return result;
    });
  }

  binomial(n: number, p: number, size?: number | number[]): ArrayStorage | number {
    return this._withState(() => {
      if (size === undefined) {
        return binomialSample(n, p, pcg64Float64);
      }
      const shape = Array.isArray(size) ? size : [size];
      const totalSize = shape.reduce((a, b) => a * b, 1);
      const result = ArrayStorage.zeros(shape, 'int64');
      const data = result.data as BigInt64Array;
      for (let i = 0; i < totalSize; i++) {
        data[i] = BigInt(binomialSample(n, p, pcg64Float64));
      }
      return result;
    });
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
  initMT19937(seedValue >>> 0);
  legacyGaussReset();
}

/**
 * Get the internal state of the random number generator
 * @returns State object that can be used with set_state
 */
export function get_state(): { mt: number[]; mti: number } {
  const { mt, mti } = getMT19937State();
  return { mt: Array.from(mt), mti };
}

/**
 * Set the internal state of the random number generator
 * @param state - State object from get_state
 */
export function set_state(state: { mt: number[]; mti: number }): void {
  setMT19937State(new Uint32Array(state.mt), state.mti);
  legacyGaussReset();
}

// ============================================================================
// Helper functions for random distributions
// ============================================================================

/**
 * NumPy's rk_interval: bounded random integer in [0, max] using rejection sampling
 * with bit masking. Matches randomkit.c line 449.
 * Uses mt19937Uint32 (rk_random) masked to smallest bitmask >= max, rejecting > max.
 */
function rkInterval(max: number): number {
  if (max === 0) return 0;

  // Smallest bit mask >= max
  let mask = max;
  mask |= mask >>> 1;
  mask |= mask >>> 2;
  mask |= mask >>> 4;
  mask |= mask >>> 8;
  mask |= mask >>> 16;

  // Rejection sampling: generate masked u32, reject if > max
  let value: number;
  do {
    value = (mt19937Uint32() & mask) >>> 0;
  } while (value > max);

  return value;
}

// ============================================================================
// NumPy-exact legacy distribution helpers
// ============================================================================

/**
 * log-gamma function matching NumPy's random_loggam (distributions.c)
 */
function randomLoggam(x: number): number {
  const a = [
    8.333333333333333e-2, -2.777777777777778e-3, 7.936507936507937e-4, -5.952380952380952e-4,
    8.417508417508418e-4, -1.917526917526918e-3, 6.41025641025641e-3, -2.955065359477124e-2,
    1.796443723688307e-1, -1.3924322169059,
  ];

  if (x === 1.0 || x === 2.0) return 0.0;

  let n: number;
  if (x < 7.0) {
    n = Math.floor(7 - x);
  } else {
    n = 0;
  }
  let x0 = x + n;
  const x2 = (1.0 / x0) * (1.0 / x0);
  const lg2pi = 1.8378770664093453;
  let gl0 = a[9]!;
  for (let k = 8; k >= 0; k--) {
    gl0 *= x2;
    gl0 += a[k]!;
  }
  let gl = gl0 / x0 + 0.5 * lg2pi + (x0 - 0.5) * Math.log(x0) - x0;
  if (x < 7.0) {
    for (let k = 1; k <= n; k++) {
      gl -= Math.log(x0 - 1.0);
      x0 -= 1.0;
    }
  }
  return gl;
}

/**
 * Legacy standard gamma — exact match for NumPy's legacy_standard_gamma
 * Uses legacyGauss(), legacyStandardExponential(), mt19937Float64()
 */
function legacyStandardGamma(shape: number): number {
  return wasmLegacyStandardGamma(shape);
}

/**
 * Legacy chisquare: 2 * legacyStandardGamma(df/2)
 */
function legacyChisquare(df: number): number {
  return 2.0 * legacyStandardGamma(df / 2.0);
}

// ============================================================================
// Hypergeometric: NumPy-exact (HYP + HRUA) from legacy-distributions.c
// ============================================================================

function hypergeometricHyp(good: number, bad: number, sample: number): number {
  const d1 = bad + good - sample;
  let d2 = Math.min(bad, good);
  let y = d2;
  let k = sample;
  while (y > 0.0) {
    const u = mt19937Float64();
    y -= Math.floor(u + y / (d1 + k));
    k--;
    if (k === 0) break;
  }
  const z = d2 - y;
  if (good > bad) return sample - z;
  return z;
}

function hypergeometricHrua(good: number, bad: number, sample: number): number {
  const D1 = 1.7155277699214135;
  const D2 = 0.8989161620588988;

  const mingoodbad = Math.min(good, bad);
  const popsize = good + bad;
  const maxgoodbad = Math.max(good, bad);
  const m = Math.min(sample, popsize - sample);
  const d4 = mingoodbad / popsize;
  const d5 = 1.0 - d4;
  const d6 = m * d4 + 0.5;
  const d7 = Math.sqrt(((popsize - m) * sample * d4 * d5) / (popsize - 1) + 0.5);
  const d8 = D1 * d7 + D2;
  const d9 = Math.floor(((m + 1) * (mingoodbad + 1)) / (popsize + 2));
  const d10 =
    randomLoggam(d9 + 1) +
    randomLoggam(mingoodbad - d9 + 1) +
    randomLoggam(m - d9 + 1) +
    randomLoggam(maxgoodbad - m + d9 + 1);
  const d11 = Math.min(Math.min(m, mingoodbad) + 1.0, Math.floor(d6 + 16 * d7));

  let Z: number;
  while (true) {
    const X = mt19937Float64();
    const Y = mt19937Float64();
    const W = d6 + (d8 * (Y - 0.5)) / X;

    if (W < 0.0 || W >= d11) continue;

    Z = Math.floor(W);
    const T =
      d10 -
      (randomLoggam(Z + 1) +
        randomLoggam(mingoodbad - Z + 1) +
        randomLoggam(m - Z + 1) +
        randomLoggam(maxgoodbad - m + Z + 1));

    if (X * (4.0 - X) - 3.0 <= T) break;
    if (X * (X - T) >= 1) continue;
    if (2.0 * Math.log(X) <= T) break;
  }

  if (good > bad) Z = m - Z;
  if (m < sample) Z = good - Z;
  return Z;
}

// ============================================================================
// Poisson: NumPy-exact (random_poisson_mult + random_poisson_ptrs)
// ============================================================================

function poissonMult(lam: number, rng: () => number): number {
  const enlam = Math.exp(-lam);
  let X = 0;
  let prod = 1.0;
  while (true) {
    const U = rng();
    prod *= U;
    if (prod > enlam) {
      X += 1;
    } else {
      return X;
    }
  }
}

function poissonPtrs(lam: number, rng: () => number): number {
  const slam = Math.sqrt(lam);
  const loglam = Math.log(lam);
  const b = 0.931 + 2.53 * slam;
  const a = -0.059 + 0.02483 * b;
  const invalpha = 1.1239 + 1.1328 / (b - 3.4);
  const vr = 0.9277 - 3.6224 / (b - 2);

  while (true) {
    const U = rng() - 0.5;
    const V = rng();
    const us = 0.5 - Math.abs(U);
    const k = Math.floor(((2 * a) / us + b) * U + lam + 0.43);
    if (us >= 0.07 && V <= vr) return k;
    if (k < 0 || (us < 0.013 && V > us)) continue;
    if (
      Math.log(V) + Math.log(invalpha) - Math.log(a / (us * us) + b) <=
      -lam + k * loglam - randomLoggam(k + 1)
    ) {
      return k;
    }
  }
}

function poissonSample(lam: number, rng: () => number): number {
  if (lam >= 10) return poissonPtrs(lam, rng);
  else if (lam === 0) return 0;
  else return poissonMult(lam, rng);
}

// ============================================================================
// Binomial: NumPy-exact (legacy_random_binomial_inversion + random_binomial_btpe)
// ============================================================================

/** Cached binomial state for BTPE/inversion */
interface BinomialState {
  has_binomial: boolean;
  nsave: number;
  psave: number;
  r: number;
  q: number;
  fm: number;
  m: number;
  p1: number;
  xm: number;
  xl: number;
  xr: number;
  c: number;
  laml: number;
  lamr: number;
  p2: number;
  p3: number;
  p4: number;
}

function makeBinomialState(): BinomialState {
  return {
    has_binomial: false,
    nsave: 0,
    psave: 0,
    r: 0,
    q: 0,
    fm: 0,
    m: 0,
    p1: 0,
    xm: 0,
    xl: 0,
    xr: 0,
    c: 0,
    laml: 0,
    lamr: 0,
    p2: 0,
    p3: 0,
    p4: 0,
  };
}

function binomialBtpe(n: number, p: number, rng: () => number, binomial: BinomialState): number {
  let r: number, q: number, fm: number, p1: number, xm: number, xl: number, xr: number;
  let c: number, laml: number, lamr: number, p2: number, p3: number, p4: number;
  let a: number, u: number, v: number, s: number, F: number, rho: number, t: number, A: number;
  let nrq: number, x1: number, x2: number, f1: number, f2: number;
  let z: number, z2: number, w: number, w2: number, x: number;
  let m: number, y: number, k: number;

  if (!binomial.has_binomial || binomial.nsave !== n || binomial.psave !== p) {
    binomial.nsave = n;
    binomial.psave = p;
    binomial.has_binomial = true;
    binomial.r = r = Math.min(p, 1.0 - p);
    binomial.q = q = 1.0 - r;
    binomial.fm = fm = n * r + r;
    binomial.m = m = Math.floor(fm);
    binomial.p1 = p1 = Math.floor(2.195 * Math.sqrt(n * r * q) - 4.6 * q) + 0.5;
    binomial.xm = xm = m + 0.5;
    binomial.xl = xl = xm - p1;
    binomial.xr = xr = xm + p1;
    binomial.c = c = 0.134 + 20.5 / (15.3 + m);
    a = (fm - xl) / (fm - xl * r);
    binomial.laml = laml = a * (1.0 + a / 2.0);
    a = (xr - fm) / (xr * q);
    binomial.lamr = lamr = a * (1.0 + a / 2.0);
    binomial.p2 = p2 = p1 * (1.0 + 2.0 * c);
    binomial.p3 = p3 = p2 + c / laml;
    binomial.p4 = p4 = p3 + c / lamr;
  } else {
    r = binomial.r;
    q = binomial.q;
    m = binomial.m;
    p1 = binomial.p1;
    xm = binomial.xm;
    xl = binomial.xl;
    xr = binomial.xr;
    c = binomial.c;
    laml = binomial.laml;
    lamr = binomial.lamr;
    p2 = binomial.p2;
    p3 = binomial.p3;
    p4 = binomial.p4;
  }

  // Main loop (gotos replaced with labeled loop + continue)
  outer: while (true) {
    // Step10
    nrq = n * r * q;
    u = rng() * p4;
    v = rng();

    if (u <= p1) {
      y = Math.floor(xm - p1 * v + u);
      // goto Step60
    } else if (u <= p2) {
      // Step20
      x = xl + (u - p1) / c;
      v = v * c + 1.0 - Math.abs(m - x + 0.5) / p1;
      if (v > 1.0) continue outer;
      y = Math.floor(x);
      // goto Step50
      k = Math.abs(y - m);
      if (k > 20 && k < nrq / 2.0 - 1) {
        // Step52
        rho = (k / nrq) * ((k * (k / 3.0 + 0.625) + 0.16666666666666666) / nrq + 0.5);
        t = (-k * k) / (2 * nrq);
        A = Math.log(v);
        if (A < t - rho) {
          /* accept */
        } else if (A > t + rho) continue outer;
        else {
          x1 = y + 1;
          f1 = m + 1;
          z = n + 1 - m;
          w = n - y + 1;
          x2 = x1 * x1;
          f2 = f1 * f1;
          z2 = z * z;
          w2 = w * w;
          if (
            A >
            xm * Math.log(f1 / x1) +
              (n - m + 0.5) * Math.log(z / w) +
              (y - m) * Math.log((w * r) / (x1 * q)) +
              (13680 - (462 - (132 - (99 - 140 / f2) / f2) / f2) / f2) / f1 / 166320 +
              (13680 - (462 - (132 - (99 - 140 / z2) / z2) / z2) / z2) / z / 166320 +
              (13680 - (462 - (132 - (99 - 140 / x2) / x2) / x2) / x2) / x1 / 166320 +
              (13680 - (462 - (132 - (99 - 140 / w2) / w2) / w2) / w2) / w / 166320
          ) {
            continue outer;
          }
        }
      } else {
        s = r / q;
        a = s * (n + 1);
        F = 1.0;
        if (m < y) {
          for (let i = m + 1; i <= y; i++) F *= a / i - s;
        } else if (m > y) {
          for (let i = y + 1; i <= m; i++) F /= a / i - s;
        }
        if (v > F) continue outer;
      }
    } else if (u <= p3) {
      // Step30
      y = Math.floor(xl + Math.log(v) / laml);
      if (y < 0 || v === 0.0) continue outer;
      v = v * (u - p2) * laml;
      // goto Step50
      k = Math.abs(y - m);
      if (k > 20 && k < nrq / 2.0 - 1) {
        rho = (k / nrq) * ((k * (k / 3.0 + 0.625) + 0.16666666666666666) / nrq + 0.5);
        t = (-k * k) / (2 * nrq);
        A = Math.log(v);
        if (A < t - rho) {
          /* accept */
        } else if (A > t + rho) continue outer;
        else {
          x1 = y + 1;
          f1 = m + 1;
          z = n + 1 - m;
          w = n - y + 1;
          x2 = x1 * x1;
          f2 = f1 * f1;
          z2 = z * z;
          w2 = w * w;
          if (
            A >
            xm * Math.log(f1 / x1) +
              (n - m + 0.5) * Math.log(z / w) +
              (y - m) * Math.log((w * r) / (x1 * q)) +
              (13680 - (462 - (132 - (99 - 140 / f2) / f2) / f2) / f2) / f1 / 166320 +
              (13680 - (462 - (132 - (99 - 140 / z2) / z2) / z2) / z2) / z / 166320 +
              (13680 - (462 - (132 - (99 - 140 / x2) / x2) / x2) / x2) / x1 / 166320 +
              (13680 - (462 - (132 - (99 - 140 / w2) / w2) / w2) / w2) / w / 166320
          ) {
            continue outer;
          }
        }
      } else {
        s = r / q;
        a = s * (n + 1);
        F = 1.0;
        if (m < y) {
          for (let i = m + 1; i <= y; i++) F *= a / i - s;
        } else if (m > y) {
          for (let i = y + 1; i <= m; i++) F /= a / i - s;
        }
        if (v > F) continue outer;
      }
    } else {
      // Step40
      y = Math.floor(xr - Math.log(v) / lamr);
      if (y > n || v === 0.0) continue outer;
      v = v * (u - p3) * lamr;
      // goto Step50
      k = Math.abs(y - m);
      if (k > 20 && k < nrq / 2.0 - 1) {
        rho = (k / nrq) * ((k * (k / 3.0 + 0.625) + 0.16666666666666666) / nrq + 0.5);
        t = (-k * k) / (2 * nrq);
        A = Math.log(v);
        if (A < t - rho) {
          /* accept */
        } else if (A > t + rho) continue outer;
        else {
          x1 = y + 1;
          f1 = m + 1;
          z = n + 1 - m;
          w = n - y + 1;
          x2 = x1 * x1;
          f2 = f1 * f1;
          z2 = z * z;
          w2 = w * w;
          if (
            A >
            xm * Math.log(f1 / x1) +
              (n - m + 0.5) * Math.log(z / w) +
              (y - m) * Math.log((w * r) / (x1 * q)) +
              (13680 - (462 - (132 - (99 - 140 / f2) / f2) / f2) / f2) / f1 / 166320 +
              (13680 - (462 - (132 - (99 - 140 / z2) / z2) / z2) / z2) / z / 166320 +
              (13680 - (462 - (132 - (99 - 140 / x2) / x2) / x2) / x2) / x1 / 166320 +
              (13680 - (462 - (132 - (99 - 140 / w2) / w2) / w2) / w2) / w / 166320
          ) {
            continue outer;
          }
        }
      } else {
        s = r / q;
        a = s * (n + 1);
        F = 1.0;
        if (m < y) {
          for (let i = m + 1; i <= y; i++) F *= a / i - s;
        } else if (m > y) {
          for (let i = y + 1; i <= m; i++) F /= a / i - s;
        }
        if (v > F) continue outer;
      }
    }

    // Step60
    if (p > 0.5) y = n - y;
    return y;
  }
}

function binomialInversion(
  n: number,
  p: number,
  rng: () => number,
  binomial: BinomialState
): number {
  let q: number, qn: number, np: number, px: number, U: number;
  let X: number, bound: number;

  if (!binomial.has_binomial || binomial.nsave !== n || binomial.psave !== p) {
    binomial.nsave = n;
    binomial.psave = p;
    binomial.has_binomial = true;
    binomial.q = q = 1.0 - p;
    binomial.r = qn = Math.exp(n * Math.log(q));
    binomial.c = np = n * p;
    binomial.m = bound = Math.min(n, Math.floor(np + 10.0 * Math.sqrt(np * q + 1)));
  } else {
    q = binomial.q;
    qn = binomial.r;
    bound = binomial.m;
  }
  X = 0;
  px = qn;
  U = rng();
  while (U > px) {
    X++;
    if (X > bound) {
      X = 0;
      px = qn;
      U = rng();
    } else {
      U -= px;
      px = ((n - X + 1) * p * px) / (X * q);
    }
  }
  return X;
}

/** Per-call binomial state (reset each call to match NumPy behavior) */
function binomialSample(n: number, p: number, rng: () => number): number {
  if (n === 0 || p === 0.0) return 0;

  const binomial = makeBinomialState();
  if (p <= 0.5) {
    if (p * n <= 30.0) {
      return binomialInversion(n, p, rng, binomial);
    } else {
      return binomialBtpe(n, p, rng, binomial);
    }
  } else {
    const q = 1.0 - p;
    if (q * n <= 30.0) {
      return n - binomialInversion(n, q, rng, binomial);
    } else {
      return n - binomialBtpe(n, q, rng, binomial);
    }
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
    return mt19937Float64();
  }
  const shape = Array.isArray(size) ? size : [size];
  const totalSize = shape.reduce((a, b) => a * b, 1);
  const data = fillUniformF64MT(totalSize);
  return ArrayStorage.fromData(new Float64Array(data), shape, 'float64');
}

/**
 * Random values in a given shape (alias for random with shape)
 * @param d0, d1, ..., dn - The dimensions of the returned array
 */
export function rand(...shape: number[]): ArrayStorage | number {
  if (shape.length === 0) {
    return mt19937Float64();
  }
  const totalSize = shape.reduce((a, b) => a * b, 1);
  const data = fillUniformF64MT(totalSize);
  return ArrayStorage.fromData(new Float64Array(data), shape, 'float64');
}

/**
 * Return random floats from standard normal distribution
 * @param d0, d1, ..., dn - The dimensions of the returned array
 */
export function randn(...shape: number[]): ArrayStorage | number {
  if (shape.length === 0) {
    return legacyGauss();
  }
  const totalSize = shape.reduce((a, b) => a * b, 1);
  const data = fillLegacyGauss(totalSize);
  return ArrayStorage.fromData(new Float64Array(data), shape, 'float64');
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

  // NumPy's legacy randint uses rk_interval (rejection sampling with bit masking)
  // rk_interval generates rk_random() values masked to the smallest bitmask >= max,
  // rejecting values > max. range-1 because rk_interval is inclusive [0, max].
  const rng = range - 1;

  if (size === undefined) {
    return rkInterval(rng) + low;
  }

  const shape = Array.isArray(size) ? size : [size];
  const totalSize = shape.reduce((a, b) => a * b, 1);

  // Dispatch by dtype to match NumPy's buffered bounded sampling exactly.
  // Small dtypes (u8/u16) extract multiple values per mt19937 u32 call.
  if (dtype === 'int8' || dtype === 'uint8') {
    const raw = fillRandintU8(totalSize, rng, low);
    const result = ArrayStorage.zeros(shape, dtype);
    const data = result.data as Int8Array | Uint8Array;
    data.set(dtype === 'int8' ? new Int8Array(raw.buffer) : raw);
    return result;
  } else if (dtype === 'int16' || dtype === 'uint16') {
    const raw = fillRandintU16(totalSize, rng, low);
    const result = ArrayStorage.zeros(shape, dtype);
    const data = result.data as Int16Array | Uint16Array;
    data.set(dtype === 'int16' ? new Int16Array(raw.buffer) : raw);
    return result;
  } else if (isBigIntDType(dtype)) {
    const data = fillRandintI64(totalSize, rng, low);
    return ArrayStorage.fromData(new BigInt64Array(data), shape, dtype);
  } else {
    const rawData = fillRkInterval(totalSize, rng);
    const result = ArrayStorage.zeros(shape, dtype);
    const numData = result.data as Exclude<typeof result.data, BigInt64Array | BigUint64Array>;
    for (let i = 0; i < totalSize; i++) {
      numData[i] = rawData[i]! + low;
    }
    return result;
  }
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
    return mt19937Float64() * (high - low) + low;
  }
  const shape = Array.isArray(size) ? size : [size];
  const totalSize = shape.reduce((a, b) => a * b, 1);
  const rawData = fillUniformF64MT(totalSize);
  const data = new Float64Array(totalSize);
  const range = high - low;
  for (let i = 0; i < totalSize; i++) {
    data[i] = rawData[i]! * range + low;
  }
  return ArrayStorage.fromData(data, shape, 'float64');
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
    return legacyGauss() * scale + loc;
  }
  const shape = Array.isArray(size) ? size : [size];
  const totalSize = shape.reduce((a, b) => a * b, 1);
  const rawData = fillLegacyGauss(totalSize);
  const data = new Float64Array(totalSize);
  for (let i = 0; i < totalSize; i++) {
    data[i] = rawData[i]! * scale + loc;
  }
  return ArrayStorage.fromData(data, shape, 'float64');
}

/**
 * Draw samples from a standard normal distribution (mean=0, std=1)
 * @param size - Output shape
 */
export function standard_normal(size?: number | number[]): ArrayStorage | number {
  if (size === undefined) {
    return legacyGauss();
  }
  const shape = Array.isArray(size) ? size : [size];
  const totalSize = shape.reduce((a, b) => a * b, 1);
  const data = fillLegacyGauss(totalSize);
  return ArrayStorage.fromData(new Float64Array(data), shape, 'float64');
}

/**
 * Draw samples from an exponential distribution
 * @param scale - The scale parameter (beta = 1/lambda) (default 1)
 * @param size - Output shape
 */
export function exponential(scale: number = 1, size?: number | number[]): ArrayStorage | number {
  if (size === undefined) {
    return legacyStandardExponential() * scale;
  }
  const shape = Array.isArray(size) ? size : [size];
  const totalSize = shape.reduce((a, b) => a * b, 1);
  const rawData = fillLegacyStandardExponential(totalSize);
  const data = new Float64Array(totalSize);
  for (let i = 0; i < totalSize; i++) {
    data[i] = rawData[i]! * scale;
  }
  return ArrayStorage.fromData(data, shape, 'float64');
}

/**
 * Draw samples from a Poisson distribution
 * @param lam - Expected number of events (lambda) (default 1)
 * @param size - Output shape
 */
export function poisson(lam: number = 1, size?: number | number[]): ArrayStorage | number {
  if (size === undefined) {
    return poissonSample(lam, mt19937Float64);
  }
  const shape = Array.isArray(size) ? size : [size];
  const totalSize = shape.reduce((a, b) => a * b, 1);
  const data = fillPoisson(totalSize, lam);
  return ArrayStorage.fromData(new BigInt64Array(data), shape, 'int64');
}

/**
 * Draw samples from a binomial distribution
 * @param n - Number of trials
 * @param p - Probability of success
 * @param size - Output shape
 */
export function binomial(n: number, p: number, size?: number | number[]): ArrayStorage | number {
  if (size === undefined) {
    return binomialSample(n, p, mt19937Float64);
  }
  const shape = Array.isArray(size) ? size : [size];
  const totalSize = shape.reduce((a, b) => a * b, 1);
  const data = fillBinomial(totalSize, n, p);
  return ArrayStorage.fromData(new BigInt64Array(data), shape, 'int64');
}

/**
 * Implementation of choice
 */
function choiceImpl(
  a: number | ArrayStorage,
  size?: number | number[],
  replace: boolean = true,
  p?: ArrayStorage | number[],
  rng: () => number = mt19937Float64,
  usePCG: boolean = false
): ArrayStorage | number {
  // Fast path: integer a, replace=true, no p — use randint/integers (matches NumPy)
  const n = typeof a === 'number' ? a : a.size;
  if (typeof a === 'number' && replace && p === undefined) {
    if (n === 0) throw new Error('cannot take a sample from an empty sequence');
    if (usePCG) {
      if (size === undefined) {
        return Number(pcg64BoundedUint64(0, n - 1));
      }
      const shape = Array.isArray(size) ? size : [size];
      const totalSize = shape.reduce((s, d) => s * d, 1);
      const result = ArrayStorage.zeros(shape, 'int64');
      const data = result.data as BigInt64Array;
      for (let i = 0; i < totalSize; i++) {
        data[i] = pcg64BoundedUint64(0, n - 1);
      }
      return result;
    }
    // Legacy: use randint(0, n, size) which matches NumPy's choice exactly
    if (size === undefined) {
      return randint(0, n) as number;
    }
    return randint(0, n, Array.isArray(size) ? size : [size]);
  }

  let population: number[];
  if (typeof a === 'number') {
    population = Array.from({ length: a }, (_, i) => i);
  } else {
    population = [];
    for (let i = 0; i < n; i++) {
      population.push(Number(a.iget(i)));
    }
  }
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
    if (!probabilities) {
      if (usePCG) {
        // NumPy Generator.choice(replace=False) uses two algorithms:
        // 1. Floyd's algorithm for small pop or small sample (pop_size <= 10000 or size <= pop_size/50)
        // 2. Tail-shuffle for large pop with large sample (pop_size > 10000 and size > pop_size/50)
        const CUTOFF = 50;
        if (n > 10000 && totalSize > n / CUTOFF) {
          // Large pop_size path: full Fisher-Yates shuffle, take last `size` elements
          const perm = fillPermutationPCG(n);
          for (let i = 0; i < totalSize; i++) {
            const permIdx = Number(perm[n - totalSize + i]!);
            if (typeof a === 'number') {
              data[i] = permIdx;
            } else {
              data[i] = Number(a.iget(permIdx));
            }
          }
        } else {
          // Floyd's algorithm: efficient for small pop or small sample
          const idx = new Array<number>(totalSize);
          const hashSet = new Set<number>();
          for (let j = n - totalSize; j < n; j++) {
            const val = Number(pcg64BoundedUint64(0, j));
            if (!hashSet.has(val)) {
              hashSet.add(val);
              idx[j - (n - totalSize)] = val;
            } else {
              hashSet.add(j);
              idx[j - (n - totalSize)] = j;
            }
          }
          // Shuffle the result (NumPy always shuffles Floyd output)
          for (let i = totalSize - 1; i > 0; i--) {
            const j = Number(pcg64BoundedUint64(0, i));
            const tmp = idx[i]!;
            idx[i] = idx[j]!;
            idx[j] = tmp;
          }
          for (let i = 0; i < totalSize; i++) {
            if (typeof a === 'number') {
              data[i] = idx[i]!;
            } else {
              data[i] = Number(a.iget(idx[i]!));
            }
          }
        }
      } else {
        // Legacy: permutation(pop_size)[:size] — use WASM permutation for exact match
        const perm = permutationImpl(n, rng, false);
        const permData = perm.data;
        for (let i = 0; i < totalSize; i++) {
          const permIdx = Number(permData instanceof BigInt64Array ? permData[i]! : permData[i]!);
          if (typeof a === 'number') {
            data[i] = permIdx;
          } else {
            data[i] = Number(a.iget(permIdx));
          }
        }
      }
    } else {
      const available = [...population];
      const availableProbs = [...probabilities];

      for (let i = 0; i < totalSize; i++) {
        const sum = availableProbs.reduce((a, b) => a + b, 0);
        const r = rng() * sum;
        let cumsum = 0;
        let idx = 0;
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

        data[i] = available[idx]!;
        available.splice(idx, 1);
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
  return choiceImpl(a, size, replace, p, mt19937Float64);
}

/**
 * Implementation of permutation
 */
function permutationImpl(
  x: number | ArrayStorage,
  rng: () => number = mt19937Float64,
  usePCG: boolean = false
): ArrayStorage {
  // Fast path: integer n — do entire shuffle in WASM
  if (typeof x === 'number') {
    if (usePCG) {
      const data = fillPermutationPCG(x);
      return ArrayStorage.fromData(new BigInt64Array(data), [x], 'int64');
    }
    if (rng === mt19937Float64) {
      const data = fillPermutation(x);
      return ArrayStorage.fromData(new Float64Array(data), [x], 'float64');
    }
  }

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
  return permutationImpl(x, mt19937Float64);
}

/**
 * Implementation of shuffle (in-place).
 * Uses WASM permutation to generate index order, then applies it.
 * Matches NumPy's shuffle exactly (rk_interval for legacy, random_interval for Generator).
 */
function shuffleImpl(
  x: ArrayStorage,
  rng: () => number = mt19937Float64,
  usePCG: boolean = false
): void {
  const n = x.size;
  if (n <= 1) return;
  // Generate a permutation of indices via WASM, then reorder x in-place
  const perm = permutationImpl(n, rng, usePCG);
  const permData = perm.data;
  // Copy original values
  const orig = new Array(n);
  for (let i = 0; i < n; i++) {
    orig[i] = x.iget(i);
  }
  // Apply permutation
  for (let i = 0; i < n; i++) {
    const idx = Number(permData instanceof BigInt64Array ? permData[i]! : permData[i]!);
    x.iset(i, orig[idx]!);
  }
}

/**
 * Modify a sequence in-place by shuffling its contents
 * @param x - Array to be shuffled
 */
export function shuffle(x: ArrayStorage): void {
  shuffleImpl(x, mt19937Float64);
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
  // NumPy's rk_fill: extracts 4 bytes per u32 (little-endian), matching randomkit.c line 487
  const result = new Uint8Array(length);
  let i = 0;
  // Process 4 bytes at a time from each u32
  for (; i + 3 < length; i += 4) {
    const r = mt19937Uint32();
    result[i] = r & 0xff;
    result[i + 1] = (r >>> 8) & 0xff;
    result[i + 2] = (r >>> 16) & 0xff;
    result[i + 3] = (r >>> 24) & 0xff;
  }
  // Handle remaining bytes (0-3) from one more u32
  if (i < length) {
    let r = mt19937Uint32();
    for (; i < length; i++, r >>>= 8) {
      result[i] = r & 0xff;
    }
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
  state: {},
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
 * Uses NumPy's legacy_standard_gamma algorithm exactly
 * @param shape - Shape parameter (alpha, must be > 0)
 * @param size - Output shape
 */
export function standard_gamma(shape: number, size?: number | number[]): ArrayStorage | number {
  if (shape <= 0) {
    throw new Error('shape must be positive');
  }
  if (size === undefined) {
    return legacyStandardGamma(shape);
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const data = fillLegacyStandardGamma(totalSize, shape);
  return ArrayStorage.fromData(new Float64Array(data), shapeArr, 'float64');
}

/**
 * Draw samples from a standard Cauchy distribution
 * @param size - Output shape
 */
export function standard_cauchy(size?: number | number[]): ArrayStorage | number {
  // NumPy: legacy_gauss / legacy_gauss (ratio of two polar-method normals with caching)
  if (size === undefined) {
    return legacyGauss() / legacyGauss();
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const data = fillStandardCauchy(totalSize);
  return ArrayStorage.fromData(new Float64Array(data), shapeArr, 'float64');
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

  // NumPy: num = legacyGauss(), denom = legacy_standard_gamma(df/2),
  // return sqrt(df/2) * num / sqrt(denom)
  const generateSample = (): number => {
    const num = legacyGauss();
    const denom = legacyStandardGamma(df / 2);
    return (Math.sqrt(df / 2) * num) / Math.sqrt(denom);
  };

  if (size === undefined) {
    return generateSample();
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const data = fillStandardT(totalSize, df);
  return ArrayStorage.fromData(new Float64Array(data), shapeArr, 'float64');
}

// ============================================================================
// Gamma distribution (needed for many other distributions)
// ============================================================================

/**
 * Draw samples from a Gamma distribution
 * NumPy: scale * legacy_standard_gamma(shape)
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
    return scale * legacyStandardGamma(shape);
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const rawData = fillLegacyStandardGamma(totalSize, shape);
  const data = new Float64Array(totalSize);
  for (let i = 0; i < totalSize; i++) {
    data[i] = rawData[i]! * scale;
  }
  return ArrayStorage.fromData(data, shapeArr, 'float64');
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

  // NumPy legacy_beta: Johnk's algorithm when a<=1 && b<=1, else gamma ratio
  const generateSample = (): number => {
    if (a <= 1.0 && b <= 1.0) {
      // Johnk's algorithm
      while (true) {
        const U = mt19937Float64();
        const V = mt19937Float64();
        const X = Math.pow(U, 1.0 / a);
        const Y = Math.pow(V, 1.0 / b);
        if (X + Y <= 1.0) {
          if (X + Y > 0) {
            return X / (X + Y);
          } else {
            // Log-space fallback
            const logX = Math.log(U) / a;
            const logY = Math.log(V) / b;
            const logM = logX > logY ? logX : logY;
            const lX = logX - logM;
            const lY = logY - logM;
            return Math.exp(lX - Math.log(Math.exp(lX) + Math.exp(lY)));
          }
        }
      }
    } else {
      const Ga = legacyStandardGamma(a);
      const Gb = legacyStandardGamma(b);
      return Ga / (Ga + Gb);
    }
  };

  if (size === undefined) {
    return generateSample();
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((aa, bb) => aa * bb, 1);
  const data = fillBeta(totalSize, a, b);
  return ArrayStorage.fromData(new Float64Array(data), shapeArr, 'float64');
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

  // NumPy random_laplace: branch on U >= 0.5 or U > 0, reject U==0
  const generateSample = (): number => {
    while (true) {
      const U = mt19937Float64();
      if (U >= 0.5) {
        return loc - scale * Math.log(2.0 - U - U);
      } else if (U > 0.0) {
        return loc + scale * Math.log(U + U);
      }
      // Reject U == 0.0
    }
  };

  if (size === undefined) {
    return generateSample();
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const data = fillLaplace(totalSize, loc, scale);
  return ArrayStorage.fromData(new Float64Array(data), shapeArr, 'float64');
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

  // NumPy random_logistic: reject U==0
  const generateSample = (): number => {
    while (true) {
      const U = mt19937Float64();
      if (U > 0.0) {
        return loc + scale * Math.log(U / (1.0 - U));
      }
    }
  };

  if (size === undefined) {
    return generateSample();
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const data = fillLogistic(totalSize, loc, scale);
  return ArrayStorage.fromData(new Float64Array(data), shapeArr, 'float64');
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

  // NumPy: exp(legacy_gauss() * sigma + mean) — uses polar-gauss with caching
  if (size === undefined) {
    return Math.exp(legacyGauss() * sigma + mean);
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const data = fillLognormal(totalSize, mean, sigma);
  return ArrayStorage.fromData(new Float64Array(data), shapeArr, 'float64');
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

  // NumPy random_gumbel: reject U == 1.0 (where 1-next_double==1)
  const generateSample = (): number => {
    while (true) {
      const U = 1.0 - mt19937Float64();
      if (U < 1.0) return loc - scale * Math.log(-Math.log(U));
    }
  };

  if (size === undefined) {
    return generateSample();
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const data = fillGumbel(totalSize, loc, scale);
  return ArrayStorage.fromData(new Float64Array(data), shapeArr, 'float64');
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

  // NumPy legacy_pareto: exp(legacy_standard_exponential() / a) - 1
  const generateSample = (): number => {
    return Math.exp(legacyStandardExponential() / a) - 1;
  };

  if (size === undefined) {
    return generateSample();
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((aa, bb) => aa * bb, 1);
  const data = fillPareto(totalSize, a);
  return ArrayStorage.fromData(new Float64Array(data), shapeArr, 'float64');
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

  // NumPy legacy_power: pow(1 - exp(-legacy_standard_exponential()), 1/a)
  const generateSample = (): number => {
    return Math.pow(1 - Math.exp(-legacyStandardExponential()), 1.0 / a);
  };

  if (size === undefined) {
    return generateSample();
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((aa, bb) => aa * bb, 1);
  const data = fillPower(totalSize, a);
  return ArrayStorage.fromData(new Float64Array(data), shapeArr, 'float64');
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

  // NumPy: mode * sqrt(-2.0 * log1p(-U)) = mode * sqrt(2 * standard_exponential)
  // legacy_rayleigh uses next_double (= mt19937Float64) via legacy_standard_exponential
  if (size === undefined) {
    return scale * Math.sqrt(2.0 * legacyStandardExponential());
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const data = fillRayleigh(totalSize, scale);
  return ArrayStorage.fromData(new Float64Array(data), shapeArr, 'float64');
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

  // NumPy random_triangular: exact match
  const base = right - left;
  const leftbase = mode - left;
  const ratio = leftbase / base;
  const leftprod = leftbase * base;
  const rightprod = (right - mode) * base;

  const generateSample = (): number => {
    const U = mt19937Float64();
    if (U <= ratio) {
      return left + Math.sqrt(U * leftprod);
    } else {
      return right - Math.sqrt((1.0 - U) * rightprod);
    }
  };

  if (size === undefined) {
    return generateSample();
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const data = fillTriangular(totalSize, left, mode, right);
  return ArrayStorage.fromData(new Float64Array(data), shapeArr, 'float64');
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

  // NumPy legacy_wald: uses legacyGauss and mt19937Float64
  const generateSample = (): number => {
    const mu_2l = mean / (2 * scale);
    let Y = legacyGauss();
    Y = mean * Y * Y;
    const X = mean + mu_2l * (Y - Math.sqrt(4 * scale * Y + Y * Y));
    const U = mt19937Float64();
    if (U <= mean / (mean + X)) {
      return X;
    } else {
      return (mean * mean) / X;
    }
  };

  if (size === undefined) {
    return generateSample();
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const data = fillWald(totalSize, mean, scale);
  return ArrayStorage.fromData(new Float64Array(data), shapeArr, 'float64');
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

  // NumPy legacy_weibull: pow(legacy_standard_exponential(), 1/a)
  const generateSample = (): number => {
    if (a === 0.0) return 0.0;
    return Math.pow(legacyStandardExponential(), 1.0 / a);
  };

  if (size === undefined) {
    return generateSample();
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((aa, bb) => aa * bb, 1);
  const data = fillWeibull(totalSize, a);
  return ArrayStorage.fromData(new Float64Array(data), shapeArr, 'float64');
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
  // NumPy: 2.0 * legacy_standard_gamma(df / 2.0)
  if (size === undefined) {
    return legacyChisquare(df);
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const data = fillLegacyChisquare(totalSize, df);
  return ArrayStorage.fromData(new Float64Array(data), shapeArr, 'float64');
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

  // NumPy legacy_noncentral_chisquare
  const generateSample = (): number => {
    if (nonc === 0) {
      return legacyChisquare(df);
    }
    if (1 < df) {
      const Chi2 = legacyChisquare(df - 1);
      const n2 = legacyGauss() + Math.sqrt(nonc);
      return Chi2 + n2 * n2;
    } else {
      const i = poissonSample(nonc / 2.0, mt19937Float64);
      return legacyChisquare(df + 2 * i);
    }
  };

  if (size === undefined) {
    return generateSample();
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const data = fillNoncentralChisquare(totalSize, df, nonc);
  return ArrayStorage.fromData(new Float64Array(data), shapeArr, 'float64');
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

  // NumPy legacy_f: (chisquare(dfnum) * dfden) / (chisquare(dfden) * dfnum)
  const generateSample = (): number => {
    const subexpr1 = legacyChisquare(dfnum) * dfden;
    const subexpr2 = legacyChisquare(dfden) * dfnum;
    return subexpr1 / subexpr2;
  };

  if (size === undefined) {
    return generateSample();
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const data = fillF(totalSize, dfnum, dfden);
  return ArrayStorage.fromData(new Float64Array(data), shapeArr, 'float64');
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

  // NumPy: legacy_noncentral_chisquare(dfnum, nonc) * dfden / (legacy_chisquare(dfden) * dfnum)
  const generateNcChi2 = (): number => {
    if (nonc === 0) return legacyChisquare(dfnum);
    if (1 < dfnum) {
      const Chi2 = legacyChisquare(dfnum - 1);
      const n2 = legacyGauss() + Math.sqrt(nonc);
      return Chi2 + n2 * n2;
    } else {
      const i = poissonSample(nonc / 2.0, mt19937Float64);
      return legacyChisquare(dfnum + 2 * i);
    }
  };

  const generateSample = (): number => {
    const t = generateNcChi2() * dfden;
    return t / (legacyChisquare(dfden) * dfnum);
  };

  if (size === undefined) {
    return generateSample();
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const data = fillNoncentralF(totalSize, dfnum, dfden, nonc);
  return ArrayStorage.fromData(new Float64Array(data), shapeArr, 'float64');
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

  // NumPy: two algorithms depending on p threshold (1/3)
  // p >= 1/3: random_geometric_search — sequential search
  // p < 1/3: legacy_geometric_inversion — ceil(log1p(-U) / log(1-p))
  const generateSample = (): number => {
    if (p >= 1 / 3) {
      // random_geometric_search: sequential CDF search
      let X = 1;
      let sum = p;
      let prod = p;
      const q = 1.0 - p;
      const U = mt19937Float64();
      while (U > sum) {
        prod *= q;
        sum += prod;
        X++;
      }
      return X;
    } else {
      // legacy_geometric_inversion: ceil(log1p(-U) / log(1-p))
      return Math.ceil(Math.log1p(-mt19937Float64()) / Math.log(1 - p));
    }
  };

  if (size === undefined) {
    return generateSample();
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const data = fillGeometric(totalSize, p);
  return ArrayStorage.fromData(new BigInt64Array(data), shapeArr, 'int64');
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

  // NumPy legacy: sample > 10 -> HRUA, sample > 0 -> HYP, else 0
  const generateSample = (): number => {
    if (nsample > 10) {
      return hypergeometricHrua(ngood, nbad, nsample);
    } else if (nsample > 0) {
      return hypergeometricHyp(ngood, nbad, nsample);
    } else {
      return 0;
    }
  };

  if (size === undefined) {
    return generateSample();
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const data = fillHypergeometric(totalSize, ngood, nbad, nsample);
  return ArrayStorage.fromData(new BigInt64Array(data), shapeArr, 'int64');
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

  // NumPy legacy_logseries: exact match
  const r = Math.log(1.0 - p);

  const generateSample = (): number => {
    while (true) {
      const V = mt19937Float64();
      if (V >= p) return 1;
      const U = mt19937Float64();
      const q = 1.0 - Math.exp(r * U);
      if (V <= q * q) {
        const result = Math.floor(1 + Math.log(V) / Math.log(q));
        if (result < 1 || V === 0.0) continue;
        return result;
      }
      if (V >= q) return 1;
      return 2;
    }
  };

  if (size === undefined) {
    return generateSample();
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const data = fillLogseries(totalSize, p);
  return ArrayStorage.fromData(new BigInt64Array(data), shapeArr, 'int64');
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

  // NumPy: Y = legacy_gamma(n, (1-p)/p), return random_poisson(Y)
  const generateSample = (): number => {
    if (p === 1) return 0;
    const Y = legacyStandardGamma(n) * ((1 - p) / p);
    return poissonSample(Y, mt19937Float64);
  };

  if (size === undefined) {
    return generateSample();
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const data = fillNegativeBinomial(totalSize, n, p);
  return ArrayStorage.fromData(new BigInt64Array(data), shapeArr, 'int64');
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

  // NumPy legacy_random_zipf: exact match
  const am1 = a - 1.0;
  const b = Math.pow(2.0, am1);
  const RAND_INT_MAX = 9007199254740991; // Number.MAX_SAFE_INTEGER

  const generateSample = (): number => {
    while (true) {
      const U = 1.0 - mt19937Float64();
      const V = mt19937Float64();
      const X = Math.floor(Math.pow(U, -1.0 / am1));
      if (X > RAND_INT_MAX || X < 1.0) continue;
      const T = Math.pow(1.0 + 1.0 / X, am1);
      if ((V * X * (T - 1.0)) / (b - 1.0) <= T / b) {
        return X;
      }
    }
  };

  if (size === undefined) {
    return generateSample();
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((aa, bb) => aa * bb, 1);
  const data = fillZipf(totalSize, a);
  return ArrayStorage.fromData(new BigInt64Array(data), shapeArr, 'int64');
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

  // NumPy: random_multinomial calls random_binomial(p/remaining_p, dn) for each category
  const generateSample = (): number[] => {
    const result = new Array(k).fill(0);
    let remaining = n;
    let pRemaining = 1.0;

    for (let i = 0; i < k - 1; i++) {
      const bi = binomialSample(remaining, normalizedProbs[i]! / pRemaining, mt19937Float64);
      result[i] = bi;
      remaining -= bi;
      if (remaining <= 0) break;
      pRemaining -= normalizedProbs[i]!;
    }
    if (remaining > 0) {
      result[k - 1] = remaining;
    }
    return result;
  };

  if (size === undefined) {
    const samp = generateSample();
    const result = ArrayStorage.zeros([k], 'int64');
    const data = result.data as BigInt64Array;
    for (let i = 0; i < k; i++) {
      data[i] = BigInt(samp[i]!);
    }
    return result;
  }

  const shapeArr = Array.isArray(size) ? size : [size];
  const numSamples = shapeArr.reduce((a, b) => a * b, 1);
  const outShape = [...shapeArr, k];
  const result = ArrayStorage.zeros(outShape, 'int64');
  const data = result.data as BigInt64Array;

  for (let i = 0; i < numSamples; i++) {
    const samp = generateSample();
    for (let j = 0; j < k; j++) {
      data[i * k + j] = BigInt(samp[j]!);
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

  // SVD decomposition of covariance matrix (matches NumPy's approach)
  const covStorage = ArrayStorage.fromData(new Float64Array(covArr.flat()), [n, n], 'float64');

  const { s: singVals, vt } = svd(covStorage, true, true) as {
    u: ArrayStorage;
    s: ArrayStorage;
    vt: ArrayStorage;
  };

  const vtData_ = vt.data as Float64Array;

  // Check for negative singular values
  const sData = singVals.data as Float64Array;
  for (let i = 0; i < n; i++) {
    if (sData[i]! < -tol) {
      if (check_valid === 'raise') {
        throw new Error('covariance matrix is not positive semi-definite');
      } else if (check_valid === 'warn') {
        console.warn('covariance matrix is not positive semi-definite');
      }
    }
  }

  // Build transform matrix: sqrt(s)[:, None] * v  →  each row i of vt scaled by sqrt(s[i])
  // Result is n x n matrix T where T[i][j] = sqrt(s[i]) * vt[i][j]
  const vtData = vtData_;
  const sqrtS = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    sqrtS[i] = Math.sqrt(Math.max(0, sData[i]!));
  }

  if (size === undefined) {
    const x = fillLegacyGauss(n);
    const result = ArrayStorage.zeros([n], 'float64');
    const data = result.data as Float64Array;
    // x_out[j] = sum_i(x[i] * sqrtS[i] * vt[i][j]) + mean[j]
    for (let j = 0; j < n; j++) {
      let val = meanArr[j]!;
      for (let i = 0; i < n; i++) {
        val += x[i]! * sqrtS[i]! * vtData[i * n + j]!;
      }
      data[j] = val;
    }
    return result;
  }

  const shapeArr = Array.isArray(size) ? size : [size];
  const numSamples = shapeArr.reduce((a, b) => a * b, 1);
  const outShape = [...shapeArr, n];
  const allX = fillLegacyGauss(numSamples * n);
  const data = new Float64Array(numSamples * n);

  for (let s = 0; s < numSamples; s++) {
    const base = s * n;
    for (let j = 0; j < n; j++) {
      let val = meanArr[j]!;
      for (let i = 0; i < n; i++) {
        val += allX[base + i]! * sqrtS[i]! * vtData[i * n + j]!;
      }
      data[base + j] = val;
    }
  }

  return ArrayStorage.fromData(data, outShape, 'float64');
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

  // NumPy: dirichlet uses standard_gamma for each alpha, then normalizes
  // Must preserve per-sample call order: gamma(alpha[0]), gamma(alpha[1]), ... for each sample
  const generateSample = (out: Float64Array, offset: number): void => {
    let sum = 0;
    for (let i = 0; i < k; i++) {
      const g = wasmLegacyStandardGamma(alphaArr[i]!);
      out[offset + i] = g;
      sum += g;
    }
    for (let i = 0; i < k; i++) {
      out[offset + i] = out[offset + i]! / sum;
    }
  };

  if (size === undefined) {
    const data = new Float64Array(k);
    generateSample(data, 0);
    return ArrayStorage.fromData(data, [k], 'float64');
  }

  const shapeArr = Array.isArray(size) ? size : [size];
  const numSamples = shapeArr.reduce((a, b) => a * b, 1);
  const outShape = [...shapeArr, k];
  const data = new Float64Array(numSamples * k);

  for (let i = 0; i < numSamples; i++) {
    generateSample(data, i * k);
  }

  return ArrayStorage.fromData(data, outShape, 'float64');
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

  // NumPy legacy_vonmises: exact match
  const generateSample = (): number => {
    if (kappa < 1e-8) {
      // Uniform on circle
      return Math.PI * (2 * mt19937Float64() - 1);
    }

    let s: number;
    if (kappa < 1e-5) {
      s = 1.0 / kappa + kappa;
    } else {
      const r = 1 + Math.sqrt(1 + 4 * kappa * kappa);
      const rho = (r - Math.sqrt(2 * r)) / (2 * kappa);
      s = (1 + rho * rho) / (2 * rho);
    }

    let W: number;
    while (true) {
      const U = mt19937Float64();
      const Z = Math.cos(Math.PI * U);
      W = (1 + s * Z) / (s + Z);
      const Y = kappa * (s - W);
      const V = mt19937Float64();
      if (Y * (2 - Y) - V >= 0 || Math.log(Y / V) + 1 - Y >= 0) break;
    }

    const U2 = mt19937Float64();
    let result = Math.acos(W);
    if (U2 < 0.5) result = -result;
    result += mu;
    const neg = result < 0;
    let mod = Math.abs(result);
    mod = ((mod + Math.PI) % (2 * Math.PI)) - Math.PI;
    if (neg) mod *= -1;
    return mod;
  };

  if (size === undefined) {
    return generateSample();
  }
  const shapeArr = Array.isArray(size) ? size : [size];
  const totalSize = shapeArr.reduce((a, b) => a * b, 1);
  const data = fillVonmises(totalSize, mu, kappa);
  return ArrayStorage.fromData(new Float64Array(data), shapeArr, 'float64');
}
