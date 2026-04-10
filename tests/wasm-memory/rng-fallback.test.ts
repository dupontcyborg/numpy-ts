/**
 * RNG JS-fallback coverage: exercises the bulkFill-based fill* functions
 * that only run when ArrayStorage falls back to JS-backed TypedArrays.
 *
 * Uses a tiny 4 MiB WASM memory so random array allocations exceed the
 * heap and fall back to JS, exercising the fillXxx → bulkFill path
 * in rng.ts instead of the directFill path.
 *
 * IMPORTANT: wasmMemoryConfig must be set BEFORE any WASM operation.
 */

import { wasmMemoryConfig } from '../../src/common/wasm/config';

// Constrain memory BEFORE any WASM initialization — forces JS fallback
// 8 MiB is the minimum for WASM binary statics (~6.4 MiB), leaving a
// tiny heap that overflows for large array allocations.
wasmMemoryConfig.maxMemoryBytes = 8 * 1024 * 1024; // 8 MiB total
wasmMemoryConfig.scratchBytes = 1 * 1024 * 1024; // 1 MiB scratch

import { describe, it, expect } from 'vitest';
import { random } from '../../src/index';

// N large enough that the allocation exceeds our tiny WASM heap,
// forcing JS-backed storage and the bulkFill code path.
const N = 50000;

describe('RNG JS-fallback fill paths', () => {
  it('random.random (fillUniformF64MT)', () => {
    random.seed(42);
    const result = random.random(N) as any;
    expect(result.shape).toEqual([N]);
    const data = result.toArray() as number[];
    expect(data.every((v: number) => v >= 0 && v < 1)).toBe(true);
  });

  it('random.standard_normal (fillLegacyGauss)', () => {
    random.seed(42);
    const result = random.standard_normal(N) as any;
    expect(result.shape).toEqual([N]);
    const data = result.toArray() as number[];
    const mean = data.reduce((a: number, b: number) => a + b, 0) / N;
    expect(Math.abs(mean)).toBeLessThan(0.1);
  });

  it('random.standard_exponential (fillLegacyStandardExponential)', () => {
    random.seed(42);
    const result = random.standard_exponential(N) as any;
    expect(result.shape).toEqual([N]);
    const data = result.toArray() as number[];
    expect(data.every((v: number) => v > 0)).toBe(true);
  });

  it('random.gamma (fillLegacyStandardGamma)', () => {
    random.seed(42);
    const result = random.gamma(5, 1.0, N) as any;
    expect(result.shape).toEqual([N]);
    const data = result.toArray() as number[];
    const mean = data.reduce((a: number, b: number) => a + b, 0) / N;
    expect(Math.abs(mean - 5)).toBeLessThan(0.3);
  });

  it('random.chisquare (fillLegacyChisquare)', () => {
    random.seed(42);
    const result = random.chisquare(4, N) as any;
    expect(result.shape).toEqual([N]);
    const data = result.toArray() as number[];
    const mean = data.reduce((a: number, b: number) => a + b, 0) / N;
    expect(Math.abs(mean - 4)).toBeLessThan(0.3);
  });

  it('random.beta (fillBeta)', () => {
    random.seed(42);
    const result = random.beta(2, 5, N) as any;
    expect(result.shape).toEqual([N]);
    const data = result.toArray() as number[];
    expect(data.every((v: number) => v >= 0 && v <= 1)).toBe(true);
  });

  it('random.f (fillF)', () => {
    random.seed(42);
    const result = random.f(5, 10, N) as any;
    expect(result.shape).toEqual([N]);
    const data = result.toArray() as number[];
    expect(data.every((v: number) => v > 0)).toBe(true);
  });

  it('random.pareto (fillPareto)', () => {
    random.seed(42);
    const result = random.pareto(3, N) as any;
    expect(result.shape).toEqual([N]);
    const data = result.toArray() as number[];
    expect(data.every((v: number) => v >= 0)).toBe(true);
  });

  it('random.power (fillPower)', () => {
    random.seed(42);
    const result = random.power(3, N) as any;
    expect(result.shape).toEqual([N]);
    const data = result.toArray() as number[];
    expect(data.every((v: number) => v >= 0 && v <= 1)).toBe(true);
  });

  it('random.weibull (fillWeibull)', () => {
    random.seed(42);
    const result = random.weibull(2, N) as any;
    expect(result.shape).toEqual([N]);
    const data = result.toArray() as number[];
    expect(data.every((v: number) => v >= 0)).toBe(true);
  });

  it('random.logistic (fillLogistic)', () => {
    random.seed(42);
    const result = random.logistic(0, 1, N) as any;
    expect(result.shape).toEqual([N]);
  });

  it('random.gumbel (fillGumbel)', () => {
    random.seed(42);
    const result = random.gumbel(0, 1, N) as any;
    expect(result.shape).toEqual([N]);
  });

  it('random.laplace (fillLaplace)', () => {
    random.seed(42);
    const result = random.laplace(0, 1, N) as any;
    expect(result.shape).toEqual([N]);
  });

  it('random.rayleigh (fillRayleigh)', () => {
    random.seed(42);
    const result = random.rayleigh(1, N) as any;
    expect(result.shape).toEqual([N]);
    const data = result.toArray() as number[];
    expect(data.every((v: number) => v >= 0)).toBe(true);
  });

  it('random.triangular (fillTriangular)', () => {
    random.seed(42);
    const result = random.triangular(0, 0.5, 1, N) as any;
    expect(result.shape).toEqual([N]);
    const data = result.toArray() as number[];
    expect(data.every((v: number) => v >= 0 && v <= 1)).toBe(true);
  });

  it('random.lognormal (fillLognormal)', () => {
    random.seed(42);
    const result = random.lognormal(0, 1, N) as any;
    expect(result.shape).toEqual([N]);
    const data = result.toArray() as number[];
    expect(data.every((v: number) => v > 0)).toBe(true);
  });

  it('random.wald (fillWald)', () => {
    random.seed(42);
    const result = random.wald(1, 1, N) as any;
    expect(result.shape).toEqual([N]);
    const data = result.toArray() as number[];
    expect(data.every((v: number) => v > 0)).toBe(true);
  });

  it('random.standard_t (fillStandardT)', () => {
    random.seed(42);
    const result = random.standard_t(5, N) as any;
    expect(result.shape).toEqual([N]);
  });

  it('random.noncentral_chisquare (fillNoncentralChisquare)', () => {
    random.seed(42);
    const result = random.noncentral_chisquare(5, 2, N) as any;
    expect(result.shape).toEqual([N]);
  });

  it('random.noncentral_f (fillNoncentralF)', () => {
    random.seed(42);
    const result = random.noncentral_f(5, 10, 2, N) as any;
    expect(result.shape).toEqual([N]);
  });

  it('random.vonmises (fillVonmises)', () => {
    random.seed(42);
    const result = random.vonmises(0, 1, N) as any;
    expect(result.shape).toEqual([N]);
    const data = result.toArray() as number[];
    expect(data.every((v: number) => v >= -Math.PI && v <= Math.PI)).toBe(true);
  });

  it('random.geometric (fillGeometric)', () => {
    random.seed(42);
    const result = random.geometric(0.3, N) as any;
    expect(result.shape).toEqual([N]);
    const data = result.toArray().map(Number) as number[];
    expect(data.every((v: number) => v >= 1 && Number.isInteger(v))).toBe(true);
  });

  it('random.poisson (fillPoisson)', () => {
    random.seed(42);
    const result = random.poisson(7, N) as any;
    expect(result.shape).toEqual([N]);
    const data = result.toArray().map(Number) as number[];
    const mean = data.reduce((a: number, b: number) => a + b, 0) / N;
    expect(Math.abs(mean - 7)).toBeLessThan(0.3);
  });

  it('random.binomial (fillBinomial)', () => {
    random.seed(42);
    const result = random.binomial(20, 0.3, N) as any;
    expect(result.shape).toEqual([N]);
    const data = result.toArray().map(Number) as number[];
    const mean = data.reduce((a: number, b: number) => a + b, 0) / N;
    expect(Math.abs(mean - 6)).toBeLessThan(0.3);
  });

  it('random.negative_binomial (fillNegativeBinomial)', () => {
    random.seed(42);
    const result = random.negative_binomial(5, 0.5, N) as any;
    expect(result.shape).toEqual([N]);
    const data = result.toArray().map(Number) as number[];
    expect(data.every((v: number) => v >= 0)).toBe(true);
  });

  it('random.hypergeometric (fillHypergeometric)', () => {
    random.seed(42);
    const result = random.hypergeometric(20, 30, 10, N) as any;
    expect(result.shape).toEqual([N]);
    const data = result.toArray().map(Number) as number[];
    expect(data.every((v: number) => v >= 0 && v <= 10)).toBe(true);
  });

  it('random.logseries (fillLogseries)', () => {
    random.seed(42);
    const result = random.logseries(0.5, N) as any;
    expect(result.shape).toEqual([N]);
    const data = result.toArray().map(Number) as number[];
    expect(data.every((v: number) => v >= 1 && Number.isInteger(v))).toBe(true);
  });

  it('random.zipf (fillZipf)', () => {
    random.seed(42);
    const result = random.zipf(2, N) as any;
    expect(result.shape).toEqual([N]);
    const data = result.toArray().map(Number) as number[];
    expect(data.every((v: number) => v >= 1 && Number.isInteger(v))).toBe(true);
  });

  it('random.permutation (fillPermutation)', () => {
    random.seed(42);
    const result = random.permutation(N) as any;
    expect(result.shape).toEqual([N]);
  });

  it('random.randint (fillRkInterval)', () => {
    random.seed(42);
    const result = random.randint(0, 100, N) as any;
    expect(result.shape).toEqual([N]);
    const data = result.toArray().map(Number) as number[];
    expect(data.every((v: number) => v >= 0 && v < 100)).toBe(true);
  });
});
