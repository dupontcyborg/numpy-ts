/**
 * Result and input types shared by the random namespaces of numpy-ts and numpy-ts/core.
 * The two entries return different array classes, so the types take an array family that maps a
 * dtype to the class a given entry returns.
 */

import type { DType } from '../common/dtype';
import type { NDArrayCore } from '../common/ndarray-core';
import type * as randomOps from '../common/ops/random';

export type SizeArg = number | readonly number[];

// Stand-in for a generic array class: the type of an array of dtype D is ArrayOf<F, D>.
export interface ArrayFamily {
  readonly dtype: DType;
  readonly type: unknown;
}

export type ArrayOf<F extends ArrayFamily, D extends DType = DType> = (F & {
  readonly dtype: D;
})['type'];

export interface CoreArrays extends ArrayFamily {
  readonly type: NDArrayCore<this['dtype']>;
}

// Input arrays for random functions. NDArray extends NDArrayCore, so both entries accept it.
export type RngArray = NDArrayCore;

// A sampling function returns an array of dtype R when size is given and a plain number otherwise.
// Req are the leading required parameters and Opt the optional ones between them and size, each
// typed with undefined so it can be skipped to reach size.
export interface Sampler<
  F extends ArrayFamily,
  Req extends unknown[],
  Opt extends unknown[],
  R extends DType = 'float64',
> {
  (...args: [...Req, ...Opt, size: SizeArg]): ArrayOf<F, R>;
  (...args: [...Req, ...Partial<Opt>, size?: undefined]): number;
}

// Shape-from-arguments generators: no arguments gives a number, any dimensions give an array.
export interface DimSampler<F extends ArrayFamily> {
  (): number;
  (...shape: [number, ...number[]]): ArrayOf<F, 'float64'>;
}

export interface RandInt<F extends ArrayFamily> {
  <D extends DType = 'int64'>(
    low: number,
    high: number | null | undefined,
    size: SizeArg,
    dtype?: D,
  ): ArrayOf<F, D>;
  (low: number, high?: number | null, size?: undefined, dtype?: DType): number;
}

// Unlike the samplers, choice draws from an array, so the element dtype is not known statically.
export interface Choice<F extends ArrayFamily> {
  (
    a: number | RngArray,
    size: SizeArg,
    replace?: boolean,
    p?: RngArray | readonly number[],
  ): ArrayOf<F>;
  (
    a: number | RngArray,
    size?: undefined,
    replace?: boolean,
    p?: RngArray | readonly number[],
  ): number;
}

// Methods of a Generator returned by default_rng.
export interface GeneratorApi<F extends ArrayFamily> {
  random: Sampler<F, [], []>;
  integers: Sampler<F, [low: number], [high: number | undefined], 'int64'>;
  standard_normal: Sampler<F, [], []>;
  normal: Sampler<F, [], [loc: number | undefined, scale: number | undefined]>;
  uniform: Sampler<F, [], [low: number | undefined, high: number | undefined]>;
  choice: Choice<F>;
  permutation(x: number | RngArray): ArrayOf<F>;
  shuffle(x: RngArray): void;
  exponential: Sampler<F, [], [scale: number | undefined]>;
  poisson: Sampler<F, [], [lam: number | undefined], 'int64'>;
  binomial: Sampler<F, [n: number, p: number], [], 'int64'>;
}

type Loc = [loc: number | undefined, scale: number | undefined];

// The public random namespace of an entry point.
export interface RandomApi<F extends ArrayFamily> {
  seed: typeof randomOps.seed;
  get_state: typeof randomOps.get_state;
  set_state: typeof randomOps.set_state;
  get_bit_generator: typeof randomOps.get_bit_generator;
  set_bit_generator: typeof randomOps.set_bit_generator;
  default_rng(seedValue?: number): GeneratorApi<F>;
  Generator: typeof randomOps.Generator;
  bytes: typeof randomOps.bytes;

  random: Sampler<F, [], []>;
  rand: DimSampler<F>;
  randn: DimSampler<F>;
  randint: RandInt<F>;
  random_sample: Sampler<F, [], []>;
  ranf: Sampler<F, [], []>;
  sample: Sampler<F, [], []>;
  random_integers: Sampler<F, [low: number], [high: number | undefined], 'int64'>;

  uniform: Sampler<F, [], [low: number | undefined, high: number | undefined]>;
  normal: Sampler<F, [], Loc>;
  standard_normal: Sampler<F, [], []>;
  exponential: Sampler<F, [], [scale: number | undefined]>;
  standard_exponential: Sampler<F, [], []>;
  gamma: Sampler<F, [shape: number], [scale: number | undefined]>;
  standard_gamma: Sampler<F, [shape: number], []>;
  beta: Sampler<F, [a: number, b: number], []>;
  chisquare: Sampler<F, [df: number], []>;
  noncentral_chisquare: Sampler<F, [df: number, nonc: number], []>;
  f: Sampler<F, [dfnum: number, dfden: number], []>;
  noncentral_f: Sampler<F, [dfnum: number, dfden: number, nonc: number], []>;
  standard_cauchy: Sampler<F, [], []>;
  standard_t: Sampler<F, [df: number], []>;
  laplace: Sampler<F, [], Loc>;
  logistic: Sampler<F, [], Loc>;
  lognormal: Sampler<F, [], [mean: number | undefined, sigma: number | undefined]>;
  gumbel: Sampler<F, [], Loc>;
  pareto: Sampler<F, [a: number], []>;
  power: Sampler<F, [a: number], []>;
  rayleigh: Sampler<F, [], [scale: number | undefined]>;
  triangular: Sampler<F, [left: number, mode: number, right: number], []>;
  wald: Sampler<F, [mean: number, scale: number], []>;
  weibull: Sampler<F, [a: number], []>;
  vonmises: Sampler<F, [mu: number, kappa: number], []>;

  poisson: Sampler<F, [], [lam: number | undefined], 'int64'>;
  binomial: Sampler<F, [n: number, p: number], [], 'int64'>;
  geometric: Sampler<F, [p: number], [], 'int64'>;
  hypergeometric: Sampler<F, [ngood: number, nbad: number, nsample: number], [], 'int64'>;
  logseries: Sampler<F, [p: number], [], 'int64'>;
  negative_binomial: Sampler<F, [n: number, p: number], [], 'int64'>;
  zipf: Sampler<F, [a: number], [], 'int64'>;

  multinomial(n: number, pvals: RngArray | number[], size?: SizeArg): ArrayOf<F, 'int64'>;
  multivariate_normal(
    mean: RngArray | number[],
    cov: RngArray | number[][],
    size?: SizeArg,
    check_valid?: 'warn' | 'raise' | 'ignore',
    tol?: number,
  ): ArrayOf<F, 'float64'>;
  dirichlet(alpha: RngArray | number[], size?: SizeArg): ArrayOf<F, 'float64'>;

  choice: Choice<F>;
  permutation(x: number | RngArray): ArrayOf<F>;
  shuffle(x: RngArray): void;
}
