/**
 * Shared benchmark utilities — setup arrays and execute operations.
 * Used by the benchmark runner and the WASM leak test suite.
 *
 * Operation dispatch uses a Record lookup instead of a long if/else chain,
 * turning ~400 string comparisons into an O(1) property access.
 */

import * as np from '../../src/core';
import { serializeNpy, parseNpy, serializeNpzSync, parseNpzSync } from '../../src/io';
import type { BenchmarkSetup } from './types';

export function setupArrays(setup: BenchmarkSetup, operation?: string): Record<string, any> {
  const arrays: Record<string, any> = {};

  for (const [key, spec] of Object.entries(setup)) {
    const { shape, dtype = 'float64', fill = 'zeros', value } = spec;

    if (
      key === 'n' ||
      key === 'axis' ||
      key === 'new_shape' ||
      key === 'shape' ||
      key === 'fill_value' ||
      key === 'target_shape' ||
      key === 'dims' ||
      key === 'kth'
    ) {
      arrays[key] = shape[0];
      if (key === 'new_shape' || key === 'shape' || key === 'target_shape' || key === 'dims') {
        arrays[key] = shape;
        // Store dtype for operations that need it (e.g., randint)
        if (dtype) arrays['dtype'] = dtype;
      }
      continue;
    }

    if (key === 'subscripts') {
      arrays[key] = spec.value;
      continue;
    }

    if (key === 'indices') {
      arrays[key] = shape;
      continue;
    }

    if (value !== undefined) {
      arrays[key] = np.full(shape, value, dtype);
    } else if (fill === 'zeros') {
      arrays[key] = np.zeros(shape, dtype);
    } else if (fill === 'ones') {
      arrays[key] = np.ones(shape, dtype);
    } else if (fill === 'complex' || fill === 'complex_small') {
      const size = shape.reduce((a: number, b: number) => a * b, 1);
      const complexValues = [];
      for (let i = 0; i < size; i++) {
        const v = fill === 'complex_small' ? (i % 10) + 1 : i + 1;
        complexValues.push(new np.Complex(v, v));
      }
      const flat = np.array(complexValues);
      const reshaped = np.reshape(flat, shape);
      flat.dispose();
      // Cast to target dtype (e.g. complex64) if specified
      if (dtype === 'complex64') {
        const casted = np.asarray(reshaped, 'complex64');
        reshaped.dispose();
        arrays[key] = casted;
      } else {
        arrays[key] = reshaped;
      }
    } else if (fill === 'random') {
      const size = shape.reduce((a: number, b: number) => a * b, 1);
      const flat = np.arange(0, size, 1, dtype);
      arrays[key] = np.reshape(flat, shape);
      flat.dispose();
    } else if (fill === 'shuffled') {
      // Deterministic Fisher-Yates shuffle of arange using LCG (seed=42)
      // Create arange with target dtype first (wraps for narrow types), then shuffle
      const size = shape.reduce((a: number, b: number) => a * b, 1);
      const wrapped = np.arange(0, size, 1, dtype);
      const flattened = np.flatten(wrapped);
      const vals = (np.tolist(flattened) as number[]).flat() as number[];
      flattened.dispose();
      wrapped.dispose();
      let seed = 42;
      for (let i = size - 1; i > 0; i--) {
        seed = (seed * 1664525 + 1013904223) >>> 0;
        const j = seed % (i + 1);
        [vals[i], vals[j]] = [vals[j]!, vals[i]!];
      }
      const shuffledFlat = np.array(vals, dtype);
      arrays[key] = np.reshape(shuffledFlat, shape);
      shuffledFlat.dispose();
    } else if (fill === 'arange') {
      const size = shape.reduce((a: number, b: number) => a * b, 1);
      const flat = np.arange(0, size, 1, dtype);
      arrays[key] = np.reshape(flat, shape);
      flat.dispose();
    } else if (fill === 'invertible') {
      const n = shape[0];
      const size = shape.reduce((a: number, b: number) => a * b, 1);
      const flat = np.arange(0, size, 1, dtype);
      const arangeMatrix = np.reshape(flat, shape);
      flat.dispose();
      const identity = np.eye(n, undefined, undefined, dtype);
      const scaled = np.multiply(identity, n * n);
      identity.dispose();
      arrays[key] = np.add(arangeMatrix, scaled);
      arangeMatrix.dispose();
      scaled.dispose();
    }
  }

  if (operation === 'linalg_cholesky' && arrays['a']) {
    // Pre-compute positive-definite matrix: A^T·A + trace(A^T·A)·I
    const a = arrays['a'];
    const n = a.shape[0];
    const aT = np.transpose(a);
    const aTa = np.matmul(aT, a);
    const tr = np.trace(aTa);
    const eye = np.eye(n, undefined, 0, a.dtype);
    const scaled = np.multiply(eye, tr);
    arrays['_posdef'] = np.add(aTa, scaled);
    aT.dispose();
    aTa.dispose();
    eye.dispose();
    scaled.dispose();
  }

  if (operation === 'parseNpy' && arrays['a']) {
    arrays['_npyBytes'] = serializeNpy(arrays['a']);
  } else if (operation === 'serializeNpzSync' && arrays['a']) {
    const npzArrays: Record<string, any> = {};
    for (const [key, val] of Object.entries(arrays)) {
      if (val && val.shape !== undefined) {
        npzArrays[key] = val;
      }
    }
    arrays['_npzArrays'] = npzArrays;
  } else if (operation === 'parseNpzSync' && arrays['a']) {
    const npzArrays: Record<string, any> = {};
    for (const [key, val] of Object.entries(arrays)) {
      if (val && val.shape !== undefined) {
        npzArrays[key] = val;
      }
    }
    arrays['_npzBytes'] = serializeNpzSync(npzArrays);
  }

  return arrays;
}

// ---------------------------------------------------------------------------
// Operation dispatch table
//
// Each entry is `name -> fn(arrays) -> result`.  Using a Record turns dispatch
// from ~400 string comparisons into an O(1) property access.
// ---------------------------------------------------------------------------

export type OpFn = (arrays: Record<string, any>) => any;

const OPERATIONS: Record<string, OpFn> = {
  // Array creation
  zeros: (a) => np.zeros(a['shape']),
  ones: (a) => np.ones(a['shape']),
  empty: (a) => np.empty(a['shape']),
  full: (a) => np.full(a['shape'], a['fill_value']),
  arange: (a) => np.arange(0, a['n'], 1),
  linspace: (a) => np.linspace(0, 100, a['n']),
  logspace: (a) => np.logspace(0, 3, a['n']),
  geomspace: (a) => np.geomspace(1, 1000, a['n']),
  eye: (a) => np.eye(a['n']),
  identity: (a) => np.identity(a['n']),
  copy: (a) => np.copy(a['a']),
  zeros_like: (a) => np.zeros_like(a['a']),
  ones_like: (a) => np.ones_like(a['a']),
  empty_like: (a) => np.empty_like(a['a']),
  full_like: (a) => np.full_like(a['a'], 7),

  // Arithmetic
  add: (a) => np.add(a['a'], a['b']),
  subtract: (a) => np.subtract(a['a'], a['b']),
  multiply: (a) => np.multiply(a['a'], a['b']),
  divide: (a) => np.divide(a['a'], a['b']),
  mod: (a) => np.mod(a['a'], a['b']),
  floor_divide: (a) => np.floor_divide(a['a'], a['b']),
  reciprocal: (a) => np.reciprocal(a['a']),
  positive: (a) => np.positive(a['a']),
  cbrt: (a) => np.cbrt(a['a']),
  fabs: (a) => np.fabs(a['a']),
  divmod: (a) => np.divmod(a['a'], a['b']),
  gcd: (a) => {
    const scalar = a['b'].size === 1 ? a['b'].get([0]) : a['b'];
    return np.gcd(a['a'], scalar);
  },
  lcm: (a) => {
    const scalar = a['b'].size === 1 ? a['b'].get([0]) : a['b'];
    return np.lcm(a['a'], scalar);
  },
  float_power: (a) => {
    const scalar = a['b'].size === 1 ? a['b'].get([0]) : a['b'];
    return np.float_power(a['a'], scalar);
  },
  square: (a) => np.square(a['a']),
  remainder: (a) => np.remainder(a['a'], a['b']),
  heaviside: (a) => np.heaviside(a['a'], a['b']),
  fmod: (a) => np.fmod(a['a'], a['b']),
  frexp: (a) => np.frexp(a['a']),
  ldexp: (a) => np.ldexp(a['a'], a['b']),
  modf: (a) => np.modf(a['a']),

  // Math
  sqrt: (a) => np.sqrt(a['a']),
  power: (a) => np.power(a['a'], a['b']),
  absolute: (a) => np.absolute(a['a']),
  negative: (a) => np.negative(a['a']),
  sign: (a) => np.sign(a['a']),

  // Trig
  sin: (a) => np.sin(a['a']),
  cos: (a) => np.cos(a['a']),
  tan: (a) => np.tan(a['a']),
  arcsin: (a) => np.arcsin(a['a']),
  arccos: (a) => np.arccos(a['a']),
  arctan: (a) => np.arctan(a['a']),
  arctan2: (a) => np.arctan2(a['a'], a['b']),
  hypot: (a) => np.hypot(a['a'], a['b']),

  // Hyperbolic
  sinh: (a) => np.sinh(a['a']),
  cosh: (a) => np.cosh(a['a']),
  tanh: (a) => np.tanh(a['a']),
  arcsinh: (a) => np.arcsinh(a['a']),
  arccosh: (a) => np.arccosh(a['a']),
  arctanh: (a) => np.arctanh(a['a']),

  // Exponential
  exp: (a) => np.exp(a['a']),
  exp2: (a) => np.exp2(a['a']),
  expm1: (a) => np.expm1(a['a']),
  log: (a) => np.log(a['a']),
  log2: (a) => np.log2(a['a']),
  log10: (a) => np.log10(a['a']),
  log1p: (a) => np.log1p(a['a']),
  logaddexp: (a) => np.logaddexp(a['a'], a['b']),
  logaddexp2: (a) => np.logaddexp2(a['a'], a['b']),

  // Gradient
  diff: (a) => np.diff(a['a']),
  gradient: (a) => np.gradient(a['a']),
  cross: (a) => np.cross(a['a'], a['b']),

  // Linear algebra
  dot: (a) => np.dot(a['a'], a['b']),
  inner: (a) => np.inner(a['a'], a['b']),
  outer: (a) => np.outer(a['a'], a['b']),
  tensordot: (a) => np.tensordot(a['a'], a['b'], a['axes'] ?? 2),
  matmul: (a) => np.matmul(a['a'], a['b']),
  trace: (a) => np.trace(a['a']),
  transpose: (a) => np.transpose(a['a']),
  diagonal: (a) => np.diagonal(a['a']),
  kron: (a) => np.kron(a['a'], a['b']),
  einsum: (a) => np.einsum(a['subscripts'], a['a'], a['b']),
  deg2rad: (a) => np.deg2rad(a['a']),
  rad2deg: (a) => np.rad2deg(a['a']),

  // numpy.linalg
  linalg_det: (a) => np.linalg.det(a['a']),
  linalg_inv: (a) => np.linalg.inv(a['a']),
  linalg_solve: (a) => np.linalg.solve(a['a'], a['b']),
  linalg_qr: (a) => np.linalg.qr(a['a']),
  linalg_cholesky: (a) => np.linalg.cholesky(a['_posdef']),
  linalg_svd: (a) => np.linalg.svd(a['a']),
  linalg_eig: (a) => np.linalg.eig(a['a']),
  linalg_eigh: (a) => {
    const arr = a['a'];
    const aT = np.transpose(arr);
    const sum = np.add(arr, aT);
    const sym = np.multiply(sum, 0.5);
    const result = np.linalg.eigh(sym);
    sum.dispose();
    sym.dispose();
    return result;
  },
  linalg_norm: (a) => np.linalg.norm(a['a']),
  linalg_matrix_rank: (a) => np.linalg.matrix_rank(a['a']),
  linalg_pinv: (a) => np.linalg.pinv(a['a']),
  linalg_cond: (a) => np.linalg.cond(a['a']),
  linalg_matrix_power: (a) => np.linalg.matrix_power(a['a'], 3),
  linalg_lstsq: (a) => np.linalg.lstsq(a['a'], a['b']),
  linalg_cross: (a) => np.linalg.cross(a['a'], a['b']),
  linalg_slogdet: (a) => np.linalg.slogdet(a['a']),
  linalg_svdvals: (a) => np.linalg.svdvals(a['a']),
  linalg_multi_dot: (a) => np.linalg.multi_dot([a['a'], a['b'], a['c']]),
  vdot: (a) => np.vdot(a['a'], a['b']),
  vecdot: (a) => np.vecdot(a['a'], a['b']),
  matrix_transpose: (a) => np.matrix_transpose(a['a']),
  matvec: (a) => np.matvec(a['a'], a['b']),
  vecmat: (a) => np.vecmat(a['a'], a['b']),

  // Bitwise
  bitwise_and: (a) => np.bitwise_and(a['a'], a['b']),
  bitwise_or: (a) => np.bitwise_or(a['a'], a['b']),
  bitwise_xor: (a) => np.bitwise_xor(a['a'], a['b']),
  bitwise_not: (a) => np.bitwise_not(a['a']),
  invert: (a) => np.invert(a['a']),
  left_shift: (a) => np.left_shift(a['a'], a['b']),
  right_shift: (a) => np.right_shift(a['a'], a['b']),
  packbits: (a) => np.packbits(a['a']),
  unpackbits: (a) => np.unpackbits(a['a']),
  bitwise_count: (a) => np.bitwise_count(a['a']),

  // Reductions
  sum: (a) => np.sum(a['a'], a['axis']),
  mean: (a) => np.mean(a['a'], a['axis']),
  max: (a) => np.max(a['a'], a['axis']),
  min: (a) => np.min(a['a'], a['axis']),
  prod: (a) => np.prod(a['a'], a['axis']),
  argmin: (a) => np.argmin(a['a'], a['axis']),
  argmax: (a) => np.argmax(a['a'], a['axis']),
  var: (a) => np.var_(a['a'], a['axis']),
  std: (a) => np.std(a['a'], a['axis']),
  all: (a) => np.all(a['a'], a['axis']),
  any: (a) => np.any(a['a'], a['axis']),
  cumsum: (a) => np.cumsum(a['a']),
  cumprod: (a) => np.cumprod(a['a']),
  ptp: (a) => np.ptp(a['a']),
  median: (a) => np.median(a['a']),
  percentile: (a) => np.percentile(a['a'], 50),
  quantile: (a) => np.quantile(a['a'], 0.5),
  average: (a) => np.average(a['a']),
  nansum: (a) => np.nansum(a['a']),
  nanmean: (a) => np.nanmean(a['a']),
  nanmin: (a) => np.nanmin(a['a']),
  nanmax: (a) => np.nanmax(a['a']),
  nanquantile: (a) => np.nanquantile(a['a'], 0.5),
  nanpercentile: (a) => np.nanpercentile(a['a'], 50),

  // Array creation - extra
  asarray_chkfinite: (a) => np.asarray_chkfinite(a['a']),
  require: (a) => np.require(a['a'], undefined, 'C'),

  // Reshape
  reshape: (a) => np.reshape(a['a'], a['new_shape']),
  flatten: (a) => np.flatten(a['a']),
  ravel: (a) => np.ravel(a['a']),
  squeeze: (a) => np.squeeze(a['a']),

  // Slicing
  slice: (a) => a['a'].slice('0:100', '0:100'),

  // Manipulation
  swapaxes: (a) => np.swapaxes(a['a'], 0, 1),
  concatenate: (a) => np.concatenate([a['a'], a['b']], 0),
  stack: (a) => np.stack([a['a'], a['b']], 0),
  vstack: (a) => np.vstack([a['a'], a['b']]),
  hstack: (a) => np.hstack([a['a'], a['b']]),
  tile: (a) => np.tile(a['a'], [2, 2]),
  repeat: (a) => np.repeat(a['a'], 2),
  broadcast_to: (a) => np.broadcast_to(a['a'], a['target_shape']),
  take: (a) => np.take(a['a'], a['indices']),
  flip: (a) => np.flip(a['a']),
  rot90: (a) => np.rot90(a['a']),
  roll: (a) => np.roll(a['a'], 10),
  pad: (a) => np.pad(a['a'], 2),
  concat: (a) => np.concat([a['a'], a['b']], 0),
  unstack: (a) => np.unstack(a['a'], 0),
  block: (a) => np.block([a['a'], a['b']]),
  item: (a) => np.item(a['a'], 0),
  tolist: (a) => np.tolist(a['a']),

  // Creation - misc
  diag: (a) => np.diag(a['a']),
  tri: (a) => np.tri(a['shape'][0], a['shape'][1]),
  tril: (a) => np.tril(a['a']),
  triu: (a) => np.triu(a['a']),

  // Indexing
  take_along_axis: (a) => np.take_along_axis(a['a'], a['b'], 0),
  compress: (a) => np.compress(a['b'], a['a'], 0),
  diag_indices: (a) => np.diag_indices(a['n']),
  tril_indices: (a) => np.tril_indices(a['n']),
  triu_indices: (a) => np.triu_indices(a['n']),
  indices: (a) => np.indices(a['shape']),
  ravel_multi_index: (a) => np.ravel_multi_index([a['a'], a['b']], a['dims']),
  unravel_index: (a) => np.unravel_index(a['a'], a['dims']),

  // IO
  serializeNpy: (a) => serializeNpy(a['a']),
  parseNpy: (a) => parseNpy(a['_npyBytes']),
  serializeNpzSync: (a) => serializeNpzSync(a['_npzArrays']),
  parseNpzSync: (a) => parseNpzSync(a['_npzBytes']),

  // Sorting
  sort: (a) => np.sort(a['a']),
  argsort: (a) => np.argsort(a['a']),
  partition: (a) => np.partition(a['a'], a['kth'] || 0),
  argpartition: (a) => np.argpartition(a['a'], a['kth'] || 0),
  lexsort: (a) => np.lexsort([a['a'], a['b']]),
  sort_complex: (a) => np.sort_complex(a['a']),

  // Searching
  nonzero: (a) => np.nonzero(a['a']),
  argwhere: (a) => np.argwhere(a['a']),
  flatnonzero: (a) => np.flatnonzero(a['a']),
  where: (a) => np.where(a['a'], a['b'], a['c']),
  searchsorted: (a) => np.searchsorted(a['a'], a['b']),
  extract: (a) => np.extract(a['condition'], a['a']),
  count_nonzero: (a) => np.count_nonzero(a['a']),

  // Statistics
  bincount: (a) => np.bincount(a['a']),
  digitize: (a) => np.digitize(a['a'], a['b']),
  histogram: (a) => np.histogram(a['a'], 10),
  histogram2d: (a) => np.histogram2d(a['a'], a['b'], 10),
  correlate: (a) => np.correlate(a['a'], a['b'], 'full'),
  convolve: (a) => np.convolve(a['a'], a['b'], 'full'),
  cov: (a) => np.cov(a['a']),
  corrcoef: (a) => np.corrcoef(a['a']),
  histogram_bin_edges: (a) => np.histogram_bin_edges(a['a'], 10),
  trapezoid: (a) => np.trapezoid(a['a']),

  // Set
  trim_zeros: (a) => np.trim_zeros(a['a']),
  unique_values: (a) => np.unique_values(a['a']),
  unique_counts: (a) => np.unique_counts(a['a']),

  // Logic
  logical_and: (a) =>
    a['b'] ? np.logical_and(a['a'], a['b']) : np.logical_and(a['a'], a['scalar']),
  logical_or: (a) =>
    a['b'] ? np.logical_or(a['a'], a['b']) : np.logical_or(a['a'], a['scalar']),
  logical_not: (a) => np.logical_not(a['a']),
  logical_xor: (a) =>
    a['b'] ? np.logical_xor(a['a'], a['b']) : np.logical_xor(a['a'], a['scalar']),
  isfinite: (a) => np.isfinite(a['a']),
  isinf: (a) => np.isinf(a['a']),
  isnan: (a) => np.isnan(a['a']),
  isneginf: (a) => np.isneginf(a['a']),
  isposinf: (a) => np.isposinf(a['a']),
  isreal: (a) => np.isreal(a['a']),
  signbit: (a) => np.signbit(a['a']),
  copysign: (a) =>
    a['b'] ? np.copysign(a['a'], a['b']) : np.copysign(a['a'], a['scalar']),

  // Random (legacy)
  random_random: (a) => np.random.random(a['shape']),
  random_rand: (a) => np.random.rand(...a['shape']),
  random_randn: (a) => np.random.randn(...a['shape']),
  random_randint: (a) => np.random.randint(0, 100, a['shape'], a['dtype'] || 'int64'),
  random_uniform: (a) => np.random.uniform(0, 1, a['shape']),
  random_normal: (a) => np.random.normal(0, 1, a['shape']),
  random_standard_normal: (a) => np.random.standard_normal(a['shape']),
  random_exponential: (a) => np.random.exponential(1, a['shape']),
  random_poisson: (a) => np.random.poisson(5, a['shape']),
  random_binomial: (a) => np.random.binomial(10, 0.5, a['shape']),
  random_choice: (a) => np.random.choice(a['n'], 100),
  random_permutation: (a) => np.random.permutation(a['n']),
  random_gamma: (a) => np.random.gamma(2, 1, a['shape']),
  random_beta: (a) => np.random.beta(2, 5, a['shape']),
  random_chisquare: (a) => np.random.chisquare(5, a['shape']),
  random_laplace: (a) => np.random.laplace(0, 1, a['shape']),
  random_geometric: (a) => np.random.geometric(0.5, a['shape']),
  random_dirichlet: (a) => np.random.dirichlet([1, 2, 3], a['shape'][0]),
  random_standard_exponential: (a) => np.random.standard_exponential(a['shape']),
  random_logistic: (a) => np.random.logistic(0, 1, a['shape']),
  random_lognormal: (a) => np.random.lognormal(0, 1, a['shape']),
  random_gumbel: (a) => np.random.gumbel(0, 1, a['shape']),
  random_pareto: (a) => np.random.pareto(3, a['shape']),
  random_power: (a) => np.random.power(3, a['shape']),
  random_rayleigh: (a) => np.random.rayleigh(1, a['shape']),
  random_weibull: (a) => np.random.weibull(3, a['shape']),
  random_triangular: (a) => np.random.triangular(0, 0.5, 1, a['shape']),
  random_standard_cauchy: (a) => np.random.standard_cauchy(a['shape']),
  random_standard_t: (a) => np.random.standard_t(5, a['shape']),
  random_wald: (a) => np.random.wald(1, 1, a['shape']),
  random_vonmises: (a) => np.random.vonmises(0, 1, a['shape']),
  random_zipf: (a) => np.random.zipf(2, a['shape']),

  // Generator (PCG64) random
  gen_random: (a) => np.random.default_rng(42).random(a['shape']),
  gen_uniform: (a) => np.random.default_rng(42).uniform(0, 1, a['shape']),
  gen_standard_normal: (a) => np.random.default_rng(42).standard_normal(a['shape']),
  gen_normal: (a) => np.random.default_rng(42).normal(0, 1, a['shape']),
  gen_exponential: (a) => np.random.default_rng(42).exponential(1, a['shape']),
  gen_integers: (a) => np.random.default_rng(42).integers(0, 100, a['shape']),
  gen_permutation: (a) => np.random.default_rng(42).permutation(a['n']),

  // Complex
  complex_zeros: (a) => np.zeros(a['shape'], 'complex128'),
  complex_ones: (a) => np.ones(a['shape'], 'complex128'),
  complex_add: (a) => np.add(a['a'], a['b']),
  complex_multiply: (a) => np.multiply(a['a'], a['b']),
  complex_divide: (a) => np.divide(a['a'], a['b']),
  complex_real: (a) => np.real(a['a']),
  complex_imag: (a) => np.imag(a['a']),
  complex_conj: (a) => np.conj(a['a']),
  complex_angle: (a) => np.angle(a['a']),
  complex_abs: (a) => np.abs(a['a']),
  complex_sqrt: (a) => np.sqrt(a['a']),
  complex_sum: (a) => np.sum(a['a']),
  complex_mean: (a) => np.mean(a['a']),
  complex_prod: (a) => np.prod(a['a']),

  // Other math
  clip: (a) => np.clip(a['a'], 10, 100),
  maximum: (a) => np.maximum(a['a'], a['b']),
  minimum: (a) => np.minimum(a['a'], a['b']),
  fmax: (a) => np.fmax(a['a'], a['b']),
  fmin: (a) => np.fmin(a['a'], a['b']),
  nan_to_num: (a) => np.nan_to_num(a['a']),
  interp: (a) => np.interp(a['x'], a['xp'], a['fp']),
  unwrap: (a) => np.unwrap(a['a']),
  sinc: (a) => np.sinc(a['a']),
  i0: (a) => np.i0(a['a']),

  // Polynomial
  poly: (a) => np.poly(a['a']),
  polyadd: (a) => np.polyadd(a['a'], a['b']),
  polyder: (a) => np.polyder(a['a']),
  polydiv: (a) => np.polydiv(a['a'], a['b']),
  polyfit: (a) => np.polyfit(a['a'], a['b'], 2),
  polyint: (a) => np.polyint(a['a']),
  polymul: (a) => np.polymul(a['a'], a['b']),
  polysub: (a) => np.polysub(a['a'], a['b']),
  polyval: (a) => np.polyval(a['a'], a['b']),
  roots: (a) => np.roots(a['a']),

  // Type checking
  can_cast: () => np.can_cast('int32', 'float64'),
  result_type: () => np.result_type('int32', 'float64'),
  min_scalar_type: () => np.min_scalar_type(1000),
  issubdtype: () => np.issubdtype('int32', 'integer'),

  // FFT
  fft: (a) => np.fft.fft(a['a']),
  ifft: (a) => np.fft.ifft(a['a']),
  fft2: (a) => np.fft.fft2(a['a']),
  ifft2: (a) => np.fft.ifft2(a['a']),
  fftn: (a) => np.fft.fftn(a['a']),
  ifftn: (a) => np.fft.ifftn(a['a']),
  rfft: (a) => np.fft.rfft(a['a']),
  irfft: (a) => np.fft.irfft(a['a']),
  rfft2: (a) => np.fft.rfft2(a['a']),
  irfft2: (a) => np.fft.irfft2(a['a']),
  rfftn: (a) => np.fft.rfftn(a['a']),
  irfftn: (a) => np.fft.irfftn(a['a']),
  hfft: (a) => np.fft.hfft(a['a']),
  ihfft: (a) => np.fft.ihfft(a['a']),
  fftfreq: (a) => np.fft.fftfreq(a['n']),
  rfftfreq: (a) => np.fft.rfftfreq(a['n']),
  fftshift: (a) => np.fft.fftshift(a['a']),
  ifftshift: (a) => np.fft.ifftshift(a['a']),
};

/** Execute a named operation via O(1) Record lookup. */
export function executeOperation(operation: string, arrays: Record<string, any>): any {
  const fn = OPERATIONS[operation];
  if (!fn) throw new Error(`Unknown operation: ${operation}`);
  return fn(arrays);
}

/**
 * Eagerly free WASM-backed memory from a benchmark result.
 * Without this, the WASM heap exhausts during tight benchmark loops
 * because FinalizationRegistry can't keep up with allocation rate.
 */
