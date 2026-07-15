/**
 * Bridge test: the *runtime* op result dtype must equal the SSOT result-dtype
 * helper its compile-time type is generated from.
 *
 * The types (src/common/dtype-promotion.ts) are generated from the helpers in
 * src/common/dtype.ts, and those helpers are proven against NumPy in
 * tests/validation/dtype-result-rules.numpy.test.ts. This test pins the *third*
 * edge of the triangle — that the actual op emits the dtype the helper predicts —
 * so the emitted type can't silently lie even if an op computes its result dtype
 * on a code path separate from the SSOT (which is exactly how several reductions
 * used to hardcode float64 while typed float16/float32).
 *
 * Coverage is the full newly-typed reduction surface plus the separate-path
 * elementwise helpers (abs/angle/divide). Methods route through the same core
 * ops, so free-function coverage guards both. Binary promotion ops aren't here:
 * their kernels call the very `promoteDTypes` the table is generated from.
 */

import { describe, expect, it } from 'vitest';
import * as np from '../../src';
import {
  absResultDtype,
  angleResultDtype,
  DTYPES,
  type DType,
  promoteDTypes,
  reductionAccumDtype,
  roundResultDtype,
  stdVarResultDtype,
  trueDivideResultDtype,
} from '../../src/common/dtype';

const without = (...ex: DType[]): DType[] => DTYPES.filter((d) => !ex.includes(d));
const ALL = [...DTYPES];
const NO_COMPLEX = without('complex64', 'complex128');
const ORDERABLE = without('complex64', 'complex128', 'bool'); // percentile/quantile reject bool + complex

const preserve = (d: DType): DType => d;
const bool = (): DType => 'bool';

/** 2-D input so an axis-0 reduction still yields an NDArray with a readable `.dtype`. */
const mat = (d: DType) => np.ones([2, 3], d);
const dtypeOf = (r: unknown): DType => (r as { dtype: DType }).dtype;

// Each entry: a reduction whose axis-0 result dtype must equal `rule(inputDtype)`.
const REDUCTIONS: {
  name: string;
  call: (a: ReturnType<typeof mat>) => unknown;
  rule: (d: DType) => DType;
  dtypes: DType[];
}[] = [
  // TrueDivide (mean / median / average family)
  { name: 'mean', call: (a) => np.mean(a, 0), rule: trueDivideResultDtype, dtypes: ALL },
  { name: 'median', call: (a) => np.median(a, 0), rule: trueDivideResultDtype, dtypes: ALL },
  { name: 'average', call: (a) => np.average(a, 0), rule: trueDivideResultDtype, dtypes: ALL },
  { name: 'nanmean', call: (a) => np.nanmean(a, 0), rule: trueDivideResultDtype, dtypes: ALL },
  { name: 'nanmedian', call: (a) => np.nanmedian(a, 0), rule: trueDivideResultDtype, dtypes: ALL },
  {
    name: 'percentile',
    call: (a) => np.percentile(a, 50, 0),
    rule: trueDivideResultDtype,
    dtypes: ORDERABLE,
  },
  {
    name: 'quantile',
    call: (a) => np.quantile(a, 0.5, 0),
    rule: trueDivideResultDtype,
    dtypes: ORDERABLE,
  },
  {
    name: 'nanpercentile',
    call: (a) => np.nanpercentile(a, 50, 0),
    rule: trueDivideResultDtype,
    dtypes: ORDERABLE,
  },
  {
    name: 'nanquantile',
    call: (a) => np.nanquantile(a, 0.5, 0),
    rule: trueDivideResultDtype,
    dtypes: ORDERABLE,
  },

  // StdVar (std / var family)
  { name: 'std', call: (a) => np.std(a, 0), rule: stdVarResultDtype, dtypes: ALL },
  { name: 'variance', call: (a) => np.variance(a, 0), rule: stdVarResultDtype, dtypes: ALL },
  { name: 'nanstd', call: (a) => np.nanstd(a, 0), rule: stdVarResultDtype, dtypes: ALL },
  { name: 'nanvar', call: (a) => np.nanvar(a, 0), rule: stdVarResultDtype, dtypes: ALL },

  // ReductionAccum (sum / prod family)
  { name: 'sum', call: (a) => np.sum(a, 0), rule: reductionAccumDtype, dtypes: ALL },
  { name: 'prod', call: (a) => np.prod(a, 0), rule: reductionAccumDtype, dtypes: ALL },
  { name: 'nansum', call: (a) => np.nansum(a, 0), rule: reductionAccumDtype, dtypes: ALL },
  { name: 'nanprod', call: (a) => np.nanprod(a, 0), rule: reductionAccumDtype, dtypes: ALL },

  // Preserve (min / max / ptp family). ptp excludes int64/uint64: the runtime
  // lacks bigint ptp support (throws) — a separate gap, not a dtype divergence.
  { name: 'min', call: (a) => np.min(a, 0), rule: preserve, dtypes: NO_COMPLEX },
  { name: 'max', call: (a) => np.max(a, 0), rule: preserve, dtypes: NO_COMPLEX },
  {
    name: 'ptp',
    call: (a) => np.ptp(a, 0),
    rule: preserve,
    dtypes: without('complex64', 'complex128', 'bool', 'int64', 'uint64'),
  },
  { name: 'nanmin', call: (a) => np.nanmin(a, 0), rule: preserve, dtypes: NO_COMPLEX },
  { name: 'nanmax', call: (a) => np.nanmax(a, 0), rule: preserve, dtypes: NO_COMPLEX },

  // Predicate → bool
  { name: 'all', call: (a) => np.all(a, 0), rule: bool, dtypes: ALL },
  { name: 'any', call: (a) => np.any(a, 0), rule: bool, dtypes: ALL },
];

// Index family — numpy-ts convention. NumPy returns int64 (`intp`) for these,
// but numpy-ts deliberately does NOT: int64 is BigInt-backed, so int64 indices
// would make element access a `bigint` and break `arr[idx]` ergonomics. Instead
// argmin/argmax/nanarg* return int32 and the sort/where family returns float64 —
// both `number`-yielding. The types reflect this; the NumPy divergence is an
// explicit carveout (see tests/validation/dtype-result-rules.numpy.test.ts).
const INDEX_INT32: { name: string; call: (a: ReturnType<typeof mat>) => unknown }[] = [
  { name: 'argmin', call: (a) => np.argmin(a, 0) },
  { name: 'argmax', call: (a) => np.argmax(a, 0) },
  { name: 'nanargmin', call: (a) => np.nanargmin(a, 0) },
  { name: 'nanargmax', call: (a) => np.nanargmax(a, 0) },
];
const INDEX_FLOAT64: { name: string; call: (a: ReturnType<typeof mat>) => unknown }[] = [
  { name: 'argsort', call: (a) => np.argsort(a) },
  { name: 'argpartition', call: (a) => np.argpartition(a, 1) },
  { name: 'argwhere', call: (a) => np.argwhere(a) },
  { name: 'flatnonzero', call: (a) => np.flatnonzero(a) },
];

describe('index family dtype matches its (number-yielding) type', () => {
  for (const { name, call } of INDEX_INT32) {
    describe(`${name} → int32`, () => {
      it.each(NO_COMPLEX)('%s', (d) => expect(dtypeOf(call(mat(d)))).toBe('int32'));
    });
  }
  for (const { name, call } of INDEX_FLOAT64) {
    describe(`${name} → float64`, () => {
      it.each(NO_COMPLEX)('%s', (d) => expect(dtypeOf(call(mat(d)))).toBe('float64'));
    });
  }
});

describe('runtime reduction dtype matches its SSOT result-dtype helper', () => {
  for (const { name, call, rule, dtypes } of REDUCTIONS) {
    describe(name, () => {
      it.each(dtypes)('%s', (d) => {
        expect(dtypeOf(call(mat(d)))).toBe(rule(d));
      });
    });
  }
});

// Separate-path elementwise helpers (their runtime dtype is computed apart from
// the SSOT function, so they need direct pinning).
const vec = (d: DType) => np.ones([3], d);

describe('runtime elementwise dtype matches its SSOT result-dtype helper', () => {
  describe('absolute → absResultDtype', () => {
    it.each(ALL)('%s', (d) => {
      expect(dtypeOf(np.absolute(vec(d)))).toBe(absResultDtype(d));
    });
  });
  describe('angle → angleResultDtype', () => {
    it.each(ALL)('%s', (d) => {
      expect(dtypeOf(np.angle(vec(d)))).toBe(angleResultDtype(d));
    });
  });
  describe('divide → trueDivideResultDtype', () => {
    // Same dtype on both operands ⇒ Promote<D,D> = D, so the rule is trueDivide(D).
    it.each(ALL)('%s', (d) => {
      expect(dtypeOf(np.divide(vec(d), vec(d)))).toBe(trueDivideResultDtype(d));
    });
  });
  describe('round → roundResultDtype (preserve, except bool → float16)', () => {
    it.each(ALL)('%s', (d) => {
      expect(dtypeOf(np.round(vec(d)))).toBe(roundResultDtype(d));
    });
  });
});

// Binary poly ops promote both coefficient dtypes (Promote<A,B>). Pairs list the
// WIDER dtype second on purpose — these ops used to return the first operand's
// dtype, which only slips past a widest-first pair ordering.
const POLY_PAIRS: [DType, DType][] = [
  ['int32', 'float32'],
  ['int8', 'int16'],
  ['float32', 'float64'],
  ['float16', 'int8'],
  ['int64', 'int64'],
];
const polyOps = { polyadd: np.polyadd, polysub: np.polysub, polymul: np.polymul };

describe('poly binary op dtype matches Promote<A,B>', () => {
  for (const [name, fn] of Object.entries(polyOps)) {
    describe(name, () => {
      it.each(POLY_PAIRS)('%s,%s', (a, b) => {
        expect(dtypeOf(fn(np.array([1, 2, 3], a), np.array([1, 2, 3], b)))).toBe(
          promoteDTypes(a, b),
        );
      });
    });
  }
});
