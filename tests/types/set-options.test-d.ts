/**
 * Compile-time assertions for the options of the set functions, isclose and put_along_axis:
 * the default call keeps its array result, return_indices: true gives the tuple.
 *
 * Checked by `pnpm run typecheck:types`.
 */

import { Complex } from '../../src/common/complex';
import * as np from '../../src/full/index';
import type { NDArray } from '../../src/full/ndarray';

type Equal<A, B> =
  (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2 ? true : false;
type Expect<T extends true> = T;

const a = np.zeros([3], 'int32');
const b = np.zeros([3], 'float32');

// intersect1d: bare array unless return_indices is literally true.
const plain = np.intersect1d(a, b);
type _plain = Expect<Equal<typeof plain, NDArray<'float64'>>>;
const unique = np.intersect1d(a, b, true);
type _unique = Expect<Equal<typeof unique, NDArray<'float64'>>>;
const off = np.intersect1d(a, b, false, false);
type _off = Expect<Equal<typeof off, NDArray<'float64'>>>;
const tuple = np.intersect1d(a, b, false, true);
type _tuple = Expect<Equal<typeof tuple, [NDArray<'float64'>, NDArray<'int32'>, NDArray<'int32'>]>>;
const skipped = np.intersect1d(a, a, undefined, true);
type _skipped = Expect<
  Equal<typeof skipped, [NDArray<'int32'>, NDArray<'int32'>, NDArray<'int32'>]>
>;
declare const flag: boolean;
const either = np.intersect1d(a, a, false, flag);
type _either = Expect<
  Equal<typeof either, NDArray<'int32'> | [NDArray<'int32'>, NDArray<'int32'>, NDArray<'int32'>]>
>;

// invert and assume_unique keep the result types.
const inv = np.isin(a, b, false, true);
type _inv = Expect<Equal<typeof inv, NDArray<'bool'>>>;
const inv1d = np.in1d(a, b, true, true);
type _inv1d = Expect<Equal<typeof inv1d, NDArray<'bool'>>>;
const diff = np.setdiff1d(a, b, true);
type _diff = Expect<Equal<typeof diff, NDArray<'int32'>>>;
const xor = np.setxor1d(a, b, true);
type _xor = Expect<Equal<typeof xor, NDArray<'float64'>>>;

// isclose and allclose: equal_nan, array tolerances, and the method forms.
const close = np.isclose(a, b, 1e-5, 1e-8, true);
type _close = Expect<Equal<typeof close, NDArray<'bool'>>>;
const closeScalar = np.isclose(a, 2, np.ones([3]), 0, true);
type _closeScalar = Expect<Equal<typeof closeScalar, NDArray<'bool'>>>;
const method = a.isclose(b, 1e-5, np.ones([3]), true);
type _method = Expect<Equal<typeof method, NDArray<'bool'>>>;
const all: boolean = np.allclose(a, b, 1e-5, 1e-8, true);
const allMethod: boolean = a.allclose(2, 1e-5, 1e-8, true);

// put_along_axis accepts a scalar of each kind and an array.
np.put_along_axis(a, np.zeros([3, 1], 'int32'), 1, 1);
np.put_along_axis(a, np.zeros([3, 1], 'int32'), 1n, 1);
np.put_along_axis(a, np.zeros([3, 1], 'int32'), new Complex(1, 2), 1);
np.put_along_axis(a, np.zeros([3, 1], 'int32'), np.ones([3, 1]), 1);

export type {
  _close,
  _closeScalar,
  _diff,
  _either,
  _inv,
  _inv1d,
  _method,
  _off,
  _plain,
  _skipped,
  _tuple,
  _unique,
  _xor,
};
export { all, allMethod };
