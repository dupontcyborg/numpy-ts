/**
 * Tree-shaking test fixture: Full import (baseline)
 * Expected: Largest bundle - includes everything
 */
import * as np from 'numpy-ts';

// Use various functions to ensure they're all included
const a = np.zeros([2, 3]);
const b = np.ones([3, 2]);
const c = np.array([
  [1, 2],
  [3, 4],
]);
const d = np.sin(c);
const e = np.linalg.inv(c);
const f = np.fft.fft(np.array([1, 2, 3, 4]));
np.random.seed(42);
const g = np.random.random([2, 2]);
// @ts-expect-error - serializeNpy accepts NDArray but zeros() returns NDArrayCore (compatible at runtime)
const h = np.serializeNpy(a);

// Access shape via type guard for g (random returns union type)
const gShape = typeof g === 'object' && 'shape' in g ? g.shape : null;
console.log(a.shape, b.shape, c.shape, d.shape, e.shape, f.shape, gShape, h.byteLength);
export { a, b, c, d, e, f, g, h };
