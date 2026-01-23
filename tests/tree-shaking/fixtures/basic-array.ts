/**
 * Tree-shaking test fixture: Basic array creation functions
 * Expected: Should include array creation but not FFT, random, linalg
 */
import { zeros, ones, array, arange, linspace, reshape } from 'numpy-ts';

const a = zeros([2, 3]);
const b = ones([3, 2]);
const c = array([
  [1, 2],
  [3, 4],
]);
const d = arange(0, 10, 2);
const e = linspace(0, 1, 5);
const f = reshape(a, [6]);

console.log(a.shape, b.shape, c.shape, d.shape, e.shape, f.shape);
export { a, b, c, d, e, f };
