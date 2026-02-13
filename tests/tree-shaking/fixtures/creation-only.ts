/**
 * Tree-shaking test fixture: Creation module only (modular import)
 * Expected: Much smaller than full-import - should NOT include any ops modules
 *
 * This imports from the new modular creation module which doesn't
 * depend on arithmetic, linalg, fft, random, or other ops.
 */
import { zeros, ones, array, arange, linspace, eye } from 'numpy-ts';

const a = zeros([2, 3]);
const b = ones([3, 2]);
const c = array([
  [1, 2],
  [3, 4],
]);
const d = arange(0, 10, 2);
const e = linspace(0, 1, 5);
const f = eye(3);

console.log(a.shape, b.shape, c.shape, d.shape, e.shape, f.shape);
export { a, b, c, d, e, f };
