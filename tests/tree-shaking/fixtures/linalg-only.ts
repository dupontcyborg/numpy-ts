/**
 * Tree-shaking test fixture: Linear algebra operations only
 * Expected: Should include linalg but not FFT, random, IO
 */
import { array, linalg, dot, transpose, NDArray } from 'numpy-ts';

const a = array([
  [1, 2],
  [3, 4],
]);
const b = array([
  [5, 6],
  [7, 8],
]);

const c = dot(a, b) as NDArray;
const d = transpose(a);
const e = linalg.inv(a);
const f = linalg.det(a); // Returns number
const g = linalg.norm(a); // Returns number

console.log(c.shape, d.shape, e.shape, f, g);
export { c, d, e, f, g };
