/**
 * Tree-shaking test fixture: Linear algebra from standalone entry point
 * Expected: Small bundle with only linalg operations
 */
import { array, dot, transpose, linalg } from 'numpy-ts/core';

const a = array([
  [1, 2],
  [3, 4],
]);
const b = array([
  [5, 6],
  [7, 8],
]);

const c = dot(a, b);
const d = transpose(a);
const e = linalg.inv(a);
const f = linalg.det(a); // Returns number
const g = linalg.norm(a); // Returns number

// c returns NDArrayCore | number - use type guard for safe access
const cShape = typeof c === 'object' && 'shape' in c ? c.shape : null;
console.log(cShape, d.shape, e.shape, f, g);
export { c, d, e, f, g };
