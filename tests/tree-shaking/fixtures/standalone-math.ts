/**
 * Tree-shaking test fixture: Math operations from standalone entry point
 * Expected: Small bundle with only math operations
 */
import { array, sin, cos, sqrt, add, multiply } from 'numpy-ts/core';

const a = array([1, 2, 3, 4]);
const b = sin(a);
const c = cos(a);
const d = sqrt(a);
const e = add(a, multiply(b, c));

console.log(b.shape, c.shape, d.shape, e.shape);
export { b, c, d, e };
