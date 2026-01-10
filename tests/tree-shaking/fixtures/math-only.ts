/**
 * Tree-shaking test fixture: Math operations only
 * Expected: Should include math ops but not FFT, random, linalg, IO
 */
import {
  array,
  sin,
  cos,
  exp,
  log,
  sqrt,
  abs,
  maximum,
  minimum,
  clip
} from '../../../src/index';

const a = array([1, 2, 3, 4, 5]);
const b = sin(a);
const c = cos(a);
const d = exp(a);
const e = log(a);
const f = sqrt(a);
const g = abs(array([-1, -2, 3]));
const h = maximum(a, array([3, 3, 3, 3, 3]));
const i = minimum(a, array([3, 3, 3, 3, 3]));
const j = clip(a, 2, 4);

console.log(b.shape, c.shape, d.shape, e.shape, f.shape, g.shape, h.shape, i.shape, j.shape);
export { b, c, d, e, f, g, h, i, j };
