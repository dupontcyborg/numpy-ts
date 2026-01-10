/**
 * Tree-shaking test fixture: Full import (baseline)
 * Expected: Largest bundle - includes everything
 */
import * as np from '../../../src/index';

// Use various functions to ensure they're all included
const a = np.zeros([2, 3]);
const b = np.ones([3, 2]);
const c = np.array([[1, 2], [3, 4]]);
const d = np.sin(c);
const e = np.linalg.inv(c);
const f = np.fft.fft(np.array([1, 2, 3, 4]));
np.random.seed(42);
const g = np.random.random([2, 2]);
const h = np.serializeNpy(a);

console.log(a.shape, b.shape, c.shape, d.shape, e.shape, f.shape, g.shape, h.byteLength);
export { a, b, c, d, e, f, g, h };
