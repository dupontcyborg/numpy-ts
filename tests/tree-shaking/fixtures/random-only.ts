/**
 * Tree-shaking test fixture: Random operations only
 * Expected: Should include random but not FFT, linalg (except what random needs), IO
 */
import { random } from '../../../src/index';

random.seed(42);

const a = random.random([3, 3]);
const b = random.randn(5);
const c = random.randint(0, 10, [4]);
const d = random.normal(0, 1, [2, 2]);
const e = random.uniform(0, 1, [3]);

console.log(a.shape, b.shape, c.shape, d.shape, e.shape);
export { a, b, c, d, e };
