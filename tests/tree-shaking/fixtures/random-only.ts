/**
 * Tree-shaking test fixture: Random operations only
 * Expected: Should include random but not FFT, linalg (except what random needs), IO
 */
import { random } from 'numpy-ts';

random.seed(42);

// These return NDArray when given shape argument
const a = random.random([3, 3]);
const b = random.randn(5);
const c = random.randint(0, 10, [4]);
const d = random.normal(0, 1, [2, 2]);
const e = random.uniform(0, 1, [3]);

// @ts-expect-error - random functions return NDArray when shape is provided (union type includes number for scalar case)
console.log(a.shape, b.shape, c.shape, d.shape, e.shape);
export { a, b, c, d, e };
