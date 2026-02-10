/**
 * Tree-shaking test fixture: IO operations only
 * Expected: Should include IO parsing but not FFT, random, full linalg
 */
import { parseNpy, serializeNpy, array, zeros } from 'numpy-ts';

const a = array([
  [1, 2],
  [3, 4],
]);
const serialized = serializeNpy(a);
const z = zeros([2, 2]);

// parseNpy needs bytes input - just reference to ensure it's included
console.log(typeof parseNpy, serialized.byteLength, z.shape);
export { serialized, parseNpy };
