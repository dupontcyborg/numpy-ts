/**
 * Tree-shaking test fixture: FFT operations only
 * Expected: Should include FFT but not random, full linalg, IO
 */
import { array, fft } from '../../../src/index';

const a = array([1, 2, 3, 4, 5, 6, 7, 8]);

const b = fft.fft(a);
const c = fft.ifft(b);
const d = fft.rfft(a);
const e = fft.fftfreq(8);

console.log(b.shape, c.shape, d.shape, e.shape);
export { b, c, d, e };
