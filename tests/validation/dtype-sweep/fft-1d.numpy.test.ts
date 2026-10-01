/**
 * DType Sweep: 1D FFT functions + utilities.
 * Tests across ALL dtypes, validated against NumPy.
 * Uses batched oracle — all Python computations run in a single subprocess.
 */
import { beforeAll, describe, it } from 'vitest';
import * as np from '../../../src';
import type { NumPyResult } from '../numpy-oracle';
import {
  ALL_DTYPES,
  asDtypeData,
  checkNumPyAvailable,
  expectBothReject,
  expectComplexFixture,
  expectMatchPre,
  isComplex,
  npDtype,
  pyArrayCast,
  runNumPyBatch,
} from './_helpers';

const { array } = np;

/**
 * The one definition of the input for a dtype, shared by the oracle snippets and
 * the test bodies. Two independent copies would let the two sides drift apart and
 * compare different inputs.
 */
function fftData(dtype: string) {
  return asDtypeData(dtype === 'bool' ? [1, 0, 1, 0] : [1, 2, 3, 4], dtype);
}

// Pre-computed oracle results — filled in beforeAll
let oracle: Map<string, NumPyResult & { error?: string }>;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');

  const snippets: Record<string, string> = {};

  for (const dtype of ALL_DTYPES) {
    const data = fftData(dtype);
    // FFT results are complex → always cast to np.complex128 for value comparison
    // fftshift/ifftshift preserve dtype → use pyArrayCast
    const ac = pyArrayCast(dtype);

    snippets[`fft_${dtype}`] = `
_result_orig = np.fft.fft(np.array(${data.py}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.complex128)`;

    snippets[`ifft_${dtype}`] = `
_result_orig = np.fft.ifft(np.fft.fft(np.array(${data.py}, dtype=${npDtype(dtype)})))
result = _result_orig.astype(np.complex128)`;

    snippets[`rfft_${dtype}`] = `
_result_orig = np.fft.rfft(np.array(${data.py}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.complex128)`;

    snippets[`irfft_${dtype}`] =
      `result = np.fft.irfft(np.fft.rfft(np.array(${data.py}, dtype=${npDtype(dtype)})))`;

    snippets[`hfft_${dtype}`] =
      `result = np.fft.hfft(np.array(${data.py}, dtype=${npDtype(dtype)}))`;

    snippets[`ihfft_${dtype}`] = `
_result_orig = np.fft.ihfft(np.array(${data.py}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.complex128)`;

    snippets[`fftshift_${dtype}`] = `
_result_orig = np.fft.fftshift(np.array(${data.py}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(${ac})`;

    snippets[`ifftshift_${dtype}`] = `
_result_orig = np.fft.ifftshift(np.array(${data.py}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(${ac})`;
  }

  // fftfreq/rfftfreq are dtype-independent, just need one each
  snippets['fftfreq'] = `result = np.fft.fftfreq(4)`;
  snippets['rfftfreq'] = `result = np.fft.rfftfreq(4)`;

  oracle = runNumPyBatch(snippets);
});

describe('DType Sweep: FFT 1D', () => {
  for (const dtype of ALL_DTYPES) {
    const data = fftData(dtype);
    const tol = dtype === 'float32' || dtype === 'complex64' ? 1e-2 : 1e-4;

    const makeInput = (label: string) => {
      const a = array(data.js as never, dtype);
      expectComplexFixture(a, dtype, label);
      return a;
    };

    it(`fft.fft ${dtype}`, () => {
      const jsResult = np.fft.fft(makeInput(`fft ${dtype}`));
      expectMatchPre(jsResult, oracle.get(`fft_${dtype}`)!, { rtol: tol });
    });

    it(`fft.ifft ${dtype}`, () => {
      const fftResult = np.fft.fft(makeInput(`ifft ${dtype}`));
      const jsResult = np.fft.ifft(fftResult);
      expectMatchPre(jsResult, oracle.get(`ifft_${dtype}`)!, { rtol: tol });
    });

    it(`fft.rfft ${dtype}`, () => {
      const a = makeInput(`rfft ${dtype}`);
      const pyCode = `
_result_orig = np.fft.rfft(np.array(${data.py}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.complex128)`;
      if (isComplex(dtype)) {
        {
          const _r = expectBothReject(
            'rfft expects real-valued input, not complex',
            () => np.fft.rfft(a),
            pyCode,
          );
          if (_r === 'both-reject') return;
        }
      }
      const jsResult = np.fft.rfft(a);
      expectMatchPre(jsResult, oracle.get(`rfft_${dtype}`)!, { rtol: tol });
    });

    it(`fft.irfft ${dtype}`, () => {
      const a = makeInput(`irfft ${dtype}`);
      const pyCode = `result = np.fft.irfft(np.fft.rfft(np.array(${data.py}, dtype=${npDtype(dtype)})))`;
      if (isComplex(dtype)) {
        {
          const _r = expectBothReject(
            'rfft(complex) not supported, so irfft round-trip fails',
            () => {
              const r = np.fft.rfft(a);
              np.fft.irfft(r);
            },
            pyCode,
          );
          if (_r === 'both-reject') return;
        }
      }
      const rfftResult = np.fft.rfft(a);
      const jsResult = np.fft.irfft(rfftResult);
      expectMatchPre(jsResult, oracle.get(`irfft_${dtype}`)!, { rtol: tol });
    });

    it(`fft.hfft ${dtype}`, () => {
      const jsResult = np.fft.hfft(makeInput(`hfft ${dtype}`));
      expectMatchPre(jsResult, oracle.get(`hfft_${dtype}`)!, { rtol: tol });
    });

    it(`fft.ihfft ${dtype}`, () => {
      const a = makeInput(`ihfft ${dtype}`);
      const pyCode = `
_result_orig = np.fft.ihfft(np.array(${data.py}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.complex128)`;
      if (isComplex(dtype)) {
        {
          const _r = expectBothReject(
            'ihfft expects real-valued input',
            () => np.fft.ihfft(a),
            pyCode,
          );
          if (_r === 'both-reject') return;
        }
      }
      const jsResult = np.fft.ihfft(a);
      expectMatchPre(jsResult, oracle.get(`ihfft_${dtype}`)!, { rtol: tol });
    });

    it(`fft.fftfreq ${dtype}`, () => {
      const jsResult = np.fft.fftfreq(4);
      expectMatchPre(jsResult, oracle.get('fftfreq')!, { rtol: 1e-10 });
    });

    it(`fft.rfftfreq ${dtype}`, () => {
      const jsResult = np.fft.rfftfreq(4);
      expectMatchPre(jsResult, oracle.get('rfftfreq')!, { rtol: 1e-10 });
    });

    it(`fft.fftshift ${dtype}`, () => {
      const jsResult = np.fft.fftshift(makeInput(`fftshift ${dtype}`));
      expectMatchPre(jsResult, oracle.get(`fftshift_${dtype}`)!);
    });

    it(`fft.ifftshift ${dtype}`, () => {
      const jsResult = np.fft.ifftshift(makeInput(`ifftshift ${dtype}`));
      expectMatchPre(jsResult, oracle.get(`ifftshift_${dtype}`)!);
    });
  }
});
