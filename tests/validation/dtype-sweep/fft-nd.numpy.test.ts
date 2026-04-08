/**
 * DType Sweep: 2D/nD FFT functions.
 * Tests across ALL dtypes, validated against NumPy.
 * Uses batched oracle — all Python computations run in a single subprocess.
 */
import { describe, it, beforeAll } from 'vitest';
import * as np from '../../../src';
import {
  ALL_DTYPES,
  runNumPyBatch,
  checkNumPyAvailable,
  npDtype,
  isComplex,
  expectBothReject,
  expectMatchPre,
} from './_helpers';
import type { NumPyResult } from '../numpy-oracle';

const { array } = np;

// Pre-computed oracle results — filled in beforeAll
let oracle: Map<string, NumPyResult & { error?: string }>;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');

  const snippets: Record<string, string> = {};

  for (const dtype of ALL_DTYPES) {
    const data2d =
      dtype === 'bool'
        ? [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
          ]
        : [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
          ];

    snippets[`fft2_${dtype}`] = `
_result_orig = np.fft.fft2(np.array(${JSON.stringify(data2d)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.complex128)`;

    snippets[`ifft2_${dtype}`] = `
_result_orig = np.fft.ifft2(np.fft.fft2(np.array(${JSON.stringify(data2d)}, dtype=${npDtype(dtype)})))
result = _result_orig.astype(np.complex128)`;

    snippets[`rfft2_${dtype}`] = `
_result_orig = np.fft.rfft2(np.array(${JSON.stringify(data2d)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.complex128)`;

    snippets[`irfft2_${dtype}`] =
      `result = np.fft.irfft2(np.fft.rfft2(np.array(${JSON.stringify(data2d)}, dtype=${npDtype(dtype)})))`;

    snippets[`fftn_${dtype}`] = `
_result_orig = np.fft.fftn(np.array(${JSON.stringify(data2d)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.complex128)`;

    snippets[`ifftn_${dtype}`] = `
_result_orig = np.fft.ifftn(np.fft.fftn(np.array(${JSON.stringify(data2d)}, dtype=${npDtype(dtype)})))
result = _result_orig.astype(np.complex128)`;

    snippets[`rfftn_${dtype}`] = `
_result_orig = np.fft.rfftn(np.array(${JSON.stringify(data2d)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.complex128)`;

    snippets[`irfftn_${dtype}`] =
      `result = np.fft.irfftn(np.fft.rfftn(np.array(${JSON.stringify(data2d)}, dtype=${npDtype(dtype)})))`;
  }

  oracle = runNumPyBatch(snippets);
});

describe('DType Sweep: FFT 2D/nD', () => {
  for (const dtype of ALL_DTYPES) {
    const data2d =
      dtype === 'bool'
        ? [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
          ]
        : [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
          ];
    const tol = dtype === 'float32' || dtype === 'complex64' ? 1e-2 : 1e-4;

    it(`fft.fft2 ${dtype}`, () => {
      const jsResult = np.fft.fft2(array(data2d, dtype));
      expectMatchPre(jsResult, oracle.get(`fft2_${dtype}`)!, { rtol: tol });
    });

    it(`fft.ifft2 ${dtype}`, () => {
      const fft2Result = np.fft.fft2(array(data2d, dtype));
      const jsResult = np.fft.ifft2(fft2Result);
      expectMatchPre(jsResult, oracle.get(`ifft2_${dtype}`)!, { rtol: tol });
    });

    it(`fft.rfft2 ${dtype}`, () => {
      const pyCode = `
_result_orig = np.fft.rfft2(np.array(${JSON.stringify(data2d)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.complex128)`;
      if (isComplex(dtype)) {
        {
          const _r = expectBothReject(
            'rfft2 expects real-valued input',
            () => np.fft.rfft2(array(data2d, dtype)),
            pyCode
          );
          if (_r === 'both-reject') return;
        }
      }
      const jsResult = np.fft.rfft2(array(data2d, dtype));
      expectMatchPre(jsResult, oracle.get(`rfft2_${dtype}`)!, { rtol: tol });
    });

    it(`fft.irfft2 ${dtype}`, () => {
      const pyCode = `result = np.fft.irfft2(np.fft.rfft2(np.array(${JSON.stringify(data2d)}, dtype=${npDtype(dtype)})))`;
      if (isComplex(dtype)) {
        {
          const _r = expectBothReject(
            'rfft2(complex) not supported',
            () => {
              const r = np.fft.rfft2(array(data2d, dtype));
              np.fft.irfft2(r);
            },
            pyCode
          );
          if (_r === 'both-reject') return;
        }
      }
      const rfft2Result = np.fft.rfft2(array(data2d, dtype));
      const jsResult = np.fft.irfft2(rfft2Result);
      expectMatchPre(jsResult, oracle.get(`irfft2_${dtype}`)!, { rtol: tol });
    });

    it(`fft.fftn ${dtype}`, () => {
      const jsResult = np.fft.fftn(array(data2d, dtype));
      expectMatchPre(jsResult, oracle.get(`fftn_${dtype}`)!, { rtol: tol });
    });

    it(`fft.ifftn ${dtype}`, () => {
      const fftnResult = np.fft.fftn(array(data2d, dtype));
      const jsResult = np.fft.ifftn(fftnResult);
      expectMatchPre(jsResult, oracle.get(`ifftn_${dtype}`)!, { rtol: tol });
    });

    it(`fft.rfftn ${dtype}`, () => {
      const pyCode = `
_result_orig = np.fft.rfftn(np.array(${JSON.stringify(data2d)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.complex128)`;
      if (isComplex(dtype)) {
        {
          const _r = expectBothReject(
            'rfftn expects real-valued input',
            () => np.fft.rfftn(array(data2d, dtype)),
            pyCode
          );
          if (_r === 'both-reject') return;
        }
      }
      const jsResult = np.fft.rfftn(array(data2d, dtype));
      expectMatchPre(jsResult, oracle.get(`rfftn_${dtype}`)!, { rtol: tol });
    });

    it(`fft.irfftn ${dtype}`, () => {
      const pyCode = `result = np.fft.irfftn(np.fft.rfftn(np.array(${JSON.stringify(data2d)}, dtype=${npDtype(dtype)})))`;
      if (isComplex(dtype)) {
        {
          const _r = expectBothReject(
            'rfftn(complex) not supported',
            () => {
              const r = np.fft.rfftn(array(data2d, dtype));
              np.fft.irfftn(r);
            },
            pyCode
          );
          if (_r === 'both-reject') return;
        }
      }
      const rfftnResult = np.fft.rfftn(array(data2d, dtype));
      const jsResult = np.fft.irfftn(rfftnResult);
      expectMatchPre(jsResult, oracle.get(`irfftn_${dtype}`)!, { rtol: tol });
    });
  }
});
