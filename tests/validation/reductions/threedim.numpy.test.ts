/**
 * NumPy validation: 3D array reductions
 */

import { describe, it, expect, beforeAll, afterEach } from 'vitest';
import * as np from '../../../src/full/index';
import { wasmConfig } from '../../../src';
import { runNumPy, arraysClose, checkNumPyAvailable, getPythonInfo } from '../numpy-oracle';
import { WASM_MODES } from './_helpers';

for (const mode of WASM_MODES) {
  describe(`NumPy Validation: 3D Reductions [${mode.name}]`, () => {
    beforeAll(() => {
      wasmConfig.thresholdMultiplier = mode.multiplier;
      if (!checkNumPyAvailable()) {
        throw new Error('Python NumPy not available');
      }
    });

    afterEach(() => {
      wasmConfig.thresholdMultiplier = mode.multiplier;
    });

    const data3d = [
      [
        [1, 2],
        [3, 4],
      ],
      [
        [5, 6],
        [7, 8],
      ],
    ];

    for (const op of ['sum', 'mean', 'max', 'min'] as const) {
      for (const axis of [0, 1, 2] as const) {
        it(`${op} axis=${axis} float64`, () => {
          const a = np.array(data3d);
          let jsResult: any;
          if (op === 'sum') jsResult = a.sum(axis);
          else if (op === 'mean') jsResult = np.mean(a, axis);
          else if (op === 'max') jsResult = np.max(a, axis);
          else if (op === 'min') jsResult = np.min(a, axis);

          const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data3d)}, dtype=np.float64)
result = np.${op}(a, axis=${axis})
`);
          expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
        });
      }
    }
  });
}
