/**
 * NumPy validation: core reductions (sum, mean, max, min, prod)
 */

import { describe, it } from 'vitest';
import { ALL_DTYPES, compareReduction, SMALL_DATA, setupWasmMode, WASM_MODES } from './_helpers';

for (const mode of WASM_MODES) {
  describe(`NumPy Validation: Core Reductions [${mode.name}]`, () => {
    setupWasmMode(mode);

    for (const op of ['sum', 'mean', 'max', 'min', 'prod'] as const) {
      describe(`${op}()`, () => {
        for (const dtype of ALL_DTYPES) {
          for (const axis of [undefined, 0, 1] as const) {
            const axisLabel = axis !== undefined ? `axis=${axis}` : 'no axis';
            it(`${dtype} ${axisLabel}`, () => {
              compareReduction(op, SMALL_DATA, dtype, axis);
            });
          }
        }
      });
    }
  });
}
