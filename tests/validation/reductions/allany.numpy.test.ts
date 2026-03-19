/**
 * NumPy validation: all, any reductions
 */

import { describe, it } from 'vitest';
import { WASM_MODES, ALL_DTYPES, compareReduction, setupWasmMode } from './_helpers';

for (const mode of WASM_MODES) {
  describe(`NumPy Validation: all/any [${mode.name}]`, () => {
    setupWasmMode(mode);

    for (const op of ['all', 'any'] as const) {
      describe(`${op}()`, () => {
        const dataWithZero = [
          [0, 1, 2],
          [3, 4, 5],
        ];
        for (const dtype of ALL_DTYPES) {
          for (const axis of [undefined, 0, 1] as const) {
            const axisLabel = axis !== undefined ? `axis=${axis}` : 'no axis';
            it(`${dtype} ${axisLabel}`, () => {
              compareReduction(op, dataWithZero, dtype, axis);
            });
          }
        }
      });
    }
  });
}
