/**
 * NumPy validation: argmin, argmax reductions
 */

import { describe, it } from 'vitest';
import { WASM_MODES, ALL_DTYPES, SMALL_DATA, compareReduction, setupWasmMode } from './_helpers';

for (const mode of WASM_MODES) {
  describe(`NumPy Validation: argmin/argmax [${mode.name}]`, () => {
    setupWasmMode(mode);

    for (const op of ['argmin', 'argmax'] as const) {
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
