/**
 * NumPy validation: nan* reductions (nansum, nanmean, nanmin, nanmax)
 */

import { describe, it } from 'vitest';
import {
  WASM_MODES,
  FLOAT_DTYPES,
  INT_DTYPES,
  SMALL_DATA,
  compareReduction,
  setupWasmMode,
} from './_helpers';

for (const mode of WASM_MODES) {
  describe(`NumPy Validation: nan* Reductions [${mode.name}]`, () => {
    setupWasmMode(mode);

    for (const op of ['nansum', 'nanmean', 'nanmin', 'nanmax'] as const) {
      describe(`${op}()`, () => {
        for (const dtype of FLOAT_DTYPES) {
          for (const axis of [undefined, 0, 1] as const) {
            const axisLabel = axis !== undefined ? `axis=${axis}` : 'no axis';
            it(`${dtype} ${axisLabel}`, () => {
              compareReduction(op, SMALL_DATA, dtype, axis, op, 1e-5);
            });
          }
        }
        for (const dtype of INT_DTYPES) {
          it(`${dtype} no axis (routes to non-nan)`, () => {
            compareReduction(op, SMALL_DATA, dtype, undefined, op);
          });
        }
      });
    }
  });
}
