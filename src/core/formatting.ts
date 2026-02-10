/**
 * Formatting and printing functions
 *
 * Tree-shakeable standalone functions that wrap the underlying ops.
 */

import { NDArrayCore } from '../common/ndarray-core';
import { ArrayStorage } from '../common/storage';
import * as formattingOps from '../common/ops/formatting';

// Helper to convert NDArrayCore to ArrayStorage
function toStorage(a: NDArrayCore): ArrayStorage {
  return (a as unknown as { _storage: ArrayStorage })._storage;
}

// ============================================================
// Print Options
// ============================================================

export const set_printoptions = formattingOps.set_printoptions;
export const get_printoptions = formattingOps.get_printoptions;
export const printoptions = formattingOps.printoptions;

// ============================================================
// Number Formatting
// ============================================================

export const format_float_positional = formattingOps.format_float_positional;
export const format_float_scientific = formattingOps.format_float_scientific;
export const base_repr = formattingOps.base_repr;
export const binary_repr = formattingOps.binary_repr;

// ============================================================
// Array String Representation
// ============================================================

export function array2string(
  a: NDArrayCore,
  options?: {
    max_line_width?: number;
    precision?: number;
    suppress_small?: boolean;
    separator?: string;
    prefix?: string;
    suffix?: string;
    threshold?: number;
    edgeitems?: number;
    sign?: ' ' | '+' | '-';
    floatmode?: 'fixed' | 'unique' | 'maxprec' | 'maxprec_equal';
  }
): string {
  return formattingOps.array2string(
    toStorage(a),
    options?.max_line_width ?? null,
    options?.precision ?? null,
    options?.suppress_small ?? null,
    options?.separator ?? ' ',
    options?.prefix ?? '',
    options?.suffix ?? '',
    options?.threshold ?? null,
    options?.edgeitems ?? null
  );
}

export function array_repr(
  a: NDArrayCore,
  max_line_width?: number,
  precision?: number,
  suppress_small?: boolean
): string {
  return formattingOps.array_repr(toStorage(a), max_line_width, precision, suppress_small);
}

export function array_str(a: NDArrayCore, max_line_width?: number, precision?: number): string {
  return formattingOps.array_str(toStorage(a), max_line_width, precision);
}
