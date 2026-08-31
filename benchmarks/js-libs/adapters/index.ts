/** Competitor adapter registry. numpy-ts is the reference (handled separately). */

import type { JsLibAdapter } from '../lib/types';
import { jaxjs } from './jaxjs';
import { mathjsAdapter } from './mathjs';
import { mlmatrix } from './mlmatrix';
import { numericAdapter } from './numeric';
import { numjsAdapter } from './numjs';
import { pyodideNumpy } from './pyodide';
import { stdlib } from './stdlib';
import { tfjs } from './tfjs';

export const ADAPTERS: JsLibAdapter[] = [
  pyodideNumpy,
  jaxjs,
  tfjs,
  mathjsAdapter,
  numericAdapter,
  mlmatrix,
  numjsAdapter,
  stdlib,
];
