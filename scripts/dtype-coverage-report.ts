#!/usr/bin/env npx tsx
/**
 * DType Coverage Report v2 — static analysis of test files.
 *
 * Parses tests/validation/dtype-sweep/*.numpy.test.ts to extract
 * which (function, dtype) pairs are tested, whether they're oracle-validated
 * (runNumPy present), and whether they're structural (// structural comment).
 *
 * The expected function list is derived from src/core/index.ts exports.
 * Every function is expected to be tested with ALL 13 dtypes.
 *
 * Usage:
 *   npx tsx scripts/dtype-coverage-report.ts            # full matrix
 *   npx tsx scripts/dtype-coverage-report.ts --missing   # gaps only
 *   npx tsx scripts/dtype-coverage-report.ts --summary   # summary only
 */

import { readFileSync, readdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { ALL_DTYPES } from '../tests/validation/dtype-sweep/_dtype-matrix';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const SWEEP_DIR = join(__dirname, '..', 'tests', 'validation', 'dtype-sweep');
const CORE_INDEX = join(__dirname, '..', 'src', 'core', 'index.ts');
const showMissing = process.argv.includes('--missing');
const showSummary = process.argv.includes('--summary');

/** Normalize function names from tests to match core/index.ts exports */
const FN_ALIASES: Record<string, string> = {
  var: 'variance',
};

function normalizeFnName(name: string): string {
  // Strip axis variants: "sum axis=0" → "sum"
  name = name.replace(/\s+axis=\d+.*$/, '');
  return FN_ALIASES[name] ?? name;
}

// ============================================================
// Extract function list from src/core/index.ts
// ============================================================

/** Functions/types that are NOT dtype-parametric and should be excluded */
const EXCLUDED = new Set([
  // Types and classes
  'DType', 'TypedArray', 'NDArrayCore', 'Complex', 'ArrayStorage',
  // Type utilities
  'can_cast', 'common_type', 'result_type', 'min_scalar_type', 'issubdtype',
  'typename', 'mintypecode', 'promote_types', 'isdtype',
  // Type checking / introspection (return bool/scalar, not dtype-parametric)
  'isscalar', 'iterable', 'isfortran', 'iscomplex', 'iscomplexobj',
  'isreal', 'isrealobj', 'real_if_close', 'isnat',
  // Formatting & printing
  'set_printoptions', 'get_printoptions', 'printoptions',
  'format_float_positional', 'format_float_scientific',
  'base_repr', 'binary_repr', 'array2string', 'array_repr', 'array_str',
  // Utility (shape/size introspection)
  'ndim', 'shape', 'size', 'item', 'tolist', 'tobytes', 'byteswap', 'view', 'tofile', 'fill',
  // Error handling
  'geterr', 'seterr',
  // IO types/errors
  'UnsupportedDTypeError', 'InvalidNpyError', 'SUPPORTED_DTYPES', 'DTYPE_TO_DESCR',
  'NpyHeader', 'NpyMetadata', 'NpyVersion',
  'NpzParseOptions', 'NpzParseResult', 'NpzArraysInput', 'NpzSerializeOptions',
  'ParseTxtOptions', 'SerializeTxtOptions',
  // IO functions (not dtype-parametric in the sweep sense)
  'parseNpy', 'parseNpyHeader', 'parseNpyData', 'serializeNpy',
  'parseNpz', 'parseNpzSync', 'loadNpz', 'loadNpzSync',
  'serializeNpz', 'serializeNpzSync',
  'parseTxt', 'genfromtxt', 'fromregex', 'serializeTxt',
  // Aliases (tracked under their canonical name)
  'abs', 'pow', 'true_divide', 'amax', 'amin', 'var_', 'var',
  'asin', 'acos', 'atan', 'atan2', 'asinh', 'acosh', 'atanh',
  'conjugate', 'row_stack', 'concat',
  'cumulative_sum', 'cumulative_prod', 'nancumsum', 'nancumprod',
  'bitwise_invert', 'bitwise_left_shift', 'bitwise_right_shift',
  'round',
  // Index utilities (not dtype-parametric)
  'indices', 'ix_', 'ravel_multi_index', 'unravel_index',
  'diag_indices', 'diag_indices_from',
  'tril_indices', 'tril_indices_from', 'triu_indices', 'triu_indices_from',
  'mask_indices',
  // Advanced utilities
  'apply_along_axis', 'apply_over_axes',
  'may_share_memory', 'shares_memory',
  'place', 'putmask', 'copyto',
  'iindex', 'bindex',
  'broadcast_shapes',
  // Creation from special sources (not dtype-sweep)
  'frombuffer', 'fromfunction', 'fromiter', 'fromstring', 'fromfile',
  'meshgrid', 'vander',
  // Niche creation
  'zeros_like', 'ones_like', 'empty_like', 'full_like',
  'copy', 'asanyarray', 'asarray_chkfinite', 'require',
  'tri', 'tril', 'triu',
  // Polynomial
  'poly', 'polyadd', 'polyder', 'polydiv', 'polyfit', 'polyint',
  'polymul', 'polysub', 'polyval', 'roots',
  // Einsum
  'einsum', 'einsum_path',
  // Misc
  'vdot', 'matrix_transpose', 'permute_dims',
  'rollaxis', 'block',
  'array_equal',
  // Unique variants (tracked under 'unique')
  'unique_all', 'unique_counts', 'unique_inverse', 'unique_values',
  // Histogram variants
  'histogram2d', 'histogramdd', 'histogram_bin_edges',
  // Misc extra
  'modf', 'unwrap',
  // Namespace objects (methods tracked separately as linalg.*, fft.*)
  'linalg', 'fft', 'random',
  // delete is alias for delete_
  'delete',
]);

function extractExportedFunctions(): string[] {
  const content = readFileSync(CORE_INDEX, 'utf-8');
  const fns: string[] = [];
  // Match exported identifiers from export { ... } blocks
  const exportBlocks = content.matchAll(/export\s*\{([^}]+)\}/g);
  for (const block of exportBlocks) {
    const items = block[1]!.split(',');
    for (const item of items) {
      const trimmed = item.trim();
      if (!trimmed || trimmed.startsWith('type ') || trimmed.startsWith('//')) continue;
      // Handle "foo as bar" — use the exported name
      const asM = trimmed.match(/(\w+)\s+as\s+(\w+)/);
      const name = asM ? asM[2]! : trimmed.split(/\s/)[0]!;
      if (!EXCLUDED.has(name) && /^[a-z_]\w*$/i.test(name)) {
        fns.push(name);
      }
    }
  }
  return [...new Set(fns)].sort();
}

// Also add linalg.* and fft.* namespace functions
function extractNamespaceFunctions(): string[] {
  const fns: string[] = [];

  // Parse namespace objects by matching method definitions: `  methodName: (`
  function parseNamespaceObject(content: string, nsName: string, exportName: string) {
    const nsRegex = new RegExp(`export\\s+const\\s+${exportName}\\s*=\\s*\\{([^]*?)\\n\\};`);
    const nsM = content.match(nsRegex);
    if (!nsM) return;
    // Match lines like: `  methodName: (` or `  methodName: function`
    const methodRegex = /^\s{2}(\w+)\s*:\s*\(/gm;
    let m;
    while ((m = methodRegex.exec(nsM[1]!)) !== null) {
      fns.push(`${nsName}.${m[1]}`);
    }
  }

  // linalg namespace from src/core/linalg.ts
  const linalgPath = join(__dirname, '..', 'src', 'core', 'linalg.ts');
  try {
    parseNamespaceObject(readFileSync(linalgPath, 'utf-8'), 'linalg', 'linalg');
  } catch {}

  // fft namespace from src/core/index.ts
  const content = readFileSync(CORE_INDEX, 'utf-8');
  parseNamespaceObject(content, 'fft', 'fft');

  return fns;
}

// ============================================================
// Dtype set resolution — maps variable names to arrays
// ============================================================

// Known dtype set names that appear in test files (for variable resolution)
const ALL = [...ALL_DTYPES];
const FLOAT = ['float64', 'float32'];
const REAL = ['float64', 'float32', 'int64', 'uint64', 'int32', 'uint32', 'int16', 'uint16', 'int8', 'uint8'];
const NUMERIC = ['float64', 'float32', 'complex128', 'complex64', 'int64', 'uint64', 'int32', 'uint32', 'int16', 'uint16', 'int8', 'uint8'];
const FLOAT_COMPLEX = ['float64', 'float32', 'complex128', 'complex64'];
const INTEGER = ['int64', 'uint64', 'int32', 'uint32', 'int16', 'uint16', 'int8', 'uint8'];
const BITWISE = [...INTEGER, 'bool'];

const KNOWN_SETS: Record<string, readonly string[]> = {
  ALL_DTYPES: ALL,
  ALL, FLOAT, REAL, NUMERIC, FLOAT_COMPLEX, INTEGER, BITWISE,
  FLOAT64_ONLY: ['float64'],
  COMPLEX: ['complex128', 'complex64'],
  'SETS.ALL': ALL,
  'SETS.FLOAT': FLOAT,
  'SETS.REAL': REAL,
  'SETS.NUMERIC': NUMERIC,
  'SETS.FLOAT_COMPLEX': FLOAT_COMPLEX,
  'SETS.INTEGER': INTEGER,
  'SETS.BITWISE': BITWISE,
  'SETS.FLOAT64_ONLY': ['float64'],
};

type CoverageEntry = {
  fn: string;
  dtype: string;
  oracle: boolean;
  structural: boolean;
  file: string;
};

// ============================================================
// Parser
// ============================================================

function resolveLoopDtypes(line: string, localVars: Map<string, readonly string[]>): readonly string[] | null {
  const m = line.match(/for\s*\(\s*const\s+\w+\s+of\s+(.+?)\s*\)/);
  if (!m) return null;
  let expr = m[1]!.trim().replace(/[{)]\s*$/, '').trim();

  if (localVars.has(expr)) return localVars.get(expr)!;
  if (KNOWN_SETS[expr]) return KNOWN_SETS[expr]!;

  // Filter pattern: REAL.filter(d => d !== 'int64' && d !== 'uint64')
  const filterM = expr.match(/^(\w+)\.filter\(/);
  if (filterM) {
    const base = localVars.get(filterM[1]!) ?? KNOWN_SETS[filterM[1]!];
    if (base) {
      const excludeMatches = [...expr.matchAll(/!==\s*'(\w+)'/g)];
      if (excludeMatches.length > 0) {
        const excludes = new Set(excludeMatches.map(m => m[1]!));
        return base.filter(d => !excludes.has(d));
      }
      return base;
    }
  }

  // Inline array: ['float64', 'float32']
  const arrM = expr.match(/^\[([^\]]+)\]/);
  if (arrM) {
    return arrM[1]!.split(',').map(s => s.trim().replace(/['"]/g, '')).filter(Boolean);
  }

  return null;
}

function resolveLocalVars(lines: string[]): Map<string, readonly string[]> {
  const vars = new Map<string, readonly string[]>();
  for (const line of lines) {
    const m = line.match(/const\s+(\w+)\s*=\s*(SETS\.\w+|[\w.]+)\s*;/);
    if (m) {
      const name = m[1]!;
      const val = m[2]!;
      if (KNOWN_SETS[val]) vars.set(name, KNOWN_SETS[val]!);
    }
    const arrM = line.match(/const\s+(\w+)\s*=\s*\[([^\]]+)\]\s*(?:as\s+const)?\s*;/);
    if (arrM) {
      const name = arrM[1]!;
      const items = arrM[2]!.split(',').map(s => s.trim().replace(/['"]/g, '')).filter(Boolean);
      if (items.length > 0 && items.every(s => /^[a-z]\w+$/i.test(s))) {
        vars.set(name, items);
      }
    }
    const filterM = line.match(/const\s+(\w+)\s*=\s*(\w+)\.filter\((.+)\)/);
    if (filterM) {
      const name = filterM[1]!;
      const base = vars.get(filterM[2]!) ?? KNOWN_SETS[filterM[2]!];
      if (base) {
        const excludeMatches = [...filterM[3]!.matchAll(/!==\s*'(\w+)'/g)];
        if (excludeMatches.length > 0) {
          const excludes = new Set(excludeMatches.map(m => m[1]!));
          vars.set(name, base.filter(d => !excludes.has(d)));
        } else {
          vars.set(name, [...base]);
        }
      }
    }
  }
  return vars;
}

function parseTestFile(filePath: string): CoverageEntry[] {
  const content = readFileSync(filePath, 'utf-8');
  const lines = content.split('\n');
  const entries: CoverageEntry[] = [];
  const localVars = resolveLocalVars(lines);
  const fileName = filePath.split('/').pop()!;

  type ForLoop = { varName: string; dtypes: readonly string[] };

  const forLoopStack: ForLoop[] = [];
  const describeStack: { name: string }[] = [];
  let braceDepth = 0;
  const forLoopBraceDepth: number[] = [];
  const describeBraceDepth: number[] = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!;
    const trimmed = line.trim();

    // Track brace depth
    for (const ch of line) {
      if (ch === '{') braceDepth++;
      else if (ch === '}') {
        braceDepth--;
        while (forLoopBraceDepth.length > 0 && forLoopBraceDepth[forLoopBraceDepth.length - 1]! >= braceDepth) {
          forLoopBraceDepth.pop();
          forLoopStack.pop();
        }
        while (describeBraceDepth.length > 0 && describeBraceDepth[describeBraceDepth.length - 1]! >= braceDepth) {
          describeBraceDepth.pop();
          describeStack.pop();
        }
      }
    }

    // Match for-loop
    const forM = trimmed.match(/for\s*\(\s*const\s+(\w+)\s+of\s+/);
    if (forM) {
      const dtypes = resolveLoopDtypes(trimmed, localVars);
      if (dtypes) {
        forLoopStack.push({ varName: forM[1]!, dtypes });
        forLoopBraceDepth.push(braceDepth - 1);
      }
    }

    // Match describe block
    const descM = trimmed.match(/describe\s*\(\s*(?:`([^`]*)`|'([^']*)'|"([^"]*)")\s*,/);
    if (descM) {
      const name = (descM[1] ?? descM[2] ?? descM[3])!;
      describeStack.push({ name });
      describeBraceDepth.push(braceDepth - 1);
    }
    const descVarM = trimmed.match(/describe\s*\(\s*(\w+)\s*,/);
    if (descVarM && !descM) {
      describeStack.push({ name: `$\{${descVarM[1]}}` });
      describeBraceDepth.push(braceDepth - 1);
    }

    // Match it() block
    const itM = trimmed.match(/it\s*\(\s*(?:`([^`]*)`|'([^']*)'|"([^"]*)")\s*[,)]/);
    if (!itM) continue;

    const isStructural = trimmed.includes('// structural');
    const testName = (itM[1] ?? itM[2] ?? itM[3])!;

    // Scan the test body for runNumPy
    let hasOracle = false;
    let bodyBrace = braceDepth;
    for (let j = i; j < Math.min(i + 30, lines.length); j++) {
      if (lines[j]!.includes('runNumPy(') || lines[j]!.includes('runNumPy`')) {
        hasOracle = true;
        break;
      }
      if (j > i) {
        for (const ch of lines[j]!) {
          if (ch === '{') bodyBrace++;
          else if (ch === '}') bodyBrace--;
        }
        if (bodyBrace < braceDepth) break;
      }
    }

    // Resolve function name and dtype
    const allDtypeSet = new Set(ALL_DTYPES);

    // Pattern: `functionName ${dtype}` or `functionName extra ${dtype}`
    const dtypeVarM = testName.match(/\$\{(\w+)\}/);
    if (dtypeVarM) {
      const varName = dtypeVarM[1]!;
      const loop = forLoopStack.find(l => l.varName === varName);
      const fnPart = testName.replace(/\s*\$\{.*?\}\s*$/, '').trim();
      let fnName = fnPart || null;

      if (!fnName) {
        for (let d = describeStack.length - 1; d >= 0; d--) {
          const desc = describeStack[d]!;
          if (!desc.name.startsWith('DType Sweep') && !desc.name.startsWith('$')) {
            fnName = desc.name;
            break;
          }
        }
      }

      if (loop && fnName) {
        for (const dtype of loop.dtypes) {
          entries.push({ fn: fnName, dtype, oracle: hasOracle, structural: isStructural, file: fileName });
        }
      }
      continue;
    }

    // Test name IS a dtype
    if (allDtypeSet.has(testName)) {
      let fnName: string | null = null;
      for (let d = describeStack.length - 1; d >= 0; d--) {
        const desc = describeStack[d]!;
        if (!desc.name.startsWith('DType Sweep') && !desc.name.startsWith('$')) {
          fnName = desc.name;
          break;
        }
      }
      if (fnName) {
        entries.push({ fn: fnName, dtype: testName, oracle: hasOracle, structural: isStructural, file: fileName });
      }
      continue;
    }

    // Test name is function name only — dtype from parent for-loop
    const dtypeLoop = forLoopStack.find(l => l.varName === 'dtype');
    if (dtypeLoop) {
      for (const dtype of dtypeLoop.dtypes) {
        entries.push({ fn: testName, dtype, oracle: hasOracle, structural: isStructural, file: fileName });
      }
      continue;
    }

    // Test name: `functionName literal_dtype`
    const spaceM = testName.match(/^(.+?)\s+([\w]+)$/);
    if (spaceM && allDtypeSet.has(spaceM[2]!)) {
      entries.push({ fn: spaceM[1]!, dtype: spaceM[2]!, oracle: hasOracle, structural: isStructural, file: fileName });
      continue;
    }
  }

  return entries;
}

// Handle the nested describe + ops array pattern
function parseOpsPattern(filePath: string): CoverageEntry[] {
  const content = readFileSync(filePath, 'utf-8');
  const lines = content.split('\n');
  const localVars = resolveLocalVars(lines);
  const fileName = filePath.split('/').pop()!;
  const entries: CoverageEntry[] = [];

  const opsEntries: { name: string; dtypes: readonly string[] }[] = [];

  // Find ops arrays: const ops = [{ name: 'sum', fn: np.sum }, ...]
  // May or may not have a `dtypes:` field per entry
  const opsArrays: { names: string[]; startLine: number; endLine: number }[] = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!.trim();
    // Detect start of ops array
    if (line.match(/const\s+\w+\s*:\s*\{/) && line.includes('name:')) {
      // Multi-line ops array — collect all names
      const names: string[] = [];
      const startLine = i;
      for (let j = i; j < lines.length; j++) {
        const nameM = lines[j]!.match(/name:\s*'(\w[\w.]+)'/);
        if (nameM) names.push(nameM[1]!);
        if (lines[j]!.trim() === '];') {
          opsArrays.push({ names, startLine, endLine: j });
          break;
        }
      }
    }
  }

  // For each ops array, find the for-loop that iterates it and the dtype loop inside
  for (const { names, endLine } of opsArrays) {
    // Look for: for (const { name, fn } of ops) { describe(name, () => { for (const dtype of X) { it(...)
    for (let j = endLine + 1; j < Math.min(endLine + 10, lines.length); j++) {
      const forM = lines[j]!.match(/for\s*\(\s*const\s+\{/);
      if (forM) {
        // Find the dtype loop inside
        for (let k = j + 1; k < Math.min(j + 10, lines.length); k++) {
          const dtypeForM = lines[k]!.match(/for\s*\(\s*const\s+\w+\s+of\s+(\w[\w.]*)\s*\)/);
          if (dtypeForM) {
            const dtypes = localVars.get(dtypeForM[1]!) ?? KNOWN_SETS[dtypeForM[1]!];
            if (dtypes) {
              for (const name of names) {
                opsEntries.push({ name, dtypes });
              }
            }
            break;
          }
        }
        break;
      }
    }
  }

  // Also handle explicit dtypes field per entry
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!.trim();
    const nameM = line.match(/name:\s*'(\w[\w.]+)'/);
    if (nameM) {
      const name = nameM[1]!;
      const dtypesM = line.match(/dtypes:\s*(\w[\w.]*)/);
      let dtypes: readonly string[] | undefined;
      if (dtypesM) {
        dtypes = localVars.get(dtypesM[1]!) ?? KNOWN_SETS[dtypesM[1]!];
      }
      if (!dtypes) {
        for (let j = i + 1; j < Math.min(i + 3, lines.length); j++) {
          const dm = lines[j]!.match(/dtypes:\s*(\w[\w.]*)/);
          if (dm) {
            dtypes = localVars.get(dm[1]!) ?? KNOWN_SETS[dm[1]!];
            break;
          }
        }
      }
      if (dtypes) {
        opsEntries.push({ name, dtypes });
      }
    }
  }

  // compOps-style: const compOps = ['greater', ...]; for (name of compOps) describe(name)
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!.trim();
    const arrM = line.match(/const\s+(\w+)\s*=\s*\[([^\]]+)\]\s*;/);
    if (arrM) {
      const items = arrM[2]!.split(',').map(s => s.trim().replace(/['"]/g, '')).filter(Boolean);
      if (items.length > 0 && items.every(s => /^[a-z_]+$/i.test(s)) && !ALL_DTYPES.includes(items[0]!)) {
        for (let j = i + 1; j < Math.min(i + 10, lines.length); j++) {
          const forM = lines[j]!.match(/for\s*\(\s*const\s+(\w+)\s+of\s+(\w+)\s*\)/);
          if (forM && forM[2] === arrM[1]) {
            for (let k = j + 1; k < Math.min(j + 10, lines.length); k++) {
              const dtypeForM = lines[k]!.match(/for\s*\(\s*const\s+\w+\s+of\s+(\w[\w.]*)\s*\)/);
              if (dtypeForM) {
                const dtypes = localVars.get(dtypeForM[1]!) ?? KNOWN_SETS[dtypeForM[1]!];
                if (dtypes) {
                  for (const name of items) {
                    opsEntries.push({ name, dtypes });
                  }
                }
                break;
              }
            }
            break;
          }
        }
      }
    }
  }

  const hasOracleInFile = content.includes('runNumPy(') || content.includes('runNumPy`');

  for (const { name, dtypes } of opsEntries) {
    for (const dtype of dtypes) {
      entries.push({ fn: name, dtype, oracle: hasOracleInFile, structural: false, file: fileName });
    }
  }

  return entries;
}

// ============================================================
// Main
// ============================================================

// Build expected function list from src/core/index.ts
const coreFunctions = extractExportedFunctions();
const namespaceFunctions = extractNamespaceFunctions();
const allExpectedFns = [...coreFunctions, ...namespaceFunctions].sort();

// Build expected matrix: every function x ALL dtypes
const EXPECTED: Record<string, readonly string[]> = {};
for (const fn of allExpectedFns) {
  EXPECTED[fn] = ALL_DTYPES;
}

// Parse test files
const testFiles = readdirSync(SWEEP_DIR)
  .filter(f => f.endsWith('.numpy.test.ts'))
  .map(f => join(SWEEP_DIR, f));

const allEntries: CoverageEntry[] = [];
const seenPairs = new Set<string>();

for (const file of testFiles) {
  const content = readFileSync(file, 'utf-8');
  const hasOpsArray = /\bname:\s*'/.test(content);
  const hasCompOps = /const\s+\w+\s*=\s*\[['"](?:greater|add|sum)/.test(content);

  let entries: CoverageEntry[] = [];
  if (hasOpsArray || hasCompOps) {
    entries = parseOpsPattern(file);
  }

  const generalEntries = parseTestFile(file);

  for (const e of [...entries, ...generalEntries]) {
    const key = `${e.fn}|${e.dtype}`;
    if (!seenPairs.has(key)) {
      seenPairs.add(key);
      allEntries.push(e);
    } else if (e.oracle) {
      const existing = allEntries.find(x => x.fn === e.fn && x.dtype === e.dtype);
      if (existing) existing.oracle = true;
    }
  }
}

// Build coverage map
const coverageMap = new Map<string, Map<string, { oracle: boolean; structural: boolean }>>();

for (const e of allEntries) {
  e.fn = normalizeFnName(e.fn);
  if (!coverageMap.has(e.fn)) coverageMap.set(e.fn, new Map());
  const fnMap = coverageMap.get(e.fn)!;
  const existing = fnMap.get(e.dtype);
  fnMap.set(e.dtype, {
    oracle: e.oracle || (existing?.oracle ?? false),
    structural: e.structural || (existing?.structural ?? false),
  });
}

// ============================================================
// Report
// ============================================================

const abbrev = (d: string) =>
  d.replace('float', 'f').replace('complex', 'c').replace('uint', 'u').replace('int', 'i');

const colW = 5;
const nameW = 32;

let totalExpected = 0;
let totalOracle = 0;
let totalStructural = 0;
let totalMissing = 0;
const missingByDtype = new Map<string, string[]>();
const missingByFn = new Map<string, string[]>();

const fns = Object.keys(EXPECTED).sort();

for (const fn of fns) {
  const expected = new Set(EXPECTED[fn]!);
  const tested = coverageMap.get(fn) ?? new Map();

  for (const d of expected) {
    totalExpected++;
    const entry = tested.get(d);
    if (entry) {
      if (entry.oracle) totalOracle++;
      else totalStructural++;
    } else {
      totalMissing++;
      if (!missingByDtype.has(d)) missingByDtype.set(d, []);
      missingByDtype.get(d)!.push(fn);
      if (!missingByFn.has(fn)) missingByFn.set(fn, []);
      missingByFn.get(fn)!.push(d);
    }
  }
}

const totalTested = totalOracle + totalStructural;
const pct = totalExpected > 0 ? ((totalTested / totalExpected) * 100).toFixed(1) : '0.0';

console.log(`\nDTYPE COVERAGE: ${totalTested}/${totalExpected} expected pairs (${pct}%)`);
console.log(`  Oracle-validated: ${totalOracle}  |  Structural: ${totalStructural}  |  Missing: ${totalMissing}`);
console.log(`  Functions in core/index.ts: ${fns.length}  |  Covered in sweep: ${coverageMap.size}\n`);

if (showSummary) {
  for (const d of ALL_DTYPES) {
    const missing = missingByDtype.get(d) || [];
    const total = fns.length; // every function expected for every dtype
    const tested = total - missing.length;
    const dpct = ((tested / total) * 100).toFixed(0);
    const status = missing.length === 0
      ? '\x1b[32m100%\x1b[0m'
      : `\x1b[31m${missing.length} missing\x1b[0m`;
    console.log(`  ${d.padEnd(12)} ${dpct.padStart(4)}% (${tested}/${total}) ${status}`);
  }
  process.exit(0);
}

if (!showMissing) {
  const hdr = 'Function'.padEnd(nameW) + ' ' + ALL_DTYPES.map(d => abbrev(d).padStart(colW)).join(' ');
  console.log(hdr);
  console.log('-'.repeat(hdr.length));

  for (const fn of fns) {
    const tested = coverageMap.get(fn) ?? new Map();
    const cols = ALL_DTYPES.map(d => {
      const entry = tested.get(d);
      if (!entry) return `\x1b[31m${' '.repeat(colW - 1)}X\x1b[0m`;
      if (entry.oracle) return `\x1b[32m${' '.repeat(colW - 1)}V\x1b[0m`;
      return `\x1b[33m${' '.repeat(colW - 1)}O\x1b[0m`;
    });
    console.log(`${fn.padEnd(nameW)} ${cols.join(' ')}`);
  }

  console.log('-'.repeat(hdr.length));
  console.log(`\nV = oracle-validated   O = structural   X = missing`);
}

// Missing details
if (missingByFn.size > 0 && (showMissing || !showSummary)) {
  console.log(`\nMissing coverage by function (${missingByFn.size}):`);
  const sorted = [...missingByFn.entries()].sort((a, b) => b[1].length - a[1].length);
  for (const [fn, dtypes] of sorted.slice(0, 50)) {
    const count = dtypes.length === 13 ? 'ALL' : `${dtypes.length}/13`;
    console.log(`  ${fn} (${count}): ${dtypes.join(', ')}`);
  }
  if (sorted.length > 50) console.log(`  ... and ${sorted.length - 50} more functions`);
}

// Functions never tested
const untested = fns.filter(f => !coverageMap.has(f));
if (untested.length > 0) {
  console.log(`\nFunctions NEVER tested (${untested.length}):`);
  console.log(`  ${untested.join(', ')}`);
}

// Functions in tests but not in expected (extra coverage)
const unmatched = [...coverageMap.keys()].filter(f => !EXPECTED[f]).sort();
if (unmatched.length > 0) {
  console.log(`\nFunctions tested but NOT in core/index.ts exports (${unmatched.length}):`);
  console.log(`  ${unmatched.join(', ')}`);
}
