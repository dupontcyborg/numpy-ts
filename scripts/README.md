# Scripts

Utility scripts for numpy-ts development and maintenance.

## API Coverage Scripts

### `compare-api-coverage.py`

Main script for tracking API coverage against NumPy.

**Usage:**
```bash
# Update README + docs coverage page with current coverage
python scripts/compare-api-coverage.py

# Show detailed list of missing functions
python scripts/compare-api-coverage.py --verbose
python scripts/compare-api-coverage.py -v
```

**What it does:**
1. Runs `audit-numpy-api.py` to extract NumPy's API
2. Runs `audit-numpyts-api.ts` to extract numpy-ts's API
3. Compares them properly (member-vs-member, global-vs-global)
4. Updates `README.md` with accurate coverage statistics
5. Updates `docs/v1.0.x/guides/api-coverage.mdx` with category/missing details
6. With `--verbose`: Shows complete list of missing/extra functions

**When to run:**
- After implementing new functions
- Before releases to update coverage stats
- When investigating what to implement next

### `compare-docs-api-coverage.py`

Audit docs API coverage against implemented `numpy-ts` exports.

**Usage:**
```bash
python scripts/compare-docs-api-coverage.py
python scripts/compare-docs-api-coverage.py --verbose
python scripts/compare-docs-api-coverage.py --no-refresh
```

**What it does:**
1. Runs `audit-numpyts-api.ts` to extract current exports
2. Detects latest docs API folder (from `docs/docs.json` redirect or `docs/v*/api`)
3. Parses `function ...(` signatures in `docs/<version>/api/**/*.mdx`
4. Reports missing docs entries for implemented functions
5. Reports doc signatures that do not map to current implementation
6. Treats known aliases as covered when they are mentioned in API docs text
7. Checks signature parity by default (parameter names/types + return type, with `NDArrayCore` normalized to `NDArray`)

**Output:** `scripts/docs-api-audit.json`

### `audit-numpy-api.py`

Extracts and categorizes all NumPy functions.

**Output:** `scripts/numpy-api-audit.json`

**Categories:**
- Array Creation, Manipulation
- Arithmetic, Trigonometric, Hyperbolic, Exponential
- Reductions, Comparisons, Logic
- Linear Algebra, Random, FFT
- And more...

### `audit-numpyts-api.ts`

Extracts numpy-ts implementation (top-level functions and NDArray methods).

**Output:** `scripts/numpyts-api-audit.json`

**Extracted:**
- All exported functions from `src/index.ts`
- All NDArray prototype methods and properties

## Generated Files

These files are generated automatically and ignored by git:

- `numpy-api-audit.json` - NumPy API audit results
- `numpyts-api-audit.json` - numpy-ts API audit results

## Requirements

**Python scripts:**
- Python 3.x
- NumPy (conda environment: `py313`)

**TypeScript scripts:**
- Node.js
- tsx (via npx)

## Examples

### Check coverage and update README
```bash
source ~/.zshrc && conda activate py313
python scripts/compare-api-coverage.py
```

### See what's missing
```bash
source ~/.zshrc && conda activate py313
python scripts/compare-api-coverage.py -v | grep "Missing from"
```

### Run audits manually
```bash
# Audit NumPy
source ~/.zshrc && conda activate py313
python scripts/audit-numpy-api.py

# Audit numpy-ts
npx tsx scripts/audit-numpyts-api.ts
```
