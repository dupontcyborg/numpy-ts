# TODO Before 1.0.0 Release

This document tracks remaining tasks before declaring numpy-ts v1.0.0.

## Critical (Must Fix)

### Test Coverage

- [ ] **Increase test coverage to 80%+ lines** (currently 76.6%)
  - Priority modules:
    - `src/core/complex.ts` (38% → 80%) - Complex class methods
    - `src/io/npy/parser.ts` (51% → 80%) - NPY format edge cases
    - `src/io/npy/serializer.ts` (52% → 80%) - Large array serialization
    - `src/ops/bitwise.ts` (56% → 80%) - packbits/unpackbits edge cases
    - `src/ops/linalg.ts` (67% → 80%) - Decomposition edge cases
    - `src/ops/reduction.ts` (67% → 80%) - Axis handling, nan-aware variants

- [ ] **Fix flaky validation test timeout**
  - File: `tests/validation/dtype-promotion-matrix.numpy.test.ts:287`
  - Test: "small integer + float32 promotes to float32..."
  - Solution: Increase timeout or optimize Python subprocess calls

### Documentation

- [ ] **Review all documentation for accuracy**
  - Ensure all code examples work
  - Verify API coverage numbers match reality
  - Check for any other stale content

---

## Important (Should Fix)

### Tree-Shaking Support

- [ ] **Add tree-shaking validation tests**
  - Create `tests/tree-shaking/` directory
  - Test that importing single functions produces small bundles
  - Test that unused modules are excluded

- [ ] **Consider modular exports** (post-1.0 candidate)
  - Add subpath exports: `numpy-ts/core`, `numpy-ts/linalg`, `numpy-ts/random`, etc.
  - Allow users to import only what they need
  - Would require build system changes

### Code Quality

- [ ] **Address TODO comment in production code**
  - File: `src/ops/linalg.ts:675`
  - Content: `// TODO: Add float32 support using sgemm`
  - Either implement or track in GitHub issue

- [ ] **Review console.warn usage**
  - `src/ops/random.ts:2039` - covariance matrix warning
  - `src/ops/linalg.ts:3391` - matrix operation warning
  - Consider making these configurable via error state

### Version Management

- [ ] **Sync __version__ fallback with package.json**
  - File: `src/index.ts:691-692`
  - Currently hardcoded to `'0.12.0'`
  - Should either auto-sync or be removed to force build-time injection

---

## Nice to Have (Consider for 1.x)

### Performance

- [ ] **Add float32 matmul optimization** (sgemm)
- [ ] **Profile and optimize hot paths**
- [ ] **Consider WASM acceleration for compute-intensive operations**

### Features

- [ ] **Implement errstate() context manager**
  - Currently only have `seterr()`/`geterr()`
  - Missing context manager for scoped error handling

- [ ] **Support tuple of axes for reductions**
  - Currently: `arr.sum(axis=0)`
  - NumPy also supports: `arr.sum(axis=(0, 1))`

- [ ] **Ufunc system** (major undertaking)
  - Would enable `out` parameter for memory reuse
  - Would enable `where` parameter for conditional ops
  - Would enable `.reduce()`, `.accumulate()`, `.outer()` methods

### Testing

- [ ] **Add property-based testing** (fast-check)
- [ ] **Add fuzzing for parser edge cases**
- [ ] **Add memory leak detection tests**

---

## Completed

- [x] Add `sideEffects: false` to package.json
- [x] Fix README bundle size claim (~60kb gzip, not <50kb)
- [x] Fix API-REFERENCE.md typo (174/47 → 47/47)
- [x] Update FEATURE-GAPS.md (FFT, Polynomials now implemented)

---

## Release Checklist

When ready to release 1.0.0:

1. [ ] All "Critical" items above are complete
2. [ ] All tests pass (`npm run test:ci`)
3. [ ] Bundle tests pass (`npm run test:bundles`)
4. [ ] Documentation is up to date
5. [ ] CHANGELOG.md is updated
6. [ ] Version bumped in package.json
7. [ ] Git tag created
8. [ ] Published to npm
