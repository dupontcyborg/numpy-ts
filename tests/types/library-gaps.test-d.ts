/**
 * Compile-time assertions for the friendlier full-API signatures: overloads whose return type
 * depends on the arguments, and parameters that accept readonly or array-like input.
 *
 * Checked by `pnpm run typecheck:types`.
 */

import type { NDArrayCore } from '../../src/common/ndarray-core';
import type { ArrayStorage } from '../../src/common/storage';
import type * as coreEntry from '../../src/core';
import * as core from '../../src/core/index';
import * as np from '../../src/full/index';
import type { NDArray } from '../../src/full/ndarray';
import * as main from '../../src/index';

type Equal<A, B> =
  (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2 ? true : false;
type Expect<T extends true> = T;

declare const f64: NDArray<'float64'>;

// ── reshape accepts a shape array as well as variadic dims ──────────────────
const rs1 = f64.reshape([2, 3]);
type _reshapeArr = Expect<Equal<typeof rs1, NDArray<'float64'>>>;
const rs2 = f64.reshape(2, 3);
type _reshapeVar = Expect<Equal<typeof rs2, NDArray<'float64'>>>;
const rs3 = f64.reshape(f64.shape);
type _reshapeReadonly = Expect<Equal<typeof rs3, NDArray<'float64'>>>;

// ── shape parameters accept readonly arrays such as NDArray.shape ───────────
const ro: readonly number[] = [2, 3];
np.zeros(ro);
np.ones(ro, 'int32');
np.empty(ro);
np.full(ro, 1);
np.reshape(f64, ro);
np.resize(f64, ro);
np.broadcast_to(f64, ro);
np.indices(ro);
np.unravel_index(0, f64.shape);
np.ravel_multi_index([f64], f64.shape);
f64.resize(f64.shape);

// ── core linalg: svd and qr return types follow compute_uv and mode ─────────
declare const mat: NDArrayCore;
const csvd = core.linalg.svd(mat);
type _coreSvd = Expect<Equal<typeof csvd, { u: NDArrayCore; s: NDArrayCore; vt: NDArrayCore }>>;
const csvdFull = core.linalg.svd(mat, false, true);
type _coreSvdTrue = Expect<Equal<typeof csvdFull, typeof csvd>>;
const csvdS = core.linalg.svd(mat, false, false);
type _coreSvdNoUv = Expect<Equal<typeof csvdS, NDArrayCore>>;
declare const flag: boolean;
const csvdAny = core.linalg.svd(mat, true, flag);
type _coreSvdUnion = Expect<Equal<typeof csvdAny, typeof csvd | NDArrayCore>>;

const cqr = core.linalg.qr(mat);
type _coreQr = Expect<Equal<typeof cqr, { q: NDArrayCore; r: NDArrayCore }>>;
const cqrC = core.linalg.qr(mat, 'complete');
type _coreQrComplete = Expect<Equal<typeof cqrC, typeof cqr>>;
const cqrR = core.linalg.qr(mat, 'r');
type _coreQrR = Expect<Equal<typeof cqrR, NDArrayCore>>;
const cqrRaw = core.linalg.qr(mat, 'raw');
type _coreQrRaw = Expect<Equal<typeof cqrRaw, { h: NDArrayCore; tau: NDArrayCore }>>;
declare const modeAny: 'reduced' | 'complete' | 'r' | 'raw';
const cqrAny = core.linalg.qr(mat, modeAny);
type _coreQrUnion = Expect<Equal<typeof cqrAny, typeof cqr | NDArrayCore | typeof cqrRaw>>;

// ── unique: result shape follows the flags ──────────────────────────────────
const u0 = np.unique(f64);
type _unique0 = Expect<Equal<typeof u0, NDArray>>;
const u1 = np.unique(f64, true, false, true);
type _unique1 = Expect<
  Equal<typeof u1, { values: NDArray; indices: NDArray; inverse?: NDArray; counts: NDArray }>
>;
const u2 = np.unique(f64, false, true);
type _unique2 = Expect<
  Equal<typeof u2, { values: NDArray; indices?: NDArray; inverse: NDArray; counts?: NDArray }>
>;
const uAny = np.unique(f64, flag);
type _uniqueAny = Expect<
  Equal<
    typeof uAny,
    NDArray | { values: NDArray; indices?: NDArray; inverse?: NDArray; counts?: NDArray }
  >
>;
const cu0 = core.unique(mat);
type _coreUnique0 = Expect<Equal<typeof cu0, NDArrayCore>>;
const cu1 = core.unique(mat, true, true, true);
type _coreUnique1 = Expect<
  Equal<
    typeof cu1,
    { values: NDArrayCore; indices: NDArrayCore; inverse: NDArrayCore; counts: NDArrayCore }
  >
>;

// ── where: one argument gives index arrays, three give an array ─────────────
const w1 = np.where(f64);
type _where1 = Expect<Equal<typeof w1, NDArray[]>>;
const w3 = np.where(f64, 1, 0);
type _where3 = Expect<Equal<typeof w3, NDArray>>;
const cw1 = core.where(mat);
type _coreWhere1 = Expect<Equal<typeof cw1, NDArrayCore[]>>;
const cw3 = core.where(mat, 1, 0);
type _coreWhere3 = Expect<Equal<typeof cw3, NDArrayCore>>;

// ── atleast_Nd: one argument gives an array, several give a list ────────────
const a1 = np.atleast_1d(f64);
type _atleast1 = Expect<Equal<typeof a1, NDArray>>;
const a2 = np.atleast_2d(f64, f64);
type _atleast2 = Expect<Equal<typeof a2, NDArray[]>>;
const a3 = np.atleast_3d(f64);
type _atleast3 = Expect<Equal<typeof a3, NDArray>>;
const ca1 = core.atleast_1d(mat);
type _coreAtleast1 = Expect<Equal<typeof ca1, NDArrayCore>>;
const ca2 = core.atleast_2d(mat, mat);
type _coreAtleast2 = Expect<Equal<typeof ca2, NDArrayCore[]>>;

// ── reductions: an explicit axis with keepdims true yields an array ─────────
const k1 = np.sum(f64, 0, true);
type _keepSum = Expect<Equal<typeof k1, NDArray<'float64'>>>;
const k2 = np.mean(f64, [0, 1], true);
type _keepMean = Expect<Equal<typeof k2, NDArray<'float64'>>>;
const k3 = np.std(f64, 0, 1, true);
type _keepStd = Expect<Equal<typeof k3, NDArray<'float64'>>>;
const k4 = np.argmax(f64, 0, true);
type _keepArg = Expect<Equal<typeof k4, NDArray<'int32'>>>;
const k5 = np.percentile(f64, 50, 0, true);
type _keepPercentile = Expect<Equal<typeof k5, NDArray<'float64'>>>;
// Without keepdims, or without an axis, the result may still be a scalar.
const k6 = np.sum(f64, 0);
type _noKeep = Expect<Equal<typeof k6, NDArray<'float64'> | number>>;

// ── random: an array when size is given, a number otherwise ─────────────────
const rn1 = main.random.randn(2, 3);
type _randn = Expect<Equal<typeof rn1, NDArray<'float64'>>>;
const rn0 = main.random.randn();
type _randn0 = Expect<Equal<typeof rn0, number>>;
const rr = main.random.random([2, 3]);
type _random = Expect<Equal<typeof rr, NDArray<'float64'>>>;
const rr0 = main.random.random();
type _random0 = Expect<Equal<typeof rr0, number>>;
const rno = main.random.normal(0, 1, [4]);
type _normal = Expect<Equal<typeof rno, NDArray<'float64'>>>;
const rno0 = main.random.normal(0, 1);
type _normal0 = Expect<Equal<typeof rno0, number>>;
const rnoSkip = main.random.normal(undefined, undefined, 5);
type _normalSkip = Expect<Equal<typeof rnoSkip, NDArray<'float64'>>>;
const rb = main.random.beta(1, 2, [3]);
type _beta = Expect<Equal<typeof rb, NDArray<'float64'>>>;
const rp = main.random.poisson(3, 5);
type _poisson = Expect<Equal<typeof rp, NDArray<'int64'>>>;
const ri = main.random.randint(0, 10, [3]);
type _randint = Expect<Equal<typeof ri, NDArray<'int64'>>>;
const ri32 = main.random.randint(0, 10, [3], 'int32');
type _randint32 = Expect<Equal<typeof ri32, NDArray<'int32'>>>;
const ri0 = main.random.randint(10);
type _randint0 = Expect<Equal<typeof ri0, number>>;
const rch = main.random.choice(f64, 3);
type _choice = Expect<Equal<typeof rch, NDArray>>;
const rch0 = main.random.choice(5);
type _choice0 = Expect<Equal<typeof rch0, number>>;
// @ts-expect-error a plain array population must be wrapped with np.array
main.random.choice([1, 2, 3], 2);
const rng = main.random.default_rng(1);
const g1 = rng.random([2]);
type _rngRandom = Expect<Equal<typeof g1, NDArray<'float64'>>>;
const g0 = rng.random();
type _rngRandom0 = Expect<Equal<typeof g0, number>>;
const gi = rng.integers(0, 10, [2]);
type _rngIntegers = Expect<Equal<typeof gi, NDArray<'int64'>>>;
const gc = rng.choice(f64, 2, false);
type _rngChoice = Expect<Equal<typeof gc, NDArray>>;

// ── fill_diagonal, append and mintypecode accept the wider inputs ───────────
np.fill_diagonal(f64, 1);
np.fill_diagonal(f64, f64);
core.fill_diagonal(mat, mat);
np.append(f64, [[5, 6]], 0);
np.append(f64, f64);
np.append(f64, 5);
core.append(mat, [[5, 6]]);
np.mintypecode('GDF');
np.mintypecode(['i', 'f']);
const typechars: readonly string[] = ['d'];
np.mintypecode(typechars);

// ── apply_along_axis passes full NDArrays to the callback ───────────────────
const applied = np.apply_along_axis((row) => row.multiply(2).reshape([-1]), 0, f64);
type _applied = Expect<Equal<typeof applied, NDArray>>;
np.apply_along_axis((row, k) => row.multiply(k), 1, f64, 3);

// ── dtype predicates are public ─────────────────────────────────────────────
type _isInt = Expect<Equal<ReturnType<typeof main.isIntegerDType>, boolean>>;
type _isFloat = Expect<Equal<ReturnType<typeof main.isFloatDType>, boolean>>;
type _isComplex = Expect<Equal<ReturnType<typeof main.isComplexDType>, boolean>>;
type _coreIsInt = Expect<Equal<ReturnType<typeof coreEntry.isIntegerDType>, boolean>>;

// ── full linalg returns NDArray, with the svd and qr overloads kept ─────────
const linInv = main.linalg.inv(f64);
type _linInv = Expect<Equal<typeof linInv, NDArray>>;
const linSvd = main.linalg.svd(f64);
type _linSvd = Expect<Equal<typeof linSvd, { u: NDArray; s: NDArray; vt: NDArray }>>;
const linSvdS = main.linalg.svd(f64, false, false);
type _linSvdS = Expect<Equal<typeof linSvdS, NDArray>>;
const linQr = main.linalg.qr(f64);
type _linQr = Expect<Equal<typeof linQr, { q: NDArray; r: NDArray }>>;
const linQrR = main.linalg.qr(f64, 'r');
type _linQrR = Expect<Equal<typeof linQrR, NDArray>>;
const linQrRaw = main.linalg.qr(f64, 'raw');
type _linQrRaw = Expect<Equal<typeof linQrRaw, { h: NDArray; tau: NDArray }>>;
const linEig = main.linalg.eig(f64);
type _linEig = Expect<Equal<typeof linEig, { w: NDArray; v: NDArray }>>;
main.linalg.inv(f64).T.sum();

// ── fft takes NDArray and NDArrayCore, not plain arrays or raw storage ──────
main.fft.fft(f64);
main.fft.fft(mat);
main.fft.fftshift(f64, 0);
core.fft.fft(mat);
core.fft.fft(f64);
core.fft.ifftshift(mat);
// @ts-expect-error a plain array must be wrapped with np.array
main.fft.fft([1, 2, 3]);
// @ts-expect-error a plain array must be wrapped with np.array
core.fft.rfft([1, 2, 3]);
// @ts-expect-error raw storage is internal
main.fft.fft(f64.storage);
// @ts-expect-error raw storage is internal
core.fft.fft(mat.storage);

// ── core random: an NDArrayCore when size is given, a number otherwise ──────
const crr = core.random.random([2, 3]);
type _coreRandom = Expect<Equal<typeof crr, NDArrayCore<'float64'>>>;
const crr0 = core.random.random();
type _coreRandom0 = Expect<Equal<typeof crr0, number>>;
const crn = core.random.randn(2, 3);
type _coreRandn = Expect<Equal<typeof crn, NDArrayCore<'float64'>>>;
const crn0 = core.random.randn();
type _coreRandn0 = Expect<Equal<typeof crn0, number>>;
const cno = core.random.normal(0, 1, [4]);
type _coreNormal = Expect<Equal<typeof cno, NDArrayCore<'float64'>>>;
const cnoSkip = core.random.normal(undefined, undefined, 5);
type _coreNormalSkip = Expect<Equal<typeof cnoSkip, NDArrayCore<'float64'>>>;
const cno0 = core.random.normal(0, 1);
type _coreNormal0 = Expect<Equal<typeof cno0, number>>;
const cpo = core.random.poisson(3, 5);
type _corePoisson = Expect<Equal<typeof cpo, NDArrayCore<'int64'>>>;
const cpo0 = core.random.poisson(3);
type _corePoisson0 = Expect<Equal<typeof cpo0, number>>;
const cri = core.random.randint(0, 10, [3]);
type _coreRandint = Expect<Equal<typeof cri, NDArrayCore<'int64'>>>;
const cri32 = core.random.randint(0, 10, [3], 'int32');
type _coreRandint32 = Expect<Equal<typeof cri32, NDArrayCore<'int32'>>>;
const cri0 = core.random.randint(10);
type _coreRandint0 = Expect<Equal<typeof cri0, number>>;
const cch = core.random.choice(mat, 3);
type _coreChoice = Expect<Equal<typeof cch, NDArrayCore>>;
const cch0 = core.random.choice(5);
type _coreChoice0 = Expect<Equal<typeof cch0, number>>;
const cchP = core.random.choice(mat, 3, true, [0.2, 0.8]);
type _coreChoiceP = Expect<Equal<typeof cchP, NDArrayCore>>;
const cpm = core.random.permutation(mat);
type _corePermutation = Expect<Equal<typeof cpm, NDArrayCore>>;
const cmn = core.random.multinomial(5, [0.5, 0.5], 3);
type _coreMultinomial = Expect<Equal<typeof cmn, NDArrayCore<'int64'>>>;
const cdi = core.random.dirichlet(mat, 2);
type _coreDirichlet = Expect<Equal<typeof cdi, NDArrayCore<'float64'>>>;
const cmv = core.random.multivariate_normal([0, 0], mat);
type _coreMvn = Expect<Equal<typeof cmv, NDArrayCore<'float64'>>>;
const crng = core.random.default_rng(1);
const cg1 = crng.random([2]);
type _coreRngRandom = Expect<Equal<typeof cg1, NDArrayCore<'float64'>>>;
const cg0 = crng.random();
type _coreRngRandom0 = Expect<Equal<typeof cg0, number>>;
const cgi = crng.integers(0, 10, [2]);
type _coreRngIntegers = Expect<Equal<typeof cgi, NDArrayCore<'int64'>>>;
const cgb = crng.binomial(10, 0.5, 3);
type _coreRngBinomial = Expect<Equal<typeof cgb, NDArrayCore<'int64'>>>;
const cgc = crng.choice(mat, 2, false);
type _coreRngChoice = Expect<Equal<typeof cgc, NDArrayCore>>;
const cgc0 = crng.choice(4);
type _coreRngChoice0 = Expect<Equal<typeof cgc0, number>>;
const cgp = crng.permutation(mat);
type _coreRngPermutation = Expect<Equal<typeof cgp, NDArrayCore>>;
type _coreShuffle = Expect<Equal<ReturnType<typeof core.random.shuffle>, void>>;

// ── random inputs are NDArray or plain number arrays, not raw storage ───────
declare const storage: ArrayStorage;
main.random.multinomial(5, f64);
main.random.multinomial(5, [0.5, 0.5]);
main.random.dirichlet(f64);
main.random.dirichlet([1, 2]);
main.random.multivariate_normal(f64, f64);
main.random.multivariate_normal(
  [0, 0],
  [
    [1, 0],
    [0, 1],
  ],
);
main.random.choice(f64, 2, true, f64);
main.random.permutation(f64);
main.random.shuffle(f64);
core.random.multinomial(5, mat);
core.random.multivariate_normal(mat, mat);
core.random.choice(f64, 2, true, mat);
core.random.shuffle(f64);
// @ts-expect-error raw storage is internal
main.random.multinomial(5, storage);
// @ts-expect-error raw storage is internal
main.random.dirichlet(storage);
// @ts-expect-error raw storage is internal
main.random.multivariate_normal(storage, [[1]]);
// @ts-expect-error raw storage is internal
main.random.choice(storage, 2);
// @ts-expect-error raw storage is internal
main.random.choice(f64, 2, true, storage);
// @ts-expect-error raw storage is internal
main.random.permutation(storage);
// @ts-expect-error raw storage is internal
main.random.shuffle(storage);
// @ts-expect-error raw storage is internal
core.random.multinomial(5, storage);
// @ts-expect-error raw storage is internal
core.random.dirichlet(storage);
// @ts-expect-error raw storage is internal
core.random.multivariate_normal([0, 0], storage);
// @ts-expect-error raw storage is internal
core.random.choice(storage, 2);
// @ts-expect-error raw storage is internal
core.random.choice(mat, 2, true, storage);
// @ts-expect-error raw storage is internal
core.random.permutation(storage);
// @ts-expect-error raw storage is internal
core.random.shuffle(storage);
// @ts-expect-error raw storage is internal
crng.choice(storage, 2);
// @ts-expect-error a plain array population must be wrapped with np.array
core.random.choice([1, 2, 3], 2);
// @ts-expect-error a plain array population must be wrapped with np.array
crng.permutation([1, 2, 3]);
