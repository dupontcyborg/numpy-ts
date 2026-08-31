/** Probe dtype support across competitor libs. Run from this dir: npx tsx dtype-probe.ts */
async function main() {
  // @stdlib/ndarray — richest dtype system among JS libs
  try {
    const sd: any = (await import('@stdlib/ndarray')).default ?? (await import('@stdlib/ndarray'));
    console.log('stdlib dtypes:', sd.dtypes().sort().join(','));
  } catch (e) { console.log('stdlib FAIL', (e as Error).message); }

  // jax-js
  try {
    const jax: any = await import('@jax-js/jax');
    const np = jax.numpy;
    const cands = ['float64', 'float32', 'float16', 'int64', 'int32', 'int16', 'int8', 'uint64', 'uint32', 'uint16', 'uint8', 'bool', 'complex64', 'complex128'];
    const ok: string[] = [];
    for (const dt of cands) {
      try { const a = np.array([1, 2, 3], dt); ok.push(`${dt}->${a.dtype ?? a.aval?.dtype ?? '?'}`); a.dispose?.(); }
      catch { /* unsupported */ }
    }
    console.log('jax-js dtypes accepted:', ok.join(', ') || 'none via array(_,dt)');
    console.log('jax-js default dtype:', (() => { const a = np.array([1, 2, 3]); const d = a.dtype ?? a.aval?.dtype; a.dispose?.(); return d; })());
    console.log('jax-js DType export:', jax.DType ? Object.keys(jax.DType).join(',') : 'n/a');
  } catch (e) { console.log('jax FAIL', (e as Error).message); }

  // tfjs
  try {
    const tf: any = await import('@tensorflow/tfjs-core');
    const cands = ['float32', 'float64', 'int32', 'int8', 'uint8', 'bool', 'complex64', 'string'];
    const ok: string[] = [];
    for (const dt of cands) { try { const t = tf.tensor1d([1, 2, 3], dt); ok.push(t.dtype); t.dispose(); } catch { } }
    console.log('tfjs dtypes accepted:', [...new Set(ok)].join(','));
  } catch (e) { console.log('tfjs FAIL', (e as Error).message); }

  // @d4c/numjs
  try {
    const nj: any = (await import('@d4c/numjs')).default ?? (await import('@d4c/numjs'));
    const cands = ['float64', 'float32', 'int32', 'int16', 'int8', 'uint32', 'uint16', 'uint8', 'uint8_clamped'];
    const ok: string[] = [];
    for (const dt of cands) { try { const a = nj.array([1, 2, 3], dt); ok.push(a.dtype); } catch { } }
    console.log('numjs dtypes accepted:', [...new Set(ok)].join(','));
  } catch (e) { console.log('numjs FAIL', (e as Error).message); }

  // mathjs — JS number(float64) + bigint + Complex + Fraction + BigNumber
  console.log('mathjs storage: float64 (JS number) default; bigint, Complex, Fraction, BigNumber as object types — no int8/16/32, no float32, no uint*');
  // numeric.js — JS Array<number>=float64; complex via numeric.T(real,imag)
  console.log('numeric.js storage: float64 (JS number) only; complex via T(re,im) tensor — no integer/float32 dtypes');
  // ml-matrix — Float64Array internal
  console.log('ml-matrix storage: float64 (Float64Array) only');
}
main();
