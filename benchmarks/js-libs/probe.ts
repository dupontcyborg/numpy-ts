/**
 * API-surface probe: dumps the callable surface of each competitor JS library
 * so we can map numpy-ts benchmark operations onto what each library supports.
 *
 * Deps live in this directory's own package.json (NOT numpy-ts's). Run:
 *   npm run bench:js-libs:install   # from repo root, once
 *   cd benchmarks/js-libs && npm run probe
 */

function fns(obj: any): string[] {
  const out = new Set<string>();
  let o = obj;
  while (o && o !== Object.prototype) {
    for (const k of Object.getOwnPropertyNames(o)) {
      if (k.startsWith('_')) continue;
      try {
        if (typeof o[k] === 'function') out.add(k);
      } catch {}
    }
    o = Object.getPrototypeOf(o);
  }
  return [...out].sort();
}

function show(label: string, names: string[]) {
  console.log(`\n=== ${label} (${names.length}) ===`);
  console.log(names.join(', '));
}

async function main() {
  try {
    const math = await import('mathjs');
    show('mathjs (top-level)', fns(math).filter((n) => typeof (math as any)[n] === 'function'));
  } catch (e) {
    console.log('mathjs FAILED', (e as Error).message);
  }

  try {
    const nj: any = (await import('@d4c/numjs')).default ?? (await import('@d4c/numjs'));
    show('@d4c/numjs (top-level)', fns(nj));
    const sample = nj.arange(9).reshape(3, 3);
    show('@d4c/numjs NdArray methods', fns(sample));
  } catch (e) {
    console.log('@d4c/numjs FAILED', (e as Error).message);
  }

  try {
    const numeric: any = (await import('numeric')).default ?? (await import('numeric'));
    show('numeric (top-level)', fns(numeric));
  } catch (e) {
    console.log('numeric FAILED', (e as Error).message);
  }

  try {
    const mlm: any = await import('ml-matrix');
    show('ml-matrix (exports)', Object.keys(mlm).sort());
    const m = new mlm.Matrix([[1, 2], [3, 4]]);
    show('ml-matrix instance methods', fns(m));
  } catch (e) {
    console.log('ml-matrix FAILED', (e as Error).message);
  }

  try {
    const sd: any = (await import('@stdlib/ndarray')).default ?? (await import('@stdlib/ndarray'));
    show('@stdlib/ndarray (namespace)', fns(sd));
  } catch (e) {
    console.log('@stdlib/ndarray FAILED', (e as Error).message);
  }

  try {
    const jax: any = await import('@jax-js/jax');
    show('@jax-js/jax (exports)', Object.keys(jax).sort());
    if (jax.numpy) show('@jax-js/jax numpy namespace', fns(jax.numpy));
  } catch (e) {
    console.log('@jax-js/jax FAILED', (e as Error).message);
  }

  try {
    const tf: any = await import('@tensorflow/tfjs-core');
    show('tfjs-core (ops)', fns(tf).filter((n) => /^[a-z]/.test(n)));
  } catch (e) {
    console.log('tfjs-core FAILED', (e as Error).message);
  }
}

main();
