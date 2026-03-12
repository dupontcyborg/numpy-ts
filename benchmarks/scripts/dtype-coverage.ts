import { getBenchmarkSpecs } from '../src/specs';

const specs = getBenchmarkSpecs('full');

// Known dtypes to strip from spec names
const DTYPES = new Set([
  'float64', 'float32',
  'complex128', 'complex64',
  'int64', 'int32', 'int16', 'int8',
  'uint64', 'uint32', 'uint16', 'uint8',
]);

// Parse spec name into base + dtype
function parseSpec(name: string): { base: string; dtype: string } {
  const words = name.split(' ');
  const last = words[words.length - 1]!;
  if (DTYPES.has(last) && words.length > 1) {
    return { base: words.slice(0, -1).join(' '), dtype: last };
  }
  return { base: name, dtype: 'float64' };
}

const ops: Record<string, Set<string>> = {};
for (const s of specs) {
  const { base, dtype } = parseSpec(s.name);
  if (!ops[base]) ops[base] = new Set();
  ops[base].add(dtype);
}

// All dtypes in a nice order
const dtypeOrder = [
  'float64', 'float32',
  'complex128', 'complex64',
  'int64', 'int32', 'int16', 'int8',
];

// Print table
const colW = 8;
const nameW = 45;
const hdr = 'Operation'.padEnd(nameW) + '| ' + dtypeOrder.map(d => d.padEnd(colW)).join('| ');
console.log(hdr);
console.log('-'.repeat(hdr.length));

for (const [op, dtypes] of Object.entries(ops).sort((a, b) => a[0].localeCompare(b[0]))) {
  const row = op.padEnd(nameW) + '| ' + dtypeOrder.map(d => (dtypes.has(d) ? 'x' : '').padEnd(colW)).join('| ');
  console.log(row);
}

console.log();
console.log('Total specs:', specs.length);
console.log('Unique operations:', Object.keys(ops).length);

// Summary: count by category
const catCounts: Record<string, number> = {};
for (const s of specs) {
  catCounts[s.category] = (catCounts[s.category] || 0) + 1;
}
console.log('\nSpecs per category:');
for (const [cat, count] of Object.entries(catCounts).sort()) {
  console.log(`  ${cat}: ${count}`);
}
