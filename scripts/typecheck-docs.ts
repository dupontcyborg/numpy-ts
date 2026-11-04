/**
 * Type-check the TypeScript code fences in the docs against the library source.
 *
 * Each fence becomes its own module so declarations in one snippet cannot leak into another, and
 * the 'numpy-ts' specifiers resolve to src/ so the check does not depend on a build. Errors are
 * reported at the doc file and line, not the generated file.
 *
 * An MDX comment directly above a fence tunes how it is checked and renders as nothing on the page.
 * The comment text `typecheck: skip` skips a deliberately incomplete or illustrative snippet, and
 * `typecheck-setup: <code>` prepends declarations that earlier snippets on the page would have
 * provided. Markdown files wrap the same text in an HTML comment instead.
 *
 * Usage: tsx scripts/typecheck-docs.ts [path ...]   (files or directories; defaults to the current docs)
 */

import { spawnSync } from 'node:child_process';
import { mkdtempSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, relative, resolve } from 'node:path';

const ROOT = resolve(import.meta.dirname, '..');
const DEFAULT_TARGETS = ['docs/index.mdx', 'docs/latest', 'docs/next', 'README.md'];
const TS_LANGS = new Set(['ts', 'typescript']);

interface Snippet {
  docPath: string;
  /** 1-based doc line of the first line of code, i.e. the line after the opening fence. */
  startLine: number;
  code: string;
  /** Declarations from a typecheck-setup directive, kept on the first line so doc line numbers hold. */
  setup: string;
}

const DIRECTIVE = /^\s*(?:\{\/\*|<!--)\s*typecheck(-setup)?:\s*(.*?)\s*(?:\*\/\}|-->)\s*$/;

/** Directives in the comment lines directly above a fence, ignoring blank lines. */
function readDirectives(lines: string[], fence: number): { skip: boolean; setup: string } {
  let skip = false;
  const setup: string[] = [];
  for (let k = fence - 1; k >= 0; k--) {
    if (!lines[k]!.trim()) continue;
    const m = DIRECTIVE.exec(lines[k]!);
    if (!m) break;
    if (m[1]) setup.unshift(m[2]!);
    else if (m[2] === 'skip') skip = true;
  }
  return { skip, setup: setup.join(' ') };
}

/**
 * True for an API reference signature such as `copy(): NDArray` or `function linalg.eig(a): X`.
 * These are declarations without bodies, and a return annotation after a call's closing paren
 * cannot occur in a runnable expression. On API pages a class, interface or type declaration is
 * documentation of a shape, not an example, and often names types it does not define.
 */
function isSignature(code: string, docPath: string): boolean {
  const first = code.split('\n').find((l) => l.trim()) ?? '';
  return (
    (docPath.includes('/api/') && /^(declare\s+)?(class|interface|type)\s+\w/.test(first)) ||
    /^function\s/.test(first) ||
    /^(static\s+|get\s+|set\s+)+\w/.test(first) ||
    /^\w+\??\s*(<.*>)?\(.*\)\s*:/.test(first) ||
    /^\w+\??\s*(<.*>)?\(\s*(\w+\??:.*)?$/.test(first)
  );
}

function listDocs(target: string): string[] {
  const abs = resolve(ROOT, target);
  if (statSync(abs).isFile()) return [abs];
  return readdirSync(abs, { withFileTypes: true }).flatMap((e) => {
    const p = join(abs, e.name);
    if (e.isDirectory()) return listDocs(p);
    return /\.mdx?$/.test(e.name) ? [p] : [];
  });
}

function extractSnippets(docPath: string): Snippet[] {
  const lines = readFileSync(docPath, 'utf8').split('\n');
  const snippets: Snippet[] = [];
  for (let i = 0; i < lines.length; i++) {
    const open = /^(\s*)```(\w+)?(.*)$/.exec(lines[i]!);
    if (!open) continue;
    const indent = open[1]!.length;
    const lang = open[2]?.toLowerCase();
    const body: string[] = [];
    let j = i + 1;
    for (; j < lines.length && !/^\s*```\s*$/.test(lines[j]!); j++) {
      body.push(lines[j]!.slice(Math.min(indent, lines[j]!.length - lines[j]!.trimStart().length)));
    }
    if (lang && TS_LANGS.has(lang)) {
      const code = body.join('\n');
      const { skip, setup } = readDirectives(lines, i);
      if (!skip && !isSignature(code, docPath)) {
        snippets.push({ docPath, startLine: i + 2, code, setup });
      }
    }
    i = j;
  }
  return snippets;
}

const targets = process.argv.slice(2);
const snippets = (targets.length ? targets : DEFAULT_TARGETS)
  .flatMap(listDocs)
  .flatMap(extractSnippets);

const work = mkdtempSync(join(tmpdir(), 'docs-typecheck-'));
try {
  const owners = new Map<string, Snippet>();
  snippets.forEach((s, n) => {
    const file = join(work, `snippet-${n}.ts`);
    // A snippet with no import or export is a script, so its top-level names would collide across files.
    const isModule = /^\s*(import|export)\b/m.test(s.code);
    // Pages import the namespace once, so later snippets use np without repeating it.
    const needsNp =
      /\bnp\./.test(s.code) && !/\bas np\b|\bnp\s*=|\{[^}]*\bnp\b[^}]*\}\s*from/.test(s.code);
    const prelude = `${needsNp ? "import * as np from 'numpy-ts'; " : ''}${s.setup ? `${s.setup} ` : ''}`;
    writeFileSync(file, `${prelude}${s.code}${isModule || prelude ? '' : '\nexport {};'}\n`);
    owners.set(file, s);
  });

  const src = (p: string) => [join(ROOT, 'src', p)];
  const tsc = join(ROOT, 'node_modules/.bin/tsc');

  /** Runs tsc over the given files and returns each error with the snippet it belongs to. */
  const check = (files: string[]) => {
    writeFileSync(
      join(work, 'tsconfig.json'),
      JSON.stringify({
        extends: join(ROOT, 'tsconfig.json'),
        compilerOptions: {
          noEmit: true,
          rootDir: '/',
          declaration: false,
          declarationMap: false,
          sourceMap: false,
          noUncheckedIndexedAccess: false,
          noUnusedLocals: false,
          noUnusedParameters: false,
          typeRoots: [join(ROOT, 'node_modules/@types')],
          paths: {
            'numpy-ts': src('index.ts'),
            'numpy-ts/node': src('index.ts'),
            'numpy-ts/core': src('core.ts'),
          },
        },
        include: [],
        files,
      }),
    );
    const res = spawnSync(tsc, ['-p', join(work, 'tsconfig.json'), '--pretty', 'false'], {
      cwd: ROOT,
      encoding: 'utf8',
      maxBuffer: 64 * 1024 * 1024,
    });
    const errors: { file?: string; text: string; syntax: boolean }[] = [];
    for (const line of `${res.stdout}${res.stderr}`.split('\n')) {
      const m = /^(.+?)\((\d+),(\d+)\): (error TS(\d+): .*)$/.exec(line);
      const file = m && resolve(ROOT, m[1]!);
      const owner = file && owners.get(file);
      if (!m || !owner) {
        if (line.trim() && !line.startsWith(' ')) errors.push({ text: line, syntax: false });
        continue;
      }
      const docLine = owner.startLine + Number(m[2]) - 1;
      const text = `${relative(ROOT, owner.docPath)}:${docLine}:${m[3]} ${m[4]}`;
      errors.push({ file, text, syntax: Number(m[5]) < 2000 });
    }
    return errors;
  };

  // tsc reports only syntax errors while any exist, so a snippet that fails to parse would hide
  // every type error in the rest. Report those first, then check the remainder without them.
  const first = check([...owners.keys()]);
  const broken = new Set(first.filter((e) => e.syntax).map((e) => e.file));
  const rest = broken.size ? check([...owners.keys()].filter((f) => !broken.has(f))) : [];
  const problems = [...first, ...rest].map((e) => e.text);

  if (problems.length) {
    console.error(problems.join('\n'));
    console.error(
      `\ndocs typecheck failed (${problems.length} problem(s) in ${snippets.length} snippets)`,
    );
    process.exitCode = 1;
  } else {
    console.log(`docs typecheck passed (${snippets.length} snippets)`);
  }
} finally {
  rmSync(work, { recursive: true, force: true });
}
