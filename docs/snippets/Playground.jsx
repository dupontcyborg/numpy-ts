export const Playground = ({
  example = "quickstart",
  examples = null,
  defaultExample = null,
  code: singleCode = null,
  label = "",
  height = null,
  startingHeight = null,
  showImportHeader = false,
  showCopyButton = false,
  showTiming = true
}) => {
  const DEFAULT_EXAMPLES = {
    quickstart: {
      label: "Quickstart",
      code: `// Create arrays
const a = np.array([[1, 2, 3], [4, 5, 6]]);
console.log("Array:\\n" + a);
console.log("Shape:", a.shape);
console.log("Dtype:", a.dtype, "\\n");

// Basic arithmetic
const b = np.multiply(a, 2);
console.log("Multiply by 2:\\n" + b, "\\n");

// Element-wise operations
const c = np.add(a, np.array([[10, 20, 30], [40, 50, 60]]));
console.log("Element-wise add:\\n" + c, "\\n");

// Row slicing
const row = a.row(0);
console.log("First row:", row);`,
    },
    linalg: {
      label: "Linear Algebra",
      code: `// Matrix multiplication
const A = np.array([[1, 2], [3, 4]]);
const B = np.array([[5, 6], [7, 8]]);
const C = np.matmul(A, B);
console.log("A @ B =\\n" + C, "\\n");

// Determinant
const det = np.linalg.det(A);
console.log("det(A) =", det, "\\n");

// Inverse
const inv = np.linalg.inv(A);
console.log("inv(A) =\\n" + inv, "\\n");

// Verify A @ inv(A) = I
const I = np.matmul(A, inv);
console.log("A @ inv(A) =\\n" + I, "\\n");

// Eigenvalues
const eig = np.linalg.eig(A);
console.log("Eigenvalues:", eig.w);`,
    },
    broadcasting: {
      label: "Broadcasting",
      code: `// Broadcasting: different shapes work together
const matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
const row = np.array([10, 20, 30]);

// Add row to each row of matrix
const result = np.add(matrix, row);
console.log("Matrix + row vector:");
console.log(result, "\\n");

// Column vector broadcasting
const col = np.array([[100], [200], [300]]);
const result2 = np.add(matrix, col);
console.log("Matrix + column vector:");
console.log(result2, "\\n");

// Scalar broadcasting
console.log("Matrix * 10:");
console.log(np.multiply(matrix, 10), "\\n");`,
    },
    random: {
      label: "Random",
      code: `// Seeded random for reproducibility
np.random.seed(42);

// Random integers
const dice = np.random.randint(1, 7, [3, 3]);
console.log("Dice rolls (3x3):\\n" + dice, "\\n");

// Normal distribution
const normal = np.random.randn(5);
console.log("Normal samples:", normal, "\\n");

// Uniform distribution
const uniform = np.random.rand(2, 3);
console.log("Uniform [0,1):\\n" + uniform, "\\n");

// Statistics of random data
const data = np.random.randn(1000);
console.log("1000 normal samples:");
console.log("  Mean:", Number(np.mean(data)).toFixed(3));
console.log("  Std:", Number(np.std(data)).toFixed(3));`,
    },
    reductions: {
      label: "Reductions",
      code: `const a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
console.log("Array:\\n" + a, "\\n");

// Global reductions
console.log("Sum:", Number(np.sum(a)));
console.log("Mean:", Number(np.mean(a)));
console.log("Std:", Number(np.std(a)).toFixed(4));
console.log("Min:", Number(np.min(a)));
console.log("Max:", Number(np.max(a)), "\\n");

// Axis reductions
console.log("Sum along axis 0 (columns):", np.sum(a, 0));
console.log("Sum along axis 1 (rows):", np.sum(a, 1));
console.log("Mean along axis 0:", np.mean(a, 0), "\\n");

// Cumulative
console.log("Cumsum:", np.cumsum(a));
console.log("Cumprod axis 0:\\n" + np.cumprod(a, 0));`,
    },
    fft: {
      label: "FFT",
      code: `// Generate a signal: 5 Hz + 12 Hz components
const N = 128;
const dt = 1 / 128;
const t = np.arange(0, N * dt, dt);

// Create composite signal
const signal = np.add(
  np.sin(np.multiply(t, 2 * Math.PI * 5)),   // 5 Hz
  np.multiply(np.sin(np.multiply(t, 2 * Math.PI * 12)), 0.5) // 12 Hz
);
console.log("Signal (first 8):\\n", signal.slice('0:8'), "\\n");

// Compute FFT
const spectrum = np.fft.fft(signal);
const magnitudes = np.abs(spectrum);
const freqs = np.fft.fftfreq(N, dt);

// Find peak frequencies (first half only)
const halfN = Math.floor(N / 2);
const halfMag = magnitudes.slice('0:' + halfN);
const halfFreqs = freqs.slice('0:' + halfN);

// Top 2 peaks
const sorted = np.argsort(np.multiply(halfMag, -1));
const i0 = Number(sorted.get([0]));
const i1 = Number(sorted.get([1]));
console.log("Top frequencies:");
console.log("  " + Number(halfFreqs.get([i0])) + " Hz (magnitude: " + Number(halfMag.get([i0])).toFixed(1) + ")");
console.log("  " + Number(halfFreqs.get([i1])) + " Hz (magnitude: " + Number(halfMag.get([i1])).toFixed(1) + ")");`,
    },
  };

  const CDN_URLS = {
    numpyTs: "https://cdn.jsdelivr.net/npm/numpy-ts@0.13.1/dist/numpy-ts.browser.js",
    prismCore: "https://cdn.jsdelivr.net/npm/prismjs@1/prism.min.js",
    prismTS: "https://cdn.jsdelivr.net/npm/prismjs@1/components/prism-typescript.min.js",
    prismCSSLight: "https://cdn.jsdelivr.net/npm/prismjs@1/themes/prism.min.css",
    prismCSSDark: "https://cdn.jsdelivr.net/npm/prismjs@1/themes/prism-tomorrow.min.css",
  };

  const SHARED_FONT = {
    fontFamily: "'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace",
    fontSize: "13px",
    lineHeight: "1.5",
    tabSize: 2,
  };

  const THEME_COLORS = {
    light: {
      editorBg: '#f5f5f5',
      editorText: '#24292e',
      editorBorder: '#d1d5da',
      toolbarBg: '#ffffff',
      outputBg: '#fafbfc',
      outputText: '#24292e',
      selectBg: '#ffffff',
      selectText: '#24292e',
      selectBorder: '#d1d5da',
      caretColor: '#000000',
      placeholderText: '#6a737d',
      tabActiveBg: '#ffffff',
      tabInactiveBg: '#f6f8fa',
      tabHoverBg: '#e1e4e8',
      tabBorder: '#d1d5da',
      tabActiveText: '#24292e',
      tabInactiveText: '#586069',
      resizeHandleBg: '#d1d5da',
      resizeHandleHoverBg: '#959da5',
    },
    dark: {
      editorBg: '#1e1e1e',
      editorText: '#d4d4d4',
      editorBorder: '#333333',
      toolbarBg: '#161616',
      outputBg: '#1a1a1a',
      outputText: '#d4d4d4',
      selectBg: '#2a2a2a',
      selectText: '#cccccc',
      selectBorder: '#444444',
      caretColor: '#ffffff',
      placeholderText: '#666666',
      tabActiveBg: '#1e1e1e',
      tabInactiveBg: '#161616',
      tabHoverBg: '#2a2a2a',
      tabBorder: '#333333',
      tabActiveText: '#d4d4d4',
      tabInactiveText: '#888888',
      resizeHandleBg: '#333333',
      resizeHandleHoverBg: '#555555',
    }
  };

  const BASE_MIN_HEIGHT = 100;
  const MAX_HEIGHT = 800;
  const IMPORT_HEADER_TEXT = `import * as np from 'numpy-ts';\n\n`;
  const parseHeightPx = (value) => {
    if (value == null) return null;
    if (typeof value === "number" && Number.isFinite(value)) return value;
    if (typeof value === "string") {
      const parsed = parseFloat(value);
      if (Number.isFinite(parsed)) return parsed;
    }
    return null;
  };
  const minHeightPx = BASE_MIN_HEIGHT;
  const clampHeight = (h) => Math.max(minHeightPx, Math.min(MAX_HEIGHT, h));
  const estimateSingleCodeHeight = (script, includeImportHeader) => {
    const normalized = (script || "").replace(/\r\n/g, "\n");
    const lineCount = normalized.length > 0 ? normalized.split("\n").length : 1;
    const fontSizePx = parseFloat(SHARED_FONT.fontSize) || 13;
    const lineHeightRaw = parseFloat(SHARED_FONT.lineHeight);
    const lineHeightPx = (Number.isFinite(lineHeightRaw) ? lineHeightRaw : 1.5) * fontSizePx;
    const topPadding = includeImportHeader ? (16 + (3 * fontSizePx)) : 16;
    const bottomPadding = 16;
    return Math.ceil(topPadding + bottomPadding + (lineCount * lineHeightPx) + 4);
  };

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const existing = document.querySelector(`script[src="${src}"]`);
      if (existing) {
        if (existing.dataset.loaded === 'true') { resolve(); return; }
        existing.addEventListener('load', resolve, { once: true });
        existing.addEventListener('error', reject, { once: true });
        return;
      }
      const s = document.createElement("script");
      s.src = src;
      s.onload = () => { s.dataset.loaded = 'true'; resolve(); };
      s.onerror = reject;
      document.head.appendChild(s);
    });
  }

  function loadCSS(href, removeOldId = null) {
    if (document.querySelector(`link[href="${href}"]`)) return;
    if (removeOldId) {
      const old = document.querySelector(`link[data-prism-theme="${removeOldId}"]`);
      if (old) old.remove();
    }
    const l = document.createElement("link");
    l.rel = "stylesheet";
    l.href = href;
    if (removeOldId) l.setAttribute("data-prism-theme", removeOldId);
    document.head.appendChild(l);
  }

  const resolvedExamples = examples && typeof examples === "object" && Object.keys(examples).length > 0
    ? examples
    : (typeof singleCode === "string"
      ? { custom: { label: label || "", code: singleCode } }
      : DEFAULT_EXAMPLES);
  const exampleKeys = Object.keys(resolvedExamples);
  const fallbackKey = exampleKeys[0];
  const initialExampleKey = (defaultExample && resolvedExamples[defaultExample])
    ? defaultExample
    : ((example && resolvedExamples[example]) ? example : fallbackKey);
  const initialCode = resolvedExamples[initialExampleKey]?.code || "";
  const startHeightPx = parseHeightPx(startingHeight) ?? parseHeightPx(height);
  const initialEditorHeight = startHeightPx != null
    ? clampHeight(startHeightPx)
    : (typeof singleCode === "string"
      ? clampHeight(estimateSingleCodeHeight(singleCode, showImportHeader))
      : 340);

  const [code, setCode] = useState(initialCode);
  const [output, setOutput] = useState("");
  const [loaded, setLoaded] = useState(false);
  const [loadError, setLoadError] = useState(null);
  const [running, setRunning] = useState(false);
  const [selectedExample, setSelectedExample] = useState(initialExampleKey);
  const [isDarkMode, setIsDarkMode] = useState(
    () => document.documentElement.classList.contains('dark')
  );
  const [editorHeight, setEditorHeight] = useState(initialEditorHeight);
  const [isResizing, setIsResizing] = useState(false);
  const [copied, setCopied] = useState(false);
  const [timing, setTiming] = useState(null);
  const [copyHover, setCopyHover] = useState(false);
  const [scrollbarWidth, setScrollbarWidth] = useState(0);
  const textareaRef = useRef(null);
  const preRef = useRef(null);
  const codeRef = useRef(null);
  const resizeStartY = useRef(null);
  const resizeStartHeight = useRef(null);
  const runSeqRef = useRef(0);

  const colors = THEME_COLORS[isDarkMode ? 'dark' : 'light'];
  const copyTimeoutRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        loadCSS(isDarkMode ? CDN_URLS.prismCSSDark : CDN_URLS.prismCSSLight, "prism");
        await loadScript(CDN_URLS.numpyTs);
        await loadScript(CDN_URLS.prismCore);
        await loadScript(CDN_URLS.prismTS);
        if (!cancelled) setLoaded(true);
      } catch (e) {
        if (!cancelled) setLoadError("Failed to load dependencies. Check your connection.");
      }
    }
    load();
    return () => { cancelled = true; };
  }, []);

  const syncScroll = useCallback(() => {
    if (!textareaRef.current || !codeRef.current) return;
    const ta = textareaRef.current;
    codeRef.current.style.transform = `translate(${-ta.scrollLeft}px, ${-ta.scrollTop}px)`;
  }, []);

  useEffect(() => {
    // Watch for Mintlify theme changes (observes 'dark' class on html element)
    const observer = new MutationObserver(() => {
      const isDark = document.documentElement.classList.contains('dark');
      setIsDarkMode(isDark);
      loadCSS(isDark ? CDN_URLS.prismCSSDark : CDN_URLS.prismCSSLight, "prism");
      if (loaded && window.Prism) {
        setTimeout(() => {
          window.Prism.highlightAll();
          requestAnimationFrame(syncScroll);
        }, 50);
      }
    });

    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class']
    });

    return () => observer.disconnect();
  }, [loaded, syncScroll]);

  useEffect(() => {
    if (loaded && window.Prism) {
      window.Prism.highlightAll();
      requestAnimationFrame(syncScroll);
    }
  }, [code, loaded, syncScroll]);

  const handleScroll = useCallback(() => {
    syncScroll();
  }, [syncScroll]);

  const handleExampleChange = useCallback((key) => {
    setSelectedExample(key);
    setCode(resolvedExamples[key].code);
    setOutput("");

    // Reset scroll position
    if (textareaRef.current) {
      textareaRef.current.scrollTop = 0;
      textareaRef.current.scrollLeft = 0;
    }
    requestAnimationFrame(syncScroll);
  }, [resolvedExamples, syncScroll]);

  const handleKeyDown = useCallback((e) => {
    const ta = e.target;
    const start = ta.selectionStart;
    const end = ta.selectionEnd;
    const hasSelection = start !== end;

    if (e.key === "Tab") {
      e.preventDefault();
      const newVal = ta.value.substring(0, start) + "  " + ta.value.substring(end);
      setCode(newVal);
      requestAnimationFrame(() => { ta.selectionStart = ta.selectionEnd = start + 2; });
      return;
    }

    if (e.key === "Enter") {
      e.preventDefault();
      const lineStart = ta.value.lastIndexOf("\n", start - 1) + 1;
      const lineTextUpToCursor = ta.value.substring(lineStart, start);
      const indent = (lineTextUpToCursor.match(/^[ \t]*/) || [""])[0];
      const insert = `\n${indent}`;
      const newVal = ta.value.substring(0, start) + insert + ta.value.substring(end);
      const nextPos = start + insert.length;
      setCode(newVal);
      requestAnimationFrame(() => { ta.selectionStart = ta.selectionEnd = nextPos; });
      return;
    }

    if ((e.metaKey || e.ctrlKey) && !e.shiftKey && !e.altKey && e.key.toLowerCase() === "a") {
      e.preventDefault();
      ta.focus();
      ta.selectionStart = 0;
      ta.selectionEnd = ta.value.length;
      return;
    }

    if ((e.metaKey || e.ctrlKey) && !e.shiftKey && !e.altKey && e.key.toLowerCase() === "c") {
      e.preventDefault();
      const prevStart = ta.selectionStart;
      const prevEnd = ta.selectionEnd;
      let copyStart = prevStart;
      let copyEnd = prevEnd;

      if (!hasSelection) {
        const lineStart = ta.value.lastIndexOf("\n", start - 1) + 1;
        const lineEndRaw = ta.value.indexOf("\n", start);
        const lineEnd = lineEndRaw === -1 ? ta.value.length : lineEndRaw;
        copyStart = lineStart;
        copyEnd = lineEnd;
      }

      ta.focus();
      ta.selectionStart = copyStart;
      ta.selectionEnd = copyEnd;

      let copied = false;
      try {
        copied = typeof document.execCommand === "function" ? document.execCommand("copy") : false;
      } catch {
        copied = false;
      }

      if (!copied && navigator.clipboard?.writeText) {
        void navigator.clipboard.writeText(ta.value.substring(copyStart, copyEnd)).catch(() => {});
      }

      requestAnimationFrame(() => {
        ta.selectionStart = prevStart;
        ta.selectionEnd = prevEnd;
      });
      return;
    }

    if ((e.metaKey || e.ctrlKey) && !e.shiftKey && !e.altKey && (e.key === "/" || e.code === "Slash")) {
      e.preventDefault();
      const lineStart = ta.value.lastIndexOf("\n", start - 1) + 1;
      const endForLineCalc = hasSelection && ta.value[end - 1] === "\n" ? end - 1 : end;
      const lineEndRaw = ta.value.indexOf("\n", endForLineCalc);
      const lineEnd = lineEndRaw === -1 ? ta.value.length : lineEndRaw;
      const block = ta.value.substring(lineStart, lineEnd);
      const lines = block.split("\n");
      const isCommentedLine = (line) => /^(\s*)\/\//.test(line);
      const nonEmptyLines = lines.filter((line) => line.trim().length > 0);
      const shouldUncomment = nonEmptyLines.length > 0 && nonEmptyLines.every(isCommentedLine);

      const toggleLine = (line) => {
        if (line.trim().length === 0) return line;
        if (shouldUncomment) return line.replace(/^(\s*)\/\/ ?/, "$1");
        const match = line.match(/^(\s*)/);
        const indent = match ? match[1] : "";
        return `${indent}// ${line.slice(indent.length)}`;
      };

      const updatedLines = lines.map(toggleLine);
      const replacedBlock = updatedLines.join("\n");
      const newVal = ta.value.substring(0, lineStart) + replacedBlock + ta.value.substring(lineEnd);
      setCode(newVal);

      requestAnimationFrame(() => {
        if (hasSelection) {
          ta.selectionStart = lineStart;
          ta.selectionEnd = lineStart + replacedBlock.length;
          return;
        }

        const line = lines[0] || "";
        const cursorCol = start - lineStart;
        let newCursorCol = cursorCol;
        if (line.trim().length > 0) {
          if (shouldUncomment) {
            const uncommentMatch = line.match(/^(\s*)\/\/ ?/);
            if (uncommentMatch) {
              const indentLen = uncommentMatch[1].length;
              const removedLen = uncommentMatch[0].length - indentLen;
              if (cursorCol > indentLen) {
                newCursorCol = Math.max(indentLen, cursorCol - removedLen);
              }
            }
          } else {
            const indentLen = (line.match(/^(\s*)/) || [""])[0].length;
            if (cursorCol > indentLen) {
              newCursorCol = cursorCol + 3;
            }
          }
        }

        const newCursor = lineStart + newCursorCol;
        ta.selectionStart = ta.selectionEnd = newCursor;
      });
    }
  }, []);

  const handleResizeStart = useCallback((e) => {
    e.preventDefault();
    setIsResizing(true);
    resizeStartY.current = e.clientY;
    resizeStartHeight.current = editorHeight;
  }, [editorHeight]);

  useEffect(() => {
    if (!isResizing) return;

    const handleMouseMove = (e) => {
      const delta = e.clientY - resizeStartY.current;
      const newHeight = Math.max(minHeightPx, Math.min(MAX_HEIGHT, resizeStartHeight.current + delta));
      setEditorHeight(newHeight);
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizing, minHeightPx]);

  useEffect(() => {
    if (startHeightPx != null) {
      setEditorHeight(clampHeight(startHeightPx));
      return;
    }
    if (typeof singleCode === "string") {
      setEditorHeight(clampHeight(estimateSingleCodeHeight(singleCode, showImportHeader)));
    }
  }, [startHeightPx, singleCode, showImportHeader, minHeightPx]);

  const run = useCallback(async () => {
    if (!loaded || !window.np) return;
    const runId = runSeqRef.current + 1;
    runSeqRef.current = runId;
    setRunning(true);
    setTiming(null);
    const logs = [];
    const origLog = console.log;
    const origError = console.error;
    const origWarn = console.warn;
    const fmt = (...args) =>
      args.map((a) => {
        if (a == null) return String(a);
        if (typeof a === "object" && typeof a.toString === "function" && a.toString !== Object.prototype.toString) return a.toString();
        if (typeof a === "object") { try { return JSON.stringify(a); } catch { return String(a); } }
        return String(a);
      }).join(" ");

    console.log = (...args) => logs.push(fmt(...args));
    console.error = (...args) => logs.push("Error: " + fmt(...args));
    console.warn = (...args) => logs.push("Warning: " + fmt(...args));

    let hasError = false;
    const shouldRunAsync = /\bawait\b/.test(code);
    const t0 = performance.now();
    try {
      let result;
      if (shouldRunAsync) {
        const executeAsync = new Function("np", `"use strict"; return (async () => {\n${code}\n})();`);
        result = await executeAsync(window.np);
      } else {
        const executeSync = new Function("np", code);
        result = executeSync(window.np);
      }
      if (result !== undefined) {
        logs.push(typeof result === "object" && typeof result?.toString === "function" && result.toString !== Object.prototype.toString ? result.toString() : String(result));
      }
    } catch (e) {
      hasError = true;
      logs.push("Error: " + (e?.message || String(e)));
    } finally {
      console.log = origLog;
      console.error = origError;
      console.warn = origWarn;
    }

    const elapsed = performance.now() - t0;
    if (runId !== runSeqRef.current) return;

    setOutput(logs.join("\n"));
    if (showTiming && !hasError) {
      const formatted = elapsed < 0.15 ? "< 0.10 ms" : elapsed < 1 ? elapsed.toFixed(2) + " ms" : elapsed < 1000 ? elapsed.toFixed(1) + " ms" : (elapsed / 1000).toFixed(2) + " s";
      setTiming(formatted);
    }
    setRunning(false);
  }, [loaded, code, showTiming]);

  const handleCopy = useCallback(async () => {
    const textToCopy = `${showImportHeader ? IMPORT_HEADER_TEXT : ""}${code}`;
    try {
      await navigator.clipboard.writeText(textToCopy);
      setCopied(true);
      if (copyTimeoutRef.current) clearTimeout(copyTimeoutRef.current);
      copyTimeoutRef.current = setTimeout(() => setCopied(false), 1200);
    } catch {
      setCopied(false);
    }
  }, [code, showImportHeader]);

  useEffect(() => {
    return () => {
      if (copyTimeoutRef.current) clearTimeout(copyTimeoutRef.current);
    };
  }, []);

  useEffect(() => {
    const measureScrollbar = () => {
      if (!textareaRef.current) return;
      const el = textareaRef.current;
      const width = Math.max(el.offsetWidth - el.clientWidth, 0);
      setScrollbarWidth(width);
    };

    measureScrollbar();
    window.addEventListener('resize', measureScrollbar);
    return () => window.removeEventListener('resize', measureScrollbar);
  }, [editorHeight, code, showImportHeader]);

  if (loadError) {
    return (
      <div style={{ padding: "20px", color: "#e55", background: colors.outputBg, borderRadius: "8px", fontSize: "14px" }}>
        {loadError}
      </div>
    );
  }

  const tabsContainer = { display: "flex", gap: "2px", marginBottom: "-1px", marginTop: "8px", overflowX: "auto", overflowY: "hidden" };
  const tabStyle = (isActive) => ({
    padding: "8px 16px",
    background: isActive ? colors.tabActiveBg : colors.tabInactiveBg,
    color: isActive ? colors.tabActiveText : colors.tabInactiveText,
    border: `1px solid ${colors.tabBorder}`,
    borderBottom: isActive ? "none" : `1px solid ${colors.tabBorder}`,
    borderRadius: "8px 8px 0 0",
    fontSize: "13px",
    fontWeight: isActive ? 500 : 400,
    cursor: "pointer",
    outline: "none",
    transition: "background 0.15s, color 0.15s",
    whiteSpace: "nowrap",
    marginBottom: isActive ? "1px" : "0"
  });
  const editorRadius = exampleKeys.length > 1 ? "0" : "8px 8px 0 0";
  const editorContentRadius = exampleKeys.length > 1 ? "0" : "8px 8px 0 0";
  const editorWrap = { position: "relative", height: `${editorHeight}px`, overflow: "hidden" };
  const editorBorderOverlay = { position: "absolute", inset: 0, borderRadius: editorRadius, border: `1px solid ${colors.editorBorder}`, borderBottom: "none", borderTop: exampleKeys.length > 1 ? "none" : `1px solid ${colors.editorBorder}`, pointerEvents: "none", zIndex: 4 };
  const copyWrapStyle = { position: "absolute", top: "8px", right: `${10 + scrollbarWidth}px`, zIndex: 3, display: "inline-flex", alignItems: "center", gap: "6px" };
  const copyTooltipVisible = copied || copyHover;
  const copyTooltipStyle = {
    fontSize: "11px",
    lineHeight: 1,
    color: "#ffffff",
    background: "#3b82f6",
    border: "1px solid #3b82f6",
    borderRadius: "6px",
    padding: "4px 8px",
    opacity: copyTooltipVisible ? 1 : 0,
    transform: copyTooltipVisible ? "translateY(0)" : "translateY(2px)",
    transition: "opacity 0.15s, transform 0.15s",
    pointerEvents: "none",
    whiteSpace: "nowrap"
  };
  const copyBtnStyle = {
    width: "24px",
    height: "24px",
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    borderRadius: "6px",
    border: "none",
    background: "transparent",
    color: isDarkMode ? "rgba(255,255,255,0.4)" : "#9ca3af",
    cursor: "pointer",
    padding: 0,
    lineHeight: 1
  };
  const sharedTextStyle = { margin: 0, border: 0, padding: "16px", whiteSpace: "pre", overflowWrap: "normal", wordBreak: "normal", verticalAlign: "baseline", textRendering: "auto" };
  const preStyle = { ...SHARED_FONT, ...sharedTextStyle, position: "absolute", top: 0, left: 0, right: 0, bottom: 0, background: colors.editorBg, overflow: "hidden", pointerEvents: "none", padding: 0, borderRadius: editorContentRadius };
  const codeStyle = { display: "block", ...SHARED_FONT, background: "transparent", padding: "16px", margin: 0, border: 0, whiteSpace: "pre", overflowWrap: "normal", wordBreak: "normal", lineHeight: "inherit", willChange: "transform" };
  const textareaStyle = { ...SHARED_FONT, ...sharedTextStyle, position: "absolute", top: 0, left: 0, right: 0, bottom: 0, paddingTop: showImportHeader ? "calc(16px + 3em)" : "16px", background: "transparent", color: "transparent", caretColor: colors.caretColor, outline: "none", resize: "none", overflow: "auto", WebkitTextFillColor: "transparent", zIndex: 1, width: "100%", height: "100%", borderRadius: editorContentRadius };
  const resizeHandle = { height: "4px", background: isResizing ? colors.resizeHandleHoverBg : colors.resizeHandleBg, cursor: "ns-resize", transition: "background 0.15s", borderLeft: `1px solid ${colors.editorBorder}`, borderRight: `1px solid ${colors.editorBorder}` };
  const toolbarStyle = { display: "flex", alignItems: "center", gap: "8px", padding: "8px 12px", background: colors.toolbarBg, borderLeft: `1px solid ${colors.editorBorder}`, borderRight: `1px solid ${colors.editorBorder}`, flexWrap: "wrap" };
  const btnStyle = { display: "inline-flex", alignItems: "center", gap: "6px", padding: "6px 16px", background: "#3179C7", color: "#fff", border: "none", borderRadius: "6px", fontSize: "13px", fontWeight: 500, cursor: loaded ? "pointer" : "not-allowed", opacity: loaded ? 1 : 0.5, transition: "background 0.15s" };
  const selectStyle = { padding: "6px 10px", background: colors.selectBg, color: colors.selectText, border: `1px solid ${colors.selectBorder}`, borderRadius: "6px", fontSize: "13px", cursor: "pointer", outline: "none" };
  const badgeStyle = { marginLeft: "auto", fontSize: "11px", color: colors.placeholderText, display: "flex", alignItems: "center", gap: "4px", whiteSpace: "nowrap" };
  const outputStyle = { margin: 0, padding: "16px", minHeight: "60px", maxHeight: "240px", overflow: "auto", ...SHARED_FONT, fontSize: "12.5px", background: colors.outputBg, color: colors.outputText, borderRadius: "0 0 8px 8px", border: `1px solid ${colors.editorBorder}`, borderTop: "none", whiteSpace: "pre-wrap", overflowWrap: "break-word" };
  const renderedCode = `${showImportHeader ? IMPORT_HEADER_TEXT : ""}${code}`;
  const singleLabel = (resolvedExamples[exampleKeys[0]]?.label || "").trim();
  const runSpinnerStyle = { animation: "playgroundSpin 0.9s linear infinite", display: "inline-block" };

  return (
    <div>
      <style>{`
        @keyframes playgroundSpin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        @keyframes playgroundPulse {
          0%, 100% { opacity: 0.45; transform: scale(1); }
          50% { opacity: 1; transform: scale(1.12); }
        }
        code[class*="language-"] {
          white-space: pre !important;
          word-break: normal !important;
          overflow-wrap: normal !important;
        }
      `}</style>
      {exampleKeys.length > 1 ? (
        <div style={tabsContainer}>
          {Object.entries(resolvedExamples).map(([key, val]) => (
            <button
              key={key}
              onClick={() => handleExampleChange(key)}
              style={tabStyle(key === selectedExample)}
              onMouseEnter={(e) => {
                if (key !== selectedExample) {
                  e.target.style.background = colors.tabHoverBg;
                }
              }}
              onMouseLeave={(e) => {
                if (key !== selectedExample) {
                  e.target.style.background = colors.tabInactiveBg;
                }
              }}
              aria-label={`Select ${val.label} example`}
            >
              {val.label}
            </button>
          ))}
        </div>
      ) : (
        singleLabel ? (
          <div style={{ marginTop: "8px", marginBottom: "8px", fontSize: "14px", fontWeight: 500, color: colors.tabActiveText }}>
            {singleLabel}
          </div>
        ) : null
      )}
      <div style={editorWrap}>
        <div style={editorBorderOverlay} aria-hidden="true" />
        {showCopyButton ? (
          <div style={copyWrapStyle}>
            <span style={copyTooltipStyle}>{copied ? "Copied" : "Copy"}</span>
            <button
              type="button"
              onClick={handleCopy}
              style={copyBtnStyle}
              onMouseEnter={(e) => {
                setCopyHover(true);
                e.currentTarget.style.color = isDarkMode ? "rgba(255,255,255,0.6)" : "#6b7280";
              }}
              onMouseLeave={(e) => {
                setCopyHover(false);
                e.currentTarget.style.color = isDarkMode ? "rgba(255,255,255,0.4)" : "#9ca3af";
              }}
              aria-label="Copy code"
            >
              {copied ? (
                <svg width="18" height="18" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                  <path d="M4.5 10.5L8.2 14.2L15.5 6.8" stroke="#3b82f6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              ) : (
                <svg width="18" height="18" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                  <path d="M14.25 5.25H7.25C6.14543 5.25 5.25 6.14543 5.25 7.25V14.25C5.25 15.3546 6.14543 16.25 7.25 16.25H14.25C15.3546 16.25 16.25 15.3546 16.25 14.25V7.25C16.25 6.14543 15.3546 5.25 14.25 5.25Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  <path d="M2.80103 11.998L1.77203 5.07397C1.61003 3.98097 2.36403 2.96397 3.45603 2.80197L10.38 1.77297C11.313 1.63397 12.19 2.16297 12.528 3.00097" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              )}
            </button>
          </div>
        ) : null}
        <pre ref={preRef} style={preStyle}>
          <code ref={codeRef} className="language-typescript" style={codeStyle}>{renderedCode}</code>
        </pre>
        <textarea
          ref={textareaRef}
          value={code}
          onChange={(e) => setCode(e.target.value)}
          onScroll={handleScroll}
          onKeyDown={handleKeyDown}
          style={textareaStyle}
          spellCheck={false}
          wrap="off"
          autoCapitalize="off"
          autoCorrect="off"
          aria-label="Code editor"
        />
      </div>
      <div
        style={resizeHandle}
        onMouseDown={handleResizeStart}
        onMouseEnter={(e) => !isResizing && (e.target.style.background = colors.resizeHandleHoverBg)}
        onMouseLeave={(e) => !isResizing && (e.target.style.background = colors.resizeHandleBg)}
        aria-label="Resize editor"
        role="separator"
      />
      <div style={toolbarStyle}>
        <button
          onClick={run}
          disabled={!loaded}
          style={btnStyle}
          onMouseEnter={(e) => loaded && (e.target.style.background = "#2668b0")}
          onMouseLeave={(e) => loaded && (e.target.style.background = "#3179C7")}
        >
          {running ? <><span style={runSpinnerStyle}>{"\u21BB"}</span> Run</> : <><span>{"\u25B6"}</span> Run</>}
        </button>
        {timing && !running && (
          <span style={{ fontSize: "12px", color: "#4caf50", fontWeight: 500 }}>
            Completed in {timing}
          </span>
        )}
        <span style={badgeStyle}>
          {loaded ? (
            <><span style={{ color: "#4caf50", display: "inline-block", animation: "playgroundPulse 2.2s ease-in-out infinite", transformOrigin: "center" }}>{"\u25CF"}</span> Runs in your browser</>
          ) : (
            <><span style={{ animation: "spin 1s linear infinite", display: "inline-block" }}>{"\u21BB"}</span> Loading numpy-ts...</>
          )}
        </span>
      </div>
      <pre style={outputStyle}>
        {output || <span style={{ color: colors.placeholderText, fontStyle: "italic" }}>Click Run to execute the code above</span>}
      </pre>
    </div>
  );
};
