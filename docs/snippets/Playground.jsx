export const Playground = ({ example = "quickstart", height = "340px" }) => {
  const EXAMPLES = {
    quickstart: {
      label: "Quickstart",
      code: `// Create arrays
const a = np.array([[1, 2, 3], [4, 5, 6]]);
console.log("Array:\\n" + a);
console.log("Shape:", a.shape);
console.log("Dtype:", a.dtype);

// Basic arithmetic
const b = np.multiply(a, 2);
console.log("\\nMultiply by 2:\\n" + b);

// Element-wise operations
const c = np.add(a, np.array([[10, 20, 30], [40, 50, 60]]));
console.log("Element-wise add:\\n" + c);

// Row slicing
const row = a.row(0);
console.log("\\nFirst row:", row);`,
    },
    linalg: {
      label: "Linear Algebra",
      code: `// Matrix multiplication
const A = np.array([[1, 2], [3, 4]]);
const B = np.array([[5, 6], [7, 8]]);
const C = np.matmul(A, B);
console.log("A @ B =\\n" + C);

// Determinant
const det = np.linalg.det(A);
console.log("\\ndet(A) =", det);

// Inverse
const inv = np.linalg.inv(A);
console.log("inv(A) =\\n" + inv);

// Verify A @ inv(A) = I
const I = np.matmul(A, inv);
console.log("\\nA @ inv(A) =\\n" + I);

// Eigenvalues
const eig = np.linalg.eig(A);
console.log("\\nEigenvalues:", eig.eigenvalues);`,
    },
    broadcasting: {
      label: "Broadcasting",
      code: `// Broadcasting: different shapes work together
const matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
const row = np.array([10, 20, 30]);

// Add row to each row of matrix
const result = np.add(matrix, row);
console.log("Matrix + row vector:");
console.log(result);

// Column vector broadcasting
const col = np.array([[100], [200], [300]]);
const result2 = np.add(matrix, col);
console.log("\\nMatrix + column vector:");
console.log(result2);

// Scalar broadcasting
console.log("\\nMatrix * 10:");
console.log(np.multiply(matrix, 10));`,
    },
    random: {
      label: "Random",
      code: `// Seeded random for reproducibility
np.random.seed(42);

// Random integers
const dice = np.random.randint(1, 7, [3, 3]);
console.log("Dice rolls (3x3):\\n" + dice);

// Normal distribution
const normal = np.random.randn(5);
console.log("\\nNormal samples:", normal);

// Uniform distribution
const uniform = np.random.rand(2, 3);
console.log("\\nUniform [0,1):\\n" + uniform);

// Statistics of random data
const data = np.random.randn(1000);
console.log("\\n1000 normal samples:");
console.log("  Mean:", Number(np.mean(data)).toFixed(3));
console.log("  Std:", Number(np.std(data)).toFixed(3));`,
    },
    reductions: {
      label: "Reductions",
      code: `const a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
console.log("Array:\\n" + a);

// Global reductions
console.log("\\nSum:", Number(np.sum(a)));
console.log("Mean:", Number(np.mean(a)));
console.log("Std:", Number(np.std(a)).toFixed(4));
console.log("Min:", Number(np.min(a)));
console.log("Max:", Number(np.max(a)));

// Axis reductions
console.log("\\nSum along axis 0 (columns):", np.sum(a, 0));
console.log("Sum along axis 1 (rows):", np.sum(a, 1));
console.log("Mean along axis 0:", np.mean(a, 0));

// Cumulative
console.log("\\nCumsum:", np.cumsum(a));
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
console.log("Signal (first 8):", signal.slice('0:8'));

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
console.log("\\nTop frequencies:");
console.log("  " + Number(halfFreqs.get([i0])) + " Hz (magnitude: " + Number(halfMag.get([i0])).toFixed(1) + ")");
console.log("  " + Number(halfFreqs.get([i1])) + " Hz (magnitude: " + Number(halfMag.get([i1])).toFixed(1) + ")");`,
    },
  };

  const CDN_URLS = {
    numpyTs: "https://cdn.jsdelivr.net/npm/numpy-ts@0.13.0/dist/numpy-ts.browser.js",
    prismCore: "https://cdn.jsdelivr.net/npm/prismjs@1/prism.min.js",
    prismTS: "https://cdn.jsdelivr.net/npm/prismjs@1/components/prism-typescript.min.js",
    prismCSS: "https://cdn.jsdelivr.net/npm/prismjs@1/themes/prism-tomorrow.min.css",
  };

  const SHARED_FONT = {
    fontFamily: "'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace",
    fontSize: "13px",
    lineHeight: "1.5",
    tabSize: 2,
  };

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      if (document.querySelector(`script[src="${src}"]`)) { resolve(); return; }
      const s = document.createElement("script");
      s.src = src;
      s.onload = resolve;
      s.onerror = reject;
      document.head.appendChild(s);
    });
  }

  function loadCSS(href) {
    if (document.querySelector(`link[href="${href}"]`)) return;
    const l = document.createElement("link");
    l.rel = "stylesheet";
    l.href = href;
    document.head.appendChild(l);
  }

  const [code, setCode] = useState(EXAMPLES[example]?.code || EXAMPLES.quickstart.code);
  const [output, setOutput] = useState("");
  const [loaded, setLoaded] = useState(false);
  const [loadError, setLoadError] = useState(null);
  const [running, setRunning] = useState(false);
  const [selectedExample, setSelectedExample] = useState(example);
  const textareaRef = useRef(null);
  const preRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        loadCSS(CDN_URLS.prismCSS);
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

  useEffect(() => {
    if (loaded && window.Prism) {
      window.Prism.highlightAll();
    }
  }, [code, loaded]);

  const handleScroll = useCallback(() => {
    if (preRef.current && textareaRef.current) {
      preRef.current.scrollTop = textareaRef.current.scrollTop;
      preRef.current.scrollLeft = textareaRef.current.scrollLeft;
    }
  }, []);

  const handleExampleChange = useCallback((key) => {
    setSelectedExample(key);
    setCode(EXAMPLES[key].code);
    setOutput("");
  }, []);

  const handleTab = useCallback((e) => {
    if (e.key === "Tab") {
      e.preventDefault();
      const ta = e.target;
      const start = ta.selectionStart;
      const end = ta.selectionEnd;
      const newVal = ta.value.substring(0, start) + "  " + ta.value.substring(end);
      setCode(newVal);
      requestAnimationFrame(() => { ta.selectionStart = ta.selectionEnd = start + 2; });
    }
  }, []);

  const run = useCallback(() => {
    if (!loaded || !window.np) return;
    setRunning(true);
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
    try {
      const result = new Function("np", code)(window.np);
      if (result !== undefined) {
        logs.push(typeof result === "object" && typeof result?.toString === "function" && result.toString !== Object.prototype.toString ? result.toString() : String(result));
      }
    } catch (e) {
      logs.push("Error: " + e.message);
    }
    console.log = origLog;
    console.error = origError;
    console.warn = origWarn;
    setOutput(logs.join("\n"));
    setRunning(false);
  }, [loaded, code]);

  if (loadError) {
    return (
      <div style={{ padding: "20px", color: "#e55", background: "#1a1a1a", borderRadius: "8px", fontSize: "14px" }}>
        {loadError}
      </div>
    );
  }

  const editorWrap = { position: "relative", height, borderRadius: "8px 8px 0 0", overflow: "hidden", border: "1px solid #333", borderBottom: "none" };
  const preStyle = { ...SHARED_FONT, position: "absolute", top: 0, left: 0, right: 0, bottom: 0, margin: 0, padding: "16px", background: "#1e1e1e", overflow: "auto", pointerEvents: "none", whiteSpace: "pre-wrap", wordWrap: "break-word" };
  const codeStyle = { ...SHARED_FONT, background: "transparent", padding: 0, margin: 0, whiteSpace: "pre-wrap", wordWrap: "break-word" };
  const textareaStyle = { ...SHARED_FONT, position: "absolute", top: 0, left: 0, right: 0, bottom: 0, padding: "16px", margin: 0, background: "transparent", color: "transparent", caretColor: "#fff", border: "none", outline: "none", resize: "none", overflow: "auto", whiteSpace: "pre-wrap", wordWrap: "break-word", WebkitTextFillColor: "transparent", zIndex: 1, width: "100%", height: "100%" };
  const toolbarStyle = { display: "flex", alignItems: "center", gap: "8px", padding: "8px 12px", background: "#161616", borderLeft: "1px solid #333", borderRight: "1px solid #333", flexWrap: "wrap" };
  const btnStyle = { display: "inline-flex", alignItems: "center", gap: "6px", padding: "6px 16px", background: "#3179C7", color: "#fff", border: "none", borderRadius: "6px", fontSize: "13px", fontWeight: 500, cursor: loaded ? "pointer" : "not-allowed", opacity: loaded ? 1 : 0.5, transition: "background 0.15s" };
  const selectStyle = { padding: "6px 10px", background: "#2a2a2a", color: "#ccc", border: "1px solid #444", borderRadius: "6px", fontSize: "13px", cursor: "pointer", outline: "none" };
  const badgeStyle = { marginLeft: "auto", fontSize: "11px", color: "#888", display: "flex", alignItems: "center", gap: "4px", whiteSpace: "nowrap" };
  const outputStyle = { margin: 0, padding: "16px", minHeight: "60px", maxHeight: "240px", overflow: "auto", ...SHARED_FONT, fontSize: "12.5px", background: "#1a1a1a", color: "#d4d4d4", borderRadius: "0 0 8px 8px", border: "1px solid #333", borderTop: "none", whiteSpace: "pre-wrap", wordWrap: "break-word" };

  return (
    <div style={{ marginTop: "8px" }}>
      <div style={editorWrap}>
        <pre ref={preRef} style={preStyle}>
          <code className="language-typescript" style={codeStyle}>{code}</code>
        </pre>
        <textarea
          ref={textareaRef}
          value={code}
          onChange={(e) => setCode(e.target.value)}
          onScroll={handleScroll}
          onKeyDown={handleTab}
          style={textareaStyle}
          spellCheck={false}
          autoCapitalize="off"
          autoCorrect="off"
          aria-label="Code editor"
        />
      </div>
      <div style={toolbarStyle}>
        <button
          onClick={run}
          disabled={!loaded}
          style={btnStyle}
          onMouseEnter={(e) => loaded && (e.target.style.background = "#2668b0")}
          onMouseLeave={(e) => loaded && (e.target.style.background = "#3179C7")}
        >
          {running ? "..." : "\u25B6"} Run
        </button>
        <select
          value={selectedExample}
          onChange={(e) => handleExampleChange(e.target.value)}
          style={selectStyle}
          aria-label="Select example"
        >
          {Object.entries(EXAMPLES).map(([key, val]) => (
            <option key={key} value={key}>{val.label}</option>
          ))}
        </select>
        <span style={badgeStyle}>
          {loaded ? (
            <><span style={{ color: "#4caf50" }}>{"\u25CF"}</span> Runs in your browser</>
          ) : (
            <><span style={{ animation: "spin 1s linear infinite", display: "inline-block" }}>{"\u21BB"}</span> Loading numpy-ts...</>
          )}
        </span>
      </div>
      <pre style={outputStyle}>
        {output || <span style={{ color: "#666", fontStyle: "italic" }}>Click Run to execute the code above</span>}
      </pre>
    </div>
  );
};
