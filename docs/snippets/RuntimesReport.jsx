export const RuntimesReport = ({ data, detailUrl }) => {
  const meta = data?.meta || {};
  const runtimes = Array.isArray(data?.runtimes) ? data.runtimes : [];
  const summaries = data?.summaries || {};
  const categories = Array.isArray(data?.categories) ? data.categories : [];

  const RUNTIME_COLORS = {
    node: '#22c55e',
    deno: '#3b82f6',
    bun: '#f97316',
  };

  const RUNTIME_LABELS = {
    node: 'Node.js',
    deno: 'Deno',
    bun: 'Bun',
  };

  const THEME_COLORS = {
    light: {
      cardBg: '#ffffff',
      mutedCardBg: '#fafafa',
      border: '#e5e7eb',
      text: '#111827',
      mutedText: '#6b7280',
      chartTrack: '#f3f4f6',
      tableHeadBg: '#f9fafb',
      tableHeadText: '#374151',
      tableRowBorder: '#e5e7eb',
      tableRowAlt: '#fafafa',
      ratioGoodText: '#1f9d55',
      ratioGoodBg: '#dcfce7',
      ratioOkText: '#b7791f',
      ratioOkBg: '#fef3c7',
      ratioBadText: '#c53030',
      ratioBadBg: '#fee2e2',
      chartGood: '#22c55e',
      chartOk: '#f59e0b',
      chartBad: '#ef4444',
    },
    dark: {
      cardBg: '#1e1e1e',
      mutedCardBg: '#1a1a1a',
      border: '#333333',
      text: '#d4d4d4',
      mutedText: '#888888',
      chartTrack: '#2a2a2a',
      tableHeadBg: '#161616',
      tableHeadText: '#d4d4d4',
      tableRowBorder: '#333333',
      tableRowAlt: '#161616',
      ratioGoodText: '#86efac',
      ratioGoodBg: '#14532d',
      ratioOkText: '#fde68a',
      ratioOkBg: '#78350f',
      ratioBadText: '#fca5a5',
      ratioBadBg: '#7f1d1d',
      chartGood: '#22c55e',
      chartOk: '#f59e0b',
      chartBad: '#ef4444',
    },
  };

  const DTYPE_COLORS = {
    float64:    { bg: '#3b6cc9', text: '#ffffff' },
    float32:    { bg: '#4a80d8', text: '#ffffff' },
    float16:    { bg: '#6196e2', text: '#ffffff' },
    complex128: { bg: '#8b5cc6', text: '#ffffff' },
    complex64:  { bg: '#a478d4', text: '#ffffff' },
    int64:      { bg: '#2a8a82', text: '#ffffff' },
    int32:      { bg: '#3ba69d', text: '#ffffff' },
    int16:      { bg: '#55bfb7', text: '#1a4a46' },
    int8:       { bg: '#7ad4cd', text: '#1a4a46' },
    uint64:     { bg: '#c98a2e', text: '#ffffff' },
    uint32:     { bg: '#d9a044', text: '#3a2400' },
    uint16:     { bg: '#e4b85e', text: '#3a2400' },
    uint8:      { bg: '#eecb7c', text: '#3a2400' },
    bool:       { bg: '#cc4466', text: '#ffffff' },
  };

  const DTYPE_RE = /\s+(float64|float32|float16|complex128|complex64|int64|int32|int16|int8|uint64|uint32|uint16|uint8|bool)$/;

  const [isDarkMode, setIsDarkMode] = useState(() =>
    document.documentElement.classList.contains('dark')
  );
  const [openCategories, setOpenCategories] = useState({});
  const [isMobile, setIsMobile] = useState(() => window.innerWidth < 640);
  const [isNarrow, setIsNarrow] = useState(() => window.innerWidth < 900);
  const [metaOpen, setMetaOpen] = useState(false);

  // Detail benchmarks: prefetched in the background after first paint so opening
  // a drawer is instant. The worker fetches, parses, AND indexes the JSON into a
  // { categoryName: benchmarks[] } map so the main thread does no prep work.
  // Browser HTTP cache handles cross-page-load reuse.
  const [detailMap, setDetailMap] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const detailFetched = useRef(false);

  const fetchDetail = () => {
    if (detailFetched.current || !detailUrl) return;
    detailFetched.current = true;
    setDetailLoading(true);
    const absUrl = new URL(detailUrl, window.location.href).href;
    const buildMap = (d) => {
      const cats = (d && d.categories) || [];
      const m = {};
      for (const c of cats) m[c.name] = c.benchmarks || [];
      return m;
    };
    const fallback = () => {
      fetch(absUrl)
        .then((r) => r.json())
        .then((d) => setDetailMap(buildMap(d)))
        .catch(() => { detailFetched.current = false; })
        .finally(() => setDetailLoading(false));
    };
    try {
      if (typeof Worker === 'undefined' || typeof Blob === 'undefined') return fallback();
      const code = "self.onmessage=async e=>{try{const{url,key}=e.data;const r=await fetch(url);const d=await r.json();const cats=(d&&d[key])||[];const m={};for(const c of cats)m[c.name]=c.benchmarks||[];self.postMessage({ok:true,map:m});}catch(err){self.postMessage({ok:false});}}";
      const blobUrl = URL.createObjectURL(new Blob([code], { type: 'application/javascript' }));
      const worker = new Worker(blobUrl);
      const cleanup = () => { worker.terminate(); URL.revokeObjectURL(blobUrl); };
      worker.onmessage = (e) => {
        if (e.data && e.data.ok) setDetailMap(e.data.map);
        else detailFetched.current = false;
        setDetailLoading(false);
        cleanup();
      };
      worker.onerror = () => { detailFetched.current = false; setDetailLoading(false); cleanup(); };
      worker.postMessage({ url: absUrl, key: 'categories' });
    } catch { fallback(); }
  };

  useEffect(() => {
    if (!detailUrl) return;
    let ricId, timeoutId;
    const schedule = () => {
      const ric = window.requestIdleCallback;
      if (ric) ricId = ric(() => fetchDetail(), { timeout: 2000 });
      else timeoutId = setTimeout(fetchDetail, 200);
    };
    if (document.readyState === 'complete') schedule();
    else window.addEventListener('load', schedule, { once: true });
    return () => {
      window.removeEventListener('load', schedule);
      if (ricId && window.cancelIdleCallback) window.cancelIdleCallback(ricId);
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, []);

  const toggleCategory = (name) => {
    fetchDetail();
    setOpenCategories((prev) => ({ ...prev, [name]: !prev[name] }));
  };

  useEffect(() => {
    const observer = new MutationObserver(() => {
      setIsDarkMode(document.documentElement.classList.contains('dark'));
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const handler = () => {
      setIsMobile(window.innerWidth < 640);
      setIsNarrow(window.innerWidth < 900);
    };
    window.addEventListener('resize', handler);
    return () => window.removeEventListener('resize', handler);
  }, []);

  const colors = THEME_COLORS[isDarkMode ? 'dark' : 'light'];

  // HoverBar tooltip uses position: fixed with computed viewport coords so it
  // escapes every ancestor stacking context (Mintlify's layout has parents
  // with their own contexts that otherwise win over our z-index after a beat).
  // The brightness `filter` lives on an inner bar div — applying it to the
  // outer element would create a containing block for fixed descendants and
  // re-trap the tooltip. useMemo with [] deps keeps the component identity
  // stable across parent re-renders.
  const HoverBar = useMemo(() => function HoverBar({ tip, width, color, rounded, isDarkMode }) {
    const [show, setShow] = useState(false);
    const [pos, setPos] = useState(null);
    const ref = useRef(null);
    const lastPointerType = useRef('mouse');
    useEffect(() => {
      if (!show) { setPos(null); return; }
      const updatePos = () => {
        if (!ref.current) return;
        const r = ref.current.getBoundingClientRect();
        setPos({ top: r.top, left: r.left + r.width / 2 });
      };
      updatePos();
      const onDocPointerDown = (e) => {
        if (ref.current && !ref.current.contains(e.target)) setShow(false);
      };
      document.addEventListener('pointerdown', onDocPointerDown);
      window.addEventListener('scroll', updatePos, true);
      window.addEventListener('resize', updatePos);
      return () => {
        document.removeEventListener('pointerdown', onDocPointerDown);
        window.removeEventListener('scroll', updatePos, true);
        window.removeEventListener('resize', updatePos);
      };
    }, [show]);
    return (
      <div ref={ref} style={{ height: '100%', width, position: 'relative', cursor: 'default' }}
        onPointerEnter={(e) => { if (e.pointerType === 'mouse') setShow(true); }}
        onPointerLeave={(e) => { if (e.pointerType === 'mouse') setShow(false); }}
        onPointerDown={(e) => { lastPointerType.current = e.pointerType; }}
        onClick={() => { if (lastPointerType.current !== 'mouse') setShow((s) => !s); }}>
        <div style={{ height: '100%', width: '100%', background: color, borderRadius: rounded ? 7 : 0, filter: show ? 'brightness(1.3)' : 'none', transition: 'filter 0.15s' }} />
        {tip && show && pos && (
          <div style={{ position: 'fixed', top: pos.top - 6, left: pos.left, transform: 'translate(-50%, -100%)', padding: '4px 8px', borderRadius: 6, background: isDarkMode ? '#2a2a2a' : '#111', color: '#fff', fontSize: 11, whiteSpace: 'nowrap', zIndex: 2147483647, pointerEvents: 'none', boxShadow: '0 2px 8px rgba(0,0,0,0.4)' }}>
            {tip}
          </div>
        )}
      </div>
    );
  }, []);

  const formatOps = (ops) => {
    if (!Number.isFinite(ops) || ops <= 0) return '-';
    if (ops >= 1_000_000_000) return `${(ops / 1_000_000_000).toFixed(2)}B/s`;
    if (ops >= 1_000_000) return `${(ops / 1_000_000).toFixed(2)}M/s`;
    if (ops >= 1_000) return `${(ops / 1_000).toFixed(2)}K/s`;
    return `${ops.toFixed(1)}/s`;
  };

  const formatRatio = (ratio) => {
    if (!Number.isFinite(ratio)) return '-';
    return `${ratio.toFixed(2)}x`;
  };

  const ratioTip = (ratio, rt) => {
    if (!Number.isFinite(ratio)) return '';
    const label = rt ? (RUNTIME_LABELS[rt] || rt) + ': ' : '';
    return label + (ratio >= 1 ? `${ratio.toFixed(2)}x faster than Node.js` : `${ratio.toFixed(2)}x slower than Node.js`);
  };

  const formatOpsShort = (ops) => {
    if (!Number.isFinite(ops) || ops <= 0) return '';
    if (ops >= 1_000_000) return `${(ops / 1_000_000).toFixed(1)}M ops/s`;
    if (ops >= 1_000) return `${(ops / 1_000).toFixed(1)}K ops/s`;
    return `${ops.toFixed(0)} ops/s`;
  };


  const RatioDisplay = ({ ratio }) => {
    if (!Number.isFinite(ratio)) return <span>-</span>;
    return (
      <span>
        <span style={{ fontSize: 20, fontWeight: 700, color: colors.text }}>{ratio.toFixed(2)}x</span>
        <span style={{ fontSize: 11, fontWeight: 400, color: colors.mutedText, marginLeft: 4 }}>{ratio >= 1 ? 'faster than Node.js' : 'slower than Node.js'}</span>
      </span>
    );
  };

  const ratioColor = (ratio) => {
    if (ratio >= 1) return colors.ratioGoodText;
    if (ratio >= 0.75) return colors.ratioOkText;
    return colors.ratioBadText;
  };

  const ratioBg = (ratio) => {
    if (ratio >= 1) return colors.ratioGoodBg;
    if (ratio >= 0.75) return colors.ratioOkBg;
    return colors.ratioBadBg;
  };

  const BenchmarkName = ({ name }) => {
    const s = String(name);
    const m = s.match(DTYPE_RE);
    const dtype = m ? m[1] : 'float64';
    const base = m ? s.slice(0, -m[0].length) : s;
    const dc = DTYPE_COLORS[dtype] || DTYPE_COLORS.float64;
    return (
      <>
        {base}{' '}
        <span style={{ display: 'inline-block', fontSize: '0.72em', fontWeight: 600, padding: '2px 8px', borderRadius: 999, background: dc.bg, color: dc.text, verticalAlign: 'middle', whiteSpace: 'nowrap' }}>
          {dtype}
        </span>
      </>
    );
  };

  const SummaryCard = ({ label, color, data }) => (
    <div style={{ border: `1px solid ${colors.border}`, borderRadius: 10, padding: 14, background: colors.cardBg}}>

      <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 8, color }}>{label}</div>
      <div style={{ marginBottom: 8 }}>
        <div style={{ fontSize: 11, color: colors.mutedText }}>Average</div>
        <RatioDisplay ratio={data.avgSpeedup} />
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
        <div>
          <div style={{ fontSize: 11, color: colors.mutedText }}>Best</div>
          <div style={{ fontSize: 14, fontWeight: 600, color: colors.text }}>{formatRatio(data.bestCase)}</div>
        </div>
        <div>
          <div style={{ fontSize: 11, color: colors.mutedText }}>Worst</div>
          <div style={{ fontSize: 14, fontWeight: 600, color: colors.text }}>{formatRatio(data.worstCase)}</div>
        </div>
      </div>
    </div>
  );

  const BASELINE_PCT = 20;
  const barScale = (value, max) => {
    if (value <= 0) return 2;
    if (value <= 1) return Math.max(2, BASELINE_PCT * value);
    if (max <= 1) return BASELINE_PCT;
    return Math.min(100, BASELINE_PCT + (Math.log(value) / Math.log(max)) * (100 - BASELINE_PCT));
  };
  const BarTrack = ({ children }) => (
    <div style={{ background: colors.chartTrack, height: 14, borderRadius: 7, position: 'relative' }}>
      <div style={{ position: 'absolute', left: `${BASELINE_PCT}%`, top: 0, bottom: 0, width: 1, borderLeft: `1px dashed ${colors.mutedText}`, opacity: 0.4, zIndex: 1 }} />
      {children}
    </div>
  );
  const maxCategorySpeedup = Math.max(
    ...categories.flatMap((c) =>
      runtimes.map((rt) => c.runtimes?.[rt]?.avgSpeedup || 0)
    ),
    1
  );

  const barGridCols = isMobile ? '90px 1fr 52px' : '180px 1fr 80px';

  return (
    <div style={{ color: colors.text }}>
      {/* Summary callout - generated comparison sentence */}
      {(() => {
        const describeRatio = (ratio) => {
          if (ratio >= 1.2) return 'much faster';
          if (ratio >= 1.05) return 'faster';
          if (ratio >= 1.01) return 'slightly faster';
          if (ratio >= 0.99) return 'on par';
          if (ratio >= 0.95) return 'slightly slower';
          if (ratio >= 0.8) return 'slower';
          return 'much slower';
        };
        const pctDiff = (ratio) => {
          const pct = Math.abs((ratio - 1) * 100).toFixed(0);
          return `${pct}%`;
        };
        const others = runtimes.filter((rt) => rt !== 'node');
        const maxSpread = Math.max(...others.map((rt) => Math.abs((summaries[rt]?.avgSpeedup || 1) - 1)));
        const spreadPct = (maxSpread * 100).toFixed(0);

        const nodeStyle = { color: RUNTIME_COLORS.node, fontWeight: 600 };
        const parts = others.map((rt) => {
          const ratio = summaries[rt]?.avgSpeedup || 1;
          const label = RUNTIME_LABELS[rt] || rt;
          const colorStyle = { color: RUNTIME_COLORS[rt], fontWeight: 600 };
          const desc = describeRatio(ratio);
          const isOnPar = ratio >= 0.99 && ratio < 1.01;
          return (
            <span key={rt}>
              <span style={colorStyle}>{label}</span>{' is '}
              {isOnPar ? (
                <span>on par with <span style={nodeStyle}>Node.js</span></span>
              ) : (
                <span>{desc} than <span style={nodeStyle}>Node.js</span> ({ratio >= 1 ? '+' : '-'}{pctDiff(ratio)})</span>
              )}
            </span>
          );
        });

        return (
          <div style={{ border: `1px solid ${colors.border}`, borderRadius: 10, padding: 14, background: colors.mutedCardBg, marginBottom: 20, fontSize: 14, lineHeight: 1.6 }}>
            <span style={{ fontWeight: 600 }}>All runtimes perform within {spreadPct}% of each other on average. </span>
            {parts.map((p, i) => (
              <span key={i}>{i > 0 ? ', ' : ''}{p}</span>
            ))}
            {'. '}The category breakdown below shows where they diverge.
          </div>
        );
      })()}

      {/* Category comparison chart */}
      <div style={{ border: `1px solid ${colors.border}`, borderRadius: 10, padding: 16, background: colors.cardBg, marginBottom: 20 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12, flexWrap: 'wrap', gap: 8 }}>
          <div style={{ fontSize: 16, fontWeight: 600, color: colors.text }}>Average Performance vs. Node.js by Category</div>
          <div style={{ display: 'flex', gap: 12 }}>
            {runtimes.map((rt) => (
              <div key={rt} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 12 }}>
                <span style={{ width: 10, height: 10, borderRadius: 3, background: RUNTIME_COLORS[rt], display: 'inline-block' }} />
                <span style={{ color: colors.mutedText }}>{RUNTIME_LABELS[rt] || rt}</span>
              </div>
            ))}
          </div>
        </div>
        <div style={{ display: 'grid', gap: 14 }}>
          {categories.map((category, catIdx) => {
            const catName = String(category.name);
            return (
              <div key={catName}>
                <div style={{ fontFamily: 'monospace', fontSize: 12, color: colors.text, marginBottom: 4, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  <a
                    href={`#cat-${catName.replace(/\s+/g, '-').toLowerCase()}`}
                    onClick={(e) => {
                      e.preventDefault();
                      fetchDetail();
                      setOpenCategories((prev) => ({ ...prev, [catName]: true }));
                      setTimeout(() => {
                        const el = document.getElementById(`cat-${catName.replace(/\s+/g, '-').toLowerCase()}`);
                        if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
                      }, 50);
                    }}
                    style={{ color: 'inherit', textDecoration: 'none', cursor: 'pointer' }}
                    onMouseEnter={(e) => (e.currentTarget.style.textDecoration = 'underline')}
                    onMouseLeave={(e) => (e.currentTarget.style.textDecoration = 'none')}
                  >
                    {catName}
                  </a>
                </div>
                <div style={{ display: 'grid', gap: 3 }}>
                  {runtimes.map((rt, rtIdx) => {
                    const ratio = Number(category.runtimes?.[rt]?.avgSpeedup) || 0;
                    const widthPct = barScale(ratio, maxCategorySpeedup);
                    return (
                      <div key={rt} style={{ display: 'grid', gridTemplateColumns: isMobile ? '40px 1fr 46px' : '50px 1fr 60px', gap: 8, alignItems: 'center' }}>
                        <div style={{ fontSize: 10, color: RUNTIME_COLORS[rt], fontWeight: 600 }}>{RUNTIME_LABELS[rt] || rt}</div>
                        <BarTrack >
                          <HoverBar tip={rt === 'node' ? `${RUNTIME_LABELS[rt]}: baseline` : ratioTip(ratio, rt)} width={`${widthPct}%`} color={RUNTIME_COLORS[rt]} rounded isDarkMode={isDarkMode} />
                        </BarTrack>
                        <div style={{ textAlign: 'right', fontWeight: 600, fontSize: 12, color: colors.text }}>{formatRatio(ratio)}</div>
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Collapsible meta info card */}
      <div style={{ border: `1px solid ${colors.border}`, borderRadius: 10, background: colors.mutedCardBg, marginBottom: 20, overflow: 'hidden' }}>
        <button
          onClick={() => setMetaOpen((prev) => !prev)}
          style={{ display: 'flex', alignItems: 'center', gap: 6, width: '100%', padding: '12px 14px', background: 'transparent', border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 'inherit', color: colors.text, textAlign: 'left' }}
        >
          <span style={{ fontSize: 10, display: 'inline-block', transform: metaOpen ? 'rotate(90deg)' : 'rotate(0deg)', transition: 'transform 0.15s' }}>&#x25B6;</span>
          <span>Benchmark Details</span>
        </button>
        <div style={{ maxHeight: metaOpen ? 500 : 0, overflow: 'hidden', transition: metaOpen ? 'max-height 0.3s ease-in' : 'max-height 0.3s ease-out' }}>
          <div style={{ padding: '0 14px 14px' }}>
            <div><strong>Generated:</strong> {meta.generatedAt || '-'}</div>
            {meta.machine ? <div><strong>Machine:</strong> <code>{meta.machine}</code></div> : null}
            <div><strong>numpy-ts:</strong> <code>{meta.numpyTsVersion || '-'}</code></div>
            {meta.runtimes && Object.entries(meta.runtimes).map(([rt, ver]) => (
              <div key={rt}>
                <strong style={{ color: RUNTIME_COLORS[rt] || colors.text }}>{RUNTIME_LABELS[rt] || rt}:</strong>{' '}
                <code>{ver}</code>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Detailed results */}
      <h2>Detailed Results</h2>
      <p style={{ color: colors.mutedText }}>
        Performance relative to Node.js. Over <code>1.00x</code> means faster than Node.js.
      </p>

      {categories.map((category) => {
        const catName = String(category.name);
        const isOpen = !!openCategories[catName];
        const benchmarks = isOpen
          ? (Array.isArray(category.benchmarks) ? category.benchmarks : (detailMap?.[category.name] || []))
          : null;
        const catRuntimes = category.runtimes || {};
        const subtitle = runtimes.map((rt) => {
          const s = catRuntimes[rt];
          return s ? `${RUNTIME_LABELS[rt] || rt}: ${formatRatio(s.avgSpeedup)}` : null;
        }).filter(Boolean).join(' | ');

        return (
          <div key={catName} id={`cat-${catName.replace(/\s+/g, '-').toLowerCase()}`} style={{ marginTop: 18, border: `1px solid ${colors.border}`, borderRadius: 10, background: colors.cardBg, overflow: 'hidden', scrollMarginTop: 80 }}>
            <button
              onClick={() => toggleCategory(catName)}
              style={{ display: 'flex', alignItems: 'center', gap: 6, width: '100%', padding: '12px 14px', background: 'transparent', border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 'inherit', color: colors.text, textAlign: 'left' }}
            >
              <span style={{ fontSize: 10, display: 'inline-block', transform: isOpen ? 'rotate(90deg)' : 'rotate(0deg)', transition: 'transform 0.15s' }}>&#x25B6;</span>
              <span>
                {catName}
                <span style={{ fontWeight: 400, fontSize: '0.85em', color: colors.mutedText, marginLeft: 8 }}>
                  ({catRuntimes[runtimes[0]]?.count ?? 0} benchmarks)
                </span>
              </span>
            </button>
            {isOpen && (
              <div style={{ padding: '0 14px 14px' }}>
                <div style={{ fontSize: 13, color: colors.mutedText, marginBottom: 8 }}>{subtitle}</div>
                <div style={{ overflowX: 'auto' }}>
                  {benchmarks.length === 0 && detailLoading ? (
                    <div style={{ padding: '20px 10px', color: colors.mutedText, fontSize: 13 }}>Loading benchmarks…</div>
                  ) : (
                  <table style={{ minWidth: '100%', borderCollapse: 'collapse', margin: 0 }}>
                    <thead>
                      <tr>
                        <th style={{ textAlign: 'left', padding: '8px 10px', background: colors.tableHeadBg, color: colors.tableHeadText, borderBottom: `1px solid ${colors.tableRowBorder}` }}>Benchmark</th>
                        {runtimes.map((rt) => (
                          <th key={rt} style={{ textAlign: 'left', padding: '8px 10px', background: colors.tableHeadBg, borderBottom: `1px solid ${colors.tableRowBorder}` }}>
                            <span style={{ color: RUNTIME_COLORS[rt], fontWeight: 700 }}>{RUNTIME_LABELS[rt] || rt}</span>
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {benchmarks.map((b, idx) => (
                        <tr key={`${String(b.name)}-${idx}`} style={{ background: idx % 2 === 0 ? 'transparent' : colors.tableRowAlt }}>
                          <td style={{ fontFamily: 'monospace', fontSize: 12, padding: '8px 10px', borderBottom: `1px solid ${colors.tableRowBorder}`, color: colors.text }}>
                            <BenchmarkName name={b.name} />
                          </td>
                          {runtimes.map((rt) => {
                            const entry = b[rt] || {};
                            const ratio = Number(entry.ratio);
                            return (
                              <td key={rt} style={{ padding: '8px 10px', borderBottom: `1px solid ${colors.tableRowBorder}` }}>
                                {Number.isFinite(ratio) ? (
                                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                                    <span style={{ background: ratioBg(ratio), color: ratioColor(ratio), padding: '2px 8px', borderRadius: 999, fontWeight: 700, fontSize: 12, whiteSpace: 'nowrap' }}>
                                      {formatRatio(ratio)}
                                    </span>
                                    {!isMobile && entry.ops != null && (
                                      <span style={{ fontSize: 11, color: colors.mutedText, whiteSpace: 'nowrap' }}>{formatOps(entry.ops)}</span>
                                    )}
                                  </span>
                                ) : (
                                  <span style={{ color: colors.mutedText }}>-</span>
                                )}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  )}
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};
