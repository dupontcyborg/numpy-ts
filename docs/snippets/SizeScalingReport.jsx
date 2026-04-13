export const SizeScalingReport = ({ data, detailUrl }) => {
  const meta = data?.meta || {};
  const sizes = Array.isArray(data?.sizes) ? data.sizes : [];

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

  const SIZE_COLORS = {
    0: { bar: '#3b82f6', label: '#3b82f6' },  // Small = blue
    1: { bar: '#8b5cf6', label: '#8b5cf6' },  // Medium = violet
    2: { bar: '#a855f7', label: '#a855f7' },  // Large = purple
  };

  const [isDarkMode, setIsDarkMode] = useState(() =>
    document.documentElement.classList.contains('dark')
  );
  const [openCategories, setOpenCategories] = useState({});
  const [isMobile, setIsMobile] = useState(() => window.innerWidth < 640);
  const [isNarrow, setIsNarrow] = useState(() => window.innerWidth < 900);
  const [tooltip, setTooltip] = useState(null);
  const [metaOpen, setMetaOpen] = useState(false);

  // Lazy-loaded detail benchmarks
  // Detail benchmarks: prefetched in the background after first paint so opening
  // a drawer is instant. The worker fetches, parses, AND indexes the JSON into a
  // { categoryName: benchmarks[] } map so the main thread does no prep work.
  // Note: this report's detail JSON uses `benchmarks` as the array of categories.
  const [detailMap, setDetailMap] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const detailFetched = useRef(false);

  const fetchDetail = () => {
    if (detailFetched.current || !detailUrl) return;
    detailFetched.current = true;
    setDetailLoading(true);
    const absUrl = new URL(detailUrl, window.location.href).href;
    const buildMap = (d) => {
      const cats = (d && d.benchmarks) || [];
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
      worker.postMessage({ url: absUrl, key: 'benchmarks' });
    } catch { fallback(); }
  };

  // Prefetch detail JSON in the background so opening a drawer is instant.
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
  const HoverBar = useMemo(() => function HoverBar({ tip, width, color, rounded, opacity, isDarkMode }) {
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
        <div style={{ height: '100%', width: '100%', background: color, borderRadius: rounded ? 7 : 0, opacity: opacity ?? 1, filter: show ? 'brightness(1.3)' : 'none', transition: 'filter 0.15s' }} />
        {tip && show && pos && (
          <div style={{ position: 'fixed', top: pos.top - 6, left: pos.left, transform: 'translate(-50%, -100%)', padding: '4px 8px', borderRadius: 6, background: isDarkMode ? '#2a2a2a' : '#111', color: '#fff', fontSize: 11, whiteSpace: 'nowrap', zIndex: 2147483647, pointerEvents: 'none', boxShadow: '0 2px 8px rgba(0,0,0,0.4)' }}>
            {tip}
          </div>
        )}
      </div>
    );
  }, []);

  const formatRatio = (ratio) => {
    if (!Number.isFinite(ratio)) return '-';
    return `${ratio.toFixed(2)}x`;
  };

  const ratioTip = (ratio) => {
    if (!Number.isFinite(ratio)) return '';
    return ratio >= 1 ? `${ratio.toFixed(2)}x faster than NumPy` : `${ratio.toFixed(2)}x slower than NumPy`;
  };

  const RatioDisplay = ({ ratio }) => {
    if (!Number.isFinite(ratio)) return <span>-</span>;
    return (
      <span>
        <span style={{ fontSize: 20, fontWeight: 700, color: colors.text }}>{ratio.toFixed(2)}x</span>
        <span style={{ fontSize: 11, fontWeight: 400, color: colors.mutedText, marginLeft: 4 }}>{ratio >= 1 ? 'faster than NumPy' : 'slower than NumPy'}</span>
      </span>
    );
  };

  const formatOps = (ops) => {
    if (!Number.isFinite(ops) || ops <= 0) return '-';
    if (ops >= 1_000_000_000) return `${(ops / 1_000_000_000).toFixed(2)}B/s`;
    if (ops >= 1_000_000) return `${(ops / 1_000_000).toFixed(2)}M/s`;
    if (ops >= 1_000) return `${(ops / 1_000).toFixed(2)}K/s`;
    return `${ops.toFixed(1)}/s`;
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

  const chartColor = (ratio) => {
    if (ratio >= 1) return colors.chartGood;
    if (ratio >= 0.75) return colors.chartOk;
    return colors.chartBad;
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

  const SummaryCard = ({ label, value, accent }) => (
    <div style={{ border: `1px solid ${colors.border}`, borderRadius: 10, padding: 14, background: colors.cardBg }}>
      <div style={{ fontSize: 12, color: colors.mutedText, marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 24, fontWeight: 700, color: accent || colors.text }}>{value}</div>
    </div>
  );

  // Collect all unique category names across sizes
  const allCategoryNames = [];
  const seenCats = new Set();
  sizes.forEach((size) => {
    (size.categories || []).forEach((cat) => {
      if (!seenCats.has(cat.name)) {
        seenCats.add(cat.name);
        allCategoryNames.push(cat.name);
      }
    });
  });

  // Build lookup: categoryName -> sizeIndex -> category data
  const catBySizeMap = {};
  allCategoryNames.forEach((name) => { catBySizeMap[name] = {}; });
  sizes.forEach((size, sIdx) => {
    (size.categories || []).forEach((cat) => {
      if (catBySizeMap[cat.name]) catBySizeMap[cat.name][sIdx] = cat;
    });
  });

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
  // Max avgSpeedup across all categories and sizes for bar scaling
  let maxSpeedup = 1;
  sizes.forEach((size) => {
    (size.categories || []).forEach((cat) => {
      if (cat.avgSpeedup > maxSpeedup) maxSpeedup = cat.avgSpeedup;
    });
  });

  const barGridCols = isMobile ? '80px 1fr 52px' : '160px 1fr 80px';

  // Cross-size benchmark data (normalized names, keyed by size)
  // Inline data may have benchmarks stripped; use detailData if available
  const crossCategories = Array.isArray(data?.benchmarks) ? data.benchmarks
    : Array.isArray(detailData?.benchmarks) ? detailData.benchmarks : [];
  const SIZE_KEYS = ['small', 'medium', 'large'];

  return (
    <div style={{ color: colors.text }}>
      {/* Summary callout */}
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
        const allFaster = sizes.every((s) => (s.summary?.avgSpeedup || 0) >= 1);
        const allSlower = sizes.every((s) => (s.summary?.avgSpeedup || 0) < 1);
        const headline = allFaster
          ? 'numpy-ts is faster than NumPy across all tested array sizes.'
          : allSlower
          ? 'NumPy is faster than numpy-ts across all tested array sizes.'
          : 'Performance varies by array size.';

        return (
          <div style={{ border: `1px solid ${colors.border}`, borderRadius: 10, padding: 14, background: colors.mutedCardBg, marginBottom: 20, fontSize: 14, lineHeight: 1.6 }}>
            <span style={{ fontWeight: 600 }}>{headline} </span>
            {sizes.map((size, sIdx) => {
              const ratio = size.summary?.avgSpeedup || 0;
              const desc = describeRatio(ratio);
              const pct = Math.abs((ratio - 1) * 100).toFixed(0);
              const colorStyle = { color: SIZE_COLORS[sIdx]?.label || colors.text, fontWeight: 600 };
              return (
                <span key={size.label}>
                  {sIdx > 0 ? ', ' : ''}
                  <span style={colorStyle}>{size.label}</span>
                  {': '}
                  {ratio >= 0.99 && ratio < 1.01
                    ? 'on par'
                    : `${desc} (${ratio >= 1 ? '+' : '-'}${pct}%)`
                  }
                </span>
              );
            })}
            .
          </div>
        );
      })()}

      {/* Category comparison chart */}
      <div style={{ border: `1px solid ${colors.border}`, borderRadius: 10, padding: 16, background: colors.cardBg, marginBottom: 20 }}>
        <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 8, color: colors.text }}>Average Performance vs. NumPy by Category & Size</div>

        {/* Legend */}
        <div style={{ display: 'flex', gap: 16, marginBottom: 14, flexWrap: 'wrap' }}>
          {sizes.map((size, sIdx) => (
            <div key={size.label} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              <div style={{ width: 12, height: 12, borderRadius: 3, background: SIZE_COLORS[sIdx]?.bar || colors.chartGood }} />
              <span style={{ fontSize: 12, color: colors.mutedText }}>{size.label}</span>
            </div>
          ))}
        </div>

        <div style={{ display: 'grid', gap: 16 }}>
          {allCategoryNames.map((catName, catIdx) => {
            const catData = catBySizeMap[catName];
            return (
              <div key={catName}>
                <div style={{ fontFamily: 'monospace', fontSize: 12, color: colors.text, marginBottom: 4, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  <a
                    href={`#scaling-cat-${catName.replace(/\s+/g, '-').toLowerCase()}`}
                    onClick={(e) => {
                      e.preventDefault();
                      fetchDetail();
                      setOpenCategories((prev) => ({ ...prev, [catName]: true }));
                      setTimeout(() => {
                        const el = document.getElementById(`scaling-cat-${catName.replace(/\s+/g, '-').toLowerCase()}`);
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
                  {sizes.map((size, sIdx) => {
                    const cat = catData[sIdx];
                    const ratio = cat ? Number(cat.avgSpeedup) || 0 : 0;
                    const widthPct = barScale(ratio, maxSpeedup);
                    const barColor = SIZE_COLORS[sIdx]?.bar || colors.chartGood;
                    return (
                      <div
                        key={size.label}
                        style={{ display: 'grid', gridTemplateColumns: isMobile ? '1fr 44px' : '1fr 60px', gap: 8, alignItems: 'center' }}
                      >
                        <BarTrack >
                          <HoverBar tip={`${size.label}: ${ratioTip(ratio)}`} width={`${widthPct}%`} color={barColor} rounded opacity={ratio === 0 ? 0.2 : 1} isDarkMode={isDarkMode} />
                        </BarTrack>
                        <div style={{ textAlign: 'right', fontWeight: 600, fontSize: 12, color: ratio === 0 ? colors.mutedText : ratioColor(ratio) }}>
                          {ratio === 0 ? '-' : formatRatio(ratio)}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Meta info card (collapsible) */}
      <div style={{ border: `1px solid ${colors.border}`, borderRadius: 10, background: colors.mutedCardBg, marginBottom: 20, overflow: 'hidden' }}>
        <button
          onClick={() => setMetaOpen((prev) => !prev)}
          style={{ display: 'flex', alignItems: 'center', gap: 6, width: '100%', padding: '12px 14px', background: 'transparent', border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 'inherit', color: colors.text, textAlign: 'left' }}
        >
          <span style={{ fontSize: 10, display: 'inline-block', transform: metaOpen ? 'rotate(90deg)' : 'rotate(0deg)', transition: 'transform 0.15s' }}>&#9654;</span>
          Benchmark Details
        </button>
        <div style={{ maxHeight: metaOpen ? 500 : 0, overflow: 'hidden', transition: metaOpen ? 'max-height 0.3s ease-in' : 'max-height 0.3s ease-out' }}>
          <div style={{ padding: '0 14px 14px' }}>
            <div><strong>Generated:</strong> {meta.generatedAt || '-'}</div>
            {meta.machine ? <div><strong>Machine:</strong> <code>{meta.machine}</code></div> : null}
            {meta.pythonVersion ? <div><strong>Python:</strong> <code>{meta.pythonVersion}</code></div> : null}
            {meta.numpyVersion ? <div><strong>NumPy:</strong> <code>{meta.numpyVersion}</code></div> : null}
            <div><strong>numpy-ts:</strong> <code>{meta.numpyTsVersion || '-'}</code></div>
          </div>
        </div>
      </div>

      {/* Detailed results */}
      <h2>Detailed Results by Category</h2>
      <p style={{ color: colors.mutedText }}>
        Benchmarks matched across array sizes. Higher is better. Over <code>1.00x</code> means numpy-ts was faster.
      </p>

      {crossCategories.map((crossCat) => {
        const catName = crossCat.name;
        const isOpen = !!openCategories[catName];
        const benchmarks = isOpen
          ? (Array.isArray(crossCat.benchmarks) ? crossCat.benchmarks : (detailMap?.[catName] || []))
          : null;
        const catSummaries = sizes.map((size, sIdx) => {
          const cat = (size.categories || []).find((c) => c.name === catName);
          return cat ? `${size.label}: ${formatRatio(cat.avgSpeedup)}` : null;
        }).filter(Boolean).join(' | ');

        return (
          <div key={catName} id={`scaling-cat-${catName.replace(/\s+/g, '-').toLowerCase()}`} style={{ marginTop: 18, border: `1px solid ${colors.border}`, borderRadius: 10, background: colors.cardBg, overflow: 'hidden', scrollMarginTop: 80 }}>
            <button
              onClick={() => toggleCategory(catName)}
              style={{ display: 'flex', alignItems: 'center', gap: 6, width: '100%', padding: '12px 14px', background: 'transparent', border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 'inherit', color: colors.text, textAlign: 'left' }}
            >
              <span style={{ fontSize: 10, display: 'inline-block', transform: isOpen ? 'rotate(90deg)' : 'rotate(0deg)', transition: 'transform 0.15s' }}>&#9654;</span>
              {catName} ({catSummaries})
            </button>
            {isOpen && (
              <div style={{ padding: '0 14px 14px' }}>
                <div style={{ overflowX: 'auto' }}>
                  {benchmarks.length === 0 && detailLoading ? (
                    <div style={{ padding: '20px 10px', color: colors.mutedText, fontSize: 13 }}>Loading benchmarks…</div>
                  ) : (
                  <table style={{ minWidth: '100%', borderCollapse: 'collapse', margin: 0 }}>
                    <thead>
                      <tr>
                        <th style={{ textAlign: 'left', padding: '8px 10px', background: colors.tableHeadBg, color: colors.tableHeadText, borderBottom: `1px solid ${colors.tableRowBorder}` }}>Benchmark</th>
                        {sizes.map((size, sIdx) => (
                          <th key={size.label} style={{ textAlign: 'center', padding: '8px 10px', background: colors.tableHeadBg, color: SIZE_COLORS[sIdx]?.label || colors.tableHeadText, borderBottom: `1px solid ${colors.tableRowBorder}`, whiteSpace: 'nowrap' }}>
                            {size.label}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {benchmarks.map((b, idx) => (
                        <tr key={b.name} style={{ background: idx % 2 === 0 ? 'transparent' : colors.tableRowAlt }}>
                          <td style={{ fontFamily: 'monospace', fontSize: 12, padding: '8px 10px', borderBottom: `1px solid ${colors.tableRowBorder}`, color: colors.text }}>
                            <BenchmarkName name={b.name} />
                          </td>
                          {SIZE_KEYS.map((key, sIdx) => {
                            const ratio = b[key];
                            if (ratio == null) {
                              return (
                                <td key={key} style={{ textAlign: 'center', padding: '8px 10px', borderBottom: `1px solid ${colors.tableRowBorder}`, color: colors.mutedText }}>-</td>
                              );
                            }
                            return (
                              <td key={key} style={{ textAlign: 'center', padding: '8px 10px', borderBottom: `1px solid ${colors.tableRowBorder}` }}>
                                <span style={{ background: ratioBg(ratio), color: ratioColor(ratio), padding: '2px 8px', borderRadius: 999, fontWeight: 700, fontSize: 12, whiteSpace: 'nowrap' }}>
                                  {formatRatio(ratio)}
                                </span>
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
