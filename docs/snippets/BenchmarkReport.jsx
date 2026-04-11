export const BenchmarkReport = ({ data, detailUrl }) => {
  const summary = data?.summary || {};
  const meta = data?.meta || {};
  const categories = Array.isArray(data?.categories) ? data.categories : [];
  const dtypeStats = Array.isArray(data?.dtypeStats) ? data.dtypeStats : [];

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
    // floats: blue family
    float64:    { bg: '#3b6cc9', text: '#ffffff' },
    float32:    { bg: '#4a80d8', text: '#ffffff' },
    float16:    { bg: '#6196e2', text: '#ffffff' },
    // complex: purple family
    complex128: { bg: '#8b5cc6', text: '#ffffff' },
    complex64:  { bg: '#a478d4', text: '#ffffff' },
    // signed ints: teal family
    int64:      { bg: '#2a8a82', text: '#ffffff' },
    int32:      { bg: '#3ba69d', text: '#ffffff' },
    int16:      { bg: '#55bfb7', text: '#1a4a46' },
    int8:       { bg: '#7ad4cd', text: '#1a4a46' },
    // unsigned ints: amber/orange family
    uint64:     { bg: '#c98a2e', text: '#ffffff' },
    uint32:     { bg: '#d9a044', text: '#3a2400' },
    uint16:     { bg: '#e4b85e', text: '#3a2400' },
    uint8:      { bg: '#eecb7c', text: '#3a2400' },
    // bool: rose
    bool:       { bg: '#cc4466', text: '#ffffff' },
  };

  const DTYPE_RE = /\s+(float64|float32|float16|complex128|complex64|int64|int32|int16|int8|uint64|uint32|uint16|uint8|bool)$/;

  const [isDarkMode, setIsDarkMode] = useState(() =>
    document.documentElement.classList.contains('dark')
  );
  const [openCategories, setOpenCategories] = useState({});
  const [isMetaOpen, setIsMetaOpen] = useState(false);
  const [isDtypeOpen, setIsDtypeOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(() => window.innerWidth < 640);
  const [isNarrow, setIsNarrow] = useState(() => window.innerWidth < 900);

  // Lazy-loaded detail benchmarks (fetched on first category expand)
  const [detailData, setDetailData] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const detailFetched = useRef(false);

  const fetchDetail = () => {
    if (detailFetched.current || !detailUrl) return;
    detailFetched.current = true;
    setDetailLoading(true);
    fetch(detailUrl)
      .then((r) => r.json())
      .then((d) => setDetailData(d))
      .catch(() => {})
      .finally(() => setDetailLoading(false));
  };

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

  const ratioTip = (ratio) => {
    if (!Number.isFinite(ratio)) return '';
    return ratio >= 1 ? `${ratio.toFixed(2)}x faster than NumPy` : `${ratio.toFixed(2)}x slower than NumPy`;
  };

  const HoverBar = ({ tip, width, color, rounded }) => {
    const [show, setShow] = useState(false);
    return (
      <div style={{ height: '100%', width, background: color, borderRadius: rounded ? 7 : 0, position: 'relative', zIndex: show ? 10 : 'auto', transition: 'filter 0.15s', cursor: 'default' }}
        onMouseEnter={(e) => { e.currentTarget.style.filter = 'brightness(1.3)'; setShow(true) }}
        onMouseLeave={(e) => { e.currentTarget.style.filter = ''; setShow(false) }}>
        {tip && (
          <div style={{ position: 'absolute', bottom: 'calc(100% + 6px)', left: '50%', transform: 'translateX(-50%)', padding: '4px 8px', borderRadius: 6, background: isDarkMode ? '#2a2a2a' : '#111', color: '#fff', fontSize: 11, whiteSpace: 'nowrap', zIndex: 100, pointerEvents: 'none', boxShadow: '0 2px 8px rgba(0,0,0,0.4)', opacity: show ? 1 : 0, transition: 'opacity 0.15s' }}>
            {tip}
          </div>
        )}
      </div>
    );
  };

  const RatioDisplay = ({ ratio, size }) => {
    if (!Number.isFinite(ratio)) return <span>-</span>;
    const numSize = size === 'lg' ? 28 : 20;
    const suffixSize = size === 'lg' ? 13 : 11;
    return (
      <span>
        <span style={{ fontSize: numSize, fontWeight: 700, color: colors.text }}>{ratio.toFixed(2)}x</span>
        <span style={{ fontSize: suffixSize, fontWeight: 400, color: colors.mutedText, marginLeft: 4 }}>{ratio >= 1 ? 'faster than NumPy' : 'slower than NumPy'}</span>
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

  const chartColor = (ratio) => {
    if (ratio >= 1) return colors.chartGood;
    if (ratio >= 0.75) return colors.chartOk;
    return colors.chartBad;
  };

  const SummaryCard = ({ label, value }) => (
    <div style={{ border: `1px solid ${colors.border}`, borderRadius: 10, padding: 14, background: colors.cardBg }}>
      <div style={{ fontSize: 12, color: colors.mutedText, marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 28, fontWeight: 700, color: colors.text }}>{value}</div>
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
  const maxCategorySpeedup = Math.max(...categories.map((c) => c.avgSpeedup || 0), 1);
  const maxDtypeSpeedup = Math.max(...dtypeStats.map((d) => d.avgSpeedup || 0), 1);
  const barGridCols = isMobile ? '90px 1fr 44px' : '140px 1fr 52px';

  return (
    <div style={{ color: colors.text }}>
      <div style={{ display: 'grid', gap: 12, gridTemplateColumns: isNarrow ? '1fr' : '2fr 1fr 1fr', marginBottom: 20 }}>
        <SummaryCard label="Average" value={<RatioDisplay ratio={summary.avgSpeedup} size="lg" />} />
        <SummaryCard label="Best case" value={formatRatio(summary.bestCase)} />
        <SummaryCard label="Worst case" value={formatRatio(summary.worstCase)} />
      </div>

      <div style={{ border: `1px solid ${colors.border}`, borderRadius: 10, padding: 16, background: colors.cardBg, marginBottom: 20 }}>
        <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 12, color: colors.text }}>Average Performance vs. NumPy by Category</div>
        <div style={{ display: 'grid', gap: 10 }}>
          {categories.map((category, catIdx) => {
            const ratio = Number(category.avgSpeedup) || 0;
            const widthPct = barScale(ratio, maxCategorySpeedup);
            return (
              <div key={String(category.name)} style={{ display: 'grid', gridTemplateColumns: barGridCols, gap: 10, alignItems: 'center' }}>
                <div style={{ fontFamily: 'monospace', fontSize: 12, color: colors.text, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  <a
                    href={`#cat-${String(category.name).replace(/\s+/g, '-').toLowerCase()}`}
                    onClick={(e) => {
                      e.preventDefault();
                      fetchDetail();
                      const catName = String(category.name);
                      setOpenCategories((prev) => ({ ...prev, [catName]: true }));
                      setTimeout(() => {
                        const el = document.getElementById(`cat-${catName.replace(/\s+/g, '-').toLowerCase()}`);
                        if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
                      }, 50);
                    }}
                    style={{ color: 'inherit', textDecoration: 'none', cursor: 'pointer' }}
                    onMouseEnter={(e) => e.currentTarget.style.textDecoration = 'underline'}
                    onMouseLeave={(e) => e.currentTarget.style.textDecoration = 'none'}
                  >{String(category.name)}</a>
                </div>
                <BarTrack >
                  <HoverBar tip={ratioTip(ratio)} width={`${widthPct}%`} color={chartColor(ratio)} rounded />
                </BarTrack>
                <div style={{ textAlign: 'right', fontWeight: 600, fontSize: 12, color: colors.text }}>{formatRatio(ratio)}</div>
              </div>
            );
          })}
        </div>
      </div>

      {dtypeStats.length > 0 && (
        <div style={{ border: `1px solid ${colors.border}`, borderRadius: 10, background: colors.cardBg, marginBottom: 20, overflow: 'hidden' }}>
          <button
            onClick={() => setIsDtypeOpen((v) => !v)}
            style={{ display: 'flex', alignItems: 'center', gap: 6, width: '100%', padding: 14, background: 'transparent', border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 14, color: colors.text, textAlign: 'left' }}
          >
            <span style={{ fontSize: 10, display: 'inline-block', transform: isDtypeOpen ? 'rotate(90deg)' : 'rotate(0deg)', transition: 'transform 0.15s' }}>▶</span>
            Average Performance by DType
          </button>
          <div style={{ maxHeight: isDtypeOpen ? 2000 : 0, overflow: 'hidden', transition: isDtypeOpen ? 'max-height 0.4s ease-in' : 'max-height 0.3s ease-out' }}>
            <div style={{ padding: '0 14px 14px', display: 'grid', gap: 10 }}>
              {dtypeStats.map(({ dtype, count, avgSpeedup }) => {
                const widthPct = barScale(avgSpeedup, maxDtypeSpeedup);
                const dc = DTYPE_COLORS[dtype] || DTYPE_COLORS.float64;
                return (
                  <div key={dtype} style={{ display: 'grid', gridTemplateColumns: barGridCols, gap: 10, alignItems: 'center' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, minWidth: 0 }}>
                      <span style={{ display: 'inline-block', fontSize: '0.72em', fontWeight: 600, padding: '2px 8px', borderRadius: 999, background: dc.bg, color: dc.text, whiteSpace: 'nowrap', flexShrink: 0 }}>
                        {dtype}
                      </span>
                      {!isMobile && <span style={{ fontSize: 11, color: colors.mutedText, whiteSpace: 'nowrap' }}>({count})</span>}
                    </div>
                    <BarTrack>
                      <div style={{ height: '100%', width: `${widthPct}%`, background: chartColor(avgSpeedup), borderRadius: 7, position: 'relative', zIndex: 0 }} />
                    </BarTrack>
                    <div style={{ textAlign: 'right', fontWeight: 600, fontSize: 12, color: colors.text }}>{formatRatio(avgSpeedup)}</div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      <div style={{ border: `1px solid ${colors.border}`, borderRadius: 10, background: colors.mutedCardBg, marginBottom: 20, overflow: 'hidden' }}>
        <button
          onClick={() => setIsMetaOpen((v) => !v)}
          style={{ display: 'flex', alignItems: 'center', gap: 6, width: '100%', padding: 14, background: 'transparent', border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 14, color: colors.text, textAlign: 'left' }}
        >
          <span style={{ fontSize: 10, display: 'inline-block', transform: isMetaOpen ? 'rotate(90deg)' : 'rotate(0deg)', transition: 'transform 0.15s' }}>▶</span>
          Benchmark Details
        </button>
        <div style={{ maxHeight: isMetaOpen ? 500 : 0, overflow: 'hidden', transition: isMetaOpen ? 'max-height 0.3s ease-in' : 'max-height 0.3s ease-out' }}>
          <div style={{ padding: '0 14px 14px' }}>
            <div><strong>Generated:</strong> {meta.generatedAt || '-'}</div>
            <div><strong>Source:</strong> <code>{meta.sourceJson || '-'}</code></div>
            {meta.runtimes && Object.entries(meta.runtimes).map(([rt, ver]) => (
              <div key={rt}><strong>{rt.charAt(0).toUpperCase() + rt.slice(1)}:</strong> <code>{ver}</code></div>
            ))}
            {meta.pythonVersion ? <div><strong>Python:</strong> <code>{meta.pythonVersion}</code></div> : null}
            {meta.numpyVersion ? <div><strong>NumPy:</strong> <code>{meta.numpyVersion}</code></div> : null}
            <div><strong>numpy-ts:</strong> <code>{meta.numpyTsVersion || '-'}</code></div>
            {meta.machine ? <div><strong>Machine:</strong> <code>{meta.machine}</code></div> : null}
          </div>
        </div>
      </div>

      <h2>Detailed Results</h2>
      <p style={{ color: colors.mutedText }}>
        Higher is better. A speedup over <code>1.00x</code> means <code>numpy-ts</code> was faster than NumPy for that benchmark.
      </p>

      {categories.map((category) => {
        const detailCat = detailData?.categories?.find((c) => c.name === category.name);
        const benchmarks = Array.isArray(category.benchmarks) ? category.benchmarks : (detailCat?.benchmarks || []);
        const isOpen = !!openCategories[String(category.name)];
        return (
          <div key={String(category.name)} id={`cat-${String(category.name).replace(/\s+/g, '-').toLowerCase()}`} style={{ marginTop: 18, border: `1px solid ${colors.border}`, borderRadius: 10, background: colors.cardBg, overflow: 'hidden', scrollMarginTop: 80 }}>
            <button
              onClick={() => toggleCategory(String(category.name))}
              style={{ display: 'flex', alignItems: 'center', gap: 6, width: '100%', padding: '12px 14px', background: 'transparent', border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 'inherit', color: colors.text, textAlign: 'left' }}
            >
              <span style={{ fontSize: 10, display: 'inline-block', transform: isOpen ? 'rotate(90deg)' : 'rotate(0deg)', transition: 'transform 0.15s' }}>▶</span>
              {String(category.name)} ({category.count} benchmarks, speedup {(Number(category.avgSpeedup) || 0).toFixed(2)}x)
            </button>
            <div style={{ maxHeight: isOpen ? 10000 : 0, overflow: 'hidden', transition: isOpen ? 'max-height 0.5s ease-in' : 'max-height 0.3s ease-out' }}>
              <div style={{ padding: '0 14px 14px' }}>
                <div style={{ fontSize: 13, color: colors.mutedText, marginBottom: 8 }}>
                  Slower than NumPy: {category.slowerCount} | Faster than NumPy: {category.fasterCount}
                </div>
                <div style={{ overflowX: 'auto' }}>
                  {benchmarks.length === 0 && detailLoading ? (
                    <div style={{ padding: '20px 10px', color: colors.mutedText, fontSize: 13 }}>Loading benchmarks…</div>
                  ) : (
                  <table style={{ minWidth: '100%', borderCollapse: 'collapse', margin: 0 }}>
                    <thead>
                      <tr>
                        <th style={{ textAlign: 'left', padding: '8px 10px', background: colors.tableHeadBg, color: colors.tableHeadText, borderBottom: `1px solid ${colors.tableRowBorder}` }}>Benchmark</th>
                        <th style={{ textAlign: 'left', padding: '8px 10px', background: colors.tableHeadBg, color: colors.tableHeadText, borderBottom: `1px solid ${colors.tableRowBorder}` }}>Speedup</th>
                        <th style={{ textAlign: 'left', padding: '8px 10px', background: colors.tableHeadBg, color: colors.tableHeadText, borderBottom: `1px solid ${colors.tableRowBorder}` }}>NumPy ops/s</th>
                        <th style={{ textAlign: 'left', padding: '8px 10px', background: colors.tableHeadBg, color: colors.tableHeadText, borderBottom: `1px solid ${colors.tableRowBorder}` }}>numpy-ts ops/s</th>
                      </tr>
                    </thead>
                    <tbody>
                      {benchmarks.map((b, idx) => (
                        <tr key={`${String(b.name)}-${idx}`} style={{ background: idx % 2 === 0 ? 'transparent' : colors.tableRowAlt }}>
                          <td style={{ fontFamily: 'monospace', fontSize: 12, padding: '8px 10px', borderBottom: `1px solid ${colors.tableRowBorder}`, color: colors.text }}><BenchmarkName name={b.name} /></td>
                          <td style={{ padding: '8px 10px', borderBottom: `1px solid ${colors.tableRowBorder}` }}>
                            <span style={{ background: ratioBg(b.ratio), color: ratioColor(b.ratio), padding: '2px 8px', borderRadius: 999, fontWeight: 700, fontSize: 12 }}>
                              {formatRatio(b.ratio)}
                            </span>
                          </td>
                          <td style={{ padding: '8px 10px', borderBottom: `1px solid ${colors.tableRowBorder}`, color: colors.text }}>{formatOps(b.numpyOps)}</td>
                          <td style={{ padding: '8px 10px', borderBottom: `1px solid ${colors.tableRowBorder}`, color: colors.text }}>{formatOps(b.numpyTsOps)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  )}
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
};
