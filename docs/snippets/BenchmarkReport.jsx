export const BenchmarkReport = ({ data }) => {
  const summary = data?.summary || {};
  const meta = data?.meta || {};
  const categories = Array.isArray(data?.categories) ? data.categories : [];

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

  const [isDarkMode, setIsDarkMode] = useState(() =>
    document.documentElement.classList.contains('dark')
  );
  const [openCategories, setOpenCategories] = useState({});
  const toggleCategory = (name) => setOpenCategories((prev) => ({ ...prev, [name]: !prev[name] }));

  useEffect(() => {
    const observer = new MutationObserver(() => {
      setIsDarkMode(document.documentElement.classList.contains('dark'));
    });

    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class'],
    });

    return () => observer.disconnect();
  }, []);

  const colors = THEME_COLORS[isDarkMode ? 'dark' : 'light'];

  const DTYPE_RE = /\s+(float64|float32|complex128|complex64|int64|int32|int16|int8|uint64|uint32|uint16|uint8|bool)$/;
  const DTYPE_COLORS = {
    float64:    { bg: '#dbeafe', text: '#1e40af' },
    float32:    { bg: '#e0f2fe', text: '#0369a1' },
    complex128: { bg: '#ede9fe', text: '#6d28d9' },
    complex64:  { bg: '#f3e8ff', text: '#7e22ce' },
    int64:      { bg: '#d1fae5', text: '#065f46' },
    int32:      { bg: '#dcfce7', text: '#166534' },
    int16:      { bg: '#ecfdf5', text: '#15803d' },
    int8:       { bg: '#f0fdf4', text: '#16a34a' },
    uint64:     { bg: '#fff7ed', text: '#9a3412' },
    uint32:     { bg: '#ffedd5', text: '#c2410c' },
    uint16:     { bg: '#fef3c7', text: '#b45309' },
    uint8:      { bg: '#fef9c3', text: '#ca8a04' },
    bool:       { bg: '#fce7f3', text: '#9d174d' },
  };

  const BenchmarkName = ({ name }) => {
    const s = String(name);
    const m = s.match(DTYPE_RE);
    const dtype = m ? m[1] : 'float64';
    const base = m ? s.slice(0, -m[0].length) : s;
    const dc = DTYPE_COLORS[dtype] || DTYPE_COLORS.float64;
    const opacity = m ? 1 : 0.55;
    return (
      <>
        {base}{' '}
        <span style={{ display: 'inline-block', fontSize: '0.72em', fontWeight: 600, padding: '1px 6px', borderRadius: 3, background: dc.bg, color: dc.text, opacity, verticalAlign: 'middle', whiteSpace: 'nowrap' }}>
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

  const ratioColor = (ratio) => {
    if (ratio < 2.5) return colors.ratioGoodText;
    if (ratio <= 5) return colors.ratioOkText;
    return colors.ratioBadText;
  };

  const ratioBg = (ratio) => {
    if (ratio < 2.5) return colors.ratioGoodBg;
    if (ratio <= 5) return colors.ratioOkBg;
    return colors.ratioBadBg;
  };

  const chartColor = (ratio) => {
    if (ratio < 2.5) return colors.chartGood;
    if (ratio <= 5) return colors.chartOk;
    return colors.chartBad;
  };

  const SummaryCard = ({ label, value }) => (
    <div style={{ border: `1px solid ${colors.border}`, borderRadius: 10, padding: 14, background: colors.cardBg }}>
      <div style={{ fontSize: 12, color: colors.mutedText, marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 24, fontWeight: 700, color: colors.text }}>{value}</div>
    </div>
  );

  const maxCategorySlowdown = Math.max(...categories.map((c) => c.avgSlowdown || 0), 1);

  return (
    <div style={{ color: colors.text }}>
      <div style={{ border: `1px solid ${colors.border}`, borderRadius: 10, padding: 14, background: colors.mutedCardBg, marginBottom: 20 }}>
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

      <div style={{ display: 'grid', gap: 12, gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', marginBottom: 20 }}>
        <SummaryCard label="Average slowdown" value={formatRatio(summary.avgSlowdown)} />
        <SummaryCard label="Median slowdown" value={formatRatio(summary.medianSlowdown)} />
        <SummaryCard label="Best case" value={formatRatio(summary.bestCase)} />
        <SummaryCard label="Worst case" value={formatRatio(summary.worstCase)} />
      </div>

      <div style={{ border: `1px solid ${colors.border}`, borderRadius: 10, padding: 16, background: colors.cardBg }}>
        <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 12, color: colors.text }}>Average Slowdown by Category</div>
        <div style={{ display: 'grid', gap: 10 }}>
          {categories.map((category) => {
            const ratio = Number(category.avgSlowdown) || 0;
            const widthPct = Math.max(2, (ratio / maxCategorySlowdown) * 100);
            return (
              <div key={String(category.name)} style={{ display: 'grid', gridTemplateColumns: '180px 1fr 80px', gap: 10, alignItems: 'center' }}>
                <div style={{ fontFamily: 'monospace', fontSize: 12, color: colors.text, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  <a
                    href={`#cat-${String(category.name).replace(/\s+/g, '-').toLowerCase()}`}
                    onClick={(e) => {
                      e.preventDefault();
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
                <div style={{ background: colors.chartTrack, height: 14, borderRadius: 7, overflow: 'hidden' }}>
                  <div
                    style={{
                      height: '100%',
                      width: `${widthPct}%`,
                      background: chartColor(ratio),
                    }}
                  />
                </div>
                <div style={{ textAlign: 'right', fontWeight: 600, fontSize: 12, color: colors.text }}>{formatRatio(ratio)}</div>
              </div>
            );
          })}
        </div>
      </div>

      <h2>Detailed Results</h2>
      <p style={{ color: colors.mutedText }}>
        Lower is better. A ratio under <code>1.00x</code> means <code>numpy-ts</code> was faster than NumPy for that benchmark.
      </p>

      {categories.map((category) => {
        const benchmarks = Array.isArray(category.benchmarks) ? category.benchmarks : [];
        const isOpen = !!openCategories[String(category.name)];
        return (
          <div key={String(category.name)} id={`cat-${String(category.name).replace(/\s+/g, '-').toLowerCase()}`} style={{ marginTop: 18, border: `1px solid ${colors.border}`, borderRadius: 10, background: colors.cardBg, overflow: 'hidden', scrollMarginTop: 80 }}>
            <button
              onClick={() => toggleCategory(String(category.name))}
              style={{ display: 'flex', alignItems: 'center', gap: 6, width: '100%', padding: '12px 14px', background: 'transparent', border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 'inherit', color: colors.text, textAlign: 'left' }}
            >
              <span style={{ fontSize: 10, display: 'inline-block', transform: isOpen ? 'rotate(90deg)' : 'rotate(0deg)', transition: 'transform 0.15s' }}>▶</span>
              {String(category.name)} ({category.count} benchmarks, avg {(Number(category.avgSlowdown) || 0).toFixed(2)}x)
            </button>
            {isOpen && (
              <div style={{ padding: '0 14px 14px' }}>
                <div style={{ fontSize: 13, color: colors.mutedText, marginBottom: 8 }}>
                  Slower than NumPy: {category.slowerCount} | Faster than NumPy: {category.fasterCount}
                </div>
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ minWidth: '100%', borderCollapse: 'collapse', margin: 0 }}>
                    <thead>
                      <tr>
                        <th style={{ textAlign: 'left', padding: '8px 10px', background: colors.tableHeadBg, color: colors.tableHeadText, borderBottom: `1px solid ${colors.tableRowBorder}` }}>Benchmark</th>
                        <th style={{ textAlign: 'left', padding: '8px 10px', background: colors.tableHeadBg, color: colors.tableHeadText, borderBottom: `1px solid ${colors.tableRowBorder}` }}>Slowdown</th>
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
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};
