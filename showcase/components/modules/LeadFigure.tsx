"use client";

// Data: scripts/export_lead_figure.py -> content/generated/lead-figure.json; numbers in text come from the registry.
import { useEffect, useRef, useState } from "react";
import lead from "../../content/generated/lead-figure.json";
import { fmt } from "../../lib/results";

import { ACCENT, INK, INK_3, RULE, STAGE_NAMES } from "../../design/palette";

// Hypnogram rows, top to bottom, in the conventional order: Wake, REM, N1, N2, N3.
const ROW_OF_STAGE = [0, 2, 3, 4, 1];
const ROW_LABELS = ["Wake", "REM", "N1", "N2", "N3"];

const night = lead.night;
const n = night.expert.length;

function clock(epoch: number) {
  const minutes = Math.floor(epoch / 2);
  return `${Math.floor(minutes / 60)} h ${String(minutes % 60).padStart(2, "0")} min`;
}

// Draw at the container's real width so 11-px labels stay 11 px on a phone instead of scaling down.
function useWidth(fallback: number) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.max(280, Math.round(entry.contentRect.width))));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

function Hypnogram() {
  const [hover, setHover] = useState<number | null>(null);
  const [ref, W] = useWidth(640);
  const left = 62, right = 14, trackH = 76, gap = 34, stripH = 10, first = 24;
  const plotW = W - left - right;
  const x = (i: number) => left + (i / n) * plotW;
  const tracks = [
    { label: "Expert", stages: night.expert, color: INK, top: first },
    { label: "Model C", stages: night.model, color: ACCENT, top: first + trackH + gap }
  ];
  const stripTop = first + 2 * trackH + gap + 18;
  const H = stripTop + stripH + 30;
  const y = (top: number, stage: number) => top + (ROW_OF_STAGE[stage] / 4) * trackH;
  const path = (stages: number[], top: number) => {
    let d = `M${x(0).toFixed(1)} ${y(top, stages[0]).toFixed(1)}`;
    for (let i = 0; i < stages.length; i++) {
      d += `H${x(i + 1).toFixed(1)}`;
      if (i + 1 < stages.length && stages[i + 1] !== stages[i]) d += `V${y(top, stages[i + 1]).toFixed(1)}`;
    }
    return d;
  };
  const runs: Array<[number, number]> = [];
  for (let i = 0; i < n; i++) {
    if (night.expert[i] === night.model[i]) continue;
    const last = runs[runs.length - 1];
    if (last && last[1] === i) last[1] = i + 1; else runs.push([i, i + 1]);
  }
  const longest = runs.reduce<[number, number] | null>((best, run) =>
    best && best[1] - best[0] >= run[1] - run[0] ? best : run, null);
  const hourTicks = Array.from({ length: Math.floor(n / 120) + 1 }, (_, h) => h * 120);
  const onMove = (event: React.PointerEvent<SVGSVGElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const px = ((event.clientX - box.left) / box.width) * W;
    const i = Math.floor(((px - left) / plotW) * n);
    setHover(i >= 0 && i < n ? i : null);
  };
  const onKeyDown = (event: React.KeyboardEvent<SVGSVGElement>) => {
    if (event.key === "ArrowRight") {
      event.preventDefault();
      setHover((prev) => (prev === null ? 0 : Math.min(prev + 1, n - 1)));
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      setHover((prev) => (prev === null ? n - 1 : Math.max(prev - 1, 0)));
    } else if (event.key === "Home") {
      event.preventDefault();
      setHover(0);
    } else if (event.key === "End") {
      event.preventDefault();
      setHover(n - 1);
    } else if (event.key === "Escape") {
      setHover(null);
    }
  };
  return (
    <div className="lead-plot" ref={ref}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="lead-svg-interactive"
        role="img"
        tabIndex={0}
        onPointerMove={onMove}
        onPointerLeave={() => setHover(null)}
        onKeyDown={onKeyDown}
        aria-label={`Sleep stages assigned by the expert and Model C over ${(n / 120).toFixed(1)} scored hours, including wake. They differ on ${night.disagreements} of ${night.epochs} epochs. Hover or use arrow keys to inspect individual epochs.`}
      >
        {longest && <rect x={x(longest[0])} y={first - 4} width={x(longest[1]) - x(longest[0])} height={stripTop + stripH - first + 4} fill={ACCENT} fillOpacity={0.12} />}
        {tracks.map((t) => (
          <g key={t.label}>
            <text x={left} y={t.top - 12} className="lead-track-label">{t.label}</text>
            {ROW_LABELS.map((label, row) => (
              <g key={label}>
                <line x1={left} x2={W - right} y1={t.top + (row / 4) * trackH} y2={t.top + (row / 4) * trackH} stroke={RULE} strokeWidth={1} />
                <text x={left - 6} y={t.top + (row / 4) * trackH + 3.5} textAnchor="end" className="lead-axis">{label}</text>
              </g>
            ))}
            <path d={path(t.stages, t.top)} fill="none" stroke={t.color} strokeWidth={1.4} strokeLinejoin="round" />
          </g>
        ))}
        <text x={left - 6} y={stripTop + 8} textAnchor="end" className="lead-axis">Mismatch</text>
        <rect x={left} y={stripTop} width={plotW} height={stripH} fill="#f2f0ea" />
        {runs.map(([a, b]) => <rect key={a} x={x(a)} y={stripTop} width={Math.max(x(b) - x(a), 0.8)} height={stripH} fill={INK_3} />)}
        {hourTicks.map((e) => (
          <g key={e}>
            <line x1={x(e)} x2={x(e)} y1={stripTop + stripH} y2={stripTop + stripH + 4} stroke={INK_3} />
            <text x={x(e)} y={stripTop + stripH + 16} textAnchor="middle" className="lead-axis">{e / 120} h</text>
          </g>
        ))}
        {hover !== null ? <line x1={x(hover + 0.5)} x2={x(hover + 0.5)} y1={10} y2={stripTop + stripH} stroke={INK} strokeWidth={1} strokeDasharray="2 2" /> : null}
      </svg>
      <p className="lead-readout" aria-live="polite">
        {hover !== null
          ? <>{clock(hover)} from first scored epoch · expert <b>{STAGE_NAMES[night.expert[hover]]}</b> · model <b>{STAGE_NAMES[night.model[hover]]}</b></>
          : <>Record {night.record} · macro-F1 {fmt("lead.night.macro_f1")} · differs on {fmt("lead.night.disagreements", 0)} of {fmt("lead.night.epochs", 0)} epochs · <span className="lead-hint">Hover or use ←/→ keys to inspect epochs</span></>}
      </p>
    </div>
  );
}

const cohorts = [
  { key: "holdout", label: "Boston, model vs expert", color: ACCENT },
  { key: "coimbra_scorer1", label: "Coimbra, model vs expert 1", color: INK_3 },
  { key: "coimbra_scorer2", label: "Coimbra, model vs expert 2", color: INK_3 },
  { key: "coimbra_scorers", label: "Coimbra, expert vs expert", color: INK }
] as const;

function PatientScores() {
  const [ref, W] = useWidth(640);
  const [hoveredDot, setHoveredDot] = useState<{ label: string; value: number; key: string; index: number } | null>(null);
  const narrow = W < 520;  // labels go above each row instead of beside it
  const left = narrow ? 8 : 188, right = 40, rowH = narrow ? 56 : 40, top = 8;
  const lo = 0.3, hi = 1;
  const x = (v: number) => left + ((v - lo) / (hi - lo)) * (W - left - right);
  const H = top + cohorts.length * rowH + (narrow ? 34 : 26);
  const ticks = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
  const jitter = (i: number) => (((i * 2654435761) >>> 0) % 1000) / 1000 - 0.5;
  return (
    <div className="lead-plot" ref={ref}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" onPointerLeave={() => setHoveredDot(null)} aria-label={cohorts.map((c) => `${c.label}: median ${fmt(`lead.${c.key}.median`)}, range ${fmt(`lead.${c.key}.min`, 2)} to ${fmt(`lead.${c.key}.max`, 2)}, ${fmt(`lead.${c.key}.n`, 0)} nights`).join("; ")}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={top} y2={top + cohorts.length * rowH} stroke={RULE} />
            <text x={x(t)} y={top + cohorts.length * rowH + (narrow ? 24 : 16)} textAnchor="middle" className="lead-axis">{t.toFixed(1)}</text>
          </g>
        ))}
        {cohorts.map((c, row) => {
          const cy = top + row * rowH + rowH / 2 + (narrow ? 8 : 0);
          const spread = narrow ? rowH - 30 : rowH - 16;
          const values = lead.patients[c.key];
          const median = lead.summary[c.key].median;
          return (
            <g key={c.key}>
              {row === cohorts.length - 1 ? <line x1={0} x2={W - right} y1={top + row * rowH} y2={top + row * rowH} stroke="#b8b4a9" /> : null}
              {narrow
                ? <text x={left} y={cy - rowH / 2 + 4} className="lead-row-label">{c.label} <tspan className="lead-axis">· {fmt(`lead.${c.key}.n`, 0)}</tspan></text>
                : <><text x={0} y={cy - 2} className="lead-row-label">{c.label}</text>
                  <text x={0} y={cy + 12} className="lead-axis">{fmt(`lead.${c.key}.n`, 0)} nights</text></>}
              {values.map((v, i) => {
                const active = hoveredDot?.key === c.key && hoveredDot.index === i;
                return (
                  <circle
                    key={i}
                    cx={x(v)}
                    cy={cy + jitter(i) * spread}
                    r={active ? 4.2 : 2.3}
                    fill={c.color}
                    fillOpacity={active ? 0.95 : 0.42}
                    style={{ cursor: "pointer" }}
                    onPointerEnter={() => setHoveredDot({ label: c.label, value: v, key: c.key, index: i })}
                  >
                    <title>{`${c.label}: ${v.toFixed(3)}`}</title>
                  </circle>
                );
              })}
              <line x1={x(median)} x2={x(median)} y1={cy - spread / 2 - 4} y2={cy + spread / 2 + 4} stroke={INK} strokeWidth={2} />
              <text x={W - right + 6} y={cy + 4} className="lead-median">{median.toFixed(2)}</text>
              {c.key === "holdout" ? <circle cx={x(night.macro_f1)} cy={cy} r={5} fill="none" stroke={INK} strokeWidth={1.2}><title>Night shown in Figure 2</title></circle> : null}
            </g>
          );
        })}
      </svg>
      <p className="lead-readout" aria-live="polite">
        {hoveredDot
          ? <>{hoveredDot.label} · night #{hoveredDot.index + 1} · macro-F1 <b>{hoveredDot.value.toFixed(3)}</b></>
          : <>Each dot is one night (hover a dot to read its score). The bar and number at right show the median; the ring marks <a href="#held-out-night">Figure 2</a>.</>}
      </p>
    </div>
  );
}

export function HeldOutNightFigure() {
  return (
    <figure className="lead-figure" id="held-out-night">
      <figcaption>
        <b>Figure 2. Expert and Model C on one held-out Boston night.</b> This Boston night scored closest to the median.
        Dark marks show where the stage labels differ, and the shaded band marks the longest continuous disagreement.
        Time starts at the first scored epoch.
      </figcaption>
      <Hypnogram />
    </figure>
  );
}

export function ValidationDistributionFigure() {
  return (
    <figure className="lead-figure" id="validation-distribution">
      <figcaption>
        <b>Figure 3. Per-night macro-F1 for Model C in Boston holdout and Coimbra.</b> Each night is scored over the stages present in its
        reference labels. Held-out Boston scores vary the most. The last row shows agreement between the Coimbra experts.
        These medians summarize individual nights. The <a href="#validation-scores">test scores</a> pool epochs across each cohort.
      </figcaption>
      <PatientScores />
    </figure>
  );
}
