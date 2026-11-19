"use client";

// Figure 1: one typical held-out night (A) and every patient's score in every test set (B).
// Data: scripts/export_lead_figure.py -> content/generated/lead-figure.json; numbers in text come from the registry.
import { useEffect, useRef, useState } from "react";
import lead from "../../content/generated/lead-figure.json";
import { fmt } from "../../lib/results";

const STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"];
// Hypnogram rows, top to bottom, in the conventional order: Wake, REM, N1, N2, N3.
const ROW_OF_STAGE = [0, 2, 3, 4, 1];
const ROW_LABELS = ["Wake", "REM", "N1", "N2", "N3"];
const INK = "#1b1a17";
const INK_3 = "#6e6c65";
const RULE = "#dedbd2";
const ACCENT = "#1c5cab";

const night = lead.night;
const n = night.expert.length;

function clock(epoch: number) {
  const minutes = Math.round(((night.first_epoch + epoch) * 30) / 60);
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
  const left = 44, right = 8, trackH = 76, gap = 34, stripH = 10, first = 24;
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
  const hourTicks = Array.from({ length: Math.floor(n / 120) + 1 }, (_, h) => h * 120).filter((e) => e <= n);
  const onMove = (event: React.PointerEvent<SVGSVGElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const px = ((event.clientX - box.left) / box.width) * W;
    const i = Math.floor(((px - left) / plotW) * n);
    setHover(i >= 0 && i < n ? i : null);
  };
  return (
    <div className="lead-panel" ref={ref}>
      <p className="lead-panel-title"><b>A</b> One held-out night, expert and model</p>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" onPointerMove={onMove} onPointerLeave={() => setHover(null)}
        aria-label={`Hypnograms for held-out record ${night.record}: expert and model C over ${(n / 120).toFixed(1)} hours of scored sleep; they differ on ${night.disagreements} of ${night.epochs} epochs.`}>
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
        <text x={left - 6} y={stripTop + 8} textAnchor="end" className="lead-axis">Differ</text>
        <rect x={left} y={stripTop} width={plotW} height={stripH} fill="#f2f0ea" />
        {runs.map(([a, b]) => <rect key={a} x={x(a)} y={stripTop} width={Math.max(x(b) - x(a), 0.8)} height={stripH} fill={INK_3} />)}
        {hourTicks.map((e) => (
          <g key={e}>
            <line x1={x(e)} x2={x(e)} y1={stripTop + stripH} y2={stripTop + stripH + 4} stroke={INK_3} />
            <text x={x(e)} y={stripTop + stripH + 16} textAnchor="middle" className="lead-axis">{e / 120} h</text>
          </g>
        ))}
        {hover !== null ? <line x1={x(hover + 0.5)} x2={x(hover + 0.5)} y1={10} y2={stripTop + stripH} stroke={INK} strokeWidth={0.75} strokeDasharray="2 2" /> : null}
      </svg>
      <p className="lead-readout" aria-live="polite">
        {hover !== null
          ? <>{clock(hover)} into the recording · expert <b>{STAGE_NAMES[night.expert[hover]]}</b> · model <b>{STAGE_NAMES[night.model[hover]]}</b></>
          : <>Record {night.record} · macro-F1 {fmt("lead.night.macro_f1")} · differs on {fmt("lead.night.disagreements", 0)} of {fmt("lead.night.epochs", 0)} epochs</>}
      </p>
    </div>
  );
}

const cohorts = [
  { key: "cv", label: "Boston, cross-validation", note: "used to choose the model", color: INK_3 },
  { key: "holdout", label: "Boston, held out", note: "never used for a decision", color: ACCENT },
  { key: "coimbra_scorer1", label: "Coimbra, vs expert 1", note: "another hospital", color: INK_3 },
  { key: "coimbra_scorer2", label: "Coimbra, vs expert 2", note: "same nights", color: INK_3 },
  { key: "coimbra_scorers", label: "Expert 1 vs expert 2", note: "human agreement, same nights", color: INK }
] as const;

function PatientScores() {
  const [ref, W] = useWidth(640);
  const narrow = W < 520;  // labels go above each row instead of beside it
  const left = narrow ? 8 : 168, right = 40, rowH = narrow ? 56 : 40, top = 8;
  const lo = 0.3, hi = 1;
  const x = (v: number) => left + ((v - lo) / (hi - lo)) * (W - left - right);
  const H = top + cohorts.length * rowH + (narrow ? 34 : 26);
  const ticks = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
  const jitter = (i: number) => (((i * 2654435761) >>> 0) % 1000) / 1000 - 0.5;
  return (
    <div className="lead-panel" ref={ref}>
      <p className="lead-panel-title"><b>B</b> Every patient&rsquo;s score, model C</p>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={cohorts.map((c) => `${c.label}: median ${fmt(`lead.${c.key}.median`)}, range ${fmt(`lead.${c.key}.min`, 2)} to ${fmt(`lead.${c.key}.max`, 2)}, ${fmt(`lead.${c.key}.n`, 0)} nights`).join("; ")}>
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
                  <text x={0} y={cy + 12} className="lead-axis">{c.note} · {fmt(`lead.${c.key}.n`, 0)}</text></>}
              {values.map((v, i) => (
                <circle key={i} cx={x(v)} cy={cy + jitter(i) * spread} r={2.1} fill={c.color} fillOpacity={0.42}>
                  <title>{v.toFixed(3)}</title>
                </circle>
              ))}
              <line x1={x(median)} x2={x(median)} y1={cy - spread / 2 - 4} y2={cy + spread / 2 + 4} stroke={INK} strokeWidth={2} />
              <text x={W - right + 6} y={cy + 4} className="lead-median">{median.toFixed(2)}</text>
              {c.key === "holdout" ? <circle cx={x(night.macro_f1)} cy={cy} r={5} fill="none" stroke={INK} strokeWidth={1.2}><title>Night shown in A</title></circle> : null}
            </g>
          );
        })}
      </svg>
      <p className="lead-readout">Each dot is one night; the bar and the number at right are the median. The ringed dot is the night in A.</p>
    </div>
  );
}

export function LeadFigure() {
  return (
    <figure className="lead-figure">
      <div className="lead-panels">
        <Hypnogram />
        <PatientScores />
      </div>
      <figcaption>
        <b>Figure 1.</b> (A) The held-out night closest to the held-out median, chosen by that rule: the expert&rsquo;s
        hypnogram, model C&rsquo;s, and the epochs where they differ. (B) Macro-F1 for every night, averaged over the stages
        the expert scored that night. The bottom row scores the two Coimbra experts against each other on the same nights:
        the model&rsquo;s held-out median is {fmt("lead.holdout.median", 2)}, its Coimbra median {fmt("lead.coimbra_scorer1.median", 2)} and{" "}
        {fmt("lead.coimbra_scorer2.median", 2)}, and the experts&rsquo; {fmt("lead.coimbra_scorers.median", 2)}.
      </figcaption>
    </figure>
  );
}

export function LeadSummary() {
  return (
    <p className="lead-summary">
      A model that reads an overnight sleep recording and labels every 30 seconds as wake, light sleep (N1, N2), deep
      sleep (N3) or REM sleep. On {fmt("lead.holdout.n", 0)} Boston patients held out from development, its typical night
      scores {fmt("lead.holdout.median", 2)} macro-F1. On {fmt("lead.coimbra_scorer1.n", 0)} nights from a hospital in Coimbra
      it scores {fmt("lead.coimbra_scorer1.median", 2)} against one expert, where two experts scoring the same nights agree
      at {fmt("lead.coimbra_scorers.median", 2)}.
    </p>
  );
}
