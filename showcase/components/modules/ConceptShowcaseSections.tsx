"use client";

import { useEffect, useRef, useState } from "react";
import conceptData from "../../content/generated/concept-showcase.json";
import { ACCENT, INK, INK_3, RULE, STAGE_NAMES, modalityInk, sensorColors } from "../../design/palette";
import { fmt } from "../../lib/results";
import { formatMetric, formatPercent } from "../ui/ResearchPrimitives";

const ROW_OF_STAGE = [0, 2, 3, 4, 1];
const ROW_LABELS = ["Wake", "REM", "N1", "N2", "N3"];
const STAGES = ["Wake", "N1", "N2", "N3", "REM"] as const;

function useWidth(fallback: number) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) =>
      setWidth(Math.max(280, Math.round(entry.contentRect.width)))
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

function clock(epoch: number) {
  const totalMinutes = Math.floor(epoch / 2);
  const hours = Math.floor(totalMinutes / 60);
  const mins = String(totalMinutes % 60).padStart(2, "0");
  const secs = epoch % 2 === 1 ? "30" : "00";
  return `${hours} h ${mins} m ${secs} s`;
}

export function ConceptHypnogramFigure() {
  const night = conceptData.night_trace;
  const n = night.expert.length;
  const [hover, setHover] = useState<number | null>(null);
  const [pinned, setPinned] = useState<number | null>(null);
  const [ref, W] = useWidth(640);
  const activeIdx = hover ?? pinned;

  const left = 62;
  const right = 14;
  const trackH = 76;
  const gap = 34;
  const stripH = 10;
  const first = 24;
  const plotW = W - left - right;
  const x = (i: number) => left + (i / n) * plotW;
  const tracks = [
    { label: "Expert", stages: night.expert, color: INK, top: first },
    {
      label: "Model D (Concept-BiGRU)",
      stages: night.model,
      color: ACCENT,
      top: first + trackH + gap,
    },
  ];
  const stripTop = first + 2 * trackH + gap + 18;
  const H = stripTop + stripH + 30;
  const y = (top: number, stage: number) => top + (ROW_OF_STAGE[stage] / 4) * trackH;
  const path = (stages: number[], top: number) => {
    let d = `M${x(0).toFixed(1)} ${y(top, stages[0]).toFixed(1)}`;
    for (let i = 0; i < stages.length; i++) {
      d += `H${x(i + 1).toFixed(1)}`;
      if (i + 1 < stages.length && stages[i + 1] !== stages[i]) {
        d += `V${y(top, stages[i + 1]).toFixed(1)}`;
      }
    }
    return d;
  };

  const runs: Array<[number, number]> = [];
  const uncertainRuns: Array<[number, number]> = [];
  for (let i = 0; i < n; i++) {
    if (night.expert[i] !== night.model[i]) {
      const last = runs[runs.length - 1];
      if (last && last[1] === i) last[1] = i + 1;
      else runs.push([i, i + 1]);
    }
    if (night.set_sizes_90[i] > 1) {
      const lastU = uncertainRuns[uncertainRuns.length - 1];
      if (lastU && lastU[1] === i) lastU[1] = i + 1;
      else uncertainRuns.push([i, i + 1]);
    }
  }
  const hourTicks = Array.from({ length: Math.floor(n / 120) + 1 }, (_, h) => h * 120);

  const idxFromEvent = (event: React.MouseEvent<SVGSVGElement> | React.PointerEvent<SVGSVGElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const px = ((event.clientX - box.left) / box.width) * W;
    const i = Math.floor(((px - left) / plotW) * n);
    return i >= 0 && i < n ? i : null;
  };
  const onMove = (event: React.PointerEvent<SVGSVGElement>) => {
    setHover(idxFromEvent(event));
  };
  const onSvgClick = (event: React.MouseEvent<SVGSVGElement>) => {
    const idx = idxFromEvent(event);
    if (idx === null) return;
    setPinned((prev) => (prev === idx ? null : idx));
  };
  const onKeyDown = (event: React.KeyboardEvent<SVGSVGElement>) => {
    const cur = activeIdx;
    if (event.key === "ArrowRight") {
      event.preventDefault();
      const next = cur === null ? 0 : Math.min(cur + 1, n - 1);
      setHover(next);
      setPinned(next);
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      const next = cur === null ? n - 1 : Math.max(cur - 1, 0);
      setHover(next);
      setPinned(next);
    } else if (event.key === "Home") {
      event.preventDefault();
      setHover(0);
      setPinned(0);
    } else if (event.key === "End") {
      event.preventDefault();
      setHover(n - 1);
      setPinned(n - 1);
    } else if (event.key === "Escape") {
      setHover(null);
      setPinned(null);
    }
  };

  const isHoldout = night.cohort === "holdout";
  return (
    <figure className="lead-figure" id="held-out-night">
      <figcaption>
        <b>
          Figure 1. Expert and Model D on one{" "}
          {isHoldout ? "unseen Boston night" : "Boston development night"}.
        </b>{" "}
        A hypnogram tracks sleep depth from Wake at the top down to deep sleep (N3) at the bottom across an overnight recording. This unseen patient ({night.record}) sits right at the held-out cohort median. Shaded bands mark epochs where the model&rsquo;s 90% uncertainty set holds more than one plausible stage (concentrated at transitions); bottom ticks mark disagreements ({night.disagreements} of {night.epochs} epochs).
      </figcaption>
      <div className="lead-plot" ref={ref}>
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="lead-svg-interactive"
          role="img"
          aria-roledescription="interactive hypnogram"
          tabIndex={0}
          onPointerMove={onMove}
          onPointerLeave={() => setHover(null)}
          onClick={onSvgClick}
          onKeyDown={onKeyDown}
          aria-label={`Expert and Model D stages across ${night.epochs} epochs; macro-F1 ${formatMetric(night.macro_f1)}.`}
        >
          {uncertainRuns.map(([a, b]) => (
            <rect
              key={`u-${a}`}
              x={x(a)}
              y={first - 2}
              width={Math.max(x(b) - x(a), 0.9)}
              height={stripTop - first + 2}
              fill={ACCENT}
              fillOpacity={0.12}
            />
          ))}
          {tracks.map((t) => (
            <g key={t.label}>
              <text x={left} y={t.top - 12} className="lead-track-label">
                {t.label}
              </text>
              {ROW_LABELS.map((label, row) => (
                <g key={label}>
                  <line
                    x1={left}
                    x2={W - right}
                    y1={t.top + (row / 4) * trackH}
                    y2={t.top + (row / 4) * trackH}
                    stroke={RULE}
                    strokeWidth={1}
                  />
                  <text
                    x={left - 6}
                    y={t.top + (row / 4) * trackH + 3.5}
                    textAnchor="end"
                    className="lead-axis"
                  >
                    {label}
                  </text>
                </g>
              ))}
              <path
                d={path(t.stages, t.top)}
                fill="none"
                stroke={t.color}
                strokeWidth={1.4}
                strokeLinejoin="round"
              />
            </g>
          ))}
          <text x={left - 6} y={stripTop + 8} textAnchor="end" className="lead-axis">
            Mismatch
          </text>
          <rect x={left} y={stripTop} width={plotW} height={stripH} fill="#f2f0ea" />
          {runs.map(([a, b]) => (
            <rect
              key={`m-${a}`}
              x={x(a)}
              y={stripTop}
              width={Math.max(x(b) - x(a), 0.8)}
              height={stripH}
              fill={INK_3}
            />
          ))}
          {hourTicks.map((e) => (
            <g key={e}>
              <line x1={x(e)} x2={x(e)} y1={stripTop + stripH} y2={stripTop + stripH + 4} stroke={INK_3} />
              <text x={x(e)} y={stripTop + stripH + 16} textAnchor="middle" className="lead-axis">
                {e / 120} h
              </text>
            </g>
          ))}
          {pinned !== null ? (
            <line
              x1={x(pinned + 0.5)}
              x2={x(pinned + 0.5)}
              y1={10}
              y2={stripTop + stripH}
              stroke={ACCENT}
              strokeWidth={1.6}
            />
          ) : null}
          {hover !== null && hover !== pinned ? (
            <line
              x1={x(hover + 0.5)}
              x2={x(hover + 0.5)}
              y1={10}
              y2={stripTop + stripH}
              stroke={INK}
              strokeWidth={1}
              strokeDasharray="2 2"
            />
          ) : null}
        </svg>
        <div className="lead-readout-stack" aria-live="polite">
          <p className="lead-readout lead-readout-summary">
            <span>
              Record <b>{night.record}</b> · macro-F1 <b>{formatMetric(night.macro_f1)}</b> ·{" "}
              {night.disagreements} mismatches of {night.epochs} epochs
            </span>
            {pinned !== null ? (
              <button
                type="button"
                className="lead-clear-btn"
                onClick={() => {
                  setPinned(null);
                  setHover(null);
                }}
              >
                Clear pin ({clock(pinned)})
              </button>
            ) : null}
          </p>
          <p className="lead-readout">
            {activeIdx !== null ? (
              <>
                {pinned !== null && activeIdx === pinned ? <b>Pinned · </b> : null}
                {clock(activeIdx)} · expert <b>{STAGE_NAMES[night.expert[activeIdx]]}</b> · model{" "}
                <b>{STAGE_NAMES[night.model[activeIdx]]}</b> · 90% prediction set:{" "}
                <b>
                  {night.set_sizes_90[activeIdx] === 1
                    ? "1 stage (confident)"
                    : `${night.set_sizes_90[activeIdx]} stages (uncertain boundary)`}
                </b>{" "}
                · top elevated branch: <b>{night.concept_labels[night.top_concept[activeIdx]]}</b>
              </>
            ) : (
              <span className="lead-hint">
                Hover, click to pin an epoch, or use ←/→ keys to inspect expert vs. model stage, 90% uncertainty set and top physiological branch
              </span>
            )}
          </p>
        </div>
      </div>
    </figure>
  );
}

export function ConceptExternalDistFigure() {
  const ext = conceptData.external_eval;
  const [ref, W] = useWidth(640);
  const [hoveredDot, setHoveredDot] = useState<{ label: string; value: number; key: string; index: number } | null>(null);
  if (!ext) return null;

  const cohorts = [
    {
      key: "holdout",
      label: "Boston holdout, Model D vs expert",
      values: ext.holdout.patients,
      median: ext.holdout.subj_median,
      color: ACCENT,
    },
    {
      key: "isruc_s1",
      label: "Coimbra, Model D vs expert 1",
      values: ext.isruc.scorer1.patients,
      median: ext.isruc.scorer1.subj_median,
      color: INK_3,
    },
    {
      key: "isruc_s2",
      label: "Coimbra, Model D vs expert 2",
      values: ext.isruc.scorer2.patients,
      median: ext.isruc.scorer2.subj_median,
      color: INK_3,
    },
    ...(ext.isruc.scorers_agreement
      ? [
          {
            key: "coimbra_scorers",
            label: "Coimbra, expert vs expert",
            values: ext.isruc.scorers_patients,
            median: ext.isruc.scorers_agreement.median,
            color: INK,
          },
        ]
      : []),
  ];

  const narrow = W < 520;
  const left = narrow ? 8 : 228;
  const right = 40;
  const rowH = narrow ? 64 : 40;
  const top = 8;
  const lo = 0.25;
  const hi = 1.0;
  const x = (v: number) => left + ((v - lo) / (hi - lo)) * (W - left - right);
  const H = top + cohorts.length * rowH + (narrow ? 34 : 26);
  const ticks = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
  const jitter = (i: number) => (((i * 2654435761) >>> 0) % 1000) / 1000 - 0.5;
  const nightF1 = conceptData.night_trace.macro_f1;

  const findNearest = (clientX: number, clientY: number, box: DOMRect) => {
    const px = ((clientX - box.left) / box.width) * W;
    const py = ((clientY - box.top) / box.height) * H;
    let best: { label: string; value: number; key: string; index: number; dist: number } | null = null;
    cohorts.forEach((c, row) => {
      const cy = top + row * rowH + rowH / 2 + (narrow ? 8 : 0);
      const spread = narrow ? rowH - 34 : rowH - 16;
      c.values.forEach((v, i) => {
        const dx = x(v) - px;
        const dy = cy + jitter(i) * spread - py;
        const dist = dx * dx + dy * dy;
        if (dist <= 324 && (!best || dist < best.dist)) {
          best = { label: c.label, value: v, key: c.key, index: i, dist };
        }
      });
    });
    return best;
  };

  const onSvgKey = (event: React.KeyboardEvent<SVGSVGElement>) => {
    if (event.key === "Escape") {
      setHoveredDot(null);
      return;
    }
    if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) return;
    event.preventDefault();
    const rowIdx = Math.max(
      0,
      hoveredDot ? cohorts.findIndex((c) => c.key === hoveredDot.key) : 0
    );
    const nextRow =
      event.key === "ArrowDown"
        ? Math.min(rowIdx + 1, cohorts.length - 1)
        : event.key === "ArrowUp"
          ? Math.max(rowIdx - 1, 0)
          : rowIdx;
    const cohort = cohorts[nextRow];
    const idx = hoveredDot && nextRow === rowIdx ? hoveredDot.index : 0;
    const nextIdx =
      event.key === "ArrowRight"
        ? Math.min(idx + 1, cohort.values.length - 1)
        : event.key === "ArrowLeft"
          ? Math.max(idx - 1, 0)
          : Math.min(idx, cohort.values.length - 1);
    setHoveredDot({ label: cohort.label, value: cohort.values[nextIdx], key: cohort.key, index: nextIdx });
  };

  return (
    <figure className="lead-figure" id="validation-distribution">
      <figcaption>
        <b>Figure 2. Model D per-night macro-F1 on unseen Boston holdout and Coimbra nights.</b>{" "}
        Each dot is one night scored over the stages present in its reference labels; the ring marks the night in Figure 1.
      </figcaption>
      <div className="lead-plot" ref={ref}>
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="lead-svg-interactive"
          role="img"
          tabIndex={0}
          aria-roledescription="interactive swarm plot"
          aria-label="Per-night macro-F1 across Boston holdout and Coimbra cohorts. Use arrow keys or hover to inspect nights."
          onPointerMove={(e) => {
            const nearest = findNearest(e.clientX, e.clientY, e.currentTarget.getBoundingClientRect());
            setHoveredDot(nearest);
          }}
          onPointerLeave={() => setHoveredDot(null)}
          onKeyDown={onSvgKey}
        >
          {ticks.map((t) => (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={top} y2={top + cohorts.length * rowH} stroke={RULE} />
              <text x={x(t)} y={top + cohorts.length * rowH + (narrow ? 24 : 16)} textAnchor="middle" className="lead-axis">
                {t.toFixed(1)}
              </text>
            </g>
          ))}
          <text x={W - right + 6} y={top + 2} className="lead-axis">
            Median
          </text>
          {cohorts.map((c, row) => {
            const cy = top + row * rowH + rowH / 2 + (narrow ? 8 : 0);
            const spread = narrow ? rowH - 34 : rowH - 16;
            const fig1Idx =
              c.key === "holdout"
                ? c.values.reduce(
                    (bestI, val, idx) =>
                      Math.abs(val - nightF1) < Math.abs(c.values[bestI] - nightF1) ? idx : bestI,
                    0
                  )
                : -1;
            return (
              <g key={c.key}>
                {row === cohorts.length - 1 ? (
                  <line x1={0} x2={W - right} y1={top + row * rowH} y2={top + row * rowH} stroke="#b8b4a9" />
                ) : null}
                {narrow ? (
                  <text x={left} y={cy - rowH / 2 + 4} className="lead-row-label">
                    {c.label} <tspan className="lead-axis">· {c.values.length}</tspan>
                  </text>
                ) : (
                  <>
                    <text x={0} y={cy - 2} className="lead-row-label">{c.label}</text>
                    <text x={0} y={cy + 12} className="lead-axis">{c.values.length} nights</text>
                  </>
                )}
                {c.values.map((v, i) => {
                  const active = hoveredDot?.key === c.key && hoveredDot.index === i;
                  return (
                    <circle
                      key={i}
                      cx={x(v)}
                      cy={cy + jitter(i) * spread}
                      r={active ? 4.4 : 2.3}
                      fill={c.color}
                      fillOpacity={active ? 1 : 0.42}
                      stroke={active ? INK : "none"}
                      strokeWidth={active ? 1.5 : 0}
                      style={{ cursor: "pointer" }}
                      onPointerEnter={() => setHoveredDot({ label: c.label, value: v, key: c.key, index: i })}
                    >
                      <title>{`${c.label}: ${v.toFixed(3)}`}</title>
                    </circle>
                  );
                })}
                <line x1={x(c.median)} x2={x(c.median)} y1={cy - spread / 2 - 4} y2={cy + spread / 2 + 4} stroke={INK} strokeWidth={2} />
                <text x={W - right + 6} y={cy + 4} className="lead-median">{c.median.toFixed(2)}</text>
                {fig1Idx >= 0 ? (
                  <circle
                    cx={x(c.values[fig1Idx])}
                    cy={cy + jitter(fig1Idx) * spread}
                    r={5.2}
                    fill="none"
                    stroke={INK}
                    strokeWidth={1.35}
                  >
                    <title>{`Night shown in Figure 1 (${conceptData.night_trace.record}: ${formatMetric(nightF1)})`}</title>
                  </circle>
                ) : null}
              </g>
            );
          })}
        </svg>
        <p className="lead-readout" aria-live="polite">
          {hoveredDot ? (
            <>
              {hoveredDot.label} · sorted night #{hoveredDot.index + 1} · macro-F1 <b>{hoveredDot.value.toFixed(3)}</b>
            </>
          ) : (
            <>
              Hover or use arrow keys to inspect individual nights. Per-night medians appear on the right; pooled all-epoch macro-F1 is{" "}
              {fmt("ext.concept79.holdout.calibrated.macro_f1")} in Boston holdout and{" "}
              {fmt("ext.concept79.isruc.calibrated_scorer1.macro_f1")} / {fmt("ext.concept79.isruc.calibrated_scorer2.macro_f1")} in Coimbra.
            </>
          )}
        </p>
      </div>
    </figure>
  );
}

export function ConceptProgressionSection() {
  const ladder = conceptData.ladder;
  const activeId = "model_d";
  const allFolds = ladder.flatMap((row) => row.folds);
  const lo = Math.floor((Math.min(...allFolds) - 0.005) * 20) / 20;
  const hi = Math.ceil((Math.max(...allFolds) + 0.005) * 20) / 20;
  const pos = (v: number) => `${((v - lo) / (hi - lo)) * 100}%`;
  const ticks = Array.from(
    { length: Math.round((hi - lo) / 0.05) + 1 },
    (_, i) => lo + i * 0.05
  );
  const baseline = ladder[0];

  return (
    <div className="research-model-compare">
      <div className="research-selected-distribution-head">
        <strong>
          Main model progression across the 320 fixed Boston CinC folds (Models A → B → C → D)
        </strong>
      </div>
      <div className="research-model-compare-axis" aria-hidden="true">
        <span />
        <div>
          {ticks.map((tick) => (
            <small key={tick} style={{ left: pos(tick) }}>
              {tick.toFixed(2)}
            </small>
          ))}
        </div>
        <span />
      </div>
      {ladder.map((row) => {
        const minF = Math.min(...row.folds);
        const maxF = Math.max(...row.folds);
        return (
          <div
            key={row.id}
            className={`research-model-compare-row${row.id === activeId ? " is-selected" : ""}`}
          >
            <strong title={`${row.kind} · ${row.size} · ${row.train_patients}`}>
              {row.label}
              <small style={{ display: "block", fontWeight: 400, color: INK_3 }}>
                {row.size} · {row.train_patients}
              </small>
            </strong>
            <div
              className="research-model-compare-track"
              role="img"
              aria-label={`${row.label}: mean ${formatMetric(row.macro_f1)}, folds ${formatMetric(minF)} to ${formatMetric(maxF)}`}
            >
              {ticks.map((tick) => (
                <i key={tick} className="grid" style={{ left: pos(tick) }} />
              ))}
              <i className="baseline" style={{ left: pos(baseline.macro_f1) }} />
              <i
                className="range"
                style={{ left: pos(minF), width: `calc(${pos(maxF)} - ${pos(minF)})` }}
              />
              {row.folds.map((v, idx) => (
                <i key={idx} className="fold" style={{ left: pos(v) }} />
              ))}
              <i className="mean" style={{ left: pos(row.macro_f1) }} />
            </div>
            <span>{formatMetric(row.macro_f1)}</span>
          </div>
        );
      })}
      <p className="research-chart-key research-model-compare-key">
        <u className="fold" /> one test fold <u className="mean" /> mean <u className="baseline" /> Model A (single epoch)
      </p>
    </div>
  );
}

export function ConceptAttributionSection() {
  const attr = conceptData.concept_attribution;
  const [selected, setSelected] = useState(attr.concepts[0].id);
  const current = attr.concepts.find((c) => c.id === selected) ?? attr.concepts[0];
  const maxDrop = Math.max(...attr.concepts.map((c) => Math.abs(c.local_macro_f1_drop)), 0.005);

  const branchColor = (id: string) =>
    id === "rem_tone"
      ? modalityInk.eog
      : id === "autonomic"
        ? sensorColors["Heart"]
        : modalityInk.eeg;

  return (
    <div className="research-evidence">
      <p className="research-evidence-lede">
        Model D scores four physiological branches separately before reading across the whole night (reaching {formatMetric(attr.local_only_macro_f1)} from the four branches alone, plus {formatMetric(attr.temporal_context_gain)} from whole-night sequence memory). Select a branch below to see which sleep stages lose accuracy when that branch is turned off:
      </p>

      <div className="research-concept-list" role="group" aria-label="Additive physiological concept branches">
        {attr.concepts.map((c) => {
          const dropAbs = Math.abs(c.local_macro_f1_drop);
          const isActive = c.id === current.id;
          return (
            <button
              key={c.id}
              type="button"
              className={`research-concept-btn${isActive ? " is-active" : ""}`}
              aria-pressed={isActive}
              onClick={() => setSelected(c.id)}
            >
              <span className="research-concept-btn-name">
                {c.label} <small>({c.dims} inputs → {c.width}-unit stem)</small>
              </span>
              <i aria-hidden="true">
                <b
                  style={{
                    width: `${(dropAbs / maxDrop) * 100}%`,
                    background: branchColor(c.id),
                  }}
                />
              </i>
              <strong>−{formatMetric(dropAbs)}</strong>
            </button>
          );
        })}
      </div>

      <p className="research-concept-detail" aria-live="polite">
        <b>{current.label}</b> (per-stage change when turned off before whole-night smoothing):{" "}
        {STAGES.map((st, idx) => {
          const delta = current.per_stage_local_f1_delta[st];
          const signed = `${delta >= 0 ? "+" : "−"}${Math.abs(delta).toFixed(3)}`;
          return `${idx ? " · " : ""}${st} ${signed}`;
        })}
        .
      </p>
      <p className="research-protocol-source">
        Because the four branches are trained together, turning off the slow-wave branch alone causes a small drop when neighbouring features (broadband brain power in the wake/drowsiness branch and frontal slow waves leaking into the eye channel) still support deep N3 sleep.
      </p>

      <h5 style={{ marginTop: "1.25rem", marginBottom: "0.5rem" }}>
        Learned past vs. future context gates by stage
      </h5>
      <div className="research-model-table-wrap">
        <table className="research-model-table">
          <thead>
            <tr>
              <th scope="col">Stage</th>
              <th scope="col">Past gate (history)</th>
              <th scope="col">Future gate (look-ahead)</th>
              <th scope="col">Physiological role</th>
            </tr>
          </thead>
          <tbody>
            {attr.stage_gates.map((g) => (
              <tr key={g.stage}>
                <th scope="row">{g.stage}</th>
                <td>{formatMetric(g.past)}</td>
                <td>{formatMetric(g.future)}</td>
                <td>
                  {g.future > g.past + 0.1
                    ? "Looks ahead into upcoming transitions"
                    : g.past > g.future + 0.15
                      ? "Anchors strongly to preceding bout"
                      : "Balanced past and future context"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function ConformalUncertaintySection() {
  const [target, setTarget] = useState<"coverage_90" | "coverage_95">("coverage_90");
  const block = conceptData.conformal[target];
  const marginal = block.marginal;
  const mondrian = block.mondrian;
  const extHo = conceptData.external_eval?.holdout?.[target === "coverage_90" ? "conformal_90" : "conformal_95"];
  const extIs = conceptData.external_eval?.isruc?.[target === "coverage_90" ? "conformal_90" : "conformal_95"];

  return (
    <div className="research-evidence">
      <div className="research-pill-group">
        <span>Conformal target coverage:</span>
        <button
          type="button"
          className={`research-pill${target === "coverage_90" ? " is-active" : ""}`}
          aria-pressed={target === "coverage_90"}
          onClick={() => setTarget("coverage_90")}
        >
          90% target coverage
        </button>
        <button
          type="button"
          className={`research-pill${target === "coverage_95" ? " is-active" : ""}`}
          aria-pressed={target === "coverage_95"}
          onClick={() => setTarget("coverage_95")}
        >
          95% target coverage
        </button>
      </div>

      <div className="research-model-table-wrap">
        <table className="research-model-table">
          <thead>
            <tr>
              <th scope="col">Cohort &amp; method</th>
              <th scope="col">Overall cov.</th>
              <th scope="col">N1 cov.</th>
              <th scope="col">N2 / N3 cov.</th>
              <th scope="col">Mean set size</th>
              <th scope="col" title="Accuracy on epochs where the prediction set contains exactly 1 stage">
                Single-stage acc.
              </th>
              <th scope="col" title="Mean set size on stage-transition epochs vs. steady interior bouts">
                Set size: Trans. / Steady
              </th>
            </tr>
          </thead>
          <tbody>
            {[
              { name: "Boston CV · Standard split conformal (1 global threshold)", m: marginal, adopted: false },
              { name: "Boston CV · Mondrian conformal (per-stage thresholds)", m: mondrian, adopted: true },
            ].map(({ name, m, adopted }) => (
              <tr key={name} className={adopted ? "is-adopted" : undefined}>
                <th scope="row">{name}</th>
                <td>{formatPercent(m.empirical_coverage)}</td>
                <td>{formatPercent(m.per_stage.N1.coverage)}</td>
                <td>
                  {formatPercent(m.per_stage.N2.coverage)} / {formatPercent(m.per_stage.N3.coverage)}
                </td>
                <td>{m.mean_set_size.toFixed(2)} stages</td>
                <td>{formatPercent(m.singleton_accuracy)}</td>
                <td>
                  {m.transition_epochs.mean_set_size.toFixed(2)} / {m.interior_epochs.mean_set_size.toFixed(2)}
                </td>
              </tr>
            ))}
            {extHo ? (
              <tr>
                <th scope="row">Boston Holdout (200) · fixed out-of-fold Mondrian</th>
                <td>{formatPercent(extHo.empirical_coverage)}</td>
                <td>{formatPercent(extHo.per_stage_coverage.N1)}</td>
                <td>
                  {formatPercent(extHo.per_stage_coverage.N2)} / {formatPercent(extHo.per_stage_coverage.N3)}
                </td>
                <td>{extHo.mean_set_size.toFixed(2)} stages</td>
                <td>{formatPercent(extHo.singleton_accuracy)}</td>
                <td>
                  {extHo.transition_set_size.toFixed(2)} / {extHo.interior_set_size.toFixed(2)}
                </td>
              </tr>
            ) : null}
            {extIs ? (
              <tr>
                <th scope="row">Coimbra ISRUC (99, vs. Scorer 1) · fixed out-of-fold Mondrian</th>
                <td>{formatPercent(extIs.empirical_coverage)}</td>
                <td>{formatPercent(extIs.per_stage_coverage.N1)}</td>
                <td>
                  {formatPercent(extIs.per_stage_coverage.N2)} / {formatPercent(extIs.per_stage_coverage.N3)}
                </td>
                <td>{extIs.mean_set_size.toFixed(2)} stages</td>
                <td>{formatPercent(extIs.singleton_accuracy)}</td>
                <td>
                  {extIs.transition_set_size.toFixed(2)} / {extIs.interior_set_size.toFixed(2)}
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
      <p className="research-protocol-source">
        Using one threshold per stage keeps rare light sleep (N1) covered at the target rate in Boston development, unlike a single global threshold. When those fixed Boston thresholds are carried over to unseen hospitals without local recalibration, coverage dips on N2 and N3 in apnea-heavy Boston holdouts and on Wake ({extIs ? formatPercent(extIs.per_stage_coverage.Wake) : "—"}) and N3 ({extIs ? formatPercent(extIs.per_stage_coverage.N3) : "—"}) in Coimbra.
      </p>
    </div>
  );
}

export function SupplementaryExplorationSection() {
  return (
    <div className="research-evidence">
      {conceptData.supplementary_models.length > 0 ? (
        <>
          <h5>Other 704-patient &amp; capacity comparators (kept in Supplementary)</h5>
          <div className="research-model-table-wrap">
            <table className="research-model-table">
              <thead>
                <tr>
                  <th scope="col">Model / Experiment</th>
                  <th scope="col">CinC CV Macro-F1</th>
                  <th scope="col">κ</th>
                  <th scope="col">Notes &amp; complexity</th>
                </tr>
              </thead>
              <tbody>
                {conceptData.supplementary_models.map((row) => (
                  <tr key={row.id}>
                    <th scope="row">{row.label}</th>
                    <td>{formatMetric(row.macro_f1)}</td>
                    <td>{row.kappa.toFixed(2)}</td>
                    <td>{row.size}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : null}

      {conceptData.cue_screen.length > 0 ? (
        <>
          <h5 style={{ marginTop: "1.25rem", marginBottom: "0.5rem" }}>
            Experiment #80: Pre-hoc Cohen&rsquo;s d cue screen (320 dev training folds)
          </h5>
          <p className="research-other-sum-note" style={{ marginBottom: "0.5rem" }}>
            Candidate signals missing from the 474-patient cache were screened on outer-training folds before neural fitting. Although relative delta passed Stage-1 screening, a Fold-0 ablation showed no gain over existing slow-wave markers; combining the winning REM/EMG and autonomic cues across all 5 folds reached {fmt("dev.cues80.calibrated.macro_f1", 4)} macro-F1, so Model D (#79) was retained to avoid extra parameters and ECG/belt dependencies.
          </p>
          <div className="research-model-table-wrap">
            <table className="research-model-table">
              <thead>
                <tr>
                  <th scope="col">Candidate cue</th>
                  <th scope="col">Top clinical contrast</th>
                  <th scope="col">Max |d|</th>
                  <th scope="col">Screen decision</th>
                </tr>
              </thead>
              <tbody>
                {conceptData.cue_screen.map((row) => (
                  <tr key={row.id}>
                    <th scope="row">{row.label}</th>
                    <td>
                      {row.best_contrast} ({row.best_d >= 0 ? "+" : "−"}
                      {Math.abs(row.best_d).toFixed(2)})
                    </td>
                    <td>{formatMetric(row.max_abs_d)}</td>
                    <td>{row.qualified ? `Qualified (${row.branch})` : "Rejected pre-training"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : null}
    </div>
  );
}
