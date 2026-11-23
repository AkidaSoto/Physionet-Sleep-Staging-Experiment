"use client";

import { useState } from "react";
import saliencyData from "../../content/generated/staging-saliency.json";
import isrucStatus from "../../content/generated/isruc-status.json";
import intervals from "../../content/generated/metric-intervals.json";
import { familyColors, modalityInk, stageColors } from "../../design/palette";
import {
  publishedModels,
  reductionLabels,
  signalInfo,
  sweepLabels,
  type PublishedModel
} from "../../content/staging-content";
import { stagingDashboardData, type ModelResult, type SaliencyData } from "../../content/staging-dashboard-data";
import { fmt } from "../../lib/results";
import {
  Citation,
  Exp,
  SecRef,
  formatInterval,
  formatMetric,
  formatPercent
} from "../ui/ResearchPrimitives";

const data = stagingDashboardData;

const ladderIds = ["epoch", "window-features", "window-all", "adopted", "cnn"];

export function ModelExplorer() {
  const ladder = ladderIds
    .map((id) => data.models.find((item) => item.id === id))
    .filter((item): item is ModelResult => Boolean(item));
  const baseline = data.models.find((item) => item.id === "epoch") ?? data.models[0];
  const final = data.headline.diagnostics_model_id;
  const foldScores = ladder.flatMap((item) => item.folds.map((fold) => fold.macro_f1));
  const low = Math.floor((Math.min(...foldScores) - 0.005) * 20) / 20;
  const high = Math.ceil((Math.max(...foldScores) + 0.005) * 20) / 20;
  const position = (value: number) => `${((value - low) / (high - low)) * 100}%`;
  const ticks = Array.from({ length: Math.round((high - low) / 0.05) + 1 }, (_, index) => low + index * 0.05);
  return (
    <div className="research-model-compare">
      <div className="research-selected-distribution-head">
        <strong>
          Macro-F1 on each of the five test folds (<SecRef to="model" />)
        </strong>
      </div>
      <div className="research-model-compare-axis" aria-hidden="true">
        <span />
        <div>
          {ticks.map((tick) => (
            <small key={tick} style={{ left: position(tick) }}>
              {tick.toFixed(2)}
            </small>
          ))}
        </div>
        <span />
      </div>
      {ladder.map((item) => {
        const folds = item.folds.map((fold) => fold.macro_f1);
        return (
          <div key={item.id} className={`research-model-compare-row${item.id === final ? " is-selected" : ""}`}>
            <strong>
              {item.id === "window-features"
                ? "B0 · Features in context"
                : item.id === "cnn"
                  ? "Attention CNN (legacy)"
                  : item.name}
            </strong>
            <div
              className="research-model-compare-track"
              role="img"
              aria-label={`${item.name}: mean ${formatMetric(item.metrics.macro_f1)}, folds ${formatMetric(Math.min(...folds))} to ${formatMetric(Math.max(...folds))}`}
            >
              {ticks.map((tick) => (
                <i key={tick} className="grid" style={{ left: position(tick) }} />
              ))}
              <i className="baseline" style={{ left: position(baseline.metrics.macro_f1) }} />
              <i
                className="range"
                style={{
                  left: position(Math.min(...folds)),
                  width: `calc(${position(Math.max(...folds))} - ${position(Math.min(...folds))})`
                }}
              />
              {folds.map((value, index) => (
                <i key={index} className="fold" style={{ left: position(value) }} />
              ))}
              <i className="mean" style={{ left: position(item.metrics.macro_f1) }} />
            </div>
            <span>{formatMetric(item.metrics.macro_f1)}</span>
          </div>
        );
      })}
      <p className="research-chart-key research-model-compare-key">
        <u className="fold" /> one test fold <u className="mean" /> mean <u className="baseline" /> A, single epoch
      </p>
      <p className="research-ablation-note">
        Paired differences vs. C on the same patients (95% patient-bootstrap CI):{" "}
        {ladder
          .filter((item) => item.id !== "adopted")
          .map((item, index) => {
            const d = intervals.cv_paired_vs_adopted[item.id as keyof typeof intervals.cv_paired_vs_adopted];
            const signed = (v: number) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(3)}`;
            const shortName = item.id === "window-features" ? "B0" : item.name.split(" · ")[0];
            return (
              <span key={item.id}>
                {index ? "; " : ""}
                {shortName} {signed(d.minus_adopted)} ({signed(d.ci[0])} to {signed(d.ci[1])})
              </span>
            );
          })}
        . Adding 2.5 min of context accounts for nearly all the gain over A; the legacy Attention CNN and C are statistically indistinguishable on the 255-patient folds.
      </p>
    </div>
  );
}

export function ConfusionExplorer() {
  return (
    <div className="research-confusion-wrap">
      <p className="research-confusion-scroll-hint">Scroll across to see all stages and metrics.</p>
      <div className="research-confusion research-confusion-static" role="region" aria-label="Cross-validation confusion matrix" tabIndex={0}>
        <span className="research-confusion-axis">Predicted stage →</span>
        <div className="research-confusion-head">
          <span />
          {data.source.stages.map((stage) => (
            <span key={stage}>{stage}</span>
          ))}
          <span title="Of predictions assigned to this stage, the proportion that is correct">Precision</span>
          <span title="Of expert-labelled epochs in this stage, the proportion found by the model">Sensitivity</span>
          <span title="Harmonic mean of precision and sensitivity for this stage">F1</span>
          <span title="Proportion of all other stages correctly excluded">Specificity</span>
        </div>
        {data.diagnostics.confusion_normalized.map((values, rowIndex) => {
          const stage = data.source.stages[rowIndex];
          const metric = data.diagnostics.per_stage[rowIndex];
          const counts = data.diagnostics.confusion_counts;
          const total = counts.flat().reduce((sum, value) => sum + value, 0);
          const positives = counts[rowIndex].reduce((sum, value) => sum + value, 0);
          const falsePositives = counts.reduce((sum, row, index) => sum + (index === rowIndex ? 0 : row[rowIndex]), 0);
          const specificity = 1 - falsePositives / Math.max(total - positives, 1);
          return (
            <div className="research-confusion-row" key={stage}>
              <strong className="swatch" style={{ "--swatch": stageColors[stage] } as React.CSSProperties}>
                {stage}
              </strong>
              {values.map((value, columnIndex) => (
                <span
                  key={`${stage}-${columnIndex}`}
                  className={columnIndex === rowIndex ? "is-diagonal" : undefined}
                  style={{ "--cell-alpha": Math.max(value, 0.035).toFixed(3) } as React.CSSProperties}
                >
                  {formatPercent(value)}
                </span>
              ))}
              <em>{formatPercent(metric.precision)}</em>
              <em>{formatPercent(metric.recall)}</em>
              <em className="is-f1">{formatMetric(metric.f1)}</em>
              <em>{formatPercent(specificity)}</em>
            </div>
          );
        })}
      </div>
      <p className="research-confusion-caption">
        Rows: expert stage; columns: predicted stage. Each row sums to 100%. Stage metrics pool all cross-validation predictions; headline scores average the five folds.
      </p>
    </div>
  );
}

export function ModelEvidence({ part }: { part: "signals" | "time" }) {
  const [hoveredTime, setHoveredTime] = useState<{ kind: string; offset: number; span: number; importance: number } | null>(null);
  const measurements = [...data.signal_importance].sort((a, b) => b.importance - a.importance);
  const remainingGroups = [
    { label: "Other EEG measurements", families: ["spectrum", "topography", "spindles"] },
    { label: "Other eye measurements", families: ["eye-movement"] },
    { label: "Other chin measurements", families: ["muscle-tone"] }
  ]
    .map((group) => {
      const items = measurements.slice(8).filter((item) => group.families.includes(item.family_id));
      return { ...group, count: items.length, importance: items.reduce((sum, item) => sum + item.importance, 0) };
    })
    .filter((group) => group.count > 0);
  const max = Math.max(...measurements.slice(0, 8).map((item) => item.importance), 0.001);
  const measurementColor = (family: string) =>
    family === "eye-movement" ? modalityInk.eog : family === "muscle-tone" ? modalityInk.emg : modalityInk.eeg;
  const positive = (value: number) => Math.max(value, 0);
  const fine = data.time_importance.fine;
  const coarse = data.time_importance.epoch;
  const timeMax = Math.ceil(Math.max(...fine.map((item) => item.importance), ...coarse.map((item) => item.importance), 0.001) * 100) / 100;
  const peak = fine.filter((item) => item.offset_sec >= 0 && item.offset_sec < 30).reduce((a, b) => (b.importance > a.importance ? b : a));
  const coarseAt = (offset: number) => coarse.find((item) => item.offset_sec === offset)?.importance ?? 0;
  const coarseRatio = coarseAt(-30) / Math.max(coarseAt(30), 0.0005);
  return (
    <div className="research-evidence">
      {part === "time" ? (
        <p className="research-evidence-lede">
          For the timing analysis, measurements are swapped only at the indicated position in the window. The upper plot tests a six-second step; the lower plot tests a whole epoch because those measurements repeat across its five steps. <Exp ids={[21]} />
        </p>
      ) : null}
      <div className="research-evidence-grid is-single">
        {part === "signals" ? (
          <div>
            <p className="research-signal-key">
              <span>
                <i style={{ background: modalityInk.eeg }} /> EEG
              </span>
              <span>
                <i style={{ background: modalityInk.eog }} /> Eye activity
              </span>
              <span>
                <i style={{ background: modalityInk.emg }} /> Chin tone
              </span>
              <span>Drop in macro-F1 · mean of {data.importance_folds} test folds</span>
            </p>
            <div className="research-importance-bars research-individual-bars is-compact">
              {measurements.slice(0, 8).map((item) => {
                const label =
                  item.id === "track.o_rel_alpha"
                    ? "Occipital alpha · 2-s spectrum"
                    : item.id === "topography_features.occipital_rel_alpha"
                      ? "Occipital alpha · 30-s spectrum"
                      : item.label.replace(" (1 s)", "");
                return (
                  <div key={item.id}>
                    <span>{label}</span>
                    <i>
                      <b style={{ width: `${(Math.max(item.importance, 0) / max) * 100}%`, background: measurementColor(item.family_id) }} />
                    </i>
                    <strong>{formatMetric(item.importance)}</strong>
                  </div>
                );
              })}
            </div>
            <p className="research-other-sum-note">
              The other {measurements.length - 8} measurements (
              {remainingGroups.map((group) => `${group.count} ${group.label.replace("Other ", "").replace(" measurements", "")}`).join(", ")}) each
              score at most {formatMetric(Math.max(...measurements.slice(8).map((item) => item.importance), 0))}.
            </p>
          </div>
        ) : null}
        {part === "time" ? (
          <div
            className="research-context-plot"
            role="img"
            onPointerLeave={() => setHoveredTime(null)}
            aria-label="Importance of each 6-second step and each epoch across the 150-second window"
          >
            <div className="research-context-epochs" aria-hidden="true">
              {[-2, -1, 0, 1, 2].map((epoch) => (
                <span key={epoch} className={epoch === 0 ? "is-scored" : undefined}>
                  {epoch === 0 ? "scored" : `${epoch > 0 ? "+" : "−"}${Math.abs(epoch)}`}
                </span>
              ))}
            </div>
            <div className="research-context-row-label">Measurements that change within an epoch · swap one six-second step</div>
            <div className="research-context-y-title">Drop in macro-F1 · mean of {data.time_importance.folds} test folds · shared vertical scale</div>
            <div className="research-context-scaled-row">
              <div className="research-context-y-ticks">
                <span>{formatMetric(timeMax)}</span>
                <span>{formatMetric(timeMax / 2)}</span>
                <span>0</span>
              </div>
              <div className="research-context-bars">
                {fine.map((item) => (
                  <i
                    key={item.offset_sec}
                    className={item.offset_sec >= 0 && item.offset_sec < 30 ? "is-scored" : undefined}
                    style={{ height: `${(positive(item.importance) / timeMax) * 100}%` }}
                    title={`${item.offset_sec} s: ${formatMetric(item.importance)}`}
                    onPointerEnter={() => setHoveredTime({ kind: "6-s step", offset: item.offset_sec, span: item.span_sec, importance: item.importance })}
                  />
                ))}
              </div>
            </div>
            <div className="research-context-row-label">
              Measurements calculated once per 30-second epoch · swap all five repeated six-second values together
            </div>
            <div className="research-context-scaled-row">
              <div className="research-context-y-ticks">
                <span>{formatMetric(timeMax)}</span>
                <span>{formatMetric(timeMax / 2)}</span>
                <span>0</span>
              </div>
              <div className="research-context-bars is-epochs">
                {coarse.map((item) => (
                  <i
                    key={item.offset_sec}
                    className={item.offset_sec === 0 ? "is-scored" : undefined}
                    style={{ height: `${(positive(item.importance) / timeMax) * 100}%` }}
                    title={`${item.offset_sec} s: ${formatMetric(item.importance)}`}
                    onPointerEnter={() => setHoveredTime({ kind: "30-s epoch", offset: item.offset_sec, span: item.span_sec, importance: item.importance })}
                  />
                ))}
              </div>
            </div>
            <div className="research-context-axis" aria-hidden="true">
              {[-60, -30, 0, 30, 60, 90].map((tick) => (
                <span key={tick} style={{ left: `${((tick + 60) / 150) * 100}%` }}>
                  {tick} s
                </span>
              ))}
            </div>
            <p className="lead-readout" aria-live="polite">
              {hoveredTime ? (
                <>
                  {hoveredTime.kind} ({hoveredTime.offset} to {hoveredTime.offset + hoveredTime.span} s): macro-F1 drop{" "}
                  <b>{formatMetric(hoveredTime.importance)}</b>
                </>
              ) : (
                <span className="lead-hint">Hover any bar above to read the exact macro-F1 drop at that window offset.</span>
              )}
            </p>
            <p className="research-context-note">
              Within the scored epoch, importance peaks {peak.offset_sec}–{peak.offset_sec + 6} s after its start. Outside it, the previous
              epoch&rsquo;s features count {coarseRatio.toFixed(1)}× as much as the next epoch&rsquo;s. This matches the scoring rules, which decide
              whether N2 and REM continue by looking at earlier epochs.
            </p>
          </div>
        ) : null}
      </div>
    </div>
  );
}

export function Generalisation() {
  const adopted = data.models.find((model) => model.id === "adopted") as ModelResult;
  const g = data.generalisation;
  if (!g) return null;
  const held = g.holdout.find((row) => row.config === "g20");
  const rows = [
    ...(held
      ? [
          {
            label: `Model C · Held-out test, ${g.holdout_subjects} Boston patients`,
            note: "Model C scored twice (Look #1 & #2); Model D evaluated in Look #3 (§3.3)",
            row: { ...held, stage_f1: intervals.holdout.stage_f1 as Record<string, number> },
            ci: intervals.holdout.macro_f1_ci
          }
        ]
      : []),
    ...(isrucStatus.final_complete
      ? isrucStatus.results
          .filter((row) => ["transfer_cinc_to_isruc_scorer1", "transfer_cinc_to_isruc_scorer2"].includes(row.evaluation))
          .map((row) => ({
            label: `Coimbra, ${row.n_subjects} nights · scorer ${row.evaluation.endsWith("scorer1") ? "1" : "2"}`,
            note: "Model C · all 20 inputs",
            row,
            ci: intervals.coimbra[row.evaluation.endsWith("scorer1") ? "model_vs_scorer1" : "model_vs_scorer2"].macro_f1_ci
          }))
      : []),
    {
      label: `Cross-validation, ${data.source.subjects} Boston patients`,
      note: "Development folds used for model selection",
      row: { macro_f1: adopted.metrics.macro_f1, kappa: adopted.metrics.cohen_kappa, stage_f1: adopted.stage_f1 },
      ci: intervals.cv.adopted.macro_f1_ci
    }
  ];
  return (
    <div className="research-ablations">
      <div className="research-evidence-head">
        <h4 id="validation-scores">Overall and per-stage scores</h4>
        <p>
          Macro-F1 averages the F1 scores for the five stages, giving each stage equal weight. F1 combines how many expert labels the model finds with how often its predictions for that stage are correct.
        </p>
      </div>
      <p className="research-validation-scroll-hint">Scroll across for all five sleep stages.</p>
      <div className="research-validation-scroll" role="region" tabIndex={0} aria-labelledby="validation-scores">
        <table className="research-validation-table research-validation-table-stages" aria-labelledby="validation-scores">
          <thead>
            <tr>
              <th scope="col">Test</th>
              <th scope="col">Macro-F1</th>
              <th scope="col">κ</th>
              {data.source.stages.map((stage) => (
                <th scope="col" key={stage} className="swatch" style={{ "--swatch": stageColors[stage] } as React.CSSProperties}>
                  {stage}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(({ label, note, row, ci }) => (
              <tr key={label}>
                <th scope="row">
                  {label}
                  <small className="research-row-note">{note}</small>
                </th>
                <td>
                  {formatMetric(row.macro_f1)}
                  <small className="research-row-note">{formatInterval(ci)}</small>
                </td>
                <td>{row.kappa.toFixed(2)}</td>
                {data.source.stages.map((stage) => (
                  <td key={stage}>{row.stage_f1 ? (row.stage_f1 as Record<string, number>)[stage].toFixed(2) : "—"}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="research-ablation-note">
        Small figures under each macro-F1 are 95% patient-bootstrap intervals ({intervals.reps.toLocaleString()} resamples of patients; within each test fold for cross-validation).
      </p>
    </div>
  );
}

export function IsrucResults() {
  if (!isrucStatus.final_complete) return null;
  const results = isrucStatus.results as Array<{ evaluation: string; macro_f1: number; accuracy: number; kappa: number; n_subjects: number }>;
  const rows = [
    { id: "transfer_cinc_to_isruc_scorer1", label: "Model vs scorer 1" },
    { id: "transfer_cinc_to_isruc_scorer2", label: "Model vs scorer 2" },
    { id: "isruc_scorer1_vs_scorer2", label: "Scorer 1 vs scorer 2" },
    { id: "transfer_on_epochs_where_scorers_agree", label: "Model on scorer-agreement epochs" }
  ].flatMap((item) => {
    const result = results.find((row) => row.evaluation === item.id);
    return result ? [{ ...result, label: item.label }] : [];
  });
  if (rows.length !== 4) return null;
  return (
    <div className="research-ablations">
      <div className="research-evidence-head">
        <h4 id="coimbra-scorers">Coimbra scorer comparisons</h4>
        <p>
          The adopted 20-signal model trained on Boston recordings, applied to {isrucStatus.evaluated} Coimbra nights. Night 8 lacked frontal EEG and was excluded.
        </p>
      </div>
      <p className="research-validation-scroll-hint research-validation-scroll-hint-coimbra">Scroll across for all metrics.</p>
      <div className="research-validation-scroll" role="region" tabIndex={0} aria-labelledby="coimbra-scorers">
        <table className="research-validation-table" aria-labelledby="coimbra-scorers">
          <thead>
            <tr>
              {["Comparison", "Nights", "Macro-F1", "Accuracy", "κ"].map((label) => (
                <th scope="col" key={label}>
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.evaluation}>
                <th scope="row">{row.label}</th>
                <td>{row.n_subjects}</td>
                <td>{formatMetric(row.macro_f1)}</td>
                <td>{(100 * row.accuracy).toFixed(1)}%</td>
                <td>{row.kappa.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="research-ablation-note">
        The agreement-only row scores only epochs where both experts agreed, so it reflects an easier subset of the night.
      </p>
    </div>
  );
}

export function ExternalTestingDetails() {
  return (
    <>
      <Generalisation />
      <IsrucResults />
    </>
  );
}

export function ComparisonCompact() {
  const models = publishedModels();
  const fold = (value?: number) => (value === undefined ? "—" : formatMetric(value));
  return (
    <div className="research-comparison">
      <table className="research-performance-comparison" role="table" aria-label="Comparison with published models">
        <thead role="rowgroup">
          <tr role="row">
            {["Model", "Macro-F1", "Model size", "Training data", "Evaluation"].map((label) => (
              <th role="columnheader" scope="col" key={label}>
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody role="rowgroup">
          {models
            .filter((model) => model.macroF1 !== undefined && !model.name.startsWith("C "))
            .map((model) => (
              <tr role="row" key={model.name} className={model.ours ? "is-adopted" : undefined}>
                <th role="rowheader" scope="row">
                  <strong>{model.name.replace(", this page", "").replace(" (this page)", "")}</strong>
                  {model.cite ? (
                    <>
                      {" "}
                      <Citation number={model.cite} />
                    </>
                  ) : null}
                  <small className="research-row-note">{model.kind}</small>
                </th>
                <td role="cell">
                  <span className="research-comparison-label" aria-hidden="true">
                    Macro-F1
                  </span>
                  {fold(model.macroF1)}
                </td>
                <td role="cell">
                  <span className="research-comparison-label" aria-hidden="true">
                    Model size
                  </span>
                  {model.learnedShort}
                  {model.name.startsWith("C ") ? " tree values" : model.name.startsWith("D ") ? " weights" : ""}
                </td>
                <td role="cell">
                  <span className="research-comparison-label" aria-hidden="true">
                    Training data
                  </span>
                  {model.trainShort}
                </td>
                <td role="cell">
                  <span className="research-comparison-label" aria-hidden="true">
                    Evaluation
                  </span>
                  {model.testShort}
                </td>
              </tr>
            ))}
        </tbody>
      </table>
      <p className="research-protocol-source">
        Model D is scored across the 320-patient 5-fold Boston development split (with 474 fit-only Boston nights added inside training folds); external Boston Holdout ({fmt("ext.minimal82.holdout.calibrated.macro_f1", 3)}) and Coimbra ({fmt("ext.minimal82.isruc.calibrated_scorer1.macro_f1", 3)}) results follow in &sect;3.3. U-Sleep and U-Time report results on different Boston subsets; U-Sleep&rsquo;s study spans 15,660 participants across 16 clinical studies. <Citation number={26} />
      </p>
    </div>
  );
}

function TradeoffChart({ models, sweep: showSweep = false }: { models: PublishedModel[]; sweep?: boolean }) {
  const points = models.filter((model) => model.learned && model.macroF1);
  const sweep = showSweep ? [...data.complexity].sort((a, b) => a.learned_values - b.learned_values) : [];
  const xMin = 1;
  const xMax = 7;
  const yMin = showSweep ? 0.55 : 0.7;
  const yMax = showSweep ? 0.8 : 0.82;
  const x = (value: number) => 60 + ((Math.log10(value) - xMin) / (xMax - xMin)) * 500;
  const y = (value: number) => 200 - ((value - yMin) / (yMax - yMin)) * 180;
  const path = sweep.map((row, index) => `${index === 0 ? "M" : "L"}${x(row.learned_values).toFixed(1)} ${y(row.macro_f1).toFixed(1)}`).join(" ");
  return (
    <figure className="research-tradeoff">
      <svg viewBox="0 0 600 240" role="img" aria-label="Macro-F1 against number of learned values, log scale">
        {(showSweep ? [0.6, 0.65, 0.7, 0.75, 0.8] : [0.7, 0.75, 0.8]).map((tick) => (
          <g key={tick}>
            <line className="grid" x1="60" x2="560" y1={y(tick)} y2={y(tick)} />
            <text className="tick" x="52" y={y(tick) + 4} textAnchor="end">
              {tick.toFixed(2)}
            </text>
          </g>
        ))}
        {[1, 2, 3, 4, 5, 6, 7].map((power) => (
          <text key={power} className="tick" x={x(10 ** power)} y="222" textAnchor="middle">
            {["", "10", "100", "1k", "10k", "100k", "1M", "10M"][power]}
          </text>
        ))}
        <text className="axis" x="310" y="238" textAnchor="middle">
          learned values (log scale)
        </text>
        <text className="axis" x="14" y="110" textAnchor="middle" transform="rotate(-90 14 110)">
          macro-F1, Boston
        </text>
        {sweep.length ? <path className="sweep" d={path} /> : null}
        {sweep
          .filter((row) => sweepLabels[row.learner])
          .map((row) => (
            <g key={row.learner} className="is-sweep">
              <circle cx={x(row.learned_values)} cy={y(row.macro_f1)} r={3.5} />
              <text
                className="label small"
                x={x(row.learned_values) + sweepLabels[row.learner].dx}
                y={y(row.macro_f1) + sweepLabels[row.learner].dy}
                textAnchor={sweepLabels[row.learner].dx < 0 ? "end" : "start"}
              >
                {sweepLabels[row.learner].text}
              </text>
            </g>
          ))}
        {points.map((model) => {
          const px = x(model.learned as number);
          const py = y(model.macroF1 as number);
          const rightSide = px > 350;
          return (
            <g key={model.name} className={model.ours ? "is-ours" : undefined}>
              <circle cx={px} cy={py} r={model.ours ? 6 : 5} />
              {showSweep && model.name.startsWith("C ") ? null : (
                <text
                  className="label"
                  x={px + (rightSide ? -9 : 9)}
                  y={py + (model.name.includes("Attention") ? 14 : -8)}
                  textAnchor={rightSide ? "end" : "start"}
                >
                  {model.name.replace(", this page", "").replace(" (this page)", "")}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <figcaption>
        {showSweep
          ? "The thin line is the final model's inputs with simpler learners, down to one readable tree."
          : "Learned values: split thresholds and leaf values for the trees, weights for the networks."}
      </figcaption>
    </figure>
  );
}

export function PublishedModelSize() {
  const models = publishedModels();
  const cModel = models.find((m) => m.name.startsWith("C "));
  const dModel = models.find((m) => m.name.startsWith("D "));
  const cnnModel = data.models.find((m) => m.id === "cnn");
  return (
    <div className="research-comparison">
      <TradeoffChart models={models} />
      <p className="research-protocol-source">
        Model D (Concept-BiGRU) uses {dModel?.learned?.toLocaleString()} weights per fold, Model C uses {cModel?.learned?.toLocaleString()} tree values (each a split threshold or leaf value across {data.model_size.trees} trees), and the legacy Attention CNN uses {data.model_size.cnn_weights.toLocaleString()} weights (macro-F1 {cnnModel ? formatMetric(cnnModel.metrics.macro_f1) : "—"}). Published networks and the expert-agreement reference used different test samples.
      </p>
    </div>
  );
}

export function ModelPath() {
  const byId = (id: string) => data.models.find((model) => model.id === id);
  const held = data.generalisation?.holdout.find((row) => row.config === "nocardio27" && row.look === 1);
  const prune = data.reduction?.find((row) => row.config === "prune6");
  const steps = [
    { name: "B · Context", signals: 33, f1: byId("window-all")?.metrics.macro_f1, note: "33 measurements in context, heart and breathing included." },
    { name: "Features in context", signals: 19, f1: byId("window-features")?.metrics.macro_f1, note: "The 19 features alone in context: most of the gain over A is context itself." },
    {
      name: "27 signals",
      signals: 27,
      f1: byId("window")?.metrics.macro_f1,
      note: `Heart and breathing removed (${fmt("delta.cardio_with_eeg", 3, true)} with them).${held ? ` Scored at held-out look 1: ${formatMetric(held.macro_f1)}.` : ""}`
    },
    { name: "21 signals", signals: 21, f1: prune?.macro_f1, note: "The six least-used inputs removed, no loss." },
    {
      name: "C · 20 signals",
      signals: 20,
      f1: byId("adopted")?.metrics.macro_f1,
      note: `Spindle bursts in for 1/f offset and gamma residual (${fmt("delta.swap_spindle", 4)} macro-F1 vs 21 signals): trades a small drop for an explicit, validated marker.`
    }
  ];
  return (
    <div className="research-ablations">
      <p className="research-evidence-lede">
        The main page shows four models. These are the steps between B and C, each tested on the same folds, and the swap rounds that tried more direct markers. A swap had to stay within {fmt("protocol.swap_tolerance")} macro-F1 of the 21-signal model overall and in every fold, and every new feature was first checked against the stage it targets. <Exp ids={[28, 33, 34, 35, 36]} />
      </p>
      <div className="research-ablation-table" role="table" aria-label="Steps from B to the adopted model">
        <div className="research-ablation-row research-noeeg-row is-head" role="row">
          <span role="columnheader">Step</span>
          <span role="columnheader">Signals</span>
          <span role="columnheader">Macro-F1</span>
        </div>
        {steps.map((step) => (
          <div className="research-ablation-row research-noeeg-row" role="row" key={step.name}>
            <span role="cell">
              <strong>{step.name}</strong>
              <small className="research-row-note">{step.note}</small>
            </span>
            <span role="cell" className="num">
              {step.signals}
            </span>
            <span role="cell" className="num">
              {step.f1 !== undefined ? formatMetric(step.f1) : "—"}
            </span>
          </div>
        ))}
      </div>
      {data.reduction?.length ? (
        <>
          <h5 className="research-agreement-title">Swap rounds</h5>
          <div className="research-ablation-table" role="table" aria-label="Feature reduction and swap rounds">
            <div className="research-ablation-row research-noeeg-row is-head" role="row">
              <span role="columnheader">Variant</span>
              <span role="columnheader">Signals</span>
              <span role="columnheader">Macro-F1</span>
            </div>
            {data.reduction
              .filter((row) => reductionLabels[row.config])
              .map((row) => (
                <div className={`research-ablation-row research-noeeg-row${row.config === "swapS2" ? " is-adopted" : ""}`} role="row" key={row.config}>
                  <span role="cell">{reductionLabels[row.config]}</span>
                  <span role="cell" className="num">
                    {row.signals}
                  </span>
                  <span role="cell" className="num">
                    {formatMetric(row.macro_f1)}
                  </span>
                </div>
              ))}
          </div>
          <p className="research-ablation-note">
            Rejected candidate markers, checked before any model used them: REM eye bursts (fired mostly on wake blinks), K-complex events and slow eye movements (did not separate their stage as well as the inputs they would replace). Sawtooth theta and alpha dropout passed but the model barely used them.
          </p>
        </>
      ) : null}
    </div>
  );
}

export function ModelSimplification() {
  const rows = data.complexity;
  if (!rows.length) return null;
  const by = (id: string) => rows.find((row) => row.learner === id);
  const full = by("hgb_l31_i100");
  const stumps = by("hgb_l2_i200");
  const d4 = by("tree_d4");
  return (
    <div className="research-simplify">
      <p className="research-comparison-lede">
        The same {(data.source.tracks + data.source.features) * 25} inputs with simpler learners, on the same folds; learned values are the mean over the five fold models. Accuracy declines slowly until the model is reduced to a single tree. Boosting with one split per tree gives an additive model, in which each input&rsquo;s effect is a single curve; it scores {formatMetric(stumps?.macro_f1 ?? NaN)} with {stumps?.learned_values.toLocaleString()} parameters. A single depth-4 tree can be read in full but scores {formatMetric(d4?.macro_f1 ?? NaN)}.
      </p>
      <div className="research-ablation-table" role="table" aria-label="The window model with simpler learners">
        <div className="research-ablation-row research-simplify-row is-head" role="row">
          <span role="columnheader">Learner</span>
          <span role="columnheader">Learned values</span>
          <span role="columnheader">Macro-F1</span>
          <span role="columnheader">κ</span>
          <span role="columnheader">N1 F1</span>
        </div>
        {rows.map((row) => (
          <div key={row.learner} className={`research-ablation-row research-simplify-row${row.learner === "hgb_l31_i100" ? " is-adopted" : ""}`} role="row">
            <span role="cell">{row.label}</span>
            <span role="cell" className="num">
              {row.learned_values.toLocaleString()}
            </span>
            <span role="cell" className="num">
              {formatMetric(row.macro_f1)}
            </span>
            <span role="cell" className="num">
              {row.kappa.toFixed(2)}
            </span>
            <span role="cell" className="num">
              {row.stage_f1.N1.toFixed(2)}
            </span>
          </div>
        ))}
      </div>
      <TradeoffChart models={publishedModels().filter((model) => model.name.startsWith("C "))} sweep />
      <div className="research-readable-tree">
        <h5>The depth-4 tree, read aloud (fold 0, top three levels)</h5>
        <ol>
          <li>
            <b>Is the beta share low 6–12 s into the epoch?</b> Low beta means the cortex is not aroused.
            <ol>
              <li>
                <b>Yes → was occipital alpha low 12 s before?</b> Then N3 if the eye channel carries slow waves leaking from the frontal EEG, otherwise N2.
              </li>
              <li>
                <b>Yes, but alpha present →</b> mostly N2.
              </li>
            </ol>
          </li>
          <li>
            <b>No (beta higher) → is chin tone low 18–24 s in?</b>
            <ol>
              <li>
                <b>Yes →</b> REM: an active-looking EEG with an atonic chin.
              </li>
              <li>
                <b>No →</b> Wake or N1, split by the 1/f slope and theta share: a flatter spectrum and less theta mean Wake.
              </li>
            </ol>
          </li>
        </ol>
        <p>
          Each branch corresponds to a standard scoring cue: arousal (beta), posterior alpha, slow waves, REM atonia and the EEG slowing at sleep onset. The full model combines {data.model_size.trees} trees of this kind to go from {formatMetric(d4?.macro_f1 ?? NaN)} to {formatMetric(full?.macro_f1 ?? NaN)}.
        </p>
      </div>
    </div>
  );
}

export function SubjectVariability() {
  const [hoveredSubject, setHoveredSubject] = useState<{ subject: string; macro_f1: number; ahi_band: string } | null>(null);
  const subjects = [...data.diagnostics.subject_metrics].sort((a, b) => a.macro_f1 - b.macro_f1);
  const min = subjects[0];
  const max = subjects[subjects.length - 1];
  return (
    <div className="research-subjects">
      <div>
        <h3>Performance by patient (Model C baseline archive)</h3>
        <p>Each bar is one of the 320 Boston development patients scored in Model C&rsquo;s test folds (median {fmt("lead.cv.median")}), ordered from lowest to highest macro-F1.</p>
      </div>
      <div className="research-subject-plot">
        <div className="research-subject-axis" aria-hidden="true">
          <span className="research-subject-axis-title">Macro-F1</span>
          {[1, 0.75, 0.5, 0.25, 0].map((tick) => (
            <span key={tick} style={{ bottom: `${tick * 100}%` }}>
              {tick.toFixed(2)}
            </span>
          ))}
        </div>
        <div
          className="research-subject-strip"
          role="img"
          onPointerLeave={() => setHoveredSubject(null)}
          aria-label={`Subject macro-F1 from ${min.macro_f1.toFixed(3)} to ${max.macro_f1.toFixed(3)}`}
        >
          {subjects.map((subject) => (
            <i
              key={subject.subject}
              style={{ height: `${Math.max(subject.macro_f1 * 100, 1)}%` }}
              title={`${subject.subject}: ${subject.macro_f1.toFixed(3)}`}
              onPointerEnter={() => setHoveredSubject({ subject: subject.subject, macro_f1: subject.macro_f1, ahi_band: subject.ahi_band })}
            />
          ))}
        </div>
      </div>
      <div className="research-subject-range">
        <span>
          lowest <strong>{min.macro_f1.toFixed(3)}</strong> ({min.subject})
        </span>
        <span>
          {hoveredSubject ? (
            <>
              <strong>{hoveredSubject.subject}</strong> · {hoveredSubject.macro_f1.toFixed(3)} ({hoveredSubject.ahi_band})
            </>
          ) : (
            <>
              median <strong>{fmt("lead.cv.median")}</strong> · {subjects.length} patients
            </>
          )}
        </span>
        <span>
          highest <strong>{max.macro_f1.toFixed(3)}</strong> ({max.subject})
        </span>
      </div>
    </div>
  );
}

const ABLATION_NOTE =
  "Per-stage columns are F1 without the decoder. Four-class κ merges N1 and N2 into light sleep (the better of with and without the decoder), as heart-rate staging studies report it. \"Unbalanced\" trains with equal class weights, which favours accuracy and κ over N1.";

export function SignalAblations() {
  const configs = [...new Set(data.signal_ablations.map((row) => row.config))];
  const rows = configs.flatMap((config) => {
    const epoch = data.signal_ablations.find((row) => row.config === config && row.decoder === "epoch-only");
    const hmm = data.signal_ablations.find((row) => row.config === config && row.decoder === "hmm");
    return epoch && hmm
      ? [{ config, label: epoch.label, epoch: epoch.macro_f1, hmm: hmm.macro_f1, kappa4: epoch.kappa4, kappa4Hmm: hmm.kappa4, stages: epoch.stage_f1 }]
      : [];
  });
  const full = rows.find((row) => row.config === "window33");
  const brain = rows.find((row) => row.config === "nocardio27");
  const cardio = rows.filter((row) => row.config.startsWith("cardio"));
  if (!full || !brain || !cardio.length) return null;
  const bestCardio = cardio.reduce((a, b) => (Math.max(b.epoch, b.hmm) > Math.max(a.epoch, a.hmm) ? b : a));
  return (
    <div className="research-ablations">
      <div className="research-evidence-head">
        <h3>Heart and breathing, with and without the EEG</h3>
        <p>
          The same model retrained with groups of signals removed or used alone, on the same folds. Removing heart and breathing changes macro-F1 only from {formatMetric(full.epoch)} to {formatMetric(brain.epoch)}; with EEG, EOG and chin EMG present they add virtually nothing, so Model C leaves them out. On their own they still carry stage information, especially with additional heart features and a longer window: the best heart-and-breathing tree model reaches {formatMetric(Math.max(bestCardio.epoch, bestCardio.hmm))} macro-F1 and a four-class κ of {Math.max(bestCardio.kappa4, bestCardio.kappa4Hmm).toFixed(2)}. Published non-EEG models using whole-night networks report four-class κ of {fmt("lit.cardio_kappa4.low")} with ECG HRV alone on 292 subjects <Citation number={31} /> (compare <Citation number={12} />) up to {fmt("lit.cardio_kappa4.high")} on thousands of nights when adding respiratory effort and airflow <Citation number={32} />. Here, adding a whole-night hidden Markov decoder <Citation number={25} /> helps far more without EEG than on the full model.
        </p>
      </div>
      <div className="research-ablation-table" role="table" aria-label="Window model with signal groups removed or isolated">
        <div className="research-ablation-row is-head" role="row">
          <span role="columnheader">Signals</span>
          <span role="columnheader">Macro-F1</span>
          <span role="columnheader">+ decoder</span>
          <span role="columnheader">4-class κ</span>
          {data.source.stages.map((stage) => (
            <span role="columnheader" key={stage} className="swatch" style={{ "--swatch": stageColors[stage] } as React.CSSProperties}>
              {stage}
            </span>
          ))}
        </div>
        {rows.map((row) => (
          <div className="research-ablation-row" role="row" key={row.config}>
            <span role="cell">{row.label}</span>
            <span role="cell" className="num">
              {formatMetric(row.epoch)}
            </span>
            <span role="cell" className="num">
              {formatMetric(row.hmm)}
            </span>
            <span role="cell" className="num">
              {Math.max(row.kappa4, row.kappa4Hmm).toFixed(2)}
            </span>
            {data.source.stages.map((stage) => (
              <span role="cell" className="num" key={stage}>
                {row.stages[stage].toFixed(2)}
              </span>
            ))}
          </div>
        ))}
      </div>
      <p className="research-ablation-note">{ABLATION_NOTE}</p>
      <h5 className="research-agreement-title">
        Heart and breathing without the EEG <Exp ids={[31]} />
      </h5>
      <p className="research-evidence-lede">
        Heart rate and breathing change slowly with sleep stage, over minutes, so a model without EEG needs a much longer window than 2.5 minutes. Whole-night networks with breath-by-breath respiration and blood oxygen added raise the heart-and-breathing model from {fmt("cv.cardiofull_c10.hmm.macro_f1")} to {fmt("noeeg.ensemble.macro_f1")} macro-F1 (four-class κ from {fmt("cv.cardiofull_c10.hmm.kappa4", 2)} to {fmt("noeeg.ensemble.kappa4", 2)}). The table shows what each sensor achieves on its own, which is the relevant question for a wearable device.
      </p>
      <div className="research-ablation-table" role="table" aria-label="Heart and breathing models without the EEG">
        <div className="research-ablation-row research-noeeg-row is-head" role="row">
          <span role="columnheader">Sensors and model</span>
          <span role="columnheader">Macro-F1</span>
          <span role="columnheader">4-class κ</span>
        </div>
        {[
          ["ECG + belts, trees over 10 minutes (row above)", "cv.cardiofull_c10.hmm"],
          ["ECG + belts + oximeter, whole-night networks (4 GRU + 4 TCN seeds, averaged post hoc)", "noeeg.ensemble"],
          ["ECG + oximeter only, whole-night GRU", "noeeg.gru_ecgonly_spo2_pos.epoch"],
          ["Effort belts only, whole-night GRU", "noeeg.gru_beltonly_pos.epoch"],
          ["ECG only, whole-night TCN", "noeeg.tcn_ecgonly_pos.epoch"]
        ]
          .map(([label, id]) => [label, fmt(`${id}.macro_f1`), fmt(`${id}.kappa4`, 2)])
          .map(([label, f1, k]) => (
            <div className="research-ablation-row research-noeeg-row" role="row" key={label}>
              <span role="cell">{label}</span>
              <span role="cell" className="num">
                {f1}
              </span>
              <span role="cell" className="num">
                {k}
              </span>
            </div>
          ))}
      </div>
      <p className="research-ablation-note">
        The oximeter row averages the probabilities of eight networks after the fact; the best single architecture (a GRU, two seeds) scores {fmt("noeeg.gru_base_fine_raw_pos.epoch.macro_f1")}. Published heart-rate and cardiorespiratory models reach four-class κ {fmt("lit.cardio_kappa4.low")}–{fmt("lit.cardio_kappa4.high")} on 292 to 6,312 recordings, compared with about 255 per fold here. Oxygen saturation and breath-by-breath respiration account for most of the gain; additional heart-rate variability measures add nothing once heart rate is included. Without EEG, N1 F1 stays at {fmt("noeeg.ensemble.f1_N1", 2)}.
      </p>
    </div>
  );
}

export function BlackBoxAgreement() {
  const saliency = saliencyData as unknown as SaliencyData;
  const stages = data.source.stages;
  const systems = [
    { id: "eeg", label: "Brain (EEG)", color: familyColors.spectrum, families: ["spectrum", "spindles", "topography"] },
    { id: "eog", label: "Eyes (EOG)", color: familyColors["eye-movement"], families: ["eye-movement"] },
    { id: "emg", label: "Chin (EMG)", color: familyColors["muscle-tone"], families: ["muscle-tone"] },
    { id: "cardio", label: "Heart and breathing", color: familyColors.heart, families: ["heart", "breathing"] }
  ];
  const trees = stages.map((stage) => {
    const drops = systems.map((system) =>
      system.families.reduce((sum, id) => sum + Math.max(data.feature_importance.find((item) => item.family_id === id)?.stage_drops[stage] ?? 0, 0), 0)
    );
    const total = drops.reduce((a, b) => a + b, 0) || 1;
    return drops.map((value) => value / total);
  });
  const cnn = stages.map((stage) => {
    const row = saliency.by_stage.find((item) => item.stage === stage);
    return systems.map((system) =>
      saliency.signals.reduce((sum, name, index) => sum + (system.families.includes(signalInfo[name]?.family) ? row?.signal_share[index] ?? 0 : 0), 0)
    );
  });
  const topOf = (row: number[]) => row.indexOf(Math.max(...row));
  const agree = stages.filter((_, k) => topOf(trees[k]) === topOf(cnn[k])).length;
  const pct = (value: number) => Math.round(value * 100);
  return (
    <div className="research-agreement">
      <p className="research-evidence-lede">
        As an exploratory check against earlier sequence-network baselines, <Citation number={4} /> <Citation number={13} /> this grid compares which body systems Model C (via input shuffling) and the legacy 2.5-min Attention CNN (via input gradients) rely on. Because the models use different windows and attribution methods are not directly equivalent, treat this as a rough comparison of broad signal share. <Exp ids={[20]} />
      </p>
      <div className="research-agreement-table is-paired" style={{ gridTemplateColumns: `minmax(0, 1.5fr) repeat(${stages.length}, minmax(0, 1fr))` }}>
        <span className="research-agreement-key">
          % share · <b>C</b> / CNN
        </span>
        {stages.map((stage) => (
          <b key={stage} className="swatch" style={{ "--swatch": stageColors[stage] } as React.CSSProperties}>
            {stage}
          </b>
        ))}
        {systems.map((system, f) => (
          <div key={system.id} className="research-agreement-row">
            <span>
              <u style={{ background: system.color }} />
              {system.label}
            </span>
            {stages.map((stage, k) => (
              <em key={stage} style={{ "--cell-alpha": Math.min(trees[k][f] * 2, 1).toFixed(3) } as React.CSSProperties}>
                <b>{pct(trees[k][f])}</b> / {pct(cnn[k][f])}
              </em>
            ))}
          </div>
        ))}
      </div>
      <p className="research-agreement-takeaway">
        Both models rely most on the EEG in {agree} of 5 stages, and both shift toward eye movements and chin tone in REM (C{" "}
        {formatPercent(trees[4][1] + trees[4][2])}, CNN {formatPercent(cnn[4][1] + cnn[4][2])}). The CNN assigns{" "}
        {formatPercent(Math.min(...cnn.map((row) => row[3])))}–{formatPercent(Math.max(...cnn.map((row) => row[3])))} to heart and breathing, in part
        because gradients spread some credit to every input; C does not use these signals and scores the same within uncertainty (CNN − C{" "}
        {intervals.cv_paired_vs_adopted.cnn.minus_adopted.toFixed(3).replace("-", "−")}, 95% interval{" "}
        {intervals.cv_paired_vs_adopted.cnn.ci[0].toFixed(3).replace("-", "−")} to {intervals.cv_paired_vs_adopted.cnn.ci[1].toFixed(3).replace("-", "−")}
        ). Shading follows C; each column sums to 100% for each model.
      </p>
    </div>
  );
}

export function SleepStructure() {
  const matrix = data.sleep_structure.transition_matrix;
  const stages = data.source.stages;
  const bouts = data.sleep_structure.bouts;
  return (
    <div className="research-sleep-structure">
      <div className="research-sleep-structure-copy">
        <p>
          Across all {data.source.subjects} expert hypnograms, stages tend to persist from one epoch to the next, wake usually passes through N1 before deeper sleep, and N3 is entered almost only from N2. So an epoch is easier to stage when its neighbours are visible, and the model reads 2.5 minutes around each one.
        </p>
        <div className="research-bout-list">
          <strong>Typical bout length</strong>
          {bouts.map((bout) => (
            <span key={bout.stage}>
              <b className="swatch" style={{ "--swatch": stageColors[bout.stage] } as React.CSSProperties}>
                {bout.stage}
              </b>
              median {bout.median_epochs * 0.5} min · {formatPercent(bout.share_single_epoch)} last one epoch
            </span>
          ))}
        </div>
      </div>
      <div className="research-confusion research-confusion-static research-transition-matrix">
        <span className="research-confusion-axis">Next epoch →</span>
        <div className="research-confusion-head">
          <span />
          {stages.map((stage) => (
            <span key={stage}>{stage}</span>
          ))}
        </div>
        {matrix.map((values, rowIndex) => (
          <div className="research-confusion-row" key={stages[rowIndex]}>
            <strong className="swatch" style={{ "--swatch": stageColors[stages[rowIndex]] } as React.CSSProperties}>
              {stages[rowIndex]}
            </strong>
            {values.map((value, columnIndex) => (
              <span
                key={columnIndex}
                className={columnIndex === rowIndex ? "is-diagonal" : undefined}
                style={{ "--cell-alpha": Math.max(columnIndex === rowIndex ? value : Math.min(value * 4, 1), 0.035).toFixed(3) } as React.CSSProperties}
              >
                {value >= 0.995 ? ">99%" : value < 0.001 ? "<0.1%" : value < 0.1 ? `${(value * 100).toFixed(1)}%` : formatPercent(value)}
              </span>
            ))}
          </div>
        ))}
        <p className="research-confusion-caption">
          Probability of each stage in the next epoch, given the current stage. Rows sum to 100%; off-diagonal shading is boosted 4×.
        </p>
      </div>
    </div>
  );
}
