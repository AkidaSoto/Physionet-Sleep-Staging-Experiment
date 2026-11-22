"use client";

import Link from "next/link";
import conceptData from "../../content/generated/concept-showcase.json";
import isrucStatus from "../../content/generated/isruc-status.json";
import { sensorColors, stageColors } from "../../design/palette";
import { references, stagePrimerRows, uSleepWeights } from "../../content/staging-content";
import { stagingDashboardData, type ModelResult } from "../../content/staging-dashboard-data";
import { fmt, pct, r } from "../../lib/results";
import {
  ConceptAttributionSection,
  ConceptExternalDistFigure,
  ConceptHypnogramFigure,
  ConceptProgressionSection,
  ConformalUncertaintySection,
  SupplementaryExplorationSection,
} from "./ConceptShowcaseSections";
import { PhysiologyExplorer, SpindleComparison } from "./PhysiologyExplorer";
import {
  BlackBoxAgreement,
  ComparisonCompact,
  ConfusionExplorer,
  ExternalTestingDetails,
  ModelEvidence,
  ModelExplorer,
  ModelPath,
  ModelSimplification,
  PublishedModelSize,
  SignalAblations,
  SleepStructure,
  SubjectVariability
} from "./ResultsSections";
import {
  Citation,
  Evidence,
  Exp,
  SecRef,
  SectionHeading,
  SubHeading,
  formatMetric,
  formatPercent,
  useOpenOnHash
} from "../ui/ResearchPrimitives";

export { ExperimentLogPage } from "./ExperimentLogPage";

const data = stagingDashboardData;

function SleepStudyPrimer() {
  return (
    <div className="research-introduction">
      <div className="research-introduction-copy">
        <p>
          Sleep staging turns an overnight sleep study—a polysomnogram (PSG) recording brain waves (EEG), eye movements (EOG) and chin muscle tone (EMG)—into a night-long timeline called a hypnogram. Each 30-second slice (an <em>epoch</em>) receives one of five labels: wake, light sleep (N1 and N2), deep slow-wave sleep (N3) or rapid-eye-movement sleep (REM). Because clinicians read sleep efficiency, fragmentation and breathing events against that timeline, staging mistakes distort downstream metrics. <Citation number={2} />
        </p>
        <p>
          Manual scoring is slow and inherently noisy at stage boundaries. <Citation number={5} /> In the American Academy of Sleep Medicine (AASM) inter-scorer program, over 2,500 scorers agreed with the majority consensus on {pct("lit.aasm.agreement")} of epochs overall and {pct("lit.aasm.agreement_N1")} of N1 epochs across nine short excerpts. <Citation number={29} /> Pairwise agreement between two individual clinicians on full nights is slightly lower—around 82% in the Coimbra test cohort below.
        </p>
      </div>

      <div className="research-introduction-figure" role="img" aria-label="EEG, EOG, and chin EMG are scored in 30-second epochs to produce a whole-night hypnogram">
        <div className="research-introduction-signals">
          <span>
            <strong>EEG</strong>
            <small>brain activity</small>
          </span>
          <span>
            <strong>EOG</strong>
            <small>eye movement</small>
          </span>
          <span>
            <strong>EMG</strong>
            <small>chin tone</small>
          </span>
        </div>
        <i>recorded together through the night</i>
        <div className="research-introduction-flow">
          <span className="research-introduction-scored-epoch">
            <b>30-second scoring epoch</b>
            <small>one expert stage label</small>
          </span>
          <em>→</em>
          <span>
            <b>Hypnogram</b>
            <small>sleep architecture</small>
          </span>
        </div>
      </div>
    </div>
  );
}

function StagePrimer() {
  const stages = data.source.stages;
  return (
    <div className="research-stage-primer">
      <p className="research-stage-primer-lede">
        Scorers identify sleep stages from combinations of brain activity, eye movements and muscle tone. These patterns can overlap across stages, so scorers consider them together. Heart rate and breathing also change across stages.
      </p>
      <div className="research-stage-primer-scroll">
        <table>
          <thead>
            <tr>
              <th scope="col">System</th>
              {stages.map((stage) => (
                <th scope="col" key={stage} className="swatch" style={{ "--swatch": stageColors[stage] } as React.CSSProperties}>
                  {stage}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {stagePrimerRows.map((row) => (
              <tr key={row.system}>
                <th scope="row">
                  <strong>
                    <i className="research-sensor-dot" style={{ background: sensorColors[row.system] }} aria-hidden="true" />
                    {row.system}
                  </strong>
                  <small>{row.signal}</small>
                  <em className={row.scoring ? "is-scoring" : "is-supporting"}>{row.scoring ? "AASM scoring" : "Follows stage"}</em>
                  <span className="research-row-cites">
                    {row.cites.map((number) => (
                      <Citation key={number} number={number} />
                    ))}
                  </span>
                </th>
                {row.cells.map((cell, index) => (
                  <td key={stages[index]} data-stage={stages[index]}>
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function TestingProtocol() {
  return (
    <div className="research-model-section">
      <p className="research-takeaway" id="testing">
        The data serve three roles: early feature prototyping in Dublin, model selection and held-out testing in Boston, and external validation in Coimbra. Patient-grouped splits test generalization to unseen people, and Coimbra tests transfer to a separate hospital.
      </p>
      <p>
        Before each experiment, I recorded the model variants and the gain required for adoption, then checked new measurements against their intended sleep stage. All model selection stayed inside 5-fold patient-grouped cross-validation; the Boston holdout was scored three times (twice during Model C pruning and once in a locked final evaluation for Model D) and Coimbra once per adopted model. The <Link href="/history">experiment log</Link> preserves dead ends alongside successes, including early label, alignment and leakage fixes (<Exp ids={[7, 8, 10]} />) and the locked evaluation runs (<Exp ids={[30, 37, 38, 81]} />).
      </p>
    </div>
  );
}

function CompactStudyProtocol() {
  const total = data.diagnostics.per_stage.reduce((sum, stage) => sum + stage.support, 0);
  return (
    <div className="research-protocol">
      <div className="research-ablation-table" role="table" aria-label="Datasets and how each is used">
        <div className="research-ablation-row research-data-row is-head" role="row">
          <span role="columnheader">Dataset</span>
          <span role="columnheader">Recordings</span>
          <span role="columnheader">Role</span>
        </div>
        <div className="research-ablation-row research-data-row" role="row">
          <span role="cell">
            <strong>UCDDB</strong>
            <small className="research-row-note">
              St Vincent&rsquo;s University Hospital, Dublin <Citation number={3} />
            </small>
          </span>
          <span role="cell">
            25 nights<small className="research-row-note">suspected sleep apnea; older R&amp;K scoring</small>
          </span>
          <span role="cell">
            <b>Early development.</b> Used to build the initial features and staging models <Exp range={[1, 14]} />.
          </span>
        </div>
        <div className="research-ablation-row research-data-row is-adopted" role="row">
          <span role="cell">
            <strong>Boston · CinC 2018, development</strong>
            <small className="research-row-note">
              Massachusetts General Hospital, Boston <Citation number={24} />
            </small>
          </span>
          <span role="cell">
            {data.source.subjects} CV nights + 474 fit-only nights<small className="research-row-note">{data.source.epochs.toLocaleString()} CV epochs</small>
          </span>
          <span role="cell">
            <b>Development and cross-validation.</b> Fixed 5-fold patient splits on the 320 patients (<Exp ids={[15]} /> onward); 474 labeled recordings added strictly inside training folds from <Exp ids={[73]} /> onward.
          </span>
        </div>
        <div className="research-ablation-row research-data-row" role="row">
          <span role="cell">
            <strong>Boston · CinC 2018, held out</strong>
            <small className="research-row-note">Same hospital, set aside before modelling</small>
          </span>
          <span role="cell">
            {data.generalisation?.holdout_subjects || 200} nights<small className="research-row-note">more apnea: 59% moderate or severe</small>
          </span>
          <span role="cell">
            <b>Scored three times.</b> Initial Model C look <Exp ids={[30]} />, post-pruning Model C <Exp ids={[37]} />, and locked final Model D (#79) evaluation <Exp ids={[81]} />.
          </span>
        </div>
        <div className="research-ablation-row research-data-row" role="row">
          <span role="cell">
            <strong>Coimbra · ISRUC-Sleep</strong>
            <small className="research-row-note">
              Coimbra, Portugal; subgroup I; two scorers per night <Citation number={30} />
            </small>
          </span>
          <span role="cell">{isrucStatus.final_complete ? isrucStatus.evaluated : 99} nights</span>
          <span role="cell">
            <b>External test.</b> Scored once per adopted model (Model C <Exp ids={[38]} />, then Model D <Exp ids={[81]} />); predictions compared against both scorers.
          </span>
        </div>
      </div>
      <p className="research-dataset-line">
        <span>Stage distribution in the development set:</span>
      </p>
      <div className="research-protocol-balance">
        <div className="research-class-stack" role="img" aria-label="Distribution of labeled sleep stages">
          {data.diagnostics.per_stage.map((stage) => (
            <i
              key={stage.stage}
              style={{ width: `${(stage.support / total) * 100}%`, background: stageColors[stage.stage] }}
              title={`${stage.stage}: ${stage.support.toLocaleString()} epochs`}
            />
          ))}
        </div>
        <div className="research-class-legend">
          {data.diagnostics.per_stage.map((stage) => (
            <span key={stage.stage}>
              <i style={{ background: stageColors[stage.stage] }} />
              {stage.stage} <strong>{formatPercent(stage.support / total)}</strong>
            </span>
          ))}
        </div>
      </div>
      <p className="research-protocol-source">
        The development set includes adults aged 18–87 (median 53; 185 men, 135 women), selected to cover the range of apnea severity: 115 with AHI below 5, 69 mild, 96 moderate and 40 severe.
      </p>
    </div>
  );
}

function WindowModelFigure() {
  const nConcepts = conceptData.concept_attribution.concepts.length;
  return (
    <div className="research-window-model">
      <div
        className="research-window-figure"
        role="img"
        aria-label={`Model D Concept-BiGRU architecture: ${conceptData.input_dim} non-redundant 6-s and 30-s cues enter ${nConcepts} additive physiological branches, merge into a 1-layer past and future GRU, and produce Mondrian conformal prediction sets`}
      >
        <div className="research-window-rows">
          <span>
            <strong>{conceptData.input_dim} pruned physiological inputs</strong>
            <small>13 EEG, EOG &amp; chin EMG tracks every 6 s plus 16 epoch markers — isolated into {nConcepts} organ/band branches with zero cross-talk</small>
          </span>
        </div>
        <div className="research-introduction-flow research-window-flow">
          <span>
            <b>{nConcepts} additive stems</b>
            <small>slow_wave, n2_events, wake_n1, rem_tone → local log-odds</small>
          </span>
          <em>→</em>
          <span>
            <b>1-layer BiGRU gates</b>
            <small>past + future context ({conceptData.model_params.toLocaleString("en-US")} weights)</small>
          </span>
          <em>→</em>
          <span className="research-introduction-scored-epoch">
            <b>Stage + Conformal Set</b>
            <small>90%/95% stage-conditional guarantee</small>
          </span>
        </div>
      </div>
    </div>
  );
}

function ModelSection() {
  const models = [
    { name: "A · Single epoch", inputs: "19 epoch-averaged features", window: "30 s", purpose: "Baseline without neighbouring epochs (255/fold)", refs: [16, 17] },
    { name: "B · Context trees", inputs: "33 measurements every 6 s", window: "150 s", purpose: "Establish the gain from 2.5-min context (255/fold)", refs: [19, 23] },
    { name: "C · Compact trees", inputs: "20 signals over 150 s (364 of 500 inputs used; 14,100 splits)", window: "150 s", purpose: `Earlier ${data.model_size.trees}-tree baseline tested on holdout & Coimbra (255/fold)`, refs: [28, 33, 36] },
    { name: `D · Concept-BiGRU (#${conceptData.active_experiment})`, inputs: `29 features (${conceptData.input_dim} inputs: 13 six-second tracks × 5 + 16 epoch markers)`, window: "Whole night", purpose: "Adopted model: 4 additive stems + 64-unit BiGRUs + conformal (704/fold)", refs: [79] }
  ];
  const nConcepts = conceptData.concept_attribution.concepts.length;
  return (
    <div className="research-model-section">
      <p className="research-takeaway">
        Model D (Concept-BiGRU, #{conceptData.active_experiment}) isolates <em>four domain-grouped branch encoders</em> from <em>whole-night sequence gating</em>: {nConcepts} private feature branches sum into additive feature-only stage log-odds, then 64-unit causal and anti-causal GRUs adjust predictions via bounded per-stage gates.
      </p>
      <WindowModelFigure />
      <p>
        At each 30-second epoch, 13 six-second tracks (5 steps each) and 16 epoch/context markers enter four private branch MLPs (<code>slow_wave</code>, <code>n2_events</code>, <code>wake_n1</code>, and <code>rem_tone</code>; 16 units each) whose linear heads sum with a stage bias into feature-only local log-odds (with zero cross-branch interaction prior to summation). Their concatenated 64-unit embedding then enters 1-layer past and future GRUs (64 units per direction) fused by bounded per-stage gates (<code>logits = local + past_gate · past + future_gate · future</code>; {conceptData.model_params.toLocaleString("en-US")} weights per fold, averaged across 5 fold checkpoints on external cohorts). Without ECG or respiratory belts, out-of-fold Mondrian conformal quantiles map calibrated probabilities into stage-conditional prediction sets.
      </p>
      <div className="research-design-rationale">
        <h5>Architectural design choices</h5>
        <ul className="research-glance-takeaways">
          <li>
            <b>Grouped additive stems before sequence mixing:</b> Routing features through four domain-grouped stems <Citation number={1} /> exposes an exact stage-by-branch logit sum ({formatMetric(conceptData.concept_attribution.local_only_macro_f1)} feature-only macro-F1) before downstream recurrence mixes representations across the night.
          </li>
          <li>
            <b>Two-timescale decoupling:</b> Following sequence-to-sequence staging that separates epoch-level feature encoding from macro-architecture <Citation number={13} />, 64-unit directional GRUs model sleep-stage hysteresis in {Math.round(conceptData.model_params / 1000)}k weights per fold (~{Math.round(uSleepWeights / conceptData.model_params)}× fewer than U-Sleep&rsquo;s ~{(uSleepWeights / 1_000_000).toFixed(1)} million <Citation number={26} />), trading offline whole-night lookahead for temporal accuracy.
          </li>
          <li>
            <b>Tempered training + Mondrian uncertainty:</b> Because transitional N1 accounts for ~10% of epochs, pairing tempered inverse-frequency weights with label smoothing and exponential moving average (EMA) weights protects N1 recall without severe over-prediction, while stage-conditional Mondrian conformal quantiles prevent rare N1 boundaries from being under-covered.
          </li>
        </ul>
      </div>
      <div className="research-models-compared">
        <h5>Models compared (A → B → C → D)</h5>
        <div className="research-model-table-wrap">
          <table className="research-model-table">
            <thead>
              <tr>
                <th scope="col">Model</th>
                <th scope="col">Inputs</th>
                <th scope="col">Window</th>
                <th scope="col">Role</th>
              </tr>
            </thead>
            <tbody>
              {models.map((model) => (
                <tr key={model.name} className={model.name.startsWith("D ·") ? "is-adopted" : undefined}>
                  <th scope="row">{model.name}</th>
                  <td>{model.inputs}</td>
                  <td>{model.window}</td>
                  <td>
                    {model.purpose} <Exp ids={model.refs} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <Evidence summary="Training, fold-split math (255 vs. 230 + 474 = 704) &amp; normalization guardrails">
        <p>
          Each of the 5 folds on the 320 Boston development patients holds out 64 test nights, leaving ~255 non-test nights. Models A–C fit trees directly on all ~255 nights; Model D reserves ~25 nights per fold for inner early-stopping validation, combining the remaining 230 inner-training nights with 474 fit-only Boston nights (230 + 474 = 704 fitted nights/fold). Per-subject robust track scaling follows the prior pipeline, while feature standardization means and SDs are computed strictly on each fold&rsquo;s fitting pool. Exploratory neural/tree comparators (dense BiGRU #75, multi-scale HGB #76, ensemble #78, and cue-screened #80) remain archived in <SecRef to="supp-cardio" />.
        </p>
      </Evidence>
    </div>
  );
}

function StudySummary() {
  const adopted = data.models.find((model) => model.id === "adopted") as ModelResult;
  const activeF1 = conceptData.calibrated.mean_fold_macro_f1;
  const activeKappa = conceptData.calibrated.kappa;
  const activeParams = conceptData.model_params;
  const mon90 = conceptData.conformal.coverage_90.mondrian;
  return (
    <section className="research-section research-summary" id="summary">
      <SectionHeading title="Abstract" />
      <div className="research-orientation">
        <p>
          <b>Plain-English guide:</b> Overnight sleep studies divide an 8-hour recording of brain waves (EEG), eye movements (EOG), and chin muscle tone (EMG) into 30-second chunks (<em>epochs</em>) labeled Wake, light sleep (N1, N2), deep slow-wave sleep (N3), or REM. We score models with <b>Macro-F1</b> (0 to 1 scale weighting all five stages equally). Across 81 logged experiments, our models progressed from single-epoch decision trees (<b>Model A</b>) and 2.5-minute context trees (<b>Models B &amp; C</b>) to <b>Model D</b> (#79): a compact sequence network that adds up four separate physiological branch scores before smoothing across the whole night, and outputs <em>Mondrian conformal prediction sets</em> (returning 2–3 plausible stages on blurry transitions so rare stages like N1 keep a 90% coverage guarantee). All model choices stayed inside 320-patient Boston cross-validation (CV); reserved Boston holdout nights (200) and external Coimbra hospital nights (99, scored by two independent experts) were tested in a locked final run (Look #3).
        </p>
      </div>
      <div className="research-exec-strip" role="region" aria-label="Model D headline scorecard">
        <div className="research-kpi-card">
          <span>Boston 5-Fold CV (320 nights)</span>
          <strong>{formatMetric(activeF1)} Macro-F1</strong>
          <small>+{formatMetric(activeF1 - adopted.metrics.macro_f1)} vs. Model C · κ {activeKappa.toFixed(2)}</small>
        </div>
        <div className="research-kpi-card">
          <span>Boston Holdout (200 nights)</span>
          <strong>{fmt("ext.concept79.holdout.calibrated.macro_f1")} Macro-F1</strong>
          <small>{fmt("delta.concept79_holdout_vs_model_c", 3, true)} vs. Model C · locked test (Look #3)</small>
        </div>
        <div className="research-kpi-card">
          <span>Coimbra Hospital (99 nights)</span>
          <strong>{fmt("ext.concept79.isruc.calibrated_scorer1.macro_f1")} Macro-F1</strong>
          <small>vs. Scorer 1 (Scorer 2: {fmt("ext.concept79.isruc.calibrated_scorer2.macro_f1")} · human: {fmt("coimbra.isruc_scorer1_vs_scorer2.macro_f1")})</small>
        </div>
        <div className="research-kpi-card">
          <span>Model D Footprint</span>
          <strong>{Math.round(activeParams / 1000)}k weights / fold</strong>
          <small>29 features ({conceptData.input_dim} inputs: 13×5 + 16) · 4 stems</small>
        </div>
      </div>
      <div className="research-summary-copy research-abstract">
        <p>
          Sleep staging models typically trade off accuracy (raw-waveform U-Nets with millions of weights) against transparency (decision trees over flattened windows). <b>Model D (Concept-BiGRU, #{conceptData.active_experiment})</b> targets an interpretable middle ground on 29 distinct EEG, EOG, and chin EMG features ({conceptData.input_dim} inputs per epoch): four domain-grouped branch encoders sum into feature-only stage log-odds before 64-unit past and future GRUs apply whole-night sequence gates ({activeParams.toLocaleString("en-US")} weights per fold—~{Math.round(uSleepWeights / activeParams)}× fewer than U-Sleep <Citation number={26} />, though ~{(activeParams / data.model_size.nodes).toFixed(1)}× more learned values than Model C&rsquo;s {data.model_size.trees} trees and requiring offline whole-night lookahead). Combining 230 inner-training nights per fold with 474 fit-only Boston nights (704 fitted nights/fold vs. Model C&rsquo;s 255) reached <b>{formatMetric(activeF1)}</b> 5-fold Macro-F1 on the 320 development patients (+{formatMetric(activeF1 - adopted.metrics.macro_f1)} over Model C).
        </p>
        <p>
          In locked external Look #3 (<Exp ids={[81]} />, averaging all 5 fold checkpoints), Model D reached <b>{fmt("ext.concept79.holdout.calibrated.macro_f1", 4)}</b> Macro-F1 on 200 apnea-heavy Boston holdout patients ({fmt("delta.concept79_holdout_vs_model_c", 4, true)} over Model C, with a smaller CV-to-holdout drop) and <b>{fmt("ext.concept79.isruc.calibrated_scorer1.macro_f1", 4)} / {fmt("ext.concept79.isruc.calibrated_scorer2.macro_f1", 4)}</b> across 99 external Coimbra hospital nights—ahead of Model C against both scorers, though cross-hospital N3 shift compressed Boston&rsquo;s margin. At a 90% target, stage-conditional <b>Mondrian conformal sets</b> maintain {formatPercent(mon90.empirical_coverage)} coverage across every stage in Boston CV and {pct("ext.concept79.holdout.conformal.coverage_90.coverage")} overall on Boston holdouts ({pct("ext.concept79.holdout.conformal.coverage_95.coverage")} at 95%), while dipping on holdout N2/N3 and Coimbra N3 under cross-cohort shift.
        </p>
      </div>
    </section>
  );
}

function DiscussionExplorer() {
  const full = data.complexity.find((row) => row.learner === "hgb_l31_i100");
  const smaller = data.complexity.find((row) => row.learner === "hgb_l8_i100");
  const mon90 = conceptData.conformal.coverage_90.mondrian;
  const gates = conceptData.concept_attribution.stage_gates;
  const wakeGate = gates.find((g) => g.stage === "Wake");
  const n3Gate = gates.find((g) => g.stage === "N3");
  const attr = conceptData.concept_attribution;
  const adopted = data.models.find((model) => model.id === "adopted") as ModelResult;
  return (
    <div className="research-conclusion">
      <p>
        <b>1. Disentangling data scale vs. architecture (+{formatMetric(attr.full_macro_f1 - adopted.metrics.macro_f1)} over Model C):</b> In Experiment #73 (<Exp ids={[73]} />), expanding training from 230 to 704 Boston nights on a fixed sequence architecture added {fmt("dev.additional_training.delta", 4, true)} Macro-F1 ({fmt("dev.additional_training.reference.macro_f1", 4)} → {fmt("dev.additional_training.candidate.macro_f1", 4)}). On that same 704-patient pool (<SecRef to="supp-cardio" />), unifying complementary slow-wave, spindle/K-complex, and night-history markers inside whole-night recurrence added {fmt("dev.concept79.vs_73.delta", 4, true)} in Model D (#{conceptData.active_experiment}, {fmt("dev.concept79.calibrated.macro_f1", 4)}), matching 309-feature Multi-scale HGB (#76, {fmt("dev.hgb704.calibrated.macro_f1", 4)}) and trading {formatMetric(r("dev.bigru75.calibrated.macro_f1") - attr.full_macro_f1)} vs. 266k-weight Dense BiGRU (#75, {fmt("dev.bigru75.calibrated.macro_f1", 4)}) for 4× fewer weights and four-branch additive local scoring ({formatMetric(attr.local_only_macro_f1)} local-only; +{formatMetric(attr.temporal_context_gain)} from directional gates, e.g., Wake future {wakeGate ? formatMetric(wakeGate.future) : "—"} vs. N3 past {n3Gate ? formatMetric(n3Gate.past) : "—"}). Adding ECG heart rate and respiratory envelopes in #80 ({fmt("dev.cues80.calibrated.macro_f1", 4)}) yielded no net 5-fold gain.
      </p>
      <p>
        <b>2. Cross-hospital shift, calibration, and conformal limits:</b> Out-of-fold Mondrian thresholds hold ~90% / 95% per-stage coverage in Boston CV (single-stage accuracy {formatPercent(mon90.singleton_accuracy)} at 90%) and {pct("ext.concept79.holdout.conformal.coverage_95.coverage")} overall at 95% on Boston holdouts. Under distribution shift without local recalibration, 90%-target coverage dips to {formatPercent(conceptData.external_eval.holdout.conformal_90.per_stage_coverage.N2)}/{formatPercent(conceptData.external_eval.holdout.conformal_90.per_stage_coverage.N3)} on apnea-heavy Boston holdout N2/N3, and drops to {formatPercent(conceptData.external_eval.isruc.conformal_90.per_stage_coverage.Wake)} on Wake and {formatPercent(conceptData.external_eval.isruc.conformal_90.per_stage_coverage.N3)} on N3 against Coimbra Scorer 1. Likewise, Boston-fitted prior-power calibration slightly helped Boston holdout ({fmt("ext.concept79.holdout.raw.macro_f1", 4)} → {fmt("ext.concept79.holdout.calibrated.macro_f1", 4)}) but trimmed raw Coimbra logits ({fmt("ext.concept79.isruc.raw_scorer1.macro_f1", 4)} → {fmt("ext.concept79.isruc.calibrated_scorer1.macro_f1", 4)} Scorer 1), pointing to clinic differences in stage prevalence and ≥75 µV frontal slow-wave amplitude thresholds. Because detecting under-coverage requires labeled target-clinic hypnograms, transferring to a new hospital warrants lightweight site recalibration on a small labeled sample.
      </p>
      <p>
        <b>3. Deployment constraints &amp; next engineering steps:</b> Model D is strictly an offline post-recording scorer: in addition to the anti-causal future GRU, several multi-minute markers (<code>atonia_fraction_10min</code>, <code>nearby_eye_atonia</code>, <code>slow_burden_vs_clock</code>) and per-night robust normalization depend on the complete recording, so real-time bedside staging would require causal feature and causal-GRU replacements. Finally, per-night holdout Macro-F1 spans {fmt("ext.concept79.holdout.calibrated.subj_median")} median down to {conceptData.external_eval.holdout.subj_min.toFixed(3)} (<a href="#validation-distribution">Figure 2</a>), motivating automated lead-dropout/artifact triage on the lowest-decile apnea recordings.
      </p>
      {full && smaller ? (
        <p className="research-protocol-source">
          Archived Model C complexity reference (<SecRef to="supp-simplify" />): capping tree depth to 8 leaves reduced tree values by {Math.round((1 - smaller.learned_values / full.learned_values) * 100)}% for a {formatMetric(full.macro_f1 - smaller.macro_f1)} drop in macro-F1.
        </p>
      ) : null}
    </div>
  );
}

function ReferencesSection() {
  return (
    <section className="research-section research-references" id="references">
      <SectionHeading title="References" />
      <ol className="research-reference-list">
        {references.map((reference) => (
          <li id={`reference-${reference.number}`} key={reference.number}>
            <span>{reference.number}</span>
            <a href={reference.href} target="_blank" rel="noreferrer">
              {reference.citation}
            </a>
          </li>
        ))}
      </ol>
      <footer className="research-footer">
        <span>Boston sleep-staging experiments</span>
        <a href="#top">Back to top ↑</a>
      </footer>
    </section>
  );
}

export function StagingResearchDashboard() {
  useOpenOnHash();
  const adopted = data.models.find((model) => model.id === "adopted") as ModelResult;
  const epochReference = data.models.find((model) => model.id === "epoch");
  return (
    <main className="research-page">
      <section className="research-hero" id="top">
        <div className="research-hero-copy">
          <h1>Automatic sleep staging from overnight polysomnography</h1>
        </div>
      </section>

      <StudySummary />

      <section className="research-section" id="lead-overview">
        <ConceptHypnogramFigure />
        <ConceptExternalDistFigure />
      </section>

      <section className="research-section" id="introduction">
        <SectionHeading title="1 Introduction" />
        <div className="research-method-block" id="staging">
          <SubHeading number="1.1" title="Sleep staging" />
          <SleepStudyPrimer />
        </div>
        <div className="research-method-block" id="stages">
          <SubHeading number="1.2" title="The physiology of each stage" />
          <StagePrimer />
        </div>
      </section>

      <section className="research-section" id="methods">
        <SectionHeading title="2 Methods" />
        <div className="research-method-block" id="data">
          <SubHeading number="2.1" title="Data and testing" />
          <TestingProtocol />
          <CompactStudyProtocol />
        </div>
        <div className="research-method-block" id="features">
          <SubHeading number="2.2" title="Features" />
          <p>
            The interactive catalog below covers the 33 baseline physiological measurements developed across Models A–C (badging which 13 were pruned in 20-signal Model C). Model D retains 17 core EEG, EOG, and chin EMG cues from this set and adds 12 complementary slow-wave, spindle/K-complex, and multi-minute night-history markers (29 distinct features = {conceptData.input_dim} inputs per epoch; §2.3).
          </p>
          <div className="research-feature-explorer-section">
            <h4>Explore the measurements by stage</h4>
            <Evidence summary="Signal types and feature labels">
              <div className="research-signal-key" aria-label="Measurement colors">
                {[
                  { name: "Brain", label: "EEG" },
                  { name: "Eyes", label: "EOG" },
                  { name: "Chin muscle", label: "EMG" },
                  { name: "Heart", label: "ECG" },
                  { name: "Breathing", label: "Respiratory effort" }
                ].map((sensor) => (
                  <span key={sensor.name}>
                    <i style={{ background: sensorColors[sensor.name] }} />
                    {sensor.label}
                  </span>
                ))}
              </div>
              <dl className="research-feature-legend">
                <dt>AASM rule</dt>
                <dd>Measures a cue used in manual staging.</dd>
                <dt>Published method</dt>
                <dd>Follows a published calculation and its standard parameters.</dd>
                <dt>Adapted</dt>
                <dd>Uses a detector, frequency band or time window adapted from published work.</dd>
              </dl>
            </Evidence>
            <PhysiologyExplorer />
          </div>
        </div>
        <div className="research-method-block" id="model">
          <SubHeading number="2.3" title="Model" />
          <ModelSection />
        </div>
      </section>

      <section className="research-section" id="results">
        <SectionHeading title="3 Results" />
        <div className="research-results-block research-block" id="performance">
          <SubHeading number="3.1" title="Performance" />
          <p className="research-takeaway">
            On 320 Boston development patients, expanding training data (255 → 704 nights/fold) and adding whole-night sequence context lifted performance from Model C ({formatMetric(adopted.metrics.macro_f1)}) to {formatMetric(conceptData.calibrated.mean_fold_macro_f1)}, which Model D (#{conceptData.active_experiment}) achieves at 4× fewer weights than #75.
          </p>
          <h4 id="comparison">Comparison with published models</h4>
          <ComparisonCompact />
          <h4>Model progression (A → B → C → D) on 320 development patients</h4>
          <p>
            From single-epoch trees (A {epochReference ? formatMetric(epochReference.metrics.macro_f1) : "—"}) and 2.5-minute context trees (B {fmt("cv.window33.epoch.macro_f1")} / Model C {formatMetric(adopted.metrics.macro_f1)}), Model D reached <b>{formatMetric(conceptData.calibrated.mean_fold_macro_f1)}</b> macro-F1 (κ {conceptData.calibrated.kappa.toFixed(2)}: Wake {formatMetric(conceptData.calibrated.stage_f1.Wake)}, N1 {formatMetric(conceptData.calibrated.stage_f1.N1)}, N2 {formatMetric(conceptData.calibrated.stage_f1.N2)}, N3 {formatMetric(conceptData.calibrated.stage_f1.N3)}, REM {formatMetric(conceptData.calibrated.stage_f1.REM)}) with {conceptData.model_params.toLocaleString("en-US")} weights/fold. On the same 704-patient folds (<SecRef to="supp-cardio" />), Model D matches Multi-scale HGB (#76, {fmt("dev.hgb704.calibrated.macro_f1")}) and gives up {formatMetric(r("dev.bigru75.calibrated.macro_f1") - conceptData.calibrated.mean_fold_macro_f1)} vs. Dense BiGRU (#75, {fmt("dev.bigru75.calibrated.macro_f1")}, 266k weights) for 4× smaller size and additive stems. Brief transitional N1 stays hardest ({pct("lit.aasm.agreement_N1")} AASM human consensus).
          </p>
          <ConceptProgressionSection />
        </div>
        <div className="research-results-block research-block" id="evidence">
          <SubHeading number="3.2" title="Additive interpretability &amp; conformal uncertainty" />
          <p className="research-takeaway">
            Model D separates per-epoch additive physiology from directional sequence gates and converts uncertain transitions into calibrated Mondrian prediction sets.
          </p>
          <h4>Additive physiological concept attributions &amp; temporal gates</h4>
          <ConceptAttributionSection />
          <h4 className="research-subsection-gap">Calibrated conformal prediction sets (marginal vs. Mondrian)</h4>
          <ConformalUncertaintySection />
        </div>
        <div className="research-results-block research-block" id="robustness">
          <SubHeading number="3.3" title="Locked external validation (Boston holdout &amp; Coimbra ISRUC)" />
          <p className="research-takeaway">
            Scoring the 5-fold ensemble in locked Look #3 (<Exp ids={[81]} />) held <b>{fmt("ext.concept79.holdout.calibrated.macro_f1", 4)}</b> on 200 apnea-heavy Boston holdout patients (<b>{fmt("delta.concept79_holdout_vs_model_c", 4, true)}</b> vs. Model C) and reached <b>{fmt("ext.concept79.isruc.calibrated_scorer1.macro_f1", 4)} / {fmt("ext.concept79.isruc.calibrated_scorer2.macro_f1", 4)}</b> on 99 Coimbra hospital nights—ahead of Model C on both scorers, though cross-site N3 shift narrowed Boston&rsquo;s margin.
          </p>
          <div className="research-model-table-wrap">
            <table className="research-model-table">
              <thead>
                <tr>
                  <th scope="col">Cohort &amp; evaluation mode</th>
                  <th scope="col">Macro-F1</th>
                  <th scope="col">κ</th>
                  <th scope="col">Acc.</th>
                  <th scope="col">Wake / N1 / N2 / N3 / REM</th>
                  <th scope="col">Subj. median</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <th scope="row">Boston CV (320 dev, single-fold OOF)</th>
                  <td>{fmt("dev.concept79.calibrated.macro_f1")}</td>
                  <td>{fmt("dev.concept79.calibrated.kappa")}</td>
                  <td>{formatPercent(conceptData.calibrated.accuracy)}</td>
                  <td>
                    {fmt("dev.concept79.calibrated.f1_Wake", 2)} / {fmt("dev.concept79.calibrated.f1_N1", 2)} / {fmt("dev.concept79.calibrated.f1_N2", 2)} / {fmt("dev.concept79.calibrated.f1_N3", 2)} / {fmt("dev.concept79.calibrated.f1_REM", 2)}
                  </td>
                  <td>{formatMetric(conceptData.calibrated.subj_median)}</td>
                </tr>
                <tr className="is-adopted">
                  <th scope="row">Boston Holdout (200, 5-fold ensemble, Look #3)</th>
                  <td>{fmt("ext.concept79.holdout.calibrated.macro_f1")}</td>
                  <td>{fmt("ext.concept79.holdout.calibrated.kappa")}</td>
                  <td>{pct("ext.concept79.holdout.calibrated.accuracy")}</td>
                  <td>
                    {fmt("ext.concept79.holdout.calibrated.f1_Wake", 2)} / {fmt("ext.concept79.holdout.calibrated.f1_N1", 2)} / {fmt("ext.concept79.holdout.calibrated.f1_N2", 2)} / {fmt("ext.concept79.holdout.calibrated.f1_N3", 2)} / {fmt("ext.concept79.holdout.calibrated.f1_REM", 2)}
                  </td>
                  <td>{fmt("ext.concept79.holdout.calibrated.subj_median")}</td>
                </tr>
                <tr>
                  <th scope="row">Coimbra ISRUC (99, 5-fold ensemble) vs. Scorer 1</th>
                  <td>{fmt("ext.concept79.isruc.calibrated_scorer1.macro_f1")}</td>
                  <td>{fmt("ext.concept79.isruc.calibrated_scorer1.kappa")}</td>
                  <td>{pct("ext.concept79.isruc.calibrated_scorer1.accuracy")}</td>
                  <td>
                    {fmt("ext.concept79.isruc.calibrated_scorer1.f1_Wake", 2)} / {fmt("ext.concept79.isruc.calibrated_scorer1.f1_N1", 2)} / {fmt("ext.concept79.isruc.calibrated_scorer1.f1_N2", 2)} / {fmt("ext.concept79.isruc.calibrated_scorer1.f1_N3", 2)} / {fmt("ext.concept79.isruc.calibrated_scorer1.f1_REM", 2)}
                  </td>
                  <td>{fmt("ext.concept79.isruc.calibrated_scorer1.subj_median")}</td>
                </tr>
                <tr>
                  <th scope="row">Coimbra ISRUC (99, 5-fold ensemble) vs. Scorer 2</th>
                  <td>{fmt("ext.concept79.isruc.calibrated_scorer2.macro_f1")}</td>
                  <td>{fmt("ext.concept79.isruc.calibrated_scorer2.kappa")}</td>
                  <td>{pct("ext.concept79.isruc.calibrated_scorer2.accuracy")}</td>
                  <td>
                    {fmt("ext.concept79.isruc.calibrated_scorer2.f1_Wake", 2)} / {fmt("ext.concept79.isruc.calibrated_scorer2.f1_N1", 2)} / {fmt("ext.concept79.isruc.calibrated_scorer2.f1_N2", 2)} / {fmt("ext.concept79.isruc.calibrated_scorer2.f1_N3", 2)} / {fmt("ext.concept79.isruc.calibrated_scorer2.f1_REM", 2)}
                  </td>
                  <td>{fmt("ext.concept79.isruc.calibrated_scorer2.subj_median")}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p>
            On Boston holdouts (59% moderate or severe apnea), Model D beat Model C in every stage and shrank the CV-to-holdout drop (−{formatMetric(conceptData.calibrated.mean_fold_macro_f1 - r("ext.concept79.holdout.calibrated.macro_f1"))} vs. −{formatMetric(adopted.metrics.macro_f1 - r("holdout.look2.g20.macro_f1"))}), matching U-Time&rsquo;s single-channel raw-EEG Boston mark ({fmt("lit.utime.macro_f1", 2)} on another split) <Citation number={27} /> while trailing U-Sleep ({fmt("lit.usleep.macro_f1", 2)} on 100 other Boston records) <Citation number={26} /> and SLEEPNET ({pct("lit.sleepnet.accuracy")} accuracy on 1,000 MGH nights vs. Model D&rsquo;s {pct("ext.concept79.holdout.calibrated.accuracy")}) <Citation number={28} />. On Coimbra ISRUC nights, Model D improved over Model C on both scorers ({fmt("delta.concept79_isruc_s1_vs_model_c", 4, true)} Scorer 1; {fmt("delta.concept79_isruc_s2_vs_model_c", 4, true)} Scorer 2) and trailed same-cohort inter-scorer agreement ({fmt("coimbra.isruc_scorer1_vs_scorer2.macro_f1", 3)}; <a href="#validation-distribution">Figure 2</a>). Archived Model C holdout tables remain in <SecRef to="supp-models" />.
          </p>
        </div>
      </section>

      <section className="research-section research-discussion" id="discussion">
        <SectionHeading title="4 Discussion and conclusion" />
        <DiscussionExplorer />
      </section>

      <section className="research-section" id="supplementary">
        <SectionHeading title="Supplementary" />
        <div className="research-supplement-disclosures">
          <Evidence id="supp-patients" summary="S1 Variation between Boston development patients">
            <SubjectVariability />
          </Evidence>
          <Evidence id="supp-models" summary="S2 Model C development, simplification, confusion &amp; archived external baseline">
            <h4>Early Model C vs Window B0 / Attention CNN comparison</h4>
            <ModelExplorer />
            <h4>Model C confusion breakdown by stage (#36)</h4>
            <ConfusionExplorer />
            <h4>Model C held-out and Coimbra baseline archive (#30, #37, #38)</h4>
            <ExternalTestingDetails />
            <h4>From 33 inputs to 20</h4>
            <ModelPath />
            <SpindleComparison />
            <div id="supp-simplify">
              <h4>Simpler models</h4>
              <ModelSimplification />
            </div>
            <div id="supp-published-size">
              <h4>Published models and parameter counts</h4>
              <PublishedModelSize />
              <p>
                SLEEPNET reported {pct("lit.sleepnet.accuracy")} accuracy and κ {fmt("lit.sleepnet.kappa", 2)} on 1,000 MGH recordings after training on about 10,000 patients. <Citation number={28} /> The AASM inter-scorer program reported {pct("lit.aasm.agreement")} agreement with the majority on a separate sample. <Citation number={29} />
              </p>
            </div>
          </Evidence>
          <Evidence id="supp-importance" summary="S3 Model C permutation reliance, bout structure &amp; network attribution archive">
            <h4>Interpreting Model C signal permutation importance</h4>
            <p>
              Importance is the fall in macro-F1 after swapping one measurement&rsquo;s complete input window between test epochs, keeping the stage labels and all other inputs unchanged ({data.importance_folds} folds). The trained model is scored again without being refitted. Drops are not additive; correlated measurements can substitute for one another, so a short bar indicates little additional reliance. EOG also contains frontal EEG, so eye-channel reliance is not specific evidence of eye movements. It shows what the model depends on; it does not establish a mechanism or explain individual predictions.
            </p>
            <div className="research-evidence-stack">
              <div className="research-feature-tab">
                <ModelEvidence part="signals" />
              </div>
              <ModelEvidence part="time" />
            </div>
            <div id="night">
              <h4>Stage transitions and bout lengths</h4>
              <SleepStructure />
            </div>
            <div id="supp-network">
              <h4>Exploratory neural-network attribution</h4>
              <BlackBoxAgreement />
            </div>
          </Evidence>
          <Evidence id="supp-cardio" summary="S4 Exploratory comparators (#75, #76, #78, #80), Cohen&rsquo;s d cue screen &amp; cardiac/respiratory ablations">
            <SupplementaryExplorationSection />
            <h4>Earlier cardiac and respiratory ablations (Model C era)</h4>
            <SignalAblations />
          </Evidence>
        </div>
        <div className="research-method-block" id="supp-history">
          <SubHeading number="S5" title="Experiment log and implementation" />
          <p>
            Every experiment, numbered when it was declared, with its finding, decision and evidence folder, and the Python behind each measurement, are on a <Link href="/history">separate page</Link>.
          </p>
        </div>
      </section>
      <ReferencesSection />
    </main>
  );
}
