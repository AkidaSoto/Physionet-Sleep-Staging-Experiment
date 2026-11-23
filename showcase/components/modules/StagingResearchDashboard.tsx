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
    { name: "A · Single epoch", inputs: "19 epoch-averaged features", window: "30 s", purpose: "Baseline without neighbouring epochs", refs: [16, 17] },
    { name: "B · Context trees", inputs: "33 measurements every 6 s", window: "150 s", purpose: "Test the gain from 2.5 minutes of surrounding context", refs: [19, 23] },
    { name: "C · Compact trees", inputs: "20 brain, eye and chin signals", window: "150 s", purpose: `Earlier ${data.model_size.trees}-tree baseline evaluated on Boston and Coimbra`, refs: [28, 33, 36] },
    { name: "D · Concept sequence model", inputs: "29 brain, eye and chin features (81 per-epoch inputs)", window: "Whole night", purpose: "Adopted model: four physiological branches, whole-night memory, and uncertainty sets", refs: [79] }
  ];
  const nConcepts = conceptData.concept_attribution.concepts.length;
  return (
    <div className="research-model-section">
      <p className="research-takeaway">
        Model D separates <em>physiological evidence</em> from <em>whole-night context</em>: {nConcepts} independent branches score each body system on its own before a compact recurrent layer reads backward and forward across the night.
      </p>
      <WindowModelFigure />
      <p>
        At each 30-second epoch, the measurements enter four separate physiological branches—deep slow waves, light-sleep events (spindles and K-complexes), wake and drowsiness cues, and eye/chin tone—whose scores add together before any cross-branch mixing. A compact bidirectional recurrent layer then reads across the full night and adjusts each stage through learned past and future gates, keeping total model size to {conceptData.model_params.toLocaleString("en-US")} weights (~{Math.round(uSleepWeights / conceptData.model_params)}× smaller than U-Sleep <Citation number={26} />). Rather than forcing a single guess on ambiguous transitions, stage-calibrated prediction sets return multiple plausible stages whenever confidence drops.
      </p>
      <div className="research-design-rationale">
        <h5>Why this design</h5>
        <ul className="research-glance-takeaways">
          <li>
            <b>Four separate physiological branches:</b> Grouping measurements by sleep physiology <Citation number={1} /> lets each system contribute an inspectable score before sequence modeling combines them across the night.
          </li>
          <li>
            <b>Short-epoch physiology separated from night-long patterns:</b> Following sequence-to-sequence sleep staging <Citation number={13} />, splitting local scoring from whole-night recurrence captures multi-minute sleep structure in a fraction of the size of raw-waveform networks <Citation number={26} />.
          </li>
          <li>
            <b>Balanced training and per-stage uncertainty:</b> Because brief light sleep (N1) makes up only ~10% of a night, tempered class weighting protects N1 without inflating false alarms, and per-stage uncertainty thresholds keep rare stages from being under-covered.
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
      <Evidence summary="Implementation details (64-unit GRUs, fold splits, normalization &amp; calibration)">
        <p>
          Each branch encodes its inputs into 16 units and outputs additive stage log-odds; their concatenated 64-unit representation feeds 1-layer causal and anti-causal GRUs (64 units per direction) merged via bounded per-stage gates (<code>logits = local + past_gate · past + future_gate · future</code>; {conceptData.model_params.toLocaleString("en-US")} weights/fold, averaged across 5 fold checkpoints on external cohorts). Each of the 5 development folds holds out 64 test nights, leaving ~255 non-test nights; Models A–C fit directly on all ~255, while Model D reserves ~25 for inner early stopping and adds 474 fit-only Boston nights (704 fitted nights/fold). Training uses fit-only standardization, tempered inverse-frequency weights, label smoothing, exponential moving average weights, and out-of-fold Mondrian conformal quantiles. Additional 704-patient comparators (#75, #76, #78, #80) are archived in <SecRef to="supp-cardio" />.
        </p>
      </Evidence>
    </div>
  );
}

function StudySummary() {
  const activeF1 = conceptData.calibrated.mean_fold_macro_f1;
  const activeParams = conceptData.model_params;
  return (
    <section className="research-section research-summary" id="summary">
      <SectionHeading title="Abstract" />
      <div className="research-summary-copy research-abstract">
        <p>
          Clinicians stage overnight sleep studies by reading brain waves, eye movements, and chin muscle tone in 30-second slices, assigning each slice to wake, light sleep, deep sleep, or REM. Modern deep-learning networks can automate this task accurately from raw waveforms, but their millions of parameters hide which physiological cues drove a call and force a single confident label even when a stage transition is genuinely ambiguous.
        </p>
        <p>
          This study develops a compact, structured alternative—<b>Model D</b>—that separates standard brain, eye, and chin measurements into four independent physiological branches before reading backward and forward across the night. With roughly {Math.round(uSleepWeights / activeParams)} times fewer weights than leading raw-waveform networks <Citation number={26} />, it keeps per-system reasoning inspectable and pairs every decision with a calibrated uncertainty set that lists multiple plausible stages on blurry transitions.
        </p>
        <p>
          Across 320 Boston development patients, Model D reached <b>{formatMetric(activeF1)}</b> macro-F1. In locked testing on 200 unseen Boston patients with heavier sleep apnea, it held <b>{fmt("ext.concept79.holdout.calibrated.macro_f1")}</b>—improving substantially over our earlier decision-tree baseline—and transferred without retraining to 99 patients from an independent hospital in Coimbra, Portugal at <b>{fmt("ext.concept79.isruc.calibrated_scorer1.macro_f1")}</b> against clinical scorers. Checking its uncertainty sets across hospitals also revealed a slow-wave calibration shift in Coimbra, showing where lightweight local calibration is needed before cross-clinic deployment.
        </p>
      </div>
    </section>
  );
}

function DiscussionExplorer() {
  const full = data.complexity.find((row) => row.learner === "hgb_l31_i100");
  const smaller = data.complexity.find((row) => row.learner === "hgb_l8_i100");
  const gates = conceptData.concept_attribution.stage_gates;
  const wakeGate = gates.find((g) => g.stage === "Wake");
  const n3Gate = gates.find((g) => g.stage === "N3");
  const attr = conceptData.concept_attribution;
  const adopted = data.models.find((model) => model.id === "adopted") as ModelResult;
  return (
    <div className="research-conclusion">
      <p>
        <b>What drove the gain over earlier models.</b> Comparing models trained on the same expanded Boston pool (<SecRef to="supp-cardio" />) shows that both extra training recordings (<Exp ids={[73]} />, {fmt("dev.additional_training.delta", 3, true)}) and richer sleep-event features with whole-night recurrence ({fmt("dev.concept79.vs_73.delta", 3, true)}) contributed to moving from Model C ({formatMetric(adopted.metrics.macro_f1)}) to Model D ({formatMetric(attr.full_macro_f1)}). Model D gives up a tiny fraction of cross-validation accuracy relative to a dense black-box recurrent network (#75) in exchange for four times fewer weights and an inspectable four-branch decomposition, where directional memory adds +{formatMetric(attr.temporal_context_gain)} and recovers natural sleep-stage asymmetry (Wake looks ahead at {wakeGate ? formatMetric(wakeGate.future) : "—"}; deep N3 anchors to preceding slow waves at {n3Gate ? formatMetric(n3Gate.past) : "—"}). Adding heart rate and breathing signals (#80) gave no net gain once brain, eye, and chin cues were present.
      </p>
      <p>
        <b>Generalization limits across hospitals.</b> Model D held up cleanly on the harder Boston holdout cohort, but transferring to Coimbra without retraining narrowed its margin over Model C. Its per-stage uncertainty checks pinpointed why: fixed Boston thresholds held well in development, dipped modestly on N2 and N3 in apnea-heavy Boston holdouts, and under-covered deep N3 sleep ({formatPercent(conceptData.external_eval.isruc.conformal_90.per_stage_coverage.N3)}) and Wake ({formatPercent(conceptData.external_eval.isruc.conformal_90.per_stage_coverage.Wake)}) in Coimbra. Because hospital montages, scoring habits, and slow-wave amplitudes differ across clinics, moving a staging model to a new hospital still warrants checking and recalibrating thresholds on a small labeled local sample.
      </p>
      <p>
        <b>Practical constraints and next steps.</b> Because Model D uses both future context and whole-night normalization, it is an offline post-recording scorer rather than a real-time bedside monitor; real-time use would require causal-only features and forward-only recurrence. In addition, patient-to-patient spread remains wide on the most severely fragmented recordings (<a href="#validation-distribution">Figure 2</a>), pointing to automated lead-quality and artifact triage as the clearest next improvement.
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
            Moving from single-epoch trees to whole-night sequence modeling lifted Boston development performance from Model C ({formatMetric(adopted.metrics.macro_f1)}) to Model D ({formatMetric(conceptData.calibrated.mean_fold_macro_f1)}) while staying orders of magnitude smaller than raw-waveform networks.
          </p>
          <h4 id="comparison">Comparison with published models</h4>
          <ComparisonCompact />
          <h4>Model progression (A → B → C → D) on 320 development patients</h4>
          <p>
            Single-epoch trees (A, {epochReference ? formatMetric(epochReference.metrics.macro_f1) : "—"}) improved substantially once 2.5 minutes of surrounding context was added (B and C). Expanding the training pool and adding whole-night sequence memory in Model D lifted macro-F1 to <b>{formatMetric(conceptData.calibrated.mean_fold_macro_f1)}</b> (κ {conceptData.calibrated.kappa.toFixed(2)}), with the biggest remaining bottleneck in brief transitional N1 sleep ({pct("lit.aasm.agreement_N1")} human scorer consensus).
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
          <SubHeading number="3.3" title="Locked external validation (Boston holdout &amp; Coimbra)" />
          <p className="research-takeaway">
            On 200 reserved Boston patients with heavier sleep apnea, Model D held <b>{fmt("ext.concept79.holdout.calibrated.macro_f1")}</b> macro-F1 ({fmt("delta.concept79_holdout_vs_model_c", 3, true)} over Model C). On 99 patients from an independent hospital in Coimbra, it reached <b>{fmt("ext.concept79.isruc.calibrated_scorer1.macro_f1")}</b> against clinical scorers—still ahead of Model C, though cross-hospital differences narrowed Boston&rsquo;s margin. <Exp ids={[81]} />
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
          <ConceptHypnogramFigure />
          <ConceptExternalDistFigure />
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
