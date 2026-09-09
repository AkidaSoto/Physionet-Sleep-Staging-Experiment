"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  stagingDashboardData,
  type FeatureDistribution,
  type FeatureSignalExample,
  type ModelResult
} from "../../content/staging-dashboard-data";

const data = stagingDashboardData;
const stageColors: Record<string, string> = {
  Wake: "#91a9ce",
  N1: "#72d2ff",
  N2: "#6fc4a2",
  N3: "#f0b36d",
  REM: "#ef8ab5"
};

const stageOrder = ["Wake", "REM", "N1", "N2", "N3"];

function Citation({ number }: { number: number }) {
  return <a className="research-citation" href={`#reference-${number}`} aria-label={`Reference ${number}`}>[{number}]</a>;
}
const modelDescriptions: Record<string, string> = {
  "raw-ensemble": "The ensemble classifies each 30-second epoch from its physiological feature vector.",
  "context-5": "A second model receives five consecutive vectors of baseline probabilities.",
  "tcn-6s": "A dilated convolutional network receives features sampled every six seconds.",
  hierarchical: "Separate encoders summarize short feature sequences and neighboring epochs.",
  "context-9": "A second model receives nine epochs of baseline logits and confidence."
};

type FeatureSpec = {
  id: string;
  name: string;
  signal: string;
  summary: string;
  measurement: string;
  stageUse: string;
};

type FeatureFamily = {
  id: string;
  label: string;
  signal: string;
  summary: string;
  evidence: string;
  limitation: string;
  sources: number[];
  features: FeatureSpec[];
};

const bandPurpose: Record<string, string> = {
  delta: "Deep sleep and slow-wave activity",
  theta: "Sleep onset and lighter sleep",
  alpha: "Relaxed wakefulness and wake-to-sleep transitions",
  sigma: "Spindle-range activity associated with N2",
  beta: "Higher-frequency activation and wake-like activity",
  gamma: "Fast activity that can help identify wake-like or noisy intervals"
};

const eegSpectrumFeatures: FeatureSpec[] = [
  {
    id: "spindle_features.aperiodic_exponent",
    name: "Aperiodic exponent",
    signal: "EEG",
    summary: "Describes the slope of the broadband EEG background rather than a single oscillatory band.",
    measurement: "Slope of the modeled aperiodic component of the power spectrum.",
    stageUse: "Separates changes in overall spectral shape across sleep depth."
  },
  {
    id: "spindle_features.aperiodic_intercept",
    name: "Aperiodic intercept",
    signal: "EEG",
    summary: "Describes the broadband level of the EEG spectrum after its slope is modeled.",
    measurement: "Offset of the modeled aperiodic component of the power spectrum.",
    stageUse: "Provides a subject-normalized measure of background spectral power."
  },
  ...["delta", "theta", "alpha", "sigma", "beta", "gamma"].flatMap((band) => [
    {
      id: `spindle_features.${band}_power`,
      name: `${band[0].toUpperCase()}${band.slice(1)} power`,
      signal: "EEG",
      summary: `Measures EEG energy in the ${band} band.`,
      measurement: `Band-limited spectral power within each analysis window.`,
      stageUse: bandPurpose[band]
    },
    {
      id: `spindle_features.${band}_residual_power`,
      name: `${band[0].toUpperCase()}${band.slice(1)} residual power`,
      signal: "EEG",
      summary: `Measures ${band}-band activity after accounting for the broadband spectral background.`,
      measurement: `Observed ${band} power minus the modeled aperiodic contribution.`,
      stageUse: `${bandPurpose[band]}; emphasizes band-specific structure.`
    }
  ])
];

const featureFamilies: FeatureFamily[] = [
  {
    id: "spindles",
    label: "Spindles",
    signal: "EEG",
    summary: "Three ratios summarize whether sigma-range activity stands out from faster background activity.",
    evidence: "Sleep spindles are brief sigma-frequency bursts characteristic of N2. Large population studies also show that spindle expression varies with age, sleep cycle, and recording context.",
    limitation: "A high sigma ratio is spindle-like evidence, not an event detector. Muscle and cardiac contamination can also raise power in this range.",
    sources: [1, 11],
    features: [
      {
        id: "spindle_features.splindex",
        name: "Spindle index",
        signal: "EEG",
        summary: "A compact spindle-like activity score derived from sigma-to-beta structure.",
        measurement: "Sigma/beta ratio combined with the residual sigma–beta contrast.",
        stageUse: "Targets spindle evidence used to distinguish N2."
      },
      {
        id: "spindle_features.sigma_beta_ratio",
        name: "Sigma / beta ratio",
        signal: "EEG",
        summary: "Compares spindle-range power with higher-frequency beta activity.",
        measurement: "Sigma-band power divided by beta-band power.",
        stageUse: "Higher values can support N2-like sigma prominence."
      },
      {
        id: "spindle_features.sigma_beta_residual_diff",
        name: "Residual sigma–beta difference",
        signal: "EEG",
        summary: "Tests whether sigma activity remains prominent after broadband background power is removed.",
        measurement: "Residual sigma power minus residual beta power.",
        stageUse: "Adds a background-corrected spindle cue for N2."
      }
    ]
  },
  {
    id: "spectrum",
    label: "EEG spectrum",
    signal: "EEG",
    summary: "Fourteen features describe canonical frequency bands and their residual power above the broadband background.",
    evidence: "Sleep depth changes the shape of the EEG spectrum: delta tends to rise with deeper non-REM sleep while alpha falls. Spectral ratios can separate some stages strongly, but no single measure separates all five reliably.",
    limitation: "Band power is continuous and subject-dependent. It summarizes an epoch; it does not encode every transient waveform a scorer can see.",
    sources: [1, 10],
    features: eegSpectrumFeatures
  },
  {
    id: "slow-waves",
    label: "Slow waves",
    signal: "EEG",
    summary: "Five features measure slow-wave amplitude, power, and contrast with faster EEG activity.",
    evidence: "High-amplitude slow EEG is the defining visual evidence for N3. Relative slow-to-fast power was also one of the strongest individual measurements in a broad comparison of hand-engineered staging features.",
    limitation: "Slow activity can appear in N2 and can be inflated by drift or movement. Amplitude, frequency, and the amount of the epoch involved still matter.",
    sources: [1, 10],
    features: [
      { id: "slow_wave_features.slow_wave_envelope", name: "Slow-wave envelope", signal: "EEG", summary: "Tracks the amplitude envelope of low-frequency EEG activity.", measurement: "Envelope of the slow-wave-filtered signal.", stageUse: "Strong slow activity supports N3." },
      { id: "slow_wave_features.high_frequency_envelope", name: "High-frequency envelope", signal: "EEG", summary: "Tracks faster EEG activity alongside the slow-wave channel.", measurement: "Envelope of the complementary higher-frequency signal.", stageUse: "Provides contrast between deep-sleep and activated EEG." },
      { id: "slow_wave_features.slow_wave_raw_power", name: "Slow-wave raw power", signal: "EEG", summary: "Measures unnormalized power in the slow-wave signal.", measurement: "Mean power of the slow-wave-filtered EEG.", stageUse: "Captures the absolute prominence of deep-sleep activity." },
      { id: "slow_wave_features.slow_wave_ratio_power", name: "Slow-wave power ratio", signal: "EEG", summary: "Expresses slow-wave power relative to faster EEG activity.", measurement: "Slow-wave power divided by the comparison-band power.", stageUse: "Makes N3-like dominance easier to compare across subjects." },
      { id: "slow_wave_features.slow_wave_index", name: "Slow-wave index", signal: "EEG", summary: "A compact summary of relative slow-wave prominence.", measurement: "Derived index based on slow-wave and comparison-band power.", stageUse: "Provides a direct N3-oriented model input." }
    ]
  },
  {
    id: "eye-movement",
    label: "Eye movement",
    signal: "EOG",
    summary: "Five features describe overall eye activity and frequency content associated with slow and rapid movements.",
    evidence: "EOG carries slow-eye-movement evidence around sleep onset and rapid-eye-movement evidence in REM. Subject-independent studies have shown that two EOG channels alone contain substantial five-stage information.",
    limitation: "Eye movements also occur during wake, and EOG electrodes pick up frontal EEG. Interpret the channel alongside EEG and chin tone.",
    sources: [1, 12, 13],
    features: [
      { id: "eye_movement_activity.eye_movement_activity", name: "Eye-movement activity", signal: "EOG", summary: "Measures moment-to-moment EOG activity above a local baseline.", measurement: "Derived activity envelope from the EOG channels.", stageUse: "Helps separate REM and wake from quieter non-REM stages." },
      { id: "eye_movement_activity.eye_movement_baseline", name: "Eye-movement baseline", signal: "EOG", summary: "Estimates the local background level used to interpret eye activity.", measurement: "Rolling baseline of the EOG activity envelope.", stageUse: "Makes eye-movement activity comparable across a night." },
      { id: "eog_rem_features.rem_band_power_fast", name: "Rapid-eye band power", signal: "EOG", summary: "Measures faster EOG activity associated with rapid eye movements.", measurement: "Band-limited EOG power in the rapid-movement range.", stageUse: "Supports REM when paired with reduced chin tone." },
      { id: "eog_rem_features.sem_band_power_fast", name: "Slow-eye band power", signal: "EOG", summary: "Measures lower-frequency EOG activity associated with slow eye movements.", measurement: "Band-limited EOG power in the slow-movement range.", stageUse: "Provides evidence for sleep onset and N1." },
      { id: "eog_rem_features.noise_band_power_fast", name: "EOG noise-band power", signal: "EOG", summary: "Measures faster EOG content that may reflect artifact or wake-like activity.", measurement: "Power in the EOG comparison/noise band.", stageUse: "Helps qualify whether apparent eye movement is reliable." }
    ]
  },
  {
    id: "muscle-tone",
    label: "Muscle tone",
    signal: "Chin EMG",
    summary: "Two features describe current chin-muscle activity and its local reference level.",
    evidence: "Chin tone typically falls from wake through non-REM and is lowest in REM. Combined EOG and EMG features can recover much of the information used in multichannel staging.",
    limitation: "Movement, snoring, electrode contact, and REM sleep without atonia can break the expected pattern. Local baseline matters more than one absolute voltage.",
    sources: [1, 13, 14],
    features: [
      { id: "emg_tone_features.emg_tone", name: "Chin EMG tone", signal: "Chin EMG", summary: "Measures current chin-muscle activity.", measurement: "Smoothed amplitude of the chin EMG signal.", stageUse: "Higher tone supports wake; reduced tone supports REM." },
      { id: "emg_tone_features.emg_tone_baseline", name: "Chin EMG baseline", signal: "Chin EMG", summary: "Provides a local reference for interpreting muscle-tone changes.", measurement: "Rolling baseline of the EMG tone measure.", stageUse: "Allows relative suppression to be compared across subjects." }
    ]
  },
  {
    id: "cardiac",
    label: "Cardiac",
    signal: "ECG",
    summary: "Five secondary features summarize heart rate and beat-to-beat variability.",
    evidence: "Heart rate and autonomic balance change with sleep state; non-REM generally shifts toward parasympathetic dominance while REM is more wake-like and variable.",
    limitation: "ECG is secondary context here. Heart-rate features are not AASM stage-defining criteria and are affected by age, disease, medication, arousals, and respiratory events.",
    sources: [15, 16],
    features: [
      { id: "pan_tompkins.heart_rate_bpm_track", name: "Heart rate", signal: "ECG", summary: "Tracks instantaneous heart rate derived from detected beats.", measurement: "Beats per minute from consecutive R–R intervals.", stageUse: "Adds weak autonomic context across sleep stages." },
      { id: "pan_tompkins.hrv_rmssd_track", name: "HRV RMSSD", signal: "ECG", summary: "Measures short-term beat-to-beat variability.", measurement: "Root mean square of successive R–R interval differences.", stageUse: "Describes short-timescale autonomic variation." },
      { id: "pan_tompkins.hrv_sdnn_track", name: "HRV SDNN", signal: "ECG", summary: "Measures overall variability in recent beat intervals.", measurement: "Standard deviation of normal R–R intervals.", stageUse: "Adds broader autonomic variation to the feature set." },
      { id: "pan_tompkins.heart_rate_min_track", name: "Minimum heart rate", signal: "ECG", summary: "Records the lowest recent heart-rate estimate.", measurement: "Minimum beats-per-minute value within the analysis window.", stageUse: "Captures sustained slowing during sleep." },
      { id: "pan_tompkins.heart_rate_range_track", name: "Heart-rate range", signal: "ECG", summary: "Measures how much heart rate changes inside the window.", measurement: "Maximum minus minimum heart rate in the analysis window.", stageUse: "Adds a simple measure of autonomic instability." }
    ]
  }
];

type StageFeatureTarget = {
  stage: string;
  cue: string;
  contrast: string;
  featureIds: string[];
};

const stageFeatureTargets: StageFeatureTarget[] = [
  {
    stage: "Wake",
    cue: "Higher-frequency EEG, sustained chin tone, eye activity, and heart rate provide evidence against sleep.",
    contrast: "The targeted inputs emphasize activation relative to NREM and the muscle-tone contrast with REM.",
    featureIds: [
      "slow_wave_features.high_frequency_envelope",
      "emg_tone_features.emg_tone_baseline",
      "eye_movement_activity.eye_movement_baseline",
      "pan_tompkins.heart_rate_bpm_track"
    ]
  },
  {
    stage: "N1",
    cue: "Sleep onset weakens wake-like EEG structure and often introduces slow eye movements.",
    contrast: "The targeted inputs look for a shallow transition state without the spindle or slow-wave evidence of deeper NREM.",
    featureIds: [
      "spindle_features.aperiodic_exponent",
      "eye_movement_activity.eye_movement_activity",
      "eog_rem_features.sem_band_power_fast"
    ]
  },
  {
    stage: "N2",
    cue: "Sigma activity and sleep spindles are the main stage-specific cues, supported here by spectral shape and HRV.",
    contrast: "The targeted inputs emphasize spindle-range structure that separates N2 from N1 and N3.",
    featureIds: [
      "spindle_features.splindex",
      "spindle_features.sigma_residual_power",
      "spindle_features.aperiodic_exponent",
      "pan_tompkins.hrv_rmssd_track"
    ]
  },
  {
    stage: "N3",
    cue: "High-amplitude slow EEG is the defining visual evidence for deep NREM sleep.",
    contrast: "The targeted inputs measure both absolute slow activity and its dominance over faster EEG.",
    featureIds: [
      "slow_wave_features.slow_wave_envelope",
      "slow_wave_features.slow_wave_index",
      "slow_wave_features.slow_wave_ratio_power",
      "slow_wave_features.high_frequency_envelope"
    ]
  },
  {
    stage: "REM",
    cue: "Rapid eye activity paired with reduced chin tone distinguishes REM from wake and NREM.",
    contrast: "The targeted inputs combine EOG movement bands with the muscle-tone reference needed to resolve wake-like EEG.",
    featureIds: [
      "eye_movement_activity.eye_movement_activity",
      "eog_rem_features.rem_band_power_fast",
      "eog_rem_features.sem_band_power_fast",
      "emg_tone_features.emg_tone_baseline"
    ]
  }
];

const featureDerivations: Record<string, { formula: string; method: string }> = {
  spindles: {
    formula: "SBI = [Σ(Pσ / Pβ)] × [Σ(Rσ − Rβ)]",
    method: "A 2-second EEG spectrogram with 90% overlap estimates sigma (11–16 Hz) and beta (16–30 Hz) power. R is log-power remaining after a fitted 1/f background is removed."
  },
  spectrum: {
    formula: "Rb(f) = log₁₀|STFT(x)| − [a + b log₁₀(f)]",
    method: "Band medians come from 2-second EEG spectra. The aperiodic line is fitted over 2–7 Hz and 14–24 Hz; residual band power measures oscillatory structure above that background."
  },
  "slow-waves": {
    formula: "SWI = (L′ / H) × (L′ − H),   L′ = L − median₃₀s(L)",
    method: "L is the Hilbert envelope of 0.3–1.5 Hz EEG and H is the 30–100 Hz envelope. The 30-second rolling median removes the local slow-wave baseline."
  },
  "eye-movement": {
    formula: "EOGactivity = MA₁s(max[E₀.₁–₅ − median₃₀s(E₀.₁–₅), 0])",
    method: "The activity channel uses a band-limited Hilbert envelope. Separate 30-second spectra summarize slow-eye power at 0.1–0.5 Hz, rapid-eye power at 0.5–5 Hz, and a 5–30 Hz comparison band."
  },
  "muscle-tone": {
    formula: "Tone = MA₁s(|Hilbert(BP₁₀–₁₀₀{EMG})|)",
    method: "A 10–100 Hz chin-EMG envelope is smoothed for one second. A 60-second rolling median supplies the local tone baseline used to detect relative suppression."
  },
  cardiac: {
    formula: "HR = 60 / RR,   RMSSD = √mean[(RRᵢ₊₁ − RRᵢ)²]",
    method: "R peaks are detected from ECG. Heart rate and 30-second beat-domain summaries provide secondary autonomic context rather than stage-defining evidence."
  }
};

type ModelProfile = {
  question: string;
  input: string;
  context: string;
  architecture: string[];
  training: string;
  lesson: string;
};

const modelProfiles: Record<string, ModelProfile> = {
  "raw-ensemble": {
    question: "How well does the feature vector work without sequence information?",
    input: "Normalized physiological features from one 30-second epoch",
    context: "None; every epoch is classified independently",
    architecture: ["feature vector", "balanced feature ensemble", "5 stage probabilities"],
    training: "Each epoch is classified independently. This is the reference point for the other models.",
    lesson: "This establishes how far the physiological representation goes before sequence modeling is added."
  },
  "context-5": {
    question: "Does a five-epoch probability smoother improve the baseline?",
    input: "Five consecutive vectors of baseline class probabilities",
    context: "5 epochs · 2.5 minutes",
    architecture: ["epoch ensemble", "5 probability vectors", "learned Stage 2 smoother", "stage label"],
    training: "The Stage 2 model uses the ensemble's probability outputs. The physiological representation remains fixed.",
    lesson: "Performance improves while the underlying physiological feature model remains unchanged."
  },
  "tcn-6s": {
    question: "Does six-second feature sampling improve classification?",
    input: "A denser sequence of physiological features sampled every 6 seconds",
    context: "Within-epoch and neighboring short-timescale structure",
    architecture: ["6-second features", "dilated temporal convolutions", "pooled representation", "stage label"],
    training: "Dilated convolutions combine nearby 6-second steps without a recurrent state.",
    lesson: "Denser sampling improves on the epoch baseline but does not beat the strongest second-stage model."
  },
  hierarchical: {
    question: "Does a joint short-step and cross-epoch encoder improve the result?",
    input: "Short feature sequences organized within and across 30-second epochs",
    context: "Joint within-epoch and cross-epoch context",
    architecture: ["short feature steps", "epoch encoder", "cross-epoch encoder", "stage sequence"],
    training: "The raw hierarchical output is evaluated directly; a later smoothing stage did not become the default.",
    lesson: "The hierarchy improves on the epoch baseline, but its added complexity does not produce the best result."
  },
  "context-9": {
    question: "Do a wider window and baseline confidence add useful information?",
    input: "Nine epochs of baseline logits plus prediction confidence",
    context: "9 epochs · 4.5 minutes",
    architecture: ["epoch ensemble", "9 logit + confidence vectors", "learned Stage 2 smoother", "stage label"],
    training: "Logits preserve relative evidence across all classes, while confidence adds the baseline model's uncertainty.",
    lesson: "This configuration produces the highest mean macro-F1 in the comparison."
  }
};

function formatMetric(value: number) {
  return value.toFixed(3);
}

function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`;
}

function linePath(values: number[], width: number, height: number) {
  const sampled = values.filter((_, index) => index % Math.max(Math.floor(values.length / 700), 1) === 0);
  const finite = sampled.filter(Number.isFinite);
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  const span = max - min || 1;
  return sampled
    .map((value, index) => {
      const x = (index / Math.max(sampled.length - 1, 1)) * width;
      const y = height - ((value - min) / span) * height;
      return `${index === 0 ? "M" : "L"}${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

function SectionHeading({
  label,
  title,
  children
}: {
  label: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <header className="research-section-heading" aria-label={label}>
      <div>
        <h2>{title}</h2>
        <p className="research-section-lede">{children}</p>
      </div>
    </header>
  );
}

function ContextWindowExplorer() {
  const [windowSize, setWindowSize] = useState(5);
  const stages = ["Wake", "N1", "N2", "N2", "N2", "N2", "N3", "N3", "REM"];
  const center = 4;
  const radius = Math.floor(windowSize / 2);
  const explanations: Record<number, { title: string; body: string; span: string }> = {
    1: {
      title: "Single-epoch baseline",
      body: "The baseline receives the EEG, EOG, EMG, and ECG features measured within the selected 30 seconds.",
      span: "30 seconds"
    },
    5: {
      title: "Five-epoch smoother",
      body: "This model receives the baseline probabilities for the target epoch and two neighbors on either side.",
      span: "2.5 minutes"
    },
    9: {
      title: "Nine-epoch smoother",
      body: "This model receives logits and confidence for the target epoch and four neighbors on either side.",
      span: "4.5 minutes"
    }
  };
  const selected = explanations[windowSize];

  return (
    <div className="research-context-explorer">
      <div className="research-intro-copy">
        <p>
          A 30-second epoch can contain weak, mixed, or transitional evidence. Manual scorers disagree most often around Wake/N1, N1/N2, and N2/N3 boundaries because many transition epochs are physiologically equivocal. <Citation number={7} />
        </p>
        <p>
          Three model variants differ mainly in how many consecutive epoch predictions they receive. Prior multichannel work has tested similar inputs. <Citation number={4} />
        </p>
        <div className="research-question">
          <span>Comparison detail</span>
          <strong>The target prediction is the same. Only the information available around it changes.</strong>
        </div>
      </div>

      <div className="research-context-panel">
        <div className="research-context-panel-head">
          <div>
            <span>Selected prediction</span>
            <strong>{selected.span} of visible context</strong>
          </div>
          <div className="research-context-controls" aria-label="Choose a temporal context window">
            {[1, 5, 9].map((size) => (
              <button key={size} type="button" aria-pressed={windowSize === size} onClick={() => setWindowSize(size)}>
                {size} epoch{size > 1 ? "s" : ""}
              </button>
            ))}
          </div>
        </div>
        <div className="research-context-track" role="img" aria-label={`${windowSize} epoch context window centered on an N2 prediction`}>
          {stages.map((stage, index) => {
            const active = Math.abs(index - center) <= radius;
            return (
              <span key={`${stage}-${index}`} className={`${active ? "is-visible" : ""} ${index === center ? "is-target" : ""}`}>
                <i style={{ background: stageColors[stage] }} />
                <small>{stage}</small>
              </span>
            );
          })}
        </div>
        <div className="research-context-explanation" aria-live="polite">
          <strong>{selected.title}</strong>
          <p>{selected.body}</p>
        </div>
      </div>
    </div>
  );
}

function SignalTrace({ signal, active = false }: { signal: { label: string; values: number[] }; active?: boolean }) {
  const path = useMemo(() => linePath(signal.values, 720, 70), [signal.values]);
  return (
    <div className={`research-signal-row ${active ? "is-active" : ""}`}>
      <span>{signal.label}</span>
      <svg viewBox="0 0 720 70" role="img" aria-label={`${signal.label} signal excerpt`} preserveAspectRatio="none">
        <path d={path} />
      </svg>
    </div>
  );
}

function SleepStudyPrimer() {
  return (
    <div className="research-introduction">
      <div className="research-introduction-copy">
        <p>
          A <strong>polysomnogram (PSG)</strong> records several body systems on the same overnight timeline: brain activity (EEG), eye movement (EOG), chin-muscle tone (EMG), heart rhythm, breathing, and oxygen. Clinicians use the recording to understand how sleep is organized and to place respiratory events, arousals, and movements in physiological context. <Citation number={2} />
        </p>
        <p>
          For sleep staging, a technologist reads the EEG, EOG, and chin EMG in consecutive <strong>30-second epochs</strong> and assigns one label: Wake, N1, N2, N3, or REM. Joining those labels produces a hypnogram, the night’s sleep architecture. The labels support measures such as total sleep time, sleep efficiency, sleep latency, and time spent in each stage. <Citation number={1} />
        </p>
        <p>
          Manual staging is slow, and difficult epochs are often genuinely ambiguous, especially near transitions between adjacent stages. This project represents each epoch with physiological measurements and compares five classifiers against the expert labels. <Citation number={7} />
        </p>
      </div>

      <div className="research-introduction-figure" role="img" aria-label="A sleep study is divided into 30-second scoring epochs. Each epoch receives one stage label, and the labels form a whole-night hypnogram">
        <div className="research-introduction-signals">
          <span><strong>EEG</strong><small>brain activity</small></span>
          <span><strong>EOG</strong><small>eye movement</small></span>
          <span><strong>EMG</strong><small>chin tone</small></span>
        </div>
        <i>recorded together through the night</i>
        <div className="research-introduction-flow">
          <span className="research-introduction-scored-epoch"><b>30-second scoring epoch</b><small>one label: Wake, N1, N2, N3, or REM</small></span>
          <em>→</em>
          <span><b>Hypnogram</b><small>sleep architecture</small></span>
        </div>
        <p>This project uses EEG, EOG, chin EMG, and ECG from UCDDB. Respiratory-event detection is outside the present task. <Citation number={3} /></p>
      </div>

      <div className="research-objective">
        <span>Study objective</span>
        <p>Build a sleep-staging pipeline from physiological measurements, compare several classifiers under one validation protocol, and trace the errors back to the input representation.</p>
      </div>
    </div>
  );
}

function featureValue(value: number) {
  const magnitude = Math.abs(value);
  if (magnitude >= 100) return value.toFixed(0);
  if (magnitude >= 10) return value.toFixed(1);
  if (magnitude >= 1) return value.toFixed(2);
  return value.toFixed(3);
}

function ordinal(value: number) {
  const rounded = Math.round(value);
  const remainder100 = rounded % 100;
  if (remainder100 >= 11 && remainder100 <= 13) return `${rounded}th`;
  if (rounded % 10 === 1) return `${rounded}st`;
  if (rounded % 10 === 2) return `${rounded}nd`;
  if (rounded % 10 === 3) return `${rounded}rd`;
  return `${rounded}th`;
}

function FeatureDistributionChart({
  feature,
  marker
}: {
  feature: FeatureDistribution;
  marker?: { stage: string; value: number; percentile: number };
}) {
  const values = [
    ...feature.stages.flatMap((stage) => [stage.p10, stage.p90]),
    ...(marker ? [marker.value] : [])
  ];
  const rawMin = Math.min(...values);
  const rawMax = Math.max(...values);
  const padding = Math.max((rawMax - rawMin) * 0.08, 0.0001);
  const min = rawMin - padding;
  const max = rawMax + padding;
  const x = (value: number) => 18 + ((value - min) / Math.max(max - min, Number.EPSILON)) * 564;

  return (
    <div className="research-distribution-chart">
      <div className="research-distribution-axis" aria-hidden="true">
        <span>{featureValue(rawMin)}</span><span>{feature.label}</span><span>{featureValue(rawMax)}</span>
      </div>
      {feature.stages.map((stage) => (
        <div className="research-distribution-row" key={stage.stage}>
          <strong style={{ color: stageColors[stage.stage] }}>{stage.stage}</strong>
          <svg viewBox="0 0 600 40" role="img" aria-label={`${stage.stage}: median ${featureValue(stage.median)}, middle half ${featureValue(stage.p25)} to ${featureValue(stage.p75)}`} preserveAspectRatio="none">
            <line className="range" x1={x(stage.p10)} x2={x(stage.p90)} y1="20" y2="20" />
            <line className="cap" x1={x(stage.p10)} x2={x(stage.p10)} y1="14" y2="26" />
            <line className="cap" x1={x(stage.p90)} x2={x(stage.p90)} y1="14" y2="26" />
            <rect x={x(stage.p25)} y="9" width={Math.max(x(stage.p75) - x(stage.p25), 2)} height="22" fill={stageColors[stage.stage]} />
            <line className="median" x1={x(stage.median)} x2={x(stage.median)} y1="7" y2="33" />
            {marker?.stage === stage.stage ? <circle className="example-marker" cx={x(marker.value)} cy="20" r="6" /> : null}
          </svg>
          <span>{featureValue(stage.median)}</span>
        </div>
      ))}
      <p className="research-chart-key"><span /> 10th to 90th percentile <i /> middle 50% <b /> median {marker ? <em>● selected epoch</em> : null}</p>
    </div>
  );
}

function formatClock(totalSeconds: number) {
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = Math.round(totalSeconds % 60);
  return `${hours}:${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`;
}

function selectedSignalId(familyId: string) {
  if (familyId === "eye-movement") return "eog";
  if (familyId === "muscle-tone") return "emg";
  if (familyId === "cardiac") return "ecg";
  return "eeg_c3a2";
}

function NightHypnogram({ example }: { example: FeatureSignalExample }) {
  const values = example.hypnogram.values;
  const plotLeft = 58;
  const plotWidth = 834;
  const stepWidth = plotWidth / Math.max(values.length, 1);
  const y = (label: string) => 18 + stageOrder.indexOf(label) * 23;
  const path = values.map((value, index) => {
    const x = plotLeft + index * stepWidth;
    const stage = STAGES_FROM_INDEX[Number(value)] ?? "Wake";
    return index === 0 ? `M${x} ${y(stage)}` : `H${x} V${y(stage)}`;
  }).join(" ");
  const highlightX = plotLeft + (example.epoch_start_sec / example.hypnogram.end_sec) * plotWidth;

  return (
    <div className="research-integrated-night">
      <div className="research-integrated-night-head">
        <div><strong>{example.record_id}</strong><span>whole-night expert labels</span></div>
        <span>{formatClock(example.epoch_start_sec)} selected</span>
      </div>
      <svg viewBox="0 0 920 145" role="img" aria-label={`Whole-night hypnogram for ${example.record_id}; selected ${example.stage} epoch at ${formatClock(example.epoch_start_sec)}`}>
        {stageOrder.map((stage) => (
          <g key={stage}>
            <line x1={plotLeft} x2={plotLeft + plotWidth} y1={y(stage)} y2={y(stage)} />
            <text x="0" y={y(stage) + 4} fill={stageColors[stage]}>{stage}</text>
          </g>
        ))}
        <rect className="research-epoch-highlight" x={Math.max(highlightX - 4, plotLeft)} y="5" width="9" height="116" />
        <path className="research-hypnogram-path" d={path} />
        <circle cx={highlightX} cy={y(example.stage)} r="5" fill={stageColors[example.stage]} />
        <text className="research-hypnogram-time" x={plotLeft} y="141">lights out</text>
        <text className="research-hypnogram-time" x="842" y="141">morning</text>
      </svg>
    </div>
  );
}

const STAGES_FROM_INDEX = ["Wake", "N1", "N2", "N3", "REM"];

function PhysiologyExplorer() {
  const [stageId, setStageId] = useState("N2");
  const [examples, setExamples] = useState<FeatureSignalExample[] | null>(null);
  const [loadFailed, setLoadFailed] = useState(false);
  const explorerRef = useRef<HTMLDivElement>(null);
  const stageTarget = stageFeatureTargets.find((item) => item.stage === stageId) ?? stageFeatureTargets[2];
  const targetedFeatures = stageTarget.featureIds.flatMap((id) => {
    const family = featureFamilies.find((item) => item.features.some((feature) => feature.id === id));
    const feature = family?.features.find((item) => item.id === id);
    return family && feature ? [{ family, feature }] : [];
  });
  const [featureId, setFeatureId] = useState("spindle_features.splindex");
  const selectedTarget = targetedFeatures.find((item) => item.feature.id === featureId) ?? targetedFeatures[0];
  const family = selectedTarget?.family ?? featureFamilies[0];
  const feature = selectedTarget?.feature ?? family.features[0];
  const example = examples?.find((item) => item.family_id === family.id) ?? examples?.[0];
  const distribution = data.feature_evidence.distributions.find((item) => item.id === feature.id) ?? data.feature_evidence.distributions[0];
  const exampleValue = example?.feature_values[feature.id];
  const derivation = featureDerivations[family.id];

  useEffect(() => {
    let cancelled = false;
    const loadExamples = () => {
      void import("../../content/generated/staging-feature-examples.json")
        .then((module) => {
          if (!cancelled) setExamples(module.default as unknown as FeatureSignalExample[]);
        })
        .catch(() => {
          if (!cancelled) setLoadFailed(true);
        });
    };
    const node = explorerRef.current;
    if (!node || !("IntersectionObserver" in window)) {
      loadExamples();
      return () => { cancelled = true; };
    }
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        observer.disconnect();
        loadExamples();
      }
    }, { rootMargin: "500px 0px" });
    observer.observe(node);
    return () => {
      cancelled = true;
      observer.disconnect();
    };
  }, []);

  function chooseStage(next: StageFeatureTarget) {
    setStageId(next.stage);
    setFeatureId(next.featureIds[0]);
  }

  return (
    <div className="research-physiology-explorer" ref={explorerRef} aria-busy={!examples && !loadFailed}>
      <div className="research-explorer-intro">
        <div>
          <h4>Start with the label</h4>
        </div>
        <p>Choose a sleep stage, then inspect the measurements designed to separate it from the other labels.</p>
      </div>
      <div className="research-family-tabs research-stage-tabs" role="tablist" aria-label="Choose an expert sleep-stage label">
        {stageFeatureTargets.map((item) => (
          <button key={item.stage} type="button" role="tab" aria-selected={item.stage === stageTarget.stage} onClick={() => chooseStage(item)}>
            <span style={{ color: item.stage === stageTarget.stage ? stageColors[item.stage] : undefined }}>{item.stage}</span>
            <small>{item.featureIds.length} targeted inputs</small>
          </button>
        ))}
      </div>

      <div className="research-stage-target-summary">
        <span style={{ color: stageColors[stageTarget.stage] }}>{stageTarget.stage}</span>
        <p>{stageTarget.cue} <Citation number={1} /></p>
        <small>{stageTarget.contrast}</small>
      </div>

      <div className="research-feature-workbench">
        <article className="research-family-reading" aria-live="polite">
          <div><span>One-vs-rest view</span><strong>{stageTarget.stage} feature subset</strong></div>
          <p>The stage-expert experiments used these inputs to ask whether a focused physiological representation improves {stageTarget.stage} separability.</p>
          <div className="research-feature-list-inline research-stage-feature-list" role="listbox" aria-label={`${stageTarget.stage} targeted features`}>
            {targetedFeatures.map((item) => (
              <button key={item.feature.id} type="button" role="option" aria-selected={item.feature.id === feature.id} onClick={() => setFeatureId(item.feature.id)}>
                <span>{item.feature.name}</span>
                <small>{item.family.label} · {item.family.signal}</small>
              </button>
            ))}
          </div>
          <p className="research-feature-scope-note">The best multiclass model still receives the full feature vector. This label-first view explains the design logic behind selected inputs.</p>
        </article>

        <div className="research-feature-analysis">
          <div className="research-feature-definition">
            <span>{family.label} · {family.signal}</span>
            <h4>{feature.name}</h4>
            <p>{feature.summary}</p>
            <dl>
              <div><dt>Computed as</dt><dd>{feature.measurement}</dd></div>
              <div><dt>Target role</dt><dd>{feature.stageUse}</dd></div>
              <div><dt>Example value</dt><dd>{exampleValue ? `${featureValue(exampleValue.value)} · ${ordinal(exampleValue.percentile)} percentile` : "Loading example…"}</dd></div>
            </dl>
            <div className="research-feature-derivation">
              <strong>Derivation</strong>
              <code>{derivation.formula}</code>
              <small>{derivation.method}</small>
            </div>
            <details className="research-feature-family-note">
              <summary>Physiological basis and limits</summary>
              <p>{family.evidence} {family.sources.map((number) => <Citation key={number} number={number} />)}</p>
              <p><strong>Limit:</strong> {family.limitation}</p>
            </details>
          </div>
          <div className="research-selected-distribution">
            <div className="research-selected-distribution-head">
              <strong>Distribution by expert stage</strong>
              <span>target: {stageTarget.stage}{example ? ` · ● example: ${example.stage}` : ""}</span>
            </div>
            <FeatureDistributionChart
              feature={distribution}
              marker={example && exampleValue ? { stage: example.stage, value: exampleValue.value, percentile: exampleValue.percentile } : undefined}
            />
          </div>
        </div>
      </div>

      {!example ? (
        <div className="research-example-loading" role="status">
          {loadFailed ? "The waveform example could not be loaded. Refresh the page to retry." : "Loading one representative epoch…"}
        </div>
      ) : (
        <details className="research-example-disclosure" open>
          <summary>
            <span>Signal example</span>
            <strong>{example.stage} at {formatClock(example.epoch_start_sec)}</strong>
            <small>See how the selected feature appears in its source signals</small>
          </summary>
          <div className="research-example-context">
            <NightHypnogram example={example} />
            <div className="research-epoch-panel">
              <div className="research-epoch-heading">
                <div><span className="research-stage-dot" style={{ background: stageColors[example.stage] }} /> <strong>{example.stage}</strong><span>{example.title.replace(`${example.stage} `, "")}</span></div>
                <span>{formatClock(example.epoch_start_sec)} to {formatClock(example.epoch_start_sec + 30)}</span>
              </div>
              <div className="research-signal-stack">
                {example.signals.map((signal) => (
                  <SignalTrace key={signal.id} signal={signal} active={signal.id === selectedSignalId(family.id)} />
                ))}
              </div>
              <p className="research-example-method">{example.selection_note}</p>
            </div>
          </div>
        </details>
      )}
      <p className="research-data-note">The distribution uses every expert label. The waveform is a high-signal example for the selected feature family; its stage is shown explicitly above.</p>
    </div>
  );
}

function CompactStudyProtocol() {
  const total = data.diagnostics.per_stage.reduce((sum, stage) => sum + stage.support, 0);
  return (
    <div className="research-protocol">
      <div className="research-protocol-copy">
        <p>
          The experiments use the <strong>25 overnight recordings</strong> in the University College Dublin Sleep Apnea Database. The cohort consists of adults referred for suspected sleep-disordered breathing. <Citation number={3} />
        </p>
        <p>
          Original Rechtschaffen and Kales stage 3 and stage 4 labels are combined as N3. After preparation, the task contains <strong>{data.source.epochs.toLocaleString()} labeled 30-second epochs</strong> across Wake, N1, N2, N3, and REM.
        </p>
      </div>
      <div className="research-protocol-balance">
        <span>Class distribution</span>
        <div className="research-class-stack" role="img" aria-label="Distribution of labeled sleep stages">
          {data.diagnostics.per_stage.map((stage) => (
            <i key={stage.stage} style={{ width: `${(stage.support / total) * 100}%`, background: stageColors[stage.stage] }} title={`${stage.stage}: ${stage.support.toLocaleString()} epochs`} />
          ))}
        </div>
        <div className="research-class-legend">
          {data.diagnostics.per_stage.map((stage) => <span key={stage.stage}><i style={{ background: stageColors[stage.stage] }} />{stage.stage} <strong>{formatPercent(stage.support / total)}</strong></span>)}
        </div>
      </div>
    </div>
  );
}

function MethodPipeline() {
  const steps = [
    { number: "01", label: "Signals", detail: "EEG · EOG · chin EMG · ECG" },
    { number: "02", label: "Epoch representation", detail: "Physiological summary features" },
    { number: "03", label: "Model comparison", detail: "Ensemble, smoothers, TCN, and hierarchy" },
    { number: "04", label: "Stage prediction", detail: "One of five labels for each epoch" },
    { number: "05", label: "Evaluation", detail: "Macro-F1 by fold, stage, and subject" }
  ];
  return (
    <div className="research-method-pipeline">
      {steps.map((step, index) => (
        <div className="research-method-step" key={step.number}>
          <span>{step.number}</span>
          <strong>{step.label}</strong>
          <small>{step.detail}</small>
          {index < steps.length - 1 ? <i aria-hidden="true">→</i> : null}
        </div>
      ))}
    </div>
  );
}

function ModelDetail({ model, className = "" }: { model: ModelResult; className?: string }) {
  const profile = modelProfiles[model.id];
  const baseline = data.headline.baseline_macro_f1;
  const delta = model.metrics.macro_f1 - baseline;
  return (
    <div className={`research-model-detail ${className}`} aria-live="polite">
      <div className="research-model-detail-head">
        <div>
          <h3>{model.name}</h3>
        </div>
        {model.id === "context-9" ? <span className="research-winner">best mean macro-F1</span> : null}
      </div>
      <div className="research-inline-metrics research-model-metrics">
        <span><small>Accuracy</small><strong>{formatMetric(model.metrics.accuracy)}</strong></span>
        <span><small>vs. baseline</small><strong>{delta === 0 ? "reference" : `+${delta.toFixed(3)}`}</strong></span>
      </div>
      <div className="research-model-primary">
        <div className="research-model-architecture">
          <span className="research-model-kicker">Architecture</span>
          <div className="research-model-architecture-flow">
            {profile.architecture.map((step, index) => (
              <span key={step}><b>{step}</b>{index < profile.architecture.length - 1 ? <i>→</i> : null}</span>
            ))}
          </div>
          <dl>
            <div><dt>Input</dt><dd>{profile.input}</dd></div>
            <div><dt>Visible context</dt><dd>{profile.context}</dd></div>
          </dl>
        </div>
        <div className="research-model-overview">
          <span>Experiment question</span>
          <h4>{profile.question}</h4>
          <p>{modelDescriptions[model.id] ?? model.hypothesis}</p>
          <div>
            <strong>Result</strong>
            <p>{profile.lesson}</p>
          </div>
        </div>
      </div>
      <details className="research-model-training-disclosure">
        <summary><span>Training details</span><small>imbalance, selection, and model-specific setup</small></summary>
        <div className="research-model-training">
          <dl>
            <div><dt>Imbalance handling</dt><dd>{model.id === "raw-ensemble" || model.id.startsWith("context") ? "Balanced class weights." : "Weighted neural sampling."}</dd></div>
            <div><dt>Model selection</dt><dd>{model.id === "tcn-6s" || model.id === "hierarchical" ? "One training-only subject used for early stopping." : "All tuning remains inside the outer training subjects."}</dd></div>
            <div><dt>Model-specific setup</dt><dd>{profile.training}</dd></div>
          </dl>
        </div>
      </details>
    </div>
  );
}

function ModelResults() {
  const [selectedId, setSelectedId] = useState("context-9");
  const selected = data.models.find((model) => model.id === selectedId) ?? data.models[0];
  const scaleMin = 0.55;
  const scaleMax = 0.70;

  return (
    <div className="research-results-grid">
      <div className="research-model-list" role="list" aria-label="Model comparison by macro-F1">
        <div className="research-model-scale" aria-hidden="true"><span>Mean macro-F1</span><small>0.55</small><small>0.70</small></div>
        {data.models.map((model) => {
          const active = model.id === selected.id;
          const position = ((model.metrics.macro_f1 - scaleMin) / (scaleMax - scaleMin)) * 100;
          return (
            <div role="listitem" className="research-model-item" key={model.id}>
              <button
                type="button"
                className={active ? "is-selected" : ""}
                aria-pressed={active}
                onClick={() => setSelectedId(model.id)}
              >
                <span className="research-model-label">
                  <span><strong>{model.name}</strong><small>{model.role}</small></span>
                  <b>{formatMetric(model.metrics.macro_f1)}</b>
                </span>
                <span className="research-model-track"><i style={{ left: `${Math.min(Math.max(position, 0), 100)}%` }} /></span>
              </button>
              {active ? <ModelDetail model={selected} className="research-model-detail-mobile" /> : null}
            </div>
          );
        })}
      </div>
      <ModelDetail key={selected.id} model={selected} className="research-model-detail-desktop" />
    </div>
  );
}

function ConfusionExplorer() {
  const [selectedStage, setSelectedStage] = useState("N2");
  const selectedIndex = data.source.stages.indexOf(selectedStage);
  const metric = data.diagnostics.per_stage[selectedIndex];
  const row = data.diagnostics.confusion_normalized[selectedIndex];
  const largestErrorIndex = row.reduce((best, value, index) => {
    if (index === selectedIndex) return best;
    return value > row[best] || best === selectedIndex ? index : best;
  }, selectedIndex === 0 ? 1 : 0);

  return (
    <div className="research-error-grid">
      <div className="research-confusion-wrap">
        <div className="research-figure-header">
          <div>
            <h3>Normalized confusion matrix</h3>
            <p>Select a row to see how that stage was classified.</p>
          </div>
        </div>
        <div className="research-confusion">
          <span className="research-confusion-axis">Predicted stage →</span>
          <div className="research-confusion-head">
            <span />
            {data.source.stages.map((stage) => <span key={stage}>{stage}</span>)}
          </div>
          {data.diagnostics.confusion_normalized.map((values, rowIndex) => {
            const stage = data.source.stages[rowIndex];
            return (
              <button
                type="button"
                className={stage === selectedStage ? "is-selected" : ""}
                aria-pressed={stage === selectedStage}
                key={stage}
                onClick={() => setSelectedStage(stage)}
              >
                <strong>{stage}</strong>
                {values.map((value, columnIndex) => (
                  <span
                    key={`${stage}-${columnIndex}`}
                    style={{ "--cell-alpha": Math.max(value, 0.035).toFixed(3) } as React.CSSProperties}
                  >
                    {formatPercent(value)}
                  </span>
                ))}
              </button>
            );
          })}
        </div>
      </div>

      <div className="research-error-detail" aria-live="polite">
        <h3>{selectedStage} errors</h3>
        <div className="research-inline-metrics">
          <span><small>Precision</small><strong>{formatMetric(metric.precision)}</strong></span>
          <span><small>Recall</small><strong>{formatMetric(metric.recall)}</strong></span>
          <span><small>F1</small><strong>{formatMetric(metric.f1)}</strong></span>
        </div>
        <p>
          {formatPercent(row[selectedIndex])} of true {selectedStage} epochs are classified correctly. The most common error is {selectedStage} → {data.source.stages[largestErrorIndex]} at {formatPercent(row[largestErrorIndex])}.
        </p>
        {selectedStage === "N2" ? (
          <p className="research-diagnosis-next"><strong>Next test:</strong> I would improve the spindle and K-complex detectors, then rerun the same subject-grouped folds.</p>
        ) : null}
      </div>
    </div>
  );
}

function FeatureImportance() {
  const max = Math.max(...data.feature_importance.map((item) => item.importance));
  return (
    <div className="research-importance">
      <div>
        <h3>Feature importance by family</h3>
        <p>
          Slow-wave features account for 61% of normalized importance. Spindle features account for 7%.
        </p>
      </div>
      <div className="research-importance-bars">
        {data.feature_importance.map((item) => (
          <div key={item.group}>
            <span>{item.group}</span>
            <i><b style={{ width: `${(item.importance / max) * 100}%` }} /></i>
            <strong>{formatPercent(item.importance)}</strong>
          </div>
        ))}
      </div>
      <a className="research-text-link" href="#features">Review the feature distributions ↑</a>
    </div>
  );
}

function SubjectVariability() {
  const subjects = [...data.diagnostics.subject_metrics].sort((a, b) => a.macro_f1 - b.macro_f1);
  const min = subjects[0];
  const max = subjects[subjects.length - 1];
  const [selectedSubjectId, setSelectedSubjectId] = useState(subjects[Math.floor(subjects.length / 2)].subject);
  const selectedSubject = subjects.find((subject) => subject.subject === selectedSubjectId) ?? subjects[0];
  return (
    <div className="research-subjects">
      <div>
        <h3>Performance varies across subjects</h3>
        <p>Subject macro-F1 ranges from {min.macro_f1.toFixed(3)} to {max.macro_f1.toFixed(3)}.</p>
      </div>
      <div className="research-subject-strip" aria-label="Select a subject to inspect its macro-F1">
        {subjects.map((subject) => (
          <button
            type="button"
            key={subject.subject}
            className={subject.subject === selectedSubject.subject ? "is-selected" : undefined}
            style={{ height: `${Math.max(subject.macro_f1 * 100, 4)}%` }}
            title={`${subject.subject}: ${subject.macro_f1.toFixed(3)}`}
            aria-label={`${subject.subject}: macro-F1 ${subject.macro_f1.toFixed(3)}`}
            aria-pressed={subject.subject === selectedSubject.subject}
            onClick={() => setSelectedSubjectId(subject.subject)}
          />
        ))}
      </div>
      <div className="research-subject-selected" aria-live="polite">
        <strong>{selectedSubject.subject}</strong>
        <span>{selectedSubject.epochs.toLocaleString()} epochs</span>
        <span>accuracy {selectedSubject.accuracy.toFixed(3)}</span>
        <span>macro-F1 {selectedSubject.macro_f1.toFixed(3)}</span>
        <span>κ {selectedSubject.cohen_kappa.toFixed(3)}</span>
      </div>
      <div className="research-subject-range">
        <span>lowest <strong>{min.macro_f1.toFixed(3)}</strong></span>
        <span>Select a bar for subject metrics</span>
        <span>highest <strong>{max.macro_f1.toFixed(3)}</strong></span>
      </div>
    </div>
  );
}

function DiscussionExplorer() {
  return (
    <div className="research-discussion-body">
      <article>
        <h3>Model comparison</h3>
        <p>The feature ensemble reaches 0.608 mean macro-F1 across the outer folds. The other four models score between 0.654 and 0.683. A small second-stage model has the highest score in this set of experiments. Architecture comparisons on other datasets remain open.</p>
      </article>
      <article>
        <h3>N2 errors</h3>
        <p>N2 has the lowest class F1 at 0.469 and is often assigned to Wake or N3. The model may capture broad slow-wave structure more reliably than brief N2 events. Spindle features account for 7% of normalized feature importance. Next, I would replace or ablate the spindle and K-complex detectors and repeat the evaluation.</p>
      </article>
      <article>
        <h3>Between-subject variation</h3>
        <p>Subject-level macro-F1 ranges from 0.431 to 0.823. These results describe UCDDB only. External datasets are needed to assess other cohorts, recording systems, and clinical settings.</p>
      </article>
      <details className="research-limitations">
        <summary>Limitations and next experiments</summary>
        <ul>
          <li>Evaluate the locked pipeline on a second PSG dataset without retuning the test cohort.</li>
          <li>Report calibration and transition-specific performance alongside aggregate metrics.</li>
          <li>Relate difficult subjects to signal quality and per-stage support.</li>
          <li>Test improved spindle and K-complex detectors through controlled ablation.</li>
        </ul>
      </details>
    </div>
  );
}

function RelatedWork() {
  return (
    <div className="research-related-work">
      <h3>Related work</h3>
      <p>
        Automatic staging systems commonly combine an epoch representation with a sequence model. DeepSleepNet learns raw-EEG features with multiscale CNNs and then models transitions with bidirectional LSTMs; SeqSleepNet explicitly separates within-epoch encoding from across-epoch sequence modeling. <Citation number={17} /> <Citation number={18} />
      </p>
      <p>
        Published UCDDB studies use different channels, preprocessing, label mappings, and validation designs. Their methods provide context for this work; their headline scores are not placed on the model chart. <Citation number={5} /> <Citation number={8} />
      </p>
      <p className="research-related-position">All five models use the same folds and outcome metrics.</p>
    </div>
  );
}

const references = [
  {
    number: 1,
    citation: "Silber MH, Ancoli-Israel S, Bonnet MH, et al. The Visual Scoring of Sleep in Adults. Journal of Clinical Sleep Medicine. 2007;3:121–131.",
    href: "https://pubmed.ncbi.nlm.nih.gov/17557422/",
    note: "Adult visual-scoring recommendations, including the retained 30-second epoch and recommended EEG, EOG, and chin-EMG channels."
  },
  {
    number: 2,
    citation: "Shrivastava D, Jung S, Saadat M, Sirohi R, Crewson K. How to interpret the results of a sleep study. Journal of Community Hospital Internal Medicine Perspectives. 2014;4:24983.",
    href: "https://pmc.ncbi.nlm.nih.gov/articles/PMC4246141/",
    note: "Clinical interpretation of sleep time, sleep efficiency, latency, stage distribution, fragmentation, respiratory events, and related PSG outputs."
  },
  {
    number: 3,
    citation: "University College Dublin and St. Vincent’s University Hospital. Sleep Apnea Database v1.0.0. PhysioNet. 2007.",
    href: "https://physionet.org/content/ucddb/1.0.0/",
    note: "Primary dataset description, recorded channels, cohort, and R&K stage annotations."
  },
  {
    number: 4,
    citation: "Chambon S, Galtier MN, Arnal PJ, Wainrib G, Gramfort A. A Deep Learning Architecture for Temporal Sleep Stage Classification Using Multivariate and Multimodal Time Series. IEEE TNSRE. 2018;26:758–769.",
    href: "https://pubmed.ncbi.nlm.nih.gov/29641380/",
    note: "End-to-end multichannel staging and a direct experiment on how surrounding minutes affect an epoch prediction."
  },
  {
    number: 5,
    citation: "Längkvist M, Karlsson L, Loutfi A. Sleep Stage Classification Using Unsupervised Feature Learning. Advances in Artificial Neural Systems. 2012;2012:107046.",
    href: "https://doi.org/10.1155/2012/107046",
    note: "UCDDB comparison of handmade features and deep belief networks, with HMM sequence modeling and discussion of subject variation."
  },
  {
    number: 6,
    citation: "Wang S, Hua G, Hao G, Xie C. A Cycle Deep Belief Network Model for Multivariate Time Series Classification. Mathematical Problems in Engineering. 2017;2017:9549323.",
    href: "https://doi.org/10.1155/2017/9549323",
    note: "Temporal multivariate classification applied to UCDDB using the previous label in the current prediction."
  },
  {
    number: 7,
    citation: "Younes M, Kuna ST, Pack AI, Walsh JK, Kushida CA, Staley B, Pien GW. Staging Sleep in Polysomnograms: Analysis of Inter-Scorer Variability. Journal of Clinical Sleep Medicine. 2016;12:885–894.",
    href: "https://pubmed.ncbi.nlm.nih.gov/27070243/",
    note: "Primary analysis showing that much scoring disagreement comes from genuinely equivocal epochs, especially adjacent-stage boundaries."
  },
  {
    number: 8,
    citation: "Zhu T, Luo W, Yu F. Multi-Branch Convolutional Neural Network for Automatic Sleep Stage Classification with Embedded Stage Refinement and Residual Attention Channel Fusion. Sensors. 2020;20:6592.",
    href: "https://pubmed.ncbi.nlm.nih.gov/33218040/",
    note: "UCDDB and Sleep-EDFx evaluation of multichannel CNN feature extraction, learned refinement, and attention-based fusion."
  },
  {
    number: 9,
    citation: "Pei W, Li Y, Wen P, Yang F, Ji X. An automatic method using MFCC features for sleep stage classification. Brain Informatics. 2024;11:6.",
    href: "https://doi.org/10.1186/s40708-024-00219-w",
    note: "UCDDB and SHHS experiments using MFCC representations with a compact CNN–LSTM architecture."
  },
  {
    number: 10,
    citation: "Šušmáková K, Krakovská A. Discrimination ability of individual measures used in sleep stages classification. Artificial Intelligence in Medicine. 2008;44:261–277.",
    href: "https://doi.org/10.1016/j.artmed.2008.07.005",
    note: "Original comparison of 818 polysomnographic measurements; delta/beta power was the strongest single five-stage measure, but remained insufficient on its own."
  },
  {
    number: 11,
    citation: "Purcell SM, Manoach DS, Demanuele C, et al. Characterizing sleep spindles in 11,630 individuals from the National Sleep Research Resource. Nature Communications. 2017;8:15930.",
    href: "https://pmc.ncbi.nlm.nih.gov/articles/PMC5490197/",
    note: "Large original study of automated spindle and spectral measurements, their N2 context, variability, and measurement confounds."
  },
  {
    number: 12,
    citation: "Virkkala J, Hasan J, Värri A, Himanen SL, Müller K. Automatic sleep stage classification using two-channel electro-oculography. Journal of Neuroscience Methods. 2007;166:109–115.",
    href: "https://pubmed.ncbi.nlm.nih.gov/17681382/",
    note: "Subject-independent validation showing that slow-eye movements, spectral content, and two EOG channels carry substantial staging information."
  },
  {
    number: 13,
    citation: "Li Y, Xu Z, Zhang Y, Cao Z, Chen H. Automatic sleep stage classification based on a two-channel electrooculogram and one-channel electromyogram. Physiological Measurement. 2022;43:7.",
    href: "https://pubmed.ncbi.nlm.nih.gov/35487205/",
    note: "Original five-stage study quantifying how EOG and chin-EMG features complement EEG-based staging."
  },
  {
    number: 14,
    citation: "Silvani A, Ferri R, Lo Martire V, et al. Muscle Activity During Sleep in Human Subjects, Rats, and Mice: Towards Translational Models of REM Sleep Without Atonia. Sleep. 2017;40:4.",
    href: "https://pubmed.ncbi.nlm.nih.gov/28329117/",
    note: "Original analysis showing progressive reduction of normalized human chin EMG from wake to non-REM and REM."
  },
  {
    number: 15,
    citation: "Trinder J, Kleiman J, Carrington M, et al. Autonomic activity during human sleep as a function of time and sleep stage. Journal of Sleep Research. 2001;10:253–264.",
    href: "https://pubmed.ncbi.nlm.nih.gov/11903855/",
    note: "Repeated-measures study separating sleep-stage and time-of-night effects on heart rate and autonomic measures."
  },
  {
    number: 16,
    citation: "Fonseca P, van Gilst MM, Radha M, et al. Automatic sleep staging using heart rate variability, body movements, and recurrent neural networks in a sleep disordered population. Sleep. 2020;43:zsaa048.",
    href: "https://pubmed.ncbi.nlm.nih.gov/32249911/",
    note: "Independent validation of HRV-based sleep staging in a broad sleep-disordered cohort; supports cardiac data as useful secondary context."
  },
  {
    number: 17,
    citation: "Supratak A, Dong H, Wu C, Guo Y. DeepSleepNet: A Model for Automatic Sleep Stage Scoring Based on Raw Single-Channel EEG. IEEE Transactions on Neural Systems and Rehabilitation Engineering. 2017;25:1998–2008.",
    href: "https://pubmed.ncbi.nlm.nih.gov/28287969/",
    note: "Influential end-to-end baseline combining multiresolution CNN feature learning with bidirectional LSTM sequence modeling."
  },
  {
    number: 18,
    citation: "Phan H, Andreotti F, Cooray N, Chen OY, De Vos M. SeqSleepNet: End-to-End Hierarchical Recurrent Neural Network for Sequence-to-Sequence Automatic Sleep Staging. IEEE Transactions on Neural Systems and Rehabilitation Engineering. 2019;27:400–410.",
    href: "https://pmc.ncbi.nlm.nih.gov/articles/PMC6481557/",
    note: "Primary sequence-to-sequence study separating within-epoch encoding from longer-range sleep-stage dynamics."
  }
];

function ReferencesSection() {
  return (
    <section className="research-section research-references" id="references">
      <SectionHeading label="References" title="References">
        Clinical scoring rules, feature-family evidence, dataset documentation, and related models.
      </SectionHeading>
      <ol className="research-reference-list">
        {references.map((reference) => (
          <li id={`reference-${reference.number}`} key={reference.number}>
            <span>{reference.number}</span>
            <div><p>{reference.citation}</p><small>{reference.note}</small><a href={reference.href} target="_blank" rel="noreferrer">Open source ↗</a></div>
          </li>
        ))}
      </ol>
      <footer className="research-footer">
        <span>UCDDB sleep-staging experiments</span>
        <a href="#top">Back to top ↑</a>
      </footer>
    </section>
  );
}

export function StagingResearchDashboard() {
  return (
    <main className="research-page">
      <section className="research-hero" id="top">
        <div className="research-hero-copy">
          <h1>Automatic sleep staging from overnight polysomnography</h1>
        </div>
      </section>

      <section className="research-section" id="introduction">
        <SectionHeading label="Introduction" title="Introduction">
          Clinical context for the prediction task.
        </SectionHeading>
        <SleepStudyPrimer />
        <RelatedWork />
      </section>

      <section className="research-section" id="methods">
        <SectionHeading label="Methods" title="Methods">
          Data preparation and model evaluation.
        </SectionHeading>

        <div className="research-method-block">
          <div className="research-method-block-heading"><span>2.1</span><div><h3>Dataset and prediction task</h3><p>UCDDB provides the overnight signals and expert stage labels.</p></div></div>
          <CompactStudyProtocol />
        </div>

        <div className="research-method-block" id="features">
          <div className="research-method-block-heading"><span>2.2</span><div><h3>Feature representation</h3><p>Start with an expert label, then trace its targeted inputs from derivation to stage distribution and a real signal segment. The full classifier uses all 34 normalized measurements from EEG, EOG, chin EMG, and ECG.</p></div></div>
          <PhysiologyExplorer />
        </div>

        <div className="research-method-block">
          <div className="research-method-block-heading"><span>2.3</span><div><h3>Model comparison</h3><p>The five configurations test a feature ensemble, two compact second-stage models, a six-second TCN, and a hierarchical encoder.</p></div></div>
          <MethodPipeline />
          <details className="research-secondary-disclosure research-context-disclosure">
            <summary><span>Detail</span><strong>Context-window inputs</strong><small>Compare the information available to the 1-, 5-, and 9-epoch configurations</small></summary>
            <ContextWindowExplorer />
          </details>
        </div>

        <div className="research-method-block">
          <div className="research-method-block-heading"><span>2.4</span><div><h3>Cross-validation</h3><p>Models are compared with five-fold grouped cross-validation. All epochs from one subject remain in the same fold. Neural model selection uses training folds only. Class weights or weighted sampling address imbalance, and macro-F1 is the primary metric.</p></div></div>
        </div>
      </section>

      <section className="research-section" id="results">
        <SectionHeading label="Results" title="Results">
          Cross-validated performance by model, sleep stage, and subject.
        </SectionHeading>
        <div className="research-results-block">
          <div className="research-method-block-heading"><span>3.1</span><div><h3>Model comparison</h3><p>Select a model to compare its architecture, inputs, and result.</p></div></div>
          <ModelResults />
        </div>
        <div className="research-results-block">
          <div className="research-method-block-heading"><span>3.2</span><div><h3>Class-level errors</h3><p>The confusion matrix shows how predictions are distributed for each expert stage.</p></div></div>
          <ConfusionExplorer />
        </div>
        <details className="research-secondary-disclosure">
          <summary><span>3.3</span><strong>Additional diagnostics</strong><small>Feature-family importance and variation across subjects</small></summary>
          <div className="research-secondary-analysis">
            <FeatureImportance />
            <SubjectVariability />
          </div>
        </details>
      </section>

      <section className="research-section research-discussion" id="discussion">
        <SectionHeading label="Discussion" title="Discussion">
          Interpretation and limitations.
        </SectionHeading>
        <DiscussionExplorer />
      </section>
      <ReferencesSection />
    </main>
  );
}
