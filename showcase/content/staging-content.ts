import algorithmMetadata from "./generated/algorithm-metadata.json";
import conceptShowcase from "./generated/concept-showcase.json";
import { stagingDashboardData, type ModelResult } from "./staging-dashboard-data";
import { fmt, pct, r } from "../lib/results";

const data = stagingDashboardData;

export type FeatureSpec = {
  id: string;
  name: string;
  signal: string;
  summary: string;
  measurement: string;
  stageUse?: string;
};

export type FeatureFamily = {
  id: string;
  label: string;
  signal: string;
  summary: string;
  evidence: string;
  limitation: string;
  sources: number[];
  features: FeatureSpec[];
};

export type FeatureBasis = "aasm" | "published" | "adapted";

export type StageFeatureTarget = {
  stage: string;
  cue: string;
  featureIds: string[];
  expect: Record<string, "high" | "low">;
};

export type PublishedModel = {
  name: string;
  kind: string;
  macroF1?: number;
  stageF1?: number[];
  kappa?: number;
  accuracy?: number;
  size: string;
  learned?: number;
  learnedShort: string;
  trainShort: string;
  testShort: string;
  training: string;
  inputs: string;
  evaluation: string;
  cite?: number;
  ours?: boolean;
};

export type HistoryStep = {
  n: number;
  date: string;
  question: string;
  finding: string;
  decision: string;
  score?: number;
  holdout?: boolean;
};

// Band definitions imported directly from src/physionet_sleep/algorithms/ via scripts/export_showcase_metadata.py.
export const bandEdges: Record<string, string> = algorithmMetadata.epoch_ratio_bands;
export const trackBands: Record<string, string> = algorithmMetadata.track_bands;

const relativeBandUse: Record<string, string> = {
  theta: "Prominent at sleep onset and in N1.",
  beta: "Higher in wake and arousal."
};

// One-second tracks: each comes from a 2-second Hann-windowed spectrum stepped every second.
const trackMethod = "One value per second from a 2-second Hann-windowed spectrum, stepped every second.";
export const trackDerivation =
  "Before modelling, values are scaled by the night’s median and interquartile range, then averaged into six-second steps.";

const eegSpectrumFeatures: FeatureSpec[] = [
  {
    id: "spindle_features.aperiodic_exponent",
    name: "1/f slope",
    signal: "EEG",
    summary: "Slope of the 1/f background of the EEG spectrum.",
    measurement:
      "Negative slope of a linear fit of log spectral magnitude against log frequency over 2–7 Hz and 14–24 Hz, per 2-second window. This version uses different bands, and spectral magnitude in place of the power spectrum used in the cited method.",
    stageUse:
      "Background spectral shape is associated with arousal; this measurement is not a direct assay of excitation/inhibition balance."
  },
  {
    id: "spindle_features.aperiodic_intercept",
    name: "1/f offset",
    signal: "EEG",
    summary: "Overall level of the 1/f background.",
    measurement: "Intercept of the same log–log fit.",
    stageUse: "Rises with the larger, slower waves of stable NREM."
  },
  {
    id: "spindle_features.delta_power",
    name: "Delta magnitude",
    signal: "EEG",
    summary: "EEG spectral magnitude in 1–4 Hz.",
    measurement: "Median spectral magnitude across 1–4 Hz in a 2-second window, averaged over C3 and C4.",
    stageUse: "Rises with deeper NREM sleep."
  },
  ...["theta", "beta"].map((band) => ({
    id: `epoch_ratio_features.rel_${band}`,
    name: `${band[0].toUpperCase()}${band.slice(1)} share`,
    signal: "EEG",
    summary: `Share of the epoch's 1–30 Hz EEG power that falls in ${bandEdges[band]}.`,
    measurement:
      "Welch spectrum (4-second Hann segments) of each 30-second epoch, band power over 1–30 Hz power, averaged over C3 and C4. A ratio from the same channel cancels differences in amplifier gain and electrode contact between recordings.",
    stageUse: relativeBandUse[band]
  })),
  ...["theta", "alpha", "beta", "gamma"].map((band) => ({
    id: `spindle_features.${band}_residual_power`,
    name: `${band[0].toUpperCase()}${band.slice(1)} residual power`,
    signal: "EEG",
    summary: `${band[0].toUpperCase()}${band.slice(1)} power above the fitted 1/f background.`,
    measurement: `Median of log power minus the aperiodic fit across ${bandEdges[band]}.`
  })),
  ...([
    ["delta", trackBands.delta, "Rises through N2 into N3."],
    ["theta", trackBands.theta, "Rises at sleep onset; theta runs mark N1 and REM."],
    ["alpha", trackBands.alpha, "High in relaxed wake; second-long alpha bursts mark arousals."],
    ["beta", trackBands.beta, "High in wake and arousals."]
  ] as const).map(([band, edges, use]) => ({
    id: `track.c_rel_${band}`,
    name: `Central ${band} share`,
    signal: "EEG",
    summary: `Share of each second's 0.5–30 Hz central EEG power that falls in ${edges}.`,
    measurement: `${trackMethod} Band power over 0.5–30 Hz power, from the average of the C3 and C4 spectra.`,
    stageUse: use
  })),
  {
    id: "track.c_log_power",
    name: "Central EEG power",
    signal: "EEG",
    summary: "Total 0.5–30 Hz central EEG power each second, on a log scale.",
    measurement: `${trackMethod} log10 of 0.5–30 Hz power, C3 and C4 spectra averaged. Scaled per night (median and IQR) before modelling.`,
    stageUse: "High-voltage slow waves and K-complexes raise it; wake and REM are low-voltage."
  }
];

export const featureFamilies: FeatureFamily[] = [
  {
    id: "spectrum",
    label: "EEG spectrum",
    signal: "EEG",
    summary: "Measurements describe the central EEG spectrum: its background slope, overall power, and activity in different frequency bands.",
    evidence:
      "Sleep depth changes the shape of the EEG spectrum: delta tends to rise with deeper non-REM sleep while alpha falls. Spectral ratios can separate some stages strongly, but no single measure separates all five reliably.",
    limitation:
      "A spectral summary describes the whole window, so a K-complex and a burst of movement can raise the same band. Muscle activity inflates beta and gamma on the scalp.",
    sources: [1, 6],
    features: eegSpectrumFeatures
  },
  {
    id: "spindles",
    label: "Spindles",
    signal: "EEG",
    summary: "Spindle-band power and detected spindle bursts describe brief N2 activity.",
    evidence:
      "Sleep spindles are 0.5–2 second bursts of 11–16 Hz activity and the defining feature of N2. Averaged over 30 seconds a few spindles barely move the spindle-band share; at 1-second resolution each one is a separate peak. Model C includes both spindle-band share and a spindle-burst detector; the latter more directly targets brief events.",
    limitation:
      "Sigma-band power reflects spindle activity without detecting individual spindles. Alpha bursts at 11–12 Hz overlap the band, and spindle density falls with age.",
    sources: [1, 7],
    features: [
      {
        id: "spindle_bursts",
        name: "Spindle bursts",
        signal: "EEG",
        summary: "Seconds in the epoch with a spindle-band burst.",
        measurement: `${trackMethod} Per second, the ${trackBands.sigma} share divided by the ${trackBands.alpha} plus ${trackBands.beta} shares (C3 and C4 spectra averaged); a burst is a second more than two robust SDs above that night's median, counted per 30-second epoch.`,
        stageUse: "Several per epoch in N2 (and N3, where spindles also occur), rare in wake, N1 and REM."
      },
      {
        id: "track.c_rel_sigma",
        name: "Spindle-band share",
        signal: "EEG",
        summary: `Share of each second's 0.5–30 Hz central EEG power that falls in ${trackBands.sigma}.`,
        measurement: `${trackMethod} ${trackBands.sigma} power over 0.5–30 Hz power, C3 and C4 spectra averaged.`,
        stageUse: "Brief peaks during N2 spindles."
      }
    ]
  },
  {
    id: "topography",
    label: "Frontal slow waves / occipital alpha",
    signal: "EEG",
    summary: "Frontal slow waves and occipital alpha describe the location and amplitude of EEG staging cues.",
    evidence:
      "AASM scores N3 from slow waves of at least 75 µV at 0.5–2 Hz on frontal leads, and wake from posterior alpha that attenuates at sleep onset. Calibrated microvolts make the amplitude rule possible.",
    limitation:
      "Slow-wave amplitude varies with age. This automated coverage measure follows the AASM rule; it does not reproduce visual scoring exactly. Eyes-open wake shows little alpha.",
    sources: [1],
    features: [
      {
        id: "topography_features.frontal_slow_wave_fraction",
        name: "Frontal slow-wave fraction",
        signal: "EEG",
        summary: "Share of the epoch covered by 0.5–2 Hz waves of at least 75 µV peak-to-peak on F3/F4.",
        measurement:
          "Band-pass 0.5–2 Hz, split into full waves at upward zero crossings, keep waves of 0.5–2 s with peak-to-peak ≥ 75 µV; fraction of the epoch they cover, averaged over F3-M2 and F4-M1.",
        stageUse: "AASM scores N3 when this reaches 20%."
      },
      {
        id: "topography_features.occipital_rel_alpha",
        name: "Occipital alpha share",
        signal: "EEG",
        summary: `Share of occipital EEG power at ${trackBands.alpha}.`,
        measurement: `Welch spectrum (4-s segments) per epoch on O1-M2 and O2-M1, ${trackBands.alpha} over 1–30 Hz power, averaged.`,
        stageUse: "High in relaxed wake, drops out at sleep onset (N1)."
      },
      {
        id: "track.f_rel_delta",
        name: "Frontal delta share",
        signal: "EEG",
        summary: `Share of each second's 0.5–30 Hz frontal EEG power in ${trackBands.delta}.`,
        measurement: `${trackMethod} From the average of the F3 and F4 spectra.`,
        stageUse: "Highest in N3, where slow waves are largest frontally."
      },
      {
        id: "track.o_rel_alpha",
        name: "Occipital alpha track (1 s)",
        signal: "EEG",
        summary: `Share of each second's 0.5–30 Hz occipital EEG power in ${trackBands.alpha}.`,
        measurement: `${trackMethod} From the average of the O1 and O2 spectra.`,
        stageUse: "High in eyes-closed wake; returns briefly during arousals."
      }
    ]
  },
  {
    id: "eye-movement",
    label: "Eye movement",
    signal: "EOG",
    summary: "Eye activity and slow- and fast-band EOG power describe movement and the shape of the eye-channel spectrum.",
    evidence:
      "EOG carries slow-eye-movement evidence around sleep onset and rapid-eye-movement evidence in REM. Subject-independent studies have shown that two EOG channels alone contain substantial five-stage information.",
    limitation:
      "Eye movements also occur during wake, and EOG electrodes pick up frontal EEG, so N3 slow waves leak into the slow EOG band. This dataset records a single eye channel (E1-M2).",
    sources: [1, 8, 9],
    features: [
      {
        id: "eye_movement_activity.eye_movement_activity",
        name: "Eye-movement activity",
        signal: "EOG",
        summary: "EOG amplitude above its local baseline.",
        measurement: "0.1–5 Hz envelope minus its 30-second rolling median, floored at zero and smoothed over 1 second."
      },
      {
        id: "eye_movement_activity.eye_movement_baseline",
        name: "Eye-movement baseline",
        signal: "EOG",
        summary: "The local baseline itself.",
        measurement: "30-second rolling median of the 0.1–5 Hz EOG envelope."
      },
      {
        id: "epoch_ratio_features.eog_rem_vs_noise_db",
        name: "Eye-movement power",
        signal: "EOG",
        summary: "Eye-movement-range EOG power (0.5–5 Hz) relative to the same channel's 5–30 Hz floor, in decibels.",
        measurement: "One 30-second spectrum of the E1-M2 channel per epoch; mean magnitude 0.5–5 Hz over 5–30 Hz, in dB.",
        stageUse: "Rapid eye movements over a quiet, atonic floor."
      },
      {
        id: "epoch_ratio_features.eog_rem_vs_sem_db",
        name: "Fast-to-slow eye movement ratio",
        signal: "EOG",
        summary: "0.5–5 Hz against 0.1–0.5 Hz EOG power, in decibels.",
        measurement: "From the same per-epoch EOG spectrum.",
        stageUse: "Low when slow rolling eye movements dominate, as at sleep onset."
      },
      {
        id: "track.eog_low",
        name: "Slow EOG power",
        signal: "EOG",
        summary: "EOG power at 0.5–2 Hz each second, on a log scale.",
        measurement: `${trackMethod} log10 of 0.5–2 Hz power on E1-M2.`,
        stageUse: "Slow rolling eye movements at sleep onset; also frontal slow waves in N3."
      },
      {
        id: "track.eog_high",
        name: "Fast EOG power",
        signal: "EOG",
        summary: "EOG power at 2–8 Hz each second, on a log scale.",
        measurement: `${trackMethod} log10 of 2–8 Hz power on E1-M2.`,
        stageUse: "The sharp deflections of rapid eye movements in REM and blinks in wake."
      }
    ]
  },
  {
    id: "muscle-tone",
    label: "Muscle tone",
    signal: "Chin EMG",
    summary: "Chin-tone measurements describe muscle activity relative to the night’s quietest periods.",
    evidence:
      "Chin tone typically falls from wake through non-REM and is lowest in REM. Combined EOG and EMG features can recover much of the information used in multichannel staging.",
    limitation:
      "Movement, snoring, electrode contact, and REM sleep without atonia can break the expected pattern. Mains interference on the chin lead can mask atonia in a few recordings.",
    sources: [1, 9, 10],
    features: [
      {
        id: "epoch_ratio_features.emg_floor_db",
        name: "Chin muscle tone",
        signal: "Chin EMG",
        summary: "Chin EMG tone relative to the night's quietest epochs, in decibels.",
        measurement: "10·log10 of the epoch's 10–100 Hz tone over the 5th percentile of all epochs in the recording. No labels are used.",
        stageUse: "High in wake, lowest in REM."
      },
      {
        id: "track.emg_log_rms",
        name: "Chin EMG level",
        signal: "Chin EMG",
        summary: "RMS of 10–80 Hz chin EMG each second, on a log scale.",
        measurement: `${trackMethod} Half the log10 of 10–80 Hz power on Chin1-Chin2. Scaled per night before modelling, so it reads relative to that night.`,
        stageUse: "Drops at REM onset; brief bursts mark movements and arousals."
      }
    ]
  },
  {
    id: "heart",
    label: "Heart",
    signal: "ECG",
    summary: `Heart rate every second from R-peak detection, plus a 5-minute measure of heart-rate variability. Tested but not used in the final model: with EEG, EOG and chin EMG present, heart and breathing together add ${fmt("delta.cardio_with_eeg")} macro-F1.`,
    evidence:
      "Sleep changes the heart's rhythm as well as its rate. In stable NREM, and most of all N3, the rhythm is steady; in REM and wake it shows slow, correlated swings and surges. In an earlier 112-patient Boston development sub-cohort check (patients with both stages present), slow heart-rate swings were lower in N3 than in REM in 109 of 112 patients. The one-second heart-rate signal captures what those earlier 30-second heart features measured, along with their timing.",
    limitation:
      "A 5-minute window blurs stage boundaries. Age and sleep apnea flatten autonomic contrasts, and apnea-related heart-rate cycles can mimic REM-like variability. Ectopic beats and missed R-peaks create false jumps in the 1-second rate.",
    sources: [11, 14, 15],
    features: [
      {
        id: "hrv_features.hr_slow_sd_300s",
        name: "Slow heart-rate swings",
        signal: "ECG",
        summary: "How much the 10-second average heart rate wanders over 5 minutes.",
        measurement: "Standard deviation of 10-second mean heart rate across ±150 s (at least 20 of 30 bins), from cleaned beat intervals.",
        stageUse: "Lower in N3 than in REM in almost every subject with both stages."
      },
      {
        id: "track.hr",
        name: "Heart rate",
        signal: "ECG",
        summary: "Instantaneous heart rate at each second.",
        measurement: "Beat-to-beat rate (60 / RR) from Pan–Tompkins R-peaks, intervals of 0.3–2.0 s, linearly interpolated to the centre of each second.",
        stageUse: "Higher and more variable in wake and REM; surges mark arousals."
      }
    ]
  },
  {
    id: "breathing",
    label: "Breathing",
    signal: "Respiratory effort",
    summary: `Breath-timing regularity and heart–breathing coupling over 1–2 minutes, plus the effort-belt signal and its amplitude every second. Tested but not in the final model; without the EEG, heart and breathing alone reach ${fmt("noeeg.ensemble.macro_f1", 2)} macro-F1 with a whole-night model (S4).`,
    evidence:
      "Breathing becomes regular in stable NREM and irregular in REM and wake, and in stable NREM the heart rate follows breathing closely. In an earlier 119-patient Boston development sub-cohort check, breath depth was more regular in N3 than in REM in 113 of 119 patients outside scored respiratory events.",
    limitation:
      "Apneas, hypopneas, snoring, and body position change breathing directly. Effort belts measure movement, not airflow, and can slip.",
    sources: [16],
    features: [
      {
        id: "breath_regularity.breath_interval_cv_60s",
        name: "Breath timing variability",
        signal: "Respiratory effort",
        summary: "Variation of breath-to-breath intervals over 1 minute.",
        measurement:
          "Inspiratory peaks of the 0.1–0.7 Hz band-passed effort-belt sum, at least 1.5 s apart; coefficient of variation of intervals within ±30 s.",
        stageUse: "Regular in N3, irregular in REM and wake."
      },
      {
        id: "cardiorespiratory_coupling.hr_resp_coherence_120s",
        name: "Heart–breathing coupling",
        signal: "ECG + effort",
        summary: "How strongly heart rate is phase-locked to breathing over 2 minutes.",
        measurement:
          "Magnitude-squared coherence between the cleaned 4-Hz NN series and the effort signal at the dominant breathing frequency, 32-s Welch segments within ±60 s.",
        stageUse: "High in N3 and N2, lower in REM and wake."
      },
      {
        id: "track.resp_effort",
        name: "Breathing effort",
        signal: "Respiratory effort",
        summary: "The band-passed effort-belt sum, averaged over each second.",
        measurement: "Abdomen + chest belts, averaged to 10 Hz, band-passed 0.1–0.7 Hz (4th-order Butterworth, zero phase), then averaged per second.",
        stageUse: "The breathing waveform itself, so the model can see pauses and irregular breaths."
      },
      {
        id: "track.resp_env",
        name: "Breathing amplitude",
        signal: "Respiratory effort",
        summary: "Breath-by-breath amplitude of the same signal, on a log scale.",
        measurement: "log10 of the Hilbert envelope of the band-passed effort signal, averaged per second.",
        stageUse: "Steady in NREM, variable in REM and wake; drops with hypopneas."
      }
    ]
  }
];

export const basisLabels: Record<FeatureBasis, { short: string; long: string }> = {
  aasm: { short: "AASM rule", long: "Implements an AASM scoring criterion directly" },
  published: { short: "Published method", long: "Established measure from the literature, standard parameters" },
  adapted: { short: "Adapted", long: "Research-motivated idea with my own implementation choices" }
};

export const featureBasis: Record<string, { basis: FeatureBasis; mechanism: string; cite?: number }> = {
  "topography_features.frontal_slow_wave_fraction": { basis: "aasm", mechanism: "Cortical slow oscillation", cite: 1 },
  "topography_features.occipital_rel_alpha": { basis: "aasm", mechanism: "Posterior alpha rhythm", cite: 1 },
  "epoch_ratio_features.rel_theta": { basis: "aasm", mechanism: "Cortical slowing at sleep onset", cite: 1 },
  "spindle_features.delta_power": { basis: "aasm", mechanism: "Cortical slow oscillation", cite: 1 },
  "epoch_ratio_features.rel_sigma": { basis: "aasm", mechanism: "Thalamocortical spindles", cite: 1 },
  "epoch_ratio_features.emg_floor_db": { basis: "aasm", mechanism: "Brainstem muscle atonia", cite: 1 },
  "epoch_ratio_features.emg_baseline_floor_db": { basis: "aasm", mechanism: "Brainstem muscle atonia", cite: 1 },
  "spindle_features.aperiodic_exponent": { basis: "adapted", mechanism: "Background spectral shape and arousal", cite: 17 },
  "spindle_features.aperiodic_intercept": { basis: "published", mechanism: "Cortical arousal (broadband power)", cite: 17 },
  "epoch_ratio_features.rel_alpha": { basis: "published", mechanism: "Posterior alpha rhythm" },
  "epoch_ratio_features.rel_beta": { basis: "published", mechanism: "Cortical arousal" },
  "epoch_ratio_features.muscle_frac_db": { basis: "published", mechanism: "Scalp muscle activity", cite: 18 },
  "slow_wave_features.slow_wave_envelope": { basis: "published", mechanism: "Cortical slow oscillation", cite: 6 },
  "hrv_features.hr_bpm_30s": { basis: "published", mechanism: "Autonomic heart control", cite: 14 },
  "hrv_features.sdnn_150s": { basis: "published", mechanism: "Autonomic heart control", cite: 14 },
  "hrv_features.ln_lf_hf_300s": { basis: "published", mechanism: "Autonomic heart control", cite: 14 },
  "hrv_features.dfa_a2_300s": { basis: "published", mechanism: "Autonomic heart control (rhythm correlation)", cite: 15 },
  "breath_regularity.breath_interval_cv_60s": { basis: "published", mechanism: "Respiratory control", cite: 19 },
  "breath_regularity.breath_amp_cv_60s": { basis: "published", mechanism: "Respiratory control", cite: 19 },
  "cardiorespiratory_coupling.hr_resp_coherence_120s": { basis: "adapted", mechanism: "Heart–breathing coupling", cite: 16 },
  "hrv_features.hr_slow_sd_300s": { basis: "adapted", mechanism: "Autonomic heart control" },
  "hrv_features.hr_surge_30s": { basis: "adapted", mechanism: "Autonomic arousal response" },
  "spindle_features.sigma_residual_power": { basis: "adapted", mechanism: "Thalamocortical spindles" },
  "spindle_features.sigma_beta_residual_diff": { basis: "adapted", mechanism: "Thalamocortical spindles" },
  "epoch_ratio_features.spindle_rel": { basis: "adapted", mechanism: "Thalamocortical spindles" },
  "spindle_features.theta_residual_power": { basis: "adapted", mechanism: "Cortical slowing at sleep onset" },
  "spindle_features.alpha_residual_power": { basis: "adapted", mechanism: "Posterior alpha rhythm" },
  "spindle_features.beta_residual_power": { basis: "adapted", mechanism: "Cortical arousal" },
  "spindle_features.gamma_residual_power": { basis: "adapted", mechanism: "Scalp muscle activity" },
  "eye_movement_activity.eye_movement_activity": { basis: "adapted", mechanism: "Oculomotor (eye movements)" },
  "eye_movement_activity.eye_movement_baseline": { basis: "adapted", mechanism: "Oculomotor (eye movements)" },
  "epoch_ratio_features.eog_rem_vs_noise_db": { basis: "adapted", mechanism: "Oculomotor (rapid eye movements)" },
  "epoch_ratio_features.eog_rem_vs_sem_db": { basis: "adapted", mechanism: "Oculomotor (slow vs rapid eye movements)" },
  "track.c_rel_delta": { basis: "published", mechanism: "Cortical slow oscillation", cite: 6 },
  "track.c_rel_theta": { basis: "aasm", mechanism: "Cortical slowing at sleep onset", cite: 1 },
  "track.c_rel_alpha": { basis: "published", mechanism: "Posterior alpha rhythm" },
  "track.c_rel_sigma": { basis: "aasm", mechanism: "Thalamocortical spindles", cite: 1 },
  spindle_bursts: { basis: "adapted", mechanism: "Thalamocortical spindles", cite: 1 },
  "track.c_rel_beta": { basis: "published", mechanism: "Cortical arousal" },
  "track.c_log_power": { basis: "adapted", mechanism: "Cortical arousal (broadband power)" },
  "track.f_rel_delta": { basis: "aasm", mechanism: "Cortical slow oscillation", cite: 1 },
  "track.o_rel_alpha": { basis: "aasm", mechanism: "Posterior alpha rhythm", cite: 1 },
  "track.eog_low": { basis: "adapted", mechanism: "Oculomotor (slow eye movements)" },
  "track.eog_high": { basis: "adapted", mechanism: "Oculomotor (rapid eye movements)" },
  "track.emg_log_rms": { basis: "aasm", mechanism: "Brainstem muscle atonia", cite: 1 },
  "track.hr": { basis: "published", mechanism: "Autonomic heart control", cite: 14 },
  "track.resp_effort": { basis: "adapted", mechanism: "Respiratory control" },
  "track.resp_env": { basis: "adapted", mechanism: "Respiratory control" }
};

export const mechanismBiology: Array<{ key: string; text: string; cites: number[] }> = [
  {
    key: "Cortical slow oscillation",
    text: "In deep NREM, falling arousal signals from the brainstem let large areas of the cortex switch on and off in synchrony, about once a second. This synchronized cortical activity appears on the scalp as high-amplitude slow waves, largest frontally, and it increases with homeostatic sleep pressure.",
    cites: [20]
  },
  {
    key: "Thalamocortical spindles",
    text: "Spindles are 11–16 Hz bursts generated by the thalamus and relayed to the cortex once sleep has begun. They are thought to gate sensory input and stabilize sleep, and they are the defining feature of N2, fading as N3 deepens.",
    cites: [20]
  },
  {
    key: "Posterior alpha rhythm",
    text: "Alpha (defined by AASM as 8–13 Hz; sampled here at 8–12 Hz or 8–11 Hz to stay clear of spindles) is the idling rhythm of the visual cortex during relaxed, eyes-closed wakefulness. It attenuates with drowsiness, and its loss marks the wake-to-N1 transition.",
    cites: [1]
  },
  {
    key: "Cortical slowing at sleep onset",
    text: "As arousal systems disengage at sleep onset, cortical activity slows from alpha toward theta (4–7 Hz) before spindles or slow waves appear. AASM defines N1 by this low-voltage, theta-dominant pattern.",
    cites: [1]
  },
  {
    key: "Cortical arousal",
    text: "High-frequency (30–45 Hz) aperiodic 1/f slope is thought to reflect cortical excitation/inhibition balance and is steepest in REM sleep. Here, we fit lower frequencies (2–7 and 14–24 Hz) on spectral magnitude, so slow-wave power makes the slope steepest in N3 rather than REM.",
    cites: [21, 17]
  },
  {
    key: "Scalp muscle activity",
    text: "Above about 20 Hz, scalp EEG is dominated by electromyographic activity from scalp, jaw and neck muscles. Muscle tone falls with sleep, so this high-frequency power marks wakefulness, arousals and movement.",
    cites: [18]
  },
  {
    key: "Oculomotor",
    text: "Rapid eye movements are saccade-like movements generated by brainstem circuits during REM sleep's phasic periods. Slow rolling eye movements appear at sleep onset as voluntary oculomotor control lapses. The EOG detects both through the eye's electrical dipole.",
    cites: [22, 1]
  },
  {
    key: "Brainstem muscle atonia",
    text: "REM sleep includes active, brainstem-driven suppression of skeletal muscle tone (REM atonia), which keeps us from acting out dreams. Chin EMG reaches its lowest level of the night; loss of atonia is the hallmark of REM sleep behaviour disorder.",
    cites: [22]
  },
  {
    key: "Autonomic",
    text: "Autonomic balance shifts with sleep stage. Through NREM, sympathetic activity falls and parasympathetic (vagal) control of the heart dominates, so heart rate slows and becomes steady. REM brings sympathetic surges back toward waking levels, producing an irregular rhythm with slow fluctuations.",
    cites: [23, 11]
  },
  {
    key: "Respiratory control",
    text: "In NREM, breathing is under automatic metabolic control by chemoreceptors sensing CO2 and oxygen, so it becomes regular in rate and depth. In REM, behavioural and cortical inputs return and intercostal muscle tone is lost, making breathing irregular.",
    cites: [19]
  },
  {
    key: "Heart–breathing coupling",
    text: "Respiratory sinus arrhythmia is the vagally mediated rise in heart rate on inspiration and fall on expiration. It is strongest when breathing is regular and vagal tone is high, as in stable NREM, and weakens in REM and fragmented sleep.",
    cites: [16]
  }
];

export function mechanismFor(featureId: string) {
  const label = featureBasis[featureId]?.mechanism;
  return label ? mechanismBiology.find((item) => label.startsWith(item.key)) : undefined;
}

export const stageFeatureTargets: StageFeatureTarget[] = [
  {
    stage: "Wake",
    cue: "Wake shows posterior alpha when the eyes are closed, high chin tone, fast EEG activity and a faster heart rate.",
    featureIds: [
      "topography_features.occipital_rel_alpha",
      "track.o_rel_alpha",
      "track.c_rel_alpha",
      "epoch_ratio_features.rel_beta",
      "track.c_rel_beta",
      "epoch_ratio_features.emg_floor_db",
      "track.hr"
    ],
    expect: {
      "topography_features.occipital_rel_alpha": "high",
      "track.o_rel_alpha": "high",
      "track.c_rel_alpha": "high",
      "epoch_ratio_features.rel_beta": "high",
      "track.c_rel_beta": "high",
      "epoch_ratio_features.emg_floor_db": "high",
      "track.hr": "high"
    }
  },
  {
    stage: "N1",
    cue: "Sleep onset slows the EEG toward theta, brings slow rolling eye movements, and leaves breathing less regular than in deeper sleep.",
    featureIds: [
      "epoch_ratio_features.rel_theta",
      "track.c_rel_theta",
      "spindle_features.aperiodic_exponent",
      "epoch_ratio_features.eog_rem_vs_sem_db",
      "breath_regularity.breath_interval_cv_60s"
    ],
    expect: {
      "epoch_ratio_features.rel_theta": "high",
      "track.c_rel_theta": "high",
      "spindle_features.aperiodic_exponent": "low",
      "epoch_ratio_features.eog_rem_vs_sem_db": "low",
      "breath_regularity.breath_interval_cv_60s": "high"
    }
  },
  {
    stage: "N2",
    cue: "N2 is identified by sleep spindles and K-complexes. Both are brief events, so event detection complements average band power. Heart rate also follows breathing closely in stable NREM.",
    featureIds: ["spindle_bursts", "track.c_rel_sigma", "cardiorespiratory_coupling.hr_resp_coherence_120s"],
    expect: {
      spindle_bursts: "high",
      "track.c_rel_sigma": "high",
      "cardiorespiratory_coupling.hr_resp_coherence_120s": "high"
    }
  },
  {
    stage: "N3",
    cue: "N3 is defined by high-amplitude frontal slow waves. Heart rate and breathing are also at their most regular.",
    featureIds: [
      "topography_features.frontal_slow_wave_fraction",
      "track.f_rel_delta",
      "spindle_features.delta_power",
      "track.c_log_power",
      "hrv_features.hr_slow_sd_300s"
    ],
    expect: {
      "topography_features.frontal_slow_wave_fraction": "high",
      "track.f_rel_delta": "high",
      "spindle_features.delta_power": "high",
      "track.c_log_power": "high",
      "hrv_features.hr_slow_sd_300s": "low"
    }
  },
  {
    stage: "REM",
    cue: "REM is identified by rapid eye movements with low chin tone, though frontal slow waves can leak into the single eye channel. Heart rate becomes less coupled to breathing and shows slower fluctuations.",
    featureIds: [
      "track.c_log_power",
      "epoch_ratio_features.rel_theta",
      "epoch_ratio_features.eog_rem_vs_noise_db",
      "eye_movement_activity.eye_movement_activity",
      "eye_movement_activity.eye_movement_baseline",
      "track.eog_high",
      "epoch_ratio_features.emg_floor_db",
      "cardiorespiratory_coupling.hr_resp_coherence_120s",
      "track.resp_env"
    ],
    expect: {
      "track.c_log_power": "low",
      "epoch_ratio_features.rel_theta": "high",
      "epoch_ratio_features.eog_rem_vs_noise_db": "high",
      "eye_movement_activity.eye_movement_activity": "high",
      "eye_movement_activity.eye_movement_baseline": "high",
      "track.eog_high": "high",
      "epoch_ratio_features.emg_floor_db": "low",
      "cardiorespiratory_coupling.hr_resp_coherence_120s": "low",
      "track.resp_env": "low"
    }
  }
];

export const codeAnchor: Record<string, string | undefined> = {
  spectrum: "ratios",
  spindles: "tracks",
  topography: undefined,
  "eye-movement": "eye-movement",
  "muscle-tone": "muscle-tone",
  heart: "heart",
  breathing: "breathing"
};

export const featureUnits: Record<string, string> = {
  "spindle_features.aperiodic_exponent": "negative log-magnitude slope (unitless)",
  "spindle_features.aperiodic_intercept": "log₁₀ power at the 1/f fit",
  "spindle_features.delta_power": "spectral magnitude, 1–4 Hz",
  "spindle_features.theta_residual_power": "log₁₀ power above the 1/f fit",
  "spindle_features.alpha_residual_power": "log₁₀ power above the 1/f fit",
  "spindle_features.beta_residual_power": "log₁₀ power above the 1/f fit",
  "spindle_features.gamma_residual_power": "log₁₀ power above the 1/f fit",
  "epoch_ratio_features.rel_theta": "fraction of 1–30 Hz power",
  "epoch_ratio_features.rel_beta": "fraction of 1–30 Hz power",
  "track.c_rel_delta": "fraction of 0.5–30 Hz power",
  "track.c_rel_theta": "fraction of 0.5–30 Hz power",
  "track.c_rel_alpha": "fraction of 0.5–30 Hz power",
  "track.c_rel_sigma": "fraction of 0.5–30 Hz power",
  spindle_bursts: "seconds per epoch",
  "track.c_rel_beta": "fraction of 0.5–30 Hz power",
  "track.f_rel_delta": "fraction of 0.5–30 Hz power",
  "track.o_rel_alpha": "fraction of 0.5–30 Hz power",
  "topography_features.occipital_rel_alpha": "fraction of occipital power",
  "topography_features.frontal_slow_wave_fraction": "fraction of the epoch",
  "track.c_log_power": "log₁₀ power, 0.5–30 Hz",
  "track.eog_low": "log₁₀ power, 0.5–2 Hz",
  "track.eog_high": "log₁₀ power, 2–8 Hz",
  "track.emg_log_rms": "log₁₀ RMS amplitude",
  "epoch_ratio_features.eog_rem_vs_noise_db": "dB",
  "epoch_ratio_features.eog_rem_vs_sem_db": "dB",
  "epoch_ratio_features.emg_floor_db": "dB above the night's quietest epochs",
  "eye_movement_activity.eye_movement_activity": "µV above the local baseline",
  "eye_movement_activity.eye_movement_baseline": "µV",
  "track.hr": "beats per minute",
  "hrv_features.hr_slow_sd_300s": "beats per minute (SD)",
  "breath_regularity.breath_interval_cv_60s": "coefficient of variation",
  "cardiorespiratory_coupling.hr_resp_coherence_120s": "coherence (0–1)",
  "track.resp_effort": "belt signal, arbitrary units",
  "track.resp_env": "log₁₀ belt amplitude"
};

export const modelCFeatureIds = new Set(data.signal_importance.map((item) => item.id));

export const signalInfo: Record<string, { name: string; family: string }> = Object.fromEntries(
  featureFamilies.flatMap((family) =>
    family.features.map((feature) => [feature.id.split(".").pop() as string, { name: feature.name, family: family.id }])
  )
);

export const stagePrimerRows: Array<{ system: string; signal: string; scoring: boolean; cells: string[]; cites: number[] }> = [
  {
    system: "Brain",
    signal: "EEG",
    scoring: true,
    cells: [
      `Occipital alpha (8–13 Hz AASM; ${trackBands.alpha} here) with eyes closed; faster beta when alert.`,
      "Alpha gives way to slower, low-voltage theta (4–7 Hz).",
      `Sleep spindles (${trackBands.sigma}) and K-complexes on a theta background.`,
      "High-amplitude slow waves (0.5–2 Hz, ≥ 75 µV) over ≥ 20% of the epoch, largest frontally.",
      "Low-voltage and wake-like."
    ],
    cites: [1, 20]
  },
  {
    system: "Eyes",
    signal: "EOG",
    scoring: true,
    cells: [
      "Eye blinks and rapid eye movements during wakefulness.",
      "Slow eye movements (SEMs): smooth, rolling movements at sleep onset.",
      "Usually still.",
      "Still.",
      "Rapid eye movements (REMs): irregular, sharply peaked movements; may be absent during tonic REM sleep."
    ],
    cites: [1, 22]
  },
  {
    system: "Chin muscle",
    signal: "EMG",
    scoring: true,
    cells: [
      "Relatively high tonic chin EMG activity.",
      "Tonic chin EMG activity usually decreases from wakefulness.",
      "Reduced tonic chin EMG activity.",
      "Variable tonic chin EMG activity; low tone alone does not define N3.",
      "Muscle atonia: very low tonic chin EMG activity, with possible brief phasic twitches."
    ],
    cites: [1, 22]
  },
  {
    system: "Heart",
    signal: "ECG",
    scoring: false,
    cells: [
      "Fastest; sympathetic drive highest.",
      "Starts to slow.",
      "Slower and steadier as vagal (parasympathetic) control rises.",
      "Slowest and steadiest; strongest respiratory sinus arrhythmia.",
      "Irregular, with sympathetic surges and slow swings."
    ],
    cites: [23, 15]
  },
  {
    system: "Breathing",
    signal: "Effort belts",
    scoring: false,
    cells: [
      "Irregular, under voluntary and behavioural control.",
      "Can wax and wane as control shifts.",
      "Regular, under automatic chemoreceptor control.",
      "Most regular.",
      "Irregular; intercostal muscle tone is lost."
    ],
    cites: [19]
  }
];

export const reductionLabels: Record<string, string> = {
  prune6: "Remove the six least-used inputs (21 signals)",
  prune6_sigratio: "… and spindle-band ratio for spindle-band share",
  prune6_bursts: "… plus spindle bursts",
  nocardio27_bursts: "27 signals plus spindle bursts",
  swapA: "Spindle bursts + K-complex proxy for 1/f offset + gamma residual",
  swapB: "Cleaned eye signal for eye baseline + fast EOG",
  swapAB: "Both of the above",
  dropAB: "Control: the four inputs removed, nothing added",
  swapS1: "Spindle bursts for gamma residual + eye baseline",
  swapS2: "Spindle bursts for 1/f offset + gamma residual (adopted as C)",
  swapS3: "As S1, with spindle-band ratio",
  swap4_saw: "Sawtooth theta for gamma residual",
  swap4_alpha: "Alpha dropout for occipital alpha track",
  swap4_combo: "Spindle, sawtooth and alpha swaps together"
};

export const uSleepWeights = 3_100_000;

export function publishedModels(): PublishedModel[] {
  const adopted = data.models.find((model) => model.id === "adopted") as ModelResult;
  const size = data.model_size;
  const stage = (model: ModelResult) => data.source.stages.map((item) => model.stage_f1[item]);
  return [
    {
      name: `D · Concept-BiGRU (#${conceptShowcase.active_experiment}, this page)`,
      kind: "4 additive physiological stems + 1-layer BiGRU + Mondrian conformal",
      ours: true,
      macroF1: conceptShowcase.calibrated.mean_fold_macro_f1,
      stageF1: data.source.stages.map((item) => conceptShowcase.calibrated.stage_f1[item as keyof typeof conceptShowcase.calibrated.stage_f1]),
      kappa: conceptShowcase.calibrated.kappa,
      size: `${conceptShowcase.model_params.toLocaleString()} weights`,
      learned: conceptShowcase.model_params,
      learnedShort: `${Math.round(conceptShowcase.model_params / 1000)}k`,
      trainShort: "704 per fold",
      testShort: `5-fold CV, ${data.source.subjects} Boston (+ Holdout ${conceptShowcase.external_eval.holdout.n_subjects} & Coimbra ${conceptShowcase.external_eval.isruc.scorer1.n_subjects} in §3.3)`,
      training: "230 inner-train + ~26 early-stop + 474 fit-only Boston patients/fold; ~25 min/fold CPU",
      inputs: `${conceptShowcase.input_dim} inputs (${conceptShowcase.n_features} single-epoch features: 9 six-second tracks + 4 AASM markers)`,
      evaluation: `5-fold CV (${data.source.subjects} Boston), + ${conceptShowcase.external_eval.holdout.n_subjects} Boston holdout & ${conceptShowcase.external_eval.isruc.scorer1.n_subjects} Coimbra`
    },
    {
      name: "C · Compact trees (earlier baseline)",
      kind: "Gradient-boosted trees",
      macroF1: adopted.metrics.macro_f1,
      stageF1: stage(adopted),
      kappa: adopted.metrics.cohen_kappa,
      accuracy: adopted.metrics.accuracy,
      size: `${size.trees} trees, ${size.splits.toLocaleString()} splits on ${size.inputs_used} of ${size.inputs} named inputs`,
      learned: size.nodes,
      learnedShort: size.nodes.toLocaleString(),
      trainShort: `${size.train_subjects} per fold`,
      testShort: `5-fold CV, ${data.source.subjects} Boston`,
      training: `${size.train_subjects} patients per fold; ${Math.round(size.fit_seconds / 60)} min on a laptop CPU`,
      inputs: "20 EEG, eye and chin measurements, every 6 s over 150 s",
      evaluation: `5-fold CV, ${data.source.subjects} Boston records`
    },
    {
      name: "U-Sleep",
      kind: "Fully convolutional U-Net",
      macroF1: r("lit.usleep.macro_f1"),
      stageF1: ["Wake", "N1", "N2", "N3", "REM"].map((s) => r(`lit.usleep.f1_${s}`)),
      size: "≈3.1 million weights",
      learned: uSleepWeights,
      learnedShort: "≈3.1 M weights",
      trainShort: "Training subsets of 13 datasets",
      testShort: "100 other Boston patients",
      training: "Training subsets of 13 datasets; 15,660 participants across all training and evaluation datasets",
      inputs: "Raw EEG + EOG at 128 Hz",
      evaluation: "100 held-out Boston records",
      cite: 26
    },
    {
      name: "U-Time",
      kind: "Fully convolutional U-Net",
      macroF1: r("lit.utime.macro_f1"),
      size: "≈1.1 million weights",
      learned: 1.1e6,
      learnedShort: "≈1.1 M weights",
      trainShort: "Boston only (one model per dataset, 7 in the paper)",
      testShort: "Other Boston",
      training: "Trained per dataset",
      inputs: "Raw EEG",
      evaluation: "Boston, more records than U-Sleep's test set",
      cite: 27
    },
    {
      name: "SLEEPNET",
      kind: "Recurrent network",
      kappa: r("lit.sleepnet.kappa"),
      accuracy: r("lit.sleepnet.accuracy"),
      size: "Deep RNN (not reported)",
      learnedShort: "Not reported",
      trainShort: "≈10,000 MGH",
      testShort: "1,000 MGH",
      training: "≈10,000 MGH patients",
      inputs: "6 EEG channels",
      evaluation: "1,000 MGH recordings",
      cite: 28
    },
    {
      name: "Human scorers (AASM program)",
      kind: "Reference",
      accuracy: r("lit.aasm.agreement"),
      stageF1: undefined,
      size: "Agreement with the majority consensus of 2,500+ scorers",
      learnedShort: "—",
      trainShort: "2,500+ scorers",
      testShort: "1,800 epochs (9 excerpts)",
      training: "Most with 3 or more years of experience",
      inputs: "Full PSG, visual scoring",
      evaluation: "1,800 epochs from 9 short excerpts",
      cite: 29
    }
  ];
}

export const sweepLabels: Record<string, { text: string; dx: number; dy: number }> = {
  tree_d4: { text: "1 tree, depth 4", dx: 8, dy: 4 },
  tree_d6: { text: "depth 6", dx: 8, dy: 4 },
  tree_d8: { text: "depth 8", dx: 8, dy: 4 },
  hgb_l2_i200: { text: "single splits", dx: -6, dy: 16 },
  hgb_l4_i100: { text: "4-leaf", dx: 8, dy: 12 },
  hgb_l8_i100: { text: "8-leaf", dx: -8, dy: -6 },
  hgb_l31_i100: { text: "C · 5-fold mean", dx: 8, dy: -8 }
};

export const historyPhases: Array<{ phase: string; dataset: string; steps: HistoryStep[] }> = [
  {
    phase: "A first pipeline",
    dataset: "UCDDB · 25 apnea patients",
    steps: [
      { n: 1, date: "Aug 13", question: "Can named features stage sleep?", finding: "Feature recipe for EEG, EOG, chin, ECG and breathing; per-task reference diagnostics.", decision: "Build on hand-made, named features." },
      { n: 2, date: "Aug 14", question: "How good is an epoch classifier?", finding: `Bagged trees ${fmt("exp2.bagged_trees")}; with HSMM smoothing ${fmt("exp2.hsmm")}.`, decision: "Keep a temporal second stage.", score: r("exp2.hsmm") },
      { n: 3, date: "Aug 21", question: "Do sequence networks beat smoothing?", finding: `30-s CNN and 6-s TCN about ${fmt("exp3.sequence.low")}–${fmt("exp3.sequence.high")}; feature and signal experts ${fmt("exp3.experts.low")}–${fmt("exp3.experts.high")}.`, decision: "Smoothing stays ahead with 25 subjects." },
      { n: 4, date: "Aug 22", question: "Does a hierarchical network help?", finding: `${fmt("exp4.raw")} raw, ${fmt("exp4.linear_head")} with a linear head.`, decision: "No gain over smoothing." },
      { n: 5, date: "Sep 9", question: "Publish the first dashboard.", finding: "Bagged trees plus smoothers on UCDDB.", decision: "Dashboard v1 on GitHub Pages." }
    ]
  },
  {
    phase: "Audit: bugs before models",
    dataset: "UCDDB",
    steps: [
      { n: 6, date: "Oct 16", question: "Audit the dashboard for this talk.", finding: `Best smoother ${fmt("exp6.best_smoother")}; copy overstated results.`, decision: "Audit models and features in parallel.", score: r("exp6.best_smoother") },
      { n: 7, date: "Oct 16", question: `Why stuck near ${fmt("exp7.ceiling")}?`, finding: "REM and N1 stage codes were swapped.", decision: "Fix the label map." },
      { n: 8, date: "Oct 17", question: "Physiology or a bug?", finding: "18 of 29 features drifted against the labels, up to 6.5 minutes by morning (binned by sample count, not time).", decision: "Bin features by time." },
      { n: 9, date: "Oct 17", question: "Is the learner the bottleneck?", finding: `Default gradient boosting ${fmt("exp9.hgb")} vs the inherited bagged trees ${fmt("exp9.bagged_trees")}.`, decision: "Default boosting, never tuned." },
      { n: 10, date: "Oct 17", question: "Is the second stage leaking?", finding: "Smoother trained on in-sample probabilities.", decision: "Out-of-fold probabilities only." },
      { n: 11, date: "Oct 17", question: "Does amplifier gain limit us?", finding: `Per-night scaling already cancels it; gain-free ratios add ${fmt("exp11.ratio_gain", 3, true)}.`, decision: "Add ratio features." },
      { n: 12, date: "Oct 17", question: "Why no heart-rate effect?", finding: "R-peaks were fine; cleaned intervals and rhythm-structure measures separate N3 from REM.", decision: "Add heart, breathing and coupling features." },
      { n: 13, date: "Oct 17", question: "Why do some patients fail?", finding: "Flat recording dropouts were scored as Wake.", decision: "Mask dropouts." },
      { n: 14, date: "Oct 17", question: "Result of the fixes?", finding: `${fmt("exp14.before")} → ${fmt("exp14.after")}, still limited by 25 R&K-scored subjects.`, decision: "Find a bigger dataset.", score: r("exp14.after") }
    ]
  },
  {
    phase: "A better dataset",
    dataset: "Boston · 120 → 320 patients",
    steps: [
      { n: 15, date: "Oct 17", question: "Move to Boston?", finding: `${fmt("exp15.boston120")} on 120 records; calibrated µV, frontal and occipital leads, AASM labels. Trained on UCDDB, tested on Boston: ${fmt("exp15.ucddb_to_boston")}.`, decision: "Switch to Boston.", score: r("exp15.boston120") },
      { n: 16, date: "Oct 17", question: "Better model or more data?", finding: "Learning curve still rising; tuning gave nothing.", decision: "Download to 320 patients." },
      { n: 17, date: "Oct 17", question: "Can I cut features?", finding: `33 → 16 features at no loss with a decoder; new event features ${fmt("exp17.event_features", 3, true)}.`, decision: `A feature must add ≥ ${fmt("protocol.adoption_gain")} or replace two.` },
      { n: 18, date: "Oct 17", question: "Why do Boston patients fail?", finding: "Dead leads, ECG artefact in the EEG, stages with few epochs.", decision: "Report stage counts with per-patient scores." }
    ]
  },
  {
    phase: "Context, then simplify",
    dataset: "Boston · 320 patients",
    steps: [
      { n: 19, date: "Oct 18", question: "Does real context beat smoothing?", finding: `1-s tracks and 6-s features over 150 s: ${fmt("cv.window33.hmm.macro_f1")} with a decoder vs ${fmt("exp19.epoch16_hmm")}.`, decision: "Adopt the window model; retire smoothers and networks.", score: r("cv.window33.hmm.macro_f1") },
      { n: 20, date: "Oct 18", question: "What does a network on the same signals do?", finding: `Attention CNN ${((data.models.find((m) => m.id === "cnn") as ModelResult).metrics.macro_f1).toFixed(3)}; relies on the same physiology.`, decision: "Keep as the black-box comparison." },
      { n: 21, date: "Oct 19", question: "Is the start of the epoch special?", finding: `No: a tree tie-break artifact. Evidence peaks 6–18 s in; the previous epoch counts ${fmt("exp21.prev_next_ratio")}× the next.`, decision: "Report the corrected profile." },
      { n: 22, date: "Oct 19", question: "How does the model compare?", finding: `U-Sleep ${fmt("lit.usleep.macro_f1")}; human scorers agree on ${pct("lit.aasm.agreement")} of epochs.`, decision: "Add the comparison section." },
      { n: 23, date: "Oct 21", question: "Is the decoder still needed?", finding: `${fmt("delta.decoder_with_window", 3, true)} with the window, and it costs N1.`, decision: "Drop it: one idea, fewer parts.", score: r("cv.window33.epoch.macro_f1") },
      { n: 24, date: "Oct 21", question: "Do heart and breathing earn their place?", finding: `Removing them costs ${fmt("delta.cardio_with_eeg")}; alone (6 signals) they reach ${fmt("range.cardio_alone.low", 2)}–${fmt("range.cardio_alone.high", 2)}.`, decision: "Kept for now; their stand-alone value is the no-EEG case." },
      { n: 25, date: "Oct 21", question: "How simple can the learner be?", finding: `Boosted single splits (an additive model) ${fmt("exp25.stumps")} with a tenth of the values; one readable depth-4 tree ${fmt("exp25.tree_d4")}.`, decision: "Keep default boosting; show the sweep and the readable tree." },
      { n: 26, date: "Oct 21", question: "Can heart and breathing alone do better?", finding: `All 12 heart and breathing signals over 10.5 min: ${fmt("cv.cardiofull_c10.hmm.macro_f1")}, four-class κ ${fmt("cv.cardiofull_c10.hmm.kappa4", 2)} (published heart-rate models ${fmt("lit.cardio_kappa4.low")}–${fmt("lit.cardio_kappa4.high")} with whole-night networks).`, decision: "Longer context helps; the rest needs whole-night models and more data." },
      { n: 27, date: "Oct 21", question: "Does the window model transfer to another lab?", finding: `Trained in Boston on the central EEG, EOG and chin signals both labs share: ${fmt("ucddb_transfer.d27.transfer_cinc_to_ucddb.macro_f1")} on Dublin (κ ${fmt("ucddb_transfer.d27.transfer_cinc_to_ucddb.kappa", 2)}), against ${fmt("exp27.best_ucddb_trained")} for the best model ever trained on Dublin.`, decision: "Transfer holds; per-night scaling makes the signals portable." },
      { n: 28, date: "Oct 21", question: "Keep heart and breathing in the final model?", finding: `They add ${fmt("delta.cardio_with_eeg")} with EEG, eyes and chin present.`, decision: `Drop them: 27 signals, ${fmt("cv.nocardio27.epoch.macro_f1")}.`, score: r("cv.nocardio27.epoch.macro_f1") },
      { n: 29, date: "Oct 21", question: "Do a cleaner lead or dropping bad segments help?", finding: `No: ${fmt("exp29.clean_lead")} vs ${fmt("cv.nocardio27.epoch.macro_f1", 4)}; excluding 1.6–3.3% of epochs leaves the score unchanged.`, decision: "Keep all epochs; the model already copes." },
      { n: 30, date: "Oct 21", question: "Final test on 200 held-out patients (look #1)", finding: `${fmt("holdout.look1.nocardio27.macro_f1")} for the adopted model; every model drops by about the same amount on this harder, apnea-heavy group. One record that had failed to build was repaired and the same batch rerun (moved by ${fmt("holdout.look1.rerun_shift", 4)}).`, decision: "Reported as is; no choice made from it.", score: r("holdout.look1.nocardio27.macro_f1"), holdout: true },
      { n: 31, date: "Oct 22", question: "How far can heart and breathing go without the EEG?", finding: `Whole-night networks with breath-by-breath breathing and oxygen (8 networks averaged post hoc): ${fmt("noeeg.ensemble.macro_f1")} (four-class κ ${fmt("noeeg.ensemble.kappa4", 2)}), up from ${fmt("cv.cardiofull_c10.hmm.macro_f1")}. ECG + oximeter ${fmt("noeeg.gru_ecgonly_spo2_pos.epoch.macro_f1")}, belts ${fmt("noeeg.gru_beltonly_pos.epoch.macro_f1")}, ECG ${fmt("noeeg.tcn_ecgonly_pos.epoch.macro_f1")}.`, decision: "Reported as the wearable outlook; the final model stays EEG-based." }
    ]
  },
  {
    phase: "Interpretability audit",
    dataset: "Boston · 320 patients",
    steps: [
      { n: 32, date: "Oct 22", question: "Do the features behave as designed?", finding: `Each feature checked against its intended stage: N3 and wake features hold; the spindle-band share does not separate N2 (d ${fmt("effect.c_rel_sigma.d_N2", 2)}); the single EOG channel also carries frontal slow waves.`, decision: "Keep the published designs; show observed behaviour beside each (2.2)." },
      { n: 33, date: "Oct 22", question: "Can the model lose inputs without losing accuracy?", finding: `Removing the six least-used inputs: ${fmt("cv.prune6.epoch.macro_f1", 4)} vs ${fmt("cv.nocardio27.epoch.macro_f1", 4)} with 21 signals. Spindle measures added nothing (${fmt("delta.spindle_additions_best", 4, true)} at best).`, decision: "Keep the 21-signal set; look for more direct markers.", score: r("cv.prune6.epoch.macro_f1") },
      { n: 34, date: "Oct 22", question: "Can vague inputs be swapped for direct markers?", finding: "Swaps passed the accuracy rule, but the K-complex proxy ran the wrong way and the cleaned eye signal barely separated REM.", decision: "Check what a feature measures before testing it." },
      { n: 35, date: "Oct 22", question: "Which candidate markers mean what they claim?", finding: "Validated first: sawtooth theta (REM) and alpha dropout (N1) passed; REM eye bursts, K-complex events and slow eye movements failed.", decision: "Only validated markers enter the swaps." },
      { n: 36, date: "Oct 22", question: "Swap in the validated markers?", finding: `Passed the rule: spindle bursts for 1/f offset and gamma residual (${fmt("delta.swap_spindle", 4)}), sawtooth theta (${fmt("exp36.swap_sawtooth", 4)}), alpha dropout (${fmt("exp36.swap_alpha_dropout", 4)}). Only spindle bursts is used by the model (3rd-largest N2 importance); the others rank last.`, decision: "Adopt the spindle swap: model C, 20 signals.", score: r("cv.g20.epoch.macro_f1") },
      { n: 37, date: "Oct 22", question: "Final test of model C (look #2)", finding: `${fmt("holdout.look2.g20.macro_f1")} on the same 200 held-out patients (27-signal model at look #1: ${fmt("holdout.look1.nocardio27.macro_f1")}).`, decision: "Reported as is; no choice made from it.", score: r("holdout.look2.g20.macro_f1"), holdout: true },
      { n: 38, date: "Oct 22", question: "Does final Model C transfer to Coimbra?", finding: `Scored once on 99 nights: macro-F1 ${fmt("coimbra.transfer_cinc_to_isruc_scorer1.macro_f1")} against scorer 1 and ${fmt("coimbra.transfer_cinc_to_isruc_scorer2.macro_f1")} against scorer 2; scorer-to-scorer agreement ${fmt("coimbra.isruc_scorer1_vs_scorer2.macro_f1")}. Night 8 excluded before scoring because frontal EEG was missing.`, decision: "Report both scorer comparisons and patient variation as external validation." },
      { n: 39, date: "Nov 21", question: "Can Model C drop two broad spectral tracks?", finding: `An 18-signal tree model scored ${fmt("dev.waveform.reduction.macro_f1", 4)} macro-F1 vs ${fmt("cv.g20.epoch.macro_f1", 4)} for Model C on the same five development folds.`, decision: "Kept as a compact comparison; extraction still builds the 20-signal set.", score: r("dev.waveform.reduction.macro_f1") },
      { n: 40, date: "Nov 21", question: "How well does a ResNet learn from raw waveforms?", finding: `On the same development folds, a CPU-trained ResNet scored ${fmt("dev.waveform.scratch.macro_f1", 4)} macro-F1 vs ${fmt("cv.g20.epoch.macro_f1", 4)} for Model C.`, decision: "Kept as a raw-waveform baseline; well below the feature trees.", score: r("dev.waveform.scratch.macro_f1") },
      { n: 41, date: "Nov 21", question: "Does Model C still benefit from more training patients?", finding: `Training on 128 patients scored ${fmt("dev.subject_budget.train128.macro_f1", 4)} vs ${fmt("dev.subject_budget.full.macro_f1", 4)} with the full fold training split.`, decision: "More training patients still helped modestly; expansion beyond 320 tested in #73." },
      { n: 42, date: "Nov 21", question: "Do waveform probabilities complement the 18-signal tree?", finding: `Blending waveform and tree probabilities reached ${fmt("dev.waveform.blend.macro_f1", 4)} (${fmt("dev.waveform.blend.delta", 4, true)} over the 18-signal tree), improving all five folds.`, decision: "Motivated targeted error audits to see what waveform cues the tree missed.", score: r("dev.waveform.blend.macro_f1") },
      { n: 43, date: "Nov 21", question: "Which features weaken on sustained tree errors?", finding: "Frontal slow-wave coverage softened on sustained N3 misses, and sigma-band variants conflicted across stages.", decision: "Defined physiologically bounded feature tuning checks." },
      { n: 44, date: "Nov 21", question: "Can tuned feature settings improve stage separation?", finding: "A lower slow-wave amplitude threshold (with lead-quality pooling) and 30-second occipital theta qualified in all five training folds; sigma and short-window EOG candidates did not.", decision: "Evaluated the two qualifying features jointly in #45." },
      { n: 45, date: "Nov 21", question: "Do the tuned features improve staging together?", finding: `The joint recipe scored ${fmt("dev.features.joint.macro_f1", 4)} macro-F1 (${fmt("dev.features.joint.delta", 4, true)} vs the 18-signal tree) on the same development folds.`, decision: "Too small to adopt over Model C." },
      { n: 46, date: "Nov 21", question: "Can the tuned model use 16 inputs?", finding: `The 16-input recipe scored ${fmt("dev.features.reduced.macro_f1", 4)} macro-F1 (${fmt("dev.features.reduced.delta", 4, true)} vs the 18-signal tree).`, decision: "Too much drop; kept 20 signals." },
      { n: 47, date: "Nov 21", question: "Does longer context help a compact feature sequence model?", finding: `Longer context scored ${fmt("dev.context.long.macro_f1", 4)} vs ${fmt("dev.context.short.macro_f1", 4)} for the matched short-window control.`, decision: "No gain on its own." },
      { n: 48, date: "Nov 21", question: "Does a long-context sequence model complement the trees?", finding: `Blending the long-context sequence model with the 18-signal tree reached ${fmt("dev.context.long_blend.macro_f1", 4)} (${fmt("dev.context.long_blend.delta", 4, true)}).`, decision: "Logged as a sequence-blend candidate while keeping Model C as the single-model baseline.", score: r("dev.context.long_blend.macro_f1") },
      { n: 49, date: "Nov 21", question: "Where do sequence probabilities fix or break tree predictions?", finding: `Sequence log loss was ${fmt("dev.probability.sequence.log_loss", 4)} vs ${fmt("dev.probability.tree.log_loss", 4)} for the 18-signal tree.`, decision: "Tracked probability calibration separately from hard stage accuracy." },
      { n: 50, date: "Nov 21", question: "Can an additive student match tree probabilities?", finding: `Additive log loss was ${fmt("dev.probability.hard.log_loss", 4)} on hard labels and ${fmt("dev.probability.soft.log_loss", 4)} with soft targets.`, decision: "Useful calibration reference, not competitive with full trees." },
      { n: 51, date: "Nov 21", question: "Does physiological night history add beyond elapsed time?", finding: `Slow-wave history and chin/EOG continuity reached log loss ${fmt("dev.probability.physiology.log_loss", 4)} vs ${fmt("dev.probability.clock.log_loss", 4)} for a pure elapsed-time control.`, decision: "Low chin tone across neighboring epochs carried the REM continuity signal; slow-wave history added little for N2/N3." },
      { n: 52, date: "Nov 21", question: "Do unified training scripts reproduce the archived baselines?", finding: "Replayed the archived baseline and extended-window core across matched patient splits and feature tables.", decision: "Used the verified pipeline for subsequent ablations." },
      { n: 53, date: "Nov 21", question: "Does adaptive feature-and-context search beat Model C?", finding: `The searched recipe scored ${fmt("dev.recipe_search.searched.macro_f1", 4)} macro-F1; the same features with fixed context scored ${fmt("dev.recipe_search.fixed_context.macro_f1", 4)}.`, decision: "Fell short of adoption; kept the archived tree baseline." },
      { n: 54, date: "Nov 21", question: "How did the searched recipe shift errors?", finding: "New errors concentrated in N2 misclassified as N1 or N3; partial prior correction recovered N2 at the cost of N1 recall.", decision: "Separated class-prior tuning from feature gaps." },
      { n: 55, date: "Nov 21", question: "Which interactions or class weights drive those shifts?", finding: "Tempered class weights traded N1 recall for N2 recall, while longer-context trees pulled N3 split gain into distant relative beta.", decision: "Diagnostic finding; kept weights and context bounds fixed." },
      { n: 56, date: "Nov 21", question: "Can additive or pairwise boosting match full trees?", finding: `Additive boosting scored ${fmt("dev.glassbox.additive.macro_f1", 4)} macro-F1 and pairwise boosting scored ${fmt("dev.glassbox.pairwise.macro_f1", 4)}, trailing full trees in every fold.`, decision: "Kept standard gradient-boosted trees." },
      { n: 57, date: "Nov 21", question: "What characterizes sustained N2 errors?", finding: "N2 persistently called N3 had high delta share with intermediate slow-wave coverage; N2 persistently called N1 lacked spindle bursts.", decision: "Targeted N2/N3 and N2/N1 pairs with separate cues." },
      { n: 58, date: "Nov 21", question: "Can learned gates regulate past and future context?", finding: `Learned gates scored ${fmt("dev.gated_temporal.gated.macro_f1", 4)} vs ${fmt("dev.gated_temporal.ungated.macro_f1", 4)} with fixed unit gates; local heads still leaned toward the same N2 confusions.`, decision: "Used the score decomposition diagnostically; not promoted." },
      { n: 59, date: "Nov 21", question: "Does a pair-specific margin penalty fix N2/N3 or N2/N1?", finding: `The margin penalty dropped macro-F1 to ${fmt("dev.gated_temporal.gated_pair.macro_f1", 4)} and increased errors on both pairs.`, decision: "Reverted to standard class-weighted cross-entropy." },
      { n: 60, date: "Nov 21", question: "Do complementary slow-wave and event cues separate hard N2 cases?", finding: `Hard-case AUC rose from ${fmt("dev.complementary.n2_n3.original.hard_auc", 4)} to ${fmt("dev.complementary.n2_n3.complementary.hard_auc", 4)} on N2/N3 and from ${fmt("dev.complementary.n2_n1.original.hard_auc", 4)} to ${fmt("dev.complementary.n2_n1.complementary.hard_auc", 4)} on N2/N1.`, decision: "Tested both cue groups together in a full model (#61)." },
      { n: 61, date: "Nov 21", question: "Do both complementary cue groups improve the full gated classifier?", finding: `The expanded model scored ${fmt("dev.complementary.expanded.macro_f1", 4)} vs ${fmt("dev.gated_temporal.gated.macro_f1", 4)}, reducing both N2 error pairs while slightly lowering N1 recall.`, decision: "Not promoted; N1 recall traded off." },
      { n: 62, date: "Nov 21", question: "Can post-hoc adjustments recover N1 without losing κ?", finding: `Expansion raised κ from ${fmt("dev.targeted.gated18.kappa", 4)} to ${fmt("dev.targeted.expanded.kappa", 4)}, while both decision adjustments lowered κ without fixing N1.`, decision: "Kept original decisions." },
      { n: 63, date: "Nov 21", question: "What do K-like shape and prior event history contribute?", finding: "Neutralizing either input lowered hard N1/N2 ranking and κ, affecting N1→N2 and N2→N1 errors asymmetrically.", decision: "Evaluated stage-pair head sensitivity in #65–#67." },
      { n: 64, date: "Nov 21", question: "Does the interruption cue align with expert-marked arousals?", finding: "Only two folds qualified, and the cue fired in most epochs.", decision: "Dropped as an arousal proxy; kept separate event histories." },
      { n: 65, date: "Nov 21", question: "Can a fitted N1/N2 head improve N1?", finding: `The fitted pair head reached κ ${fmt("dev.stage_pair.head.kappa", 4)}, but N1 recall still fell.`, decision: "Rejected under the N1 recall constraint." },
      { n: 66, date: "Nov 21", question: "Does direct K-like shape help the N1/N2 head?", finding: "Shape shifted errors between N1 and N2, but the archived detector had low precision against annotated DREAMS K-complexes.", decision: "Kept the existing count as a coarse shape proxy." },
      { n: 67, date: "Nov 21", question: "Does splitting spindle and K-like history help N2 persistence?", finding: "Separating those two histories barely changed the matched head.", decision: "Kept pooled event history." },
      { n: 68, date: "Nov 21", question: "Can the expanded model drop redundant slow-wave inputs?", finding: `A 21-signal version scored ${fmt("dev.stage_pair.compact.macro_f1", 4)} (κ ${fmt("dev.stage_pair.compact.kappa", 4)}), missing the fold-level floor on fold 2.`, decision: "Kept 23 signals in that experimental branch." },
      { n: 69, date: "Nov 21", question: "Where do Wake and N1 errors occur?", finding: "Mid-night awakenings and re-entry transitions cause far more Wake/N1 errors than initial sleep onset: local cues miss brief N1, while past context can override sudden awakenings.", decision: "Tested awakening-specific cues in #71–#72." },
      { n: 70, date: "Nov 21", question: "How sensitive is the branch to fitting-set size?", finding: `128 fitting patients scored ${fmt("dev.current_curve.train_128.macro_f1", 4)} vs ${fmt("dev.current_curve.full.macro_f1", 4)} with all 230–231 fitting patients per fold.`, decision: "Standardized on full fold splits." },
      { n: 71, date: "Nov 21", question: "Does within-epoch theta timing separate Wake and N1?", finding: `Mean hard-case AUC moved from ${fmt("dev.wake_n1.timing.mean_only.auc", 4)} to ${fmt("dev.wake_n1.timing.mean_and_timing.auc", 4)}.`, decision: "Dropped the two tested timing features." },
      { n: 72, date: "Nov 21", question: "Can an awakening gate reduce Wake called N1?", finding: `It cut Wake-called-N1 errors by ${pct("dev.wake_n1.followup.alpha_past_gate.target_error_reduction")} while shifting errors toward N1-called-Wake (N1 F1 ${fmt("dev.wake_n1.followup.alpha_past_gate.n1_f1_delta", 4)}).`, decision: "Kept the frozen reference for #73." },
      { n: 73, date: "Nov 21", question: "Does adding 474 more Boston training patients improve staging?", finding: `Mean macro-F1 rose from ${fmt("dev.additional_training.reference.macro_f1", 4)} to ${fmt("dev.additional_training.candidate.macro_f1", 4)}, and hard Wake/N1 AUC rose from ${fmt("dev.additional_training.original.hard_wake_n1_auc", 3)} to ${fmt("dev.additional_training.expanded.hard_wake_n1_auc", 3)}.`, decision: "Expanded 704-patient pool retained for unified sequence and tree models (#75–#80)." }
    ]
  },
  {
    phase: "Concept-additive sequence & conformal uncertainty",
    dataset: "704 fitting patients per fold; evaluated strictly on fixed 320 Boston dev patients",
    steps: [
      { n: 74, date: "Nov 21", question: "Do prior boundary fixes (#23, #65–#67, #72) stack?", finding: "Auditing boundary errors showed threshold adjustments traded N1 against N2/Wake rather than compounding cleanly.", decision: "Replaced threshold stacking with unified 704-patient sequence modeling (#75)." },
      { n: 75, date: "Nov 21", question: "Does unifying occipital theta, sigma, night history and complementary markers in a whole-night BiGRU pass 0.8 macro-F1?", finding: `Calibrated macro-F1 reached ${fmt("dev.bigru75.calibrated.macro_f1", 4)} (κ ${fmt("dev.bigru75.calibrated.kappa", 4)}, log loss ${fmt("dev.bigru75.calibrated.log_loss", 4)}), beating #73 by ${fmt("dev.bigru75.delta_vs_73", 4)}.`, decision: "Validated unified cues; next tested simplifying and factorizing the model (#79).", score: r("dev.bigru75.calibrated.macro_f1") },
      { n: 76, date: "Nov 21", question: "Can tempered multi-scale gradient-boosted trees match on the same 704-patient pool?", finding: `Calibrated macro-F1 reached ${fmt("dev.hgb704.calibrated.macro_f1", 4)} (κ ${fmt("dev.hgb704.calibrated.kappa", 4)}, N3 ${fmt("dev.hgb704.calibrated.f1_N3", 4)}).`, decision: "Retained as the top tabular comparator alongside neural sequence models.", score: r("dev.hgb704.calibrated.macro_f1") },
      { n: 77, date: "Nov 21", question: "Does a sub-epoch raw-waveform BiGRU outperform feature-engineered models?", finding: "Paused on hold after fold 0 (macro-F1 below feature-based #75/#79) due to multi-hour CPU runtime per fold.", decision: "Kept on hold in favor of interpretable, compact engineered-feature sequence models (#79–#80)." },
      { n: 78, date: "Nov 21", question: "What is the 704-patient multi-family ceiling?", finding: `Out-of-fold log-odds ensembling across #75, #76 and archived sequence/tree models reached ${fmt("dev.ensemble78.best.macro_f1", 4)} macro-F1 (κ ${fmt("dev.ensemble78.best.kappa", 4)}).`, decision: "Used as a ceiling reference while refining a single deployable model.", score: r("dev.ensemble78.best.macro_f1") },
      { n: 79, date: "Nov 21", question: "Can a 4× smaller concept-additive BiGRU match #75 with conformal uncertainty?", finding: `Pruning duplicates (120→81 inputs) and separating local evidence into 4 additive organ/band heads reached ${fmt("dev.concept79.calibrated.macro_f1", 4)} macro-F1 (κ ${fmt("dev.concept79.calibrated.kappa", 4)}, N3 ${fmt("dev.concept79.calibrated.f1_N3", 4)}) with 90%/95% Mondrian conformal sets (${fmt("dev.concept79.conformal.coverage_90.mondrian.set_size", 2)} mean stages at 90% coverage).`, decision: "Initial 29-feature Concept-BiGRU prototype (61.6k weights); later pruned to 13 single-epoch features in #82.", score: r("dev.concept79.calibrated.macro_f1") },
      { n: 80, date: "Nov 22", question: "Do Cohen's d-screened missing physiological cues (EMG, HR, guarded EOG bursts, delta, resp_env) improve #79?", finding: `Training-fold screening rejected respiratory effort belt (${fmt("dev.cue_screen.resp_effort.max_abs_d", 3)} max |d|) and qualified chin EMG (${fmt("dev.cue_screen.emg_log_rms.N1_vs_REM", 3)} N1 vs REM), F3/F4-guarded EOG bursts (${fmt("dev.cue_screen.eog_burst_guarded.N2_vs_REM", 3)} N2 vs REM) and HR (${fmt("dev.cue_screen.hr.max_abs_d", 3)} max |d|, ${fmt("dev.cue_screen.hr.Wake_vs_N1", 3)} Wake vs N1). Relative delta showed no Fold-0 gain over existing slow-wave markers; combining winning REM/EMG + 5th autonomic branch across all 5 folds reached ${fmt("dev.cues80.calibrated.macro_f1", 4)} cal (${fmt("dev.cues80.raw.macro_f1", 4)} raw, κ ${fmt("dev.cues80.calibrated.kappa", 4)}, log loss ${fmt("dev.cues80.calibrated.log_loss", 4)}, N1 ${fmt("dev.cues80.calibrated.f1_N1", 4)}).`, decision: "Retained in Supplementary: matched #79 within sampling variation while losing N1 slightly, adding 26% weights and requiring ECG/respiratory channels.", score: r("dev.cues80.calibrated.macro_f1") },
      { n: 81, date: "Nov 22", holdout: true, question: "External evaluation (Look #3) of the 29-feature Concept-BiGRU prototype (#79) on Boston 200 holdout & Coimbra 99 ISRUC nights", finding: `Boston 200-patient holdout reached ${fmt("ext.concept79.holdout.calibrated.macro_f1", 4)} calibrated macro-F1 (κ ${fmt("ext.concept79.holdout.calibrated.kappa", 4)}, ${fmt("delta.concept79_holdout_vs_model_c", 4, true)} vs Model C; 90%/95% conformal ${pct("ext.concept79.holdout.conformal.coverage_90.coverage")} / ${pct("ext.concept79.holdout.conformal.coverage_95.coverage")} coverage). Coimbra 99 nights reached ${fmt("ext.concept79.isruc.calibrated_scorer1.macro_f1", 4)} (scorer 1) and ${fmt("ext.concept79.isruc.calibrated_scorer2.macro_f1", 4)} (scorer 2).`, decision: "Superseded by the 13-feature single-epoch pruned Model D (#82).", score: r("ext.concept79.holdout.calibrated.macro_f1") },
      { n: 82, date: "Nov 22", holdout: true, question: "Prune Concept-BiGRU to 13 single-epoch features (49 inputs, zero rolling/clock shortcuts) & evaluate on Boston holdout (Look #4) and Coimbra", finding: `5-fold CV reached ${fmt("dev.minimal82.calibrated.macro_f1", 4)} macro-F1 (κ ${fmt("dev.minimal82.calibrated.kappa", 4)}, pure single-epoch local ${fmt("dev.minimal82.local_only_macro_f1", 4)} + BiGRU ${fmt("dev.minimal82.temporal_gain", 4, true)}, 90% conformal ${fmt("dev.minimal82.conformal.coverage_90.mondrian.set_size", 2)} stages). Locked external eval: Boston 200 holdout ${fmt("ext.minimal82.holdout.calibrated.macro_f1", 4)} (κ ${fmt("ext.minimal82.holdout.calibrated.kappa", 4)}, acc ${pct("ext.minimal82.holdout.calibrated.accuracy")}, ${fmt("delta.minimal82_holdout_vs_model_c", 4, true)} vs Model C); Coimbra 99 nights ${fmt("ext.minimal82.isruc.calibrated_scorer1.macro_f1", 4)} (S1) / ${fmt("ext.minimal82.isruc.calibrated_scorer2.macro_f1", 4)} (S2).`, decision: "Promoted as adopted Model D (fewer features than Model A and Model C, 61.1k weights).", score: r("ext.minimal82.holdout.calibrated.macro_f1") }
    ]
  }
];

export const historyWhere: Record<number, string> = {
  1: "artifacts/reference_diagnostics",
  2: "artifacts/paper_matched_benchmarks_5fold_hsmm",
  3: "artifacts/sequence_cnn_*",
  4: "artifacts/hierarchical_epoch_staging",
  5: "git 9c671ea",
  6: "artifacts/staging_diagnostics",
  7: "exploration/coordinator",
  8: "exploration/model-audit",
  9: "exploration/coordinator",
  10: "exploration/model-audit",
  11: "exploration/ratio-normalization",
  12: "exploration/hrv-features",
  13: "exploration/subject-failures",
  14: "exploration/coordinator",
  15: "exploration/cinc2018",
  16: "exploration/cinc-models",
  17: "exploration/cinc-features",
  18: "exploration/cinc-subjects",
  19: "exploration/cinc-context",
  20: "exploration/cinc-context",
  21: "scripts/run_window_time_importance.py",
  22: "dashboard comparison",
  23: "scripts/run_window_hmm_staging.py",
  24: "staging_window_model/ablations",
  25: "scripts/run_window_complexity.py",
  26: "staging_window_model/cardio_full*",
  27: "scripts/run_window_transfer.py",
  28: "staging_window_model/ablations",
  29: "exploration/signal-quality",
  30: "holdout/LEDGER.csv",
  31: "exploration/no-eeg",
  32: "showcase explorer (2.2)",
  33: "staging_window_model/cinc2018/feature_reduction",
  34: "feature_reduction (round 2)",
  35: "feature_reduction/round4_validation.csv",
  36: "feature_reduction (rounds 3–4)",
  37: "holdout/LEDGER.csv",
  38: "staging_window_model/transfer_isruc/FINDINGS.md; single_look.json",
  39: "exploration/cinc-reduction18/FINDINGS.md",
  40: "exploration/cinc-waveform/FINDINGS.md",
  41: "exploration/cinc-cap-audit/FINDINGS.md",
  42: "exploration/cinc-cap-audit/FUSION.md",
  43: "exploration/cinc-cap-audit/FEATURE_FAILURES.md",
  44: "exploration/cinc-feature-tuning/FINDINGS.md",
  45: "exploration/cinc-feature-tuning/FINDINGS.md",
  46: "exploration/cinc-feature-tuning/FINDINGS.md",
  47: "exploration/cinc-feature-context/FINDINGS.md",
  48: "exploration/cinc-feature-context/FINDINGS.md",
  49: "exploration/cinc-probability-diagnostic/FINDINGS.md",
  50: "exploration/cinc-probability-diagnostic/FINDINGS.md",
  51: "exploration/cinc-probability-diagnostic/FINDINGS.md",
  52: "exploration/cinc-recipe-search/FINDINGS.md",
  53: "exploration/cinc-recipe-search/FINDINGS.md",
  54: "exploration/cinc-error-interactions/FINDINGS.md",
  55: "exploration/cinc-error-interactions/FINDINGS.md",
  56: "exploration/cinc-error-interactions/FINDINGS.md",
  57: "exploration/cinc-gated-temporal/FINDINGS.md",
  58: "exploration/cinc-gated-temporal/FINDINGS.md",
  59: "exploration/cinc-gated-temporal/FINDINGS.md",
  60: "exploration/cinc-complementary-markers/FINDINGS.md",
  61: "exploration/cinc-complementary-markers/FINDINGS.md",
  62: "exploration/cinc-targeted-refinement/FINDINGS.md",
  63: "exploration/cinc-targeted-refinement/FINDINGS.md",
  64: "exploration/cinc-stage-pair-revisit/FINDINGS.md",
  65: "exploration/cinc-stage-pair-revisit/FINDINGS.md",
  66: "exploration/cinc-stage-pair-revisit/FINDINGS.md",
  67: "exploration/cinc-stage-pair-revisit/FINDINGS.md",
  68: "exploration/cinc-stage-pair-revisit/FINDINGS.md",
  69: "exploration/cinc-wake-n1-data/FINDINGS.md",
  70: "exploration/cinc-wake-n1-data/FINDINGS.md",
  71: "exploration/cinc-wake-n1-followup/FINDINGS.md",
  72: "exploration/cinc-wake-n1-followup/FINDINGS.md",
  73: "exploration/cinc-all-additional-training/FINDINGS.md",
  74: "exploration/cinc-combination-audit/summary.json",
  75: "exploration/cinc-bigru-unified/FINDINGS.md",
  76: "exploration/cinc-hgb-704/FINDINGS.md",
  77: "exploration/cinc-waveform-subepoch/FINDINGS.md",
  78: "exploration/cinc-unified-ensemble/FINDINGS.md",
  79: "exploration/cinc-bigru-concept/FINDINGS.md",
  80: "exploration/cinc-bigru-cues/FINDINGS.md",
  81: "exploration/cinc-bigru-concept/external_eval/single_look.json",
  82: "exploration/cinc-bigru-pruned/FINDINGS.md"
};

export const references = [
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
    citation: "Younes M, Raneri J, Hanly P. Staging Sleep in Polysomnograms: Analysis of Inter-Scorer Variability. Journal of Clinical Sleep Medicine. 2016;12:885–894.",
    href: "https://pubmed.ncbi.nlm.nih.gov/27070243/",
    note: "Primary analysis showing that much scoring disagreement comes from genuinely equivocal epochs, especially adjacent-stage boundaries."
  },
  {
    number: 6,
    citation: "Šušmáková K, Krakovská A. Discrimination ability of individual measures used in sleep stages classification. Artificial Intelligence in Medicine. 2008;44:261–277.",
    href: "https://doi.org/10.1016/j.artmed.2008.07.005",
    note: "Original comparison of 818 polysomnographic measurements; delta/beta power achieved ~1% error separating wake from deep sleep (S4), though five-stage classification required multiple cues."
  },
  {
    number: 7,
    citation: "Purcell SM, Manoach DS, Demanuele C, et al. Characterizing sleep spindles in 11,630 individuals from the National Sleep Research Resource. Nature Communications. 2017;8:15930.",
    href: "https://pmc.ncbi.nlm.nih.gov/articles/PMC5490197/",
    note: "Large original study of automated spindle and spectral measurements, their N2 context, variability, and measurement confounds."
  },
  {
    number: 8,
    citation: "Virkkala J, Hasan J, Värri A, Himanen SL, Müller K. Automatic sleep stage classification using two-channel electro-oculography. Journal of Neuroscience Methods. 2007;166:109–115.",
    href: "https://pubmed.ncbi.nlm.nih.gov/17681382/",
    note: "Subject-independent validation showing that slow-eye movements, spectral content, and two EOG channels carry substantial staging information."
  },
  {
    number: 9,
    citation: "Li Y, Xu Z, Zhang Y, Cao Z, Chen H. Automatic sleep stage classification based on a two-channel electrooculogram and one-channel electromyogram. Physiological Measurement. 2022;43:7.",
    href: "https://pubmed.ncbi.nlm.nih.gov/35487205/",
    note: "Evaluates EEG-free five-stage classification from two EOG channels and one chin-EMG channel (81% accuracy) against full PSG (85% accuracy)."
  },
  {
    number: 10,
    citation: "Silvani A, Ferri R, Lo Martire V, et al. Muscle Activity During Sleep in Human Subjects, Rats, and Mice: Towards Translational Models of REM Sleep Without Atonia. Sleep. 2017;40:4.",
    href: "https://pubmed.ncbi.nlm.nih.gov/28329117/",
    note: "Original analysis showing progressive reduction of normalized human chin EMG from wake to non-REM and REM."
  },
  {
    number: 11,
    citation: "Trinder J, Kleiman J, Carrington M, et al. Autonomic activity during human sleep as a function of time and sleep stage. Journal of Sleep Research. 2001;10:253–264.",
    href: "https://pubmed.ncbi.nlm.nih.gov/11903855/",
    note: "Repeated-measures study separating sleep-stage and time-of-night effects on heart rate and autonomic measures."
  },
  {
    number: 12,
    citation: "Fonseca P, van Gilst MM, Radha M, et al. Automatic sleep staging using heart rate variability, body movements, and recurrent neural networks in a sleep disordered population. Sleep. 2020;43:zsaa048.",
    href: "https://pubmed.ncbi.nlm.nih.gov/32249911/",
    note: "Independent validation of HRV-based sleep staging in a broad sleep-disordered cohort; supports cardiac data as useful secondary context."
  },
  {
    number: 13,
    citation: "Phan H, Andreotti F, Cooray N, Chen OY, De Vos M. SeqSleepNet: End-to-End Hierarchical Recurrent Neural Network for Sequence-to-Sequence Automatic Sleep Staging. IEEE Transactions on Neural Systems and Rehabilitation Engineering. 2019;27:400–410.",
    href: "https://pmc.ncbi.nlm.nih.gov/articles/PMC6481557/",
    note: "Primary sequence-to-sequence study separating within-epoch encoding from longer-range sleep-stage dynamics."
  },
  {
    number: 14,
    citation: "Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology. Heart rate variability: standards of measurement, physiological interpretation and clinical use. Circulation. 1996;93:1043–1065.",
    href: "https://pubmed.ncbi.nlm.nih.gov/8598068/",
    note: "Standard definitions of time- and frequency-domain HRV, band limits, and window lengths."
  },
  {
    number: 15,
    citation: "Bunde A, Havlin S, Kantelhardt JW, Penzel T, Peter JH, Voigt K. Correlated and uncorrelated regions in heart-rate fluctuations during sleep. Physical Review Letters. 2000;85:3736–3739.",
    href: "https://arxiv.org/abs/cond-mat/0012390",
    note: "Heartbeat intervals are long-range correlated in REM and wake but close to uncorrelated in light and deep NREM sleep."
  },
  {
    number: 16,
    citation: "Thomas RJ, Mietus JE, Peng CK, Goldberger AL. An electrocardiogram-based technique to assess cardiopulmonary coupling during sleep. Sleep. 2005;28:1151–1161.",
    href: "https://pubmed.ncbi.nlm.nih.gov/16268385/",
    note: "Coupling between heart-rate variability and ECG-derived respiration distinguishes stable from unstable sleep."
  },
  {
    number: 17,
    citation: "Lendner JD, Helfrich RF, Mander BA, et al. An electrophysiological marker of arousal level in humans. eLife. 2020;9:e55092.",
    href: "https://elifesciences.org/articles/55092",
    note: "The aperiodic (1/f) slope of the EEG spectrum tracks arousal level across wake, sleep stages and anesthesia."
  },
  {
    number: 18,
    citation: "Whitham EM, Pope KJ, Fitzgibbon SP, et al. Scalp electrical recording during paralysis: quantitative evidence that EEG frequencies above 20 Hz are contaminated by EMG. Clinical Neurophysiology. 2007;118:1877–1888.",
    href: "https://pubmed.ncbi.nlm.nih.gov/17574912/",
    note: "Neuromuscular paralysis removes most scalp EEG power above 20 Hz, showing it is largely muscle."
  },
  {
    number: 19,
    citation: "Douglas NJ, White DP, Pickett CK, Weil JV, Zwillich CW. Respiration during sleep in normal man. Thorax. 1982;37:840–844.",
    href: "https://pubmed.ncbi.nlm.nih.gov/7164002/",
    note: "Breathing is regular in NREM sleep and irregular in REM sleep."
  },
  {
    number: 20,
    citation: "Steriade M, McCormick DA, Sejnowski TJ. Thalamocortical oscillations in the sleeping and aroused brain. Science. 1993;262:679–685.",
    href: "https://pubmed.ncbi.nlm.nih.gov/8235588/",
    note: "Cellular mechanisms of spindles and slow oscillations in the thalamocortical system, and how neuromodulators switch them off on arousal."
  },
  {
    number: 21,
    citation: "Gao R, Peterson EJ, Voytek B. Inferring synaptic excitation/inhibition balance from field potentials. NeuroImage. 2017;158:70–78.",
    href: "https://pubmed.ncbi.nlm.nih.gov/28676297/",
    note: "Links the aperiodic slope of the power spectrum to the ratio of synaptic excitation to inhibition."
  },
  {
    number: 22,
    citation: "Peever J, Fuller PM. The biology of REM sleep. Current Biology. 2017;27:R1237–R1248.",
    href: "https://pubmed.ncbi.nlm.nih.gov/29161567/",
    note: "Review of the brainstem circuits that generate REM sleep, its eye movements and its muscle atonia."
  },
  {
    number: 23,
    citation: "Somers VK, Dyken ME, Mark AL, Abboud FM. Sympathetic-nerve activity during sleep in normal subjects. New England Journal of Medicine. 1993;328:303–307.",
    href: "https://pubmed.ncbi.nlm.nih.gov/8419815/",
    note: "Direct nerve recordings: sympathetic activity falls through NREM sleep and rises to waking levels with surges in REM."
  },
  {
    number: 24,
    citation: "Ghassemi MM, Moody BE, Lehman LH, et al. You Snooze, You Win: the PhysioNet/Computing in Cardiology Challenge 2018. Computing in Cardiology. 2018;45.",
    href: "https://physionet.org/content/challenge-2018/1.0.0/",
    note: "Dataset description: polysomnograms from the Massachusetts General Hospital sleep laboratory with AASM stage and arousal annotations."
  },
  {
    number: 25,
    citation: "Rabiner LR. A tutorial on hidden Markov models and selected applications in speech recognition. Proceedings of the IEEE. 1989;77:257–286.",
    href: "https://doi.org/10.1109/5.18626",
    note: "Forward–backward decoding of a hidden Markov model, used here to combine per-epoch stage probabilities with stage-transition probabilities."
  },
  {
    number: 26,
    citation: "Perslev M, Darkner S, Kempfner L, Nikolic M, Jennum PJ, Igel C. U-Sleep: resilient high-frequency sleep staging. npj Digital Medicine. 2021;4:72.",
    href: "https://doi.org/10.1038/s41746-021-00440-5",
    note: "Fully convolutional network trained on subsets of 13 internal datasets (plus 8 held-out evaluation sets; 15,660 participants across 16 clinical studies); reports CinC 2018 macro-F1 on 100 held-out records."
  },
  {
    number: 27,
    citation: "Perslev M, Jensen MH, Darkner S, Jennum PJ, Igel C. U-Time: A fully convolutional network for time series segmentation applied to sleep staging. Advances in Neural Information Processing Systems. 2019;32.",
    href: "https://arxiv.org/abs/1910.11162",
    note: "U-Sleep's predecessor; its CinC 2018 score is quoted from the U-Sleep paper."
  },
  {
    number: 28,
    citation: "Biswal S, Kulas J, Sun H, Goparaju B, Westover MB, Bianchi MT, Sun J. SLEEPNET: Automated sleep staging system via deep learning. arXiv:1707.08262. 2017.",
    href: "https://arxiv.org/abs/1707.08262",
    note: "Deep recurrent network trained on PSGs from over 10,000 MGH patients; accuracy and κ on 1,000 held-out recordings."
  },
  {
    number: 29,
    citation: "Rosenberg RS, Van Hout S. The American Academy of Sleep Medicine inter-scorer reliability program: sleep stage scoring. Journal of Clinical Sleep Medicine. 2013;9:81–87.",
    href: "https://pubmed.ncbi.nlm.nih.gov/23319910/",
    note: "More than 2,500 scorers on 1,800 epochs: 82.6% agreement with the majority overall; W 84.1%, N1 63.0%, N2 85.2%, N3 67.4%, REM 90.5%."
  },
  {
    number: 30,
    citation: "Khalighi S, Sousa T, Santos JM, Nunes U. ISRUC-Sleep: A comprehensive public dataset for sleep researchers. Computer Methods and Programs in Biomedicine. 2016;124:180–192.",
    href: "https://pubmed.ncbi.nlm.nih.gov/26589468/",
    note: "Primary dataset description for the Coimbra cohort (Subgroup I, 100 adult PSGs, two independent scorers)."
  },
  {
    number: 31,
    citation: "Radha M, Fonseca P, Moreau A, et al. Sleep stage classification from heart-rate variability using long short-term memory neural networks. Scientific Reports. 2019;9:14149.",
    href: "https://pubmed.ncbi.nlm.nih.gov/31578345/",
    note: `LSTM staging from ECG heart-rate variability alone on 292 subjects (four-class κ = ${fmt("lit.cardio_kappa4.low")}, R&K scoring).`
  },
  {
    number: 32,
    citation: "Bakker JP, Ross M, Cerny A, et al. Scoring sleep with artificial intelligence enables quantification of sleep stage architecture: validation of the CReSS algorithm. Journal of Clinical Sleep Medicine. 2021;17:1209–1221.",
    href: "https://pubmed.ncbi.nlm.nih.gov/33682676/",
    note: `Cardiorespiratory sleep staging (CReSS) using HRV, airflow, and thoracic respiratory effort (up to four-class κ = ${fmt("lit.cardio_kappa4.high")}).`
  }
];

export const technicalAppendixEntries = [
  {
    id: "tracks",
    label: "P1",
    title: "One-second tracks",
    summary: "Band shares, EOG and EMG power from a 2-second spectrum every second",
    source: "src/physionet_sleep/algorithms/eeg_tracks.py · compute_tracks",
    note: "Frame k is centred at k + 0.5 s. Heart rate is interpolated from R-peaks to the same second centres, and the effort belts are band-passed at 10 Hz and averaged per second.",
    code: `P = {c: power_spectra(signals[c], n) for c in SPECTRAL_CHANNELS}  # 2-s Hann, 1-s hop
C = (P["C3-M2"] + P["C4-M1"]) / 2
total = band_power(C, 0.5, 30) + EPS
t = {f"c_rel_{k}": band_power(C, *v) / total for k, v in BANDS.items()}
t["c_log_power"] = np.log10(total)
t["f_rel_delta"] = band_power(F, 0.5, 4) / (band_power(F, 0.5, 30) + EPS)
t["o_rel_alpha"] = band_power(O, 8, 12) / (band_power(O, 0.5, 30) + EPS)
t["eog_low"] = np.log10(band_power(P["E1-M2"], 0.5, 2) + EPS)
t["eog_high"] = np.log10(band_power(P["E1-M2"], 2, 8) + EPS)
t["emg_log_rms"] = 0.5 * np.log10(band_power(P["Chin1-Chin2"], 10, 80) + EPS)`
  },
  {
    id: "spindle-bursts",
    label: "P2",
    title: "Spindle-burst count",
    summary: "Seconds per epoch above a robust spindle-ratio threshold",
    source: "src/physionet_sleep/analysis/epoch_windows.py · spindle_burst_counts",
    note: "Computed from central EEG band shares before model scaling; the threshold is the night's median plus two robust standard deviations.",
    code: `ratio = sigma / (alpha + beta + 1e-6)
median = np.nanmedian(ratio)
threshold = median + 2 * 1.4826 * np.nanmedian(np.abs(ratio - median))
burst = (ratio > threshold).astype(np.float32)
# Pad the final incomplete epoch with zeros, then count seconds per epoch.
n_epochs = (len(burst) + 29) // 30
padded_burst = np.pad(burst, (0, n_epochs * 30 - len(burst)))
counts = padded_burst.reshape(n_epochs, 30).sum(axis=1)`
  },
  {
    id: "eye-movement",
    label: "P3",
    title: "Eye-movement activity",
    summary: "Band-limited EOG activity relative to a rolling local reference",
    source: "src/physionet_sleep/algorithms/staging.py · EyeMovementActivityGenerator",
    note: "A rolling local baseline keeps within-night changes comparable.",
    code: `eye_env = _band_envelope(eog, fs, 0.1, 5.0)
eye_baseline = rolling_median(
    eye_env,
    max(int(round(fs * 30.0)), 1),
)
eye_activity = np.maximum(eye_env - eye_baseline, 0.0)
eye_activity = rolling_mean(
    eye_activity,
    max(int(round(fs * 1.0)), 1),
)`
  },
  {
    id: "muscle-tone",
    label: "P4",
    title: "Chin-muscle tone",
    summary: "A smoothed EMG envelope with a sixty-second local baseline",
    source: "src/physionet_sleep/algorithms/staging.py · EMGToneFeatureGenerator",
    note: "Chin EMG is sampled at 200 Hz, so the full 10–100 Hz band is used.",
    code: `tone = _band_envelope(emg, fs, 10.0, 100.0)
tone = rolling_mean(tone, max(int(round(fs * 1.0)), 1))
tone_baseline = rolling_median(
    tone,
    max(int(round(fs * 60.0)), 1),
)
suppression_mask = tone < (
    np.maximum(tone_baseline, 1e-12) * self.suppression_ratio
)`
  },
  {
    id: "ratios",
    label: "P5",
    title: "Epoch-grid ratio features",
    summary: "Gain-free EEG band shares and muscle fraction per 30-second epoch",
    source: "src/physionet_sleep/algorithms/staging.py · EpochRatioFeatureGenerator",
    note: "Ratios from the same channel and epoch cancel between-recording differences in gain and electrode contact, so these features enter the model without per-night scaling.",
    code: `freqs, psd = welch(seg, fs=fs, nperseg=min(nperseg, seg.size))
total = _half_open_band_sum(psd, freqs, 1.0, 30.0)
band = {b: _half_open_band_sum(psd, freqs, lo, hi) for b, (lo, hi) in self.BANDS.items()}
values = {f"rel_{b}": band[b] / total for b in self.BANDS}
flank = band["theta"] + band["beta"]
values["spindle_rel"] = band["sigma"] / flank if flank > 0 else np.nan
muscle = _half_open_band_sum(psd, freqs, 30.0, 47.0)
values["muscle_frac_db"] = 10.0 * np.log10(muscle / total) if muscle > 0 else np.nan`
  },
  {
    id: "heart",
    label: "P6",
    title: "Cleaned heart-rate variability",
    summary: "Artifact rejection and multi-minute rhythm features",
    source: "src/physionet_sleep/algorithms/ecg.py · clean_nn_mask, compute_hrv_tracks",
    note: "A single missed or extra beat dominates a 30-second RMSSD, so intervals are cleaned first and each feature uses a window long enough for what it measures.",
    code: `local = _rolling_centered_median(rr, local_beats, min_count=3)
clean = (rr >= 300) & (rr <= 2000) & (np.abs(rr - local) <= 0.2 * local)

# per epoch centre c
out["sdnn_150s"][k] = np.std(rr[sel150][ok[sel150]], ddof=1)
out["dfa_a2_300s"][k] = dfa_alpha(rr[sel300][ok[sel300]], DFA_A2_SCALES)
window = hr_bins[k0 : k0 + 30]          # 10-s mean heart rate over 5 min
out["hr_slow_sd_300s"][k] = np.nanstd(window, ddof=1)`
  },
  {
    id: "breathing",
    label: "P7",
    title: "Heart–breathing coupling",
    summary: "Coherence of heart rate and breathing at the breathing frequency",
    source: "src/physionet_sleep/algorithms/respiration.py · hr_resp_coherence_track",
    note: "Both signals are put on a 4-Hz grid; the breathing frequency is the effort-belt spectral peak between 0.1 and 0.5 Hz.",
    code: `f, pr = welch(r_seg, fs=g_hz, nperseg=nper, noverlap=nper // 2)
band = (f >= 0.1) & (f <= 0.5)
f_resp = f[band][np.argmax(pr[band])]
fc, coh = coherence(n_seg, r_seg, fs=g_hz, nperseg=nper, noverlap=nper // 2)
j = int(np.argmin(np.abs(fc - f_resp)))
out[k] = float(np.mean(coh[max(j - 1, 0) : j + 2]))`
  },
  {
    id: "validation",
    label: "P8",
    title: "Subject-grouped evaluation",
    summary: "Every epoch from one person stays on the same side of the split",
    source: "src/physionet_sleep/experiments/tabular_runner.py · build_validation_splits",
    note: "The group array holds patient identifiers, so epochs from one night cannot appear in both development and test data.",
    code: `splitter = StratifiedGroupKFold(
    n_splits=effective_folds,
    shuffle=True,
    random_state=random_state,
)
yield from splitter.split(
    np.zeros_like(y),
    y,
    groups,
)`
  },
  {
    id: "window",
    label: "P9",
    title: "Context windows",
    summary: "25 six-second steps of 20 signals around each epoch",
    source: "src/physionet_sleep/analysis/epoch_windows.py · WindowEncoder",
    note: "Standardization uses the training patients only. Steps before the start or after the end of the recording are zero, which is the training mean.",
    code: `def _steps(self, X):
    Z = np.nan_to_num(np.clip((X - self.mean) / self.sd, -10, 10))
    n = -(-len(Z) // self.step_seconds)
    Z = np.vstack([Z, np.zeros((n * self.step_seconds - len(Z), Z.shape[1]))])
    S = Z.reshape(n, self.step_seconds, Z.shape[1]).mean(1)  # 1-s -> 6-s steps
    pad = np.zeros(((self.context + 1) * self.steps_per_epoch, Z.shape[1]))
    return np.vstack([pad, S, pad])

# epoch e covers steps (e - 2) * 5 ... (e + 3) * 5 - 1 of the padded array
start = pad + (epoch_index - self.context) * self.steps_per_epoch
window = S[start[:, None] + np.arange(self.n_steps)[None, :]]`
  },
  {
    id: "hmm",
    label: "P10",
    title: "Whole-night decoder (comparison rows only)",
    summary: "Transition counts from expert hypnograms, forward–backward over the night; not in the adopted model",
    source: "src/physionet_sleep/decoders/hmm.py · learn_transition_matrix, posterior_decode",
    note: "Stage-1 probabilities come from a class-balanced model, so they act as emission likelihoods without dividing by class priors. A Laplace prior of one count keeps every transition possible.",
    code: `trans = np.full((5, 5), prior)
for y in label_nights:
    np.add.at(trans, (y[:-1], y[1:]), 1.0)
trans /= trans.sum(1, keepdims=True)

alpha[0] = initial * emis[0]; alpha[0] /= alpha[0].sum()
for t in range(1, n):
    a = (alpha[t - 1] @ trans) * emis[t]
    scale[t] = a.sum(); alpha[t] = a / scale[t]
for t in range(n - 2, -1, -1):
    beta[t] = (trans @ (emis[t + 1] * beta[t + 1])) / scale[t + 1]
posterior = alpha * beta  # normalised per epoch; argmax is the stage`
  }
];
