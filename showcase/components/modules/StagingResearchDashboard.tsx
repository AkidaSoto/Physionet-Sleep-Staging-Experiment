"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import saliencyData from "../../content/generated/staging-saliency.json";
import isrucStatus from "../../content/generated/isruc-status.json";
import holdoutDistribution from "../../content/generated/holdout-distribution.json";
import {
  stagingDashboardData,
  type FeatureDistribution,
  type FeatureSignalExample,
  type ModelResult,
  type SaliencyData
} from "../../content/staging-dashboard-data";

const data = stagingDashboardData;
const sensorColors: Record<string, string> = {
  Brain: "#78d8ff", Eyes: "#ef8ab5", "Chin muscle": "#b49cf0", Heart: "#ff7d7d", Breathing: "#9fd36f"
};

const stageColors: Record<string, string> = {
  Wake: "#91a9ce",
  N1: "#72d2ff",
  N2: "#6fc4a2",
  N3: "#f0b36d",
  REM: "#ef8ab5"
};

function Citation({ number }: { number: number }) {
  return <a className="research-citation" href={`#reference-${number}`} aria-label={`Reference ${number}`}>[{number}]</a>;
}
type FeatureSpec = {
  id: string;
  name: string;
  signal: string;
  summary: string;
  measurement: string;
  stageUse?: string;
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

// EEG is sampled at 200 Hz, so the gamma band runs to 100 Hz.
const bandEdges: Record<string, string> = {
  theta: "4–8 Hz",
  alpha: "8–11 Hz",
  beta: "16–30 Hz",
  gamma: "30–100 Hz"
};

const relativeBandUse: Record<string, string> = {
  theta: "Prominent at sleep onset and in N1.",
  beta: "Higher in wake and arousal."
};

// One-second tracks: each comes from a 2-second Hann-windowed spectrum stepped every second.
const trackMethod = "One value per second from a 2-second Hann-windowed spectrum, stepped every second.";

const eegSpectrumFeatures: FeatureSpec[] = [
  {
    id: "spindle_features.aperiodic_exponent",
    name: "1/f slope",
    signal: "EEG",
    summary: "Slope of the 1/f background of the EEG spectrum.",
    measurement: "Linear fit of log power against log frequency over 2–7 Hz and 14–24 Hz (skipping the alpha and spindle peaks), per 2-second window.",
    stageUse: "The spectrum steepens as sleep deepens."
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
    name: "Delta power",
    signal: "EEG",
    summary: "EEG power in 1–4 Hz.",
    measurement: "Median spectral density across 1–4 Hz in a 2-second window, averaged over C3 and C4.",
    stageUse: "Rises with deeper NREM sleep."
  },
  ...["theta", "beta"].map((band) => ({
    id: `epoch_ratio_features.rel_${band}`,
    name: `${band[0].toUpperCase()}${band.slice(1)} share`,
    signal: "EEG",
    summary: `Share of the epoch's 1–30 Hz EEG power that falls in ${bandEdges[band]}.`,
    measurement: "Welch spectrum (4-second Hann segments) of each 30-second epoch, band power over 1–30 Hz power, averaged over C3 and C4. A ratio from the same channel cancels differences in amplifier gain and electrode contact between recordings.",
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
    ["delta", "0.5–4 Hz", "Rises through N2 into N3."],
    ["theta", "4–8 Hz", "Rises at sleep onset; theta runs mark N1 and REM."],
    ["alpha", "8–12 Hz", "High in relaxed wake; second-long alpha bursts mark arousals."],
    ["beta", "16–30 Hz", "High in wake and arousals."]
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
    measurement: `${trackMethod} log10 of 0.5–30 Hz power, C3 and C4 spectra averaged. Scaled per subject (median and IQR) before modelling.`,
    stageUse: "High-voltage slow waves and K-complexes raise it; wake and REM are low-voltage."
  }
];

const featureFamilies: FeatureFamily[] = [
  {
    id: "spectrum",
    label: "EEG spectrum",
    signal: "EEG",
    summary: "Measurements describe the central EEG spectrum: its background slope, overall power, and activity in different frequency bands.",
    evidence: "Sleep depth changes the shape of the EEG spectrum: delta tends to rise with deeper non-REM sleep while alpha falls. Spectral ratios can separate some stages strongly, but no single measure separates all five reliably.",
    limitation: "Spectral summaries describe a window, not a waveform: a K-complex and a burst of movement can raise the same band. Muscle activity inflates beta and gamma on the scalp.",
    sources: [1, 6],
    features: eegSpectrumFeatures
  },
  {
    id: "spindles",
    label: "Spindles",
    signal: "EEG",
    summary: "Spindle-band power and detected spindle bursts describe brief N2 activity.",
    evidence: "Sleep spindles are 0.5–2 second bursts of 11–16 Hz activity and the defining feature of N2. Averaged over 30 seconds a few spindles barely move the spindle-band share; at 1-second resolution each one is a separate peak. Model C includes both spindle-band share and a spindle-burst detector; the latter more directly targets brief events.",
    limitation: "Sigma-band power suggests spindle activity but does not detect individual spindle events. Alpha bursts at 11–12 Hz overlap the band, and spindle density falls with age.",
    sources: [1, 7],
    features: [
      { id: "spindle_bursts", name: "Spindle bursts", signal: "EEG", summary: "Seconds in the epoch with a spindle-band burst.", measurement: `${trackMethod} Per second, the 11–16 Hz share divided by the 8–12 Hz plus 16–30 Hz shares (C3 and C4 spectra averaged); a burst is a second more than two robust SDs above that night's median, counted per 30-second epoch.`, stageUse: "Several per epoch in N2 (and N3, where spindles also occur), rare in wake, N1 and REM." },
      { id: "track.c_rel_sigma", name: "Spindle-band share", signal: "EEG", summary: "Share of each second's 0.5–30 Hz central EEG power that falls in 11–16 Hz.", measurement: `${trackMethod} 11–16 Hz power over 0.5–30 Hz power, C3 and C4 spectra averaged.`, stageUse: "Brief peaks during N2 spindles." }
    ]
  },
  {
    id: "topography",
    label: "Frontal slow waves / occipital alpha",
    signal: "EEG",
    summary: "Frontal slow waves and occipital alpha describe the location and amplitude of EEG staging cues.",
    evidence: "AASM scores N3 from slow waves of at least 75 µV at 0.5–2 Hz on frontal leads, and wake from posterior alpha that attenuates at sleep onset. Calibrated microvolts make the amplitude rule possible.",
    limitation: "The amplitude rule assumes adult slow-wave amplitudes; slow waves shrink with age, so older patients can reach N3 by expert judgement with fewer waves above 75 µV. Eyes-open wake shows little alpha.",
    sources: [1],
    features: [
      { id: "topography_features.frontal_slow_wave_fraction", name: "Frontal slow-wave fraction", signal: "EEG", summary: "Share of the epoch covered by 0.5–2 Hz waves of at least 75 µV peak-to-peak on F3/F4.", measurement: "Band-pass 0.5–2 Hz, split into full waves at upward zero crossings, keep waves of 0.5–2 s with peak-to-peak ≥ 75 µV; fraction of the epoch they cover, averaged over F3-M2 and F4-M1.", stageUse: "AASM scores N3 when this reaches 20%." },
      { id: "topography_features.occipital_rel_alpha", name: "Occipital alpha share", signal: "EEG", summary: "Share of occipital EEG power at 8–12 Hz.", measurement: "Welch spectrum (4-s segments) per epoch on O1-M2 and O2-M1, 8–12 Hz over 1–30 Hz power, averaged.", stageUse: "High in relaxed wake, drops out at sleep onset (N1)." },
      { id: "track.f_rel_delta", name: "Frontal delta share", signal: "EEG", summary: "Share of each second's 0.5–30 Hz frontal EEG power in 0.5–4 Hz.", measurement: `${trackMethod} From the average of the F3 and F4 spectra.`, stageUse: "Highest in N3, where slow waves are largest frontally." },
      { id: "track.o_rel_alpha", name: "Occipital alpha share", signal: "EEG", summary: "Share of each second's 0.5–30 Hz occipital EEG power in 8–12 Hz.", measurement: `${trackMethod} From the average of the O1 and O2 spectra.`, stageUse: "High in eyes-closed wake; returns briefly during arousals." }
    ]
  },
  {
    id: "eye-movement",
    label: "Eye movement",
    signal: "EOG",
    summary: "Eye activity and slow- and fast-band EOG power describe movement and the shape of the eye-channel spectrum.",
    evidence: "EOG carries slow-eye-movement evidence around sleep onset and rapid-eye-movement evidence in REM. Subject-independent studies have shown that two EOG channels alone contain substantial five-stage information.",
    limitation: "Eye movements also occur during wake, and EOG electrodes pick up frontal EEG, so N3 slow waves leak into the slow EOG band. This dataset records a single eye channel (E1-M2).",
    sources: [1, 8, 9],
    features: [
      { id: "eye_movement_activity.eye_movement_activity", name: "Eye-movement activity", signal: "EOG", summary: "EOG amplitude above its local baseline.", measurement: "0.1–5 Hz envelope minus its 30-second rolling median, floored at zero and smoothed over 1 second." },
      { id: "eye_movement_activity.eye_movement_baseline", name: "Eye-movement baseline", signal: "EOG", summary: "The local baseline itself.", measurement: "30-second rolling median of the 0.1–5 Hz EOG envelope." },
      { id: "epoch_ratio_features.eog_rem_vs_noise_db", name: "Eye-movement power", signal: "EOG", summary: "Eye-movement-range EOG power (0.5–5 Hz) relative to the same channel's 5–30 Hz floor, in decibels.", measurement: "One 30-second spectrum of the E1-M2 channel per epoch; mean magnitude 0.5–5 Hz over 5–30 Hz, in dB.", stageUse: "Rapid eye movements over a quiet, atonic floor." },
      { id: "epoch_ratio_features.eog_rem_vs_sem_db", name: "Fast-to-slow eye movement ratio", signal: "EOG", summary: "0.5–5 Hz against 0.1–0.5 Hz EOG power, in decibels.", measurement: "From the same per-epoch EOG spectrum.", stageUse: "Low when slow rolling eye movements dominate, as at sleep onset." },
      { id: "track.eog_low", name: "Slow EOG power", signal: "EOG", summary: "EOG power at 0.5–2 Hz each second, on a log scale.", measurement: `${trackMethod} log10 of 0.5–2 Hz power on E1-M2.`, stageUse: "Slow rolling eye movements at sleep onset; also frontal slow waves in N3." },
      { id: "track.eog_high", name: "Fast EOG power", signal: "EOG", summary: "EOG power at 2–8 Hz each second, on a log scale.", measurement: `${trackMethod} log10 of 2–8 Hz power on E1-M2.`, stageUse: "The sharp deflections of rapid eye movements in REM and blinks in wake." }
    ]
  },
  {
    id: "muscle-tone",
    label: "Muscle tone",
    signal: "Chin EMG",
    summary: "Chin-tone measurements describe muscle activity relative to the night’s quietest periods.",
    evidence: "Chin tone typically falls from wake through non-REM and is lowest in REM. Combined EOG and EMG features can recover much of the information used in multichannel staging.",
    limitation: "Movement, snoring, electrode contact, and REM sleep without atonia can break the expected pattern. Mains interference on the chin lead can mask atonia in a few recordings.",
    sources: [1, 9, 10],
    features: [
      { id: "epoch_ratio_features.emg_floor_db", name: "Chin muscle tone", signal: "Chin EMG", summary: "Chin EMG tone relative to the night's quietest epochs, in decibels.", measurement: "10·log10 of the epoch's 10–100 Hz tone over the 5th percentile of all epochs in the recording. No labels are used.", stageUse: "High in wake, lowest in REM." },
      { id: "track.emg_log_rms", name: "Chin EMG level", signal: "Chin EMG", summary: "RMS of 10–80 Hz chin EMG each second, on a log scale.", measurement: `${trackMethod} Half the log10 of 10–80 Hz power on Chin1-Chin2. Scaled per subject before modelling, so it reads relative to that night.`, stageUse: "Drops at REM onset; brief bursts mark movements and arousals." }
    ]
  },
  {
    id: "heart",
    label: "Heart",
    signal: "ECG",
    summary: "Heart rate every second from R-peak detection, plus a 5-minute measure of heart-rate variability. Tested but not used in the final model: with EEG, EOG and chin EMG present it adds 0.002 macro-F1.",
    evidence: "Sleep changes the heart's rhythm as well as its rate. In stable NREM, and most of all N3, the rhythm is steady; in REM and wake it shows slow, correlated swings and surges. Within subjects, slow heart-rate swings were lower in N3 than in REM in 109 of 112 subjects. The one-second heart-rate signal captures what the earlier 30-second heart features measured (rate, arousal-related surges and short-term variability), along with their timing.",
    limitation: "A 5-minute window blurs stage boundaries. Age and sleep apnea flatten autonomic contrasts, and apnea-related heart-rate cycles can mimic REM-like variability. Ectopic beats and missed R-peaks create false jumps in the 1-second rate.",
    sources: [11, 14, 15],
    features: [
      { id: "hrv_features.hr_slow_sd_300s", name: "Slow heart-rate swings", signal: "ECG", summary: "How much the 10-second average heart rate wanders over 5 minutes.", measurement: "Standard deviation of 10-second mean heart rate across ±150 s (at least 20 of 30 bins), from cleaned beat intervals.", stageUse: "Lower in N3 than in REM in almost every subject with both stages." },
      { id: "track.hr", name: "Heart rate", signal: "ECG", summary: "Instantaneous heart rate at each second.", measurement: "Beat-to-beat rate (60 / RR) from Pan–Tompkins R-peaks, intervals of 0.3–2.0 s, linearly interpolated to the centre of each second.", stageUse: "Higher and more variable in wake and REM; surges mark arousals." }
    ]
  },
  {
    id: "breathing",
    label: "Breathing",
    signal: "Respiratory effort",
    summary: "Breath-timing regularity and heart–breathing coupling over 1–2 minutes, plus the effort-belt signal and its amplitude every second. Tested but not in the final model; without the EEG, heart and breathing alone reach 0.65 macro-F1 with a whole-night model (S2).",
    evidence: "Breathing becomes regular in stable NREM and irregular in REM and wake, and in stable NREM the heart rate follows breathing closely. In this dataset breath depth was more regular in N3 than in REM in 113 of 119 subjects, outside scored respiratory events.",
    limitation: "Apneas, hypopneas, snoring, and body position change breathing directly. Effort belts measure movement, not airflow, and can slip.",
    sources: [16],
    features: [
      { id: "breath_regularity.breath_interval_cv_60s", name: "Breath timing variability", signal: "Respiratory effort", summary: "Variation of breath-to-breath intervals over 1 minute.", measurement: "Inspiratory peaks of the 0.1–0.7 Hz band-passed effort-belt sum, at least 1.5 s apart; coefficient of variation of intervals within ±30 s.", stageUse: "Regular in N3, irregular in REM and wake." },
      { id: "cardiorespiratory_coupling.hr_resp_coherence_120s", name: "Heart–breathing coupling", signal: "ECG + effort", summary: "How strongly heart rate is phase-locked to breathing over 2 minutes.", measurement: "Magnitude-squared coherence between the cleaned 4-Hz NN series and the effort signal at the dominant breathing frequency, 32-s Welch segments within ±60 s.", stageUse: "High in N3 and N2, lower in REM and wake." },
      { id: "track.resp_effort", name: "Breathing effort", signal: "Respiratory effort", summary: "The band-passed effort-belt sum, averaged over each second.", measurement: "Abdomen + chest belts, averaged to 10 Hz, band-passed 0.1–0.7 Hz (4th-order Butterworth, zero phase), then averaged per second.", stageUse: "The breathing waveform itself, so the model can see pauses and irregular breaths." },
      { id: "track.resp_env", name: "Breathing amplitude", signal: "Respiratory effort", summary: "Breath-by-breath amplitude of the same signal, on a log scale.", measurement: "log10 of the Hilbert envelope of the band-passed effort signal, averaged per second.", stageUse: "Steady in NREM, variable in REM and wake; drops with hypopneas." }
    ]
  }
];

type FeatureBasis = "aasm" | "published" | "adapted";

const basisLabels: Record<FeatureBasis, { short: string; long: string }> = {
  aasm: { short: "AASM rule", long: "Implements an AASM scoring criterion directly" },
  published: { short: "Published method", long: "Established measure from the literature, standard parameters" },
  adapted: { short: "Adapted", long: "Research-motivated idea with our own implementation choices" }
};

// Basis (how directly the feature follows a scoring rule or a published method) and the biological mechanism it measures.
const featureBasis: Record<string, { basis: FeatureBasis; mechanism: string; cite?: number }> = {
  "topography_features.frontal_slow_wave_fraction": { basis: "aasm", mechanism: "Cortical slow oscillation", cite: 1 },
  "topography_features.occipital_rel_alpha": { basis: "aasm", mechanism: "Posterior alpha rhythm", cite: 1 },
  "epoch_ratio_features.rel_theta": { basis: "aasm", mechanism: "Cortical slowing at sleep onset", cite: 1 },
  "spindle_features.delta_power": { basis: "aasm", mechanism: "Cortical slow oscillation", cite: 1 },
  "epoch_ratio_features.rel_sigma": { basis: "aasm", mechanism: "Thalamocortical spindles", cite: 1 },
  "epoch_ratio_features.emg_floor_db": { basis: "aasm", mechanism: "Brainstem muscle atonia", cite: 1 },
  "epoch_ratio_features.emg_baseline_floor_db": { basis: "aasm", mechanism: "Brainstem muscle atonia", cite: 1 },
  "spindle_features.aperiodic_exponent": { basis: "published", mechanism: "Cortical arousal (excitation/inhibition balance)", cite: 17 },
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
  "spindle_bursts": { basis: "adapted", mechanism: "Thalamocortical spindles", cite: 1 },
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

// What the research says generates each signal, and why that generator changes with sleep stage.
const mechanismBiology: Array<{ key: string; text: string; cites: number[] }> = [
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
    text: "Alpha (8–12 Hz) is the idling rhythm of the visual cortex during relaxed, eyes-closed wakefulness. It attenuates with drowsiness, and its loss is the AASM marker of the wake-to-N1 transition.",
    cites: [1]
  },
  {
    key: "Cortical slowing at sleep onset",
    text: "As arousal systems disengage at sleep onset, cortical activity slows from alpha toward theta (4–7 Hz) before spindles or slow waves appear. AASM defines N1 by this low-voltage, theta-dominant pattern.",
    cites: [1]
  },
  {
    key: "Cortical arousal",
    text: "The aperiodic (1/f) slope of the EEG spectrum is thought to reflect the cortex's balance of excitation and inhibition. It steepens as arousal falls, from wake through N1, N2 and N3, independent of any specific rhythm.",
    cites: [21, 17]
  },
  {
    key: "Scalp muscle activity",
    text: "Above about 20 Hz, scalp EEG is dominated by electromyographic activity from scalp, jaw and neck muscles rather than by the brain. Muscle tone falls with sleep, so this high-frequency power marks wakefulness, arousals and movement.",
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

function mechanismFor(featureId: string) {
  const label = featureBasis[featureId]?.mechanism;
  return label ? mechanismBiology.find((item) => label.startsWith(item.key)) : undefined;
}

function BasisTag({ featureId, compact = false }: { featureId: string; compact?: boolean }) {
  const info = featureBasis[featureId];
  if (!info) return null;
  return (
    <em className={`research-basis-tag is-${info.basis}`} title={basisLabels[info.basis].long}>
      {basisLabels[info.basis].short}{compact ? null : <> · {info.mechanism}</>}
    </em>
  );
}

type StageFeatureTarget = {
  stage: string;
  cue: string;
  featureIds: string[];
  // Direction the physiology predicts for this stage: higher or lower than in the other stages.
  expect: Record<string, "high" | "low">;
};

const stageFeatureTargets: StageFeatureTarget[] = [
  {
    stage: "Wake",
    cue: "Wake shows posterior alpha when the eyes are closed, high chin tone, fast EEG activity and a faster heart rate.",
    featureIds: ["topography_features.occipital_rel_alpha", "track.o_rel_alpha", "track.c_rel_alpha", "epoch_ratio_features.rel_beta", "track.c_rel_beta", "epoch_ratio_features.emg_floor_db", "track.hr"],
    expect: { "topography_features.occipital_rel_alpha": "high", "track.o_rel_alpha": "high", "track.c_rel_alpha": "high", "epoch_ratio_features.rel_beta": "high", "track.c_rel_beta": "high", "epoch_ratio_features.emg_floor_db": "high", "track.hr": "high" }
  },
  {
    stage: "N1",
    cue: "Sleep onset slows the EEG toward theta, brings slow rolling eye movements, and leaves breathing less regular than in deeper sleep.",
    featureIds: ["epoch_ratio_features.rel_theta", "track.c_rel_theta", "spindle_features.aperiodic_exponent", "epoch_ratio_features.eog_rem_vs_sem_db", "breath_regularity.breath_interval_cv_60s"],
    expect: { "epoch_ratio_features.rel_theta": "high", "track.c_rel_theta": "high", "spindle_features.aperiodic_exponent": "low", "epoch_ratio_features.eog_rem_vs_sem_db": "low", "breath_regularity.breath_interval_cv_60s": "high" }
  },
  {
    stage: "N2",
    cue: "N2 is identified by sleep spindles and K-complexes. Both are brief events, so event detection complements average band power. Heart rate also follows breathing closely in stable NREM.",
    featureIds: ["spindle_bursts", "track.c_rel_sigma", "cardiorespiratory_coupling.hr_resp_coherence_120s"],
    expect: { "spindle_bursts": "high", "track.c_rel_sigma": "high", "cardiorespiratory_coupling.hr_resp_coherence_120s": "high" }
  },
  {
    stage: "N3",
    cue: "N3 is defined by high-amplitude frontal slow waves. Heart rate and breathing are also at their most regular.",
    featureIds: ["topography_features.frontal_slow_wave_fraction", "track.f_rel_delta", "spindle_features.delta_power", "track.c_log_power", "hrv_features.hr_slow_sd_300s"],
    expect: { "topography_features.frontal_slow_wave_fraction": "high", "track.f_rel_delta": "high", "spindle_features.delta_power": "high", "track.c_log_power": "high", "hrv_features.hr_slow_sd_300s": "low" }
  },
  {
    stage: "REM",
    cue: "REM is identified by rapid eye movements with low chin tone. Heart rate becomes less coupled to breathing and shows slower fluctuations.",
    featureIds: ["track.c_log_power", "epoch_ratio_features.rel_theta", "epoch_ratio_features.eog_rem_vs_noise_db", "eye_movement_activity.eye_movement_activity", "eye_movement_activity.eye_movement_baseline", "track.eog_high", "epoch_ratio_features.emg_floor_db", "cardiorespiratory_coupling.hr_resp_coherence_120s", "track.resp_env", "track.resp_effort"],
    expect: { "track.c_log_power": "low", "epoch_ratio_features.rel_theta": "high", "epoch_ratio_features.eog_rem_vs_noise_db": "high", "eye_movement_activity.eye_movement_activity": "high", "eye_movement_activity.eye_movement_baseline": "high", "track.eog_high": "high", "epoch_ratio_features.emg_floor_db": "low", "cardiorespiratory_coupling.hr_resp_coherence_120s": "low", "track.resp_env": "low", "track.resp_effort": "high" }
  }
];

// How the observed distribution compares with the direction the physiology predicts for the selected stage.
function featureBehaviour(distribution: { stages: Array<{ stage: string; median: number; one_vs_rest_cohens_d: number }> }, stage: string, direction: "high" | "low") {
  const byMedian = [...distribution.stages].sort((x, y) => x.median - y.median);
  const highestStages = byMedian.filter((row) => Math.abs(row.median - byMedian[byMedian.length - 1].median) < 1e-9).map((row) => row.stage);
  const lowestStages = byMedian.filter((row) => Math.abs(row.median - byMedian[0].median) < 1e-9).map((row) => row.stage);
  const highest = highestStages.join(" and ");
  const lowest = lowestStages.join(" and ");
  const d = distribution.stages.find((row) => row.stage === stage)?.one_vs_rest_cohens_d ?? 0;
  const signed = direction === "high" ? d : -d;
  const extreme = (direction === "high" ? highestStages : lowestStages).includes(stage);
  const verdict: "expected" | "partly" | "unexpected" = signed >= 0.5 && extreme ? "expected" : signed >= 0.2 ? "partly" : "unexpected";
  return { verdict, highest, lowest, d };
}

// Which appendix snippet holds each family's code; every 1-second track points to the track code.
const codeAnchor: Record<string, string | undefined> = {
  spectrum: "ratios",
  spindles: "tracks",
  topography: undefined,
  "eye-movement": "eye-movement",
  "muscle-tone": "muscle-tone",
  heart: "heart",
  breathing: "breathing"
};

const featureDerivations: Record<string, { method: string }> = {
  spectrum: {
    method: "From a 2-second spectrogram: the 1/f fit (2–7 and 14–24 Hz, skipping alpha and spindles), delta power, and band power above the fit. From a Welch spectrum of each 30-second epoch: theta and beta as shares of 1–30 Hz power."
  },
  spindles: {
    method: "The 1-second track takes a 2-second Hann-windowed spectrum every second on C3 and C4, averages the two spectra, and divides 11–16 Hz power by 0.5–30 Hz power."
  },
  topography: {
    method: "Frontal slow waves follow the AASM definition directly: 0.5–2 Hz waves of at least 75 µV peak-to-peak on F3/F4, as a share of the epoch. Occipital alpha is the 8–12 Hz share of O1/O2 power from a per-epoch Welch spectrum."
  },
  "eye-movement": {
    method: "Eye-movement activity is the 0.1–5 Hz envelope above its rolling baseline. The two ratios come from one 30-second spectrum of the E1-M2 EOG channel per epoch (this dataset records a single eye channel)."
  },
  "muscle-tone": {
    method: "Chin EMG is band-passed to 10–100 Hz, enveloped, and smoothed over one second. Each epoch's tone is expressed in decibels above the 5th percentile of all epochs in that recording."
  },
  heart: {
    method: "R peaks are detected, refined to sub-sample timing with a cubic spline, and intervals outside 300–2000 ms or more than 20% from the local median are rejected. Slow heart-rate swings use the 10-second mean rate over ±150 s."
  },
  breathing: {
    method: "Breaths are inspiratory peaks of the 0.1–0.7 Hz band-passed effort-belt sum. Timing variability uses the breaths within ±30 s. Coupling is the coherence between the cleaned heart-rate series and the effort signal at the breathing frequency over ±60 s."
  }
};

const trackDerivation = "Before modelling, values are scaled by the night’s median and interquartile range, then averaged into six-second steps.";

type ModelProfile = {
  input: string;
  context: string;
  architecture: string[];
  fit: string;
  tradeoff: string;
  sources: number[];
};

const modelProfiles: Record<string, ModelProfile> = {
  epoch: {
    input: "The 19 features averaged over one 30-second epoch",
    context: "None; every epoch is classified on its own",
    architecture: ["19 epoch features", "gradient-boosted trees", "5 stage probabilities"],
    fit: "Histogram gradient boosting with library defaults (100 rounds, 31-leaf trees, learning rate 0.1) and class-balanced weights. Features scaled per subject by median and interquartile range; ratio and topography features stay raw. No tuning.",
    tradeoff: "The reference point: everything later is measured against what one epoch's averages can do.",
    sources: [6]
  },
  "epoch-hmm": {
    input: "The epoch model's five stage probabilities for every epoch of the night",
    context: "The whole night, through stage-to-stage transition probabilities",
    architecture: ["epoch model", "stage probabilities", "whole-night decoder", "stage label"],
    fit: "A hidden Markov model whose 5×5 transition table is counted from the training subjects' hypnograms. Forward–backward decoding weighs each epoch's probabilities against how likely each stage is, given the stages around it.",
    tradeoff: "It adds no new physiology; it removes sequences that rarely happen, such as N3 straight to REM. It matters here because this model cannot see the neighbouring epochs.",
    sources: [25]
  },
  "window-features": {
    input: "The 19 features every 6 seconds over the epoch and two neighbours on each side",
    context: "150 seconds · 25 six-second steps",
    architecture: ["19 features × 25 steps", "gradient-boosted trees", "stage label"],
    fit: "Same learner. Each epoch becomes 475 numbers, one per feature and 6-second step, so the trees can compare the epoch with what came before and after.",
    tradeoff: "Isolates the value of context: no new signals, only the same features at finer time resolution and over a longer window.",
    sources: [4]
  },
  "window-all": {
    input: "33 measurements, including heart and breathing, every 6 seconds over 150 seconds",
    context: "150 seconds · 25 six-second steps",
    architecture: ["33 signals × 25 steps", "gradient-boosted trees", "stage label"],
    fit: "825 inputs per epoch. Tracks and features are scaled per subject, then standardized with the training subjects' mean and SD and clipped at ±10. Library-default gradient boosting, class-balanced, no tuning.",
    tradeoff: "The best score on the page, but heart rate, breathing and their three features add only 0.002 over the EEG, eyes and chin.",
    sources: [4]
  },
  window: {
    input: "27 EEG, eye and chin measurements, every 6 seconds over 150 seconds",
    context: "150 seconds · 25 six-second steps",
    architecture: ["27 signals × 25 steps", "gradient-boosted trees", "stage label"],
    fit: "675 inputs per epoch (27 signals × 25 steps), scaled per subject, then standardized with the training subjects' mean and SD. Library-default gradient boosting, class-balanced, no tuning.",
    tradeoff: "The adopted model: the signals the AASM rules are written around, read in context, and nothing that does not earn its place. Every input is a named signal at a time offset.",
    sources: [4, 1]
  },
  cnn: {
    input: "The 14 tracks and 16 features at 1-second resolution over the epoch and one neighbour on each side",
    context: "90 seconds · 90 one-second steps",
    architecture: ["30 signals × 90 s", "dilated convolutions", "attention pooling", "stage label"],
    fit: "Three residual dilated convolution blocks (32 channels, kernel 5, dilation 1–4) and attention pooling over seconds, class-weighted, early stopped on 10% of training subjects; two seeds averaged.",
    tradeoff: "It learns its own temporal filters and lands just below the boosted window, with inputs to its classifier that no longer have names.",
    sources: [4, 13]
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
  title,
  children
}: {
  title: string;
  children?: React.ReactNode;
}) {
  return (
    <header className="research-section-heading">
      <div>
        <h2>{title}</h2>
        {children ? <p className="research-section-lede">{children}</p> : null}
      </div>
    </header>
  );
}

function PaperDisclosure({
  label,
  title,
  summary,
  children,
  open = false,
  className = ""
}: {
  label?: string;
  title: string;
  summary: string;
  children: React.ReactNode;
  open?: boolean;
  className?: string;
}) {
  return (
    <details className={`research-paper-disclosure ${className}`.trim()} open={open || undefined}>
      <summary>
        {label ? <span className="research-disclosure-label">{label}</span> : null}
        <strong>{title}</strong>
        <small>{summary}</small>
      </summary>
      <div className="research-paper-disclosure-body">{children}</div>
    </details>
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
          Sleep staging turns an overnight <strong>polysomnogram (PSG)</strong> into a timeline of wake, N1, N2, N3 and REM. That timeline establishes how much a patient slept, how fragmented sleep was, and the sleep state in which respiratory events and arousals occurred. Accurate staging therefore underpins the interpretation of the rest of the study. <Citation number={2} />
        </p>
        <p>Manual scoring is slow and partly subjective. In the AASM inter-scorer reliability program, experienced scorers agreed with the majority on 83% of epochs, but on only 63% of N1 and 67% of N3 epochs. <Citation number={29} /> Deep networks report strong performance on sleep staging. They learn representations from signals; connecting those representations to physiological events remains a separate task, and saliency maps provide approximate explanations. <Citation number={26} /></p>
      </div>

      <div className="research-introduction-figure" role="img" aria-label="EEG, EOG, and chin EMG are scored in 30-second epochs to produce a whole-night hypnogram">
        <div className="research-introduction-signals">
          <span><strong>EEG</strong><small>brain activity</small></span>
          <span><strong>EOG</strong><small>eye movement</small></span>
          <span><strong>EMG</strong><small>chin tone</small></span>
        </div>
        <i>recorded together through the night</i>
        <div className="research-introduction-flow">
          <span className="research-introduction-scored-epoch"><b>30-second scoring epoch</b><small>one expert stage label</small></span>
          <em>→</em>
          <span><b>Hypnogram</b><small>sleep architecture</small></span>
        </div>
      </div>
    </div>
  );
}

const stagePrimerRows: Array<{ system: string; signal: string; scoring: boolean; cells: string[]; cites: number[] }> = [
  {
    system: "Brain",
    signal: "EEG",
    scoring: true,
    cells: [
      "Occipital alpha (8–12 Hz) with eyes closed; faster beta when alert.",
      "Alpha gives way to slower, low-voltage theta (4–7 Hz).",
      "Sleep spindles (11–16 Hz) and K-complexes on a theta background.",
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
      "Blinks and voluntary rapid movements.",
      "Slow rolling eye movements.",
      "Usually still.",
      "Still.",
      "Bursts of rapid eye movements."
    ],
    cites: [1, 22]
  },
  {
    system: "Chin muscle",
    signal: "EMG",
    scoring: true,
    cells: [
      "Highest tone.",
      "Tone falls.",
      "Lower.",
      "Low.",
      "Lowest of the night: REM atonia."
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

function StagePrimer() {
  const stages = ["Wake", "N1", "N2", "N3", "REM"];
  return (
    <div className="research-stage-primer">
      <p className="research-stage-primer-lede">
        Sleep stages are identified by combinations of brain activity, eye movements and muscle tone. Heart rate and breathing change alongside them, but do not define the stages. These physiological patterns motivate the features; their overlap explains why no single measurement can identify every stage.
      </p>
      <div className="research-stage-primer-scroll">
        <table>
          <thead>
            <tr>
              <th scope="col">System</th>
              {stages.map((stage) => <th scope="col" key={stage} style={{ color: stageColors[stage] }}>{stage}</th>)}
            </tr>
          </thead>
          <tbody>
            {stagePrimerRows.map((row) => (
              <tr key={row.system}>
                <th scope="row">
                  <strong><i className="research-sensor-dot" style={{ background: sensorColors[row.system] }} aria-hidden="true" />{row.system}</strong>
                  <small>{row.signal}</small>
                  <em className={row.scoring ? "is-scoring" : "is-supporting"}>{row.scoring ? "AASM scoring" : "Follows stage"}</em>
                  <span className="research-row-cites">{row.cites.map((number) => <Citation key={number} number={number} />)}</span>
                </th>
                {row.cells.map((cell, index) => <td key={stages[index]}>{cell}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
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

type ExampleMarker = { stage: string; value: number; kind: "target" | "contrast" };

function FeatureDistributionChart({
  feature,
  markers = []
}: {
  feature: FeatureDistribution;
  markers?: ExampleMarker[];
}) {
  const values = [
    ...feature.stages.flatMap((stage) => [stage.p10, stage.p90]),
    ...markers.map((marker) => marker.value)
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
        <span>{featureValue(rawMin)}</span><span>{feature.label}{featureUnits[feature.id] ? ` · ${featureUnits[feature.id]}` : ""}</span><span>{featureValue(rawMax)}</span>
      </div>
      {feature.stages.map((stage) => (
        <div className="research-distribution-row" key={stage.stage}>
          <strong style={{ color: stageColors[stage.stage] }}>{stage.stage}</strong>
          <div className="research-distribution-plot">
            <svg viewBox="0 0 600 40" role="img" aria-label={`${stage.stage}: median ${featureValue(stage.median)}, middle half ${featureValue(stage.p25)} to ${featureValue(stage.p75)}`} preserveAspectRatio="none">
              <line className="range" x1={x(stage.p10)} x2={x(stage.p90)} y1="20" y2="20" />
              <line className="cap" x1={x(stage.p10)} x2={x(stage.p10)} y1="14" y2="26" />
              <line className="cap" x1={x(stage.p90)} x2={x(stage.p90)} y1="14" y2="26" />
              <rect x={x(stage.p25)} y="9" width={Math.max(x(stage.p75) - x(stage.p25), 2)} height="22" fill={stageColors[stage.stage]} />
              <line className="median" x1={x(stage.median)} x2={x(stage.median)} y1="7" y2="33" />
            </svg>
            {markers.filter((marker) => marker.stage === stage.stage).map((marker) => (
              <i
                key={marker.kind}
                className={`research-example-marker is-${marker.kind}`}
                style={{ left: `${(x(marker.value) / 600) * 100}%` }}
                aria-label={`${marker.stage} example epoch: ${featureValue(marker.value)}`}
              />
            ))}
          </div>
          <span>{featureValue(stage.median)}</span>
        </div>
      ))}
      <p className="research-chart-key">
        <span /> 10th to 90th percentile <i /> middle 50% <b /> median
        {markers.map((marker) => (
          <em key={marker.kind}><u className={`research-example-marker is-${marker.kind}`} />{marker.stage} example</em>
        ))}
      </p>
    </div>
  );
}

function formatClock(totalSeconds: number) {
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = Math.round(totalSeconds % 60);
  return `${hours}:${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`;
}

const behaviourLabel = { expected: "Matches the expectation", partly: "Partly matches", unexpected: "Differs from the expectation" };

function FeatureStory({ feature, stageTarget, distribution }: {
  feature: { id: string; stageUse?: string };
  stageTarget: StageFeatureTarget;
  distribution: { stages: Array<{ stage: string; median: number; one_vs_rest_cohens_d: number }> };
}) {
  const mechanism = mechanismFor(feature.id);
  const sign = (d: number) => `${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(2)}`;
  const direction = stageTarget.expect[feature.id] ?? "high";
  const b = featureBehaviour(distribution, stageTarget.stage, direction);
  const verdictBlock = (
    <div className={`research-behaviour is-${b.verdict}`}>
      <h5>Expected {direction === "high" ? "higher" : "lower"} in {stageTarget.stage} <em>{behaviourLabel[b.verdict]}</em></h5>
      <p>{feature.stageUse ? `${feature.stageUse} ` : ""}In this data: highest in {b.highest}, lowest in {b.lowest} ({stageTarget.stage} vs the rest, <abbr title="Standardized difference between this stage and the other stages; positive means higher values in this stage">d</abbr> = {sign(b.d)}).</p>
    </div>
  );
  return (
    <div className="research-feature-story">
      {mechanism ? (
        <div>
          <h5>Physiology · {featureBasis[feature.id].mechanism}</h5>
          <p>{mechanism.text} {mechanism.cites.map((number) => <Citation key={number} number={number} />)}</p>
        </div>
      ) : null}
      {verdictBlock}
    </div>
  );
}

// Unit of each feature as it appears in the distribution (values are raw, before per-night scaling).
const featureUnits: Record<string, string> = {
  "spindle_features.aperiodic_exponent": "spectral slope (unitless)",
  "spindle_features.aperiodic_intercept": "log₁₀ power at the 1/f fit",
  "spindle_features.delta_power": "log₁₀ power, 1–4 Hz",
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
  "spindle_bursts": "seconds per epoch",
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

// Features listed by how directly they follow the scoring rules: AASM rule, then published method, then adapted.
const basisOrder: Record<FeatureBasis, number> = { aasm: 0, published: 1, adapted: 2 };

// Families measured but left out of the adopted model.
const modelCFeatureIds = new Set(data.signal_importance.map((item) => item.id));

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
  const [featureId, setFeatureId] = useState("spindle_bursts");
  const selectedTarget = targetedFeatures.find((item) => item.feature.id === featureId) ?? targetedFeatures[0];
  const family = selectedTarget?.family ?? featureFamilies[0];
  const feature = selectedTarget?.feature ?? family.features[0];
  const matchingExamples = examples?.filter((item) => item.target_stage === stageTarget.stage && item.feature_id === feature.id);
  const targetExample = matchingExamples?.find((item) => item.comparison_role === "target");
  const contrastExample = matchingExamples?.find((item) => item.comparison_role === "contrast");
  const shownExamples = [
    targetExample ? { kind: "target" as const, example: targetExample } : null,
    contrastExample ? { kind: "contrast" as const, example: contrastExample } : null
  ].filter((item): item is { kind: "target" | "contrast"; example: FeatureSignalExample } => item !== null);
  const distribution = data.feature_evidence.distributions.find((item) => item.id === feature.id) ?? data.feature_evidence.distributions[0];
  const isTrack = feature.id.startsWith("track.");
  const method = isTrack ? `${feature.measurement} ${trackDerivation}` : feature.measurement;
  const anchor = feature.id === "spindle_bursts" ? "spindle-bursts" : isTrack ? "tracks" : codeAnchor[family.id];

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
      <div className="research-family-tabs research-stage-tabs" role="tablist" aria-label="Choose an expert sleep-stage label">
        {stageFeatureTargets.map((item) => (
          <button key={item.stage} type="button" role="tab" aria-selected={item.stage === stageId} onClick={() => chooseStage(item)}>
            <span style={{ color: item.stage === stageId ? stageColors[item.stage] : undefined }}>{item.stage}</span>
          </button>
        ))}

      </div>

      <div className="research-feature-picker" role="listbox" aria-label={`${stageId} features`}>
        {targetedFeatures.map((item) => (
          <button key={item.feature.id} type="button" role="option" aria-selected={item.feature.id === feature.id} onClick={() => setFeatureId(item.feature.id)}>
            <strong><i className="research-feature-sensor-dot" aria-hidden="true" style={{ background: item.family.id === "eye-movement" ? sensorColors.Eyes : item.family.id === "muscle-tone" ? sensorColors["Chin muscle"] : item.family.id === "heart" ? sensorColors.Heart : item.family.id === "breathing" ? sensorColors.Breathing : sensorColors.Brain }} />{item.feature.name}</strong>
            <span><BasisTag featureId={item.feature.id} compact />{!modelCFeatureIds.has(item.feature.id) ? <em className="research-not-in-model">not in final model</em> : null}</span>
          </button>
        ))}
      </div>

      <div className="research-feature-result" aria-live="polite">
        <div className="research-feature-result-head">
          <div>
            <h4>{feature.name}</h4>
          </div>
        </div>
        <FeatureStory feature={feature} stageTarget={stageTarget} distribution={distribution} />
        <div className="research-selected-distribution">
          <div className="research-selected-distribution-head">
            <strong>Distribution by expert stage label</strong>
          </div>
          <FeatureDistributionChart
            feature={distribution}
            markers={shownExamples.map(({ kind, example }) => ({ stage: example.stage, value: example.feature_value, kind }))}
          />
        </div>
      </div>

      {shownExamples.length === 0 ? (
        <div className="research-example-loading" role="status">
          {loadFailed ? "The epoch examples could not be loaded." : "Loading example epochs…"}
        </div>
      ) : (
        <div className="research-feature-epoch">
          <div className="research-epoch-heading">
            <strong>Example epochs: {shownExamples.map(({ example }) => example.stage).join(" vs ")}</strong>
          </div>
          <p className="research-plot-caption">Selected epochs illustrate the expected contrast; markers on the distributions above show where their measurements fall.</p>
          {shownExamples.map(({ kind, example }) => (
            <div className="research-epoch-example" key={kind}>
              <div className="research-epoch-example-label">
                <u className={`research-example-marker is-${kind}`} />
                <strong style={{ color: stageColors[example.stage] }}>{example.stage}</strong>
                <span>{feature.name} {featureValue(example.feature_value)} · {example.record_id} · {formatClock(example.epoch_start_sec)}</span>
              </div>
              <SignalTrace signal={example.signal} active />
            </div>
          ))}
        </div>
      )}

      <Tabs label="About this feature" tabs={[
        { id: "calculated", title: "How it is calculated", content: <div className="research-feature-tab"><p>{method}</p>{anchor ? <a className="research-text-link" href={`#code-${anchor}`}>View the Python implementation</a> : null}</div> },
        { id: "fools", title: "What can fool it", content: <div className="research-feature-tab"><p>{family.limitation} {family.sources.map((number) => <Citation key={number} number={number} />)}</p></div> }
      ]} />
    </div>
  );
}

function WindowModelFigure() {
  const tracks = data.source.tracks;
  const features = data.source.features;
  return (
    <div className="research-window-model">
      <div className="research-window-figure" role="img" aria-label={`Window model: ${tracks + features} signals over five epochs, 25 six-second steps, then gradient boosting`}>
        <div className="research-window-epochs">
          {[-2, -1, 0, 1, 2].map((epoch) => (
            <span key={epoch} className={epoch === 0 ? "is-scored" : undefined}>
              <b>{epoch === 0 ? "scored epoch" : `epoch ${epoch > 0 ? "+" : "−"}${Math.abs(epoch)}`}</b>
              <small>{epoch * 30 >= 0 ? `${epoch * 30}` : `−${Math.abs(epoch * 30)}`} to {epoch * 30 + 30} s</small>
              <i>{Array.from({ length: 5 }, (_, step) => <u key={step} />)}</i>
            </span>
          ))}
        </div>
        <div className="research-window-rows">
          <span><strong>{tracks + features} physiological measurements</strong><small>EEG spectrum, spindles, frontal slow waves, occipital alpha, eye activity and chin tone; supplied every six seconds</small></span>
        </div>
        <div className="research-introduction-flow research-window-flow">
          <span><b>{(tracks + features) * 25} inputs</b><small>{tracks + features} signals × 25 steps</small></span>
          <em>→</em>
          <span><b>Gradient boosting</b><small>500 trees, library defaults</small></span>
          <em>→</em>
          <span className="research-introduction-scored-epoch"><b>Stage</b><small>one label per epoch</small></span>
        </div>
      </div>
    </div>
  );
}

function SleepStructure() {
  const matrix = data.sleep_structure.transition_matrix;
  const stages = data.source.stages;
  const bouts = data.sleep_structure.bouts;
  return (
    <div className="research-sleep-structure">
      <div className="research-sleep-structure-copy">
        <p>
          Across all {data.source.subjects} expert hypnograms, stages tend to persist from one epoch to the next, wake usually passes through N1 before deeper sleep, and N3 is entered almost only from N2. An epoch is therefore easier to stage when its neighbours are visible, which is why the model looks at 2.5 minutes around each epoch.
        </p>
        <div className="research-bout-list">
          <strong>Typical bout length</strong>
          {bouts.map((bout) => (
            <span key={bout.stage}>
              <b style={{ color: stageColors[bout.stage] }}>{bout.stage}</b>
              median {bout.median_epochs * 0.5} min · {formatPercent(bout.share_single_epoch)} last one epoch
            </span>
          ))}
        </div>
      </div>
      <div className="research-confusion research-confusion-static research-transition-matrix">
        <span className="research-confusion-axis">Next epoch →</span>
        <div className="research-confusion-head">
          <span />
          {stages.map((stage) => <span key={stage}>{stage}</span>)}
        </div>
        {matrix.map((values, rowIndex) => (
          <div className="research-confusion-row" key={stages[rowIndex]}>
            <strong style={{ color: stageColors[stages[rowIndex]] }}>{stages[rowIndex]}</strong>
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
        <p className="research-confusion-caption">Probability of each stage in the next epoch, given the current stage. Rows sum to 100%; off-diagonal shading is boosted 4×.</p>
      </div>
    </div>
  );
}

function CompactStudyProtocol() {
  const total = data.diagnostics.per_stage.reduce((sum, stage) => sum + stage.support, 0);
  return (
    <div className="research-protocol">
      <div className="research-ablation-table" role="table" aria-label="Datasets and how each is used">
        <div className="research-ablation-row research-data-row is-head" role="row">
          <span role="columnheader">Dataset</span><span role="columnheader">Recordings</span><span role="columnheader">Role</span>
        </div>
        <div className="research-ablation-row research-data-row" role="row">
          <span role="cell"><strong>UCDDB</strong><small className="research-row-note">St Vincent&rsquo;s University Hospital, Dublin <Citation number={3} /></small></span>
          <span role="cell">25 nights<small className="research-row-note">suspected sleep apnea; older R&amp;K scoring</small></span>
          <span role="cell"><b>Early development.</b> Used to build the initial features and staging models <Exp range={[1, 14]} />.</span>
        </div>
        <div className="research-ablation-row research-data-row is-adopted" role="row">
          <span role="cell"><strong>Boston · CinC 2018, development</strong><small className="research-row-note">Massachusetts General Hospital, Boston <Citation number={24} /></small></span>
          <span role="cell">{data.source.subjects} nights<small className="research-row-note">{data.source.epochs.toLocaleString()} epochs</small></span>
          <span role="cell"><b>Development and cross-validation.</b> Five-fold cross-validation split by patient. <Exp ids={[15]} /></span>
        </div>
        <div className="research-ablation-row research-data-row" role="row">
          <span role="cell"><strong>Boston · CinC 2018, held out</strong><small className="research-row-note">Same hospital, set aside before modelling</small></span>
          <span role="cell">{data.generalisation?.holdout_subjects || 200} nights<small className="research-row-note">more apnea: 59% moderate or severe</small></span>
          <span role="cell"><b>Held-out test.</b> Model C fitted on all {data.source.subjects} development patients; {data.generalisation?.holdout_looks ?? 0} evaluation looks. <Exp ids={[30]} /></span>
        </div>
        <div className="research-ablation-row research-data-row" role="row">
          <span role="cell"><strong>Coimbra · ISRUC-Sleep</strong><small className="research-row-note">Coimbra, Portugal; subgroup I; two scorers per night</small></span>
          <span role="cell">{isrucStatus.final_complete ? isrucStatus.evaluated : 100} nights</span>
          <span role="cell"><b>External evaluation.</b> One recorded evaluation look; predictions compared with both scorers.</span>
        </div>
      </div>
      <p className="research-dataset-line"><span>Stage distribution in the development set:</span></p>
      <div className="research-protocol-balance">
        <div className="research-class-stack" role="img" aria-label="Distribution of labeled sleep stages">
          {data.diagnostics.per_stage.map((stage) => (
            <i key={stage.stage} style={{ width: `${(stage.support / total) * 100}%`, background: stageColors[stage.stage] }} title={`${stage.stage}: ${stage.support.toLocaleString()} epochs`} />
          ))}
        </div>
        <div className="research-class-legend">
          {data.diagnostics.per_stage.map((stage) => <span key={stage.stage}><i style={{ background: stageColors[stage.stage] }} />{stage.stage} <strong>{formatPercent(stage.support / total)}</strong></span>)}
        </div>
      </div>
      <p className="research-protocol-source">The development set includes adults aged 18–87 (median 53; 185 men, 135 women), selected to cover the range of apnea severity: 115 with AHI below 5, 69 mild, 96 moderate and 40 severe.</p>
    </div>
  );
}

const ladderIds = ["epoch", "window-all", "adopted", "cnn"];

function ModelExplorer() {
  const ladder = ladderIds.map((id) => data.models.find((item) => item.id === id)).filter((item): item is ModelResult => Boolean(item));
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
        <strong>Macro-F1 on each of the five test folds, models A–D (<SecRef to="model" />)</strong>
      </div>
      <div className="research-model-compare-axis" aria-hidden="true">
        <span />
        <div>{ticks.map((tick) => <small key={tick} style={{ left: position(tick) }}>{tick.toFixed(2)}</small>)}</div>
        <span />
      </div>
      {ladder.map((item) => {
        const folds = item.folds.map((fold) => fold.macro_f1);
        return (
          <div key={item.id} className={`research-model-compare-row${item.id === final ? " is-selected" : ""}`}>
            <strong>{item.name}</strong>
            <div className="research-model-compare-track" role="img" aria-label={`${item.name}: mean ${formatMetric(item.metrics.macro_f1)}, folds ${formatMetric(Math.min(...folds))} to ${formatMetric(Math.max(...folds))}`}>
              {ticks.map((tick) => <i key={tick} className="grid" style={{ left: position(tick) }} />)}
              <i className="baseline" style={{ left: position(baseline.metrics.macro_f1) }} />
              <i className="range" style={{ left: position(Math.min(...folds)), width: `calc(${position(Math.max(...folds))} - ${position(Math.min(...folds))})` }} />
              {folds.map((value, index) => <i key={index} className="fold" style={{ left: position(value) }} />)}
              <i className="mean" style={{ left: position(item.metrics.macro_f1) }} />
            </div>
            <span>{formatMetric(item.metrics.macro_f1)}</span>
          </div>
        );
      })}
      <p className="research-chart-key research-model-compare-key">
        <u className="fold" /> one test fold <u className="mean" /> mean <u className="baseline" /> A, single epoch
      </p>
    </div>
  );
}

function ConfusionExplorer() {
  return (
    <div className="research-confusion-wrap">
      <div className="research-confusion research-confusion-static">
        <span className="research-confusion-axis">Predicted stage →</span>
        <div className="research-confusion-head">
          <span />
          {data.source.stages.map((stage) => <span key={stage}>{stage}</span>)}
          <span title="Of predictions assigned to this stage, the proportion that is correct">Precision</span><span title="Of expert-labelled epochs in this stage, the proportion found by the model">Sensitivity</span><span title="Harmonic mean of precision and sensitivity for this stage">F1</span><span title="Proportion of all other stages correctly excluded">Specificity</span>
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
              <strong style={{ color: stageColors[stage] }}>{stage}</strong>
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
      <p className="research-confusion-caption">Rows: expert stage; columns: predicted stage. Each row sums to 100%. Stage metrics pool all cross-validation predictions; headline scores average the five folds.</p>
    </div>
  );
}

const familyColors: Record<string, string> = {
  spectrum: "#78d8ff",
  spindles: "#6fc4a2",
  topography: "#f0b36d",
  "eye-movement": "#ef8ab5",
  "muscle-tone": "#b49cf0",
  heart: "#ff7d7d",
  breathing: "#9fd36f"
};

// Short signal name (as the models and saliency export use it) → explorer name and family.
const signalInfo: Record<string, { name: string; family: string }> = Object.fromEntries(
  featureFamilies.flatMap((family) => family.features.map((feature) => [feature.id.split(".").pop() as string, { name: feature.name, family: family.id }]))
);

function PhysiologyFindings() {
  return <div className="research-feature-tab">
    <ModelEvidence part="signals" />
  </div>;
}

function PerformanceSummary() {
  const model = data.models.find((item) => item.id === data.headline.diagnostics_model_id) as ModelResult;
  const items = [
    { value: formatMetric(model.metrics.macro_f1), label: "Macro-F1", note: "Equal weight to each stage; balances precision and sensitivity." },
    { value: `${(model.metrics.accuracy * 100).toFixed(1)}%`, label: "Accuracy", note: "Share of all epochs scored correctly; common stages count more." },
    { value: `${(model.metrics.balanced_accuracy * 100).toFixed(1)}%`, label: "Balanced accuracy", note: "Average sensitivity across the five stages." },
    { value: model.metrics.cohen_kappa.toFixed(2), label: "Cohen’s κ", note: "Agreement beyond that expected from the stage frequencies." }
  ];
  return <div className="research-glance-stats research-performance-metrics">{items.map((item) => <div key={item.label}><strong>{item.value}</strong><span><b>{item.label}</b><br />{item.note}</span></div>)}</div>;
}

function IsrucResults() {
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
  const columns = { gridTemplateColumns: "minmax(200px, 2fr) repeat(4, minmax(72px, 1fr))" };
  return <div className="research-ablations">
    <div className="research-evidence-head">
      <h3>Coimbra</h3>
      <p>The adopted 20-signal model trained on Boston recordings, applied to {isrucStatus.evaluated} Coimbra nights. Night 8 lacked frontal EEG and was excluded.</p>
    </div>
    <div className="research-ablation-table" role="table" aria-label="Coimbra external evaluation">
      <div className="research-ablation-row is-head" role="row" style={columns}>
        {["Comparison", "Nights", "Macro-F1", "Accuracy", "κ"].map((label) => <span role="columnheader" key={label}>{label}</span>)}
      </div>
      {rows.map((row) => <div className="research-ablation-row" role="row" style={columns} key={row.evaluation}>
        <span role="cell">{row.label}</span>
        <span role="cell" className="num">{row.n_subjects}</span>
        <span role="cell" className="num">{formatMetric(row.macro_f1)}</span>
        <span role="cell" className="num">{(100 * row.accuracy).toFixed(1)}%</span>
        <span role="cell" className="num">{row.kappa.toFixed(2)}</span>
      </div>)}
    </div>
    <p className="research-ablation-note">Comparisons involving scorer 2 use epochs scored by both scorers. The agreement-only analysis excludes disputed epochs and describes an easier subset, rather than overall performance.</p>
  </div>;
}

function ModelEvidence({ part }: { part: "signals" | "time" }) {
  const measurements = [...data.signal_importance].sort((a, b) => b.importance - a.importance);
  const remainingGroups = [
    { label: "Other EEG measurements", families: ["spectrum", "topography", "spindles"], color: "#78d8ff" },
    { label: "Other eye measurements", families: ["eye-movement"], color: "#ef8ab5" },
    { label: "Other chin measurements", families: ["muscle-tone"], color: "#b49cf0" }
  ].map((group) => {
    const items = measurements.slice(8).filter((item) => group.families.includes(item.family_id));
    return { ...group, count: items.length, importance: items.reduce((sum, item) => sum + item.importance, 0) };
  }).filter((group) => group.count > 0);
  const max = Math.max(...measurements.slice(0, 8).map((item) => item.importance), ...remainingGroups.map((group) => group.importance), .001);
  const measurementColor = (family: string) => family === "eye-movement" ? "#ef8ab5" : family === "muscle-tone" ? "#b49cf0" : "#78d8ff";
  const positive = (value: number) => Math.max(value, 0);
  const fine = data.time_importance.fine;
  const coarse = data.time_importance.epoch;
  const timeMax = Math.ceil(Math.max(...fine.map((item) => item.importance), ...coarse.map((item) => item.importance), .001) * 100) / 100;
  const peak = fine.filter((item) => item.offset_sec >= 0 && item.offset_sec < 30).reduce((a, b) => (b.importance > a.importance ? b : a));
  const coarseAt = (offset: number) => coarse.find((item) => item.offset_sec === offset)?.importance ?? 0;
  const coarseRatio = coarseAt(-30) / Math.max(coarseAt(30), 0.0005);
  return (
    <div className="research-evidence">
      {part === "time" ? <p className="research-evidence-lede">For the timing analysis, measurements are swapped only at the indicated position in the window. The upper plot tests a six-second step; the lower plot tests a whole epoch because those measurements repeat across its five steps. <Exp ids={[21]} /></p> : null}
      <div className="research-evidence-grid is-single">
        {part === "signals" ? <div>
          <p className="research-signal-key"><span><i style={{ background: "#78d8ff" }} /> EEG</span><span><i style={{ background: "#ef8ab5" }} /> Eye activity</span><span><i style={{ background: "#b49cf0" }} /> Chin tone</span><span>Drop in macro-F1 · mean of {data.importance_folds} test folds</span></p>
          <div className="research-importance-bars research-individual-bars is-compact">
            {measurements.slice(0, 8).map((item) => {
              const label = item.id === "track.o_rel_alpha" ? "Occipital alpha · 2-s spectrum" : item.id === "topography_features.occipital_rel_alpha" ? "Occipital alpha · 30-s spectrum" : item.label.replace(" (1 s)", "");
              return <div key={item.id}>
                <span>{label}</span>
                <i><b style={{ width: `${Math.max(item.importance, 0) / max * 100}%`, background: measurementColor(item.family_id) }} /></i>
                <strong>{formatMetric(item.importance)}</strong>
              </div>;
            })}
          </div>
          <div className="research-importance-bars research-individual-bars is-compact research-other-sums">
            {remainingGroups.map((group) => <div key={group.label}>
              <span>{group.label}<small className="research-row-note">{group.count} measurements · summed</small></span>
              <i><b style={{ width: `${Math.max(group.importance, 0) / max * 100}%`, background: group.color }} /></i>
              <strong>{formatMetric(group.importance)}</strong>
            </div>)}
          </div>
          <p className="research-other-sum-note">“Other” bars sum the remaining individual importance scores; they do not measure the effect of shuffling those measurements together.</p>
        </div> : null}
        {part === "time" ? <div className="research-context-plot" role="img" aria-label="Importance of each 6-second step and each epoch across the 150-second window">
          <div className="research-context-epochs" aria-hidden="true">
            {[-2, -1, 0, 1, 2].map((epoch) => <span key={epoch} className={epoch === 0 ? "is-scored" : undefined}>{epoch === 0 ? "scored" : `${epoch > 0 ? "+" : "−"}${Math.abs(epoch)}`}</span>)}
          </div>
          <div className="research-context-row-label">Measurements that change within an epoch · swap one six-second step</div>
          <div className="research-context-y-title">Drop in macro-F1 · mean of {data.time_importance.folds} test folds · shared vertical scale</div>
          <div className="research-context-scaled-row">
          <div className="research-context-y-ticks"><span>{formatMetric(timeMax)}</span><span>{formatMetric(timeMax / 2)}</span><span>0</span></div>
          <div className="research-context-bars">
            {fine.map((item) => (
              <i key={item.offset_sec} className={item.offset_sec >= 0 && item.offset_sec < 30 ? "is-scored" : undefined}
                style={{ height: `${(positive(item.importance) / timeMax) * 100}%` }} title={`${item.offset_sec} s: ${formatMetric(item.importance)}`} />
            ))}
          </div>
          </div>
          <div className="research-context-row-label">Measurements calculated once per 30-second epoch · swap all five repeated six-second values together</div>
          <div className="research-context-scaled-row">
          <div className="research-context-y-ticks"><span>{formatMetric(timeMax)}</span><span>{formatMetric(timeMax / 2)}</span><span>0</span></div>
          <div className="research-context-bars is-epochs">
            {coarse.map((item) => (
              <i key={item.offset_sec} className={item.offset_sec === 0 ? "is-scored" : undefined}
                style={{ height: `${(positive(item.importance) / timeMax) * 100}%` }} title={`${item.offset_sec} s: ${formatMetric(item.importance)}`} />
            ))}
          </div>
          </div>
          <div className="research-context-axis" aria-hidden="true">
            {[-60, -30, 0, 30, 60, 90].map((tick) => <span key={tick} style={{ left: `${((tick + 60) / 150) * 100}%` }}>{tick} s</span>)}
          </div>
          <p className="research-context-note">
            Within the scored epoch, importance peaks {peak.offset_sec}–{peak.offset_sec + 6} s after its start. Outside it, the previous epoch&rsquo;s features count {coarseRatio.toFixed(1)}× as much as the next epoch&rsquo;s. This matches the scoring rules, which decide whether N2 and REM continue by looking at earlier epochs.
          </p>
        </div> : null}
      </div>
    </div>
  );
}

function Generalisation() {
  const adopted = data.models.find((model) => model.id === "adopted") as ModelResult;
  const g = data.generalisation;
  if (!g) return null;
  const held = g.holdout.find((row) => row.config === "g20");
  const heldRef = g.holdout.find((row) => row.config === "epoch19");
  const epochRef = data.models.find((model) => model.id === "epoch") as ModelResult;
  const rows = [
    { label: `Cross-validation, ${data.source.subjects} Boston patients`, note: "final model, each patient tested once", row: { macro_f1: adopted.metrics.macro_f1, kappa: adopted.metrics.cohen_kappa, stage_f1: adopted.stage_f1 } },
    ...(held ? [{ label: `Held-out test, ${g.holdout_subjects} Boston patients`, note: `${g.holdout_looks} evaluation looks`, row: { ...held, stage_f1: undefined as Record<string, number> | undefined } }] : []),
    ...(isrucStatus.final_complete ? isrucStatus.results.filter((row) => ["transfer_cinc_to_isruc_scorer1", "transfer_cinc_to_isruc_scorer2"].includes(row.evaluation)).map((row) => ({ label: `Coimbra, ${row.n_subjects} nights · scorer ${row.evaluation.endsWith("scorer1") ? "1" : "2"}`, note: "Model C · all 20 inputs", row })) : []),
  ];
  return (
    <div className="research-ablations">
      <div className="research-evidence-head">
        <h3>Per-stage scores for each test</h3>
      </div>
      <div className="research-ablation-table" role="table" aria-label="Held-out and cross-lab results">
        <div className="research-ablation-row is-head" role="row">
          <span role="columnheader">Test</span><span role="columnheader">Macro-F1</span><span role="columnheader">κ</span><span role="columnheader" />
          {data.source.stages.map((stage) => <span role="columnheader" key={stage} style={{ color: stageColors[stage] }}>{stage}</span>)}
        </div>
        {rows.map(({ label, note, row }) => (
          <div className="research-ablation-row" role="row" key={label}>
            <span role="cell">{label}<small className="research-row-note">{note}</small></span>
            <span role="cell" className="num">{formatMetric(row.macro_f1)}</span>
            <span role="cell" className="num">{row.kappa.toFixed(2)}</span>
            <span role="cell" />
            {data.source.stages.map((stage) => <span role="cell" className="num" key={stage}>{row.stage_f1 ? (row.stage_f1 as Record<string, number>)[stage].toFixed(2) : "—"}</span>)}
          </div>
        ))}
      </div>
    </div>
  );
}

// Supplementary: the steps from B to the adopted model C, and the swap rounds (experiments #28 and #33–36).
const reductionLabels: Record<string, string> = {
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

function ModelPath() {
  const byId = (id: string) => data.models.find((model) => model.id === id);
  const held = data.generalisation?.holdout.find((row) => row.config === "nocardio27" && row.look === 1);
  const prune = data.reduction?.find((row) => row.config === "prune6");
  const steps = [
    { name: "B · Context", signals: 33, f1: byId("window-all")?.metrics.macro_f1, note: "33 measurements in context, heart and breathing included." },
    { name: "Features in context", signals: 19, f1: byId("window-features")?.metrics.macro_f1, note: "The 19 features alone in context: most of the gain over A is context itself." },
    { name: "27 signals", signals: 27, f1: byId("window")?.metrics.macro_f1, note: `Heart and breathing removed (+0.002 with them).${held ? ` Scored at held-out look 1: ${formatMetric(held.macro_f1)}.` : ""}` },
    { name: "21 signals", signals: 21, f1: prune?.macro_f1, note: "The six least-used inputs removed, no loss." },
    { name: "C · 20 signals", signals: 20, f1: byId("adopted")?.metrics.macro_f1, note: "Spindle bursts in for the 1/f offset and gamma residual: the one validated marker the model uses." }
  ];
  return (
    <div className="research-ablations">
      <p className="research-evidence-lede">The main page shows four models. These are the steps between B and C, each tested on the same folds, and the swap rounds that tried more direct markers. A swap had to stay within 0.005 macro-F1 of the 21-signal model overall and in every fold, and every new feature was first checked against the stage it targets. <Exp ids={[28, 33, 34, 35, 36]} /></p>
      <div className="research-ablation-table" role="table" aria-label="Steps from B to the adopted model">
        <div className="research-ablation-row research-noeeg-row is-head" role="row"><span role="columnheader">Step</span><span role="columnheader">Signals</span><span role="columnheader">Macro-F1</span></div>
        {steps.map((step) => (
          <div className="research-ablation-row research-noeeg-row" role="row" key={step.name}>
            <span role="cell"><strong>{step.name}</strong><small className="research-row-note">{step.note}</small></span>
            <span role="cell" className="num">{step.signals}</span>
            <span role="cell" className="num">{step.f1 !== undefined ? formatMetric(step.f1) : "—"}</span>
          </div>
        ))}
      </div>
      {data.reduction?.length ? (
        <>
          <h5 className="research-agreement-title">Swap rounds</h5>
          <div className="research-ablation-table" role="table" aria-label="Feature reduction and swap rounds">
            <div className="research-ablation-row research-noeeg-row is-head" role="row"><span role="columnheader">Variant</span><span role="columnheader">Signals</span><span role="columnheader">Macro-F1</span></div>
            {data.reduction.filter((row) => reductionLabels[row.config]).map((row) => (
              <div className={`research-ablation-row research-noeeg-row${row.config === "swapS2" ? " is-adopted" : ""}`} role="row" key={row.config}>
                <span role="cell">{reductionLabels[row.config]}</span>
                <span role="cell" className="num">{row.signals}</span>
                <span role="cell" className="num">{formatMetric(row.macro_f1)}</span>
              </div>
            ))}
          </div>
          <p className="research-ablation-note">Rejected candidate markers, checked before any model used them: REM eye bursts (fired mostly on wake blinks), K-complex events and slow eye movements (did not separate their stage as well as the inputs they would replace). Sawtooth theta and alpha dropout passed but the model barely used them.</p>
        </>
      ) : null}
    </div>
  );
}

function SignalAblations() {
  const configs = [...new Set(data.signal_ablations.map((row) => row.config))];
  const rows = configs.flatMap((config) => {
    const epoch = data.signal_ablations.find((row) => row.config === config && row.decoder === "epoch-only");
    const hmm = data.signal_ablations.find((row) => row.config === config && row.decoder === "hmm");
    return epoch && hmm ? [{ config, label: epoch.label, epoch: epoch.macro_f1, hmm: hmm.macro_f1, kappa4: epoch.kappa4, kappa4Hmm: hmm.kappa4, stages: epoch.stage_f1 }] : [];
  });
  const full = rows.find((row) => row.config === "window33");  // all 33 signals
  const brain = rows.find((row) => row.config === "nocardio27");
  const cardio = rows.filter((row) => row.config.startsWith("cardio"));
  if (!full || !brain || !cardio.length) return null;
  const bestCardio = cardio.reduce((a, b) => (Math.max(b.epoch, b.hmm) > Math.max(a.epoch, a.hmm) ? b : a));
  return (
    <div className="research-ablations">
      <div className="research-evidence-head">
        <h3>Do heart and breathing earn their place?</h3>
        <p>
          The same model retrained with groups of signals removed or used alone, on the same folds. Removing heart and breathing changes macro-F1 only from {formatMetric(full.epoch)} to {formatMetric(brain.epoch)}; with EEG, EOG and chin EMG present they are redundant, so the final model leaves them out. On their own they still carry stage information, more so with additional heart features and a longer window: the best heart-and-breathing model reaches {formatMetric(Math.max(bestCardio.epoch, bestCardio.hmm))} macro-F1 and a four-class κ of {Math.max(bestCardio.kappa4, bestCardio.kappa4Hmm).toFixed(2)}. Published heart-rate models report four-class κ of 0.61–0.68, using whole-night context and thousands of nights; the whole-night decoder (a hidden Markov model over stage-to-stage transitions, used only in this comparison) helps here far more than on the full model.
        </p>
      </div>
      <div className="research-ablation-table" role="table" aria-label="Window model with signal groups removed or isolated">
        <div className="research-ablation-row is-head" role="row">
          <span role="columnheader">Signals</span>
          <span role="columnheader">Macro-F1</span>
          <span role="columnheader">+ decoder</span>
          <span role="columnheader">4-class κ</span>
          {data.source.stages.map((stage) => <span role="columnheader" key={stage} style={{ color: stageColors[stage] }}>{stage}</span>)}
        </div>
        {rows.map((row) => (
          <div className="research-ablation-row" role="row" key={row.config}>
            <span role="cell">{row.label}</span>
            <span role="cell" className="num">{formatMetric(row.epoch)}</span>
            <span role="cell" className="num">{formatMetric(row.hmm)}</span>
            <span role="cell" className="num">{Math.max(row.kappa4, row.kappa4Hmm).toFixed(2)}</span>
            {data.source.stages.map((stage) => <span role="cell" className="num" key={stage}>{row.stages[stage].toFixed(2)}</span>)}
          </div>
        ))}
      </div>
      <p className="research-ablation-note">{ABLATION_NOTE}</p>
      <h5 className="research-agreement-title">Without the EEG, how far can heart and breathing go? <Exp ids={[31]} /></h5>
      <p className="research-evidence-lede">
        Heart rate and breathing change slowly with sleep stage, over minutes, so a model without EEG needs a much longer window than 2.5 minutes. A recurrent network over the whole night, with breath-by-breath respiration and blood oxygen added, raises the heart-and-breathing model from 0.533 to 0.649 macro-F1 (four-class κ from 0.46 to 0.60). The table shows what each sensor achieves on its own, which is the relevant question for a wearable device.
      </p>
      <div className="research-ablation-table" role="table" aria-label="Heart and breathing models without the EEG">
        <div className="research-ablation-row research-noeeg-row is-head" role="row">
          <span role="columnheader">Sensors and model</span><span role="columnheader">Macro-F1</span><span role="columnheader">4-class κ</span>
        </div>
        {[
          ["ECG + belts, trees over 10 minutes (row above)", "0.533", "0.46"],
          ["ECG + belts + oximeter, whole-night network", "0.649", "0.60"],
          ["ECG + oximeter only", "0.604", "0.54"],
          ["Effort belts only", "0.589", "0.53"],
          ["ECG only", "0.549", "0.47"]
        ].map(([label, f1, k]) => (
          <div className="research-ablation-row research-noeeg-row" role="row" key={label}>
            <span role="cell">{label}</span><span role="cell" className="num">{f1}</span><span role="cell" className="num">{k}</span>
          </div>
        ))}
      </div>
      <p className="research-ablation-note">Whole-night results are averaged over random seeds; the best single network scores 0.640. Published heart-rate models reach four-class κ 0.61–0.68 using thousands of training nights, compared with about 255 per fold here. Oxygen saturation and breath-by-breath respiration account for most of the gain; additional heart-rate variability measures add nothing once heart rate is included. Without EEG, N1 F1 stays near 0.45.</p>
    </div>
  );
}

const ABLATION_NOTE = "Per-stage columns are F1 without the decoder. Four-class κ merges N1 and N2 into light sleep (the better of with and without the decoder), as heart-rate staging studies report it. \"Unbalanced\" trains with equal class weights, which favours accuracy and κ over N1.";

function BlackBoxAgreement() {
  const saliency = saliencyData as unknown as SaliencyData;
  const stages = data.source.stages;
  // Body systems, as in the stage table in the introduction; heart and breathing together, since D uses neither.
  const systems = [
    { id: "eeg", label: "Brain (EEG)", color: familyColors.spectrum, families: ["spectrum", "spindles", "topography"] },
    { id: "eog", label: "Eyes (EOG)", color: familyColors["eye-movement"], families: ["eye-movement"] },
    { id: "emg", label: "Chin (EMG)", color: familyColors["muscle-tone"], families: ["muscle-tone"] },
    { id: "cardio", label: "Heart and breathing", color: familyColors.heart, families: ["heart", "breathing"] }
  ];
  // D: share of each stage's F1 drop from each system (negative drops count as zero).
  const trees = stages.map((stage) => {
    const drops = systems.map((system) => system.families.reduce((sum, id) => sum + Math.max(data.feature_importance.find((item) => item.family_id === id)?.stage_drops[stage] ?? 0, 0), 0));
    const total = drops.reduce((a, b) => a + b, 0) || 1;
    return drops.map((value) => value / total);
  });
  // E: share of |gradient × input| from each system's signals, correctly classified epochs of each stage.
  const cnn = stages.map((stage) => {
    const row = saliency.by_stage.find((item) => item.stage === stage);
    return systems.map((system) => saliency.signals.reduce((sum, name, index) => sum + (system.families.includes(signalInfo[name]?.family) ? row?.signal_share[index] ?? 0 : 0), 0));
  });
  const topOf = (row: number[]) => row.indexOf(Math.max(...row));
  const agree = stages.filter((_, k) => topOf(trees[k]) === topOf(cnn[k])).length;
  const pct = (value: number) => Math.round(value * 100);
  return (
    <div className="research-agreement">
      <p className="research-evidence-lede">
        An exploratory comparison asks which body systems each model uses. C is measured by shuffling inputs; D is estimated from gradients. The models use different inputs and windows, and these attribution methods are not directly equivalent. Agreement is descriptive support, not validation of the features or their physiological interpretation. <Exp ids={[20]} />
      </p>
      <div className="research-agreement-table is-paired" style={{ gridTemplateColumns: `minmax(0, 1.5fr) repeat(${stages.length}, minmax(0, 1fr))` }}>
        <span className="research-agreement-key">% share · <b>C</b> / D</span>
        {stages.map((stage) => <b key={stage} style={{ color: stageColors[stage] }}>{stage}</b>)}
        {systems.map((system, f) => (
          <div key={system.id} className="research-agreement-row">
            <span><u style={{ background: system.color }} />{system.label}</span>
            {stages.map((stage, k) => (
              <em key={stage} style={{ "--cell-alpha": Math.min(trees[k][f] * 2, 1).toFixed(3) } as React.CSSProperties}><b>{pct(trees[k][f])}</b> / {pct(cnn[k][f])}</em>
            ))}
          </div>
        ))}
      </div>
      <p className="research-agreement-takeaway">
        Both models rely most on the EEG in {agree} of 5 stages, and both shift toward eye movements and chin tone in REM (C {formatPercent(trees[4][1] + trees[4][2])}, D {formatPercent(cnn[4][1] + cnn[4][2])}). D assigns {formatPercent(Math.min(...cnn.map((row) => row[3])))}–{formatPercent(Math.max(...cnn.map((row) => row[3])))} to heart and breathing, partly because gradients spread some credit to every input; C does not use these signals and scores higher. Shading follows C; each column sums to 100% for each model.
      </p>
    </div>
  );
}

type PublishedModel = {
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

function publishedModels(): PublishedModel[] {
  const adopted = data.models.find((model) => model.id === "adopted") as ModelResult;
  const cnn = data.models.find((model) => model.id === "cnn") as ModelResult;
  const size = data.model_size;
  const stage = (model: ModelResult) => data.source.stages.map((item) => model.stage_f1[item]);
  return [
    {
      name: "C · 20 signals (this page)", kind: "Gradient-boosted trees", ours: true,
      macroF1: adopted.metrics.macro_f1, stageF1: stage(adopted), kappa: adopted.metrics.cohen_kappa, accuracy: adopted.metrics.accuracy,
      size: `${size.trees} trees, ${size.splits.toLocaleString()} splits on ${size.inputs_used} of ${size.inputs} named inputs`,
      learned: size.nodes, learnedShort: size.nodes.toLocaleString(), trainShort: `${size.train_subjects} per fold`, testShort: `5-fold CV, ${data.source.subjects} Boston`,
      training: `${size.train_subjects} subjects per fold; ${Math.round(size.fit_seconds / 60)} min on a laptop CPU`,
      inputs: "20 EEG, eye and chin measurements, every 6 s over 150 s",
      evaluation: `5-fold CV, ${data.source.subjects} Boston records`
    },
    {
      name: "D · Attention CNN (this page)", kind: "Convolutional network", ours: true,
      macroF1: cnn.metrics.macro_f1, stageF1: stage(cnn), kappa: cnn.metrics.cohen_kappa, accuracy: cnn.metrics.accuracy,
      size: `${size.cnn_weights.toLocaleString()} weights`, learned: size.cnn_weights, learnedShort: size.cnn_weights.toLocaleString(), trainShort: `${size.train_subjects} per fold`, testShort: "Same folds",
      training: `${size.train_subjects} subjects per fold`, inputs: "Same signals at 1 s, 90 s", evaluation: "Same folds"
    },
    {
      name: "U-Sleep", kind: "Fully convolutional U-Net", macroF1: 0.79, stageF1: [0.84, 0.6, 0.84, 0.81, 0.87],
      size: "≈3.1 million weights", learned: 3.1e6, learnedShort: "≈3.1 M", trainShort: "≈15,600 people pooled from 13 datasets", testShort: "100 other Boston", training: "≈15,600 participants from 13 datasets",
      inputs: "Raw EEG + EOG at 128 Hz", evaluation: "100 held-out Boston records", cite: 26
    },
    {
      name: "U-Time", kind: "Fully convolutional U-Net", macroF1: 0.77, size: "≈1.1 million weights", learned: 1.1e6, learnedShort: "≈1.1 M", trainShort: "Boston only (one model per dataset, 7 in the paper)", testShort: "Other Boston",
      training: "Trained per dataset", inputs: "Raw EEG", evaluation: "Boston, more records than U-Sleep's test set", cite: 27
    },
    {
      name: "SLEEPNET", kind: "Recurrent network", kappa: 0.795, accuracy: 0.858, size: "Deep RNN (not reported)", learnedShort: "Not reported", trainShort: "≈10,000 MGH", testShort: "1,000 MGH",
      training: "≈10,000 MGH patients", inputs: "EEG", evaluation: "1,000 MGH recordings", cite: 28
    },
    {
      name: "Human scorers (AASM program)", kind: "Reference", accuracy: 0.826,
      stageF1: undefined, size: "Agreement with the majority of more than 2,500 scorers", learnedShort: "—", trainShort: "2,500+ scorers", testShort: "1,800 epochs",
      training: "Most with 3 or more years of experience", inputs: "Full PSG, visual scoring",
      evaluation: "1,800 epochs from 9 records", cite: 29
    }
  ];
}

// Short labels and offsets for the simplified tree models drawn on the trade-off chart.
const sweepLabels: Record<string, { text: string; dx: number; dy: number }> = {
  tree_d4: { text: "1 tree, depth 4", dx: 8, dy: 4 },
  tree_d6: { text: "depth 6", dx: 8, dy: 4 },
  tree_d8: { text: "depth 8", dx: 8, dy: 4 },
  hgb_l2_i200: { text: "single splits", dx: -6, dy: 16 },
  hgb_l4_i100: { text: "4-leaf", dx: -8, dy: -8 },
  hgb_l8_i100: { text: "8-leaf", dx: 8, dy: 12 }
};

function TradeoffChart({ models, sweep: showSweep = false }: { models: PublishedModel[]; sweep?: boolean }) {
  const points = models.filter((model) => model.learned && model.macroF1);
  const sweep = showSweep ? [...data.complexity].sort((a, b) => a.learned_values - b.learned_values) : [];
  const xMin = 1;
  const xMax = 7;
  const yMin = showSweep ? 0.55 : 0.7;
  const yMax = 0.8;
  const x = (value: number) => 60 + ((Math.log10(value) - xMin) / (xMax - xMin)) * 500;
  const y = (value: number) => 200 - ((value - yMin) / (yMax - yMin)) * 180;
  const path = sweep.map((row, index) => `${index === 0 ? "M" : "L"}${x(row.learned_values).toFixed(1)} ${y(row.macro_f1).toFixed(1)}`).join(" ");
  return (
    <figure className="research-tradeoff">
      <svg viewBox="0 0 600 240" role="img" aria-label="Macro-F1 against number of learned values, log scale">
        {(showSweep ? [0.6, 0.65, 0.7, 0.75, 0.8] : [0.7, 0.75, 0.8]).map((tick) => (
          <g key={tick}>
            <line className="grid" x1="60" x2="560" y1={y(tick)} y2={y(tick)} />
            <text className="tick" x="52" y={y(tick) + 4} textAnchor="end">{tick.toFixed(2)}</text>
          </g>
        ))}
        {[1, 2, 3, 4, 5, 6, 7].map((power) => (
          <text key={power} className="tick" x={x(10 ** power)} y="222" textAnchor="middle">{["", "10", "100", "1k", "10k", "100k", "1M", "10M"][power]}</text>
        ))}
        <text className="axis" x="310" y="238" textAnchor="middle">learned values (log scale)</text>
        <text className="axis" x="14" y="110" textAnchor="middle" transform="rotate(-90 14 110)">macro-F1, Boston</text>
        {sweep.length ? <path className="sweep" d={path} /> : null}
        {sweep.filter((row) => sweepLabels[row.learner]).map((row) => (
          <g key={row.learner} className="is-sweep">
            <circle cx={x(row.learned_values)} cy={y(row.macro_f1)} r={3.5} />
            <text className="label small" x={x(row.learned_values) + sweepLabels[row.learner].dx} y={y(row.macro_f1) + sweepLabels[row.learner].dy}
              textAnchor={sweepLabels[row.learner].dx < 0 ? "end" : "start"}>{sweepLabels[row.learner].text}</text>
          </g>
        ))}
        {points.map((model) => (
          <g key={model.name} className={model.ours ? "is-ours" : undefined}>
            <circle cx={x(model.learned as number)} cy={y(model.macroF1 as number)} r={model.ours ? 6 : 5} />
            <text className="label" x={x(model.learned as number) + 9} y={y(model.macroF1 as number) + (model.name.includes("Attention") ? 14 : -8)}>{model.name.replace(" (this page)", "")}</text>
          </g>
        ))}
      </svg>
      <figcaption>{showSweep ? "The thin line is the final model's inputs with simpler learners, down to one readable tree." : "Learned values: split thresholds and leaf values for the trees, weights for the networks."}</figcaption>
    </figure>
  );
}

function ModelSimplification() {
  const rows = data.complexity;
  if (!rows.length) return null;
  const by = (id: string) => rows.find((row) => row.learner === id);
  const full = by("hgb_l31_i100");
  const stumps = by("hgb_l2_i200");
  const d4 = by("tree_d4");
  return (
    <div className="research-simplify">
      <p className="research-comparison-lede">
        The same 675 inputs with simpler learners, on the same folds. Accuracy declines slowly until the model is reduced to a single tree. Boosting with one split per tree gives an additive model, in which each input&rsquo;s effect is a single curve; it scores {formatMetric(stumps?.macro_f1 ?? NaN)} with {stumps?.learned_values.toLocaleString()} parameters. A single depth-4 tree can be read in full but scores {formatMetric(d4?.macro_f1 ?? NaN)}.
      </p>
      <div className="research-ablation-table" role="table" aria-label="The window model with simpler learners">
        <div className="research-ablation-row research-simplify-row is-head" role="row">
          <span role="columnheader">Learner</span><span role="columnheader">Learned values</span><span role="columnheader">Macro-F1</span><span role="columnheader">κ</span><span role="columnheader">N1 F1</span>
        </div>
        {rows.map((row) => (
          <div key={row.learner} className={`research-ablation-row research-simplify-row${row.learner === "hgb_l31_i100" ? " is-adopted" : ""}`} role="row">
            <span role="cell">{row.label}</span>
            <span role="cell" className="num">{row.learned_values.toLocaleString()}</span>
            <span role="cell" className="num">{formatMetric(row.macro_f1)}</span>
            <span role="cell" className="num">{row.kappa.toFixed(2)}</span>
            <span role="cell" className="num">{row.stage_f1.N1.toFixed(2)}</span>
          </div>
        ))}
      </div>
      <TradeoffChart models={publishedModels().filter((model) => model.ours)} sweep />
      <div className="research-readable-tree">
        <h5>The depth-4 tree, read aloud (fold 0, top three levels)</h5>
        <ol>
          <li><b>Is the beta share low 6–12 s into the epoch?</b> Low beta means the cortex is not aroused.
            <ol>
              <li><b>Yes → was occipital alpha low 12 s before?</b> Then N3 if the eye channel carries slow waves leaking from the frontal EEG, otherwise N2.</li>
              <li><b>Yes, but alpha present →</b> mostly N2.</li>
            </ol>
          </li>
          <li><b>No (beta higher) → is chin tone low 18–24 s in?</b>
            <ol>
              <li><b>Yes →</b> REM: an active-looking EEG with an atonic chin.</li>
              <li><b>No →</b> Wake or N1, split by the 1/f slope and theta share: a flatter spectrum and less theta mean Wake.</li>
            </ol>
          </li>
        </ol>
        <p>Each branch corresponds to a standard scoring cue: arousal (beta), posterior alpha, slow waves, REM atonia and the EEG slowing at sleep onset. The full model combines 500 trees of this kind to go from {formatMetric(d4?.macro_f1 ?? NaN)} to {formatMetric(full?.macro_f1 ?? NaN)}.</p>
      </div>
    </div>
  );
}

function SubjectVariability() {
  const subjects = [...data.diagnostics.subject_metrics].sort((a, b) => a.macro_f1 - b.macro_f1);
  const min = subjects[0];
  const median = subjects[Math.floor(subjects.length / 2)];
  const max = subjects[subjects.length - 1];
  return (
    <div className="research-subjects">
      <div>
        <h3>Performance varies across subjects</h3>
        <p>Each bar is one Boston development patient scored in their test fold, ordered from lowest to highest macro-F1.</p>
      </div>
      <div className="research-subject-plot">
        <div className="research-subject-axis" aria-hidden="true">
          <span className="research-subject-axis-title">Macro-F1</span>
          {[1, 0.75, 0.5, 0.25, 0].map((tick) => <span key={tick} style={{ bottom: `${tick * 100}%` }}>{tick.toFixed(2)}</span>)}
        </div>
        <div className="research-subject-strip" role="img" aria-label={`Subject macro-F1 from ${min.macro_f1.toFixed(3)} to ${max.macro_f1.toFixed(3)}`}>
          {subjects.map((subject) => (
            <i key={subject.subject} style={{ height: `${Math.max(subject.macro_f1 * 100, 1)}%` }} title={`${subject.subject}: ${subject.macro_f1.toFixed(3)}`} />
          ))}
        </div>
      </div>
      <div className="research-subject-range">
        <span>lowest <strong>{min.macro_f1.toFixed(3)}</strong> ({min.subject})</span>
        <span>median <strong>{median.macro_f1.toFixed(3)}</strong> · {subjects.length} subjects, sorted</span>
        <span>highest <strong>{max.macro_f1.toFixed(3)}</strong> ({max.subject})</span>
      </div>
    </div>
  );
}

// Experiment chronology (docs/experiment-history.md). Dates from git commits and artifact file times.
type HistoryStep = { n: number; date: string; question: string; finding: string; decision: string; score?: number; holdout?: boolean };
const historyPhases: Array<{ phase: string; dataset: string; steps: HistoryStep[] }> = [
  {
    phase: "A first pipeline",
    dataset: "UCDDB · 25 apnea patients",
    steps: [
      { n: 1, date: "Aug 13", question: "Can named features stage sleep?", finding: "Feature recipe for EEG, EOG, chin, ECG and breathing; per-task reference diagnostics.", decision: "Build on hand-made, named features." },
      { n: 2, date: "Aug 14", question: "How good is an epoch classifier?", finding: "Bagged trees 0.608; with HSMM smoothing 0.629.", decision: "Keep a temporal second stage.", score: 0.629 },
      { n: 3, date: "Aug 21", question: "Do sequence networks beat smoothing?", finding: "30-s CNN and 6-s TCN about 0.58–0.60; feature and signal experts 0.39–0.47.", decision: "Smoothing stays ahead with 25 subjects." },
      { n: 4, date: "Aug 22", question: "Does a hierarchical network help?", finding: "0.40 raw, 0.62 with a linear head.", decision: "No gain over smoothing." },
      { n: 5, date: "Sep 9", question: "Publish the first dashboard.", finding: "Bagged trees plus smoothers on UCDDB.", decision: "Dashboard v1 on GitHub Pages." }
    ]
  },
  {
    phase: "Audit: bugs before models",
    dataset: "UCDDB",
    steps: [
      { n: 6, date: "Oct 16", question: "Audit the dashboard for this talk.", finding: "Best smoother 0.67; copy overstated results.", decision: "Audit models and features in parallel.", score: 0.67 },
      { n: 7, date: "Oct 16", question: "Why stuck near 0.66?", finding: "REM and N1 stage codes were swapped.", decision: "Fix the label map." },
      { n: 8, date: "Oct 17", question: "Physiology or a bug?", finding: "18 of 29 features drifted against the labels, up to 6.5 minutes by morning (binned by sample count, not time).", decision: "Bin features by time." },
      { n: 9, date: "Oct 17", question: "Is the learner the bottleneck?", finding: "Default gradient boosting 0.680 vs the inherited bagged trees 0.609.", decision: "Default boosting, never tuned." },
      { n: 10, date: "Oct 17", question: "Is the second stage leaking?", finding: "Smoother trained on in-sample probabilities.", decision: "Out-of-fold probabilities only." },
      { n: 11, date: "Oct 17", question: "Does amplifier gain limit us?", finding: "Per-subject scaling already cancels it; gain-free ratios add +0.028.", decision: "Add ratio features." },
      { n: 12, date: "Oct 17", question: "Why no heart-rate effect?", finding: "R-peaks were fine; cleaned intervals and rhythm-structure measures separate N3 from REM.", decision: "Add heart, breathing and coupling features." },
      { n: 13, date: "Oct 17", question: "Why do some patients fail?", finding: "Flat recording dropouts were scored as Wake.", decision: "Mask dropouts." },
      { n: 14, date: "Oct 17", question: "Result of the fixes?", finding: "0.669 → 0.744, still limited by 25 R&K-scored subjects.", decision: "Find a bigger dataset.", score: 0.744 }
    ]
  },
  {
    phase: "A better dataset",
    dataset: "Boston · 120 → 320 patients",
    steps: [
      { n: 15, date: "Oct 17", question: "Move to Boston?", finding: "0.749 on 120 records; calibrated µV, frontal and occipital leads, AASM labels. Trained on UCDDB, tested on Boston: 0.679.", decision: "Switch to Boston.", score: 0.749 },
      { n: 16, date: "Oct 17", question: "Better model or more data?", finding: "Learning curve still rising; tuning gave nothing.", decision: "Download to 320 patients." },
      { n: 17, date: "Oct 17", question: "Can we cut features?", finding: "33 → 16 features at no loss with a decoder; new event features +0.006.", decision: "A feature must add ≥ 0.01 or replace two." },
      { n: 18, date: "Oct 17", question: "Why do Boston patients fail?", finding: "Dead leads, ECG artefact in the EEG, stages with few epochs.", decision: "Report stage counts with per-patient scores." }
    ]
  },
  {
    phase: "Context, then simplify",
    dataset: "Boston · 320 patients",
    steps: [
      { n: 19, date: "Oct 18", question: "Does real context beat smoothing?", finding: "1-s tracks and 6-s features over 150 s: 0.779 with a decoder vs 0.751.", decision: "Adopt the window model; retire smoothers and networks.", score: 0.779 },
      { n: 20, date: "Oct 18", question: "What does a network on the same signals do?", finding: "Attention CNN 0.769; relies on the same physiology.", decision: "Keep as the black-box comparison." },
      { n: 21, date: "Oct 19", question: "Is the start of the epoch special?", finding: "No: a tree tie-break artifact. Evidence peaks 6–18 s in; the previous epoch counts 3.8× the next.", decision: "Report the corrected profile." },
      { n: 22, date: "Oct 19", question: "How do we compare?", finding: "U-Sleep 0.79; human scorers agree on 83% of epochs.", decision: "Add the comparison section." },
      { n: 23, date: "Oct 21", question: "Is the decoder still needed?", finding: "+0.002 with the window, and it costs N1.", decision: "Drop it: one idea, fewer parts.", score: 0.777 },
      { n: 24, date: "Oct 21", question: "Do heart and breathing earn their place?", finding: "Removing them costs 0.002; alone (6 signals) they reach 0.44–0.49.", decision: "Kept for now; their stand-alone value is the no-EEG case." },
      { n: 25, date: "Oct 21", question: "How simple can the learner be?", finding: "Boosted single splits (an additive model) 0.751 with a tenth of the values; one readable depth-4 tree 0.596.", decision: "Keep default boosting; show the sweep and the readable tree." },
      { n: 26, date: "Oct 21", question: "Can heart and breathing alone do better?", finding: "All 12 heart and breathing signals over 10.5 min: 0.533, four-class κ 0.46 (published heart-rate models 0.61–0.68 with whole-night networks).", decision: "Longer context helps; the rest needs whole-night models and more data." },
      { n: 28, date: "Oct 21", question: "Keep heart and breathing in the final model?", finding: "They add 0.002 with EEG, eyes and chin present.", decision: "Drop them: 27 signals, 0.775.", score: 0.775 },
      { n: 29, date: "Oct 21", question: "Do a cleaner lead or dropping bad segments help?", finding: "No: 0.7756 vs 0.7755; excluding 1.6–3.3% of epochs leaves the score unchanged.", decision: "Keep all epochs; the model already copes." },
      { n: 30, date: "Oct 21", question: "Final test on 200 held-out patients (look #1)", finding: "0.730 for the adopted model; every model drops by about the same amount on this harder, apnea-heavy group. One record that had failed to build was repaired and the same batch rerun (moved < 0.001).", decision: "Reported as is; no choice made from it.", score: 0.730, holdout: true },
      { n: 31, date: "Oct 22", question: "How far can heart and breathing go without the EEG?", finding: "Whole-night recurrent network with breath-by-breath breathing and oxygen: 0.649 (four-class κ 0.60), up from 0.533. ECG + oximeter 0.604, belts 0.589, ECG 0.549.", decision: "Reported as the wearable outlook; the final model stays EEG-based." }
    ]
  },
  {
    phase: "Interpretability audit",
    dataset: "Boston · 320 patients",
    steps: [
      { n: 32, date: "Oct 22", question: "Do the features behave as designed?", finding: "Each feature checked against its intended stage: N3 and wake features hold; the spindle-band share does not separate N2 (d 0.02); the single EOG channel also carries frontal slow waves.", decision: "Keep the published designs; show observed behaviour beside each (2.2)." },
      { n: 33, date: "Oct 22", question: "Can the model lose inputs without losing accuracy?", finding: "Removing the six least-used inputs: 0.7754 vs 0.7755 with 21 signals. Spindle measures added nothing (+0.0006 at best).", decision: "Keep the 21-signal set; look for more direct markers.", score: 0.775 },
      { n: 34, date: "Oct 22", question: "Can vague inputs be swapped for direct markers?", finding: "Swaps passed the accuracy rule, but the K-complex proxy ran the wrong way and the cleaned eye signal barely separated REM.", decision: "Check what a feature measures before testing it." },
      { n: 35, date: "Oct 22", question: "Which candidate markers mean what they claim?", finding: "Validated first: sawtooth theta (REM) and alpha dropout (N1) passed; REM eye bursts, K-complex events and slow eye movements failed.", decision: "Only validated markers enter the swaps." },
      { n: 36, date: "Oct 22", question: "Swap in the validated markers?", finding: "Passed the rule: spindle bursts for 1/f offset and gamma residual (−0.0026), sawtooth theta (−0.0019), alpha dropout (−0.0012). Only spindle bursts is used by the model (3rd-largest N2 importance); the others rank last.", decision: "Adopt the spindle swap: model C, 20 signals.", score: 0.773 },
      { n: 37, date: "Oct 22", question: "Final test of model C (look #2)", finding: "0.729 on the same 200 held-out patients (27-signal model at look #1: 0.730).", decision: "Reported as is; no choice made from it.", score: 0.729, holdout: true }
    ]
  }
];

// Where each history step's evidence lives (repo paths; artifacts/ is local, not in git).
const historyWhere: Record<number, string> = {
  1: "artifacts/reference_diagnostics", 2: "artifacts/paper_matched_benchmarks_5fold_hsmm", 3: "artifacts/sequence_cnn_*",
  4: "artifacts/hierarchical_epoch_staging", 5: "git 9c671ea", 6: "artifacts/staging_diagnostics", 7: "exploration/coordinator",
  8: "exploration/model-audit", 9: "exploration/coordinator", 10: "exploration/model-audit", 11: "exploration/ratio-normalization",
  12: "exploration/hrv-features", 13: "exploration/subject-failures", 14: "exploration/coordinator", 15: "exploration/cinc2018",
  16: "exploration/cinc-models", 17: "exploration/cinc-features", 18: "exploration/cinc-subjects", 19: "exploration/cinc-context",
  20: "exploration/cinc-context", 21: "scripts/run_window_time_importance.py", 22: "dashboard comparison", 23: "scripts/run_window_hmm_staging.py",
  24: "staging_window_model/ablations", 25: "scripts/run_window_complexity.py", 26: "staging_window_model/cardio_full*",
  27: "scripts/run_window_transfer.py", 28: "staging_window_model/ablations", 29: "exploration/signal-quality", 31: "exploration/no-eeg", 30: "holdout/LEDGER.csv", 32: "showcase explorer (2.2)", 33: "staging_window_model/cinc2018/feature_reduction", 34: "feature_reduction (round 2)", 35: "feature_reduction/round4_validation.csv", 36: "feature_reduction (rounds 3–4)", 37: "holdout/LEDGER.csv"
};

const sectionNames: Record<string, string> = {
  staging: "1.1 Sleep staging", stages: "1.2 The physiology of each stage", night: "S3 Sleep structure", "supp-models": "S2 Model development",
  data: "2.1 Data and testing", features: "2.2 Features", model: "2.3 Model", testing: "2.1 Data and testing",
  evidence: "3.2 Features", performance: "3.1 Model", robustness: "3.3 Testing",
  "history-details": "S5 Experiment history", "supp-simplify": "S2 Simpler models"
};

// A link to another section, written as "§2.4 Testing" so it reads as a place on the page.
function SecRef({ to }: { to: string }) {
  return <a className="research-secref" href={`#${to}`}>§{sectionNames[to]}</a>;
}

// Experiment numbers attached to the claim they support.
function Exp({ ids, range }: { ids?: number[]; range?: [number, number] }) {
  if (range) return <span className="research-exp-inline"><a href={`#exp-${range[0]}`}>exp. #{range[0]}–{range[1]}</a></span>;
  return <span className="research-exp-inline">exp. {(ids ?? []).map((n, i) => <a key={n} href={`#exp-${n}`}>{i ? ", " : ""}#{n}</a>)}</span>;
}

function ExpRefs({ ids }: { ids: number[] }) {
  return (
    <span className="research-exp-refs" aria-label="Experiments behind this result">
      Experiments {ids.map((n) => <a key={n} href={`#exp-${n}`}>#{n}</a>)}
    </span>
  );
}

// Dot colour = what the score was measured on.
const scoreKinds = {
  ucddb: { label: "UCDDB cross-validation (25 patients)", color: "#91a9ce" },
  cinc: { label: "Boston cross-validation", color: "#78d8ff" },
  holdout: { label: "Held-out Boston patients (two evaluation looks)", color: "#f0b36d" }
};
const scoreKind = (step: HistoryStep) => (step.holdout ? "holdout" : step.n <= 14 ? "ucddb" : "cinc");

function HowItWasRun() {
  return (
    <div className="research-workflow">
      <p className="research-comparison-lede">I set the research questions and made the decisions; AI coding sessions did most of the implementation and ran the experiments. Three working rules kept the work fast and the results verifiable.</p>
      <ol className="research-glance-takeaways">
        <li><b>One session per question.</b> A coordinating session keeps the goals, the decision log and this page. Each question (normalisation, heart-rate variability, poorly scored patients, signal quality, a second lab, a no-EEG model) has its own session with its own files, so several experiments can run in parallel without interfering.</li>
        <li><b>Adoption criteria set in advance.</b> Each experiment lists its variants and the criterion for adoption (about +0.01 macro-F1, better in 4 of 5 folds) before the results are seen. Every feature needs a physiological rationale, and a new feature is checked before any model uses it: it must separate its intended stage in the predicted direction, at least as well as the input it replaces.</li>
        <li><b>Every experiment logged.</b> Each experiment is numbered when it is declared, then recorded below with its date, finding, decision and the folder that holds its evidence. The log is checked before each new run to avoid repeating work, and claims on this page link back to it. The held-out test has a separate log of scoring runs.</li>
      </ol>
    </div>
  );
}

function ExperimentHistory() {
  const steps = historyPhases.flatMap((phase) => phase.steps);
  const last = Math.max(...steps.map((step) => step.n));
  const scored = steps.filter((step) => step.score !== undefined);
  const lo = 0.6;
  const hi = 0.8;
  const xOf = (n: number) => `${((n - 1) / (last - 1)) * 100}%`;
  const markers = historyPhases.slice(1).map((phase) => ({ n: phase.steps[0].n, label: phase.phase }));
  return (
    <div className="research-history">
      <div className="research-history-progress" role="img" aria-label={`Best macro-F1 over time, from ${scored[0].score} to ${scored[scored.length - 1].score}`}>
        {[0.6, 0.7, 0.8].map((tick) => <i key={tick} className="tick" style={{ bottom: `${((tick - lo) / (hi - lo)) * 100}%` }}>{tick.toFixed(1)}</i>)}
        {markers.map((marker, index) => (
          <em key={marker.n} className={`research-history-marker${index % 2 ? " is-low" : ""}${index === markers.length - 1 ? " is-last" : ""}`} style={{ left: `calc(${xOf(marker.n)} - 0.6%)` }}><small>#{marker.n} {marker.label}</small></em>
        ))}
        {scored.map((step) => (
          <span key={step.n} style={{ left: xOf(step.n), bottom: `${(((step.score as number) - lo) / (hi - lo)) * 100}%`, "--dot": scoreKinds[scoreKind(step)].color } as React.CSSProperties}>
            <b>{(step.score as number).toFixed(3)}</b><small>#{step.n}</small>
          </span>
        ))}
      </div>
      <p className="research-chart-key research-history-key">
        {Object.entries(scoreKinds).map(([key, kind]) => <em key={key}><u style={{ background: kind.color }} />{kind.label}</em>)}
      </p>
      <div className="research-history-phases">
        {historyPhases.map((phase) => (
          <section key={phase.phase} className="research-history-phase">
            <header><h4>{phase.phase}</h4><small>{phase.dataset}</small></header>
            <ol>
              {phase.steps.map((step) => (
                <li key={step.n} id={`exp-${step.n}`}>
                  <span className="research-history-n">{step.n}</span>
                  <span className="research-history-date">{step.date}</span>
                  <div>
                    <strong>{step.question}</strong>
                    <p>{step.finding} <em>{step.decision}</em></p>
                    {historyWhere[step.n] ? <code className="research-history-where">{historyWhere[step.n]}</code> : null}
                  </div>
                </li>
              ))}
            </ol>
          </section>
        ))}
      </div>
    </div>
  );
}

function DiscussionExplorer() {
  return <div className="research-conclusion">
    <p className="research-takeaway">Physiological measurements combined with temporal context provide a credible offline staging baseline: a default tree ensemble retains useful performance when transferred to a second hospital. Its physiological assumptions can be checked against the recorded waveforms.</p>
    <ul className="research-conclusion-objectives">
      <li><Objective n={1} /> The reliance on EEG background activity is consistent with research linking spectral slope to arousal. My slope calculation differs from the published implementation, so this is converging evidence rather than a replication. The weak N2 separation of broad spindle-band power supports measuring defining events directly instead of assuming a physiological name guarantees specificity. <Citation number={17} /> <Citation number={1} /></li>
      <li><Objective n={2} /> In these experiments, temporal context contributed more than changing to the tested neural-network architecture. A compact set of explicit measurements can therefore be a competitive design choice. Published systems such as U-Sleep still provide a stronger benchmark for broad validation; different test cohorts prevent claiming equivalent performance. <Citation number={26} /></li>
      <li><Objective n={3} /> Coimbra supports portability of the full input set, while the difficult held-out Boston nights show that transfer of an average score is not enough. The different scores against Coimbra’s two scorers also fit established evidence that the reference labels themselves vary. This strengthens the case for patient-level evaluation and multiple scorers, without attributing every model error to ambiguous labels. <Citation number={5} /></li>
    </ul>
    <h4>Limitations</h4>
    <ul className="research-limits">
      <li><b>Uneven reliability.</b> N1 and some individual nights remain difficult, even when the overall score is useful.</li>
      <li><b>Partial interpretability.</b> Named measurements show what the model uses, but proxies, mixed EOG/EEG activity and an ensemble of trees limit explanations of individual decisions.</li>
      <li><b>Limited validation scope.</b> One external dataset supports transfer, not broad clinical reliability. Future epochs and whole-night scaling also limit the current method to offline scoring.</li>
    </ul>
    <h4>Next steps</h4>
    <ul className="research-limits">
      <li><b>Test the 8-leaf ensemble on final Model C.</b> Check whether the earlier performance–complexity trade-off holds with the final 20 inputs, especially for N1 and difficult patients.</li>
      <li><b>Validate the physiological explanations.</b> Review difficult nights with a scorer and test whether event detectors and individual explanations agree with the waveforms.</li>
      <li><b>Confirm broader reliability.</b> Test the fixed model in further patient groups and recording systems, with attention to per-patient performance.</li>
    </ul>
  </div>;
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
    citation: "Younes M, Raneri J, Hanly P. Staging Sleep in Polysomnograms: Analysis of Inter-Scorer Variability. Journal of Clinical Sleep Medicine. 2016;12:885–894.",
    href: "https://pubmed.ncbi.nlm.nih.gov/27070243/",
    note: "Primary analysis showing that much scoring disagreement comes from genuinely equivocal epochs, especially adjacent-stage boundaries."
  },
  {
    number: 6,
    citation: "Šušmáková K, Krakovská A. Discrimination ability of individual measures used in sleep stages classification. Artificial Intelligence in Medicine. 2008;44:261–277.",
    href: "https://doi.org/10.1016/j.artmed.2008.07.005",
    note: "Original comparison of 818 polysomnographic measurements; delta/beta power was the strongest single five-stage measure, but remained insufficient on its own."
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
    note: "Original five-stage study quantifying how EOG and chin-EMG features complement EEG-based staging."
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
    note: "Fully convolutional network trained on about 15,600 participants; reports per-dataset F1, including held-out CinC 2018 records."
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
  }
];

const technicalAppendixEntries = [
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
    note: "Ratios from the same channel and epoch cancel between-recording differences in gain and electrode contact, so these features enter the model without per-subject scaling.",
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
    note: "The group array contains subject identifiers, preventing epochs from one night from leaking into both development and test data.",
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
    note: "Standardization uses the training subjects only. Steps before the start or after the end of the recording are zero, which is the training mean.",
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

function TechnicalAppendix() {
  return (
    <div className="research-technical-appendix" id="technical-appendix">
      <div className="research-paper-disclosures research-code-disclosures">
        {technicalAppendixEntries.map((entry) => (
          <PaperDisclosure key={entry.id} title={entry.title} summary={entry.summary} className="research-code-disclosure">
            <div className="research-code-source"><span>Canonical source</span><strong>{entry.source}</strong></div>
            <pre id={`code-${entry.id}`}><code>{entry.code}</code></pre>
            <p>{entry.note}</p>
          </PaperDisclosure>
        ))}
      </div>
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
            <a href={reference.href} target="_blank" rel="noreferrer">{reference.citation}</a>
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

// ---------------------------------------------------------------- page structure: summary, contents, tabs, expandable evidence

function Tabs({ label, tabs }: { label: string; tabs: Array<{ id: string; title: string; content: React.ReactNode }> }) {
  const [active, setActive] = useState(tabs[0].id);
  const current = tabs.find((tab) => tab.id === active) ?? tabs[0];
  return (
    <div className="research-tabs">
      <div className="research-tabs-list" role="tablist" aria-label={label}>
        {tabs.map((tab) => (
          <button key={tab.id} type="button" role="tab" aria-selected={tab.id === current.id} onClick={() => setActive(tab.id)}>{tab.title}</button>
        ))}
      </div>
      <div className="research-tabs-panel" role="tabpanel">{current.content}</div>
    </div>
  );
}

function Evidence({ summary, children, id }: { summary: string; children: React.ReactNode; id?: string }) {
  return (
    <details className="research-evidence-disclosure" id={id}>
      <summary><span>{summary}</span></summary>
      <div className="research-evidence-disclosure-body">{children}</div>
    </details>
  );
}

function Block({ id, title, takeaway, refs, children }: { id: string; title: string; takeaway: React.ReactNode; refs?: number[]; children: React.ReactNode }) {
  return (
    <div className="research-results-block research-block" id={id}>
      <div className="research-tool-heading"><h3>{title}</h3>{refs ? <ExpRefs ids={refs} /> : null}</div>
      <p className="research-takeaway">{takeaway}</p>
      {children}
    </div>
  );
}

// Open a closed <details> when a link points inside it (experiment history, code appendix).
function useOpenOnHash() {
  useEffect(() => {
    const open = () => {
      const id = window.location.hash.slice(1);
      if (!id) return;
      const target = document.getElementById(id);
      let node: HTMLElement | null = target;
      while (node) {
        if (node instanceof HTMLDetailsElement) node.open = true;
        node = node.parentElement;
      }
      target?.scrollIntoView({ block: "start", behavior: "instant" });
    };
    open();
    window.addEventListener("hashchange", open);
    return () => window.removeEventListener("hashchange", open);
  }, []);
}

// ---------------------------------------------------------------- objectives that thread the page
const objectives = [
  { n: 1, short: "Physiological interpretation", line: "Connect each measurement to sleep physiology, check its observed behaviour, and distinguish plausible cues from what the model actually uses." },
  { n: 2, short: "A simple model", line: "Use decision trees with default settings, and keep an addition only if it clearly improves accuracy." },
  { n: 3, short: "Independent validation", line: "Separate training patients from test patients, report held-out scoring history, and examine transfer to other hospitals." }
];

function Objective({ n }: { n: 1 | 2 | 3 }) {
  const item = objectives[n - 1];
  return <span className="research-objective-chip" title={item.line}><b>{n}</b>{item.short}</span>;
}

function SubHeading({ number, title, objective, refs }: { number: string; title: string; objective?: 1 | 2 | 3; refs?: number[] }) {
  return (
    <div className="research-tool-heading">
      <h3><span className="research-subnumber">{number}</span>{title}</h3>
      {objective ? <Objective n={objective} /> : null}
      {refs ? <ExpRefs ids={refs} /> : null}
    </div>
  );
}

// ---------------------------------------------------------------- 2.3 model
function ModelSection() {
  const size = data.model_size;
  const models = [
    { name: "A · Single epoch", inputs: "19 epoch-averaged features", window: "30 s", purpose: "Baseline without neighbouring epochs", refs: [16, 17] },
    { name: "B · Context", inputs: "33 measurements every 6 s; heart and breathing included", window: "150 s", purpose: "Test the gain from temporal context", refs: [19] },
    { name: "C · Adopted", inputs: "20 EEG, eye and chin measurements every 6 s; spindle-burst marker added", window: "150 s", purpose: "Retain performance with fewer, physiologically motivated inputs", refs: [28, 33, 36] },
    { name: "D · Attention CNN", inputs: "B’s 14 one-second signals, including heart and breathing", window: "90 s", purpose: "Compare learned signal combinations with the tree models", refs: [20] }
  ];
  return (
    <div className="research-model-section">
      <p className="research-takeaway">Sleep-stage cues become easier to interpret when their surrounding pattern is visible. Model C combines measurements from the scored epoch and its neighbours using decision trees: thresholds on named signals that can capture both brief events and the continuity of sleep.</p>
      <WindowModelFigure />
      <p>Measurements are scaled within each night, and class-balanced training gives rare stages more weight. The use of future epochs and whole-night scaling makes this an offline model.</p>
      <div className="research-models-compared">
        <h5>Models compared</h5>
        <div className="research-model-table-wrap">
          <table className="research-model-table">
            <thead><tr><th scope="col">Model</th><th scope="col">Inputs</th><th scope="col">Window</th><th scope="col">Question</th></tr></thead>
            <tbody>{models.map((model) => <tr key={model.name} className={model.name === "C · Adopted" ? "is-adopted" : undefined}>
              <th scope="row">{model.name}</th><td>{model.inputs}</td><td>{model.window}</td><td>{model.purpose} <Exp ids={model.refs} /></td>
            </tr>)}</tbody>
          </table>
        </div>
      </div>
      <Evidence summary="Implementation details and model reduction">
        <p>The adopted learner uses histogram gradient boosting with {size.trees} trees of up to 31 leaves. Tuning did not provide a measurable gain. Trees handle missing values without imputation and do not require inputs to share units. Input reduction and feature substitutions are documented in <SecRef to="supp-models" />; the performance cost of simplifying the ensemble to smaller trees or a single readable tree is shown in <SecRef to="supp-simplify" />.</p>
      </Evidence>
    </div>
  );
}

// ---------------------------------------------------------------- 2.4 testing
function TestingProtocol() {
  return (
    <div className="research-model-section">
      <p className="research-takeaway" id="testing">The data serve three roles: early feature development in Dublin, model development and held-out testing in Boston, and external testing in Coimbra. Splitting by patient tests whether the model works on another person; Coimbra tests whether that performance carries to another hospital.</p>
    </div>
  );
}

// ---------------------------------------------------------------- 3.3 robustness, compact
function ExternalTestingDetails() {
  const heldRef = data.generalisation?.holdout.find((row) => row.config === "epoch19");
  const epochRef = data.models.find((row) => row.id === "epoch") as ModelResult;
  return <>
    <p>The held-out Boston cohort contains more sleep apnea. The reference model also falls from {formatMetric(epochRef.metrics.macro_f1)} to {heldRef ? formatMetric(heldRef.macro_f1) : "—"}; this is consistent with a patient-mix effect but does not exclude overfitting.</p>
    <Generalisation />
    <IsrucResults />
  </>;
}

function RobustnessSummary() {
  const adopted = data.models.find((model) => model.id === "adopted") as ModelResult;
  const g = data.generalisation;
  const held = g?.holdout.find((row) => row.config === "g20");
  const heldRef = g?.holdout.find((row) => row.config === "epoch19");
  const epochRef = data.models.find((model) => model.id === "epoch") as ModelResult;
  const coimbra = isrucStatus.final_complete ? (isrucStatus.results as Array<{ evaluation: string; macro_f1: number }>).find((row) => row.evaluation === "transfer_cinc_to_isruc_scorer1") : undefined;
  const coimbraPatients = (isrucStatus as unknown as { subject_macro_f1_scorer1?: number[] }).subject_macro_f1_scorer1 ?? [];
  const spread = (values: number[]) => {
    if (!values.length) return undefined;
    const sorted = [...values].sort((a, b) => a - b);
    const quantile = (p: number) => {
      const index = (sorted.length - 1) * p;
      const low = Math.floor(index);
      return sorted[low] + (sorted[Math.ceil(index)] - sorted[low]) * (index - low);
    };
    return { min: sorted[0], q1: quantile(.25), median: quantile(.5), q3: quantile(.75), max: sorted[sorted.length - 1] };
  };
  const groups = [
    { title: "Model C · all 20 inputs", color: "#78d8ff", rows: [
      { label: "Boston · cross-validation", note: `${data.source.subjects} patients`, value: adopted.metrics.macro_f1, spread: spread(data.diagnostics.subject_metrics.map((row) => row.macro_f1)) },
      { label: "Boston · held-out test", note: `${g?.holdout_subjects ?? 200} patients · ${g?.holdout_looks ?? 2} evaluation looks`, value: held?.macro_f1, spread: holdoutDistribution.verified ? spread(holdoutDistribution.subject_macro_f1) : undefined },
      { label: "Coimbra · external test", note: `${isrucStatus.final_complete ? isrucStatus.evaluated : 100} patients · 1 recorded evaluation look`, value: coimbra?.macro_f1, spread: spread(coimbraPatients) }
    ] },

  ];
  const ticks = [0, 0.2, 0.4, 0.6, 0.8, 1];
  const position = (value: number) => `${value * 100}%`;
  return (
    <div className="research-robustness">
      <div className="research-transfer-chart" aria-label="Macro-F1 across patient cohorts, grouped by model inputs">
        <div className="research-model-compare-axis" aria-hidden="true"><span>Macro-F1</span><div>{ticks.map((tick) => <small key={tick} style={{ left: position(tick) }}>{tick.toFixed(2)}</small>)}</div><span /></div>
        {groups.map((group) => <div className="research-transfer-group" key={group.title}>
          <h4 style={{ color: group.color }}>{group.title}</h4>
          {group.rows.map((row) => <div className="research-model-compare-row" key={row.label}>
            <strong>{row.label}<small className="research-row-note">{row.note}</small></strong>
            <div className="research-model-compare-track" role="img" aria-label={`${row.label}: ${row.value === undefined ? "result not available" : formatMetric(row.value)} macro-F1`}>
              {ticks.map((tick) => <i key={tick} className="grid" style={{ left: position(tick) }} />)}
              {row.spread ? <>
                <i className="transfer-whisker" style={{ left: position(row.spread.min), width: position(row.spread.max - row.spread.min), background: group.color }} />
                {[row.spread.min, row.spread.max].map((value, index) => <i key={index} className="transfer-cap" style={{ left: position(value), background: group.color }} />)}
                <i className="transfer-box" style={{ left: position(row.spread.q1), width: position(row.spread.q3 - row.spread.q1), borderColor: group.color }} />
                <i className="transfer-median" style={{ left: position(row.spread.median), background: group.color }} />
              </> : null}
              {row.value !== undefined ? <i className="research-transfer-dot" style={{ left: position(row.value), background: group.color }} /> : null}
            </div>
            <span>{row.value === undefined ? "—" : formatMetric(row.value)}{row.spread ? <small className="research-row-note">{row.spread.min.toFixed(2)}–{row.spread.max.toFixed(2)}</small> : null}</span>
          </div>)}
        </div>)}

        <p className="research-chart-key research-transfer-key">
          <span><u className="legend-score" /> headline macro-F1</span>
          <span><u className="legend-box" /> middle 50% of patients</span>
          <span><u className="legend-median" /> patient median</span>
          <span><u className="legend-whisker" /> patient min–max</span>
        </p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- comparison, compact
function ComparisonCompact() {
  const models = publishedModels();
  const fold = (value?: number) => (value === undefined ? "—" : formatMetric(value));
  return (
    <div className="research-comparison">
      <div className="research-ablation-table" role="table" aria-label="Comparison with published models">
        <div className="research-ablation-row research-compare-row is-head" role="row">
          <span role="columnheader">Model</span><span role="columnheader">Macro-F1</span><span role="columnheader">κ / acc.</span><span role="columnheader">Learned values</span><span role="columnheader">Training patients</span><span role="columnheader">Tested on</span>
        </div>
        {models.map((model) => (
          <div key={model.name} className={`research-ablation-row research-compare-row${model.ours ? " is-adopted" : ""}`} role="row">
            <span role="cell"><strong>{model.name.replace(" (this page)", "")}</strong>{model.cite ? <> <Citation number={model.cite} /></> : null}<small className="research-row-note">{model.kind}</small></span>
            <span role="cell" className="num">{fold(model.macroF1)}</span>
            <span role="cell" className="num">{model.kappa ? model.kappa.toFixed(2) : "—"} / {model.accuracy ? formatPercent(model.accuracy) : "—"}</span>
            <span role="cell" className="num">{model.learnedShort}</span>
            <span role="cell"><small>{model.trainShort}</small></span>
            <span role="cell"><small>{model.testShort}</small></span>
          </div>
        ))}
      </div>
      <p className="research-protocol-source">Published models used different test patients; these are reference results, not head-to-head comparisons. Human scorer agreement is also from a separate sample.</p>
    </div>
  );
}

function PublishedModelSize() {
  const models = publishedModels();
  const ours = models[0];
  const cnn = models[1];
  return <div className="research-comparison">
      <TradeoffChart models={models} />
      <p className="research-protocol-source">
        Parameter count is not the whole difference: D has fewer weights ({cnn.learned?.toLocaleString()}) than C has learned values ({ours.learned?.toLocaleString()}). The difference is in what the parameters represent. Each tree split is a threshold on one named signal at one time, whereas a network combines all signals through learned filters. The comparison is approximate, because the published networks were tested on other records, and inter-scorer agreement comes from a small expert sample. C&rsquo;s lower accuracy is partly a result of class weighting, which improves N1 at some cost to the common stages.
      </p>
  </div>;
}

function StudySummary() {
  const adopted = data.models.find((model) => model.id === "adopted") as ModelResult;
  const coimbraResult = isrucStatus.results.find((row) => row.evaluation === "transfer_cinc_to_isruc_scorer1");
  const pct = (value: number) => `${Math.round(value * 100)}%`;
  const n1 = data.diagnostics.per_stage.find((row) => row.stage === "N1");
  const n3 = data.diagnostics.per_stage.find((row) => row.stage === "N3");
  const epoch = data.models.find((model) => model.id === "epoch") as ModelResult;
  const features = data.models.find((model) => model.id === "window-features") as ModelResult;
  const held = data.generalisation?.holdout.find((row) => row.config === "g20");
  return (
    <section className="research-section research-summary" id="summary">
      <SectionHeading title="Abstract" />
      <div className="research-summary-copy research-abstract">
        <p>
          <b>Background.</b> Sleep staging from polysomnography underlies most sleep diagnoses, but manual scoring is slow, and expert scorers agree on only 83% of epochs (63% for N1). Deep networks report strong staging performance; understanding which physiological information drives a decision remains a separate challenge.
        </p>
        <p>
          <b>Objective.</b> To test three aims: connect inputs to physiology, keep the model simple, and assess performance beyond its training patients:
        </p>
        <div className="research-objectives">
          <ol>
            {objectives.map((item) => <li key={item.n}><Objective n={item.n as 1 | 2 | 3} /><span>{item.line}</span></li>)}
          </ol>
        </div>
        <p>
          <b>Methods.</b> {data.source.subjects} overnight recordings from the Boston dataset. {data.source.tracks + data.source.features} measurements from the EEG, EOG and chin EMG, supplied every six seconds, over the scored epoch and two epochs on either side. Gradient-boosted decision trees with default settings. Testing: five-fold cross-validation split by patient, a held-out set of {data.generalisation?.holdout_subjects || 200} Boston recordings with two recorded evaluation looks, and external evaluation on {isrucStatus.final_complete ? isrucStatus.evaluated : 100} Coimbra recordings.
        </p>
        <p>
          <b>Results.</b> Macro-F1 {formatMetric(adopted.metrics.macro_f1)} (κ {adopted.metrics.cohen_kappa.toFixed(2)}) in cross-validation, with N1 recall {pct(n1?.recall ?? 0)} and N3 recall {pct(n3?.recall ?? 0)}. Context accounted for most of the gain ({formatMetric(epoch.metrics.macro_f1)} → {formatMetric(features.metrics.macro_f1)}).{held ? <> Held-out set: {formatMetric(held.macro_f1)}.</> : null} {isrucStatus.final_complete && coimbraResult ? <>Coimbra ({coimbraResult.n_subjects} nights, scorer 1): {formatMetric(coimbraResult.macro_f1)}.</> : null} The best published network scores 0.79 on a different Boston test set, with about 100 times more parameters. <Citation number={26} />
        </p>
        <p>
          <b>Conclusion.</b> Named physiological measurements support useful staging performance with inspectable inputs. The strongest result is the gain from context; the main limitations are N1, variability between patients, and incomplete independence of external testing. Physiological interpretation guides error investigation, but individual explanations still need verification.
        </p>
      </div>
    </section>
  );
}

export function StagingResearchDashboard() {
  useOpenOnHash();
  const adopted = data.models.find((model) => model.id === "adopted") as ModelResult;
  const n1 = data.diagnostics.per_stage.find((row) => row.stage === "N1");
  const subjects = data.diagnostics.subject_metrics.map((row) => row.macro_f1);
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
          <SubHeading number="2.1" title="Data and testing" objective={3} />
          <TestingProtocol />
          <CompactStudyProtocol />
        </div>
        <div className="research-method-block" id="features">
          <SubHeading number="2.2" title="Features: each stage's physiology, measured" objective={1} />
          <p className="research-takeaway">Each feature translates a physiological staging cue into a measurement, then checks whether it varies across stages as expected. The final model retains {data.source.tracks + data.source.features} EEG, eye and chin measurements every six seconds; heart and breathing candidates are also shown. A plausible physiological rationale is the starting point, not evidence that a feature works.</p>
          <div className="research-signal-key research-feature-sensors" aria-label="Measurement colors, matching the physiology table">
            {[
              { name: "Brain", label: "EEG" }, { name: "Eyes", label: "EOG" },
              { name: "Chin muscle", label: "EMG" }, { name: "Heart", label: "ECG" },
              { name: "Breathing", label: "Respiratory effort" }
            ].map((sensor) => <span key={sensor.name}><i style={{ background: sensorColors[sensor.name] }} />{sensor.label}</span>)}
          </div>
          <div className="research-feature-legend" aria-label="Feature origins">
            <span><em className="research-basis-tag is-aasm">AASM rule</em> measures a waveform, rhythm or muscle-tone cue used in manual staging</span>
            <span><em className="research-basis-tag is-published">Published method</em> follows a published calculation and its standard parameters</span>
            <span><em className="research-basis-tag is-adapted">Adapted</em> uses published physiology with my choices of detector, frequency band or time window</span>
          </div>
          <PhysiologyExplorer />
        </div>
        <div className="research-method-block" id="model">
          <SubHeading number="2.3" title="Model: the signals in context" objective={2} />
          <ModelSection />
        </div>
      </section>

      <section className="research-section" id="results">
        <SectionHeading title="3 Results" />
        <div className="research-results-block research-block" id="performance">
          <SubHeading number="3.1" title="Model: how well it scores" objective={2} />
          <p className="research-takeaway">Context provides the largest improvement, and reducing the inputs to Model C preserves most of that gain: {formatMetric(adopted.metrics.macro_f1)} macro-F1 in patient-separated cross-validation. N1 remains the main weakness, with {formatPercent(n1?.recall ?? 0)} sensitivity. The comparisons below show the gain, the remaining stage errors, and where published models stand.</p>
          <PerformanceSummary />
          <Tabs label="Performance views" tabs={[
            { id: "ladder", title: "Model comparisons", content: <ModelExplorer /> },
            { id: "errors", title: "Stage errors and metrics", content: <ConfusionExplorer /> },
            { id: "published", title: "Published models", content: <div id="comparison"><ComparisonCompact /></div> }
          ]} />
        </div>
        <div className="research-results-block research-block" id="evidence">
          <SubHeading number="3.2" title="Which sleep physiology does the model capture?" objective={1} />
          <p className="research-takeaway">These analyses ask which measurements Model C uses and when it needs them. Values are swapped between test examples while the stage labels stay unchanged; the fall in macro-F1 measures reliance. The tabs examine individual measurements and positions within the input window. A small drop means little additional reliance, since other inputs may carry similar information.</p>
          <Tabs label="What the model relies on" tabs={[
            { id: "signals", title: "Individual measurements", content: <PhysiologyFindings /> },
            { id: "time", title: "Why neighbouring epochs matter", content: <ModelEvidence part="time" /> }
          ]} />
        </div>
        <div className="research-results-block research-block" id="robustness">
          <SubHeading number="3.3" title="Testing: does it hold up?" objective={3} />
          <p className="research-takeaway">Model C scores lower on the held-out Boston patients, while its Coimbra score is closer to Boston cross-validation. The same 20 inputs are used throughout, suggesting useful transfer across these two hospitals alongside substantial variation between patients. Boston’s held-out cohort has two recorded evaluation looks; Coimbra has one recorded evaluation look.</p>
          <RobustnessSummary />
        </div>
      </section>

      <section className="research-section research-discussion" id="discussion">
        <SectionHeading title="4 Discussion and conclusion" />
        <DiscussionExplorer />
      </section>

      <section className="research-section" id="supplementary">
        <SectionHeading title="Supplementary" />
        <Evidence id="supp-patients" summary="S1 Patient variation and validation details">
          <h4>Variation between patients</h4><SubjectVariability />
          <div id="supp-testing"><h4>Held-out testing and per-stage scores</h4><ExternalTestingDetails /></div>
        </Evidence>
        <Evidence id="supp-models" summary="S2 Model development, simplification and size">
          <h4>From 33 inputs to 20</h4><ModelPath />
          <div id="supp-simplify"><h4>Simpler models</h4><ModelSimplification /></div>
          <div id="supp-published-size"><h4>Published models and parameter counts</h4><PublishedModelSize /></div>
        </Evidence>
        <Evidence id="supp-importance" summary="S3 Signal reliance and sleep structure">
          <h4>Interpreting signal importance</h4>
          <p>Importance is the fall in macro-F1 after swapping one measurement’s complete input window between test epochs, keeping the stage labels and all other inputs unchanged ({data.importance_folds} folds). The trained model is scored again without being refitted. Drops are not additive; correlated measurements can substitute for one another, so a short bar indicates little additional reliance. EOG also contains frontal EEG, so eye-channel reliance is not specific evidence of eye movements. This measures model dependence, not a proven mechanism or an individual explanation.</p>
          <div id="night"><h4>Stage transitions and bout lengths</h4><SleepStructure /></div>
          <div id="supp-network"><h4>Exploratory neural-network attribution</h4><BlackBoxAgreement /></div>
        </Evidence>
        <Evidence id="supp-cardio" summary="S4 Heart and breathing: alternative sensor models">
          <SignalAblations />
        </Evidence>
        <Evidence id="history-details" summary="S5 Experiment history and implementation">
          <HowItWasRun /><div id="history"><ExperimentHistory /></div>
          <div id="supp-code"><h4>Python implementation</h4><TechnicalAppendix /></div>
        </Evidence>
      </section>
      <ReferencesSection />
    </main>
  );
}
