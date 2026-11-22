// Shared ink, accent, modality, and stage colors across figures, tables, and explorers.

export const INK = "#1b1a17";
export const INK_2 = "#4a4843";
export const INK_3 = "#6e6c65";
export const RULE = "#dedbd2";
export const ACCENT = "#1a4373";

export const STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"] as const;

export const modalityInk = {
  eeg: "#1e293b",
  eog: "#0e7490",
  emg: "#b45309"
} as const;

export const sensorColors: Record<string, string> = {
  Brain: modalityInk.eeg,
  Eyes: modalityInk.eog,
  "Chin muscle": modalityInk.emg,
  Heart: "#9f1239",
  Breathing: "#7e22ce"
};

export const familyColors: Record<string, string> = {
  spectrum: modalityInk.eeg,
  spindles: modalityInk.eeg,
  topography: modalityInk.eeg,
  "eye-movement": modalityInk.eog,
  "muscle-tone": modalityInk.emg,
  heart: sensorColors.Heart,
  breathing: sensorColors.Breathing
};

// Validated with the dataviz palette checker on the paper surface (matches --stage-* in globals.css).
export const stageColors: Record<string, string> = {
  Wake: "#e34948",
  N1: "#eda100",
  N2: "#1baf7a",
  N3: "#6250d6",
  REM: "#008300"
};
