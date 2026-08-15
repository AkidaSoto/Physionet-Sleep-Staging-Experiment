export const homeIntro = {
  title: "PhysioNet Sleep Explorer"
};

export const homeBranches = [
  {
    id: "abstract",
    href: "/abstract",
    title: "Abstract",
    accent: "warm",
    stage: ["Problem", "Dataset", "Method"],
    hero: "Translate",
    note: "clinical observation to measurable features"
  },
  {
    id: "apnea",
    href: "/apnea",
    title: "Sleep Apnea",
    accent: "cool",
    stage: ["Airflow", "Effort", "SpO2"],
    hero: "Respiratory",
    note: "event reasoning"
  },
  {
    id: "staging",
    href: "/staging",
    title: "Sleep Staging",
    accent: "neutral",
    stage: ["EEG", "EOG", "EMG"],
    hero: "Sleep States",
    note: "signal to classification"
  },
  {
    id: "about",
    href: "/about",
    title: "About Me",
    accent: "muted",
    stage: ["Background", "Role", "Links"],
    hero: "Background",
    note: "portfolio context"
  }
] as const;
