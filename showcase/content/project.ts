export const projectIntro = {
  title: "Sleep data is rich, but hard to explain well",
  summary:
    "A lot of sleep modeling work jumps straight to black-box performance. This project is meant to do the opposite: start from the clinical concepts, show the signals, and keep the feature logic understandable."
};

export const problemCards = [
  {
    title: "Two different problems",
    body: "Sleep staging and sleep apnea are related, but they are not the same task. One is about global sleep state. The other is about respiratory events within sleep."
  },
  {
    title: "Signals are not interchangeable",
    body: "EEG, EOG, and EMG matter most for staging. Airflow, effort, SpO2, and ECG matter most for apnea. The project should make that distinction obvious."
  }
];

export const clinicalConceptCards = [
  {
    title: "Sleep staging",
    body: "Stage labels reflect recognizable physiologic states. The point is not just classification. The point is whether the signal evidence actually supports the stage."
  },
  {
    title: "Sleep apnea",
    body: "Respiratory events are about airflow, breathing effort, oxygenation, and cardiac response. A model should read those patterns in a way a human can follow."
  }
];

export const datasetRoleCard = {
  title: "Why UCDDB works",
  body:
    "UCDDB gives one coherent environment: overnight PSG, stage labels, and respiratory-event annotations in the same records. That makes it a strong single-dataset case study."
};

export const projectPathCards = [
  {
    title: "Sleep staging path",
    tags: ["EEG", "EOG", "Chin EMG"],
    body: "Use stage-relevant physiology and interpretable features, then add explicit context carefully."
  },
  {
    title: "Apnea path",
    tags: ["Airflow", "Effort", "SpO2", "ECG"],
    body: "Use respiratory-event physiology first, rather than reducing the whole problem to a purely opaque signal classifier."
  }
];

export const stagingPage = {
  title: "Sleep staging should start from visible stage physiology",
  summary:
    "This section should make the reviewer understand what staging is before any metrics show up: what the signal families do, what clinicians look for, and why explicit context matters."
};

export const apneaPage = {
  title: "Sleep apnea should read like respiratory-event reasoning",
  summary:
    "This section should make the respiratory logic legible: what airflow change looks like, how effort behaves, where desaturation matters, and where ECG helps."
};

export const aboutPage = {
  title: "About this project",
  summary:
    "This project is meant to show signal-processing judgment, interpretable modeling choices, and a clean technical presentation layer."
};
