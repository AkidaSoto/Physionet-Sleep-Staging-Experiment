export { getShowcaseSiteData } from "./showcase-site-data";

const overview = {
  title: "Interpretable Sleep Staging from PSG",
  subtitle: "Sleep-stage classification using physiological features from UCDDB"
};

const researchLinks = [
  {
    id: "ucddb",
    label: "UCDDB dataset",
    title: "University College Dublin Sleep Apnea Database",
    url: "https://physionet.org/content/ucddb/1.0.0/",
    summary: "Primary PhysioNet source for the overnight PSG records and annotations."
  },
  {
    id: "staging-benchmark",
    label: "UCDDB staging benchmark",
    title: "Multi-Branch CNN with stage refinement and attention fusion",
    url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC7698838/",
    summary: "Published sleep-staging reference evaluated on UCDDB."
  }
];

const taskSummary = [
  {
    slug: "sleep-staging",
    title: "Sleep staging",
    signals: ["EEG", "EOG", "Chin EMG"],
    focus: "Subject-independent staging with interpretable physiological features and explicit error analysis."
  }
];

export function getOverview() {
  return overview;
}

export function getResearchLinks() {
  return researchLinks;
}

export function getTaskSummary() {
  return taskSummary;
}
