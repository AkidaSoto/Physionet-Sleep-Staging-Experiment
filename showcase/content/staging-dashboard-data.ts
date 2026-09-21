import dashboardData from "./generated/staging-dashboard.json";

export type FoldMetric = {
  fold: number;
  macro_f1: number;
  accuracy: number;
  cohen_kappa: number;
};

export type ModelResult = {
  id: string;
  name: string;
  role: string;
  hypothesis: string;
  metrics: {
    accuracy: number;
    balanced_accuracy: number;
    macro_f1: number;
    cohen_kappa: number;
  };
  folds: FoldMetric[];
};

export type StageMetric = {
  stage: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
};

export type SubjectMetric = {
  subject: string;
  epochs: number;
  accuracy: number;
  macro_f1: number;
  cohen_kappa: number;
};

export type FeatureStageDistribution = {
  stage: string;
  count: number;
  p10: number;
  p25: number;
  median: number;
  p75: number;
  p90: number;
  one_vs_rest_cohens_d: number;
};

export type FeatureDistribution = {
  id: string;
  family_id: string;
  label: string;
  signal: string;
  stages: FeatureStageDistribution[];
};

export type FeatureSignalExample = {
  id: string;
  family_id: string;
  stage: string;
  record_id: string;
  epoch_start_sec: number;
  feature_id: string;
  feature_label: string;
  feature_value: number;
  feature_percentile: number;
  stage_percentile: number;
  direction: "high" | "low";
  selection_note: string;
  signal: {
    id: string;
    label: string;
    start_sec: number;
    end_sec: number;
    effective_sample_rate_hz: number;
    values: number[];
  };
};

export type StagingDashboardData = {
  source: {
    dataset: string;
    short_name: string;
    subjects: number;
    epochs: number;
    epoch_seconds: number;
    features: number;
    stages: string[];
  };
  validation: {
    outer_folds: number;
    train_subjects_per_fold: number;
    test_subjects_per_fold: number;
    neural_early_stop_subjects: number;
    grouping: string;
    primary_metric: string;
    folds: Array<{
      fold: number;
      train_subjects: string[];
      test_subjects: string[];
    }>;
  };
  headline: {
    baseline_macro_f1: number;
    best_macro_f1: number;
    absolute_lift: number;
  };
  models: ModelResult[];
  diagnostics: {
    rows: number;
    subjects: number;
    confusion_counts: number[][];
    confusion_normalized: number[][];
    per_stage: StageMetric[];
    subject_metrics: SubjectMetric[];
  };
  feature_importance: Array<{
    group: string;
    importance: number;
  }>;
  feature_evidence: {
    distributions: FeatureDistribution[];
  };
};

export const stagingDashboardData = dashboardData as StagingDashboardData;
