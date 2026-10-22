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
  stage_f1: Record<string, number>;
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
  ahi_band: string;
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
  target_stage: string;
  stage: string;
  record_id: string;
  epoch_start_sec: number;
  feature_id: string;
  feature_label: string;
  feature_value: number;
  feature_percentile: number;
  stage_percentile: number;
  direction: "high" | "low";
  comparison_role: "target" | "contrast";
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
    tracks: number;
    stages: string[];
  };
  validation: {
    outer_folds: number;
    train_subjects_per_fold: number;
    test_subjects_per_fold: number;
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
    best_model_id: string;
    best_macro_f1: number;
    diagnostics_model_id: string;
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
    family_id: string;
    importance: number;
    n_features: number;
    stage_drops: Record<string, number>;
  }>;
  signal_importance: Array<{
    id: string;
    label: string;
    family_id: string;
    family: string;
    kind: "track" | "feature";
    importance: number;
    stage_drops: Record<string, number>;
  }>;
  step_importance: Array<{
    offset_sec: number;
    importance: number;
    stage_drops: Record<string, number>;
  }>;
  importance_folds: number;
  time_importance: {
    fine: Array<{ offset_sec: number; span_sec: number; importance: number }>;
    epoch: Array<{ offset_sec: number; span_sec: number; importance: number }>;
    folds: number;
  };
  model_size: {
    n_iter: number;
    trees: number;
    nodes: number;
    leaves: number;
    splits: number;
    inputs_used: number;
    inputs: number;
    train_subjects: number;
    train_epochs: number;
    fit_seconds: number;
    hmm_params: number;
    cnn_weights: number;
  };
  sleep_structure: {
    transition_matrix: number[][];
    bouts: Array<{
      stage: string;
      bouts: number;
      median_epochs: number;
      mean_epochs: number;
      p90_epochs: number;
      share_single_epoch: number;
      self_transition: number;
      initial: number;
    }>;
  };
  feature_evidence: {
    distributions: FeatureDistribution[];
  };
  generalisation: {
    transfer: Array<{ evaluation: string; macro_f1: number; kappa: number; accuracy: number; subj_min: number; subj_median: number; stage_f1: Record<string, number> }>;
    holdout: Array<{ config: string; look?: number; macro_f1: number; kappa: number; accuracy: number }>;
    holdout_subjects: number;
    holdout_looks: number;
    holdout_dates: string[];
  };
  reduction?: Array<{ config: string; signals: number; macro_f1: number; folds: string; n2_f1: number }>;
  complexity: Array<{
    learner: string;
    label: string;
    macro_f1: number;
    kappa: number;
    learned_values: number;
    folds: string;
    stage_f1: Record<string, number>;
  }>;
  signal_ablations: Array<{
    config: string;
    label: string;
    decoder: string;
    macro_f1: number;
    kappa: number;
    kappa4: number;
    stage_f1: Record<string, number>;
  }>;
};

export type SaliencyData = {
  model: string;
  fold: number;
  test_subjects: number;
  test_epochs: number;
  signals: string[];
  seconds: number[];
  by_stage: Array<{ stage: string; epochs: number; signal_share: number[]; attention: number[] }>;
  examples: Array<{
    stage: string;
    record_id: string;
    epoch_start_sec: number;
    probability: number;
    input: number[][];
    saliency: number[][];
    attention: number[];
  }>;
};

export const stagingDashboardData = dashboardData as StagingDashboardData;
