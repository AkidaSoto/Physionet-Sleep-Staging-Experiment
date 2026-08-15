import siteData from "../../artifacts/showcase/site-data.json";

export type ShowcaseSignalSlice = {
  id: string;
  label: string;
  start_sec: number;
  end_sec: number;
  effective_sample_rate_hz: number;
  time_seconds: number[];
  values: number[];
};

export type ShowcaseTrackSlice = {
  id: string;
  label: string;
  start_sec: number;
  end_sec: number;
  step_seconds: number;
  time_seconds: number[];
  values: number[];
  class_map?: Record<string, string>;
};

export type ShowcaseFeatureColumn = {
  id: string;
  label: string;
  values: Array<number | null>;
};

export type ShowcaseFeatureWindow = {
  time_seconds: number[];
  absolute_time_seconds: number[];
  columns: ShowcaseFeatureColumn[];
} | null;

export type ShowcaseEvent = {
  label: string;
  kind: string;
  absolute_start_sec: number;
  absolute_end_sec: number;
  relative_start_sec: number;
  relative_end_sec: number;
  metadata: Record<string, unknown>;
};

export type ApneaExample = {
  id: string;
  segment: {
    start_sec: number;
    end_sec: number;
    duration_sec: number;
  };
  anchor_event: ShowcaseEvent | null;
  signals: ShowcaseSignalSlice[];
  tracks: ShowcaseTrackSlice[];
  overlapping_events: ShowcaseEvent[];
  feature_window: ShowcaseFeatureWindow;
};

export type StagingExample = {
  id: string;
  class_id: number;
  class_label: string;
  run_start_sec: number;
  run_end_sec: number;
  excerpt_start_sec: number;
  excerpt_end_sec: number;
  signals: ShowcaseSignalSlice[];
  feature_window: ShowcaseFeatureWindow;
};

export type ShowcaseSiteData = {
  generated_at: string;
  dataset: string;
  record_id: string;
  apnea: {
    default_example_index: number;
    examples: ApneaExample[];
  };
  staging: {
    record_duration_sec?: number;
    hypnogram?: ShowcaseTrackSlice;
    anchors?: Array<{
      example_index: number;
      class_id: number;
      class_label: string;
      time_sec: number;
    }>;
    examples: StagingExample[];
  };
};

export function getShowcaseSiteData(): ShowcaseSiteData {
  return siteData as ShowcaseSiteData;
}
