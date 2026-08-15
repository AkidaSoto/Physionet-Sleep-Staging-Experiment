"use client";

import type { ReactNode } from "react";
import { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { showcaseCase } from "../../content/showcase-case";
import type { ShowcaseChapter, ShowcaseSection } from "../../content/showcase-flow";
import { getShowcaseSiteData } from "../../lib/artifacts";
import type {
  ApneaExample,
  ShowcaseFeatureWindow,
  ShowcaseSignalSlice,
  ShowcaseTrackSlice,
  StagingExample
} from "../../lib/showcase-site-data";
import { PopupModuleTray } from "./PopupModuleTray";
import { ReferenceList } from "./ReferenceList";
import { Text } from "../ui/Text";

type Props = {
  chapter: ShowcaseChapter;
  activeSectionId: string | null;
  onSectionSelect: (sectionId: string | null) => void;
};

const siteData = getShowcaseSiteData();
type FixedSectionExampleId = keyof typeof showcaseCase.sectionExampleIds;
type StagingOverviewAnchor = { class_label: string; time_sec: number; exampleId: string };

function getPreviewLabel(section: ShowcaseSection) {
  return section.eyebrow.replace(/^\d+\.\s*/, "");
}

function getFixedExampleId(sectionId: string | null): string | null {
  if (!sectionId) {
    return null;
  }
  if (sectionId in showcaseCase.sectionExampleIds) {
    return showcaseCase.sectionExampleIds[sectionId as FixedSectionExampleId];
  }
  return null;
}

function buildLinePath(values: number[], width: number, height: number) {
  if (!values.length) {
    return "";
  }
  const finite = values.filter((value) => Number.isFinite(value));
  const min = finite.length ? Math.min(...finite) : 0;
  const max = finite.length ? Math.max(...finite) : 1;
  const span = max - min || 1;
  return values
    .map((value, index) => {
      const x = values.length === 1 ? 0 : (index / (values.length - 1)) * width;
      const normalized = Number.isFinite(value) ? (value - min) / span : 0.5;
      const y = height - normalized * height;
      return `${index === 0 ? "M" : "L"}${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

function buildStepPath(values: number[], width: number, height: number) {
  if (!values.length) {
    return "";
  }
  const numeric = values.map((value) => Number(value));
  const max = Math.max(...numeric, 1);
  const stepWidth = values.length > 1 ? width / (values.length - 1) : width;
  let path = "";
  for (let index = 0; index < numeric.length; index += 1) {
    const x = index * stepWidth;
    const y = height - (numeric[index] / max) * height;
    if (index === 0) {
      path = `M${x.toFixed(2)} ${y.toFixed(2)}`;
    } else {
      const prevX = x - stepWidth;
      path += ` L${x.toFixed(2)} ${path.endsWith(`${y.toFixed(2)}`) ? y.toFixed(2) : y.toFixed(2)}`;
      path = `${path.slice(0, path.length)}`;
      path += ` M`;
      path = path.slice(0, -1);
      path += ` L${x.toFixed(2)} ${y.toFixed(2)}`;
    }
  }
  return path.replace(/M(?![^M]*$).*?L/g, (match) => match.replace("M", "L"));
}

function getSignalTone(signalId: string) {
  if (signalId.includes("spo2")) return "d";
  if (signalId.includes("paradox")) return "c";
  if (signalId.includes("abdominal")) return "c";
  if (signalId.includes("thoracic")) return "b";
  if (signalId.includes("eog")) return "c";
  if (signalId.includes("emg")) return "d";
  return "a";
}

function getSignalById(example: ApneaExample | StagingExample | null, signalId: string) {
  return example?.signals.find((signal) => signal.id === signalId) ?? null;
}

function getTrackById(example: ApneaExample | null, trackId: string) {
  return example?.tracks.find((track) => track.id === trackId) ?? null;
}

function findPrimaryEventWindow(example: ApneaExample | null) {
  const track =
    getTrackById(example, "respiratory_obstruction_truth") ??
    getTrackById(example, "respiratory_event_truth");
  if (!track) {
    return null;
  }

  const values = track.values.map((value) => Number(value));
  const startIndex = values.findIndex((value) => value > 0);
  if (startIndex < 0) {
    return null;
  }
  let endIndex = startIndex;
  while (endIndex + 1 < values.length && values[endIndex + 1] > 0) {
    endIndex += 1;
  }

  return {
    startSec: track.time_seconds[startIndex] ?? 0,
    endSec: (track.time_seconds[endIndex] ?? 0) + track.step_seconds
  };
}

function getStageColor(stageValue: number) {
  switch (stageValue) {
    case 0:
      return "#8ea6c9";
    case 1:
      return "#78d7ff";
    case 2:
      return "#6abf9f";
    case 3:
    case 4:
      return "#f1b36b";
    case 5:
      return "#f58db7";
    default:
      return "rgba(255,255,255,0.22)";
  }
}

// The hypnogram track's numeric stage codes (see class_map in the exported artifact)
// use a different numbering than the anchor examples' class_id, so anchor-driven
// coloring is keyed off the stage label instead of relying on the two staying in sync.
function getStageColorByLabel(label: string) {
  switch (label) {
    case "wake":
      return "#8ea6c9";
    case "n1":
      return "#78d7ff";
    case "n2":
      return "#6abf9f";
    case "n3":
      return "#f1b36b";
    case "rem":
      return "#f58db7";
    default:
      return "rgba(255,255,255,0.22)";
  }
}

function getApneaOverviewRowState(activeCardIndex: number, hoveredCardIndex: number | null, signalId: string) {
  const highlightedByCard: Record<number, string[]> = {
    1: ["airflow_pressure"],
    2: ["spo2"],
    3: ["effort_thoracic", "effort_abdominal"]
  };
  const cardIndex =
    hoveredCardIndex != null && hoveredCardIndex !== 0 ? hoveredCardIndex : activeCardIndex !== 0 ? activeCardIndex : null;
  if (cardIndex == null) {
    return "neutral";
  }
  return highlightedByCard[cardIndex]?.includes(signalId) ? "active" : "muted";
}

function SignalTrace({
  signal,
  compact = false,
  emphasis = "neutral",
  eventWindow = null,
  overlay = null,
  onClick,
  onMouseEnter,
  onMouseLeave
}: {
  signal: ShowcaseSignalSlice;
  compact?: boolean;
  emphasis?: "neutral" | "active" | "muted";
  eventWindow?: { startSec: number; endSec: number } | null;
  overlay?: ReactNode;
  onClick?: () => void;
  onMouseEnter?: () => void;
  onMouseLeave?: () => void;
}) {
  const height = compact ? 72 : 96;
  const path = useMemo(() => buildLinePath(signal.values, 760, height), [signal.values, height]);
  const duration = Math.max(signal.end_sec - signal.start_sec, 1);
  const eventX = eventWindow ? (eventWindow.startSec / duration) * 760 : 0;
  const eventWidth = eventWindow ? ((eventWindow.endSec - eventWindow.startSec) / duration) * 760 : 0;
  const interactive = typeof onClick === "function";
  return (
    <div
      className={[
        "real-trace-row",
        compact ? "real-trace-row-compact" : "",
        interactive ? "real-trace-row-interactive" : "",
        emphasis === "active" ? "real-trace-row-active" : "",
        emphasis === "muted" ? "real-trace-row-muted" : ""
      ]
        .filter(Boolean)
        .join(" ")}
      role={interactive ? "button" : undefined}
      tabIndex={interactive ? 0 : undefined}
      onClick={onClick}
      onKeyDown={
        interactive
          ? (event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                onClick?.();
              }
            }
          : undefined
      }
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      <span className="real-trace-label">{signal.label}</span>
      <svg
        viewBox={`0 0 760 ${height}`}
        className="real-trace-svg"
        style={{ height }}
        preserveAspectRatio="none"
        aria-hidden="true"
      >
        {eventWindow ? (
          <rect x={eventX} y={0} width={Math.max(eventWidth, 2)} height={height} className="real-event-window" />
        ) : null}
        <path d={path} className={`real-trace-path real-trace-path-${getSignalTone(signal.id)}`} />
        {overlay}
      </svg>
    </div>
  );
}

function TrackBand({
  track,
  onClick,
  active = false
}: {
  track: ShowcaseTrackSlice;
  onClick?: () => void;
  active?: boolean;
}) {
  const path = useMemo(() => buildLinePath(track.values, 760, 34), [track.values]);
  const interactive = typeof onClick === "function";
  return (
    <div
      className={[
        "real-track-row",
        interactive ? "real-trace-row-interactive" : "",
        active ? "real-trace-row-active" : ""
      ]
        .filter(Boolean)
        .join(" ")}
      role={interactive ? "button" : undefined}
      tabIndex={interactive ? 0 : undefined}
      onClick={onClick}
      onKeyDown={
        interactive
          ? (event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                onClick?.();
              }
            }
          : undefined
      }
    >
      <span className="real-trace-label">{track.label}</span>
      <svg viewBox="0 0 760 34" className="real-track-svg" style={{ height: 34 }} preserveAspectRatio="none" aria-hidden="true">
        <path d={path} className="real-track-path" />
      </svg>
    </div>
  );
}

function FeatureWindowScene({
  featureWindow,
  title,
  activeCardIndex,
  cardKeywords
}: {
  featureWindow: ShowcaseFeatureWindow;
  title: string;
  activeCardIndex?: number;
  cardKeywords?: string[][];
}) {
  if (!featureWindow || !featureWindow.columns.length) {
    return (
      <div className="real-feature-empty">
        <Text tone="secondary">{title}</Text>
      </div>
    );
  }

  const keywords = activeCardIndex != null ? cardKeywords?.[activeCardIndex] : undefined;
  const matched = keywords?.length
    ? featureWindow.columns.filter((column) => keywords.some((keyword) => column.id.includes(keyword)))
    : [];
  const visibleColumns = matched.length ? matched : featureWindow.columns.slice(0, 6);

  return (
    <div className="real-feature-scene">
      {visibleColumns.map((column) => {
        const numeric = column.values.map((value) => (value == null ? NaN : value));
        const path = buildLinePath(numeric, 760, 74);
        return (
          <div key={column.id} className="real-feature-row">
            <span className="real-trace-label">{column.label}</span>
            <svg viewBox="0 0 760 74" className="real-trace-svg" style={{ height: 74 }} preserveAspectRatio="none" aria-hidden="true">
              <path d={path} className={`real-trace-path real-trace-path-${getSignalTone(column.id)}`} />
            </svg>
          </div>
        );
      })}
    </div>
  );
}

function AirflowAnnotation({
  signal,
  eventWindow
}: {
  signal: ShowcaseSignalSlice;
  eventWindow: { startSec: number; endSec: number } | null;
}) {
  if (!eventWindow) {
    return null;
  }
  const duration = Math.max(signal.end_sec - signal.start_sec, 1);
  const startIndex = Math.max(Math.floor((eventWindow.startSec / duration) * signal.values.length), 0);
  const endIndex = Math.min(Math.ceil((eventWindow.endSec / duration) * signal.values.length), signal.values.length - 1);
  const preStartIndex = Math.max(startIndex - Math.max(Math.floor(signal.values.length * 0.08), 8), 0);
  const preValues = signal.values.slice(preStartIndex, startIndex);
  const eventValues = signal.values.slice(startIndex, endIndex + 1);
  const baselineAmp = preValues.length ? Math.max(...preValues) - Math.min(...preValues) : 1;
  const eventAmp = eventValues.length ? Math.max(...eventValues) - Math.min(...eventValues) : 0.5;
  const maxAmp = Math.max(baselineAmp, eventAmp, 1e-6);
  const baselineHeight = (baselineAmp / maxAmp) * 84;
  const eventHeight = (eventAmp / maxAmp) * 84;
  const xBase = 102;
  const xEvent = 444;
  const durationX1 = (eventWindow.startSec / duration) * 760;
  const durationX2 = (eventWindow.endSec / duration) * 760;

  return (
    <g className="real-annotation">
      <line x1={xBase} x2={xBase} y1={90 - baselineHeight} y2={90} className="real-measure-line" />
      <line x1={xEvent} x2={xEvent} y1={90 - eventHeight} y2={90} className="real-measure-line" />
      <text x={xBase - 18} y={18} className="real-measure-text">
        baseline
      </text>
      <text x={xEvent - 14} y={18} className="real-measure-text">
        reduced
      </text>
      <line x1={durationX1} x2={durationX2} y1={18} y2={18} className="real-measure-line" />
      <text x={(durationX1 + durationX2) / 2 - 18} y={12} className="real-measure-text">
        &gt; 10 s
      </text>
    </g>
  );
}

function Spo2Annotation({
  signal,
  eventWindow
}: {
  signal: ShowcaseSignalSlice;
  eventWindow: { startSec: number; endSec: number } | null;
}) {
  if (!eventWindow) {
    return null;
  }
  const duration = Math.max(signal.end_sec - signal.start_sec, 1);
  const startIndex = Math.max(Math.floor((eventWindow.startSec / duration) * signal.values.length), 0);
  const endIndex = Math.min(Math.ceil((eventWindow.endSec / duration) * signal.values.length), signal.values.length - 1);
  const baseline = signal.values.slice(Math.max(startIndex - 20, 0), startIndex);
  const eventValues = signal.values.slice(startIndex, endIndex + 1);
  const baselineMean = baseline.length ? baseline.reduce((sum, value) => sum + value, 0) / baseline.length : 0;
  const nadir = eventValues.length ? Math.min(...eventValues) : baselineMean - 3;
  const all = signal.values;
  const min = Math.min(...all);
  const max = Math.max(...all);
  const span = max - min || 1;
  const yForValue = (value: number) => 96 - ((value - min) / span) * 96;
  const x = ((eventWindow.endSec - 2) / duration) * 760;

  return (
    <g className="real-annotation">
      <line x1={x} x2={x} y1={yForValue(baselineMean)} y2={yForValue(nadir)} className="real-measure-line" />
      <text x={x + 10} y={(yForValue(baselineMean) + yForValue(nadir)) / 2} className="real-measure-text">
        3% drop
      </text>
    </g>
  );
}

function getIndexEmphasis(activeCardIndex: number, hoveredCardIndex: number | null, rowIndex: number) {
  const current = hoveredCardIndex ?? activeCardIndex;
  return current === rowIndex ? "active" : "muted";
}

function EffortCorrelationGraphic({
  example,
  activeCardIndex,
  hoveredCardIndex,
  onCardHover,
  onCardSelect
}: {
  example: ApneaExample;
  activeCardIndex: number;
  hoveredCardIndex: number | null;
  onCardHover?: (cardIndex: number | null) => void;
  onCardSelect?: (cardIndex: number) => void;
}) {
  const thoracic = getSignalById(example, "effort_thoracic");
  const abdominal = getSignalById(example, "effort_abdominal");
  const correlation = getSignalById(example, "thor_abd_corr");
  const obstruction = getTrackById(example, "respiratory_obstruction_truth");
  const eventWindow = findPrimaryEventWindow(example);

  if (!thoracic || !abdominal) {
    return null;
  }

  return (
    <div className="real-scene-stack real-scene-stack-focus">
      <SignalTrace
        signal={thoracic}
        eventWindow={eventWindow}
        emphasis={getIndexEmphasis(activeCardIndex, hoveredCardIndex, 0)}
        onMouseEnter={() => onCardHover?.(0)}
        onMouseLeave={() => onCardHover?.(null)}
        onClick={() => onCardSelect?.(0)}
      />
      <SignalTrace
        signal={abdominal}
        eventWindow={eventWindow}
        emphasis={getIndexEmphasis(activeCardIndex, hoveredCardIndex, 1)}
        onMouseEnter={() => onCardHover?.(1)}
        onMouseLeave={() => onCardHover?.(null)}
        onClick={() => onCardSelect?.(1)}
      />
      {correlation ? (
        <SignalTrace
          signal={correlation}
          emphasis={getIndexEmphasis(activeCardIndex, hoveredCardIndex, 2)}
          onMouseEnter={() => onCardHover?.(2)}
          onMouseLeave={() => onCardHover?.(null)}
          onClick={() => onCardSelect?.(2)}
        />
      ) : null}
      {obstruction ? (
        <TrackBand
          track={obstruction}
          active={(hoveredCardIndex ?? activeCardIndex) === 3}
          onClick={() => onCardSelect?.(3)}
        />
      ) : null}
    </div>
  );
}

const APNEA_TRACK_SECTION: Record<string, string> = {
  respiratory_obstruction_truth: "apnea-signals",
  apnea_event_prediction_sec: "apnea-features"
};

function ApneaOverviewGraphic({
  example,
  activeCardIndex,
  hoveredCardIndex,
  onCardHover,
  onCardSelect,
  onNavigateSection
}: {
  example: ApneaExample | null;
  activeCardIndex: number;
  hoveredCardIndex: number | null;
  onCardHover?: (cardIndex: number | null) => void;
  onCardSelect?: (cardIndex: number) => void;
  onNavigateSection?: (sectionId: string) => void;
}) {
  if (!example) {
    return null;
  }

  const eventWindow = findPrimaryEventWindow(example);
  const airflow = getSignalById(example, "airflow_pressure");
  const spo2 = getSignalById(example, "spo2");
  const thoracic = getSignalById(example, "effort_thoracic");
  const abdominal = getSignalById(example, "effort_abdominal");

  if (activeCardIndex === 1 && airflow) {
    return (
      <div className="real-scene-stack real-scene-stack-focus">
        <SignalTrace signal={airflow} eventWindow={eventWindow} emphasis="active" overlay={<AirflowAnnotation signal={airflow} eventWindow={eventWindow} />} />
      </div>
    );
  }

  if (activeCardIndex === 2 && spo2) {
    return (
      <div className="real-scene-stack real-scene-stack-focus">
        <SignalTrace signal={spo2} emphasis="active" overlay={<Spo2Annotation signal={spo2} eventWindow={eventWindow} />} />
      </div>
    );
  }

  if (activeCardIndex === 3) {
    const cardThoracic = getSignalById(example, "effort_thoracic");
    const cardAbdominal = getSignalById(example, "effort_abdominal");
    const cardCorrelation = getSignalById(example, "thor_abd_corr");
    if (!cardThoracic || !cardAbdominal) {
      return null;
    }
    return (
      <div className="real-scene-stack real-scene-stack-focus">
        <SignalTrace signal={cardThoracic} eventWindow={eventWindow} emphasis="active" />
        <SignalTrace signal={cardAbdominal} eventWindow={eventWindow} emphasis="active" />
        {cardCorrelation ? <SignalTrace signal={cardCorrelation} emphasis="active" /> : null}
      </div>
    );
  }

  return (
    <div className="real-scene-stack">
      {airflow ? (
        <SignalTrace
          signal={airflow}
          compact
          eventWindow={eventWindow}
          emphasis={getApneaOverviewRowState(activeCardIndex, hoveredCardIndex, airflow.id)}
          onMouseEnter={() => onCardHover?.(1)}
          onMouseLeave={() => onCardHover?.(null)}
          onClick={() => onCardSelect?.(1)}
        />
      ) : null}
      {thoracic ? (
        <SignalTrace
          signal={thoracic}
          compact
          eventWindow={eventWindow}
          emphasis={getApneaOverviewRowState(activeCardIndex, hoveredCardIndex, thoracic.id)}
          onMouseEnter={() => onCardHover?.(3)}
          onMouseLeave={() => onCardHover?.(null)}
          onClick={() => onCardSelect?.(3)}
        />
      ) : null}
      {abdominal ? (
        <SignalTrace
          signal={abdominal}
          compact
          eventWindow={eventWindow}
          emphasis={getApneaOverviewRowState(activeCardIndex, hoveredCardIndex, abdominal.id)}
          onMouseEnter={() => onCardHover?.(3)}
          onMouseLeave={() => onCardHover?.(null)}
          onClick={() => onCardSelect?.(3)}
        />
      ) : null}
      {spo2 ? (
        <SignalTrace
          signal={spo2}
          compact
          emphasis={getApneaOverviewRowState(activeCardIndex, hoveredCardIndex, spo2.id)}
          onMouseEnter={() => onCardHover?.(2)}
          onMouseLeave={() => onCardHover?.(null)}
          onClick={() => onCardSelect?.(2)}
        />
      ) : null}
      {example.tracks
        .filter((track) => track.id !== "respiratory_event_truth")
        .map((track) => {
          const targetSection = APNEA_TRACK_SECTION[track.id];
          return (
            <TrackBand
              key={track.id}
              track={track}
              onClick={targetSection ? () => onNavigateSection?.(targetSection) : undefined}
            />
          );
        })}
    </div>
  );
}

function getStageCardIndex(stageValue: number) {
  if (stageValue === 0) return 0;
  if (stageValue === 5) return 2;
  return 1;
}

function HypnogramGraphic({
  hypnogram,
  anchors,
  onAnchorSelect,
  onOverviewSelect,
  selectedExampleId
}: {
  hypnogram: ShowcaseTrackSlice | undefined;
  anchors: StagingOverviewAnchor[];
  onAnchorSelect?: (exampleId: string) => void;
  onOverviewSelect?: (cardIndex: number) => void;
  selectedExampleId?: string | null;
}) {
  if (!hypnogram) {
    return null;
  }

  const duration = Math.max(hypnogram.end_sec - hypnogram.start_sec, 1);
  const barWidth = hypnogram.values.length > 0 ? 760 / hypnogram.values.length : 760;
  const barsInteractive = typeof onOverviewSelect === "function";

  return (
    <div className="hypnogram-scene">
      <svg viewBox="0 0 760 180" className="hypnogram-svg" aria-hidden="true">
        {hypnogram.values.map((value, index) => (
          <rect
            key={`${index}-${value}`}
            x={index * barWidth}
            y={54}
            width={Math.max(barWidth + 1, 2)}
            height={52}
            rx={6}
            fill={getStageColor(Number(value))}
            opacity={0.94}
            className={barsInteractive ? "hypnogram-bar-interactive" : undefined}
            onClick={barsInteractive ? () => onOverviewSelect?.(getStageCardIndex(Number(value))) : undefined}
          />
        ))}
        {anchors.map((anchor) => {
          const x = ((anchor.time_sec - hypnogram.start_sec) / duration) * 760;
          return (
            <g
              key={`${anchor.class_label}-${anchor.time_sec}`}
              transform={`translate(${x.toFixed(2)} 42)`}
              className={[
                "hypnogram-anchor-target",
                selectedExampleId === anchor.exampleId ? "hypnogram-anchor-target-active" : ""
              ]
                .filter(Boolean)
                .join(" ")}
              onClick={() => onAnchorSelect?.(anchor.exampleId)}
            >
              <circle className="hypnogram-anchor-dot" r="5" style={{ fill: getStageColorByLabel(anchor.class_label) }} />
            </g>
          );
        })}
      </svg>
      <div className="hypnogram-anchor-row">
        {anchors.slice(0, 8).map((anchor) => (
          <button
            key={`${anchor.class_label}-${anchor.time_sec}-button`}
            type="button"
            className={[
              "hypnogram-anchor-chip",
              selectedExampleId === anchor.exampleId ? "hypnogram-anchor-chip-active" : ""
            ]
              .filter(Boolean)
              .join(" ")}
            onClick={() => onAnchorSelect?.(anchor.exampleId)}
          >
            <span className="hypnogram-anchor-chip-dot" style={{ background: getStageColorByLabel(anchor.class_label) }} />
            {anchor.class_label}
          </button>
        ))}
      </div>
    </div>
  );
}

function getStagingSignalCardIndex(signalId: string) {
  if (signalId.startsWith("eeg")) return 0;
  if (signalId.startsWith("eog")) return 1;
  if (signalId.startsWith("emg")) return 2;
  return -1;
}

function StagingSignalsGraphic({
  example,
  activeCardIndex,
  hoveredCardIndex,
  onCardHover,
  onCardSelect
}: {
  example: StagingExample | null;
  activeCardIndex: number;
  hoveredCardIndex: number | null;
  onCardHover?: (cardIndex: number | null) => void;
  onCardSelect?: (cardIndex: number) => void;
}) {
  if (!example) {
    return null;
  }

  return (
    <div className="real-scene-stack">
      {example.signals.map((signal) => {
        const cardIndex = getStagingSignalCardIndex(signal.id);
        return (
          <SignalTrace
            key={signal.id}
            signal={signal}
            compact={false}
            emphasis={cardIndex >= 0 ? getIndexEmphasis(activeCardIndex, hoveredCardIndex, cardIndex) : "neutral"}
            onMouseEnter={cardIndex >= 0 ? () => onCardHover?.(cardIndex) : undefined}
            onMouseLeave={cardIndex >= 0 ? () => onCardHover?.(null) : undefined}
            onClick={cardIndex >= 0 ? () => onCardSelect?.(cardIndex) : undefined}
          />
        );
      })}
    </div>
  );
}

function ZoomOverviewScene({
  chapter,
  onSectionSelect,
  apneaExample,
  hypnogram,
  stagingAnchors,
  activeCardIndex,
  hoveredCardIndex,
  onCardHover,
  onApneaOverviewSelect,
  onStageAnchorSelect,
  onStageOverviewSelect,
  selectedStagingExampleId
}: {
  chapter: ShowcaseChapter;
  onSectionSelect: (sectionId: string) => void;
  apneaExample: ApneaExample | null;
  hypnogram: ShowcaseTrackSlice | undefined;
  stagingAnchors: StagingOverviewAnchor[];
  activeCardIndex: number;
  hoveredCardIndex: number | null;
  onCardHover: (cardIndex: number | null) => void;
  onApneaOverviewSelect: (cardIndex: number) => void;
  onStageAnchorSelect: (exampleId: string) => void;
  onStageOverviewSelect: (cardIndex: number) => void;
  selectedStagingExampleId: string | null;
}) {
  return (
    <motion.div
      key="overview"
      className="zoom-scene-mode zoom-scene-mode-overview"
      initial={{ opacity: 0, scale: 1.02 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 1.04 }}
      transition={{ duration: 0.26, ease: [0.22, 1, 0.36, 1] }}
    >
      {chapter.id === "apnea" ? (
        <ApneaOverviewGraphic
          example={apneaExample}
          activeCardIndex={activeCardIndex}
          hoveredCardIndex={hoveredCardIndex}
          onCardHover={onCardHover}
          onCardSelect={onApneaOverviewSelect}
          onNavigateSection={onSectionSelect}
        />
      ) : (
        <HypnogramGraphic
          hypnogram={hypnogram}
          anchors={stagingAnchors}
          onAnchorSelect={onStageAnchorSelect}
          onOverviewSelect={onStageOverviewSelect}
          selectedExampleId={selectedStagingExampleId}
        />
      )}
    </motion.div>
  );
}

function ZoomCardTabs({
  cards,
  activeCardIndex,
  onCardSelect,
  onCardHover
}: {
  cards: ShowcaseSection["cards"];
  activeCardIndex: number;
  onCardSelect: (cardIndex: number) => void;
  onCardHover: (cardIndex: number | null) => void;
}) {
  if (cards.length < 2) {
    return null;
  }

  return (
    <div className="zoom-card-tabs" role="tablist" aria-label="Focal points">
      {cards.map((card, index) => {
        const active = activeCardIndex === index;

        return (
          <button
            key={card.title}
            type="button"
            role="tab"
            aria-selected={active}
            className={["zoom-card-tab", active ? "zoom-card-tab-active" : ""].filter(Boolean).join(" ")}
            onClick={() => onCardSelect(index)}
            onMouseEnter={() => onCardHover(index)}
            onMouseLeave={() => onCardHover(null)}
            onFocus={() => onCardHover(index)}
            onBlur={() => onCardHover(null)}
          >
            {card.title}
          </button>
        );
      })}
    </div>
  );
}

function renderFocusScene(
  section: ShowcaseSection,
  apneaExample: ApneaExample | null,
  stagingExample: StagingExample | null,
  hypnogram: ShowcaseTrackSlice | undefined,
  stagingAnchors: StagingOverviewAnchor[],
  activeCardIndex: number,
  hoveredCardIndex: number | null,
  onCardHover: (cardIndex: number | null) => void,
  onCardSelect: (cardIndex: number) => void,
  onSectionSelect: (sectionId: string) => void,
  onStageAnchorSelect: (exampleId: string) => void,
  onStageOverviewSelect: (cardIndex: number) => void,
  selectedStagingExampleId: string | null
) {
  switch (section.id) {
    case "apnea-overview":
      return (
        <ApneaOverviewGraphic
          example={apneaExample}
          activeCardIndex={activeCardIndex}
          hoveredCardIndex={hoveredCardIndex}
          onCardHover={onCardHover}
          onCardSelect={onCardSelect}
          onNavigateSection={onSectionSelect}
        />
      );
    case "apnea-signals":
      return apneaExample ? (
        <EffortCorrelationGraphic
          example={apneaExample}
          activeCardIndex={activeCardIndex}
          hoveredCardIndex={hoveredCardIndex}
          onCardHover={onCardHover}
          onCardSelect={onCardSelect}
        />
      ) : null;
    case "apnea-features":
      return (
        <FeatureWindowScene
          featureWindow={apneaExample?.feature_window ?? null}
          title="No apnea features available for this window."
          activeCardIndex={activeCardIndex}
          cardKeywords={[["respiration_rate"], ["paradox_mask"], ["event_vector"]]}
        />
      );
    case "staging-overview":
      return (
        <HypnogramGraphic
          hypnogram={hypnogram}
          anchors={stagingAnchors}
          onAnchorSelect={onStageAnchorSelect}
          onOverviewSelect={onStageOverviewSelect}
          selectedExampleId={selectedStagingExampleId}
        />
      );
    case "staging-signals":
      return (
        <StagingSignalsGraphic
          example={stagingExample}
          activeCardIndex={activeCardIndex}
          hoveredCardIndex={hoveredCardIndex}
          onCardHover={onCardHover}
          onCardSelect={onCardSelect}
        />
      );
    case "staging-features":
      return (
        <FeatureWindowScene
          featureWindow={stagingExample?.feature_window ?? null}
          title="No staging features available for this window."
          activeCardIndex={activeCardIndex}
          cardKeywords={[["splindex", "spindle"], ["slow_wave"], ["rem_power", "sem_power", "emg_tone", "emg_suppression"]]}
        />
      );
    default:
      return null;
  }
}

function ZoomFocusScene({
  section,
  apneaExample,
  stagingExample,
  hypnogram,
  stagingAnchors,
  activeCardIndex,
  hoveredCardIndex,
  onCardHover,
  onCardSelect,
  onSectionSelect,
  onStageAnchorSelect,
  onStageOverviewSelect,
  selectedStagingExampleId
}: {
  section: ShowcaseSection;
  apneaExample: ApneaExample | null;
  stagingExample: StagingExample | null;
  hypnogram: ShowcaseTrackSlice | undefined;
  stagingAnchors: StagingOverviewAnchor[];
  activeCardIndex: number;
  hoveredCardIndex: number | null;
  onCardHover: (cardIndex: number | null) => void;
  onCardSelect: (cardIndex: number) => void;
  onSectionSelect: (sectionId: string) => void;
  onStageAnchorSelect: (exampleId: string) => void;
  onStageOverviewSelect: (cardIndex: number) => void;
  selectedStagingExampleId: string | null;
}) {
  return (
    <motion.div
      key={section.id}
      className="zoom-scene-mode zoom-scene-mode-focus"
      initial={{ opacity: 0, scale: 0.92 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.96 }}
      transition={{ duration: 0.26, ease: [0.22, 1, 0.36, 1] }}
    >
      {renderFocusScene(
        section,
        apneaExample,
        stagingExample,
        hypnogram,
        stagingAnchors,
        activeCardIndex,
        hoveredCardIndex,
        onCardHover,
        onCardSelect,
        onSectionSelect,
        onStageAnchorSelect,
        onStageOverviewSelect,
        selectedStagingExampleId
      )}
    </motion.div>
  );
}

function ZoomDetailCard({
  section,
  activeCardIndex
}: {
  section: ShowcaseSection;
  activeCardIndex: number;
}) {
  const activeCard = section.cards[activeCardIndex];

  if (!activeCard) {
    return null;
  }

  return (
    <motion.div
      key={`${section.id}-${activeCard.title}`}
      className="zoom-detail-card"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.18, ease: [0.22, 1, 0.36, 1] }}
    >
      <span className="zoom-detail-card-title">{activeCard.title}</span>
      <Text as="p" className="zoom-detail-card-body" tone="secondary">
        {activeCard.body}
      </Text>
    </motion.div>
  );
}

export function ZoomChapterStage({ chapter, activeSectionId, onSectionSelect }: Props) {
  const activeSection = chapter.sections.find((section) => section.id === activeSectionId) ?? null;
  const [activeCardIndex, setActiveCardIndex] = useState(0);
  const [hoveredCardIndex, setHoveredCardIndex] = useState<number | null>(null);
  const [selectedStagingExampleId, setSelectedStagingExampleId] = useState<string | null>(null);

  const apneaExamples = siteData.apnea.examples;
  const stagingExamples = siteData.staging.examples;
  const fixedExampleId = getFixedExampleId(activeSectionId);
  const activeApneaExample =
    chapter.id === "apnea"
      ? apneaExamples.find((example) => example.id === fixedExampleId) ??
        apneaExamples.find((example) => example.id === showcaseCase.sectionExampleIds["apnea-overview"]) ??
        apneaExamples[0] ??
        null
      : null;
  const activeStagingExample =
    chapter.id === "staging"
      ? stagingExamples.find((example) => example.id === selectedStagingExampleId) ??
        stagingExamples.find((example) => example.id === fixedExampleId) ??
        stagingExamples.find((example) => example.id === showcaseCase.sectionExampleIds["staging-signals"]) ??
        stagingExamples[0] ??
        null
      : null;
  const stagingOverviewAnchorIds = new Set<string>(showcaseCase.stagingOverviewAnchorIds);
  const stagingAnchors = (siteData.staging.anchors ?? [])
    .filter((anchor) => {
      const example = stagingExamples[anchor.example_index];
      return example ? stagingOverviewAnchorIds.has(example.id) : false;
    })
    .map((anchor) => ({
      class_label: anchor.class_label,
      time_sec: anchor.time_sec,
      exampleId: stagingExamples[anchor.example_index]?.id ?? ""
    }));
  const selectedStageAnchor =
    chapter.id === "staging" && selectedStagingExampleId
      ? stagingAnchors.find((anchor) => anchor.exampleId === selectedStagingExampleId) ?? null
      : null;

  function handleApneaOverviewSelect(cardIndex: number) {
    setHoveredCardIndex(null);
    setActiveCardIndex(cardIndex);
    onSectionSelect("apnea-overview");
  }

  function handleNavigateSection(sectionId: string) {
    setHoveredCardIndex(null);
    setActiveCardIndex(0);
    onSectionSelect(sectionId);
  }

  function handleStageAnchorSelect(exampleId: string) {
    if (!exampleId) {
      return;
    }
    setHoveredCardIndex(null);
    setActiveCardIndex(0);
    setSelectedStagingExampleId(exampleId);
    onSectionSelect("staging-signals");
  }

  function handleStageOverviewSelect(cardIndex: number) {
    setHoveredCardIndex(null);
    setActiveCardIndex(cardIndex);
    onSectionSelect("staging-overview");
  }

  function handleBack() {
    setHoveredCardIndex(null);
    setActiveCardIndex(0);
    onSectionSelect(null);
  }

  useEffect(() => {
    setSelectedStagingExampleId(null);
  }, [chapter.id]);

  return (
    <div className="zoom-chapter-shell">
      <div className="zoom-stage-frame">
        <div className="zoom-scene-shell">
          <AnimatePresence initial={false} mode="sync">
            {activeSection ? (
              <ZoomFocusScene
                section={activeSection}
                apneaExample={activeApneaExample}
                stagingExample={activeStagingExample}
                hypnogram={siteData.staging.hypnogram}
                stagingAnchors={stagingAnchors}
                activeCardIndex={activeCardIndex}
                hoveredCardIndex={hoveredCardIndex}
                onCardHover={setHoveredCardIndex}
                onCardSelect={setActiveCardIndex}
                onSectionSelect={handleNavigateSection}
                onStageAnchorSelect={handleStageAnchorSelect}
                onStageOverviewSelect={handleStageOverviewSelect}
                selectedStagingExampleId={selectedStagingExampleId}
              />
            ) : (
              <ZoomOverviewScene
                chapter={chapter}
                onSectionSelect={handleNavigateSection}
                apneaExample={activeApneaExample}
                hypnogram={siteData.staging.hypnogram}
                stagingAnchors={stagingAnchors}
                activeCardIndex={activeCardIndex}
                hoveredCardIndex={hoveredCardIndex}
                onCardHover={setHoveredCardIndex}
                onApneaOverviewSelect={handleApneaOverviewSelect}
                onStageAnchorSelect={handleStageAnchorSelect}
                onStageOverviewSelect={handleStageOverviewSelect}
                selectedStagingExampleId={selectedStagingExampleId}
              />
            )}
          </AnimatePresence>

          <AnimatePresence initial={false}>
            {activeSection ? (
              <motion.div
                key={activeSection.id}
                className="zoom-detail-rail"
                initial={{ opacity: 0, x: 28 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 18 }}
                transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
              >
                <div className="zoom-detail-header">
                  <button type="button" className="zoom-back-button" onClick={handleBack}>
                    &lt;
                  </button>

                  <div className="zoom-detail-copy">
                    <div className="zoom-detail-title-row">
                      <span className="zoom-detail-title-pill">{getPreviewLabel(activeSection)}</span>
                      {activeSection.id === "staging-signals" && selectedStageAnchor ? (
                        <span
                          className="zoom-detail-stage-badge"
                          style={{ background: getStageColorByLabel(selectedStageAnchor.class_label) }}
                        >
                          {selectedStageAnchor.class_label}
                        </span>
                      ) : null}
                    </div>
                    <Text as="p" variant="bodyStrong" className="zoom-detail-takeaway">
                      {activeSection.takeaway}
                    </Text>
                  </div>
                </div>

                <ZoomCardTabs
                  cards={activeSection.cards}
                  activeCardIndex={activeCardIndex}
                  onCardSelect={setActiveCardIndex}
                  onCardHover={setHoveredCardIndex}
                />
                <ZoomDetailCard section={activeSection} activeCardIndex={activeCardIndex} />
                <PopupModuleTray modules={activeSection.modules} />
                {chapter.references?.length ? <ReferenceList links={chapter.references} /> : null}
              </motion.div>
            ) : null}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
