"use client";

import Link from "next/link";
import { useEffect, useId, useMemo, useRef, useState } from "react";
import { sensorColors, stageColors } from "../../design/palette";
import {
  basisLabels,
  codeAnchor,
  featureBasis,
  featureFamilies,
  featureUnits,
  mechanismFor,
  modelCFeatureIds,
  stageFeatureTargets,
  trackDerivation,
  type StageFeatureTarget
} from "../../content/staging-content";
import {
  stagingDashboardData,
  type FeatureDistribution,
  type FeatureSignalExample
} from "../../content/staging-dashboard-data";
import { fmt } from "../../lib/results";
import {
  Citation,
  Exp,
  Tabs,
  featureValue,
  formatClock,
  linePath,
  movePickerFocus
} from "../ui/ResearchPrimitives";

const data = stagingDashboardData;

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

export function BasisTag({ featureId, compact = false }: { featureId: string; compact?: boolean }) {
  const info = featureBasis[featureId];
  if (!info) return null;
  return (
    <em className={`research-basis-tag is-${info.basis}`} title={basisLabels[info.basis].long}>
      {basisLabels[info.basis].short}
      {compact ? null : <> · {info.mechanism}</>}
    </em>
  );
}

type ExampleMarker = { stage: string; value: number; kind: "target" | "contrast" };

export function FeatureDistributionChart({
  feature,
  markers = [],
  compact = false
}: {
  feature: FeatureDistribution;
  markers?: ExampleMarker[];
  compact?: boolean;
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
        <span>{featureValue(rawMin)}</span>
        <span>{compact ? featureUnits[feature.id] : [feature.label, featureUnits[feature.id]].filter(Boolean).join(" · ")}</span>
        <span>{featureValue(rawMax)}</span>
      </div>
      {feature.stages.map((stage) => (
        <div className="research-distribution-row" key={stage.stage}>
          <strong className="swatch" style={{ "--swatch": stageColors[stage.stage] } as React.CSSProperties}>
            {stage.stage}
          </strong>
          <div className="research-distribution-plot">
            <svg
              viewBox="0 0 600 40"
              role="img"
              aria-label={`${stage.stage}: median ${featureValue(stage.median)}, middle half ${featureValue(stage.p25)} to ${featureValue(stage.p75)}`}
              preserveAspectRatio="none"
            >
              <line className="range" x1={x(stage.p10)} x2={x(stage.p90)} y1="20" y2="20" />
              <line className="cap" x1={x(stage.p10)} x2={x(stage.p10)} y1="14" y2="26" />
              <line className="cap" x1={x(stage.p90)} x2={x(stage.p90)} y1="14" y2="26" />
              <rect x={x(stage.p25)} y="9" width={Math.max(x(stage.p75) - x(stage.p25), 2)} height="22" fill={stageColors[stage.stage]} />
              <line className="median" x1={x(stage.median)} x2={x(stage.median)} y1="7" y2="33" />
            </svg>
            {markers
              .filter((marker) => marker.stage === stage.stage)
              .map((marker) => (
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
      {!compact && (
        <p className="research-chart-key">
          <span /> 10th to 90th percentile <i /> middle 50% <b /> median
          {markers.map((marker) => (
            <em key={marker.kind}>
              <u className={`research-example-marker is-${marker.kind}`} />
              {marker.stage} example
            </em>
          ))}
        </p>
      )}
    </div>
  );
}

export function SpindleComparison() {
  const distributions = data.feature_evidence.distributions;
  const band = distributions.find((feature) => feature.id === "track.c_rel_sigma");
  const bursts = distributions.find((feature) => feature.id === "spindle_bursts");
  if (!band || !bursts) return null;
  return (
    <figure className="research-spindle-comparison" id="spindle-example">
      <figcaption>
        <h4>A burst measure separated N2 better</h4>
        <p>
          Sleep spindles are brief bursts of brain activity used to score N2. <Citation number={1} /> The band-share measure uses the fraction of EEG power in the spindle band. The burst measure compares spindle-band power with neighboring frequencies and counts seconds above a threshold set for each night.
        </p>
      </figcaption>
      <div className="research-spindle-plots">
        <div>
          <h5>Spindle-band share</h5>
          <FeatureDistributionChart feature={band} compact />
        </div>
        <div>
          <h5>Seconds above the spindle-ratio threshold</h5>
          <FeatureDistributionChart feature={bursts} compact />
        </div>
      </div>
      <p className="research-plot-caption">
        Boxes show the middle half of values, whiskers span the 10th to 90th percentiles, and the number at right gives the median. Each panel uses its own units.
      </p>
      <p>
        For N2 versus the other stages as a group, the standardized difference was {fmt("effect.spindle_bursts.d_N2", 2)} for the burst measure and {fmt("effect.c_rel_sigma.d_N2", 2)} for band share. N2 and N3 had the same median burst count. I kept both measures, adding the burst measure in place of two other spectral inputs. Identifying individual spindles would require validation against marked events. <Exp ids={[32, 36]} />
      </p>
    </figure>
  );
}

function featureBehaviour(
  distribution: { stages: Array<{ stage: string; median: number; one_vs_rest_cohens_d: number }> },
  stage: string,
  direction: "high" | "low"
) {
  const byMedian = [...distribution.stages].sort((x, y) => x.median - y.median);
  const highestStages = byMedian.filter((row) => Math.abs(row.median - byMedian[byMedian.length - 1].median) < 1e-9).map((row) => row.stage);
  const lowestStages = byMedian.filter((row) => Math.abs(row.median - byMedian[0].median) < 1e-9).map((row) => row.stage);
  const highest = highestStages.join(" and ");
  const lowest = lowestStages.join(" and ");
  const d = distribution.stages.find((row) => row.stage === stage)?.one_vs_rest_cohens_d ?? 0;
  const signed = direction === "high" ? d : -d;
  const extremeStages = direction === "high" ? highestStages : lowestStages;
  const extreme = extremeStages.length === 1 && extremeStages[0] === stage;
  const verdict: "expected" | "partly" | "unexpected" = signed >= 0.5 && extreme ? "expected" : signed >= 0.2 ? "partly" : "unexpected";
  return { verdict, highest, lowest, d };
}

const behaviourLabel = {
  expected: "✓ Matches expectation",
  partly: "~ Partly matches",
  unexpected: "≠ Differs from expectation"
};

function FeatureStory({
  feature,
  stageTarget,
  distribution
}: {
  feature: { id: string; stageUse?: string };
  stageTarget: StageFeatureTarget;
  distribution: { stages: Array<{ stage: string; median: number; one_vs_rest_cohens_d: number }> };
}) {
  const mechanism = mechanismFor(feature.id);
  const sign = (d: number) => `${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(2)}`;
  const direction = stageTarget.expect[feature.id] ?? "high";
  const b = featureBehaviour(distribution, stageTarget.stage, direction);
  return (
    <div className="research-feature-story">
      {mechanism ? (
        <div>
          <h5>Physiology · {featureBasis[feature.id].mechanism}</h5>
          <p>
            {mechanism.text} {mechanism.cites.map((number) => <Citation key={number} number={number} />)}
          </p>
        </div>
      ) : null}
      <div className={`research-behaviour is-${b.verdict}`}>
        <h5>
          Expected {direction === "high" ? "higher" : "lower"} in {stageTarget.stage} <em>{behaviourLabel[b.verdict]}</em>
        </h5>
        <p>
          {feature.stageUse ? `${feature.stageUse} ` : ""}In this data: highest in {b.highest}, lowest in {b.lowest} ({stageTarget.stage} vs the rest,{" "}
          <abbr title="Standardized difference between this stage and the other stages; positive means higher values in this stage">d</abbr> = {sign(b.d)}).
        </p>
      </div>
    </div>
  );
}

export function PhysiologyExplorer() {
  const pickerId = useId();
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
      return () => {
        cancelled = true;
      };
    }
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          observer.disconnect();
          loadExamples();
        }
      },
      { rootMargin: "500px 0px" }
    );
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
    <div className="research-physiology-explorer" ref={explorerRef}>
      <div className="research-explorer-step">1. Choose a sleep stage to inspect its features</div>
      <div className="research-family-tabs research-stage-tabs" role="tablist" aria-label="Choose an expert sleep-stage label" onKeyDown={movePickerFocus}>
        {stageFeatureTargets.map((item) => (
          <button
            key={item.stage}
            type="button"
            role="tab"
            id={`${pickerId}-tab-${item.stage}`}
            aria-controls={`${pickerId}-panel-${item.stage}`}
            aria-selected={item.stage === stageId}
            tabIndex={item.stage === stageId ? 0 : -1}
            onClick={() => chooseStage(item)}
          >
            <span className="swatch" style={{ "--swatch": stageColors[item.stage] } as React.CSSProperties}>
              {item.stage}
            </span>
          </button>
        ))}
      </div>

      {stageFeatureTargets
        .filter((item) => item.stage !== stageId)
        .map((item) => (
          <div key={item.stage} role="tabpanel" id={`${pickerId}-panel-${item.stage}`} aria-labelledby={`${pickerId}-tab-${item.stage}`} hidden />
        ))}
      <div role="tabpanel" id={`${pickerId}-panel-${stageId}`} aria-labelledby={`${pickerId}-tab-${stageId}`}>
        <div className="research-explorer-step">2. Pick a {stageId} measurement ({targetedFeatures.length} available)</div>
        <div className="research-feature-picker" role="listbox" aria-label={`${stageId} features`} aria-orientation="horizontal" onKeyDown={movePickerFocus}>
          {targetedFeatures.map((item) => (
            <button
              key={item.feature.id}
              type="button"
              role="option"
              aria-selected={item.feature.id === feature.id}
              tabIndex={item.feature.id === feature.id ? 0 : -1}
              onClick={() => setFeatureId(item.feature.id)}
            >
              <strong>
                <i
                  className="research-feature-sensor-dot"
                  aria-hidden="true"
                  style={{
                    background:
                      item.family.id === "eye-movement"
                        ? sensorColors.Eyes
                        : item.family.id === "muscle-tone"
                          ? sensorColors["Chin muscle"]
                          : item.family.id === "heart"
                            ? sensorColors.Heart
                            : item.family.id === "breathing"
                              ? sensorColors.Breathing
                              : sensorColors.Brain
                  }}
                />
                {item.feature.name}
              </strong>
              <span>
                <BasisTag featureId={item.feature.id} compact />
                {!modelCFeatureIds.has(item.feature.id) ? <em className="research-not-in-model"> · pruned in 20-signal Model C</em> : null}
              </span>
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
            <p className="research-plot-caption">
              Selected epochs illustrate the expected contrast; markers on the distributions above show where their measurements fall.
            </p>
            {shownExamples.map(({ kind, example }) => (
              <div className="research-epoch-example" key={kind}>
                <div className="research-epoch-example-label">
                  <u className={`research-example-marker is-${kind}`} />
                  <strong className="swatch" style={{ "--swatch": stageColors[example.stage] } as React.CSSProperties}>
                    {example.stage}
                  </strong>
                  <span>
                    {feature.name} {featureValue(example.feature_value)} · {example.record_id} · {formatClock(example.epoch_start_sec)}
                  </span>
                </div>
                <SignalTrace signal={example.signal} active />
              </div>
            ))}
          </div>
        )}

        <Tabs
          label="About this feature"
          tabs={[
            {
              id: "calculated",
              title: "How it is calculated",
              content: (
                <div className="research-feature-tab">
                  <p>{method}</p>
                  {anchor ? (
                    <Link className="research-text-link" href={`/history#code-${anchor}`}>
                      View the Python implementation
                    </Link>
                  ) : null}
                </div>
              )
            },
            {
              id: "fools",
              title: "What can fool it",
              content: (
                <div className="research-feature-tab">
                  <p>
                    {family.limitation} {family.sources.map((number) => <Citation key={number} number={number} />)}
                  </p>
                </div>
              )
            }
          ]}
        />
      </div>
    </div>
  );
}
