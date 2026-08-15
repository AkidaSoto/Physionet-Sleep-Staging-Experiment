import type { ReactNode } from "react";
import type { ShowcaseSection } from "../../content/showcase-flow";
import type { StoryCard } from "../../content/story-types";
import { Text } from "../ui/Text";

type CalloutAnchor = {
  x: string;
  y: string;
  width?: string;
  align?: "left" | "right";
};

type Props = {
  section: ShowcaseSection;
  sections: ShowcaseSection[];
  activeSectionId: string | null;
  onSectionSelect: (sectionId: string) => void;
  activeCardIndex: number | null;
  onCardSelect: (cardIndex: number) => void;
  getSectionLabel: (section: ShowcaseSection) => string;
};

function SignalSvg({
  lines,
  className = ""
}: {
  lines: Array<{ path: string; tone: "a" | "b" | "c" | "d" }>;
  className?: string;
}) {
  return (
    <svg viewBox="0 0 600 220" className={["scene-svg", className].filter(Boolean).join(" ")} aria-hidden="true">
      {lines.map((line) => (
        <path key={`${line.tone}-${line.path}`} d={line.path} className={`scene-path scene-path-${line.tone}`} />
      ))}
    </svg>
  );
}

function OrbitScene() {
  return (
    <div className="scene-orbit">
      <span className="scene-orbit-ring scene-orbit-ring-a" />
      <span className="scene-orbit-ring scene-orbit-ring-b" />
      <span className="scene-orbit-ring scene-orbit-ring-c" />
      <span className="scene-orbit-core">Sleep</span>
      <span className="scene-orbit-node scene-orbit-node-a">Apnea</span>
      <span className="scene-orbit-node scene-orbit-node-b">Staging</span>
      <span className="scene-orbit-node scene-orbit-node-c">Signals</span>
      <span className="scene-orbit-node scene-orbit-node-d">Features</span>
    </div>
  );
}

function DatasetScene() {
  return (
    <div className="scene-dataset">
      <div className="scene-dataset-strip scene-dataset-strip-a" />
      <div className="scene-dataset-strip scene-dataset-strip-b" />
      <div className="scene-dataset-strip scene-dataset-strip-c" />
      <div className="scene-dataset-rail">
        <span>PSG</span>
        <span>Stages</span>
        <span>Respiratory Events</span>
      </div>
      <div className="scene-dataset-notes">
        <span>25 overnight records</span>
        <span>Shared subject context</span>
      </div>
    </div>
  );
}

function MethodScene() {
  return (
    <div className="scene-method">
      <div className="scene-method-node">Clinical read</div>
      <div className="scene-method-link" />
      <div className="scene-method-node">Signal evidence</div>
      <div className="scene-method-link" />
      <div className="scene-method-node">Interpretable features</div>
    </div>
  );
}

function ApneaEventScene() {
  return (
    <div className="scene-stack">
      <div className="scene-stack-row">
        <span className="scene-stack-label">Airflow</span>
        <SignalSvg
          lines={[{ path: "M10 110 C60 40 110 40 160 110 S260 180 310 110 S410 40 460 110 S540 180 590 110", tone: "a" }]}
        />
      </div>
      <div className="scene-stack-row">
        <span className="scene-stack-label">Effort</span>
        <SignalSvg
          lines={[{ path: "M10 110 C60 70 110 70 160 110 S260 150 310 110 S410 70 460 110 S540 150 590 110", tone: "b" }]}
        />
      </div>
      <div className="scene-stack-event">event window</div>
    </div>
  );
}

function RespiratorySignalsScene() {
  return (
    <div className="scene-stack scene-stack-dense">
      <div className="scene-stack-row">
        <span className="scene-stack-label">Airflow</span>
        <SignalSvg lines={[{ path: "M10 112 C50 54 90 54 130 112 S210 170 250 112 S330 54 370 112 S450 170 490 112 S550 54 590 112", tone: "a" }]} />
      </div>
      <div className="scene-stack-row">
        <span className="scene-stack-label">Thoracic</span>
        <SignalSvg lines={[{ path: "M10 112 C50 82 90 82 130 112 S210 142 250 112 S330 82 370 112 S450 142 490 112 S550 82 590 112", tone: "b" }]} />
      </div>
      <div className="scene-stack-row">
        <span className="scene-stack-label">Abdominal</span>
        <SignalSvg lines={[{ path: "M10 112 C50 90 90 90 130 112 S210 134 250 112 S330 90 370 112 S450 134 490 112 S550 90 590 112", tone: "c" }]} />
      </div>
      <div className="scene-stack-row">
        <span className="scene-stack-label">SpO2</span>
        <SignalSvg lines={[{ path: "M10 88 L120 88 L180 104 L270 104 L320 122 L420 122 L500 108 L590 108", tone: "d" }]} />
      </div>
    </div>
  );
}

function ApneaFeaturesScene() {
  return (
    <div className="scene-metrics">
      <div className="scene-metric-bars">
        <span style={{ height: "42%" }} />
        <span style={{ height: "68%" }} />
        <span style={{ height: "84%" }} />
        <span style={{ height: "58%" }} />
      </div>
      <div className="scene-metric-grid">
        <span className="scene-metric-cell strong" />
        <span className="scene-metric-cell" />
        <span className="scene-metric-cell mid" />
        <span className="scene-metric-cell" />
        <span className="scene-metric-cell strong" />
        <span className="scene-metric-cell mid" />
        <span className="scene-metric-cell" />
        <span className="scene-metric-cell strong" />
        <span className="scene-metric-cell mid" />
      </div>
    </div>
  );
}

function StagingArcScene() {
  return (
    <div className="scene-stages">
      <div className="scene-stage-band">
        <span>N1</span>
        <span>N2</span>
        <span>N3</span>
        <span>REM</span>
      </div>
      <div className="scene-stage-curve" />
      <div className="scene-stage-pulse scene-stage-pulse-a" />
      <div className="scene-stage-pulse scene-stage-pulse-b" />
      <div className="scene-stage-pulse scene-stage-pulse-c" />
    </div>
  );
}

function StagingSignalsScene() {
  return (
    <div className="scene-stack scene-stack-dense">
      <div className="scene-stack-row">
        <span className="scene-stack-label">EEG</span>
        <SignalSvg lines={[{ path: "M10 112 L40 94 L60 126 L90 88 L120 120 L160 100 L210 124 L250 82 L300 118 L340 92 L390 126 L430 86 L490 118 L540 94 L590 112", tone: "a" }]} />
      </div>
      <div className="scene-stack-row">
        <span className="scene-stack-label">EOG</span>
        <SignalSvg lines={[{ path: "M10 112 C60 112 60 78 110 78 S160 146 220 146 S280 92 340 92 S400 138 460 138 S520 96 590 96", tone: "c" }]} />
      </div>
      <div className="scene-stack-row">
        <span className="scene-stack-label">EMG</span>
        <SignalSvg lines={[{ path: "M10 126 L30 98 L50 132 L80 102 L110 128 L140 100 L170 130 L210 106 L260 128 L320 104 L390 126 L470 106 L590 122", tone: "d" }]} />
      </div>
    </div>
  );
}

function StagingMatrixScene() {
  return (
    <div className="scene-matrix">
      <div className="scene-matrix-grid">
        {["strong", "mid", "", "", "mid", "strong", "mid", "", "", "mid", "strong", "mid", "", "", "mid", "strong"].map(
          (tone, index) => (
            <span key={`${tone}-${index}`} className={["scene-matrix-cell", tone].filter(Boolean).join(" ")} />
          )
        )}
      </div>
      <div className="scene-matrix-metrics">
        <div>
          <Text as="span" variant="meta" tone="secondary">
            kappa
          </Text>
          <Text as="span" variant="sectionTitle">
            0.74
          </Text>
        </div>
        <div>
          <Text as="span" variant="meta" tone="secondary">
            macro-F1
          </Text>
          <Text as="span" variant="sectionTitle">
            0.78
          </Text>
        </div>
      </div>
    </div>
  );
}

function AboutScene({ kind }: { kind: ShowcaseSection["scene"] }) {
  return (
    <div className={`scene-about scene-about-${kind}`}>
      <div className="scene-about-panel scene-about-panel-a" />
      <div className="scene-about-panel scene-about-panel-b" />
    </div>
  );
}

function getCalloutAnchors(scene: ShowcaseSection["scene"], count: number): CalloutAnchor[] {
  const anchorsByScene: Record<ShowcaseSection["scene"], CalloutAnchor[]> = {
    "problem-orbit": [
      { x: "18%", y: "18%", width: "180px" },
      { x: "68%", y: "62%", width: "180px", align: "right" }
    ],
    "dataset-timeline": [{ x: "12%", y: "74%", width: "260px" }],
    "method-bridge": [{ x: "36%", y: "72%", width: "220px" }],
    "apnea-event": [
      { x: "20%", y: "12%", width: "180px" },
      { x: "62%", y: "16%", width: "180px", align: "right" },
      { x: "50%", y: "74%", width: "190px", align: "right" }
    ],
    "respiratory-signals": [
      { x: "13%", y: "9%", width: "170px" },
      { x: "74%", y: "28%", width: "170px", align: "right" },
      { x: "14%", y: "49%", width: "170px" },
      { x: "74%", y: "70%", width: "170px", align: "right" }
    ],
    "apnea-features": [
      { x: "12%", y: "12%", width: "170px" },
      { x: "36%", y: "74%", width: "170px" },
      { x: "74%", y: "20%", width: "170px", align: "right" }
    ],
    "staging-arc": [
      { x: "12%", y: "16%", width: "170px" },
      { x: "72%", y: "18%", width: "190px", align: "right" },
      { x: "44%", y: "72%", width: "190px", align: "right" }
    ],
    "staging-signals": [
      { x: "14%", y: "10%", width: "170px" },
      { x: "74%", y: "38%", width: "170px", align: "right" },
      { x: "14%", y: "72%", width: "170px" }
    ],
    "staging-matrix": [
      { x: "10%", y: "16%", width: "180px" },
      { x: "74%", y: "18%", width: "180px", align: "right" },
      { x: "14%", y: "72%", width: "180px" }
    ],
    "about-motive": [{ x: "16%", y: "18%", width: "210px" }],
    "about-role": [{ x: "66%", y: "54%", width: "210px", align: "right" }],
    "about-background": [{ x: "18%", y: "62%", width: "230px" }]
  };

  return anchorsByScene[scene].slice(0, count);
}

function SceneCallouts({
  section,
  activeCardIndex,
  onCardSelect
}: {
  section: ShowcaseSection;
  activeCardIndex: number | null;
  onCardSelect: (cardIndex: number) => void;
}) {
  const anchors = getCalloutAnchors(section.scene, section.cards.length);

  return (
    <div className="scene-callouts">
      {section.cards.map((card: StoryCard, index) => {
        const anchor = anchors[index];
        const active = activeCardIndex === index;

        if (!anchor) {
          return null;
        }

        return (
          <div
            key={card.title}
            className={["scene-callout", active ? "scene-callout-active" : "", anchor.align === "right" ? "scene-callout-right" : ""]
              .filter(Boolean)
              .join(" ")}
            style={{ left: anchor.x, top: anchor.y, width: anchor.width }}
          >
            <button type="button" className="scene-callout-trigger" onClick={() => onCardSelect(index)}>
              <Text as="span" variant="sectionTitle" className="scene-callout-title">
                {card.title}
              </Text>
            </button>

            {active ? (
              <div className="scene-callout-body">
                <Text tone="secondary">{card.body}</Text>
              </div>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}

function SceneNav({
  sections,
  activeSectionId,
  onSectionSelect,
  getSectionLabel
}: {
  sections: ShowcaseSection[];
  activeSectionId: string | null;
  onSectionSelect: (sectionId: string) => void;
  getSectionLabel: (section: ShowcaseSection) => string;
}) {
  return (
    <div className="chapter-scene-nav" role="tablist" aria-label="Chapter sections">
      {sections.map((item) => {
        const active = item.id === activeSectionId;

        return (
          <button
            key={item.id}
            type="button"
            role="tab"
            aria-selected={active}
            className={["chapter-scene-step", active ? "chapter-scene-step-active" : ""].filter(Boolean).join(" ")}
            onClick={() => onSectionSelect(item.id)}
          >
            <span className="chapter-scene-step-dot" />
            <Text as="span" variant="meta" className="chapter-scene-step-label">
              {getSectionLabel(item)}
            </Text>
          </button>
        );
      })}
    </div>
  );
}

export function ShowcaseScene({
  section,
  sections,
  activeSectionId,
  onSectionSelect,
  activeCardIndex,
  onCardSelect,
  getSectionLabel
}: Props) {
  let body: ReactNode = null;

  switch (section.scene) {
    case "problem-orbit":
      body = <OrbitScene />;
      break;
    case "dataset-timeline":
      body = <DatasetScene />;
      break;
    case "method-bridge":
      body = <MethodScene />;
      break;
    case "apnea-event":
      body = <ApneaEventScene />;
      break;
    case "respiratory-signals":
      body = <RespiratorySignalsScene />;
      break;
    case "apnea-features":
      body = <ApneaFeaturesScene />;
      break;
    case "staging-arc":
      body = <StagingArcScene />;
      break;
    case "staging-signals":
      body = <StagingSignalsScene />;
      break;
    case "staging-matrix":
      body = <StagingMatrixScene />;
      break;
    case "about-motive":
    case "about-role":
    case "about-background":
      body = <AboutScene kind={section.scene} />;
      break;
  }

  return (
    <div className={`chapter-scene chapter-scene-${section.scene}`}>
      {body}
      <SceneCallouts section={section} activeCardIndex={activeCardIndex} onCardSelect={onCardSelect} />
      <SceneNav
        sections={sections}
        activeSectionId={activeSectionId}
        onSectionSelect={onSectionSelect}
        getSectionLabel={getSectionLabel}
      />
    </div>
  );
}
