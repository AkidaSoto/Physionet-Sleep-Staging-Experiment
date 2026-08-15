import { abstractCards, apneaSections, stagingSections, aboutCards } from "./branches";
import type { StorySectionData } from "./story-types";

export type ResearchLink = {
  id: string;
  label: string;
  title: string;
  summary: string;
  url: string;
};

export type ShowcaseSceneKind =
  | "problem-orbit"
  | "dataset-timeline"
  | "method-bridge"
  | "apnea-event"
  | "respiratory-signals"
  | "apnea-features"
  | "staging-arc"
  | "staging-signals"
  | "staging-matrix"
  | "about-motive"
  | "about-role"
  | "about-background";

export type ShowcaseSection = StorySectionData & {
  scene: ShowcaseSceneKind;
};

export type ShowcaseChapter = {
  id: "abstract" | "apnea" | "staging" | "about";
  title: string;
  sections: ShowcaseSection[];
  references?: ResearchLink[];
};

export const showcaseTitle = "PhysioNet Sleep Explorer";

const abstractSections: ShowcaseSection[] = [
  {
    id: "abstract-record",
    eyebrow: "1. Shared record",
    title: "One overnight record supports two different readings",
    summary:
      "The prototype begins with one UCDDB night: synchronized PSG channels, stage labels, and respiratory-event annotations aligned to the same recording.",
    takeaway:
      "The project stays coherent because both branches read the same night instead of switching datasets or stories halfway through.",
    cards: [abstractCards[0], abstractCards[3]],
    layout: "two",
    scene: "problem-orbit",
    modules: [
      {
        id: "abstract-branch-split",
        label: "Branch split",
        title: "Why one record becomes two branches",
        description:
          "The same record supports both state reading and respiratory-event reading, but those are different clinical questions and should stay visually separate.",
        whyItMatters:
          "The split keeps the site honest: one foundation, two interpretations, no blended methods wall.",
        bullets: [
          "Sleep apnea reads the night as a respiratory-event problem.",
          "Sleep staging reads the same night as a physiologic-state problem.",
          "The branch pages hold the depth so the abstract can stay visual and short."
        ]
      }
    ]
  },
  {
    id: "abstract-detectors",
    eyebrow: "2. Detector stack",
    title: "Modular detectors keep the physiology visible",
    summary:
      "The codebase runs named algorithms in sequence, each emitting detector signals, events, and scalar features instead of hiding everything inside one end-to-end model.",
    takeaway:
      "The strongest current product story is not just prediction. It is a visible detector stack that stays close to the underlying physiology.",
    cards: [abstractCards[1]],
    layout: "two",
    scene: "dataset-timeline",
    modules: [
      {
        id: "abstract-modules",
        label: "Pipeline shape",
        title: "What the pipeline already exposes",
        description:
          "Each algorithm returns named outputs that the showcase can render directly: tracks, events, and features.",
        whyItMatters:
          "That structure is exactly what lets the site feel like a guided review rather than a vague modeling summary.",
        bullets: [
          "Respiration, SpO2, ECG, and staging algorithms all produce inspectable outputs.",
          "Aligned truth tracks are built after the detectors, not assumed upfront.",
          "Feature tables are compiled from those outputs for grouped experiments."
        ]
      }
    ]
  },
  {
    id: "abstract-evaluation",
    eyebrow: "3. Evaluation",
    title: "Grouped evaluation is part of the prototype, not an appendix",
    summary:
      "The pipeline compiles aligned windows, screens features, and runs grouped subject-wise experiments so the strongest claims have an honest evaluation surface.",
    takeaway:
      "The prototype should end in subject-wise evidence, not just in attractive signal examples.",
    cards: [abstractCards[2]],
    layout: "two",
    scene: "method-bridge",
    modules: [
      {
        id: "abstract-grouped-eval",
        label: "Why grouped",
        title: "Why grouped validation belongs in the main story",
        description:
          "The repo already uses grouped subject-aware evaluation paths, and that should stay visible in the prototype rather than getting buried in a benchmark footnote.",
        whyItMatters:
          "A technically strong reviewer will trust the work more if leakage-resistant evaluation is part of the narrative itself.",
        bullets: [
          "Feature quality should be judged by class separation and stage fit.",
          "Prediction quality should be judged with grouped folds or leave-one-subject-out logic.",
          "Metrics should stay tied to the signal families and feature families that produced them."
        ]
      }
    ]
  }
];

const apneaSceneById = {
  "apnea-overview": "apnea-event",
  "apnea-signals": "respiratory-signals",
  "apnea-features": "apnea-features"
} as const satisfies Record<string, ShowcaseSceneKind>;

const stagingSceneById = {
  "staging-overview": "staging-arc",
  "staging-signals": "staging-signals",
  "staging-features": "staging-matrix"
} as const satisfies Record<string, ShowcaseSceneKind>;

const apneaShowcaseSections: ShowcaseSection[] = apneaSections.map((section) => ({
  ...section,
  scene: apneaSceneById[section.id as keyof typeof apneaSceneById]
}));

const stagingShowcaseSections: ShowcaseSection[] = stagingSections.map((section) => ({
  ...section,
  scene: stagingSceneById[section.id as keyof typeof stagingSceneById]
}));

const aboutSections: ShowcaseSection[] = [
  {
    id: "about-purpose",
    eyebrow: "1. Why",
    title: "Why this project exists",
    summary: aboutCards[0].body,
    takeaway: "The portfolio layer should reinforce the work, not interrupt it.",
    cards: [aboutCards[0]],
    layout: "two",
    scene: "about-motive",
    modules: []
  },
  {
    id: "about-role",
    eyebrow: "2. Role",
    title: "Role in the work",
    summary: aboutCards[1].body,
    takeaway: "The role should stay concrete and tied to the technical output.",
    cards: [aboutCards[1]],
    layout: "two",
    scene: "about-role",
    modules: []
  },
  {
    id: "about-background",
    eyebrow: "3. Background",
    title: "Relevant background",
    summary: aboutCards[2].body,
    takeaway: "Keep only the context that sharpens trust in the project itself.",
    cards: [aboutCards[2]],
    layout: "two",
    scene: "about-background",
    modules: []
  }
];

export function getShowcaseChapters(researchLinks: ResearchLink[]): ShowcaseChapter[] {
  const chapterLinks = {
    abstract: researchLinks,
    apnea: researchLinks.filter((link) => ["ucddb", "apnea-benchmark"].includes(link.id)),
    staging: researchLinks.filter((link) => ["ucddb", "staging-benchmark"].includes(link.id))
  };

  return [
    {
      id: "abstract",
      title: "Abstract",
      sections: abstractSections,
      references: chapterLinks.abstract
    },
    {
      id: "apnea",
      title: "Sleep Apnea",
      sections: apneaShowcaseSections,
      references: chapterLinks.apnea
    },
    {
      id: "staging",
      title: "Sleep Staging",
      sections: stagingShowcaseSections,
      references: chapterLinks.staging
    },
    {
      id: "about",
      title: "About",
      sections: aboutSections
    }
  ];
}

export function getShowcaseChapter(
  chapterId: ShowcaseChapter["id"],
  researchLinks: ResearchLink[]
) {
  return getShowcaseChapters(researchLinks).find((chapter) => chapter.id === chapterId);
}
