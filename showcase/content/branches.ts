import type { ChapterOverviewData, StorySectionData } from "./story-types";

export const abstractPage = {
  title: "Abstract",
  summary:
    "A concise summary of the project: the clinical problems, the shared dataset, the interpretable signal-processing approach, and the strongest current claims."
};

export const abstractCards = [
  {
    title: "Shared record",
    body:
      "One overnight UCDDB record carries the signals, stage labels, and respiratory-event labels that support both branches of the project."
  },
  {
    title: "Detector stack",
    body:
      "The pipeline runs named algorithms that emit inspectable signals, events, and scalar features instead of collapsing the whole story into one hidden model."
  },
  {
    title: "Grouped evaluation",
    body:
      "Aligned feature windows, subject-wise grouping, and branch-specific experiments are part of the project design, not a cleanup step at the end."
  },
  {
    title: "Two readings",
    body:
      "The same record can be read as a respiratory-event problem or a physiologic-state problem, and the prototype should keep both readings visibly connected."
  }
];

export const apneaPage = {
  title: "Sleep Apnea",
  summary:
    "This branch should read like respiratory-event reasoning: signal trust, detector gates, and grouped evaluation tied back to the same overnight record."
};

export const apneaOverview: ChapterOverviewData = {
  eyebrow: "Apnea intro",
  title: "The overview strip is the apnea intro",
  summary:
    "Start on one real event strip. The clinical framing lives there: airflow is reduced by roughly 90% or more for at least 10 seconds, while effort and desaturation explain what kind of respiratory event this is.",
  takeaway:
    "The first read stays on the strip. Depth only begins after the user chooses airflow amplitude, SpO2 desaturation, or the thorax-abdomen effort pair."
};

export const apneaSections: StorySectionData[] = [
  {
    id: "apnea-airflow",
    eyebrow: "1. Airflow amplitude",
    title: "Airflow amplitude carries the core apnea definition",
    summary:
      "This branch should show the airflow drop directly: the amplitude falls by about 90% and stays down for at least 10 seconds.",
    takeaway:
      "Airflow is the first focused branch because it holds the actual apnea definition.",
    cards: [
      {
        title: "AASM threshold",
        body:
          "Keep the focused view on the visible rule itself: airflow amplitude is reduced by about 90% or more and stays reduced for at least 10 seconds."
      },
      {
        title: "Feature extraction",
        body:
          "The deeper layer is the extraction logic behind that visible drop: local baseline, envelope, and derived airflow features rather than a separate classifier-first explanation."
      }
    ],
    layout: "two",
    modules: [
      {
        id: "apnea-airflow-extraction",
        label: "Feature extraction",
        title: "The deeper layer should be airflow extraction, not a second intro",
        description:
          "Once the amplitude drop is visible, the next layer is how the detector extracts that evidence from the signal: local baseline, envelope, and derived airflow measures.",
        whyItMatters:
          "That keeps the site grounded in the same evidence the user already saw instead of jumping sideways into unrelated detector machinery.",
        bullets: [
          "Show the visible threshold first.",
          "Then show how the signal gets reduced into derived airflow evidence.",
          "Do not add a redundant event-overview branch above this."
        ]
      }
    ]
  },
  {
    id: "apnea-spo2",
    eyebrow: "2. SpO2 desaturation",
    title: "SpO2 shows the downstream consequence of the event",
    summary:
      "This branch should stay simple: reduced airflow is followed by oxygen desaturation, and the deeper layer is the scoring framing around 3% versus 4% drops.",
    takeaway:
      "SpO2 is about consequence and scoring framing, not a separate trust story.",
    cards: [
      {
        title: "Desaturation",
        body:
          "The focused view should show the oxygen drop that follows the respiratory event, not another restatement of the airflow change."
      },
      {
        title: "3% vs 4%",
        body:
          "The deeper layer here is the scoring cutpoint: show how the same drop can be framed against 3% and 4% desaturation conventions."
      }
    ],
    layout: "two",
    modules: [
      {
        id: "apnea-spo2-thresholds",
        label: "Desaturation framing",
        title: "Keep the deeper step on 3% versus 4%",
        description:
          "The site should use the deeper step to explain how the visible drop reads under common desaturation cutpoints.",
        whyItMatters:
          "That is clinically relevant depth. A data-quality side path is not the main story here.",
        bullets: [
          "Lead with the visible drop.",
          "Use the deeper step to compare 3% and 4% framing.",
          "Do not turn this branch into an artifact-quality explanation."
        ]
      }
    ]
  },
  {
    id: "apnea-effort",
    eyebrow: "3. Thorax + abdomen",
    title: "The effort pair explains why the event reads as obstructive",
    summary:
      "This branch should show that effort continues through the event and then deepen once into thorax-abdomen correlation or paradoxical motion.",
    takeaway:
      "Effort ends on correlation. Do not add a later prediction-confirmation step.",
    cards: [
      {
        title: "Effort persists",
        body:
          "The focused view should keep thoracic and abdominal effort visible through the event so the user can see that breathing effort continues instead of stopping."
      },
      {
        title: "Correlation / paradox",
        body:
          "The deeper step is the correlation view itself: thorax and abdomen move out of phase, which is the clean obstruction-specific ending for this branch."
      }
    ],
    layout: "two",
    modules: [
      {
        id: "apnea-effort-correlation",
        label: "Correlation",
        title: "Stop the branch at thorax-abdomen correlation",
        description:
          "Once the user sees preserved effort, the last meaningful step is the correlation or paradox view between the thoracic and abdominal channels.",
        whyItMatters:
          "That is the obstruction-specific signal logic. A prediction-confirmation layer adds clutter instead of clarity.",
        bullets: [
          "Show preserved effort first.",
          "Then show the correlation or paradox signal.",
          "Do not add a prediction-track ending after that."
        ]
      }
    ]
  }
];

export const stagingPage = {
  title: "Sleep Staging",
  summary:
    "This branch should read like state reasoning: spectral structure, signal-family roles, and grouped class separation tied back to the same overnight record."
};

export const stagingOverview: ChapterOverviewData = {
  eyebrow: "Night structure",
  title: "The whole-night hypnogram comes first",
  summary:
    "Start with the night structure and let the user choose a real stage anchor. The branch should only deepen after a stage has been selected.",
  takeaway:
    "Do not force one generic EEG ladder. The first question is always why the chosen stage label makes sense."
};

const stagingStageBase: StorySectionData = {
  id: "staging-stage",
  eyebrow: "1. Stage view",
  title: "Stage-specific signal view",
  summary:
    "A selected stage should open a real excerpt with only the depth that matches that stage.",
  takeaway:
    "The staging branch should deepen differently for Wake, N1, N2, N3, and REM.",
  cards: [
    {
      title: "Stage evidence",
      body: "Show the chosen stage excerpt first."
    }
  ],
  layout: "two",
  modules: []
};

const stagingStageByLabel: Record<string, Omit<StorySectionData, "id" | "layout">> = {
  wake: {
    eyebrow: "1. Wake",
    title: "Wake should read like the baseline state",
    summary:
      "Wake is the comparison point for everything else: wake-like mixed activity and higher muscle tone before the branch moves deeper.",
    takeaway:
      "Wake works best as a clear baseline rather than an overbuilt depth tree.",
    cards: [
      {
        title: "Wake physiology",
        body:
          "Keep the first read on wake-like activity and higher chin tone so the user understands what the other stages are diverging from."
      },
      {
        title: "Tone baseline",
        body:
          "The deeper step can stay simple here: muscle tone remains higher than it will in deeper sleep or REM."
      }
    ],
    modules: []
  },
  n1: {
    eyebrow: "2. N1",
    title: "N1 should read like transition into sleep",
    summary:
      "The N1 view should feel transitional rather than fully settled: early sleep onset, lighter state structure, and less stable wake-like patterning.",
    takeaway:
      "N1 does not need the same depth as N2 or N3. It just needs to read as transition.",
    cards: [
      {
        title: "Sleep onset",
        body:
          "Keep the first read on the transition itself: the excerpt should feel less like wake and not yet fully organized like deeper sleep."
      },
      {
        title: "Transition cues",
        body:
          "If the branch goes deeper here, keep it on simple transition cues rather than forcing a spindle or slow-wave explanation that belongs to other stages."
      }
    ],
    modules: []
  },
  n2: {
    eyebrow: "3. N2",
    title: "N2 should deepen into spindle detail",
    summary:
      "The N2 excerpt should first make the stage feel defensible, then use spindle detail as the natural deeper layer.",
    takeaway:
      "N2 is where spindle detail belongs.",
    cards: [
      {
        title: "Stage evidence",
        body:
          "Start with the excerpt itself and let the combination of EEG, EOG, and EMG make the stage label feel readable before adding feature detail."
      },
      {
        title: "Spindle detail",
        body:
          "The deeper layer for N2 is spindle activity. Keep the detail attached to the chosen excerpt instead of turning it into a generic EEG explainer."
      }
    ],
    modules: []
  },
  n3: {
    eyebrow: "4. N3",
    title: "N3 should deepen into slow-wave detail",
    summary:
      "The N3 excerpt should make deep sleep obvious first, then deepen into slow-wave detail rather than broad spectral narration.",
    takeaway:
      "N3 is where slow-wave detail belongs.",
    cards: [
      {
        title: "Stage evidence",
        body:
          "Start with the deep-sleep excerpt itself so the user sees the dominant slow-wave structure before any derived feature explanation."
      },
      {
        title: "Slow-wave detail",
        body:
          "The deeper layer for N3 is slow-wave detail tied directly to the selected excerpt."
      }
    ],
    modules: []
  },
  rem: {
    eyebrow: "5. REM",
    title: "REM should deepen into eye-movement and tone detail",
    summary:
      "The REM excerpt should first show why the stage label makes sense, then deepen into the combined EOG and EMG story.",
    takeaway:
      "REM is the one stage where eye movement and muscle tone should carry the deeper read together.",
    cards: [
      {
        title: "Stage evidence",
        body:
          "Start with the REM excerpt itself: the stage should read from the signal view before the branch opens the feature layer."
      },
      {
        title: "EOG / EMG detail",
        body:
          "The deeper layer is the eye-movement and tone combination: rapid eye activity with reduced chin tone."
      }
    ],
    modules: []
  }
};

export function getStagingStageSection(stageLabel: string | null): StorySectionData {
  const resolved = stageLabel ? stagingStageByLabel[stageLabel.toLowerCase()] : null;
  return {
    ...stagingStageBase,
    ...(resolved ?? stagingStageByLabel.n2)
  };
}

export const stagingSections: StorySectionData[] = [
  {
    ...stagingStageBase
  }
];

export const aboutPage = {
  title: "About",
  summary:
    "A lightweight edge page that explains why the project exists, what role the work demonstrates, and why the presentation stays physiology-first."
};

export const aboutCards = [
  {
    title: "Why this project exists",
    body:
      "The project is meant to show clinically grounded signal-processing judgment, modular detector design, and careful technical communication."
  },
  {
    title: "Role in the work",
    body:
      "The work combines dataset understanding, detector translation, feature design, grouped evaluation discipline, and the prototype storytelling layer."
  },
  {
    title: "Relevant background",
    body:
      "Keep this focused on the experience that supports the project itself: algorithmic thinking, signal analysis, and technical product storytelling."
  }
];
