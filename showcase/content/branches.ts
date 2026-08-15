import type { StorySectionData } from "./story-types";

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

export const apneaSections: StorySectionData[] = [
  {
    id: "apnea-overview",
    eyebrow: "1. Airflow reduction",
    title: "Airflow reduction is the first read of the obstructive event",
    summary:
      "The chapter should begin on one real obstructive strip and make the airflow reduction legible before anything else is explained.",
    takeaway:
      "The airflow trace alone makes the event legible, before any detector logic gets involved.",
    cards: [
      {
        title: "Clinical definition",
        body:
          "An apnea is scored when airflow drops sharply and stays reduced — the window below shows exactly that pattern, sustained long enough to count as an event."
      },
      {
        title: "Airflow amplitude",
        body:
          "The amplitude drop isn't a brief dip. The reduced breathing pattern holds for more than ten seconds."
      },
      {
        title: "SpO2 drop",
        body:
          "Oxygen saturation follows with a short lag, dropping about 3% as the reduced airflow continues."
      },
      {
        title: "Effort pair",
        body:
          "Thoracic and abdominal effort keep moving through the event — it's their relationship, not just the airflow drop, that turns this into an obstructive case."
      }
    ],
    layout: "four",
    modules: [
      {
        id: "apnea-consensus-logic",
        label: "Consensus logic",
        title: "How the respiratory consensus is built",
        description:
          "The detector filters candidate channels, finds phase extrema, computes per-channel rate tracks, and ranks channels by agreement with the median consensus.",
        whyItMatters:
          "This is the branch's first credibility check before any apnea or hypopnea decision is made.",
        bullets: [
          "Consensus peaks and troughs are promoted into the branch's main respiratory track.",
          "The chosen channel is explicit in the algorithm outputs.",
          "The same agreement logic can be shown visually in the prototype stage."
        ]
      },
      {
        id: "apnea-signal-trust",
        label: "Signal trust",
        title: "Why channel choice belongs in the story",
        description:
          "A respiratory event story is weaker if the site acts as if the input signal were obviously correct from the start.",
        whyItMatters:
          "Showing the trust decision helps the reviewer understand why later detector gates deserve confidence.",
        bullets: [
          "Different channels can disagree or degrade.",
          "Consensus ranking keeps the branch physiology-first instead of channel-first.",
          "This is a better prototype story than opening with a classifier score."
        ]
      }
    ]
  },
  {
    id: "apnea-signals",
    eyebrow: "2. Obstructive pattern",
    title: "Thoracic and abdominal effort turn the event into an obstruction case",
    summary:
      "The second apnea view should center on the effort channels and the paradox signal so the obstruction pattern is visible without narration.",
    takeaway:
      "Effort behavior, not narration, is what marks this event as obstructive.",
    cards: [
      {
        title: "Thoracic effort",
        body:
          "Thoracic effort keeps moving through the reduced-airflow interval instead of pausing with it."
      },
      {
        title: "Abdominal effort",
        body:
          "Abdominal effort moves against thoracic effort instead of with it — the two pull in opposite directions for the duration of the event."
      },
      {
        title: "Paradox correlation",
        body:
          "The paradox signal quantifies that opposition directly: a negative correlation between the two effort channels while the event is active."
      },
      {
        title: "Obstructive label",
        body:
          "The obstruction label marks exactly the window where thoracic and abdominal effort move out of phase."
      }
    ],
    layout: "four",
    modules: [
      {
        id: "apnea-thresholds",
        label: "Threshold logic",
        title: "How apnea and hypopnea thresholds differ",
        description:
          "The code uses different peak-drop thresholds for apnea and hypopnea and then anchors the resulting masks back to physiologic structure.",
        whyItMatters:
          "The reviewer should be able to understand what the detector is asking the signal to prove.",
        bullets: [
          "Apnea and hypopnea do not use the same drop rule.",
          "Rolling baselines make the threshold a local physiologic comparison.",
          "Event masks are anchored, cleaned, and re-tested before they become outputs."
        ]
      },
      {
        id: "apnea-spo2-gating",
        label: "SpO2 gating",
        title: "Why desaturation stays in the main story",
        description:
          "SpO2 is not decorative context in this pipeline. It contributes quality exclusion and event confirmation.",
        whyItMatters:
          "That makes the detector story more credible than a pure airflow threshold narrative.",
        bullets: [
          "Bad SpO2 periods are excluded.",
          "3-point and 4-point drops are tracked explicitly.",
          "Candidate events can be filtered by overlap with desaturation evidence."
        ]
      }
    ]
  },
  {
    id: "apnea-features",
    eyebrow: "3. Feature window",
    title: "The same event, reduced to the features a detector actually uses",
    summary:
      "Once the strip is established, the third apnea view should expose the aligned feature window for that same time segment.",
    takeaway:
      "These features are computed from the exact window shown above, not a separate summary table.",
    cards: [
      {
        title: "Respiration rate",
        body:
          "Respiration rate is derived from this same time window, so a slowdown lines up with the event you can already see."
      },
      {
        title: "Paradox mask",
        body:
          "The paradox mask is a binary trace of the same thoracoabdominal opposition seen in the effort channels."
      },
      {
        title: "Event vector",
        body:
          "The event vector is what actually reaches the grouped feature table used for evaluation — everything above compresses down to this."
      }
    ],
    layout: "three",
    modules: [
      {
        id: "apnea-feature-table",
        label: "Feature table",
        title: "How detector outputs become comparable rows",
        description:
          "The feature-table compiler reduces detector outputs into aligned windows, using different aggregation rules for continuous and binary tracks.",
        whyItMatters:
          "This is the bridge between record-level detector views and grouped experiments.",
        bullets: [
          "Detector signals become windowed columns.",
          "Binary tracks are aggregated differently from continuous tracks.",
          "The compiled rows are what the grouped experiments actually consume."
        ]
      },
      {
        id: "apnea-grouped-validation",
        label: "Grouped validation",
        title: "Why grouped subject-wise evaluation is visible here",
        description:
          "The repo already uses grouped split logic for downstream experiments, and the prototype should make that evaluation discipline part of the branch's visible payoff.",
        whyItMatters:
          "A technically strong reviewer will look for leakage-resistant evaluation immediately.",
        bullets: [
          "Show subject-aware splits, not random row mixing.",
          "Keep F1, sensitivity, and specificity near the feature groups they summarize.",
          "Use benchmark links as optional depth, not as the whole story."
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

export const stagingSections: StorySectionData[] = [
  {
    id: "staging-overview",
    eyebrow: "1. Night structure",
    title: "One full night, read as a structure rather than a sequence of guesses",
    summary:
      "The staging branch should open on the full night, not on an isolated excerpt. The first read is the structure of the night itself.",
    takeaway:
      "The hypnogram is the map; the stage entries below are fixed points into it.",
    cards: [
      {
        title: "Wake entry",
        body:
          "Wake bookends the night and sets the EEG, EOG, and EMG baseline everything else is judged against."
      },
      {
        title: "N2 / N3 entry",
        body:
          "N2/N3 marks the deep-sleep core of the night, not just its beginning and end."
      },
      {
        title: "REM entry",
        body:
          "REM is where eye-movement and muscle-tone signals start to matter as much as EEG."
      }
    ],
    layout: "three",
    modules: [
      {
        id: "staging-spectral-logic",
        label: "Spectral logic",
        title: "Why the EEG view should include more than raw bands",
        description:
          "The code keeps both canonical band power and residual spectral structure so stage evidence does not collapse into one oversimplified chart.",
        whyItMatters:
          "This lets the prototype show real spectral reasoning instead of generic brain-wave decoration.",
        bullets: [
          "Band power is the base layer, not the whole story.",
          "Residual power and aperiodic structure support a more honest spectral view.",
          "Spindle and slow-wave indices should stay attached to that same scene."
        ]
      },
      {
        id: "staging-state-reading",
        label: "State reading",
        title: "Why stage evidence should appear before the classifier",
        description:
          "The prototype should help the reviewer see what differentiates state structure before it asks them to trust a classification summary.",
        whyItMatters:
          "That ordering matches both the branch logic and the repo's interpretability goal.",
        bullets: [
          "Lead with EEG structure.",
          "Then add EOG and EMG support.",
          "Only then move into grouped separation and confusion views."
        ]
      }
    ]
  },
  {
    id: "staging-signals",
    eyebrow: "2. Stage excerpt",
    title: "One excerpt, three channels, read together",
    summary:
      "The second staging view should stop at one fixed excerpt and let the three signal families carry the explanation.",
    takeaway:
      "EEG, EOG, and EMG each carry information the others don't.",
    cards: [
      {
        title: "EEG",
        body:
          "EEG carries spectral structure, spindle content, slow-wave behavior, and short-window arousal signals."
      },
      {
        title: "EOG",
        body:
          "EOG contributes both continuous eye-movement activity and epoch-level REM / SEM summaries."
      },
      {
        title: "EMG",
        body:
          "EMG contributes tone level and suppression events, giving the branch a concrete muscle-tone view instead of a vague 'context' claim."
      }
    ],
    layout: "three",
    modules: [
      {
        id: "staging-eog-emg",
        label: "EOG / EMG logic",
        title: "What the non-EEG channels actually contribute",
        description:
          "The staging code does not treat EOG and EMG as side notes. It turns them into explicit tracks and events that can be shown directly in the prototype.",
        whyItMatters:
          "This is what keeps the branch from drifting into EEG-only staging language.",
        bullets: [
          "Eye-movement activity is continuous and local.",
          "REM / SEM power is aggregated at epoch scale.",
          "EMG suppression gives the branch an event-style muscle-tone signal."
        ]
      },
      {
        id: "staging-arousal",
        label: "Arousal signal",
        title: "Why short-window arousal belongs in the stage story",
        description:
          "The branch also computes an EEG arousal signal, giving the prototype a micro-state disturbance layer that can sit beside slower epoch-level stage summaries.",
        whyItMatters:
          "It helps the staging story feel like state physiology rather than just static label prediction.",
        bullets: [
          "Arousal is computed from short-window spectral power.",
          "The resulting events can be shown directly on the focused scene.",
          "That makes the stage branch feel more alive and less table-first."
        ]
      }
    ]
  },
  {
    id: "staging-features",
    eyebrow: "3. Feature window",
    title: "Feature windows, drawn from the same excerpt just shown",
    summary:
      "The third staging view should show the aligned feature window for the fixed excerpt so the frontend can bridge from raw signals into derived structure.",
    takeaway:
      "This feature window is the bridge from raw signals to the derived structure evaluation actually uses.",
    cards: [
      {
        title: "Spindle index",
        body:
          "Spindle-related features are pulled from the same excerpt shown above."
      },
      {
        title: "Slow-wave index",
        body:
          "Slow-wave features stay tied to this concrete excerpt rather than a detached summary chart."
      },
      {
        title: "REM / tone features",
        body:
          "REM power and muscle-tone features round out the picture EEG alone can't provide."
      }
    ],
    layout: "three",
    modules: [
      {
        id: "staging-feature-screen",
        label: "Feature screen",
        title: "How the repo already screens stage features",
        description:
          "Per-feature Cohen's d and AUC-style separation are already available as diagnostics and should become first-class chart material in the prototype.",
        whyItMatters:
          "That keeps the branch from pretending all extracted features matter equally.",
        bullets: [
          "Show which features separate which classes.",
          "Keep the ranking tied to actual harvested series.",
          "Use popup depth for benchmark caveats, not for the core interpretation."
        ]
      },
      {
        id: "staging-grouped-eval",
        label: "Grouped evaluation",
        title: "Why grouped subject-wise staging results matter",
        description:
          "The experiment runner supports grouped splits and branch-specific summaries, so the prototype should visibly reward honest subject-aware evaluation.",
        whyItMatters:
          "A reviewer's trust goes up when grouped evaluation is part of the visible story rather than buried in a footer.",
        bullets: [
          "Keep grouped split logic visible.",
          "Treat macro-F1 and kappa as consequence views, not starting points.",
          "Use confusion and pairwise difficulty views to explain where the branch is strong or weak."
        ]
      }
    ]
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
