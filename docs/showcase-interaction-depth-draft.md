# Showcase Interaction Depth

Status: current working flow for the showcase branch structure.

## Ground rules

- The top level stays shallow: `Home`, `Abstract`, `Sleep Apnea`, `Sleep Staging`, `About`.
- Depth only happens inside the branch the user chose.
- The overview itself is the intro. Do not create a redundant "overview of the overview" scene.
- Every deeper step has to stay clinically legible. Mechanics only appear after the user has already chosen what evidence to inspect.
- Staging depth is stage-specific. Do not force one generic EEG ladder on every stage.

---

## Apnea

The overview strip is the apnea intro.

That intro should do the actual AASM-style clinical framing:

- airflow is reduced by about `90% or more`
- the reduction lasts `10 seconds or longer`
- effort and desaturation help explain what kind of event this is

There is no separate `clinical relevance` branch before the real content. The strip already does that job.

### Apnea tree

```text
Sleep Apnea
│
└── Overview strip
    Airflow / Thoracic effort / Abdominal effort / SpO2
    event-window highlight
    AASM intro lives here
    │
    ├── Airflow amplitude
    │   reduced by >=90% for >=10 s
    │   └── deeper → feature extraction
    │       envelope / baseline / derived airflow logic
    │
    ├── SpO2 desaturation
    │   oxygen drop after the event
    │   └── deeper → 3% vs 4% desaturation framing
    │
    └── Thorax + abdomen
        effort continues through the event
        └── deeper → thorax-abdomen correlation / paradox
```

### Apnea rules

- No redundant event-overview branch.
- No top-level obstruction-truth or prediction-track branch.
- Airflow is the first focused branch because it carries the core apnea definition.
- SpO2 is about consequence and scoring framing, not data-quality narration.
- Effort ends on correlation / paradox. Do not add a later prediction-confirmation step.

---

## Sleep staging

The overview is the whole-night hypnogram.

The user picks a real stage anchor from the night. Only after that does the branch deepen.

### Staging tree

```text
Sleep Staging
│
└── Whole-night hypnogram
    choose Wake / N1 / N2 / N3 / REM
    │
    ├── Wake
    │   wake-like mixed activity + higher muscle tone
    │
    ├── N1
    │   transition into sleep
    │
    ├── N2
    │   stage-specific signal view
    │   └── deeper → spindle detail
    │
    ├── N3
    │   stage-specific signal view
    │   └── deeper → slow-wave detail
    │
    └── REM
        eye movements + low chin tone
        └── deeper → EOG / EMG detail
```

### Staging rules

- Do not go `EEG -> spindle -> slow wave`.
- The first question is always: why does this chosen stage label make sense?
- Any deeper feature has to match the chosen stage.
- `N2` can deepen into spindle detail.
- `N3` can deepen into slow-wave detail.
- `REM` can deepen into eye-movement and tone detail.
- `Wake` and `N1` can stay shallower if that reads more cleanly.

---

## Full site flow

```text
Home
├── Abstract
├── Sleep Apnea
│   └── overview strip + apnea intro
│       ├── Airflow amplitude → feature extraction
│       ├── SpO2 desaturation → 3% vs 4%
│       └── Thorax + abdomen → correlation / paradox
├── Sleep Staging
│   └── whole-night hypnogram
│       ├── Wake
│       ├── N1
│       ├── N2 → spindle detail
│       ├── N3 → slow-wave detail
│       └── REM → EOG / EMG detail
└── About
```

## Back behavior

- deep detail -> stage or evidence focus
- focus -> branch overview
- branch overview -> home
