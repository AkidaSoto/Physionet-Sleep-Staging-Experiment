# Dashboard audit: AI-slop design and writing (2026-11-19)

Scope: `showcase/` single-page paper (localhost:3000). Method: 10 external sources on AI-slop UI and AI writing tells, three adversarial navigation passes (hostile editorial designer, lay visitor from a résumé link, reviewer #2 in sleep-ML), plus a code audit. No code was changed.

## Sources

| # | Source | Tells used here |
|---|---|---|
| 1 | Wikipedia, *Signs of AI writing* — https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing | negative parallelism, rule of three, mid-prose bold, section-summary closers, inflated significance |
| 2 | Kobak et al., *Science Advances* 2025 — https://arxiv.org/html/2406.07016v5 | excess style words in biomedical abstracts (underscores, crucial, notably, findings) |
| 3 | Reinhart et al., *PNAS* 2025 — https://arxiv.org/html/2410.16107v2 | participial clauses, nominalisation density |
| 4 | Decrypt / Washington Post analysis, Nov 2025 — https://decrypt.co/?p=348923 | "not just X, but Y" in ~6% of messages |
| 5 | Anthropic, *Improving frontend design through Skills*, Nov 2025 — https://claude.com/blog/improving-frontend-design-through-skills | default fonts, clichéd palettes, predictable components ("distributional convergence") |
| 6 | prg.sh, *Why your AI keeps building the same purple gradient website*, Oct 2025 — https://prg.sh/ramblings/Why-Your-AI-Keeps-Building-the-Same-Purple-Gradient-Website | uniform rounded cards, faint shadows, three-card rows |
| 7 | NN/g, *AI Prototyping in Real Design Contexts*, Oct 2025 — https://www.nngroup.com/articles/ai-prototyping/ | bright accents, poor contrast, inconsistent grouping, unrequested features |
| 8 | Fountain Institute, *7 Signs a UI Has Been Vibe Coded*, 2026 — https://www.thefountaininstitute.com/blog/signs-vibe-coded-ui | neon-on-dark glow, nested cards, meaningless status dots |
| 9 | Built In, *AI Makes Bad Design Look Good Enough*, Jun 2026 — https://builtin.com/articles/ai-design-slop-era | stock card dashboards |
| 10 | Paul Bakaus, *Slop* — https://impeccable.style/slop | hero metric tiles, glow borders, neon on dark, nested cards |

## Findings

### A. Trust breakers (fix before anything visual)

Numbers that contradict each other on the page (verified in source):

- Learned values for C: 28,670 (`staging-dashboard.json:5355`) vs 30,134 (`:5832`), both presented as model size.
- Evidence window: "6–12 s" (`StagingResearchDashboard.tsx:1714`) vs "6–18 s" (`:1809`, exp #21).
- Scaling: "scaled within each night" (`:2445`) vs "scaled per subject, then standardized with the training subjects" (`:501`, `:2283`).
- Inputs: §2.3 says 20 signals × 25 = 500; S2 "Simpler models" uses "the same 675 inputs (27 signals)" (`:501`, `:1694`) without saying it is the pre-reduction set.
- Reported by agents, not yet traced: 500 vs 470 trees (`:1021` "500 trees, library defaults" vs implementation panel); previous-epoch weight 4.4× vs 3.8×; additive model 0.749 vs 0.751; experiment numbering skips #27.

Scientific framing a reviewer would challenge:

- The abstract leads with CV 0.773, which is the selection metric (features were swapped and pruned on the same folds under the 0.005 rule). Held-out 0.729 should lead.
- No confidence intervals anywhere; B/C/D ranking (0.777/0.773/0.769) is within fold noise, yet S3 argues from a 0.004 gap.
- Held-out per-stage cells are all "—" in S1.
- Coimbra headline uses scorer 1 only (0.761 vs 0.751 for scorer 2).
- Spindle panel says "Matches the expectation" for N2 > N3 while both medians are 4.0.
- §3.2 summed "Other …" bars (0.052) are drawn like real measurements and visually win.
- S5 timeline joins UCDDB, Boston CV and held-out scores into one series.

### B. Visual slop (UI)

| Tell | Where | Sources |
|---|---|---|
| Neon cyan on navy, radial blooms, glows | `design/tokens.ts`, `globals.css` (38 linear + 21 radial gradients, 17 shadows, 5 backdrop-filters; body blooms ~L1350) | 8, 10 |
| Hero metric tiles | §3.1 four KPI cards (0.773 / 78.6% / 79.5% / 0.72) | 10 |
| Pill/badge confetti | objective chips on every subheading (`.research-objective-chip`, `SubHeading` ~L2422); AASM/Published/Adapted chips; row pills in §1.2 table | 8, 10 |
| Boxed everything, box-and-arrow "diagrams" | §1.1 EEG/EOG/EMG → epoch → hypnogram; §2.3 inputs → boosting → stage | 6, 8, 10 |
| Uppercase + letter-spacing labels | 36 uppercase rules, 46 letter-spacing | 7 |
| No type system | 37 computed font sizes; most common 12.48 px; 10–11.5 px alpha text ≈3.5:1 contrast | 7 |
| Decorative chrome | gradient title rule, scroll-progress bar, halo dots | 8, 9 |
| Things that look clickable but aren't | objective chips, legend chips, metric cards with hover styling | 7 |

### C. Writing slop

| Tell | Examples | Sources |
|---|---|---|
| "Not X but Y" / "X rather than Y" (≥10) | "the starting point, not evidence that a feature works" (§2.2); "converging evidence rather than a replication" (§4); "model dependence, not a proven mechanism" (S3) | 1, 4 |
| Rule of three | three objectives, three discussion points, three limitations, three next steps, three working rules; "the gain, the remaining stage errors, and where published models stand" | 1 |
| Mid-prose bold | "**these physiological cues**", "**physiologically motivated measurements**" (abstract) | 1 |
| Summary closer | "Together, these findings show how…" (abstract) | 1 |
| Question / colon headings | "Testing: does it hold up?", "Do heart and breathing earn their place?", "Model: the signals in context" | house heuristic |
| Inflated / stacked hedges | "a credible offline staging baseline", "can therefore be a competitive design choice", "without attributing every model error to ambiguous labels" | 1, 2 |
| Process jargon repeated | "recorded evaluation look(s)" ×6; 0.773 restated ~10× | — |
| Inconsistent voice | "My study tests…" vs impersonal; "patients" vs "subjects"; S5 leaks "Audit the dashboard for this talk" | — |

### D. Structure and navigation

- Key result not findable in 10 s: first performance figure at ~y=6000 px; the human-agreement anchor (83% / scorer-to-scorer 0.802) that makes 0.77 meaningful sits in §1.1 paragraph 2.
- 16 `<details>` hide most of the page (11.4k px collapsed → 27.2k px open; S5 alone 9.5k px). The supplement outweighs the paper, while the confusion matrix and published-models table hide behind tabs.
- Same numbers restated across Abstract, §3.1, §3.3, S1, S5; the model reduction is told three times (§2.3 table, S2, log).
- Nav is a `<select>` with 22 options at every width; no persistent TOC despite an empty right margin; nav label "3.1 How well it scores" ≠ heading "Model: how well it scores".
- Mobile: §1.2 table clipped with no scroll cue; KPI tiles become empty 2×2 boxes; min–max labels wrap.
- Only 5 `:focus` rules in 9.8k lines of CSS.

### E. Code debt behind the inconsistency

- `globals.css` is 9,820 lines; 325 of 721 class selectors are never referenced in TSX; 26 `!important`; ~20 distinct breakpoints.
- `ZoomChapterStage.tsx` (1,143 lines), `ShowcaseScene.tsx`, `ShowcaseFlow.tsx`, `PopupModuleTray.tsx` are imported by nothing.
- `StagingResearchDashboard.tsx` is 2,724 lines with copy, data shaping and layout mixed together.

## Lay-visitor verdict

Quit at ~2:40 without being able to say whether it worked. Believed "scores ~77%, sounds mediocre" and saw U-Sleep 0.790 > 0.773 without being told how to read it.

## Rework plan

1. **Correctness pass (no visual change).** Reconcile every number in §A from the data JSON instead of hard-coded strings; lead with held-out 0.729 + κ; add patient-bootstrap CIs and paired differences; report both Coimbra scorers; fill or remove held-out per-stage cells; fix the spindle verdict and drop the summed "Other" bars; split the S5 timeline by dataset.
2. **Delete dead code.** Remove orphan modules and unreferenced CSS before restyling, so the new system starts clean.
3. **New visual system.** Light, print-first page: off-white paper, near-black text, one text serif at 17–18 px, 65–70ch measure, tabular figures. One accent; the five stage colours used only to encode stage, identically in every chart. No gradients, glows, pills, card borders, uppercase labels or progress bar. One small token set, three breakpoints.
4. **Lead with a figure.** Above the fold: a one-sentence plain-language summary, then Figure 1 = one median night's expert vs model hypnogram with errors marked, beside a per-patient dot plot of macro-F1 for CV / held-out / Coimbra with human scorer-to-scorer agreement and U-Sleep drawn as reference lines.
5. **Restructure.** Replace KPI tiles with the confusion matrix and per-stage F1 row (N1 visible as the stated weakness). Objectives become one framing paragraph. Promote the per-stage table and scorer-agreement table into §3.3. Un-collapse S1–S4 as normal numbered sections; move S5's log and code listing to a separate page or the repo. Sticky left TOC on desktop, short top TOC on mobile; §1.2 table becomes per-stage definition lists on narrow screens.
6. **Copy edit.** Rewrite against §C: cut the closer, mid-prose bold, "not X but Y" constructions, stacked hedges and process jargon; declarative headings; one voice; one term for patients; each number stated once and referenced after.
