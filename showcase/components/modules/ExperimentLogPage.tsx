"use client";

import Link from "next/link";
import { ACCENT, modalityInk } from "../../design/palette";
import { historyPhases, historyWhere, technicalAppendixEntries, type HistoryStep } from "../../content/staging-content";
import { fmt } from "../../lib/results";
import { PaperDisclosure, SectionHeading, useOpenOnHash } from "../ui/ResearchPrimitives";

const scoreKinds = {
  ucddb: { label: "UCDDB cross-validation (25 patients)", color: modalityInk.emg },
  cinc: { label: "Boston cross-validation", color: modalityInk.eeg },
  holdout: { label: "Held-out Boston patients (three evaluation looks)", color: ACCENT }
};

const scoreKind = (step: HistoryStep) => (step.holdout ? "holdout" : step.n <= 14 ? "ucddb" : "cinc");

function HowItWasRun() {
  return (
    <div className="research-workflow">
      <p className="research-comparison-lede">
        I set the research questions and made the decisions; AI coding sessions did most of the implementation and ran the experiments, under the working rules below.
      </p>
      <ol className="research-glance-takeaways">
        <li>
          <b>One session per question.</b> A coordinating session keeps the goals, the decision log and this page. Each question (normalisation, heart-rate variability, poorly scored patients, signal quality, a second lab, a no-EEG model) has its own session with its own files, so several experiments can run in parallel without interfering.
        </li>
        <li>
          <b>Adoption criteria set in advance.</b> Each experiment lists its variants and the criterion for adoption (about {fmt("protocol.adoption_gain", 2, true)} macro-F1, better in 4 of 5 folds) before the results are seen. Every feature needs a physiological rationale, and a new feature is checked before any model uses it: it must separate its intended stage in the predicted direction, at least as well as the input it replaces.
        </li>
        <li>
          <b>Every experiment logged.</b> Each experiment is numbered when it is declared, then recorded below with its date, finding, decision and the folder that holds its evidence. The log is checked before each new run to avoid repeating work, and claims on this page link back to it. The held-out test has a separate log of scoring runs.
        </li>
      </ol>
    </div>
  );
}

function ExperimentHistory() {
  const steps = historyPhases.flatMap((phase) => phase.steps);
  const last = Math.max(...steps.map((step) => step.n));
  const scored = steps.filter((step) => step.score !== undefined);
  const lo = 0.6;
  const hi = 0.85;
  const xOf = (n: number) => `${((n - 1) / (last - 1)) * 100}%`;
  const shortMarker: Record<string, string> = {
    "Audit: bugs before models": "Audit & fixes",
    "A better dataset": "Boston (320)",
    "Context, then simplify": "Context",
    "Interpretability audit": "Interpretability",
    "Concept-additive sequence & conformal uncertainty": "Concept-BiGRU"
  };
  const markers = historyPhases.slice(1).map((phase) => ({
    n: phase.steps[0].n,
    label: shortMarker[phase.phase] ?? phase.phase
  }));
  const quietSteps = new Set([14, 28, 33, 36, 39, 75, 76, 78, 80]);
  const belowSteps = new Set([30, 37, 40, 81]);
  return (
    <div className="research-history">
      <div className="research-history-progress" role="img" aria-label={`Best macro-F1 over time, from ${scored[0].score} to ${scored[scored.length - 1].score}`}>
        {[0.6, 0.7, 0.8].map((tick) => (
          <i key={tick} className="tick" style={{ bottom: `${((tick - lo) / (hi - lo)) * 100}%` }}>
            {tick.toFixed(1)}
          </i>
        ))}
        {markers.map((marker, index) => (
          <em
            key={marker.n}
            className={`research-history-marker${index % 2 ? " is-low" : ""}${index === markers.length - 1 ? " is-last" : ""}`}
            style={{ left: `calc(${xOf(marker.n)} - 0.6%)` }}
          >
            <small>
              #{marker.n} {marker.label}
            </small>
          </em>
        ))}
        {scored.map((step) => (
          <span
            key={step.n}
            className={`${quietSteps.has(step.n) ? "is-quiet " : ""}${belowSteps.has(step.n) ? "is-below" : ""}`.trim() || undefined}
            style={{
              left: xOf(step.n),
              bottom: `${(((step.score as number) - lo) / (hi - lo)) * 100}%`,
              "--dot": scoreKinds[scoreKind(step)].color
            } as React.CSSProperties}
            title={`#${step.n} (${step.date}): ${(step.score as number).toFixed(3)} — ${step.question}`}
          >
            <b>{(step.score as number).toFixed(3)}</b>
            <small>#{step.n}</small>
          </span>
        ))}
      </div>
      <p className="research-chart-key research-history-key">
        {Object.entries(scoreKinds).map(([key, kind]) => (
          <em key={key}>
            <u style={{ background: kind.color }} />
            {kind.label}
          </em>
        ))}
      </p>
      <div className="research-history-phases">
        {historyPhases.map((phase) => (
          <section key={phase.phase} className="research-history-phase">
            <header>
              <h4>{phase.phase}</h4>
              <small>{phase.dataset}</small>
            </header>
            <ol>
              {phase.steps.map((step) => (
                <li key={step.n} id={`exp-${step.n}`}>
                  <span className="research-history-n">{step.n}</span>
                  <span className="research-history-date">{step.date}</span>
                  <div>
                    <strong>{step.question}</strong>
                    <p>
                      {step.finding} <em>{step.decision}</em>
                    </p>
                    {historyWhere[step.n] ? <code className="research-history-where">{historyWhere[step.n]}</code> : null}
                  </div>
                </li>
              ))}
            </ol>
          </section>
        ))}
      </div>
    </div>
  );
}

function TechnicalAppendix() {
  return (
    <div className="research-technical-appendix" id="technical-appendix">
      <div className="research-paper-disclosures research-code-disclosures">
        {technicalAppendixEntries.map((entry) => (
          <PaperDisclosure key={entry.id} title={entry.title} summary={entry.summary} className="research-code-disclosure">
            <div className="research-code-source">
              <span>Canonical source</span>
              <strong>{entry.source}</strong>
            </div>
            <pre id={`code-${entry.id}`}>
              <code>{entry.code}</code>
            </pre>
            <p>{entry.note}</p>
          </PaperDisclosure>
        ))}
      </div>
    </div>
  );
}

export function ExperimentLogPage() {
  useOpenOnHash();
  return (
    <main className="research-page research-history-page">
      <section className="research-section" id="history">
        <p>
          <Link href="/">← Automatic sleep staging from overnight polysomnography</Link>
        </p>
        <SectionHeading title="Experiment log and implementation" />
        <HowItWasRun />
        <ExperimentHistory />
      </section>
      <section className="research-section" id="supp-code">
        <SectionHeading title="Python implementation" />
        <TechnicalAppendix />
      </section>
    </main>
  );
}
