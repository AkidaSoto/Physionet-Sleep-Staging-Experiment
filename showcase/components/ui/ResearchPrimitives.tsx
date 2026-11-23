"use client";

import Link from "next/link";
import { useEffect, useId, useState, type KeyboardEvent, type ReactNode } from "react";

export function formatMetric(value: number) {
  return value.toFixed(3);
}

// 95% patient-bootstrap interval (scripts/build_metric_intervals.py), e.g. "0.715–0.742".
export function formatInterval(ci: number[], digits = 3) {
  return `${ci[0].toFixed(digits)}–${ci[1].toFixed(digits)}`;
}

export function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`;
}

export function featureValue(value: number) {
  const magnitude = Math.abs(value);
  if (magnitude >= 100) return value.toFixed(0);
  if (magnitude >= 10) return value.toFixed(1);
  if (magnitude >= 1) return value.toFixed(2);
  return value.toFixed(3);
}

export function formatClock(totalSeconds: number) {
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = Math.round(totalSeconds % 60);
  return `${hours}:${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`;
}

export function linePath(values: number[], width: number, height: number) {
  const sampled = values.filter((_, index) => index % Math.max(Math.floor(values.length / 700), 1) === 0);
  const finite = sampled.filter(Number.isFinite);
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  const span = max - min || 1;
  return sampled
    .map((value, index) => {
      const x = (index / Math.max(sampled.length - 1, 1)) * width;
      const y = height - ((value - min) / span) * height;
      return `${index === 0 ? "M" : "L"}${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

export function movePickerFocus(event: KeyboardEvent<HTMLDivElement>) {
  if (event.altKey || event.ctrlKey || event.metaKey) return;
  const buttons = Array.from(event.currentTarget.querySelectorAll<HTMLButtonElement>(":scope > button"));
  const current = buttons.indexOf(event.target as HTMLButtonElement);
  if (current < 0) return;
  const targets: Partial<Record<string, number>> = {
    ArrowLeft: (current + buttons.length - 1) % buttons.length,
    ArrowRight: (current + 1) % buttons.length,
    Home: 0,
    End: buttons.length - 1
  };
  const next = targets[event.key];
  if (next === undefined) return;
  event.preventDefault();
  buttons[next].focus();
  buttons[next].click();
}

export function Citation({ number }: { number: number }) {
  return (
    <a className="research-citation" href={`#reference-${number}`} aria-label={`Reference ${number}`}>
      [{number}]
    </a>
  );
}

export function SectionHeading({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <header className="research-section-heading">
      <div>
        <h2>{title}</h2>
        {children ? <p className="research-section-lede">{children}</p> : null}
      </div>
    </header>
  );
}

export function SubHeading({ number, title, refs }: { number: string; title: string; refs?: number[] }) {
  return (
    <div className="research-tool-heading">
      <h3>
        <span className="research-subnumber">{number}</span>
        {title}
      </h3>
      {refs ? <ExpRefs ids={refs} /> : null}
    </div>
  );
}

export function Block({
  id,
  title,
  takeaway,
  refs,
  children
}: {
  id: string;
  title: string;
  takeaway: ReactNode;
  refs?: number[];
  children: ReactNode;
}) {
  return (
    <div className="research-results-block research-block" id={id}>
      <div className="research-tool-heading">
        <h3>{title}</h3>
        {refs ? <ExpRefs ids={refs} /> : null}
      </div>
      <p className="research-takeaway">{takeaway}</p>
      {children}
    </div>
  );
}

export function PaperDisclosure({
  label,
  title,
  summary,
  children,
  open = false,
  className = ""
}: {
  label?: string;
  title: string;
  summary: string;
  children: ReactNode;
  open?: boolean;
  className?: string;
}) {
  return (
    <details className={`research-paper-disclosure ${className}`.trim()} open={open || undefined}>
      <summary>
        {label ? <span className="research-disclosure-label">{label}</span> : null}
        <strong>{title}</strong>
        <small>{summary}</small>
      </summary>
      <div className="research-paper-disclosure-body">{children}</div>
    </details>
  );
}

export function Evidence({ summary, children, id }: { summary: string; children: ReactNode; id?: string }) {
  return (
    <details className="research-evidence-disclosure" id={id}>
      <summary>
        <span>{summary}</span>
      </summary>
      <div className="research-evidence-disclosure-body">{children}</div>
    </details>
  );
}

export function Tabs({
  label,
  tabs
}: {
  label: string;
  tabs: Array<{ id: string; title: string; content: ReactNode }>;
}) {
  const tabListId = useId();
  const [active, setActive] = useState(tabs[0].id);
  const current = tabs.find((tab) => tab.id === active) ?? tabs[0];
  return (
    <div className="research-tabs">
      <div className="research-tabs-list" role="tablist" aria-label={label} onKeyDown={movePickerFocus}>
        {tabs.map((tab) => (
          <button
            key={tab.id}
            type="button"
            role="tab"
            id={`${tabListId}-tab-${tab.id}`}
            aria-controls={`${tabListId}-panel-${tab.id}`}
            aria-selected={tab.id === current.id}
            tabIndex={tab.id === current.id ? 0 : -1}
            onClick={() => setActive(tab.id)}
          >
            {tab.title}
          </button>
        ))}
      </div>
      {tabs.map((tab) => (
        <div
          key={tab.id}
          className="research-tabs-panel"
          role="tabpanel"
          id={`${tabListId}-panel-${tab.id}`}
          aria-labelledby={`${tabListId}-tab-${tab.id}`}
          hidden={tab.id !== current.id}
          tabIndex={0}
        >
          {tab.id === current.id ? tab.content : null}
        </div>
      ))}
    </div>
  );
}

export function Exp({ ids, range }: { ids?: number[]; range?: [number, number] }) {
  if (range) {
    return (
      <span className="research-exp-inline">
        <Link href={`/history#exp-${range[0]}`}>exp. #{range[0]}–{range[1]}</Link>
      </span>
    );
  }
  return (
    <span className="research-exp-inline">
      exp.{" "}
      {(ids ?? []).map((n, i) => (
        <Link key={n} href={`/history#exp-${n}`}>
          {i ? ", " : ""}#{n}
        </Link>
      ))}
    </span>
  );
}

export function ExpRefs({ ids }: { ids: number[] }) {
  return (
    <span className="research-exp-refs" aria-label="Experiments behind this result">
      Experiments {ids.map((n) => <Link key={n} href={`/history#exp-${n}`}>#{n}</Link>)}
    </span>
  );
}

export const sectionNames: Record<string, string> = {
  staging: "1.1 Sleep staging",
  stages: "1.2 The physiology of each stage",
  night: "S3 Sleep structure",
  data: "2.1 Data and testing",
  features: "2.2 Features",
  model: "2.3 Model",
  testing: "2.1 Data and testing",
  performance: "3.1 Performance",
  evidence: "3.2 Additive interpretability & conformal uncertainty",
  robustness: "3.3 External validation",
  "supp-patients": "S1 Patient variation",
  "supp-models": "S2 Model C & reduction archive",
  "supp-simplify": "S2 Simpler models",
  "supp-importance": "S3 Signal reliance archive",
  "supp-cardio": "S4 #75–#80 comparators & cardio ablations",
  "supp-history": "S5 Experiment log"
};

export function SecRef({ to }: { to: string }) {
  return (
    <a className="research-secref" href={`#${to}`}>
      §{sectionNames[to]}
    </a>
  );
}

// Open a closed <details> when a link points inside it (experiment history, code appendix, S2 anchors).
export function useOpenOnHash() {
  useEffect(() => {
    const open = () => {
      const id = window.location.hash.slice(1);
      if (!id) return;
      const target = document.getElementById(id);
      let node: HTMLElement | null = target;
      while (node) {
        if (node instanceof HTMLDetailsElement) node.open = true;
        node = node.parentElement;
      }
      target?.scrollIntoView({ block: "start", behavior: "instant" });
    };
    open();
    window.addEventListener("hashchange", open);
    return () => window.removeEventListener("hashchange", open);
  }, []);
}
