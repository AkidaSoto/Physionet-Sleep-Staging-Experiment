"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

const items = [
  { id: "summary", href: "/#summary", label: "Abstract" },
  { id: "introduction", href: "/#introduction", label: "1 Introduction" },
  { id: "methods", href: "/#methods", label: "2 Methods" },
  { id: "results", href: "/#results", label: "3 Results" },
  { id: "comparison", href: "/#comparison", label: "4 Comparison" },
  { id: "discussion", href: "/#discussion", label: "5 Discussion" },
  { id: "supplementary", href: "/#supplementary", label: "Supplementary" },
  { id: "references", href: "/#references", label: "References" }
];

// Subsections listed, indented, in the jump menu.
const subsections: Record<string, Array<{ id: string; label: string }>> = {
  introduction: [
    { id: "staging", label: "1.1 Sleep staging" }, { id: "stages", label: "1.2 The physiology of each stage" }
  ],
  methods: [
    { id: "data", label: "2.1 Data and testing" }, { id: "features", label: "2.2 Features" },
    { id: "model", label: "2.3 Model" }
  ],
  results: [
    { id: "evidence", label: "3.1 What the model relies on" }, { id: "performance", label: "3.2 How well it scores" },
    { id: "robustness", label: "3.3 Does it hold up?" }
  ],
  supplementary: [
    { id: "history-details", label: "S1 How the work was run" }, { id: "supp-models", label: "S2 How the adopted model was reached" },
    { id: "night", label: "S3 How a night is organised" }, { id: "supp-cardio", label: "S4 Heart and breathing" },
    { id: "supp-simplify", label: "S5 Simpler models" }, { id: "supp-code", label: "S6 Python implementation" }
  ]
};

export function SiteNav() {
  const [activeId, setActiveId] = useState("top");
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const targets = ["top", ...items.map((item) => item.id)]
      .map((id) => document.getElementById(id))
      .filter((element): element is HTMLElement => Boolean(element));
    const updateScrollState = () => {
      const scrollable = document.documentElement.scrollHeight - window.innerHeight;
      setProgress(scrollable > 0 ? Math.min(window.scrollY / scrollable, 1) : 0);
      const readingLine = Math.min(180, window.innerHeight * 0.28);
      const visibleSection = document
        .elementFromPoint(window.innerWidth / 2, readingLine)
        ?.closest<HTMLElement>("section[id]");
      const active = visibleSection && targets.includes(visibleSection) ? visibleSection : targets[0];
      if (active?.id) setActiveId(active.id);
    };
    const syncToHash = () => {
      const id = window.location.hash.slice(1) || "top";
      if (targets.some((target) => target.id === id)) setActiveId(id);
    };
    const syncToAnchorClick = (event: MouseEvent) => {
      const anchor = (event.target as Element | null)?.closest<HTMLAnchorElement>("a[href*='#']");
      const id = anchor?.hash.slice(1);
      if (id && targets.some((target) => target.id === id)) setActiveId(id);
    };
    updateScrollState();
    window.addEventListener("scroll", updateScrollState, { passive: true });
    window.addEventListener("resize", updateScrollState);
    window.addEventListener("hashchange", syncToHash);
    document.addEventListener("click", syncToAnchorClick);
    return () => {
      window.removeEventListener("scroll", updateScrollState);
      window.removeEventListener("resize", updateScrollState);
      window.removeEventListener("hashchange", syncToHash);
      document.removeEventListener("click", syncToAnchorClick);
    };
  }, []);

  function handleMobileSection(value: string) {
    const target = document.getElementById(value);
    if (!target) return;
    let node: HTMLElement | null = target;
    while (node) {  // open a collapsed panel the target sits in
      if (node instanceof HTMLDetailsElement) node.open = true;
      node = node.parentElement;
    }
    setActiveId(value);
    const previousBehavior = document.documentElement.style.scrollBehavior;
    document.documentElement.style.scrollBehavior = "auto";
    target.scrollIntoView({ block: "start" });
    document.documentElement.style.scrollBehavior = previousBehavior;
  }

  return (
    <nav className="site-nav" aria-label="Primary">
      <div className="site-nav-inner">
        <Link href="/#top" className="site-nav-brand">
          Sleep staging
        </Link>
        <div className="site-nav-links">
          {items.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className="site-nav-link"
              aria-current={activeId === item.id ? "location" : undefined}
            >
              {item.label}
            </Link>
          ))}
        </div>
        <label className="site-nav-mobile">
          <span className="sr-only">Jump to section</span>
          <select value="" onChange={(event) => handleMobileSection(event.target.value)}>
            <option value="" disabled>Jump to…</option>
            <option value="top">Overview</option>
            {items.flatMap((item) => [
              <option key={item.id} value={item.id}>{item.label}</option>,
              ...(subsections[item.id] ?? []).map((sub) => <option key={sub.id} value={sub.id}>{"\u00a0\u00a0\u00a0\u00a0" + sub.label}</option>)
            ])}
          </select>
        </label>
      </div>
      <span className="site-nav-progress" style={{ transform: `scaleX(${progress})` }} aria-hidden="true" />
    </nav>
  );
}
