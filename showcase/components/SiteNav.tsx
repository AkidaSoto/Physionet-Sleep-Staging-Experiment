"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";

const paperItems = [
  { id: "summary", href: "/#summary", label: "Abstract" },
  { id: "lead-overview", href: "/#lead-overview", label: "Key figures" },
  { id: "introduction", href: "/#introduction", label: "1 Introduction" },
  { id: "methods", href: "/#methods", label: "2 Methods" },
  { id: "results", href: "/#results", label: "3 Results" },
  { id: "discussion", href: "/#discussion", label: "4 Discussion" },
  { id: "supplementary", href: "/#supplementary", label: "Supplementary" },
  { id: "references", href: "/#references", label: "References" }
];

const historyItems = [
  { id: "history", href: "/history#history", label: "Experiment log" },
  { id: "supp-code", href: "/history#supp-code", label: "Python code" }
];

// Subsections revealed in the rail only for the active parent section.
const subsections: Record<string, Array<{ id: string; label: string }>> = {
  introduction: [
    { id: "staging", label: "1.1 Sleep staging" },
    { id: "stages", label: "1.2 Stage physiology" }
  ],
  methods: [
    { id: "data", label: "2.1 Data & testing" },
    { id: "features", label: "2.2 Features" },
    { id: "model", label: "2.3 Model architecture (A–D)" }
  ],
  results: [
    { id: "performance", label: "3.1 Performance" },
    { id: "evidence", label: "3.2 Signal drivers & uncertainty" },
    { id: "robustness", label: "3.3 External validation" }
  ],
  supplementary: [
    { id: "supp-patients", label: "S1 Patient variation" },
    { id: "supp-models", label: "S2 Model C archive" },
    { id: "supp-importance", label: "S3 Signal reliance" },
    { id: "supp-cardio", label: "S4 Neural baselines & heart/breathing" },
    { id: "supp-history", label: "S5 Experiment log" }
  ]
};

function parentOf(id: string) {
  for (const [parentId, subs] of Object.entries(subsections)) {
    if (subs.some((sub) => sub.id === id)) return parentId;
  }
  return id;
}

export function SiteNav() {
  const pathname = usePathname();
  const router = useRouter();
  const isHistory = pathname === "/history";
  const items = isHistory ? historyItems : paperItems;
  const [activeId, setActiveId] = useState(isHistory ? "history" : "summary");
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    document.documentElement.dataset.navCollapsed = collapsed ? "true" : "false";
    return () => {
      delete document.documentElement.dataset.navCollapsed;
    };
  }, [collapsed]);

  useEffect(() => {
    const allIds = items.flatMap((item) => [item.id, ...(subsections[item.id] ?? []).map((sub) => sub.id)]);
    let frame = 0;
    let lockedUntil = 0;

    const updateScrollState = () => {
      frame = 0;
      if (Date.now() < lockedUntil) return;
      const refEl = document.getElementById("references");
      if (
        window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 6 &&
        (!refEl || refEl.getBoundingClientRect().top <= window.innerHeight * 0.55)
      ) {
        const lastItem = items[items.length - 1];
        if (lastItem) setActiveId(lastItem.id);
        return;
      }
      const readingLine = Math.min(140, window.innerHeight * 0.25);
      let current = items[0]?.id ?? "summary";
      for (const id of allIds) {
        const el = document.getElementById(id);
        if (!el) continue;
        // Skip closed <details> siblings unless it's the only element or currently open
        if (el instanceof HTMLDetailsElement && !el.open && window.location.hash !== `#${id}`) {
          continue;
        }
        if (el.getBoundingClientRect().top <= readingLine) {
          current = id;
        }
      }
      setActiveId(current);
    };

    const scheduleUpdate = () => {
      if (!frame) frame = window.requestAnimationFrame(updateScrollState);
    };

    const syncToHash = () => {
      const id = window.location.hash.slice(1);
      if (id && allIds.includes(id)) {
        lockedUntil = Date.now() + 500;
        setActiveId(id);
      }
    };

    const syncToAnchorClick = (event: MouseEvent) => {
      const anchor = (event.target as Element | null)?.closest<HTMLAnchorElement>("a[href*='#']");
      const id = anchor?.hash.slice(1);
      if (id && allIds.includes(id)) {
        lockedUntil = Date.now() + 500;
        setActiveId(id);
      }
    };

    updateScrollState();
    window.addEventListener("scroll", scheduleUpdate, { passive: true });
    window.addEventListener("resize", scheduleUpdate);
    window.addEventListener("hashchange", syncToHash);
    document.addEventListener("click", syncToAnchorClick);
    return () => {
      if (frame) window.cancelAnimationFrame(frame);
      window.removeEventListener("scroll", scheduleUpdate);
      window.removeEventListener("resize", scheduleUpdate);
      window.removeEventListener("hashchange", syncToHash);
      document.removeEventListener("click", syncToAnchorClick);
    };
  }, [pathname, isHistory]);

  function handleMobileSection(value: string) {
    const target = document.getElementById(value);
    if (!target) {
      router.push(isHistory ? `/history#${value}` : `/#${value}`);
      return;
    }
    let node: HTMLElement | null = target;
    while (node) {
      if (node instanceof HTMLDetailsElement) node.open = true;
      node = node.parentElement;
    }
    setActiveId(value);
    const previousBehavior = document.documentElement.style.scrollBehavior;
    document.documentElement.style.scrollBehavior = "auto";
    target.scrollIntoView({ block: "start" });
    document.documentElement.style.scrollBehavior = previousBehavior;
  }

  function handleSubClick(id: string) {
    const target = document.getElementById(id);
    let node: HTMLElement | null = target;
    while (node) {
      if (node instanceof HTMLDetailsElement) node.open = true;
      node = node.parentElement;
    }
    setActiveId(id);
  }

  const activeParent = parentOf(activeId);

  return (
    <nav className={`site-nav${collapsed ? " is-collapsed" : ""}`} aria-label="Contents">
      <div className="site-nav-inner">
        <Link href="/#top" className="site-nav-brand">
          Sleep staging
        </Link>
        <div className="site-nav-rail-head">
          <Link href={isHistory ? "/" : "/#top"} className="site-nav-rail-title">
            {isHistory ? "← Paper" : "Contents"}
          </Link>
          <button
            type="button"
            className="site-nav-collapse-btn"
            aria-expanded={!collapsed}
            aria-label={collapsed ? "Show navigation sidebar" : "Hide navigation sidebar"}
            onClick={() => setCollapsed((prev) => !prev)}
          >
            {collapsed ? "Contents ▸" : "Hide ◂"}
          </button>
        </div>
        <div className="site-nav-links" hidden={collapsed}>
          {items.map((item) => {
            const isActiveGroup = activeParent === item.id;
            const subs = isHistory ? [] : subsections[item.id] ?? [];
            return (
              <div key={item.href} className={`site-nav-group${isActiveGroup ? " is-open" : ""}`}>
                <Link
                  href={item.href}
                  className="site-nav-link"
                  aria-current={activeId === item.id || isActiveGroup ? "location" : undefined}
                >
                  {item.label}
                </Link>
                {isActiveGroup && subs.length > 0 ? (
                  <div className="site-nav-sublist">
                    {subs.map((sub) => (
                      <Link
                        key={sub.id}
                        href={`/#${sub.id}`}
                        className="site-nav-sublink"
                        aria-current={activeId === sub.id ? "location" : undefined}
                        onClick={() => handleSubClick(sub.id)}
                      >
                        {sub.label}
                      </Link>
                    ))}
                  </div>
                ) : null}
              </div>
            );
          })}
        </div>
        <label className="site-nav-mobile">
          <span className="site-nav-mobile-label">Jump to</span>
          <select value={activeId} onChange={(event) => handleMobileSection(event.target.value)}>
            <option value="" disabled>Select section…</option>
            {!isHistory ? <option value="top">Overview</option> : null}
            {items.flatMap((item) => [
              <option key={item.id} value={item.id}>{item.label}</option>,
              ...(isHistory ? [] : subsections[item.id] ?? []).map((sub) => (
                <option key={sub.id} value={sub.id}>{"\u00a0\u00a0\u00a0\u00a0" + sub.label}</option>
              ))
            ])}
          </select>
        </label>
      </div>
    </nav>
  );
}
