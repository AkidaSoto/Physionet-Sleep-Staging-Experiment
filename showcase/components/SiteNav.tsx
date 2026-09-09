"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

const items = [
  { id: "introduction", href: "/#introduction", label: "Introduction" },
  { id: "methods", href: "/#methods", label: "Methods" },
  { id: "results", href: "/#results", label: "Results" },
  { id: "discussion", href: "/#discussion", label: "Discussion" },
  { id: "references", href: "/#references", label: "References" }
];

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
          Sleep Staging · ML Study
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
            <option value="" disabled>Sections</option>
            <option value="top">Overview</option>
            {items.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
          </select>
        </label>
      </div>
      <span className="site-nav-progress" style={{ transform: `scaleX(${progress})` }} aria-hidden="true" />
    </nav>
  );
}
