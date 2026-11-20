"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";

const items = [
  { id: "summary", href: "/#summary", label: "Abstract" },
  { id: "introduction", href: "/#introduction", label: "1 Introduction" },
  { id: "methods", href: "/#methods", label: "2 Methods" },
  { id: "results", href: "/#results", label: "3 Results" },
  { id: "discussion", href: "/#discussion", label: "4 Discussion and conclusion" },
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
    { id: "performance", label: "3.1 Performance" }, { id: "evidence", label: "3.2 Feature evidence and model reliance" },
    { id: "robustness", label: "3.3 Held-out and external testing" }
  ],
  supplementary: [
    { id: "supp-patients", label: "S1 Development patient variation" },
    { id: "supp-models", label: "S2 Model development and size" },
    { id: "supp-importance", label: "S3 Signal reliance and sleep structure" },
    { id: "supp-cardio", label: "S4 Alternative sensor models" },
    { id: "supp-history", label: "S5 Experiment log and code" }
  ]
};

export function SiteNav() {
  const pathname = usePathname();
  const router = useRouter();
  const [activeId, setActiveId] = useState("top");

  useEffect(() => {
    const targets = ["top", ...items.map((item) => item.id)]
      .map((id) => document.getElementById(id))
      .filter((element): element is HTMLElement => Boolean(element));
    const updateScrollState = () => {
      const readingLine = Math.min(180, window.innerHeight * 0.28);
      const visibleSection = document
        .elementFromPoint(window.innerWidth / 2, readingLine)
        ?.closest<HTMLElement>("section[id]");
      const active = visibleSection && targets.includes(visibleSection) ? visibleSection : targets[0];
      setActiveId(active?.id ?? "top");
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
  }, [pathname]);

  function handleMobileSection(value: string) {
    const target = document.getElementById(value);
    if (!target) {
      router.push(`/#${value}`);
      return;
    }
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
            <div key={item.href} className="site-nav-group">
              <Link
                href={item.href}
                className="site-nav-link"
                aria-current={activeId === item.id ? "location" : undefined}
              >
                {item.label}
              </Link>
              {(subsections[item.id] ?? []).map((sub) => (
                <Link key={sub.id} href={`/#${sub.id}`} className="site-nav-sublink">{sub.label}</Link>
              ))}
            </div>
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
    </nav>
  );
}
