// Every result the dashboard shows comes from the results registry (scripts/build_results_registry.py).
// An unknown id throws, so a page that references a missing result fails to build instead of showing a guess.
import registry from "../content/generated/results-registry.json";

type Entry = { value: number; kind: string; label: string; digits?: number };
const entries = registry.entries as Record<string, Entry>;

function entry(id: string): Entry {
  const found = entries[id];
  if (!found) throw new Error(`results registry has no entry "${id}"; register it in scripts/build_results_registry.py`);
  return found;
}

/** The registered value. */
export function r(id: string): number {
  return entry(id).value;
}

/** The registered value as text: `digits` places (default: the entry's own, else 3); `sign` adds + or −. */
export function fmt(id: string, digits?: number, sign = false): string {
  const e = entry(id);
  const text = Math.abs(e.value).toFixed(digits ?? e.digits ?? 3);
  const negative = e.value < 0 && Number(text) !== 0;
  return `${negative ? "−" : sign ? "+" : ""}${text}`;
}

/** The registered proportion as a whole percentage, e.g. 0.826 -> "83%". */
export function pct(id: string): string {
  return `${Math.round(entry(id).value * 100)}%`;
}
