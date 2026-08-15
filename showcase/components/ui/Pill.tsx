import { ReactNode } from "react";

export function Pill({ children }: { children: ReactNode }) {
  return <span className="pill">{children}</span>;
}
