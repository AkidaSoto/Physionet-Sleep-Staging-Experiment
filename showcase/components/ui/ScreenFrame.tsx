import { ReactNode } from "react";

export function ScreenFrame({ children }: { children: ReactNode }) {
  return (
    <div className="screen-frame">
      <div className="screen-frame-content">{children}</div>
    </div>
  );
}
