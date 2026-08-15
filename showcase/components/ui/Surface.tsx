import { HTMLAttributes, ReactNode } from "react";

type SurfaceProps = {
  children: ReactNode;
  elevated?: boolean;
} & HTMLAttributes<HTMLElement>;

export function Surface({
  children,
  className = "",
  elevated = false,
  ...props
}: SurfaceProps) {
  return (
    <section
      {...props}
      className={[
        "card",
        elevated ? "card-elevated" : "",
        className
      ]
        .filter(Boolean)
        .join(" ")}
    >
      {children}
    </section>
  );
}

export function InsetSurface({
  children,
  className = "",
  ...props
}: HTMLAttributes<HTMLElement> & { children: ReactNode }) {
  return (
    <section {...props} className={["inset-card", className].filter(Boolean).join(" ")}>
      {children}
    </section>
  );
}
