import { CSSProperties, ElementType, ReactNode } from "react";
import { typeScale } from "../../design/tokens";

type Variant = keyof typeof typeScale;
type Tone = "primary" | "secondary" | "accent";

const toneStyles: Record<Tone, CSSProperties> = {
  primary: { color: "var(--text)" },
  secondary: { color: "var(--muted)" },
  accent: { color: "var(--accent)" }
};

type Props<T extends ElementType> = {
  as?: T;
  variant?: Variant;
  tone?: Tone;
  children: ReactNode;
  className?: string;
  style?: CSSProperties;
} & Omit<React.ComponentPropsWithoutRef<T>, "as" | "children" | "className" | "style">;

export function Text<T extends ElementType = "p">({
  as,
  variant = "body",
  tone = "primary",
  children,
  className,
  style,
  ...props
}: Props<T>) {
  const Component = (as ?? "p") as ElementType;

  return (
    <Component
      {...props}
      className={className}
      style={{
        margin: 0,
        ...typeScale[variant],
        ...toneStyles[tone],
        ...style
      }}
    >
      {children}
    </Component>
  );
}
