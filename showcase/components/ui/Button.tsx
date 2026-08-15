import { ButtonHTMLAttributes, ReactNode } from "react";
import { Text } from "./Text";

type Variant = "primary" | "ghost";

export function Button({
  children,
  className = "",
  variant = "primary",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  children: ReactNode;
  variant?: Variant;
}) {
  const variantClass =
    variant === "primary"
      ? "button button-primary"
      : "button button-ghost";

  return (
    <button {...props} className={[variantClass, className].filter(Boolean).join(" ")}>
      {typeof children === "string" ? (
        <Text as="span" variant="button">
          {children}
        </Text>
      ) : (
        children
      )}
    </button>
  );
}
