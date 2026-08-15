import { ReactNode } from "react";
import { Surface } from "../ui/Surface";
import { Text } from "../ui/Text";

export function StackSection({
  title,
  summary,
  children
}: {
  title: string;
  summary?: string;
  children?: ReactNode;
}) {
  return (
    <Surface className="section-stack">
      <div className="section-copy">
        <Text as="h2" variant="title">
          {title}
        </Text>
        {summary ? <Text tone="secondary">{summary}</Text> : null}
      </div>
      {children}
    </Surface>
  );
}
