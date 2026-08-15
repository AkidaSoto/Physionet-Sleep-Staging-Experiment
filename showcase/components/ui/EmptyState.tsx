import { ReactNode } from "react";
import { Text } from "./Text";

export function EmptyState({
  icon,
  title,
  message
}: {
  icon?: ReactNode;
  title?: string;
  message: string;
}) {
  return (
    <div className="empty-state">
      {icon ? <div>{icon}</div> : null}
      {title ? (
        <Text as="div" variant="bodyStrong">
          {title}
        </Text>
      ) : null}
      <Text as="div" variant="meta" tone="secondary">
        {message}
      </Text>
    </div>
  );
}
