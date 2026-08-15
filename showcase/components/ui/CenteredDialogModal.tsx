"use client";

import { ReactNode } from "react";
import { Surface } from "./Surface";
import { Text } from "./Text";

type Props = {
  title: ReactNode;
  description?: ReactNode;
  children: ReactNode;
  footer?: ReactNode;
  onClose: () => void;
};

export function CenteredDialogModal({
  title,
  description,
  children,
  footer,
  onClose
}: Props) {
  return (
    <div className="modal-layer" role="dialog" aria-modal="true">
      <button
        type="button"
        aria-label="Close modal backdrop"
        className="modal-backdrop"
        onClick={onClose}
      />
      <div className="modal-shell">
        <Surface elevated className="modal-panel">
          <div className="modal-header">
            <div className="modal-header-copy">
              <Text as="h2" variant="sectionTitle">
                {title}
              </Text>
              {description ? (
                <Text as="p" variant="meta" tone="secondary">
                  {description}
                </Text>
              ) : null}
            </div>
            <button type="button" className="modal-close" onClick={onClose}>
              x
            </button>
          </div>

          <div className="modal-body">{children}</div>
          {footer ? <div className="modal-footer">{footer}</div> : null}
        </Surface>
      </div>
    </div>
  );
}
