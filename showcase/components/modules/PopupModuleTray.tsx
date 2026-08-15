"use client";

import { useState } from "react";
import type { PopupModule } from "../../content/story-types";
import { Button } from "../ui/Button";
import { CenteredDialogModal } from "../ui/CenteredDialogModal";
import { Text } from "../ui/Text";

export function PopupModuleTray({ modules }: { modules: PopupModule[] }) {
  const [activeId, setActiveId] = useState<string | null>(null);
  const activeModule = modules.find((module) => module.id === activeId) ?? null;

  if (!modules.length) {
    return null;
  }

  return (
    <>
      <div className="module-trigger-row">
        {modules.map((module) => (
          <Button
            key={module.id}
            type="button"
            variant="ghost"
            className="module-trigger"
            onClick={() => setActiveId(module.id)}
          >
            {module.label}
          </Button>
        ))}
      </div>

      {activeModule ? (
        <CenteredDialogModal
          title={activeModule.title}
          description={activeModule.description}
          onClose={() => setActiveId(null)}
          footer={
            activeModule.link ? (
              <a
                href={activeModule.link.href}
                target="_blank"
                rel="noreferrer"
                className="button button-primary"
              >
                {activeModule.link.label}
              </a>
            ) : null
          }
        >
          <div className="modal-copy">
            <Text variant="bodyStrong">Why it matters</Text>
            <Text tone="secondary">{activeModule.whyItMatters}</Text>
          </div>
          <ul className="modal-list">
            {activeModule.bullets.map((bullet) => (
              <li key={bullet}>
                <Text tone="secondary">{bullet}</Text>
              </li>
            ))}
          </ul>
        </CenteredDialogModal>
      ) : null}
    </>
  );
}
