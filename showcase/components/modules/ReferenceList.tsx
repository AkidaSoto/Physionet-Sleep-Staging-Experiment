"use client";

import { useState } from "react";
import { CenteredDialogModal } from "../ui/CenteredDialogModal";
import { Text } from "../ui/Text";

type ResearchLink = {
  id: string;
  label: string;
  title: string;
  summary: string;
  url: string;
};

export function ReferenceList({ links }: { links: ResearchLink[] }) {
  const [activeId, setActiveId] = useState<string | null>(null);
  const activeLink = links.find((link) => link.id === activeId) ?? null;

  return (
    <>
      <div className="reference-list reference-list-minimal">
        {links.map((link) => (
          <button
            key={link.id}
            type="button"
            className="reference-item"
            onClick={() => setActiveId(link.id)}
          >
            <span className="reference-tag">{link.label}</span>
          </button>
        ))}
      </div>

      {activeLink ? (
        <CenteredDialogModal
          title={activeLink.title}
          description={activeLink.label}
          onClose={() => setActiveId(null)}
          footer={
            <a href={activeLink.url} target="_blank" rel="noreferrer" className="button button-primary">
              Open source
            </a>
          }
        >
          <Text tone="secondary">{activeLink.summary}</Text>
        </CenteredDialogModal>
      ) : null}
    </>
  );
}
