"use client";

import { useState } from "react";
import type { PopupModule } from "../../content/story-types";
import type { ResearchLink, ShowcaseChapter } from "../../content/showcase-flow";
import { PopupModuleTray } from "./PopupModuleTray";
import { ReferenceList } from "./ReferenceList";
import { ShowcaseScene } from "./ShowcaseScene";
import { ZoomChapterStage } from "./ZoomChapterStage";
import { Text } from "../ui/Text";

type Props = {
  title: string;
  chapters: ShowcaseChapter[];
  showCover?: boolean;
};

function getTopicLabel(section: ShowcaseChapter["sections"][number]) {
  return section.eyebrow.replace(/^\d+\.\s*/, "");
}

function getModuleRows(modules: PopupModule[]) {
  return modules.length ? <PopupModuleTray modules={modules} /> : null;
}

export function ShowcaseFlow({ title, chapters, showCover = true }: Props) {
  const [activeSections, setActiveSections] = useState<Record<string, string | null>>(() =>
    Object.fromEntries(
      chapters.map((chapter) => [
        chapter.id,
        chapter.id === "apnea" || chapter.id === "staging" ? null : chapter.sections[0]?.id ?? ""
      ])
    )
  );
  const [activeCards, setActiveCards] = useState<Record<string, number | null>>(() =>
    Object.fromEntries(
      chapters.flatMap((chapter) => chapter.sections.map((section) => [section.id, null]))
    )
  );

  return (
    <div className="flow-shell" id="top">
      {showCover ? (
        <section className="flow-cover flow-section">
          <div className="flow-cover-copy">
            <Text as="h1" variant="display" className="flow-title">
              {title}
            </Text>
          </div>
        </section>
      ) : null}

      {chapters.map((chapter) => {
        const usesZoomStage = chapter.id === "apnea" || chapter.id === "staging";
        const storedActiveId = activeSections[chapter.id];
        const activeId =
          storedActiveId === undefined
            ? chapter.sections[0]?.id ?? ""
            : storedActiveId;
        const activeIndex = Math.max(
          chapter.sections.findIndex((section) => section.id === (activeId ?? "")),
          0
        );

        return (
          <section
            key={chapter.id}
            id={chapter.id}
            className={`flow-section flow-chapter ${usesZoomStage ? "flow-chapter-zoom" : ""} flow-${chapter.id}`}
          >
            <div className="chapter-intro">
              <Text as="h2" variant="title" className="chapter-title">
                {chapter.title}
              </Text>
            </div>

            <div className={`chapter-stage ${usesZoomStage ? "chapter-stage-zoom" : ""}`}>
              {usesZoomStage ? (
                <ZoomChapterStage
                  chapter={chapter}
                  activeSectionId={activeId}
                  onSectionSelect={(nextSectionId) =>
                    setActiveSections((current) => ({
                      ...current,
                      [chapter.id]: nextSectionId
                    }))
                  }
                />
              ) : (
                <div className="chapter-track-window">
                  <div
                    className="chapter-track"
                    style={{ transform: `translate3d(-${activeIndex * 100}%, 0, 0)` }}
                  >
                    {chapter.sections.map((section) => (
                      <article key={section.id} className="chapter-panel">
                        <div className="chapter-stage-grid">
                          <ShowcaseScene
                            section={section}
                            sections={chapter.sections}
                            activeSectionId={activeId}
                            onSectionSelect={(nextSectionId) =>
                              setActiveSections((current) => ({
                                ...current,
                                [chapter.id]: nextSectionId
                              }))
                            }
                            activeCardIndex={activeCards[section.id] ?? null}
                            onCardSelect={(nextCardIndex) =>
                              setActiveCards((current) => ({
                                ...current,
                                [section.id]:
                                  current[section.id] === nextCardIndex ? null : nextCardIndex
                              }))
                            }
                            getSectionLabel={getTopicLabel}
                          />

                          <div className="chapter-rail">
                            <div className="chapter-panel-copy">
                              <Text as="h3" variant="title" className="chapter-panel-title">
                                {section.title}
                              </Text>
                              <Text as="p" variant="bodyStrong" className="chapter-panel-takeaway">
                                {section.takeaway}
                              </Text>
                            </div>

                            <div className="chapter-depth">
                              {getModuleRows(section.modules)}
                              {chapter.references?.length ? <ReferenceList links={chapter.references as ResearchLink[]} /> : null}
                            </div>
                          </div>
                        </div>
                      </article>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </section>
        );
      })}
    </div>
  );
}
