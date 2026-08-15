import { ReferenceList } from "../../components/modules/ReferenceList";
import { StackSection } from "../../components/modules/StackSection";
import { ScreenFrame } from "../../components/ui/ScreenFrame";
import { Text } from "../../components/ui/Text";
import { getResearchLinks } from "../../lib/artifacts";

export default function ReferencesPage() {
  const links = getResearchLinks();

  return (
    <ScreenFrame>
      <main className="page">
        <section className="hero-block">
          <Text as="h1" variant="display">
            References
          </Text>
          <Text style={{ maxWidth: 760 }}>
            A clean end-summary source list. Inline ? references should still
            exist throughout the project, but this page gives one place to scan
            the anchors together.
          </Text>
        </section>

        <StackSection
          title="Source list"
          summary="Dataset, benchmark, and methodological anchors."
        >
          <ReferenceList links={links} />
        </StackSection>
      </main>
    </ScreenFrame>
  );
}
