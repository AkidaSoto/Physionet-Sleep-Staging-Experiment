import { ShowcaseFlow } from "../../components/modules/ShowcaseFlow";
import { ScreenFrame } from "../../components/ui/ScreenFrame";
import { getShowcaseChapter, showcaseTitle } from "../../content/showcase-flow";
import { getResearchLinks } from "../../lib/artifacts";

export default function ApneaPage() {
  const chapter = getShowcaseChapter("apnea", getResearchLinks());

  if (!chapter) {
    return null;
  }

  return (
    <ScreenFrame>
      <main className="page page-flow">
        <ShowcaseFlow title={showcaseTitle} chapters={[chapter]} showCover={false} />
      </main>
    </ScreenFrame>
  );
}
