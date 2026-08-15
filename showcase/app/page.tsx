import { ShowcaseFlow } from "../components/modules/ShowcaseFlow";
import { ScreenFrame } from "../components/ui/ScreenFrame";
import { getShowcaseChapters, showcaseTitle } from "../content/showcase-flow";
import { getResearchLinks } from "../lib/artifacts";

export default function HomePage() {
  const researchLinks = getResearchLinks();
  const chapters = getShowcaseChapters(researchLinks);

  return (
    <ScreenFrame>
      <main className="page page-flow">
        <ShowcaseFlow title={showcaseTitle} chapters={chapters} />
      </main>
    </ScreenFrame>
  );
}
