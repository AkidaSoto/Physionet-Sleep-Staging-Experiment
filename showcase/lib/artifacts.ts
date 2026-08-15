import overview from "../../artifacts/showcase/overview.json";
import researchLinks from "../../artifacts/showcase/research-links.json";
import taskSummary from "../../artifacts/showcase/task-summary.json";
export { getShowcaseSiteData } from "./showcase-site-data";

export function getOverview() {
  return overview;
}

export function getResearchLinks() {
  return researchLinks;
}

export function getTaskSummary() {
  return taskSummary;
}
