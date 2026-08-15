export type PopupModule = {
  id: string;
  label: string;
  title: string;
  description: string;
  whyItMatters: string;
  bullets: string[];
  link?: {
    href: string;
    label: string;
  };
};

export type StoryCard = {
  title: string;
  body: string;
  tags?: string[];
};

export type StorySectionData = {
  id: string;
  eyebrow: string;
  title: string;
  summary: string;
  takeaway: string;
  cards: StoryCard[];
  modules: PopupModule[];
  layout?: "two" | "three" | "four";
};
