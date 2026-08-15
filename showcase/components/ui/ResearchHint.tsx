type ResearchHintProps = {
  title: string;
  summary: string;
  url: string;
};

export function ResearchHint({ title, summary, url }: ResearchHintProps) {
  return (
    <a className="research-hint" href={url} target="_blank" rel="noreferrer">
      <span className="q-badge">?</span>
      <span className="research-hint-copy">
        <strong>{title}</strong>
        <span className="muted">{summary}</span>
      </span>
    </a>
  );
}
