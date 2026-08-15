export const colors = {
  background: "#0b1020",
  surface: "#121934",
  surfaceElevated: "#182246",
  surfaceHigh: "#22305f",
  textPrimary: "#edf2ff",
  textSecondary: "#a8b3cf",
  accent: "#7dd3fc",
  accentSecondary: "#a78bfa",
  border: "#2b3a67",
  scrim: "rgba(11, 16, 32, 0.78)"
} as const;

export const typeScale = {
  display: {
    fontSize: "clamp(2.2rem, 6vw, 4rem)",
    lineHeight: 1.05,
    fontWeight: 700
  },
  title: {
    fontSize: "1.8rem",
    lineHeight: 1.15,
    fontWeight: 700
  },
  sectionTitle: {
    fontSize: "1.2rem",
    lineHeight: 1.2,
    fontWeight: 700
  },
  body: {
    fontSize: "1rem",
    lineHeight: 1.6,
    fontWeight: 400
  },
  bodyStrong: {
    fontSize: "1rem",
    lineHeight: 1.6,
    fontWeight: 600
  },
  meta: {
    fontSize: "0.92rem",
    lineHeight: 1.5,
    fontWeight: 400
  },
  pill: {
    fontSize: "0.92rem",
    lineHeight: 1.2,
    fontWeight: 600
  },
  eyebrow: {
    fontSize: "0.82rem",
    lineHeight: 1.2,
    fontWeight: 700,
    letterSpacing: "0.08em",
    textTransform: "uppercase" as const
  },
  button: {
    fontSize: "0.95rem",
    lineHeight: 1.2,
    fontWeight: 700
  }
} as const;

export const radius = {
  panel: "18px",
  inset: "14px",
  pill: "999px"
} as const;

export const screens = {
  contentMaxWidth: "1100px"
} as const;
