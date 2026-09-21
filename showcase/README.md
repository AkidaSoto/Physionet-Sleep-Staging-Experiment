# Sleep Staging ML Case Study

This folder is the technical presentation layer for the sleep-staging project.

It should:

- read like a compact research paper rather than a general-purpose dashboard
- make subject-grouped validation and leakage control visible
- compare model hypotheses with real saved experiment results
- expose class, feature-family, fold, and subject-level failure modes
- retain clinical signal examples as grounding for the modeling decisions

It should not own the heavy computation.

Data flow:

`scripts / algorithms -> parquet artifacts -> staging-dashboard.json -> showcase`

Refresh the dashboard data after new experiments with:

```powershell
npm.cmd run data:staging
```

The apnea implementation and artifacts remain in the repository, but they are intentionally hidden from the primary staging case-study navigation.

## Hosting

The primary dashboard is fully static and can be hosted with either GitHub Pages or Vercel.

- GitHub Pages: the included workflow builds `showcase/` as a static Next.js export and publishes `showcase/out`.
- Vercel: import the repository, set the project root to `showcase`, and keep the default Next.js build settings.

For GitHub Pages, push the repository to GitHub and select **Settings → Pages → Source → GitHub Actions**. The workflow handles the repository subpath automatically.
