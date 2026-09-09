import type { NextConfig } from "next";

const isGitHubPages = process.env.GITHUB_PAGES === "true";
const pagesBasePath = isGitHubPages ? process.env.PAGES_BASE_PATH ?? "" : "";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  output: isGitHubPages ? "export" : undefined,
  trailingSlash: isGitHubPages,
  basePath: pagesBasePath,
  assetPrefix: pagesBasePath
};

export default nextConfig;
