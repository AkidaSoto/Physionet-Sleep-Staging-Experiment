import "./globals.css";
import type { Metadata } from "next";
import { SiteNav } from "../components/SiteNav";

export const metadata: Metadata = {
  title: "Interpretable Sleep Staging from PSG",
  description: "A technical case study in subject-independent sleep-stage modeling."
};

export default function RootLayout({
  children
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <SiteNav />
        {children}
      </body>
    </html>
  );
}
