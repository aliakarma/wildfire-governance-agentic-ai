import type { Metadata } from "next";
import "./globals.css";
import { ThemeProvider, THEME_SCRIPT } from "@/components/providers/ThemeProvider";
import { LanguageProvider } from "@/components/providers/LanguageProvider";
import { SimulationProvider } from "@/components/providers/SimulationProvider";
import { BenchmarkProvider } from "@/components/providers/BenchmarkProvider";

export const metadata: Metadata = {
  title: "GOMDP Dashboard · Wildfire Governance",
  description:
    "Interactive dashboard for governance-constrained agentic AI in safety-critical wildfire monitoring.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" dir="ltr" data-theme="dark" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body>
        <ThemeProvider>
          <LanguageProvider>
            <SimulationProvider>
              <BenchmarkProvider>{children}</BenchmarkProvider>
            </SimulationProvider>
          </LanguageProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
