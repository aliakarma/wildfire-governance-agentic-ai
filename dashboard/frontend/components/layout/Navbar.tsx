"use client";
import { useLang } from "@/components/providers/LanguageProvider";
import { useSim, type View } from "@/components/providers/SimulationProvider";
import { useTheme } from "@/components/providers/ThemeProvider";

const NAV: Array<{ id: View; key: string }> = [
  { id: "live", key: "nav.live" },
  { id: "adversarial", key: "nav.adversarial" },
  { id: "governance", key: "nav.governance" },
  { id: "compare", key: "nav.compare" },
  { id: "viirs", key: "nav.viirs" },
  { id: "benchmark", key: "nav.benchmark" },
  { id: "ablation", key: "nav.ablation" },
  { id: "scalability", key: "nav.scalability" },
  { id: "learning", key: "nav.learning" },
  { id: "hitl", key: "nav.hitl" },
  { id: "cnn", key: "nav.cnn" },
  { id: "experiments", key: "nav.experiments" },
];

export function Navbar({
  onOpenPalette,
  onOpenHelp,
}: {
  onOpenPalette?: () => void;
  onOpenHelp?: () => void;
}) {
  const { t, toggle: toggleLang } = useLang();
  const { theme, toggle: toggleTheme } = useTheme();
  const { view, setView } = useSim();

  return (
    <header className="sticky top-0 z-20 border-b bg-[var(--surface)]/85 backdrop-blur">
      <div className="mx-auto flex max-w-[1600px] items-center gap-4 px-4 py-3">
        <div className="flex items-center gap-3">
          <div className="grid h-9 w-9 place-items-center rounded-lg bg-[var(--accent)] text-white">
            <span aria-hidden className="text-lg">🔥</span>
          </div>
          <div className="leading-tight">
            <div className="text-sm font-semibold">{t("app.title")}</div>
            <div className="hidden text-xs text-muted sm:block">{t("app.subtitle")}</div>
          </div>
        </div>

        <nav className="ms-auto flex items-center gap-1 text-sm" role="tablist">
          {NAV.map((n) => {
            const active = view === n.id;
            return (
              <button
                key={n.id}
                role="tab"
                aria-selected={active}
                onClick={() => setView(n.id)}
                className={`rounded-md px-3 py-1.5 font-medium transition ${
                  active ? "bg-[var(--surface-2)] text-accent" : "text-muted hover:text-text"
                }`}
              >
                {t(n.key)}
              </button>
            );
          })}
        </nav>

        <div className="flex items-center gap-2">
          <button
            onClick={onOpenPalette}
            className="hidden items-center gap-1 rounded-md border px-2.5 py-1.5 text-xs text-muted hover:bg-[var(--surface-2)] sm:inline-flex"
            aria-label="Open command palette"
          >
            <kbd className="font-sans">⌘K</kbd>
          </button>
          <button
            onClick={onOpenHelp}
            className="grid h-9 w-9 place-items-center rounded-md border hover:bg-[var(--surface-2)]"
            aria-label={t("help.open")}
          >
            ?
          </button>
          <button
            onClick={toggleLang}
            className="rounded-md border px-3 py-1.5 text-sm hover:bg-[var(--surface-2)]"
            aria-label="Switch language"
          >
            {t("lang.toggle")}
          </button>
          <button
            onClick={toggleTheme}
            className="grid h-9 w-9 place-items-center rounded-md border hover:bg-[var(--surface-2)]"
            aria-label={t("theme.toggle")}
          >
            {theme === "dark" ? "☀" : "☾"}
          </button>
        </div>
      </div>
    </header>
  );
}
