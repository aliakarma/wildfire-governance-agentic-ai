"use client";

import { useState, useRef, useEffect } from "react";
import { useLang } from "@/components/providers/LanguageProvider";
import { useSim, type View } from "@/components/providers/SimulationProvider";
import { useTheme } from "@/components/providers/ThemeProvider";

const MAIN_NAV: Array<{ id: View; key: string }> = [
  { id: "live", key: "nav.live" },
  { id: "adversarial", key: "nav.adversarial" },
  { id: "governance", key: "nav.governance" },
  { id: "compare", key: "nav.compare" },
  { id: "viirs", key: "nav.viirs" },
  { id: "benchmark", key: "nav.benchmark" },
];

const EXPERIMENT_NAV: Array<{ id: View; key: string }> = [
  { id: "ablation", key: "nav.ablation" },
  { id: "scalability", key: "nav.scalability" },
  { id: "learning", key: "nav.learning" },
  { id: "hitl", key: "nav.hitl" },
  { id: "statistics", key: "nav.statistics" },
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
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const isExpActive = EXPERIMENT_NAV.some((item) => item.id === view);
  const currentExp = EXPERIMENT_NAV.find((item) => item.id === view);

  // Close dropdown on outside click or Escape key
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

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
          {MAIN_NAV.map((n) => {
            const active = view === n.id;
            return (
              <button
                key={n.id}
                role="tab"
                aria-selected={active}
                onClick={() => {
                  setView(n.id);
                  setIsOpen(false);
                }}
                className={`rounded-md px-3 py-1.5 font-medium transition ${
                  active ? "bg-[var(--surface-2)] text-accent" : "text-muted hover:text-text"
                }`}
              >
                {t(n.key)}
              </button>
            );
          })}

          {/* Grouped Experiments Dropdown */}
          <div className="relative" ref={dropdownRef}>
            <button
              type="button"
              aria-expanded={isOpen}
              aria-haspopup="true"
              onClick={() => setIsOpen((prev) => !prev)}
              className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 font-medium transition ${
                isExpActive
                  ? "bg-[var(--surface-2)] text-accent shadow-sm"
                  : "text-muted hover:text-text hover:bg-[var(--surface-2)]/50"
              }`}
            >
              <span>
                {isExpActive && currentExp ? `${t(currentExp.key)}` : t("nav.paper_experiments")}
              </span>
              <span className={`text-[10px] transition-transform duration-200 ${isOpen ? "rotate-180" : ""}`}>
                ▼
              </span>
            </button>

            {isOpen && (
              <div className="absolute right-0 top-full mt-1.5 z-30 min-w-[190px] rounded-lg border border-[var(--border)] bg-[var(--surface)] p-1.5 shadow-xl backdrop-blur">
                <div className="px-2 py-1 text-[11px] font-semibold uppercase tracking-wider text-muted border-b border-muted/20 mb-1">
                  {t("nav.paper_experiments")}
                </div>
                {EXPERIMENT_NAV.map((n) => {
                  const active = view === n.id;
                  return (
                    <button
                      key={n.id}
                      onClick={() => {
                        setView(n.id);
                        setIsOpen(false);
                      }}
                      className={`flex w-full items-center justify-between rounded-md px-2.5 py-1.5 text-left text-sm font-medium transition ${
                        active
                          ? "bg-[var(--surface-2)] text-accent font-semibold"
                          : "text-muted hover:bg-[var(--surface-2)]/60 hover:text-text"
                      }`}
                    >
                      <span>{t(n.key)}</span>
                      {active && <span className="text-xs font-bold text-accent">✓</span>}
                    </button>
                  );
                })}
              </div>
            )}
          </div>
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
