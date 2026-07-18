"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { useLang } from "@/components/providers/LanguageProvider";
import { useSim, type View } from "@/components/providers/SimulationProvider";
import { useTheme } from "@/components/providers/ThemeProvider";

interface Cmd {
  id: string;
  label: string;
  section: string;
  hint?: string;
  action: () => void;
}

export function CommandPalette({
  open,
  onClose,
  onOpenTour,
}: {
  open: boolean;
  onClose: () => void;
  onOpenTour: () => void;
}) {
  const { t } = useLang();
  const { toggle: toggleTheme } = useTheme();
  const { toggle: toggleLang } = useLang();
  const sim = useSim();
  const [query, setQuery] = useState("");
  const [sel, setSel] = useState(0);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const commands = useMemo<Cmd[]>(() => {
    const nav: Array<{ id: View; key: string }> = [
      { id: "live", key: "nav.live" },
      { id: "adversarial", key: "nav.adversarial" },
      { id: "governance", key: "nav.governance" },
      { id: "compare", key: "nav.compare" },
      { id: "viirs", key: "nav.viirs" },
      { id: "benchmark", key: "nav.benchmark" },
    ];
    const go = (v: View) => () => {
      sim.setView(v);
      onClose();
    };
    return [
      ...nav.map((n) => ({ id: `nav-${n.id}`, label: t(n.key), section: t("cmd.section.nav"), action: go(n.id) })),
      { id: "run", label: t("cmd.run"), section: t("cmd.section.actions"), action: () => { sim.run(); onClose(); } },
      { id: "theme", label: t("cmd.theme"), section: t("cmd.section.actions"), action: () => { toggleTheme(); } },
      { id: "lang", label: t("cmd.lang"), section: t("cmd.section.actions"), action: () => { toggleLang(); } },
      { id: "tour", label: t("cmd.tour"), section: t("cmd.section.actions"), action: () => { onClose(); onOpenTour(); } },
      { id: "p-paper", label: t("cmd.preset.paper"), section: t("cmd.section.presets"), action: () => { sim.setView("live"); sim.runWith({ grid_size: 100, n_uavs: 20, n_sectors: 25, n_timesteps: 500, tau: 0.72, method: "greedy_gomdp", attack_type: "none" }); onClose(); } },
      { id: "p-inj", label: t("cmd.preset.injection"), section: t("cmd.section.presets"), action: () => { sim.setView("adversarial"); sim.runWith({ method: "greedy_gomdp", attack_type: "injection", grid_size: 60, n_uavs: 18, n_timesteps: 400 }); onClose(); } },
      { id: "p-byz", label: t("cmd.preset.byzantine"), section: t("cmd.section.presets"), action: () => { sim.setView("governance"); sim.runWith({ method: "greedy_gomdp", attack_type: "byzantine", n_byzantine: 3, grid_size: 60, n_uavs: 18, n_timesteps: 400 }); onClose(); } },
      { id: "p-ppo", label: t("cmd.preset.ppo"), section: t("cmd.section.presets"), action: () => { sim.setView("live"); sim.runWith({ method: "ppo_gomdp", grid_size: 100, n_uavs: 20, n_sectors: 25, n_timesteps: 300, tau: 0.72 }); onClose(); } },
    ];
  }, [t, sim, toggleTheme, toggleLang, onClose, onOpenTour]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return commands;
    return commands.filter((c) => c.label.toLowerCase().includes(q) || c.section.toLowerCase().includes(q));
  }, [commands, query]);

  useEffect(() => {
    if (open) {
      setQuery("");
      setSel(0);
      setTimeout(() => inputRef.current?.focus(), 0);
    }
  }, [open]);

  useEffect(() => {
    setSel((s) => Math.min(s, Math.max(0, filtered.length - 1)));
  }, [filtered.length]);

  if (!open) return null;

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") onClose();
    else if (e.key === "ArrowDown") { e.preventDefault(); setSel((s) => Math.min(s + 1, filtered.length - 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setSel((s) => Math.max(s - 1, 0)); }
    else if (e.key === "Enter") { e.preventDefault(); filtered[sel]?.action(); }
  };

  let lastSection = "";
  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/50 p-4 pt-[12vh] backdrop-blur-sm"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label={t("cmd.placeholder")}
    >
      <div
        className="w-full max-w-lg overflow-hidden rounded-card border bg-[var(--surface)] shadow-2xl"
        onClick={(e) => e.stopPropagation()}
        onKeyDown={onKeyDown}
      >
        <input
          ref={inputRef}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={t("cmd.placeholder")}
          className="w-full border-b bg-transparent px-4 py-3 text-sm outline-none"
        />
        <ul className="max-h-[50vh] overflow-y-auto p-2">
          {filtered.length === 0 && <li className="px-3 py-4 text-center text-xs text-muted">{t("cmd.empty")}</li>}
          {filtered.map((c, i) => {
            const showSection = c.section !== lastSection;
            lastSection = c.section;
            return (
              <li key={c.id}>
                {showSection && (
                  <div className="px-2 pb-1 pt-2 text-[10px] font-semibold uppercase tracking-wide text-muted">{c.section}</div>
                )}
                <button
                  onMouseEnter={() => setSel(i)}
                  onClick={c.action}
                  className={`flex w-full items-center gap-2 rounded-md px-3 py-2 text-start text-sm ${
                    i === sel ? "bg-[var(--surface-2)] text-accent" : "hover:bg-[var(--surface-2)]/60"
                  }`}
                >
                  {c.label}
                </button>
              </li>
            );
          })}
        </ul>
        <div className="border-t px-4 py-2 text-[10px] text-muted">↑↓ navigate · ↵ select · esc close</div>
      </div>
    </div>
  );
}
