"use client";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useTheme } from "@/components/providers/ThemeProvider";
import { API_BASE, DEFAULT_PARAMS } from "@/lib/api";
import type { Frame, LedgerEntry, SimParams } from "@/lib/types";
import { useEpisodeStream, type StreamStatus } from "@/lib/useEpisodeStream";

// Calmer default replay so the search → verify → encircle hand-off and the
// slow fire front are easy to follow. Users can still speed up via the controls.
const BASE_FPS = 9;

interface LayerState {
  fire: boolean;
  uavs: boolean;
  sectors: boolean;
  comms: boolean;
}

export type View = "live" | "adversarial" | "governance" | "benchmark" | "compare" | "viirs" | "experiments" | "ablation" | "scalability" | "learning" | "hitl" | "statistics";

interface SimCtx {
  view: View;
  setView: (v: View) => void;
  params: SimParams;
  setParam: (patch: Partial<SimParams>) => void;
  status: StreamStatus;
  frames: Frame[];
  currentFrame: Frame | null;
  events: LedgerEntry[];
  summary: ReturnType<typeof useEpisodeStream>["summary"];
  meta: ReturnType<typeof useEpisodeStream>["meta"];
  error: string | null;
  // playback
  index: number;
  playing: boolean;
  speed: number;
  seek: (i: number) => void;
  togglePlay: () => void;
  setSpeed: (s: number) => void;
  // layers
  layers: LayerState;
  toggleLayer: (k: keyof LayerState) => void;
  // selection (governance inspector)
  selected: LedgerEntry | null;
  setSelected: (e: LedgerEntry | null) => void;
  // actions
  run: () => void;
  runWith: (override: Partial<SimParams>) => void;
  stop: () => void;
  exporting: boolean;
  exportGif: () => Promise<void>;
  share: () => void;
  shared: boolean;
}

const Ctx = createContext<SimCtx | null>(null);

export function SimulationProvider({ children }: { children: React.ReactNode }) {
  const { theme } = useTheme();
  const [view, setView] = useState<View>("live");
  const [params, setParams] = useState<SimParams>(DEFAULT_PARAMS);
  const [shared, setShared] = useState(false);

  // Hydrate params + view from the URL query string on first mount (permalinks).
  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    if ([...q.keys()].length === 0) return;
    const patch: Partial<SimParams> = {};
    (Object.keys(DEFAULT_PARAMS) as Array<keyof SimParams>).forEach((k) => {
      const v = q.get(k);
      if (v == null) return;
      const def = DEFAULT_PARAMS[k];
      (patch as Record<string, unknown>)[k] = typeof def === "number" ? Number(v) : v;
    });
    setParams((p) => ({ ...p, ...patch }));
    const v = q.get("view");
    if (v && ["live", "adversarial", "governance", "benchmark", "compare", "viirs"].includes(v)) {
      setView(v as View);
    }
  }, []);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [layers, setLayers] = useState<LayerState>({ fire: true, uavs: true, sectors: false, comms: true });
  const [selected, setSelected] = useState<LedgerEntry | null>(null);
  const [exporting, setExporting] = useState(false);

  const { frames, status, summary, meta, error, start, stop } = useEpisodeStream();
  const rafRef = useRef<number | null>(null);
  const lastTsRef = useRef(0);
  const accRef = useRef(0);

  const setParam = useCallback((patch: Partial<SimParams>) => {
    setParams((p) => ({ ...p, ...patch }));
  }, []);

  const run = useCallback(() => {
    setIndex(0);
    setPlaying(true);
    setSelected(null);
    start(params);
  }, [params, start]);

  const runWith = useCallback(
    (override: Partial<SimParams>) => {
      const p = { ...params, ...override };
      setParams(p);
      setIndex(0);
      setPlaying(true);
      setSelected(null);
      start(p);
    },
    [params, start],
  );

  // Keep the latest frame count in a ref so the playback loop can read it
  // without re-subscribing (and thus restarting the rAF) on every frame batch.
  const framesLenRef = useRef(0);
  useEffect(() => {
    framesLenRef.current = frames.length;
  }, [frames.length]);
  const statusRef = useRef(status);
  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  // Single playback loop that advances the view at a steady BASE_FPS × speed —
  // used both while the episode is still streaming AND on replay afterwards. It
  // never snaps to the newest frame (that made the UAVs look like they were
  // teleporting/bouncing during a fast live stream). Instead it plays smoothly
  // from t=0; if it reaches the live edge while frames are still arriving it
  // waits there for more, and only stops once the stream is finished.
  useEffect(() => {
    if (!playing) {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      return;
    }
    lastTsRef.current = 0;
    accRef.current = 0;
    const tick = (ts: number) => {
      if (!lastTsRef.current) lastTsRef.current = ts;
      const dt = (ts - lastTsRef.current) / 1000;
      lastTsRef.current = ts;
      accRef.current += dt * speed * BASE_FPS;
      if (accRef.current >= 1) {
        const adv = Math.floor(accRef.current);
        accRef.current -= adv;
        setIndex((i) => {
          const liveEdge = framesLenRef.current - 1;
          const next = i + adv;
          if (next >= liveEdge) {
            // At the newest available frame. If the stream is done, stop here;
            // otherwise hold at the edge and wait for more frames to arrive.
            if (statusRef.current !== "running" && statusRef.current !== "connecting") {
              setPlaying(false);
            }
            return Math.max(0, liveEdge);
          }
          return next;
        });
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [playing, speed]);

  const currentFrame = frames[Math.min(index, frames.length - 1)] ?? null;

  const events = useMemo<LedgerEntry[]>(
    () => frames.filter((f) => f.event).map((f) => ({ t: f.t, ...(f.event as object) }) as LedgerEntry),
    [frames],
  );

  const exportGif = useCallback(async () => {
    setExporting(true);
    try {
      const res = await fetch(`${API_BASE}/api/export/gif`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ params, fps: 12, theme }),
      });
      if (!res.ok) throw new Error("export failed");
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `episode_${params.method}_seed${params.seed}.gif`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      /* ignore */
    } finally {
      setExporting(false);
    }
  }, [params, theme]);

  const share = useCallback(() => {
    const q = new URLSearchParams();
    (Object.keys(params) as Array<keyof SimParams>).forEach((k) => q.set(k, String(params[k])));
    q.set("view", view);
    const url = `${window.location.origin}${window.location.pathname}?${q.toString()}`;
    window.history.replaceState(null, "", url);
    try {
      navigator.clipboard?.writeText(url);
    } catch {
      /* ignore */
    }
    setShared(true);
    window.setTimeout(() => setShared(false), 1800);
  }, [params, view]);

  const value: SimCtx = {
    view,
    setView,
    params,
    setParam,
    status,
    frames,
    currentFrame,
    events,
    summary,
    meta,
    error,
    index,
    playing,
    speed,
    seek: (i) => {
      setPlaying(false);
      setIndex(i);
    },
    togglePlay: () => setPlaying((p) => !p),
    setSpeed,
    layers,
    toggleLayer: (k) => setLayers((l) => ({ ...l, [k]: !l[k] })),
    selected,
    setSelected,
    run,
    runWith,
    stop,
    exporting,
    exportGif,
    share,
    shared,
  };

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useSim(): SimCtx {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useSim must be used within SimulationProvider");
  return ctx;
}
