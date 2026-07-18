"use client";
import { createContext, useCallback, useContext, useState } from "react";
import { fetchPaperResults, runBenchmark } from "@/lib/api";
import type { BenchmarkResponse, PaperResponse } from "@/lib/types";

export interface BenchConfig {
  n_seeds: number;
  n_uavs: number;
  grid_size: number;
  n_timesteps: number;
  tau: number;
}

export type BenchSource = "live" | "paper";

interface BenchCtx {
  methods: string[];
  toggleMethod: (id: string) => void;
  config: BenchConfig;
  setConfig: (patch: Partial<BenchConfig>) => void;
  source: BenchSource;
  setSource: (s: BenchSource) => void;
  live: BenchmarkResponse | null;
  paper: PaperResponse | null;
  loading: boolean;
  error: string | null;
  runLive: () => Promise<void>;
  loadPaper: () => Promise<void>;
}

const DEFAULT_METHODS = ["greedy_gomdp", "ppo_cmdp", "adaptive_ai", "static"];
const DEFAULT_CONFIG: BenchConfig = { n_seeds: 3, n_uavs: 16, grid_size: 50, n_timesteps: 250, tau: 0.72 };

const Ctx = createContext<BenchCtx | null>(null);

export function BenchmarkProvider({ children }: { children: React.ReactNode }) {
  const [methods, setMethods] = useState<string[]>(DEFAULT_METHODS);
  const [config, setConfigState] = useState<BenchConfig>(DEFAULT_CONFIG);
  const [source, setSource] = useState<BenchSource>("live");
  const [live, setLive] = useState<BenchmarkResponse | null>(null);
  const [paper, setPaper] = useState<PaperResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const toggleMethod = useCallback((id: string) => {
    setMethods((m) => (m.includes(id) ? m.filter((x) => x !== id) : [...m, id]));
  }, []);

  const setConfig = useCallback((patch: Partial<BenchConfig>) => {
    setConfigState((c) => ({ ...c, ...patch }));
  }, []);

  const runLive = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await runBenchmark({ methods, ...config });
      setLive(res);
    } catch (e) {
      setError((e as Error).message || "benchmark failed");
    } finally {
      setLoading(false);
    }
  }, [methods, config]);

  const loadPaper = useCallback(async () => {
    if (paper) return;
    try {
      const res = await fetchPaperResults("table1_rl_comparison");
      setPaper(res);
    } catch {
      /* ignore */
    }
  }, [paper]);

  const value: BenchCtx = {
    methods,
    toggleMethod,
    config,
    setConfig,
    source,
    setSource,
    live,
    paper,
    loading,
    error,
    runLive,
    loadPaper,
  };
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useBench(): BenchCtx {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useBench must be used within BenchmarkProvider");
  return ctx;
}
