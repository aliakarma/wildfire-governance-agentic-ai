import type { SimParams } from "./types";

/**
 * Base URLs. By default the UI talks to its own origin (works when the FastAPI
 * backend serves the exported build). For split dev (Next on :3000, backend on
 * another port) set NEXT_PUBLIC_API_BASE / NEXT_PUBLIC_WS_BASE at build time.
 */
function defaultOrigin(): string {
  if (process.env.NEXT_PUBLIC_API_BASE) return process.env.NEXT_PUBLIC_API_BASE;
  if (typeof window !== "undefined") return window.location.origin;
  return "http://localhost:8000";
}

export const API_BASE = defaultOrigin();
export const WS_BASE =
  process.env.NEXT_PUBLIC_WS_BASE ?? API_BASE.replace(/^http/, "ws");

export const DEFAULT_PARAMS: SimParams = {
  grid_size: 80,
  n_uavs: 16,
  n_sectors: 25,
  n_timesteps: 500,
  tau: 0.72,
  seed: 0,
  policy: "greedy",
  method: "greedy_gomdp",
  attack_type: "none",
  p_spoof: 0.0,
  n_byzantine: 0,
  p_drop: 0.0,
  sensor_failure_rate: 0.0,
};

export async function fetchSchema() {
  const res = await fetch(`${API_BASE}/api/config/schema`);
  if (!res.ok) throw new Error("schema fetch failed");
  return res.json();
}

export async function fetchMethods() {
  const res = await fetch(`${API_BASE}/api/methods`);
  if (!res.ok) throw new Error("methods fetch failed");
  return res.json();
}

export function exportGifUrl() {
  return `${API_BASE}/api/export/gif`;
}

export async function fetchBreach(nValidators: number, maxByzantine: number) {
  const res = await fetch(
    `${API_BASE}/api/breach-probability?n_validators=${nValidators}&max_byzantine=${maxByzantine}`,
  );
  if (!res.ok) throw new Error("breach fetch failed");
  return res.json();
}

export async function runBenchmark(payload: {
  methods: string[];
  n_seeds: number;
  n_uavs: number;
  grid_size: number;
  n_timesteps: number;
  tau: number;
}) {
  const res = await fetch(`${API_BASE}/api/benchmark`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error("benchmark failed");
  return res.json();
}

export async function fetchPaperResults(table: string) {
  const res = await fetch(`${API_BASE}/api/paper-results/${table}`);
  if (!res.ok) throw new Error("paper results fetch failed");
  return res.json();
}

export interface ArtifactMeta {
  id: string;
  title: string;
  kind: "table" | "figure";
  paper_ref: string;
  route: string;
  provenance: "exact" | "calibration" | "reference" | "supplementary";
  metrics: string[];
  csv_present: boolean;
}

export async function fetchArtifacts(): Promise<{ artifacts: ArtifactMeta[]; note: string }> {
  const res = await fetch(`${API_BASE}/api/artifacts`);
  if (!res.ok) throw new Error("artifacts fetch failed");
  return res.json();
}
