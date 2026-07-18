export type Locale = "en" | "ar";
export type Theme = "light" | "dark";

export interface SimParams {
  grid_size: number;
  n_uavs: number;
  n_sectors: number;
  n_timesteps: number;
  tau: number;
  seed: number;
  policy: "greedy" | "ppo";
  method: string;
  attack_type: "none" | "spoofing" | "spoofing_strategic" | "injection" | "byzantine";
  p_spoof: number;
  n_byzantine: number;
  p_drop: number;
  sensor_failure_rate: number;
}

export interface Predicate {
  conf_ok: boolean;
  human_approval: boolean;
  signature_ok: boolean | null;
  consensus_ok: boolean | null;
  satisfied: boolean;
}

export interface FrameEvent {
  kind: string;
  cert?: string;
  true_fire?: boolean;
  conf?: number;
  tau?: number;
  reason?: string;
  row?: number;
  col?: number;
  predicate?: Predicate;
}

export interface LedgerEntry extends FrameEvent {
  t: number;
}

export interface BreachPoint {
  p_c: number;
  gomdp: number;
  central: number;
}

export interface BreachResponse {
  n_validators: number;
  max_byzantine: number;
  threshold: number;
  bft_safe: boolean;
  points: BreachPoint[];
  note: string;
}

export interface FrameMetrics {
  ld: number | null;
  fp_pct: number;
  n_alerts: number;
  n_false: number;
  compliance: number;
  n_injections_blocked: number;
}

/** A UAV's role in the cooperative swarm this frame. */
export type UavRole = "scout" | "verifier" | "responder" | "static";

export interface UavPoint {
  x: number;
  y: number;
  batt: number;
  /** Cooperative role driving this UAV's colour + behaviour. */
  role?: UavRole;
  /** Where the UAV is heading this step (grid col/row), for a heading cue. */
  tx?: number;
  ty?: number;
}

/** A live communication link between two UAVs (indices into `uavs`). */
export type CommLink = [a: number, b: number, kind: "alert" | "relay"];

/** An active fire the swarm is tracking (grid col/row centroid + radius). */
export interface FireCluster {
  x: number;
  y: number;
  r: number;
  confirmed: boolean;
  size: number;
}

/** Swarm coordination phase: searching → verifying a contact → tracking fire. */
export type SwarmPhase = "search" | "verify" | "track" | "static";

export interface Frame {
  type: "frame";
  t: number;
  grid_size: number;
  /** uint8 per cell: 0 = grass, 1..255 = fire age (timesteps burning). */
  fire_b64: string;
  uavs: UavPoint[];
  /** Communication links to draw between UAVs this frame. */
  links?: CommLink[];
  /** Fire clusters the swarm is encircling this frame. */
  fires?: FireCluster[];
  /** Current swarm coordination phase. */
  phase?: SwarmPhase;
  event: FrameEvent | null;
  metrics: FrameMetrics;
}

export interface DoneMessage {
  type: "done";
  summary: Record<string, number | boolean | null>;
  ledger: Array<{ t: number } & FrameEvent>;
  meta: Record<string, unknown>;
}

export interface BenchmarkRow {
  method: string;
  label: string;
  enforcement: string;
  ld_mean: number;
  ld_std: number;
  ld_ci95: number;
  fp_mean: number;
  fp_std: number;
  fp_ci95: number;
  compliance_pct: number;
  n_seeds: number;
}

export interface BenchmarkResponse {
  source: string;
  rows: BenchmarkRow[];
  raw: Array<{ method: string; seed: number; ld: number | null; fp_pct: number; compliance: number }>;
  config: Record<string, number>;
}

export interface PaperResponse {
  source: string;
  label: string;
  table: string;
  rows: Array<Record<string, string>>;
}

export interface MethodMeta {
  id: string;
  label: string;
  enforcement: string;
  color: { light?: string; dark?: string };
  governance: boolean;
  hitl: boolean;
  blockchain: boolean;
  verification: boolean;
  coordination: boolean;
  policy: string;
}
