/**
 * PyroRL-style palette: green grass base with a warm fire gradient keyed by how
 * long a cell has been burning (fresh leading edge = bright yellow, older core =
 * deep ember red).
 */

/** Warm fire ramp control stops (t → RGB), t in [0,1] over fire age. */
const FIRE_STOPS: Array<[number, [number, number, number]]> = [
  [0.0, [255, 241, 148]], // hot leading edge — near white-yellow
  [0.15, [255, 202, 40]], // yellow
  [0.35, [255, 143, 26]], // orange
  [0.6, [233, 58, 30]], // red
  [1.0, [140, 26, 12]], // dark ember
];

const AGE_MAX = 45; // ages at/above this map to the darkest ember

function lerp(a: number, b: number, f: number): number {
  return a + (b - a) * f;
}

/** Precomputed LUT indexed by fire age 0..255 (index 0 is unused = grass). */
function buildFireLut(): Uint8ClampedArray {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let age = 0; age < 256; age++) {
    const t = Math.min(age, AGE_MAX) / AGE_MAX;
    let lo = FIRE_STOPS[0];
    let hi = FIRE_STOPS[FIRE_STOPS.length - 1];
    for (let i = 0; i < FIRE_STOPS.length - 1; i++) {
      if (t >= FIRE_STOPS[i][0] && t <= FIRE_STOPS[i + 1][0]) {
        lo = FIRE_STOPS[i];
        hi = FIRE_STOPS[i + 1];
        break;
      }
    }
    const span = hi[0] - lo[0] || 1;
    const f = (t - lo[0]) / span;
    lut[age * 3] = lerp(lo[1][0], hi[1][0], f);
    lut[age * 3 + 1] = lerp(lo[1][1], hi[1][1], f);
    lut[age * 3 + 2] = lerp(lo[1][2], hi[1][2], f);
  }
  return lut;
}

export const FIRE_LUT = buildFireLut();

export function fireRGB(age: number): [number, number, number] {
  const a = Math.max(0, Math.min(255, age));
  return [FIRE_LUT[a * 3], FIRE_LUT[a * 3 + 1], FIRE_LUT[a * 3 + 2]];
}

export interface GridPalette {
  grass: string;
  gridline: string;
  uav: string;
  uavStroke: string;
}

export function gridPalette(theme: "light" | "dark"): GridPalette {
  return theme === "dark"
    ? { grass: "#17A877", gridline: "#0B0E14", uav: "#39C6FF", uavStroke: "#E8F6FF" }
    : { grass: "#25C48D", gridline: "#FFFFFF", uav: "#0B6FB8", uavStroke: "#FFFFFF" };
}

/**
 * Colour a UAV by its cooperative role so the story reads at a glance:
 * blue = scouting/searching, amber = called in to verify, green = confirmed and
 * holding station around the fire, grey = uncoordinated (static baseline).
 */
export function roleColor(role: string | undefined, theme: "light" | "dark"): string {
  const dark = theme === "dark";
  switch (role) {
    case "verifier":
      return dark ? "#FFC04D" : "#C77D0A"; // amber — converging to confirm
    case "responder":
      return dark ? "#35D0A5" : "#0E9E77"; // green — on station, encircling
    case "static":
      return dark ? "#9AA6B8" : "#6B7280"; // grey — no coordination
    case "scout":
    default:
      return dark ? "#39C6FF" : "#0B6FB8"; // blue — searching
  }
}

/** Colour for a communication link by kind (bright alert vs subtle relay mesh). */
export function linkColor(kind: string, theme: "light" | "dark"): string {
  const dark = theme === "dark";
  if (kind === "alert") return dark ? "#FF9F45" : "#E4572E"; // urgent broadcast
  return dark ? "#4FE0C0" : "#12987A"; // relay mesh
}

/** Decode a base64 string into a Uint8Array (browser-safe). */
export function b64ToU8(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
