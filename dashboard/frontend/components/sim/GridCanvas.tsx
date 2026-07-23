"use client";
import { useEffect, useMemo, useRef } from "react";
import { useTheme } from "@/components/providers/ThemeProvider";
import { b64ToU8, fireRGB, gridPalette, linkColor, roleColor } from "@/lib/colormap";
import type { Frame, UavPoint } from "@/lib/types";

interface Props {
  frame: Frame | null;
  showFire: boolean;
  showUavs: boolean;
  showSectors: boolean;
  /** Draw the UAV communication links + fire encirclement rings. */
  showComms?: boolean;
  nSectors: number;
  /** Recent past UAV position arrays (oldest first) for motion trails. */
  trail?: UavPoint[][];
}

const DISPLAY = 560;

export function GridCanvas({ frame, showFire, showUavs, showSectors, showComms = true, nSectors, trail }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const { theme } = useTheme();
  const pal = useMemo(() => gridPalette(theme), [theme]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = DISPLAY * dpr;
    canvas.height = DISPLAY * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const grid = frame?.grid_size ?? 40;
    const cs = DISPLAY / grid;
    // Cell-index → pixel-centre helper (matches how UAVs are placed).
    const px = (v: number) => (v + 0.5) * cs;

    // 1. Grass base.
    ctx.fillStyle = pal.grass;
    ctx.fillRect(0, 0, DISPLAY, DISPLAY);

    // 2. Fire cells (warm gradient by age).
    if (frame && showFire) {
      const fire = b64ToU8(frame.fire_b64);
      for (let r = 0; r < grid; r++) {
        for (let c = 0; c < grid; c++) {
          const age = fire[r * grid + c];
          if (age === 0) continue;
          const [rr, gg, bb] = fireRGB(age);
          ctx.fillStyle = `rgb(${rr},${gg},${bb})`;
          ctx.fillRect(c * cs, r * cs, cs, cs);
        }
      }
    }

    // 3. Grid lines (skip when cells get too small).
    if (cs >= 4) {
      ctx.strokeStyle = pal.gridline;
      ctx.lineWidth = Math.max(0.6, cs * 0.07);
      ctx.beginPath();
      for (let i = 0; i <= grid; i++) {
        const p = Math.round(i * cs) + 0.5;
        ctx.moveTo(p, 0);
        ctx.lineTo(p, DISPLAY);
        ctx.moveTo(0, p);
        ctx.lineTo(DISPLAY, p);
      }
      ctx.stroke();
    }

    // 4. Optional sector overlay.
    if (frame && showSectors) {
      const side = Math.ceil(Math.sqrt(nSectors));
      const cell = Math.floor(grid / side) * cs;
      ctx.strokeStyle = theme === "dark" ? "rgba(255,255,255,0.22)" : "rgba(0,0,0,0.18)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let i = 1; i < side; i++) {
        ctx.moveTo(i * cell, 0);
        ctx.lineTo(i * cell, DISPLAY);
        ctx.moveTo(0, i * cell);
        ctx.lineTo(DISPLAY, i * cell);
      }
      ctx.stroke();
    }

    if (!frame) return;
    const t = frame.t;

    // 5. Fire encirclement rings — the perimeter the swarm is tracking. Pulses
    //    while a contact is being verified, solid once confirmed. Shows the
    //    viewer "how much the fire has spread" at a glance.
    if (showComms && frame.fires && frame.fires.length) {
      for (const f of frame.fires) {
        const cx = px(f.x);
        const cy = px(f.y);
        const rad = Math.max(cs * 0.9, f.r * cs);
        const pulse = f.confirmed ? 0 : 0.5 + 0.5 * Math.sin(t * 0.35);
        ctx.save();
        ctx.setLineDash([6, 6]);
        ctx.lineDashOffset = -(t * 1.5) % 12;
        ctx.lineWidth = f.confirmed ? 2 : 2.5;
        ctx.strokeStyle = f.confirmed
          ? theme === "dark" ? "rgba(255,159,69,0.85)" : "rgba(228,87,46,0.85)"
          : theme === "dark" ? `rgba(255,196,77,${0.4 + 0.5 * pulse})` : `rgba(199,125,10,${0.4 + 0.5 * pulse})`;
        ctx.beginPath();
        ctx.arc(cx, cy, rad + cs * 0.8, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }
    }

    // 6. UAV motion trails — a fading polyline per UAV so the flight path each
    //    drone has taken is clearly visible (older segments fainter). Trails are
    //    index-aligned across frames, so uav i's path is trail[0][i] → … → now.
    const uavR = Math.max(4, Math.min(9, cs * 0.5));
    if (showUavs && trail && trail.length && frame.uavs.length) {
      const pathColor = theme === "dark" ? "57,198,255" : "11,111,184";
      const nUav = frame.uavs.length;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      for (let i = 0; i < nUav; i++) {
        // Build this UAV's recent positions (past trail + current).
        const pts: { x: number; y: number }[] = [];
        for (let d = 0; d < trail.length; d++) {
          const u = trail[d][i];
          if (u) pts.push({ x: px(u.x), y: px(u.y) });
        }
        pts.push({ x: px(frame.uavs[i].x), y: px(frame.uavs[i].y) });
        if (pts.length < 2) continue;
        for (let s = 1; s < pts.length; s++) {
          // Fade from faint (oldest) to bright (newest). Skip long jumps
          // (wrap-arounds) so the path stays clean.
          const dx = pts[s].x - pts[s - 1].x;
          const dy = pts[s].y - pts[s - 1].y;
          if (Math.hypot(dx, dy) > cs * 3) continue;
          const alpha = 0.12 + 0.5 * (s / (pts.length - 1));
          ctx.strokeStyle = `rgba(${pathColor},${alpha})`;
          ctx.lineWidth = Math.max(1, uavR * 0.35);
          ctx.beginPath();
          ctx.moveTo(pts[s - 1].x, pts[s - 1].y);
          ctx.lineTo(pts[s].x, pts[s].y);
          ctx.stroke();
        }
      }
    }

    // 6b. Heading cue — a short segment from each UAV toward its current target
    //     (where the coordinator is sending it), so intent is legible even when
    //     the drone has not moved far yet.
    if (showUavs) {
      ctx.save();
      ctx.setLineDash([2, 3]);
      ctx.lineWidth = 1;
      ctx.strokeStyle = theme === "dark" ? "rgba(232,237,244,0.35)" : "rgba(27,30,36,0.30)";
      for (const u of frame.uavs) {
        if (u.tx == null || u.ty == null) continue;
        const ox = px(u.x);
        const oy = px(u.y);
        const dx = px(u.tx) - ox;
        const dy = px(u.ty) - oy;
        const len = Math.hypot(dx, dy);
        if (len < cs * 0.6) continue; // already there
        const k = Math.min(1, (cs * 2.4) / len); // cap cue length
        ctx.beginPath();
        ctx.moveTo(ox, oy);
        ctx.lineTo(ox + dx * k, oy + dy * k);
        ctx.stroke();
      }
      ctx.restore();
    }

    // 7. Communication links between UAVs (drawn under the UAV dots).
    //    Bright animated "alert" edges = a UAV calling teammates to verify a
    //    detection; subtle "relay" edges = the encirclement comms mesh.
    if (showComms && showUavs && frame.links && frame.links.length) {
      for (const link of frame.links) {
        const a = frame.uavs[link[0]];
        const b = frame.uavs[link[1]];
        if (!a || !b) continue;
        const alert = link[2] === "alert";
        ctx.save();
        ctx.strokeStyle = linkColor(link[2], theme);
        if (alert) {
          ctx.lineWidth = 2;
          ctx.globalAlpha = 0.9;
          ctx.setLineDash([7, 5]);
          ctx.lineDashOffset = -(t * 3) % 12;
          ctx.shadowColor = ctx.strokeStyle as string;
          ctx.shadowBlur = 6;
        } else {
          ctx.lineWidth = 1;
          ctx.globalAlpha = 0.4;
          ctx.setLineDash([3, 5]);
          ctx.lineDashOffset = -(t * 1.2) % 8;
        }
        ctx.beginPath();
        ctx.moveTo(px(a.x), px(a.y));
        ctx.lineTo(px(b.x), px(b.y));
        ctx.stroke();
        ctx.restore();
      }
    }

    // 8. UAV agents — coloured by cooperative role, with a battery ring.
    if (showUavs) {
      for (const u of frame.uavs) {
        const cx = px(u.x);
        const cy = px(u.y);
        const color = showComms ? roleColor(u.role, theme) : pal.uav;
        ctx.save();
        ctx.shadowColor = color;
        ctx.shadowBlur = 6;
        ctx.beginPath();
        ctx.arc(cx, cy, uavR, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.restore();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = pal.uavStroke;
        ctx.stroke();
        // battery ring
        const batt = u.batt > 0.5 ? "#35D0A5" : u.batt > 0.2 ? "#F2B455" : "#FF5C72";
        ctx.beginPath();
        ctx.arc(cx, cy, uavR + 1.5, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * Math.max(0.05, u.batt));
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = batt;
        ctx.stroke();
      }
    }

    // 9. Event marker (alert/injection).
    const ev = frame.event;
    if (ev && ev.row != null && ev.col != null) {
      const cx = px(ev.col);
      const cy = px(ev.row);
      const color =
        ev.kind === "ALERT_APPROVED"
          ? "#35D0A5"
          : ev.kind === "ALERT_BLOCKED" || ev.kind === "INJECTION_BLOCKED"
            ? "#FF5C72"
            : "#FFFFFF";
      ctx.beginPath();
      ctx.arc(cx, cy, Math.max(10, cs), 0, Math.PI * 2);
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = color;
      ctx.stroke();
    }
  }, [frame, showFire, showUavs, showSectors, showComms, nSectors, theme, pal, trail]);

  return (
    <div dir="ltr" className="overflow-hidden rounded-card border">
      <canvas
        ref={canvasRef}
        role="img"
        aria-label={frame ? `Wildfire grid at step ${frame.t}, ${frame.uavs.length} UAVs` : "Wildfire grid (idle)"}
        style={{ width: DISPLAY, height: DISPLAY, display: "block", margin: "0 auto" }}
      />
    </div>
  );
}
