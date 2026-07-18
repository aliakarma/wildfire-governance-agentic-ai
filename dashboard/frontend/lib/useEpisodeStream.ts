"use client";
import { useCallback, useRef, useState } from "react";
import { WS_BASE } from "./api";
import type { DoneMessage, Frame, SimParams } from "./types";

export type StreamStatus = "idle" | "connecting" | "running" | "done" | "error";

export interface StreamState {
  frames: Frame[];
  status: StreamStatus;
  summary: DoneMessage["summary"] | null;
  meta: DoneMessage["meta"] | null;
  error: string | null;
  start: (params: SimParams) => void;
  stop: () => void;
}

/**
 * Opens a WebSocket to /ws/simulate, buffers streamed frames, and exposes the
 * terminal summary. Frames are kept in a ref-backed array and mirrored to state
 * in batches to avoid a re-render per frame.
 */
export function useEpisodeStream(): StreamState {
  const [frames, setFrames] = useState<Frame[]>([]);
  const [status, setStatus] = useState<StreamStatus>("idle");
  const [summary, setSummary] = useState<DoneMessage["summary"] | null>(null);
  const [meta, setMeta] = useState<DoneMessage["meta"] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const bufRef = useRef<Frame[]>([]);
  const flushRef = useRef<number | null>(null);

  const flush = useCallback(() => {
    if (bufRef.current.length) {
      setFrames([...bufRef.current]);
    }
    flushRef.current = null;
  }, []);

  const scheduleFlush = useCallback(() => {
    if (flushRef.current == null) {
      flushRef.current = window.setTimeout(flush, 60);
    }
  }, [flush]);

  const stop = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
  }, []);

  const start = useCallback(
    (params: SimParams) => {
      stop();
      bufRef.current = [];
      setFrames([]);
      setSummary(null);
      setMeta(null);
      setError(null);
      setStatus("connecting");

      const ws = new WebSocket(`${WS_BASE}/ws/simulate`);
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus("running");
        ws.send(JSON.stringify({ params }));
      };
      ws.onerror = () => {
        setStatus("error");
        setError("WebSocket connection failed. Is the backend running on :8000?");
      };
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.type === "frame") {
          bufRef.current.push(msg as Frame);
          scheduleFlush();
        } else if (msg.type === "done") {
          flush();
          setSummary((msg as DoneMessage).summary);
          setMeta((msg as DoneMessage).meta);
          setStatus("done");
        } else if (msg.type === "error") {
          setStatus("error");
          setError(msg.message ?? "Simulation error");
        }
      };
      ws.onclose = () => {
        setStatus((s) => (s === "running" ? "done" : s));
      };
    },
    [flush, scheduleFlush, stop],
  );

  return { frames, status, summary, meta, error, start, stop };
}
