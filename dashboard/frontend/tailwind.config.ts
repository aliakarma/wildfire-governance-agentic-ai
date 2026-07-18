import type { Config } from "tailwindcss";

/**
 * Theme tokens are defined as CSS variables in globals.css and referenced here
 * so both Tailwind utilities and raw canvas/chart code read one palette.
 */
const config: Config = {
  darkMode: ["selector", '[data-theme="dark"]'],
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: "var(--bg)",
        surface: "var(--surface)",
        "surface-2": "var(--surface-2)",
        border: "var(--border)",
        text: "var(--text)",
        muted: "var(--text-muted)",
        accent: "var(--accent)",
        "accent-soft": "var(--accent-soft)",
        ok: "var(--ok)",
        danger: "var(--danger)",
        warn: "var(--warn)",
        info: "var(--info)",
        focus: "var(--focus)",
      },
      fontFamily: {
        sans: ["var(--font-sans)"],
        mono: ["var(--font-mono)"],
      },
      borderRadius: {
        card: "12px",
      },
    },
  },
  plugins: [],
};

export default config;
