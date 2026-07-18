"use client";
import { createContext, useCallback, useContext, useEffect, useState } from "react";
import type { Theme } from "@/lib/types";

interface ThemeCtx {
  theme: Theme;
  toggle: () => void;
}
const Ctx = createContext<ThemeCtx>({ theme: "dark", toggle: () => {} });

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<Theme>("dark");

  useEffect(() => {
    const initial = (document.documentElement.getAttribute("data-theme") as Theme) || "dark";
    setTheme(initial);
  }, []);

  const toggle = useCallback(() => {
    setTheme((prev) => {
      const next: Theme = prev === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try {
        localStorage.setItem("theme", next);
      } catch {
        /* ignore */
      }
      return next;
    });
  }, []);

  return <Ctx.Provider value={{ theme, toggle }}>{children}</Ctx.Provider>;
}

export const useTheme = () => useContext(Ctx);

/** Inline script string that sets data-theme before paint (prevents FOUC). */
export const THEME_SCRIPT = `
(function(){try{
  var t = localStorage.getItem('theme')
    || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  document.documentElement.setAttribute('data-theme', t);
}catch(e){document.documentElement.setAttribute('data-theme','dark');}})();
`;
