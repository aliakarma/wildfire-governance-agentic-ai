"use client";
import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { translate } from "@/lib/i18n";
import type { Locale } from "@/lib/types";

interface LangCtx {
  locale: Locale;
  dir: "ltr" | "rtl";
  t: (key: string) => string;
  toggle: () => void;
}
const Ctx = createContext<LangCtx>({
  locale: "en",
  dir: "ltr",
  t: (k) => k,
  toggle: () => {},
});

export function LanguageProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocale] = useState<Locale>("en");

  useEffect(() => {
    let initial: Locale = "en";
    try {
      initial = (localStorage.getItem("locale") as Locale) || "en";
    } catch {
      /* ignore */
    }
    applyLocale(initial);
    setLocale(initial);
  }, []);

  const applyLocale = (loc: Locale) => {
    const dir = loc === "ar" ? "rtl" : "ltr";
    document.documentElement.setAttribute("lang", loc);
    document.documentElement.setAttribute("dir", dir);
  };

  const toggle = useCallback(() => {
    setLocale((prev) => {
      const next: Locale = prev === "en" ? "ar" : "en";
      applyLocale(next);
      try {
        localStorage.setItem("locale", next);
      } catch {
        /* ignore */
      }
      return next;
    });
  }, []);

  const dir = locale === "ar" ? "rtl" : "ltr";
  const t = useCallback((key: string) => translate(locale, key), [locale]);

  return <Ctx.Provider value={{ locale, dir, t, toggle }}>{children}</Ctx.Provider>;
}

export const useLang = () => useContext(Ctx);
