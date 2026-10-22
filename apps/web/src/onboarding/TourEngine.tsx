import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { ArrowLeft, ArrowRight, Check, X } from "lucide-react";
import { useI18n } from "../i18n";
import type { TourDefinition } from "./registry";

interface Props {
  tour: TourDefinition;
  onComplete: () => void;
  onSkip: () => void;
  onClose: () => void;
  confirmSkip: boolean;
}

type Rect = { left: number; top: number; width: number; height: number };

export function TourEngine({ tour, onComplete, onSkip, onClose, confirmSkip }: Props) {
  const { locale, t } = useI18n();
  const [index, setIndex] = useState(0);
  const [target, setTarget] = useState<HTMLElement | null>(null);
  const [rect, setRect] = useState<Rect | null>(null);
  const [placement, setPlacement] = useState({ left: 12, top: 12 });
  const [skipPrompt, setSkipPrompt] = useState(false);
  const dialogRef = useRef<HTMLElement>(null);
  const previousFocus = useRef<HTMLElement | null>(document.activeElement as HTMLElement | null);
  const activeStep = tour.steps[index];
  const textIndex = locale === "en" ? 1 : 0;
  const ready = Boolean(target && rect);

  useEffect(() => () => {
    if (previousFocus.current?.isConnected) previousFocus.current.focus();
  }, []);

  useEffect(() => {
    setTarget(null);
    if (!activeStep) return;
    let observer: MutationObserver | undefined;
    let cancelled = false;
    const resolve = () => {
      const element = document.querySelector<HTMLElement>(activeStep.target);
      if (!element || !element.isConnected || !element.getClientRects().length) return;
      observer?.disconnect();
      if (!cancelled) {
        setTarget(element);
        element.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "nearest" });
      }
    };
    const existing = document.querySelector<HTMLElement>(activeStep.target);
    if (existing?.getClientRects().length) resolve();
    else {
      observer = new MutationObserver(resolve);
      observer.observe(document.getElementById("root") ?? document.body, { childList: true, subtree: true });
      resolve();
    }
    const limit = window.setTimeout(() => {
      observer?.disconnect();
      if (!cancelled && !document.querySelector(activeStep.target)?.getClientRects().length) {
        if (index === tour.steps.length - 1) onClose();
        else setIndex((current) => current + 1);
      }
    }, 8000);
    return () => { cancelled = true; observer?.disconnect(); window.clearTimeout(limit); };
  }, [activeStep, index, tour]);

  useEffect(() => {
    if (!target) return;
    const measure = () => {
      const bounds = target.getBoundingClientRect();
      const pad = 8;
      const next = { left: Math.max(0, bounds.left - pad), top: Math.max(0, bounds.top - pad), width: bounds.width + pad * 2, height: bounds.height + pad * 2 };
      setRect((current) => current && Object.keys(next).every((key) => current[key as keyof Rect] === next[key as keyof Rect]) ? current : next);
      const width = Math.min(360, window.innerWidth - 24);
      const height = dialogRef.current?.offsetHeight ?? 260;
      const below = window.innerHeight - (next.top + next.height);
      const above = next.top;
      const top = below >= height + 16 || below >= above
        ? next.top + next.height + 12 : next.top - height - 12;
      const nextLeft = next.left + Math.min(next.width, 160);
      const left = Math.max(12, Math.min(nextLeft, window.innerWidth - width - 12));
      const boundedTop = Math.max(12, Math.min(top, window.innerHeight - height - 12));
      setPlacement((current) => current.left === left && current.top === boundedTop ? current : { left, top: boundedTop });
    };
    const resize = new ResizeObserver(measure);
    resize.observe(target);
    window.addEventListener("scroll", measure, true);
    window.addEventListener("resize", measure);
    measure();
    return () => { resize.disconnect(); window.removeEventListener("scroll", measure, true); window.removeEventListener("resize", measure); };
  }, [target, locale, skipPrompt]);

  useEffect(() => {
    if (activeStep?.waitFor !== "valueChange" || !target) return;
    const initialValue = (target as HTMLInputElement).value;
    const changed = () => {
      if ((target as HTMLInputElement).value !== initialValue) {
        setIndex((current) => current + 1);
      }
    };
    target.addEventListener("input", changed);
    target.addEventListener("change", changed);
    return () => { target.removeEventListener("input", changed); target.removeEventListener("change", changed); };
  }, [activeStep, target]);

  useEffect(() => {
    if (!ready) return;
    (activeStep.waitFor === "valueChange" ? target : dialogRef.current)?.focus();
  }, [activeStep, target, ready]);

  useEffect(() => {
    if (skipPrompt) dialogRef.current?.focus();
  }, [skipPrompt]);

  useEffect(() => {
    const keydown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        if (skipPrompt) setSkipPrompt(false);
        else if (confirmSkip) setSkipPrompt(true);
        else onClose();
      }
      if (event.key === "Tab" && dialogRef.current) {
        const focusables = [...dialogRef.current.querySelectorAll<HTMLElement>("button:not(:disabled)")];
        if (!skipPrompt && activeStep?.waitFor === "valueChange" && target) focusables.push(target);
        const position = focusables.indexOf(document.activeElement as HTMLElement);
        if (event.shiftKey && (position === 0 || position === -1)) { event.preventDefault(); focusables[focusables.length - 1]?.focus(); }
        else if (!event.shiftKey && (position === focusables.length - 1 || position === -1)) { event.preventDefault(); focusables[0]?.focus(); }
      }
      if (skipPrompt || !ready) return;
      if (event.key === "ArrowRight" && activeStep?.waitFor !== "valueChange") {
        event.preventDefault();
        if (index === tour.steps.length - 1) onComplete();
        else setIndex(index + 1);
      }
      if (event.key === "ArrowLeft" && index > 0 && document.activeElement !== target) { event.preventDefault(); setIndex(index - 1); }
    };
    window.addEventListener("keydown", keydown);
    return () => window.removeEventListener("keydown", keydown);
  }, [activeStep, confirmSkip, index, onClose, onComplete, ready, skipPrompt, target, tour.steps.length]);

  if (!activeStep) return null;
  const blocks = rect ? [
    { left: 0, top: 0, width: "100%", height: rect.top },
    { left: 0, top: rect.top, width: rect.left, height: rect.height },
    { left: rect.left + rect.width, top: rect.top, width: `calc(100% - ${rect.left + rect.width}px)`, height: rect.height },
    { left: 0, top: rect.top + rect.height, width: "100%", height: `calc(100% - ${rect.top + rect.height}px)` },
  ] : [{ left: 0, top: 0, width: "100%", height: "100%" }];

  return createPortal(<div className="tour-layer">
    {blocks.map((style, blockIndex) => <div key={blockIndex} className="tour-shade" style={style} />)}
    {rect && <div className="tour-spotlight" style={rect} />}
    <section ref={dialogRef} className="tour-popover" style={rect ? placement : { left: Math.max(12, (window.innerWidth - Math.min(360, window.innerWidth - 24)) / 2), top: Math.max(12, (window.innerHeight - 240) / 2) }} role="dialog" aria-modal="true" aria-labelledby="tour-title" aria-describedby="tour-description" tabIndex={-1}>
      {skipPrompt ? <>
        <div className="tour-kicker">{t("平台介绍", "PLATFORM INTRODUCTION")}</div>
        <h2 id="tour-title">{t("跳过首次引导？", "Skip the introduction?")}</h2>
        <p id="tour-description">{t("之后不会再自动显示。你仍然可以通过页面右上角的「页面引导」重新查看。", "It will not appear automatically again. You can revisit guidance from Page Tour at the top right.")}</p>
        <footer><button onClick={() => setSkipPrompt(false)}>{t("继续引导", "Continue tour")}</button><button className="tour-next" onClick={onSkip}>{t("跳过", "Skip")}</button></footer>
      </> : <>
        <div className="tour-kicker"><span>{t("页面引导", "PAGE GUIDE")}</span><span>{String(index + 1).padStart(2, "0")} / {String(tour.steps.length).padStart(2, "0")}</span></div>
        <h2 id="tour-title">{activeStep.title[textIndex]}</h2>
        <p id="tour-description">{ready ? activeStep.description[textIndex] : t("正在等待页面内容加载...", "Waiting for page content...")}</p>
        <footer>
          <button className="tour-skip" onClick={() => confirmSkip ? setSkipPrompt(true) : onClose()}>{t("跳过", "Skip")}</button>
          <div>
            <button title={t("上一步", "Previous step")} aria-label={t("上一步", "Previous step")} disabled={index === 0} onClick={() => setIndex(index - 1)}><ArrowLeft size={17} /></button>
            {activeStep.waitFor === "valueChange" && ready ? <span className="tour-wait">{t("请调整 J2", "Adjust J2")}</span> : <button className="tour-next" disabled={!ready} onClick={() => index === tour.steps.length - 1 ? onComplete() : setIndex(index + 1)}>{index === tour.steps.length - 1 ? <><Check size={16} /> {confirmSkip ? t("开始使用", "Start using") : t("完成", "Finish")}</> : <>{t("下一步", "Next")} <ArrowRight size={16} /></>}</button>}
          </div>
        </footer>
      </>}
      <button className="tour-close" title={t("关闭引导", "Close tour")} aria-label={t("关闭引导", "Close tour")} onClick={() => confirmSkip ? setSkipPrompt(true) : onClose()}><X size={16} /></button>
    </section>
  </div>, document.body);
}
