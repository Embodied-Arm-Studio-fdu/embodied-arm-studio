import { useEffect, useMemo, useState } from "react";
import { BookOpen, ChevronLeft, ChevronRight } from "lucide-react";
import { LEARNING_CHAPTERS, getLearningChapters, learningPath } from "../content/learning/chapters";
import { useI18n } from "../i18n";
import { LearningMarkdown } from "./LearningMarkdown";
import { LearningVisual } from "./LearningVisuals";

function chapterFromPath(path: string): number {
  const value = Number(path.match(/^\/learn\/(\d+)/)?.[1] ?? 1);
  return Math.min(LEARNING_CHAPTERS.length, Math.max(1, Number.isFinite(value) ? value : 1));
}

function chapterTerms(title: string): string {
  return title.replace(/^\d+\.\d+\s*/, "").replace(/^(?:实验|Experiment\s+)\d+-\d+\s*/, "");
}

function PracticeScopeLegend() {
  const { t } = useI18n();
  return <aside className="learning-practice-scope" aria-label={t("实践内容权限说明", "Practice permissions")}>
    <div><strong>{t("网页可修改", "Editable in the browser")}</strong><span>{t("仅限页面明确提供的受控参数和奖励表达式。", "Limited to the controlled parameters and reward expressions provided on the page.")}</span></div>
    <div><strong>{t("受控开发环境练习", "Controlled development exercises")}</strong><span>{t("需要在课程 allowlist 内修改源码并运行指定测试。", "Edit source files within the course allowlist and run the specified tests.")}</span></div>
    <div><strong>{t("只读教学示例", "Read-only teaching examples")}</strong><span>{t("用于手算和理解，不在网页执行任意 Python。", "For calculation and explanation; arbitrary Python is not executed in the browser.")}</span></div>
  </aside>;
}

export function LearningPage({ path, navigate }: { path: string; navigate: (path: string) => void }) {
  const { locale, t } = useI18n();
  const chapters = getLearningChapters(locale);
  const chapterNumber = chapterFromPath(path);
  const chapter = chapters[chapterNumber - 1];
  const [activeSection, setActiveSection] = useState(chapter.sections[0]?.id ?? "");
  useEffect(() => {
    setActiveSection(location.hash.slice(1) || chapter.sections[0]?.id || "");
    window.scrollTo({ top: 0 });
  }, [chapter.number]);
  useEffect(() => {
    const nodes = chapter.sections.map((section) => document.getElementById(section.id)).filter(Boolean) as HTMLElement[];
    if (!nodes.length || !("IntersectionObserver" in window)) return;
    const observer = new IntersectionObserver((entries) => {
      const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (visible) setActiveSection(visible.target.id);
    }, { rootMargin: "-12% 0px -68%", threshold: [0, .2, .6] });
    nodes.forEach((node) => observer.observe(node));
    return () => observer.disconnect();
  }, [chapter.number, chapter.sections]);
  const activeIndex = Math.max(0, chapter.sections.findIndex((section) => section.id === activeSection));
  const overallProgress = useMemo(() => Math.round((((chapter.number - 1) + (activeIndex + 1) / chapter.sections.length) / chapters.length) * 100), [chapter.number, chapter.sections.length, chapters.length, activeIndex]);
  const goSection = (id: string) => {
    history.replaceState({}, "", learningPath(chapter.number, id));
    setActiveSection(id);
    document.getElementById(id)?.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  };
  return <main className="learning-page" lang={locale} translate="no">
    <header className="learning-page-header">
      <div><span className="eyebrow">{t("具身智能 · 知识模块", "EMBODIED AI · KNOWLEDGE MODULE")}</span><h1>{t("入门学习", "Introduction to Embodied AI")}</h1><p>{t("系统讲解具身智能、机械臂、强化学习与可信评估。正文来自最新版《introduction_to_embodied》，与 Roadmap 的实验顺序和 Guide 的平台操作说明彼此独立。", "Learn the foundations of embodied AI, robot arms, reinforcement learning, and reliable evaluation. Based on the latest introduction_to_embodied manuscript, this material complements the Roadmap's experiment sequence and the Guide's platform instructions.")}</p></div>
      <div className="learning-edition"><BookOpen size={20}/><strong>{t("第1—12章", "Chapters 1–12")}</strong><small>{t("原理与渐进式实践 · 正式发布", "Foundations and progressive practice · Published edition")}</small></div>
    </header>
    <div className="learning-layout">
      <aside className="learning-chapter-nav" aria-label={t("学习导航", "Learning navigation")}>
        <section className="learning-nav-block"><h2>{t("课程目录", "Course contents")}</h2>{chapters.map((item) => <button key={item.number} className={item.number === chapter.number ? "active" : ""} aria-current={item.number === chapter.number ? "page" : undefined} onClick={() => navigate(learningPath(item.number))}><em>{String(item.number).padStart(2, "0")}</em><strong>{item.title}</strong></button>)}</section>
        <section className="learning-nav-block learning-chapter-sections"><h2>{t("本章目录", "In this chapter")}</h2>{chapter.sections.map((section) => <button key={section.id} className={section.id === activeSection ? "active" : ""} onClick={() => goSection(section.id)}>{section.title}</button>)}</section>
        <section className="learning-nav-block learning-progress"><div><h2>{t("阅读进度", "Reading progress")}</h2><strong>{overallProgress}%</strong></div><progress value={overallProgress} max="100" aria-label={t(`课程阅读进度 ${overallProgress}%`, `Course reading progress ${overallProgress}%`)}/></section>
        <section className="learning-nav-block learning-terms"><h2>{t("关键术语", "Key concepts")}</h2><p>{chapter.sections.slice(0, 5).map((section) => chapterTerms(section.title)).join(" · ")}</p></section>
      </aside>
      <article className="learning-article">
        <div className="learning-article-kicker">CHAPTER {String(chapter.number).padStart(2, "0")} / {chapters.length}</div><h2>{t(`第${chapter.number}章`, `Chapter ${chapter.number}`)} {chapter.title}</h2>
        {chapter.number >= 10 && <PracticeScopeLegend/>}<LearningVisual key={chapter.number} chapter={chapter.number}/>
        {chapter.sections.map((section) => <section className="learning-section" id={section.id} key={section.id}><h2>{section.title}</h2><LearningMarkdown source={section.markdown}/></section>)}
        <footer className="learning-pagination"><button disabled={chapter.number === 1} onClick={() => navigate(learningPath(chapter.number - 1))}><ChevronLeft size={15}/>{t("上一章", "Previous chapter")}</button><span>{t(`第 ${chapter.number} / ${chapters.length} 章`, `Chapter ${chapter.number} / ${chapters.length}`)}</span><button disabled={chapter.number === chapters.length} onClick={() => navigate(learningPath(chapter.number + 1))}>{t("下一章", "Next chapter")}<ChevronRight size={15}/></button></footer>
      </article>
    </div>
  </main>;
}
