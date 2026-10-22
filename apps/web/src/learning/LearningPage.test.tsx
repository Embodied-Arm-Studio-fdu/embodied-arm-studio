import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { I18nProvider } from "../i18n";
import { LEARNING_CHAPTERS, getLearningChapters } from "../content/learning/chapters";
import { LearningPage } from "./LearningPage";

describe("LearningPage", () => {
  it("publishes all twelve manuscript chapters in a two-column learning layout", () => {
    expect(LEARNING_CHAPTERS).toHaveLength(12);
    expect(LEARNING_CHAPTERS.map((chapter) => chapter.number)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const markup = renderToStaticMarkup(<I18nProvider initialLocale="zh-CN"><LearningPage path="/learn/1" navigate={() => undefined}/></I18nProvider>);
    expect(markup).toContain("入门学习");
    expect(markup).toContain("课程目录");
    expect(markup).toContain("本章目录");
    expect(markup).toContain("阅读进度");
    expect(markup).toContain("关键术语");
    expect(markup).not.toContain("learning-toc");
  });

  it("keeps the chapter-one embodied loop as the controllable core animation", () => {
    const markup = renderToStaticMarkup(<I18nProvider initialLocale="zh-CN"><LearningPage path="/learn/1" navigate={() => undefined}/></I18nProvider>);
    expect(markup).toContain("环境状态 → 观测 → 决策 → 动作 → 环境变化 → 新观测");
    for (const phase of ["环境状态", "观测", "决策", "动作", "环境变化", "新观测"]) expect(markup).toContain(phase);
    for (const control of ["开始", "暂停", "重置", "单步", "速度"]) expect(markup).toContain(control);
  });

  it("renders both precise chapter-two SVGs and removes the specified sentence", () => {
    const markup = renderToStaticMarkup(<I18nProvider initialLocale="zh-CN"><LearningPage path="/learn/2" navigate={() => undefined}/></I18nProvider>);
    expect(markup).toContain('data-learning-visual="arm-anatomy"');
    expect(markup).toContain("经典六自由度串联机械臂构成示意");
    expect(markup).toContain("并非S1机械臂精确结构图");
    expect(markup).toContain('data-learning-visual="planar-kinematics"');
    expect(markup).toContain("x = 0.346 + 0.000 = 0.346 m");
    expect(markup).toContain("y = 0.200 + 0.300 = 0.500 m");
    expect(markup).not.toContain("教学中应把");
  });

  it("renders practice chapters progressively without executable Python", () => {
    const markup = [10, 11, 12].map((chapter) => renderToStaticMarkup(<I18nProvider initialLocale="zh-CN"><LearningPage path={`/learn/${chapter}`} navigate={() => undefined}/></I18nProvider>)).join("\n");
    expect(markup).toContain("网页可修改");
    expect(markup).toContain("受控开发环境练习");
    expect(markup).toContain("只读教学示例");
    expect(markup).toContain("learning-details");
    expect(markup).toContain("learning-answer");
    expect(markup).toContain("复制代码");
    expect(markup).toContain("不在浏览器执行");
    expect(markup).not.toContain('type="submit"');
  });
});

describe("English learning materials", () => {
  for (let chapter = 1; chapter <= 12; chapter += 1) {
    it(`renders chapter ${chapter} entirely in English, including code, diagrams and accessibility labels`, () => {
      const markup = renderToStaticMarkup(<I18nProvider initialLocale="en"><LearningPage path={`/learn/${chapter}`} navigate={() => undefined}/></I18nProvider>);
      expect(markup).not.toMatch(/[\u3400-\u9fff]/u);
      expect(markup).toContain("Introduction to Embodied AI");
      expect(markup).toContain(`Chapter ${chapter}`);
      expect(markup).toContain("Course contents");
      expect(markup).not.toContain("[[BLOCK");
      expect(markup).not.toContain("[[REFERENCES]]");
    });
  }

  it("preserves every section link across language changes", () => {
    const english = getLearningChapters("en");
    expect(english).toHaveLength(12);
    for (const chapter of LEARNING_CHAPTERS) {
      const translated = english[chapter.number - 1];
      expect(translated.sections.map(section => section.id)).toEqual(chapter.sections.map(section => section.id));
      for (const section of translated.sections) expect(section.markdown.length).toBeGreaterThan(100);
    }
  });
});
