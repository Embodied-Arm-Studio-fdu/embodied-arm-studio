import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LearningGuidePage } from "./App";
import { I18nProvider } from "./i18n";

describe("LearningGuidePage", () => {
  it("presents the six-stage S1 guide and teaching Agent without Mock training claims", () => {
    const markup = renderToStaticMarkup(<I18nProvider><LearningGuidePage /></I18nProvider>);

    expect(markup).toContain("学习路线与实验流程");
    expect(markup.match(/aria-current="step"/g)).toHaveLength(1);
    expect(markup.match(/class="learning-guide-flow"/g)).toHaveLength(1);
    for (const stage of ["认识平台", "准备六轴场景", "理解任务与策略", "配置训练", "启动并观察 Run", "评估和复盘"]) {
      expect(markup).toContain(stage);
    }
    expect(markup).toContain("S1 Push");
    expect(markup).toContain("S1 Reach");
    expect(markup).toContain("问教学 Agent");
    expect(markup).not.toContain("Mock Reach Baseline");
  });
});
