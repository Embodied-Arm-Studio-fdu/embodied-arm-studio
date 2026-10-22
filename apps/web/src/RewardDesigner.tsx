import { useI18n } from "./i18n";
import { readRewardWeights, rewardLessons, rewardPreset, writeRewardWeights } from "./rewardTeaching";

export function RewardDesigner({ source, terms, disabled, onChange, onExplain, onViewSource }: {
  source: string; terms: string[]; disabled: boolean; onChange: (source: string) => void;
  onExplain: (question: string) => void; onViewSource: () => void;
}) {
  const { t } = useI18n();
  const weights = readRewardWeights(source, terms);
  const canTeach = terms.length > 0 && terms.every((term) => rewardLessons[term]);
  const sampleTotal = weights ? terms.reduce((sum, term) => sum + (rewardLessons[term]?.sample ?? 0) * weights[term], 0) : null;
  function applyPreset(preset: "baseline" | "progress" | "restraint") {
    if (!weights && !window.confirm(t("当前为自定义表达式。使用教学方案会替换未保存的表达式，继续？", "This is a custom expression. A teaching preset will replace it, including unsaved changes. Continue?"))) return;
    onChange(writeRewardWeights(terms, rewardPreset(terms, preset)));
  }
  return <div className="reward-designer" data-tour="workbench-reward">
    <div className="reward-learning-intro"><h2>{t("你希望机械臂更重视什么？", "What should the arm prioritize?")}</h2><p>{t("先只调整一项。1× 保持默认贡献，2× 放大一倍，0× 不计入这项。", "Read each term, then change one weight. 1× keeps its contribution, 2× doubles it, and 0× excludes it.")}</p></div>
    <div className="reward-presets" aria-label={t("奖励教学方案", "Reward teaching presets")}>
      {(["baseline", "progress", "restraint"] as const).map((preset) => <button className="secondary" key={preset} aria-pressed={Boolean(weights && terms.every((term) => weights[term] === rewardPreset(terms, preset)[term]))} disabled={disabled || !canTeach} onClick={() => applyPreset(preset)}>{preset === "baseline" ? t("默认方案", "Baseline") : preset === "progress" ? t("更重视目标进展", "Emphasize progress") : terms.includes("action_failure") ? t("更重视动作与失败成本", "Emphasize action and failure costs") : t("更重视节制动作", "Emphasize smaller actions")}</button>)}
    </div>
    {!weights && <p className="reward-custom-note" role="status">{t("当前表达式超出教学权重模式，原代码已保留。请使用进阶代码编辑，或选择教学方案并确认替换。", "This expression is outside the teaching weight mode. Your source is preserved. Use advanced code editing, or choose a preset and confirm replacement.")}</p>}
    <div className="reward-term-cards">{terms.map((term) => {
      const lesson = rewardLessons[term];
      return <article className="reward-term-card" key={term}>
        <div className="reward-term-heading"><h3>{lesson ? t(...lesson.title) : term}</h3><span>{lesson && lesson.sample < 0 ? t("扣分项", "Penalty") : t("进展 / 得分项", "Progress / bonus")}</span></div>
        <p>{lesson ? t(...lesson.explanation) : t("请查看当前任务源码了解该项。", "See the task source for this term.")}</p>
        <label className="reward-weight-control"><span>{t("权重", "Weight")}</span><input type="range" min="0" max="5" step="0.1" aria-label={t(`${lesson?.title[0] ?? term}权重滑块`, `${lesson?.title[1] ?? term} weight slider`)} value={weights?.[term] ?? 1} disabled={disabled || !weights || !canTeach} onChange={(event) => onChange(writeRewardWeights(terms, { ...weights, [term]: Number(event.target.value) }))} /><input type="number" aria-label={t(`${lesson?.title[0] ?? term}权重`, `${lesson?.title[1] ?? term} weight`)} min="0" max="5" step="0.1" value={weights?.[term] ?? ""} disabled={disabled || !weights || !canTeach} onChange={(event) => { const value = event.target.valueAsNumber; if (Number.isFinite(value) && value >= 0 && value <= 5) onChange(writeRewardWeights(terms, { ...weights, [term]: value })); }} /><span>×</span></label>
        <span className="wb-weight-change">{t("默认权重", "Default weight")}: 1×{weights && weights[term] !== 1 ? t(` · 已调整为 ${weights[term]}×`, ` · Changed to ${weights[term]}×`) : ""}</span>
        <details className="wb-reward-explanation"><summary>{t("计算说明与观察建议", "Calculation and observation tips")}</summary><code>{term}</code>{lesson && weights && <div className="reward-term-example">{t("示例贡献", "Example contribution")}: {lesson.sample.toFixed(2)} × {weights[term]} = <strong>{(lesson.sample * weights[term]).toFixed(3)}</strong></div>}
        <p>{t("示例用于理解计算，并非实际测量。训练后结合目标距离、正式评估和视频检查行为；增大权重不保证效果改善。", "This example illustrates the calculation, not measured behavior. Inspect goal distance, formal evaluation and video after training. Higher weights do not guarantee improvement.")}</p></details>
        <button className="workbench-guide-action" disabled={disabled} onClick={() => onExplain(t(`请面向初学者解释 ${term}：含义、正负号、当前系数及它与默认权重的差异。请给出一步数值示例，并说明实验后应查看什么指标，不要保证训练效果。`, `Explain ${term} to a beginner: its meaning, sign, current coefficient and difference from the baseline weight. Give a one-step numeric example and explain which metrics to inspect without guaranteeing training outcomes.`))}>{t("请 Agent 解释这项", "Ask Agent about this term")}</button>
      </article>;
    })}</div>
    {sampleTotal !== null && canTeach && <details className="wb-reward-total"><summary>{t("查看一步计算示例", "View a one-step calculation example")}</summary><div className="reward-sample-summary"><span>{t("同一步示例 · 可编辑部分合计", "Same-step example · editable subtotal")}</span><strong>{sampleTotal.toFixed(3)}</strong><small>{t("这些数值仅用于教学，不是本次 Run 的测量结果；成功与安全项由平台另算。各项已含后台基础系数，滑块是在其上乘权重。", "These values are illustrative, not measured Run results. The platform adds success and safety terms separately. Each term already includes its backend base coefficient; weights multiply those values.")}</small></div></details>}
    <p className="reward-experiment-hint">{t("实验步骤：只改一项 → 保存并检查 → 训练 → 对比相同场景、预算和种子的运行。关注成功率、最终距离和动作幅度，不能只看总奖励。", "Experiment: change one term → save and check → train → compare Runs with matching scene, budget and seed. Inspect success rate, final distance and action magnitude, alongside reward.")}</p>
    <button className="workbench-guide-action" onClick={onViewSource}>{t("查看奖励如何从状态计算 →", "See how state becomes reward →")}</button>
  </div>;
}
