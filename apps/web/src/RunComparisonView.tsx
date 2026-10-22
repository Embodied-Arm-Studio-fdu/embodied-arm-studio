import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import ReactECharts from "echarts-for-react";
import {
  Activity,
  Check,
  CircleAlert,
  Film,
  Gauge,
  Pause,
  Play,
  RotateCcw,
} from "lucide-react";
import {
  api,
  apiBlobData,
  type MetricEvent,
  type RunArtifact,
  type RunComparisonResponse,
  type RunRecord,
} from "./api";
import {
  RUN_METRIC_DEFINITIONS,
  metricSeriesData,
  missingSimulationMetricDefinitions,
  productionEvaluation,
  runMetricHistory,
  runMetricText,
  visibleRunMetricDefinitions,
  type RunMetricKey,
} from "./runMetrics";
import { type Locale, translateUiText, useI18n } from "./i18n";
import { runStatusPresentation } from "./runPresentation";

interface Props {
  comparison: RunComparisonResponse;
  runs: RunRecord[];
}

export interface EvidenceChannelSummary {
  available: boolean;
  count: number;
}

export function displayEvidenceSummary(
  summary: EvidenceChannelSummary,
  suffix: string | { zh: string; en: string },
  locale: Locale = "zh-CN",
): string {
  if (!summary.available) return locale === "en" ? "Not recorded" : "未记录";
  const localizedSuffix = typeof suffix === "string"
    ? (locale === "en" ? translateUiText(suffix) : suffix)
    : suffix[locale === "en" ? "en" : "zh"];
  return `${summary.count}${localizedSuffix}`;
}

export function summarizeDiagnosticChannel(
  events: MetricEvent[],
  channel: "collision" | "joint_limits",
): EvidenceChannelSummary {
  const values = events
    .map((event) => channel === "collision"
      ? event.payload.collision?.detected
      : event.payload.joint_limits?.violated)
    .filter((value): value is boolean => typeof value === "boolean");
  return {
    available: values.length > 0,
    count: values.filter(Boolean).length,
  };
}

export function summarizeFailureChannel(events: MetricEvent[]): EvidenceChannelSummary {
  const errorCount = events.filter((event) => event.type === "error").length;
  const hasFailureRate = events.some(
    (event) => typeof event.payload.metrics?.failure_rate === "number",
  );
  return {
    available: errorCount > 0 || hasFailureRate,
    count: errorCount,
  };
}

const RUN_COLORS = ["#277b69", "#e05a3f"] as const;

const CONFIG_FIELDS = [
  ["environment_type", "环境", "Environment"],
  ["task_type", "任务", "Task"],
  ["algorithm", "训练算法", "Training algorithm"],
  ["training_profile", "训练规格", "Training profile"],
  ["config_version", "配置版本", "Configuration version"],
  ["expected_training_steps", "计划训练步数", "Planned training steps"],
  ["video_enabled", "训练录像", "Training recording"],
] as const;

function displayValue(value: unknown, locale: Locale = "zh-CN"): string {
  if (value === null || value === undefined || value === "") return locale === "en" ? "Not recorded" : "未记录";
  if (typeof value === "boolean") return value ? (locale === "en" ? "On" : "开启") : (locale === "en" ? "Off" : "关闭");
  if (typeof value === "number") return Number.isInteger(value)
    ? value.toLocaleString(locale)
    : value.toLocaleString(locale, { maximumFractionDigits: 5 });
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

function valuesDiffer(values: unknown[]): boolean {
  return new Set(values.map((value) => JSON.stringify(value))).size > 1;
}

function comparisonDuration(run: RunRecord, locale: Locale = "zh-CN"): string {
  if (!run.started_at || !run.ended_at) return locale === "en" ? "Not recorded" : "未记录";
  const milliseconds = Date.parse(run.ended_at) - Date.parse(run.started_at);
  if (!Number.isFinite(milliseconds) || milliseconds < 0) return locale === "en" ? "Not recorded" : "未记录";
  const seconds = Math.round(milliseconds / 1000);
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m ${seconds % 60}s`;
}

function comparisonEvaluationValue(run: RunRecord, key: string, locale: Locale = "zh-CN"): string {
  const evaluation = productionEvaluation(run.metrics_summary);
  if (!evaluation) return locale === "en" ? "Not recorded" : "未记录";
  const value = (evaluation as unknown as Record<string, unknown>)[key];
  if (typeof value !== "number" || !Number.isFinite(value)) return locale === "en" ? "Not recorded" : "未记录";
  if (["success_rate", "wilson_95_lower_bound"].includes(key)) {
    return `${(value * 100).toFixed(2)}%`;
  }
  if (["mean_final_distance", "mean_object_displacement"].includes(key)) {
    return `${Number(value.toFixed(4))} m`;
  }
  return displayValue(value, locale);
}

export function runComparisonMetricRows(runs: RunRecord[], locale: Locale = "zh-CN") {
  return [
    { key: "success_rate", label: locale === "en" ? "Formal evaluation success rate" : "正式评估成功率", values: runs.map((run) => comparisonEvaluationValue(run, "success_rate", locale)) },
    { key: "wilson_95_lower_bound", label: locale === "en" ? "Wilson 95% lower bound" : "Wilson 95% 下界", values: runs.map((run) => comparisonEvaluationValue(run, "wilson_95_lower_bound", locale)) },
    { key: "mean_final_distance", label: locale === "en" ? "Mean final distance" : "平均最终距离", values: runs.map((run) => comparisonEvaluationValue(run, "mean_final_distance", locale)) },
    { key: "mean_object_displacement", label: locale === "en" ? "Mean object displacement" : "平均物体位移", values: runs.map((run) => comparisonEvaluationValue(run, "mean_object_displacement", locale)) },
    { key: "training_duration", label: locale === "en" ? "Training duration" : "训练耗时", values: runs.map((run) => comparisonDuration(run, locale)) },
  ];
}

function VideoSource({
  runId,
  artifact,
  videoRef,
  onTimeUpdate,
}: {
  runId: string;
  artifact?: RunArtifact;
  videoRef: React.RefObject<HTMLVideoElement>;
  onTimeUpdate?: () => void;
}) {
  const { t } = useI18n();
  const [source, setSource] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    if (!artifact) return;
    const controller = new AbortController();
    let objectUrl = "";
    setError("");
    void apiBlobData(
      `/api/v1/runs/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(artifact.artifact_id)}/download`,
      { signal: controller.signal },
    ).then(
      ({ blob }) => {
        objectUrl = URL.createObjectURL(blob);
        setSource(objectUrl);
      },
      (caught: unknown) => {
        if (!controller.signal.aborted) {
          setError(caught instanceof Error ? caught.message : t("录像读取失败", "Could not load recording"));
        }
      },
    );
    return () => {
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [artifact, runId]);

  if (!artifact) return <div className="comparison-video-empty">{t("本次运行没有训练录像", "No training recording is available for this run")}</div>;
  if (error) return <div className="comparison-video-empty error">{error}</div>;
  if (!source) return <div className="comparison-video-empty">{t("正在读取训练录像", "Loading training recording")}</div>;
  return <video ref={videoRef} src={source} muted playsInline preload="metadata" onTimeUpdate={onTimeUpdate} />;
}

function LinkedVideos({ runs, artifacts }: { runs: RunRecord[]; artifacts: RunArtifact[][] }) {
  const { t } = useI18n();
  const firstRef = useRef<HTMLVideoElement>(null);
  const secondRef = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const videos = [firstRef, secondRef];
  const trainingVideos = artifacts.map((items) => items.find((item) => item.artifact_type === "training_video"));
  const bothAvailable = trainingVideos.every(Boolean);

  async function togglePlayback() {
    if (!bothAvailable) return;
    if (playing) {
      videos.forEach((ref) => ref.current?.pause());
      setPlaying(false);
      return;
    }
    const start = Math.max(firstRef.current?.currentTime ?? 0, secondRef.current?.currentTime ?? 0);
    videos.forEach((ref) => {
      if (ref.current) ref.current.currentTime = start;
    });
    const results = await Promise.allSettled(videos.map((ref) => ref.current?.play()));
    setPlaying(results.some((result) => result.status === "fulfilled"));
  }

  function restart() {
    videos.forEach((ref) => {
      if (ref.current) ref.current.currentTime = 0;
    });
    setProgress(0);
  }

  function syncFromFirst() {
    const first = firstRef.current;
    const second = secondRef.current;
    if (!first) return;
    if (first.duration) setProgress((first.currentTime / first.duration) * 100);
    if (second && Math.abs(second.currentTime - first.currentTime) > 0.2) {
      second.currentTime = Math.min(first.currentTime, second.duration || first.currentTime);
    }
    if (first.ended) setPlaying(false);
  }

  function seek(value: number) {
    setProgress(value);
    videos.forEach((ref) => {
      const video = ref.current;
      if (video?.duration) video.currentTime = (value / 100) * video.duration;
    });
  }

  return (
    <section className="comparison-videos" aria-label={t("并排录像对比", "Side-by-side recording comparison")}>
      <div className="comparison-section-title">
        <div><Film size={17} /><span><strong>{t("并排录像", "Side-by-side recordings")}</strong><small>{t("按各自录像进度联动，不能证明物理时间一致", "Synchronized by recording progress; this does not establish equal physical time")}</small></span></div>
        <span className={bothAvailable ? "evidence-ready" : "evidence-missing"}>
          {bothAvailable ? t("可对比", "Ready to compare") : t("证据不足", "Insufficient evidence")}
        </span>
      </div>
      <div className="comparison-video-grid">
        {runs.map((run, index) => (
          <div key={run.run_id}>
            <span>{t("运行", "Run")} {index + 1} · {run.run_id.slice(-8)}</span>
            <VideoSource
              runId={run.run_id}
              artifact={trainingVideos[index]}
              videoRef={videos[index]}
              onTimeUpdate={index === 0 ? syncFromFirst : undefined}
            />
          </div>
        ))}
      </div>
      <div className="comparison-video-controls">
        <button className="secondary" disabled={!bothAvailable} onClick={() => void togglePlayback()}>
          {playing ? <Pause size={15} /> : <Play size={15} />}{playing ? t("联动暂停", "Pause both") : t("联动播放", "Play both")}
        </button>
        <button className="secondary" disabled={!bothAvailable} onClick={restart} title={t("回到录像起点", "Return to the start of both recordings")}>
          <RotateCcw size={15} /> {t("从头对比", "Restart comparison")}
        </button>
        <input
          aria-label={t("并排录像进度", "Side-by-side recording progress")}
          type="range"
          min="0"
          max="100"
          value={progress}
          disabled={!bothAvailable}
          onChange={(event) => seek(Number(event.target.value))}
        />
      </div>
    </section>
  );
}

export function RunComparisonView({ comparison, runs }: Props) {
  const { locale, t } = useI18n();
  const [metricKey, setMetricKey] = useState<RunMetricKey>("average_reward");
  const runIds = comparison.run_ids.slice(0, 2);
  const events = useQuery({
    queryKey: ["comparison-events", runIds],
    queryFn: () => Promise.all(runIds.map((id) => api<MetricEvent[]>(`/api/v1/runs/${encodeURIComponent(id)}/events?after_sequence=0`))),
    enabled: runIds.length === 2,
  });
  const artifacts = useQuery({
    queryKey: ["comparison-artifacts", runIds],
    queryFn: () => Promise.all(runIds.map((id) => api<RunArtifact[]>(`/api/v1/runs/${encodeURIComponent(id)}/artifacts`))),
    enabled: runIds.length === 2,
  });
  const histories = useMemo(
    () => (events.data ?? [[], []]).map((items) => runMetricHistory(items)),
    [events.data],
  );
  const comparisonMetricDefinitions = visibleRunMetricDefinitions(
    histories.flat(),
  );
  const missingSimulationMetrics = missingSimulationMetricDefinitions(
    histories.flat(),
  );
  const orderedRuns = runIds.map((id) => runs.find((run) => run.run_id === id)).filter((run): run is RunRecord => Boolean(run));
  const configRows = [
    ...CONFIG_FIELDS.map(([key, chineseLabel, englishLabel]) => ({
      key,
      label: t(chineseLabel, englishLabel),
      values: orderedRuns.map((run) => run.project_config_snapshot[key]),
    })),
    ...Array.from(new Set(orderedRuns.flatMap((run) => Object.keys(run.parameters ?? {})))).map((key) => ({
      key: `parameter.${key}`,
      label: key === "seed" ? t("随机种子", "Random seed") : key,
      values: orderedRuns.map((run) => run.parameters?.[key]),
    })),
  ];
  const evaluationRows = runComparisonMetricRows(orderedRuns, locale);
  const selectedMetric = RUN_METRIC_DEFINITIONS.find((definition) => definition.key === metricKey)!;
  const chartOption = {
    animation: false,
    tooltip: { trigger: "axis" },
    legend: { top: 2, right: 8, textStyle: { fontSize: 10 } },
    grid: { left: 18, right: 18, top: 42, bottom: 34, containLabel: true },
    xAxis: { type: "value", name: t("更新次数", "Updates"), minInterval: 1 },
    yAxis: { type: "value", scale: true, name: runMetricText(selectedMetric, "label", locale) },
    series: histories.map((history, index) => ({
      name: `${t("运行", "Run")} ${index + 1} · ${runIds[index]?.slice(-8) ?? "--"}`,
      type: "line",
      showSymbol: history.length < 30,
      connectNulls: false,
      lineStyle: { width: 2, color: RUN_COLORS[index] },
      itemStyle: { color: RUN_COLORS[index] },
      data: metricSeriesData(history, metricKey),
    })),
  };
  const artifactSets = artifacts.data ?? [[], []];
  const armSamples = (events.data ?? [[], []]).map((items) => items.filter((event) => event.payload.arm?.joint_positions?.length).length);
  const collisionEvidence = (events.data ?? [[], []]).map((items) => summarizeDiagnosticChannel(items, "collision"));
  const jointLimitEvidence = (events.data ?? [[], []]).map((items) => summarizeDiagnosticChannel(items, "joint_limits"));
  const failureEvidence = (events.data ?? [[], []]).map(summarizeFailureChannel);
  const metricCounts = histories.map((history) => history.length);
  const evaluationAvailable = orderedRuns.map(
    (run) => productionEvaluation(run.metrics_summary) !== null,
  );
  const evidenceRows = [
    { label: t("训练指标曲线", "Training metric curves"), values: metricCounts.map((count) => `${count} ${t("点", "points")}`), ready: metricCounts.every((count) => count > 0) },
    { label: t("训练录像", "Training recordings"), values: artifactSets.map((items) => `${items.filter((item) => item.artifact_type === "training_video").length} ${t("个", "recordings")}`), ready: artifactSets.every((items) => items.some((item) => item.artifact_type === "training_video")) },
    { label: t("关节状态序列", "Joint state sequences"), values: armSamples.map((count) => `${count} ${t("帧", "frames")}`), ready: armSamples.every((count) => count > 0) },
    { label: t("碰撞诊断", "Collision diagnostics"), values: collisionEvidence.map((item) => displayEvidenceSummary(item, { zh: " 次触发", en: " triggers" }, locale)), ready: collisionEvidence.every((item) => item.available) },
    { label: t("关节限制诊断", "Joint-limit diagnostics"), values: jointLimitEvidence.map((item) => displayEvidenceSummary(item, { zh: " 次触发", en: " violations" }, locale)), ready: jointLimitEvidence.every((item) => item.available) },
    { label: t("失败事件", "Failure events"), values: failureEvidence.map((item) => displayEvidenceSummary(item, { zh: " 个事件", en: " events" }, locale)), ready: failureEvidence.every((item) => item.available) },
    { label: t("证据评估", "Evaluation evidence"), values: evaluationAvailable.map((available) => available ? t("1 份", "1 report") : t("未记录", "Not recorded")), ready: evaluationAvailable.every(Boolean) },
  ];
  const finalRewards = orderedRuns.map((run) => run.metrics_summary.average_reward).filter((value): value is number => typeof value === "number");
  const teachingConclusion = finalRewards.length === 2
    ? finalRewards[1] > finalRewards[0]
      ? t(`运行 2 的末次平均奖励高于运行 1（${displayValue(finalRewards[1])} 对 ${displayValue(finalRewards[0])}）。这只说明本次记录结果，不代表算法普遍更优。`, `Run 2 has a higher final average reward than Run 1 (${displayValue(finalRewards[1], locale)} vs. ${displayValue(finalRewards[0], locale)}). This describes only these recorded results and does not establish general algorithm superiority.`)
      : t(`运行 1 的末次平均奖励不低于运行 2（${displayValue(finalRewards[0])} 对 ${displayValue(finalRewards[1])}）。仍需更多随机种子验证稳定性。`, `Run 1's final average reward is no lower than Run 2's (${displayValue(finalRewards[0], locale)} vs. ${displayValue(finalRewards[1], locale)}). More random seeds are required to assess stability.`)
    : t("当前没有两组完整的末次平均奖励，暂不生成优劣结论。", "Two complete final average rewards are not available, so no ranking is produced.");

  return (
    <section className="run-comparison-view" aria-label={t("运行可视化对比", "Run visualization comparison")}>
      <header>
        <div>
          <span className="eyebrow">{t("双运行证据", "Two-run evidence")}</span>
          <h2>{t("可视化对比", "Visualization comparison")}</h2>
          <p>{t("只比较服务端冻结配置、实际训练事件和已登记制品。空值保持为空，不按单次结果推断泛化能力。", "Compare only server-frozen configuration, actual training events, and registered artifacts. Empty values remain empty; a single run does not establish generalization.")}</p>
        </div>
        <div className="comparison-run-legend">
          {orderedRuns.map((run, index) => <span key={run.run_id}><i style={{ background: RUN_COLORS[index] }} />{t("运行", "Run")} {index + 1}<small>{run.run_id.slice(-8)}</small></span>)}
        </div>
      </header>

      <section className="comparison-summary-band">
        {orderedRuns.map((run, index) => (
          <div key={run.run_id}>
            <span>{t("运行", "Run")} {index + 1}</span>
            <strong>{runStatusPresentation(run.status, locale).label}</strong>
            <small>{run.training_steps.toLocaleString(locale)} {t("步", "steps")} · {run.update_steps} {t("次更新", "updates")}</small>
          </div>
        ))}
        <div className="comparison-teaching-conclusion">
          <Activity size={18} />
          <span><strong>{t("本次证据说明", "What this evidence shows")}</strong><small>{teachingConclusion}</small></span>
        </div>
      </section>

      <div className="comparison-layout">
        <section className="comparison-main">
          <section className="comparison-config">
            <div className="comparison-section-title"><div><Activity size={17} /><span><strong>{t("正式评估与耗时", "Formal evaluation and duration")}</strong><small>{t("缺失评估保持未记录，不按 completed 推断", "Missing evaluations remain unrecorded; completed status is not treated as proof")}</small></span></div></div>
            <div className="comparison-table">
              <div className="comparison-table-head"><span>{t("指标", "Metric")}</span><span>{t("运行", "Run")} 1</span><span>{t("运行", "Run")} 2</span></div>
              {evaluationRows.map((row) => (
                <div className={valuesDiffer(row.values) ? "different" : ""} key={row.key}>
                  <strong>{row.label}</strong>
                  {row.values.map((value, index) => <span key={index}>{value}</span>)}
                </div>
              ))}
            </div>
          </section>
          <section className="comparison-config">
            <div className="comparison-section-title"><div><Gauge size={17} /><span><strong>{t("训练配置", "Training configuration")}</strong><small>{t("变化项已高亮", "Changed values are highlighted")}</small></span></div></div>
            <div className="comparison-table">
              <div className="comparison-table-head"><span>{t("配置项", "Configuration")}</span><span>{t("运行", "Run")} 1</span><span>{t("运行", "Run")} 2</span></div>
              {configRows.map((row) => (
                <div className={valuesDiffer(row.values) ? "different" : ""} key={row.key}>
                  <strong>{row.label}</strong>
                  {row.values.map((value, index) => <span key={index}>{displayValue(value, locale)}</span>)}
                </div>
              ))}
            </div>
          </section>

          <section className="comparison-metrics">
            <div className="comparison-section-title"><div><Activity size={17} /><span><strong>{t("训练曲线", "Training curves")}</strong><small>{t("按 Worker 上报的更新次数对齐", "Aligned by update counts reported by the Worker")}</small></span></div></div>
            <div className="metric-selector" role="tablist" aria-label={t("对比指标", "Comparison metrics")}>
              {comparisonMetricDefinitions.map((definition) => (
                <button title={`${runMetricText(definition, "description", locale)}; ${t("单位", "Unit")}: ${runMetricText(definition, "unit", locale)}`} key={definition.key} className={metricKey === definition.key ? "active" : ""} onClick={() => setMetricKey(definition.key)}>{runMetricText(definition, "label", locale)}</button>
              ))}
            </div>
            {events.isLoading ? <div className="comparison-loading">{t("正在读取两次运行的指标事件", "Reading metric events for both runs")}</div> : events.isError ? <div className="comparison-loading error">{t("指标事件读取失败", "Could not read metric events")}</div> : <div className="comparison-chart"><ReactECharts option={chartOption} style={{ width: "100%", height: 330 }} /></div>}
            {missingSimulationMetrics.length > 0 && <p className="comparison-metric-gaps">{t("接口缺口：", "Missing interface fields: ")}{missingSimulationMetrics.map((item) => runMetricText(item, "label", locale)).join(locale === "en" ? ", " : "、")}{t("未由两次 Run 的 Event 输出，不作推算。", " are not emitted by either Run's Events and are not inferred.")}</p>}
          </section>

          {orderedRuns.length === 2 && <LinkedVideos runs={orderedRuns} artifacts={artifactSets} />}
        </section>

        <aside className="comparison-evidence-matrix">
            <div className="comparison-section-title"><div><Check size={17} /><span><strong>{t("证据完整度", "Evidence completeness")}</strong><small>{t("仅显示实际采集能力，缺失项不作推断", "Shows collected evidence only; missing items are not inferred")}</small></span></div></div>
          {evidenceRows.map((row) => (
            <div className="evidence-matrix-row" key={row.label}>
              <span>{row.ready ? <Check size={14} /> : <CircleAlert size={14} />}{row.label}</span>
              <div>{row.values.map((value, index) => <small key={index}>{t("运行", "Run")} {index + 1}<strong>{value}</strong></small>)}</div>
              {!row.ready && <p>{t("当前 Run 未完整采集，不能进行可信对比。", "This Run was not fully collected, so a reliable comparison is not possible.")}</p>}
            </div>
          ))}
          <div className="comparison-boundary-note">
            <strong>{t("行为证据边界", "Behavior evidence boundary")}</strong>
            <p>{t("末端三维轨迹、关节速度、力矩、动作裁剪和关节限制需要 Worker 事件显式上报。当前页面不会从录像、奖励曲线或关节位置推算这些数据。", "End-effector trajectories, joint velocity, torque, action clipping, and joint limits require explicit Worker events. This page does not infer them from recordings, reward curves, or joint positions.")}</p>
          </div>
          {comparison.limitations.map((item) => <small className="comparison-limitation" key={item}>{item}</small>)}
        </aside>
      </div>
    </section>
  );
}
