import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import * as echarts from "echarts";
import { Code2, Download, FileJson, Film, LineChart, RefreshCw } from "lucide-react";
import { api, apiBlob, type CanonicalEvaluationReport, type RunArtifact, type RunRecord } from "./api";
import { normalizeRunEventsResponse } from "./runEvents";
import { buildRunMetricsCsv, metricSeriesData, runMetricHistory, runMetricText, visibleRunMetricDefinitions } from "./runMetrics";
import { runDisplayName } from "./runLabel";
import { runStatusLabel } from "./runPresentation";
import { useI18n } from "./i18n";

function saveBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function frozenRewardSource(run: RunRecord): string | null {
  const extension = run.project_config_snapshot.reward_extension;
  if (!extension || typeof extension !== "object") return null;
  const source = (extension as Record<string, unknown>).source;
  return typeof source === "string" && source.trim() ? source : null;
}

export function artifactCategory(type: string) {
  if (type.includes("video")) return "video";
  if (type.includes("checkpoint") || type.includes("model")) return "model";
  if (type.includes("log")) return "log";
  return "data";
}

export function RunArtifactsPage() {
  const { locale, t } = useI18n();
  const [runId, setRunId] = useState("");
  const [category, setCategory] = useState("all");
  const [metricKey, setMetricKey] = useState("average_reward");
  const [busy, setBusy] = useState("");
  const [feedback, setFeedback] = useState("");
  const [failed, setFailed] = useState(false);
  const runs = useQuery({ queryKey: ["artifact-center-runs"], queryFn: () => api<RunRecord[]>("/api/v1/runs") });
  const selected = (runs.data ?? []).find((run) => run.run_id === runId) ?? runs.data?.[0];
  const artifacts = useQuery({ queryKey: ["artifact-center-files", selected?.run_id], queryFn: () => api<RunArtifact[]>(`/api/v1/runs/${encodeURIComponent(selected!.run_id)}/artifacts`), enabled: Boolean(selected) });
  const events = useQuery({ queryKey: ["artifact-center-events", selected?.run_id], queryFn: async () => normalizeRunEventsResponse(await api<unknown>(`/api/v1/runs/${encodeURIComponent(selected!.run_id)}/events?after_sequence=0`)), enabled: Boolean(selected) });
  const reports = useQuery({ queryKey: ["artifact-center-reports", selected?.run_id], queryFn: () => api<CanonicalEvaluationReport[]>(`/api/v1/evaluations?run_id=${encodeURIComponent(selected!.run_id)}`), enabled: Boolean(selected) });
  const points = runMetricHistory(events.data ?? []);
  const metrics = visibleRunMetricDefinitions(points).filter((definition) => points.some((point) => typeof point.values[definition.key] === "number"));
  const metric = metrics.find((item) => item.key === metricKey) ?? metrics[0];
  const rewardSource = selected ? frozenRewardSource(selected) : null;
  const chartOption: echarts.EChartsOption = {
    animation: false,
    backgroundColor: "#fff",
    grid: { left: 80, right: 30, bottom: 50, top: 35 },
    xAxis: { type: "value", name: t("更新", "Update"), minInterval: 1 },
    yAxis: { type: "value", scale: true },
    series: metric ? [{ name: runMetricText(metric, "label", locale), type: "line", data: metricSeriesData(points, metric.key), showSymbol: true, connectNulls: false, itemStyle: { color: "#287e6f" } }] : [],
  };
  function chartSvg(): string {
    const renderer = echarts.init(null, undefined, { renderer: "svg", ssr: true, width: 1000, height: 360 });
    try { renderer.setOption(chartOption); return renderer.renderToSVGString(); }
    finally { renderer.dispose(); }
  }
  function json(value: unknown, name: string) { saveBlob(new Blob([JSON.stringify(value, null, 2)], { type: "application/json;charset=utf-8" }), `${selected!.run_id}-${name}.json`); }
  async function download(artifact: RunArtifact) {
    if (!selected || busy) return;
    setBusy(artifact.artifact_id); setFeedback(""); setFailed(false);
    try {
      const response = await apiBlob(`/api/v1/runs/${encodeURIComponent(selected.run_id)}/artifacts/${encodeURIComponent(artifact.artifact_id)}/download`);
      saveBlob(await response.blob(), artifact.path.split("/").pop() ?? "artifact");
      setFeedback(t("下载已开始", "Download started"));
    } catch (error) { setFailed(true); setFeedback(error instanceof Error ? error.message : "下载失败"); }
    finally { setBusy(""); }
  }
  async function exportCanonical() {
    if (!selected) return;
    setBusy("canonical"); setFeedback(""); setFailed(false);
    try {
      const response = await apiBlob("/api/v1/runs/export", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ run_ids: [selected.run_id], format: "json" }) });
      const filename = (response.headers.get("content-disposition") ?? "").match(/filename="?([^";]+)"?/)?.[1] ?? `${selected.run_id}-evidence.json`;
      saveBlob(await response.blob(), filename);
      setFeedback(t("下载已开始", "Download started"));
    } catch (error) { setFailed(true); setFeedback(error instanceof Error ? error.message : "导出失败"); }
    finally { setBusy(""); }
  }
  async function exportChart(type: "png" | "svg") {
    if (!selected || !metric) return;
    if (type === "svg") {
      saveBlob(new Blob([chartSvg()], { type: "image/svg+xml;charset=utf-8" }), `${selected.run_id}-${metric.key}.svg`);
      return;
    }
    try {
      const response = await apiBlob(`/api/v1/runs/${encodeURIComponent(selected.run_id)}/charts/${metric.key}.png`);
      saveBlob(await response.blob(), `${selected.run_id}-${metric.key}.png`);
    } catch (error) { setFailed(true); setFeedback(error instanceof Error ? error.message : "PNG export failed"); }
  }
  if (runs.isLoading) return <div className="loading">{t("读取运行制品…", "Loading artifacts…")}</div>;
  if (runs.isError) return <div className="loading error">{t("运行记录读取失败", "Unable to load Runs")}<button onClick={() => void runs.refetch()}>{t("重试", "Retry")}</button></div>;
  if (!selected) return <main className="evaluation-empty"><FileJson size={30} /><h1>{t("暂无运行制品", "No Run artifacts yet")}</h1></main>;
  return <main className="artifact-center">
    <header className="operations-header"><div><span className="eyebrow">{t("Run Artifact", "RUN ARTIFACTS")}</span><h1>{t("运行制品", "Run artifacts")}</h1><p>{t("代码、权重、运行录像、日志、指标曲线与评估报告", "Code, weights, recordings, logs, metric charts and evaluation reports")}</p></div><button className="secondary" onClick={() => { void artifacts.refetch(); void events.refetch(); void reports.refetch(); }}><RefreshCw size={16} /> {t("刷新", "Refresh")}</button></header>
    <section className="artifact-center-context"><label>{t("运行记录", "Run")}<select value={selected.run_id} disabled={Boolean(busy)} onChange={(event) => { setRunId(event.target.value); setFeedback(""); setFailed(false); }} data-tour="artifact-run-select">{(runs.data ?? []).map((run) => <option key={run.run_id} value={run.run_id}>{runDisplayName(run, locale)} · {runStatusLabel(run.status, locale)} · {run.run_id.slice(-8)}</option>)}</select></label><code>{selected.run_id}</code></section>
    <section className="artifact-center-exports"><h2>{t("实验结果导出", "Experiment exports")}</h2><div className="artifact-export-actions">
      <button className="secondary" disabled={Boolean(busy) || selected.status !== "completed"} onClick={() => void exportCanonical()} data-tour="json-export"><FileJson size={16} /> {t("权威运行 JSON", "Canonical Run JSON")}</button>
      <button className="secondary" onClick={() => json(selected.project_config_snapshot, "config")}><Download size={16} /> {t("冻结配置 / 场景", "Frozen config / scene")}</button>
      <button className="secondary" disabled={!rewardSource} title={!rewardSource ? t("此 Run 未保存可下载源码，不能从 hash 重建", "No source was frozen; a hash cannot reconstruct code") : undefined} onClick={() => rewardSource && saveBlob(new Blob([rewardSource], { type: "text/x-python;charset=utf-8" }), `${selected.run_id}-reward.py`)}><Code2 size={16} /> {t("本次奖励源码", "Frozen reward code")}</button>
      <button className="secondary" disabled={!points.length || events.isError} onClick={() => saveBlob(new Blob([buildRunMetricsCsv(selected, points)], { type: "text/csv;charset=utf-8" }), `${selected.run_id}-metrics.csv`)}><Download size={16} /> {t("指标 CSV", "Metrics CSV")}</button>
      <button className="secondary" disabled={!events.data || events.isError} onClick={() => json(events.data, "events")}><Download size={16} /> {t("运行事件 JSON", "Run events JSON")}</button>
      <button className="secondary" disabled={!reports.data?.length || reports.isError} onClick={() => json(reports.data, "evaluations")}><Download size={16} /> {t("评估报告 JSON", "Evaluation reports JSON")}</button>
    </div>{feedback && <p className={failed ? "action-feedback error" : "action-feedback success"} role={failed ? "alert" : "status"}>{feedback}</p>}{reports.isError && <p role="alert">{t("评估报告读取失败，可刷新重试", "Evaluation reports unavailable; refresh to retry")}</p>}</section>
    <section className="artifact-center-files"><header><h2>{t("实际运行文件", "Registered files")}</h2><select aria-label={t("文件分类", "File category")} value={category} onChange={(event) => setCategory(event.target.value)}>{[["all", "全部", "All"], ["video", "视频 MP4", "Video MP4"], ["model", "模型权重", "Model weights"], ["log", "日志", "Logs"], ["data", "配置与指标", "Config and metrics"]].map(([value, zh, en]) => <option key={value} value={value}>{t(zh, en)}</option>)}</select></header>
      {artifacts.isLoading ? <p>{t("读取文件目录…", "Loading files…")}</p> : artifacts.isError ? <p role="alert">{t("制品目录读取失败，请刷新重试", "Artifact catalog unavailable; refresh to retry")}</p> : <div className="artifact-file-table">{(artifacts.data ?? []).filter((file) => category === "all" || artifactCategory(file.artifact_type) === category).map((file) => <div key={file.artifact_id}>{artifactCategory(file.artifact_type) === "video" ? <Film size={18} /> : <FileJson size={18} />}<span><strong>{file.path}</strong><small>{file.artifact_type} · SHA256 {file.sha256?.slice(0, 16) ?? "--"}</small></span><button className="secondary" disabled={Boolean(busy)} onClick={() => void download(file)}><Download size={14} /> {busy === file.artifact_id ? t("下载中", "Downloading") : t("下载", "Download")}</button></div>)}{!(artifacts.data ?? []).some((file) => category === "all" || artifactCategory(file.artifact_type) === category) && <p>{t("没有该分类的已登记文件。录像只能下载实际生成的 MP4，不能补录历史直播。", "No registered files in this category. Recordings are available only when an MP4 was generated; past streams cannot be recreated.")}</p>}</div>}
    </section>
    <section className="artifact-center-chart"><header><h2><LineChart size={17} /> {t("指标曲线", "Metric charts")}</h2><div className="artifact-export-actions"><select aria-label={t("选择曲线", "Choose metric")} value={metric?.key ?? ""} onChange={(event) => setMetricKey(event.target.value)}>{metrics.map((item) => <option key={item.key} value={item.key}>{runMetricText(item, "label", locale)}</option>)}</select><button className="secondary" disabled={!metric} onClick={() => void exportChart("png")}><Download size={14} /> PNG</button><button className="secondary" disabled={!metric} onClick={() => void exportChart("svg")}><Download size={14} /> SVG</button></div></header>
      {events.isError ? <p role="alert">{t("指标事件读取失败", "Metric events unavailable")}</p> : metric ? <div className="artifact-chart-preview" translate="no" dangerouslySetInnerHTML={{ __html: chartSvg() }} /> : <p>{t("没有已记录的数值曲线", "No numeric metric history recorded")}</p>}
    </section>
  </main>;
}
