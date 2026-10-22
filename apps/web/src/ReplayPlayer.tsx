import { forwardRef, useEffect, useRef, useState } from "react";
import { Download, Play, RotateCcw } from "lucide-react";
import { apiBlobData, type RunArtifact, type RunStatus } from "./api";
import { type Locale, useI18n } from "./i18n";

export interface ReplayPlayerProps {
  runId: string;
  artifact: RunArtifact;
}

export interface ReplayPresentation {
  title: string;
  description: string;
  loading: string;
  failure: string;
  retry: string;
  download: string;
  fallbackFilename: string;
}

export function replayPresentation(
  artifactType: string,
  locale: Locale = "zh-CN",
): ReplayPresentation {
  if (locale === "en") {
    if (artifactType === "demo_video") {
      return {
        title: "Demo Recording",
        description: "Authenticated recording produced by this Demo Run.",
        loading: "Loading Demo recording",
        failure: "Could not load the Demo recording",
        retry: "Retry Demo recording",
        download: "Download Demo recording",
        fallbackFilename: "demo-overview.mp4",
      };
    }
    return artifactType === "evaluation_video"
      ? {
          title: "Complete Successful Episode",
          description: "After training, a frozen checkpoint replays a successful episode in the complete robot simulation environment.",
          loading: "Loading evaluation replay",
          failure: "Could not load the evaluation replay",
          retry: "Retry evaluation replay",
          download: "Download evaluation replay",
          fallbackFilename: "successful-evaluation-replay.mp4",
        }
      : {
          title: "Training Recording",
          description: "Camera footage captured during training.",
          loading: "Loading training recording",
          failure: "Could not load the training recording",
          retry: "Retry training recording",
          download: "Download training recording",
          fallbackFilename: "training-overview.mp4",
        };
  }
  if (artifactType === "demo_video") {
    return {
      title: "Demo 录像",
      description: "该录像由当前 Demo Run 生成，并通过鉴权 Artifact 接口读取。",
      loading: "正在加载 Demo 录像",
      failure: "Demo 录像读取失败",
      retry: "重试 Demo 录像",
      download: "下载 Demo 录像",
      fallbackFilename: "demo-overview.mp4",
    };
  }
  return artifactType === "evaluation_video"
    ? {
        title: "成功回合完整场景",
        description: "训练完成后，使用冻结 checkpoint 重放成功回合；画面来自完整机械臂仿真环境。",
        loading: "正在加载训练结果回放",
        failure: "训练结果回放读取失败",
        retry: "重试训练结果回放",
        download: "下载训练结果回放",
        fallbackFilename: "successful-evaluation-replay.mp4",
      }
    : {
        title: "训练录像",
        description: "训练期间由 Camera 采集的过程录像。",
        loading: "正在加载训练过程录像",
        failure: "训练录像读取失败",
        retry: "重试训练录像",
        download: "下载训练录像",
        fallbackFilename: "training-overview.mp4",
      };
}

export function evaluationReplayPlaceholder(
  status: RunStatus,
  artifactsPending: boolean,
  artifactsFailed: boolean,
  locale: Locale = "zh-CN",
): string {
  if (locale === "en") {
    if (["queued", "initializing", "running"].includes(status)) {
      return "Training is in progress. A complete robot replay will appear here after training and policy evaluation if a successful episode is available.";
    }
    if (status === "finalizing") return "Training has ended. Evaluation results and replay are being generated.";
    if (status === "failed" || status === "stopped") return "Training did not complete normally, so no evaluation replay was generated.";
    if (artifactsFailed) return "Could not load evaluation replay information. Please retry.";
    if (artifactsPending) return "Loading evaluation replay information.";
    return "Training is complete; this evaluation did not produce an available successful-episode replay.";
  }
  if (["queued", "initializing", "running"].includes(status)) {
    return "正在训练。训练完成并完成策略评估后，如有成功回合，可在此查看完整机械臂回放。";
  }
  if (status === "finalizing") {
    return "训练已结束，正在生成评估结果与回放。";
  }
  if (status === "failed" || status === "stopped") {
    return "本次训练未正常完成，因此没有生成训练结果回放。";
  }
  if (artifactsFailed) {
    return "训练结果回放信息读取失败，请重试。";
  }
  if (artifactsPending) {
    return "正在读取训练结果回放信息。";
  }
  return "训练已完成；本次评估没有生成可用的成功回合回放。";
}

export function selectReplayArtifacts(artifacts: RunArtifact[]) {
  return {
    trainingVideo: artifacts.find((item) => item.artifact_type === "training_video"),
    demoVideo: artifacts.find((item) => item.artifact_type === "demo_video"),
    evaluationVideo: artifacts.find((item) => item.artifact_type === "evaluation_video"),
  };
}

export type RunVideoMode =
  | "disabled"
  | "live"
  | "finalizing"
  | "preparing"
  | "replay"
  | "unavailable"
  | "error"
  | "lifecycle";

export function runVideoEnabled(snapshot: Record<string, unknown>): boolean {
  const trainingConfig = snapshot.training_config;
  const taskConfig = snapshot.task_config;
  return snapshot.video_enabled === true ||
    (trainingConfig !== null && typeof trainingConfig === "object" &&
      (trainingConfig as Record<string, unknown>).video_enabled === true) ||
    (taskConfig !== null && typeof taskConfig === "object" &&
      (taskConfig as Record<string, unknown>).video_enabled === true);
}

export function runVideoMode(
  videoEnabled: boolean,
  status: RunStatus,
  hasTrainingVideo: boolean,
  artifactsPending: boolean,
  artifactsFailed: boolean,
): RunVideoMode {
  if (status === "running") return "live";
  if (!videoEnabled) return "disabled";
  if (status === "finalizing") return "finalizing";
  if (["completed", "failed", "stopped"].includes(status)) {
    if (hasTrainingVideo) return "replay";
    if (artifactsFailed) return "error";
    if (artifactsPending) return "preparing";
    return "unavailable";
  }
  return "lifecycle";
}

export function replayArtifactPath(runId: string, artifactId: string): string {
  return `/api/v1/runs/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(artifactId)}/download`;
}

export function autoplayNeedsManualPlay(error: unknown): boolean {
  void error;
  return true;
}

export interface ReplayVideoElementProps {
  src: string;
  loop: boolean;
  onError: () => void;
}

export const ReplayVideoElement = forwardRef<HTMLVideoElement, ReplayVideoElementProps>(
  function ReplayVideoElement({ src, loop, onError }, ref) {
    return (
      <video
        ref={ref}
        src={src}
        controls
        autoPlay
        muted
        playsInline
        loop={loop}
        onError={onError}
      />
    );
  },
);

export class ReplayBlobGeneration {
  private generation = 0;
  private objectUrl: string | null = null;

  begin(): number {
    this.generation += 1;
    this.revoke();
    return this.generation;
  }

  install(generation: number, blob: Blob): string | null {
    if (generation !== this.generation) return null;
    this.objectUrl = URL.createObjectURL(blob);
    return this.objectUrl;
  }

  isCurrent(generation: number): boolean {
    return generation === this.generation;
  }

  revoke(): void {
    if (this.objectUrl) URL.revokeObjectURL(this.objectUrl);
    this.objectUrl = null;
  }
}

export function ReplayPlayer({ runId, artifact }: ReplayPlayerProps) {
  const { locale, t } = useI18n();
  const videoRef = useRef<HTMLVideoElement>(null);
  const generations = useRef(new ReplayBlobGeneration());
  const blobRef = useRef<Blob | null>(null);
  const [objectUrl, setObjectUrl] = useState("");
  const presentation = replayPresentation(artifact.artifact_type, locale);
  const [filename, setFilename] = useState(presentation.fallbackFilename);
  const [loading, setLoading] = useState(true);
  const [failure, setFailure] = useState("");
  const [manualPlay, setManualPlay] = useState(false);
  const [loopEnabled, setLoopEnabled] = useState(true);
  const [retryGeneration, setRetryGeneration] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    const generation = generations.current.begin();
    blobRef.current = null;
    setObjectUrl("");
    setLoading(true);
    setFailure("");
    setManualPlay(false);
    void apiBlobData(
      replayArtifactPath(runId, artifact.artifact_id),
      { signal: controller.signal },
    ).then(
      ({ blob, filename: downloadedName }) => {
        if (!generations.current.isCurrent(generation)) return;
        const nextUrl = generations.current.install(generation, blob);
        if (!nextUrl) return;
        blobRef.current = blob;
        setFilename(
          downloadedName
          ?? artifact.path.split("/").pop()
          ?? presentation.fallbackFilename,
        );
        setObjectUrl(nextUrl);
        setLoading(false);
      },
      (error: unknown) => {
        if (controller.signal.aborted || !generations.current.isCurrent(generation)) return;
        setLoading(false);
        setFailure(error instanceof Error ? error.message : presentation.failure);
      },
    );
    return () => {
      controller.abort();
      generations.current.begin();
      blobRef.current = null;
    };
  }, [
    artifact.artifact_id,
    artifact.path,
    presentation.failure,
    presentation.fallbackFilename,
    retryGeneration,
    runId,
  ]);

  useEffect(() => () => generations.current.revoke(), []);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !objectUrl) return;
    void video.play().then(
      () => setManualPlay(false),
      (error: unknown) => {
        if (autoplayNeedsManualPlay(error)) setManualPlay(true);
      },
    );
  }, [objectUrl]);

  async function playManually() {
    const video = videoRef.current;
    if (!video) return;
    try {
      await video.play();
      setManualPlay(false);
    } catch {
      setManualPlay(true);
    }
  }

  async function restart() {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = 0;
    await playManually();
  }

  function download() {
    if (!objectUrl || !blobRef.current) return;
    const anchor = document.createElement("a");
    anchor.href = objectUrl;
    anchor.download = filename;
    anchor.click();
  }

  if (loading) {
    return <div className="replay-state">{presentation.loading}</div>;
  }
  if (failure) {
    return (
      <div className="replay-state replay-error">
        <p role="alert">{presentation.failure}：{failure}</p>
        <button className="secondary" onClick={() => setRetryGeneration((value) => value + 1)}>
          <RotateCcw size={14} /> {presentation.retry}
        </button>
      </div>
    );
  }
  return (
    <section className="replay-player" data-replay-state={manualPlay ? "manual-play" : "ready"}>
      <header className="replay-heading">
        <strong>{presentation.title}</strong>
        <small>{presentation.description}</small>
      </header>
      <ReplayVideoElement
        ref={videoRef}
        src={objectUrl}
        loop={loopEnabled}
        onError={() => setFailure(t("录像文件无法播放", "The recording cannot be played"))}
      />
      <small className="replay-seek-note">{t("可使用播放器时间轴拖动定位；结果导出位于下方“训练结果与导出”。", "Use the player timeline to seek. Result exports are available under Training Results & Export below.")}</small>
      {manualPlay && (
        <button className="primary replay-manual-play" onClick={() => void playManually()}>
          <Play size={14} /> {t("点击播放", "Play")}
        </button>
      )}
      <div className="replay-controls">
        <label>
          <input
            type="checkbox"
            checked={loopEnabled}
            onChange={(event) => setLoopEnabled(event.target.checked)}
          />
          {t("循环播放", "Loop")}
        </label>
        <button className="secondary" onClick={() => void restart()}>
          <RotateCcw size={14} /> {t("从头播放", "Restart")}
        </button>
        <button className="secondary" onClick={download}>
          <Download size={14} /> {presentation.download}
        </button>
      </div>
    </section>
  );
}
