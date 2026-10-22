import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  api,
  type RunStatus,
  type StreamDescriptor,
  type StreamStatus,
} from "./api";
import {
  boundedRetryDelay,
  createWhepEffectCleanup,
  negotiateWhep,
  proxiedWhepUrl,
  WhepNegotiationSessionError,
  WHEP_NEGOTIATION_RETRY_DELAYS_MS,
  whepReconnectDelay,
  type WhepReconnectReason,
} from "./whep";
import { useI18n } from "./i18n";

const TERMINAL_RUN_STATUSES = new Set<RunStatus>(["completed", "failed", "stopped"]);
const MAX_WHEP_RETRIES = WHEP_NEGOTIATION_RETRY_DELAYS_MS.length;
const DISCONNECTED_GRACE_MS = 3_000;

export type VideoPresentation =
  | "starting"
  | "connecting"
  | "playing"
  | "reconnecting"
  | "unavailable"
  | "closing"
  | "closed";

const VIDEO_LABELS: Record<VideoPresentation, string> = {
  starting: "视频正在启动，请稍候",
  connecting: "正在连接",
  playing: "正在播放",
  reconnecting: "正在重连",
  unavailable: "视频暂时不可用，训练仍在继续",
  closing: "视频正在关闭",
  closed: "训练已结束，视频已关闭",
};

const EN_VIDEO_LABELS: Record<VideoPresentation, string> = {
  starting: "Video is starting",
  connecting: "Connecting",
  playing: "Playing",
  reconnecting: "Reconnecting",
  unavailable: "Video is temporarily unavailable; training continues",
  closing: "Video is closing",
  closed: "Training has ended and the video is closed",
};

export function descriptorPresentation(
  runStatus: RunStatus,
  descriptorStatus?: StreamStatus,
  descriptorFailed = false,
): VideoPresentation {
  if (TERMINAL_RUN_STATUSES.has(runStatus)) return "closed";
  if (runStatus === "stopping" || runStatus === "finalizing") return "closing";
  if (runStatus !== "running") return "starting";
  if (descriptorFailed) return "unavailable";
  switch (descriptorStatus) {
    case "READY":
      return "connecting";
    case "ERROR":
    case "STOPPED":
      return "unavailable";
    case "STOPPING":
      return "closing";
    case "CREATED":
    case "STARTING":
    default:
      return "starting";
  }
}

export function descriptorPollingEnabled(runStatus: RunStatus): boolean {
  return runStatus === "running";
}

export function streamDescriptorPath(runId: string): string {
  return `/api/v1/runs/${encodeURIComponent(runId)}/stream?camera_id=overview`;
}

export function descriptorRetryAllowed(runStatus: RunStatus, failureCount: number): boolean {
  return descriptorPollingEnabled(runStatus) && failureCount < 4;
}

export function descriptorRefetchInterval(
  status: StreamStatus | undefined,
  successfulFetchCount: number,
): number | false {
  if (status !== "CREATED" && status !== "STARTING") return false;
  return boundedRetryDelay(Math.max(0, successfulFetchCount - 1));
}

export interface WhepPlayerProps {
  url: string;
  active: boolean;
  onStateChange?: (state: VideoPresentation) => void;
  onPeerConnection?: (peer: RTCPeerConnection | null) => void;
}

export function WhepPlayer({
  url,
  active,
  onStateChange,
  onPeerConnection,
}: WhepPlayerProps) {
  const { locale } = useI18n();
  const videoRef = useRef<HTMLVideoElement>(null);
  const retryCountRef = useRef(0);
  const [generation, setGeneration] = useState(0);
  const [state, setState] = useState<VideoPresentation>(active ? "connecting" : "closed");

  useEffect(() => {
    retryCountRef.current = 0;
    setGeneration(0);
  }, [url, active]);

  useEffect(() => {
    const updateState = (next: VideoPresentation) => {
      setState(next);
      onStateChange?.(next);
    };
    if (!active || !url || !videoRef.current || !window.RTCPeerConnection) {
      updateState(active ? "unavailable" : "closed");
      onPeerConnection?.(null);
      return;
    }

    const peer = new RTCPeerConnection();
    const negotiationController = new AbortController();
    const cleanupOnce = createWhepEffectCleanup();
    let disposed = false;
    let reconnectScheduled = false;
    let retryTimer = 0;
    let disconnectTimer = 0;
    let mediaStream: MediaStream | null = null;
    let sessionLocation: string | null = null;
    let negotiation: ReturnType<typeof negotiateWhep> | null = null;
    const video = videoRef.current;
    onPeerConnection?.(peer);

    const beginCleanup = () => {
      const ownedLocation = sessionLocation;
      const ownedNegotiation = negotiation;
      const ownedPeer = peer;
      const ownedStream = mediaStream;
      const ownedVideo = video;
      void cleanupOnce({
        sessionLocation: ownedLocation,
        negotiation: ownedNegotiation,
        peer: ownedPeer,
        stream: ownedStream,
        video: ownedVideo,
      });
    };

    const scheduleReconnect = (reason: WhepReconnectReason) => {
      if (disposed || reconnectScheduled) return;
      reconnectScheduled = true;
      updateState("reconnecting");
      const delay = whepReconnectDelay(reason, retryCountRef.current, MAX_WHEP_RETRIES);
      if (delay === null) {
        updateState("unavailable");
        beginCleanup();
        return;
      }
      retryCountRef.current += 1;
      retryTimer = window.setTimeout(() => {
        if (!disposed) setGeneration((value) => value + 1);
      }, delay);
    };

    peer.ontrack = (event) => {
      if (disposed) return;
      mediaStream = event.streams[0] ?? new MediaStream([event.track]);
      video.srcObject = mediaStream;
      void video.play().then(
        () => { if (!disposed) updateState("playing"); },
        () => { if (!disposed) updateState("unavailable"); },
      );
    };
    peer.onconnectionstatechange = () => {
      if (disposed) return;
      if (peer.connectionState === "connected") {
        window.clearTimeout(disconnectTimer);
        retryCountRef.current = 0;
      } else if (peer.connectionState === "failed") {
        window.clearTimeout(disconnectTimer);
        scheduleReconnect("failed");
      } else if (peer.connectionState === "disconnected") {
        window.clearTimeout(disconnectTimer);
        disconnectTimer = window.setTimeout(() => {
          if (!disposed && peer.connectionState === "disconnected") scheduleReconnect("disconnected");
        }, DISCONNECTED_GRACE_MS);
      }
    };
    peer.oniceconnectionstatechange = () => {
      if (!disposed && peer.iceConnectionState === "failed") scheduleReconnect("failed");
    };

    updateState(generation === 0 ? "connecting" : "reconnecting");
    negotiation = negotiateWhep({
      peer,
      whepUrl: url,
      signal: negotiationController.signal,
    });
    void negotiation.then(
      (result) => {
        sessionLocation = result.sessionLocation;
      },
      (error: unknown) => {
        if (error instanceof WhepNegotiationSessionError) {
          sessionLocation = error.sessionLocation;
        }
        if (disposed || (error instanceof DOMException && error.name === "AbortError")) return;
        scheduleReconnect("negotiation");
      },
    );

    return () => {
      disposed = true;
      negotiationController.abort();
      window.clearTimeout(retryTimer);
      window.clearTimeout(disconnectTimer);
      beginCleanup();
      onPeerConnection?.(null);
    };
  }, [active, generation, onPeerConnection, onStateChange, url]);

  return (
    <div className={`whep-player ${state}`} data-video-state={state}>
      <video ref={videoRef} controls autoPlay muted playsInline />
      <span>{locale === "en" ? EN_VIDEO_LABELS[state] : VIDEO_LABELS[state]}</span>
    </div>
  );
}

export function RunVideoPanel({ runId, runStatus }: { runId: string; runStatus: RunStatus }) {
  const { locale, t } = useI18n();
  const [viewerOpen, setViewerOpen] = useState(true);
  const queryClient = useQueryClient();
  const queryKey = ["run-stream", runId] as const;
  const enabled = viewerOpen && descriptorPollingEnabled(runStatus);
  const stream = useQuery({
    queryKey,
    queryFn: ({ signal }) => api<StreamDescriptor>(
      streamDescriptorPath(runId),
      { signal },
    ),
    enabled,
    retry: (failureCount) => descriptorRetryAllowed(runStatus, failureCount),
    retryDelay: boundedRetryDelay,
    refetchInterval: (query) => {
      return descriptorRefetchInterval(
        query.state.data?.status,
        query.state.dataUpdateCount,
      );
    },
    refetchIntervalInBackground: false,
  });

  useEffect(() => {
    if (!enabled) void queryClient.cancelQueries({ queryKey, exact: true });
    return () => {
      void queryClient.cancelQueries({ queryKey, exact: true });
    };
  }, [enabled, queryClient, runId]);

  const descriptor = stream.data;
  const identityMatches = !descriptor || descriptor.run_id === runId;
  const presentation = descriptorPresentation(
    runStatus,
    identityMatches ? descriptor?.status : undefined,
    stream.isError || !identityMatches,
  );


  return (
    <section className="run-video-panel" data-video-presentation={presentation}>
      <div className="button-row video-viewer-controls">
        <button className="secondary" type="button" onClick={() => setViewerOpen((open) => !open)}>
          {viewerOpen ? t("关闭实时视频", "Close live video") : t("打开实时视频", "Open live video")}
        </button>
        <small>{t("关闭只会释放浏览器视频连接；服务端录像仍会继续保存。", "Closing releases only the browser video connection; server recording continues.")}</small>
      </div>
      {!viewerOpen ? (
        <div className="whep-player idle">
          <span>{t("实时视频已关闭，可随时打开", "Live video is closed; open it at any time")}</span>
        </div>
      ) : descriptor?.status === "READY" && enabled && identityMatches ? (
        <WhepPlayer
          key={`${descriptor.stream_id}:${descriptor.whep_url}`}
          url={proxiedWhepUrl(descriptor.whep_url, location.origin)}
          active
        />
      ) : (
        <div className={`whep-player ${presentation}`}>
          <span>{locale === "en" ? EN_VIDEO_LABELS[presentation] : VIDEO_LABELS[presentation]}</span>
        </div>
      )}
      {viewerOpen && stream.isError && <p className="action-feedback error" role="alert">{stream.error instanceof Error ? stream.error.message : t("视频服务响应失败", "Video service request failed")}</p>}
      {viewerOpen && descriptor && identityMatches && (
        <div className="run-facts">
          <div><span>{t("视频流", "STREAM")}</span><strong>{descriptor.status}</strong></div>
          <div><span>{t("相机", "CAMERA")}</span><strong>{descriptor.camera_id}</strong></div>
          <div><span>{t("格式", "FORMAT")}</span><strong>{descriptor.width} x {descriptor.height} · {descriptor.fps} FPS</strong></div>
          <div><span>{t("传输协议", "TRANSPORT")}</span><strong>WHEP / WEBRTC</strong></div>
        </div>
      )}
      {viewerOpen && enabled && (stream.isError || !identityMatches) && (
        <button
          className="secondary video-retry"
          disabled={stream.isFetching}
          onClick={() => void stream.refetch()}
        >
          {stream.isFetching ? t("正在重试", "Retrying") : t("重试视频", "Retry video")}
        </button>
      )}
    </section>
  );
}
