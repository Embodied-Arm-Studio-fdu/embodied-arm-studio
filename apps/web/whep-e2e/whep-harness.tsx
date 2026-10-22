import { useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import type { StreamDescriptor } from "../src/api";
import { WhepPlayer, type VideoPresentation } from "../src/WhepPlayer";
import "../src/styles.css";

interface HarnessEvidence {
  schema_version: "whep-browser-evidence/v1";
  captured_at: string;
  identity: {
    run_id: string;
    task_id: string;
    stream_id: string;
    camera_id: string;
    whep_url: string;
  };
  connection_state: string;
  ice_connection_state: string;
  selected_candidate_pair: boolean;
  remote_candidate_address: string | null;
  remote_candidate_protocol: string | null;
  remote_candidate_type: string | null;
  last_whep_post_status: number | null;
  last_whep_response_type: string | null;
  last_whep_session_location: string | null;
  whep_post_count: number;
  whep_delete_count: number;
  last_whep_delete_status: number | null;
  reconnect_count: number;
  video_width: number;
  video_height: number;
  first_playing_time: number | null;
  current_time: number;
  playback_progress_seconds: number;
  stop_requests: number;
}

declare global {
  interface Window {
    __WHEP_HARNESS__?: {
      states: VideoPresentation[];
      reconnect: () => void;
      evidence: () => Promise<HarnessEvidence>;
      sdp: () => { offer: string | null; answer: string | null };
    };
  }
}

const EXPECTED_WHEP_ORIGIN = "http://192.0.2.10:18889";
const states: VideoPresentation[] = [];
let stopRequests = 0;
let lastWhepPostStatus: number | null = null;
let lastWhepResponseType: string | null = null;
let whepPostCount = 0;
let whepDeleteCount = 0;
let lastWhepDeleteStatus: number | null = null;
let reconnectCount = 0;
let lastWhepSessionLocation: string | null = null;
let lastWhepOfferSdp: string | null = null;
let lastWhepAnswerSdp: string | null = null;
let firstPlayingTime: number | null = null;
const query = new URLSearchParams(location.search);
const originalFetch = window.fetch.bind(window);

window.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
  const requestUrl = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
  const method = (init?.method ?? (input instanceof Request ? input.method : "GET")).toUpperCase();
  let parsedUrl: URL | null = null;
  try {
    parsedUrl = new URL(requestUrl, location.href);
  } catch {
    // The original fetch remains authoritative for malformed input.
  }
  const approvedWhepOrigin = parsedUrl?.origin === EXPECTED_WHEP_ORIGIN;
  const isWhepPost = approvedWhepOrigin && method === "POST" && parsedUrl?.pathname.endsWith("/whep");
  const isWhepDelete = approvedWhepOrigin && method === "DELETE";
  if (/\/runs\/[^/]+\/stop(?:$|\?)/.test(requestUrl)) stopRequests += 1;
  if (isWhepPost) {
    whepPostCount += 1;
    lastWhepOfferSdp = typeof init?.body === "string" ? init.body : null;
  }
  if (isWhepDelete) whepDeleteCount += 1;
  const response = await originalFetch(input, init);
  if (isWhepPost) {
    lastWhepPostStatus = response.status;
    lastWhepResponseType = (response.headers.get("content-type") ?? "")
      .split(";", 1)[0]
      .trim()
      .toLowerCase() || null;
    lastWhepSessionLocation = response.headers.get("location");
    lastWhepAnswerSdp = await response.clone().text();
  }
  if (isWhepDelete) lastWhepDeleteStatus = response.status;
  return response;
};

function validateManualDescriptor(): StreamDescriptor | null {
  const values = new URLSearchParams(location.hash.replace(/^#/, ""));
  const streamId = values.get("stream_id");
  const whepValue = values.get("whep_url");
  if (!streamId && !whepValue) return null;
  if (!streamId || !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(streamId) || !whepValue) {
    throw new Error("Manual descriptor input is invalid");
  }
  const whepUrl = new URL(whepValue);
  if (
    whepUrl.origin !== EXPECTED_WHEP_ORIGIN
    || whepUrl.username
    || whepUrl.password
    || whepUrl.search
    || whepUrl.hash
    || !whepUrl.pathname.endsWith(`/${streamId}/whep`)
  ) {
    throw new Error("Manual WHEP URL is outside the approved test origin or path");
  }
  return {
    stream_id: streamId,
    run_id: "whep-harness",
    task_id: "whep-harness-task",
    camera_id: "overview",
    status: "READY",
    webrtc_url: whepUrl.href.replace(/\/whep$/, ""),
    whep_url: whepUrl.href,
    width: 1280,
    height: 720,
    fps: 30,
  };
}

function evidenceIdentity(descriptor: StreamDescriptor): HarnessEvidence["identity"] {
  return {
    run_id: query.get("run_id") ?? descriptor.run_id,
    task_id: query.get("task_id") ?? descriptor.task_id,
    stream_id: query.get("stream_id") ?? descriptor.stream_id,
    camera_id: query.get("camera_id") ?? descriptor.camera_id,
    whep_url: query.get("whep_url") ?? descriptor.whep_url,
  };
}

async function selectedCandidateEvidence(
  peer: RTCPeerConnection | null,
  descriptor: StreamDescriptor,
): Promise<HarnessEvidence> {
  let pair: (RTCStats & { remoteCandidateId?: string }) | undefined;
  let remote: (RTCStats & {
    address?: string;
    ip?: string;
    protocol?: string;
    candidateType?: string;
  }) | undefined;
  if (peer) {
    const stats = await peer.getStats();
    stats.forEach((item) => {
      if (item.type === "candidate-pair" && (item.selected || (item.nominated && item.state === "succeeded"))) {
        pair = item;
      }
    });
    if (pair?.remoteCandidateId) remote = stats.get(pair.remoteCandidateId) as typeof remote;
  }
  const video = document.querySelector("video");
  if (video && firstPlayingTime === null && video.currentTime > 0) {
    firstPlayingTime = video.currentTime;
  }
  return {
    schema_version: "whep-browser-evidence/v1",
    captured_at: new Date().toISOString(),
    identity: evidenceIdentity(descriptor),
    connection_state: peer?.connectionState ?? "closed",
    ice_connection_state: peer?.iceConnectionState ?? "closed",
    selected_candidate_pair: Boolean(pair),
    remote_candidate_address: remote?.address ?? remote?.ip ?? null,
    remote_candidate_protocol: remote?.protocol ?? null,
    remote_candidate_type: remote?.candidateType ?? null,
    last_whep_post_status: lastWhepPostStatus,
    last_whep_response_type: lastWhepResponseType,
    last_whep_session_location: lastWhepSessionLocation,
    whep_post_count: whepPostCount,
    whep_delete_count: whepDeleteCount,
    last_whep_delete_status: lastWhepDeleteStatus,
    reconnect_count: reconnectCount,
    video_width: video?.videoWidth ?? 0,
    video_height: video?.videoHeight ?? 0,
    first_playing_time: firstPlayingTime,
    current_time: video?.currentTime ?? 0,
    playback_progress_seconds:
      firstPlayingTime === null
        ? 0
        : Math.max(0, (video?.currentTime ?? 0) - firstPlayingTime),
    stop_requests: stopRequests,
  };
}

function HarnessApp({ descriptor }: { descriptor: StreamDescriptor }) {
  const peerRef = useRef<RTCPeerConnection | null>(null);
  const [playerKey, setPlayerKey] = useState(0);
  const [mounted, setMounted] = useState(true);
  const [state, setState] = useState<VideoPresentation>("connecting");
  const [evidence, setEvidence] = useState<HarnessEvidence | null>(null);
  const recordState = useCallback((value: VideoPresentation) => {
    if (value === "reconnecting" && states[states.length - 1] !== "reconnecting") {
      reconnectCount += 1;
    }
    states.push(value);
    setState(value);
  }, []);
  const recordPeer = useCallback((value: RTCPeerConnection | null) => {
    peerRef.current = value;
  }, []);
  const reconnect = useCallback(() => {
    reconnectCount += 1;
    states.push("reconnecting");
    setState("reconnecting");
    setMounted(false);
    window.setTimeout(() => {
      setPlayerKey((value) => value + 1);
      setMounted(true);
    }, 250);
  }, []);

  useEffect(() => {
    const update = () => {
      void selectedCandidateEvidence(peerRef.current, descriptor).then(setEvidence);
    };
    update();
    const timer = window.setInterval(update, 500);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    window.__WHEP_HARNESS__ = {
      states,
      reconnect,
      evidence: () => selectedCandidateEvidence(peerRef.current, descriptor),
      sdp: () => ({ offer: lastWhepOfferSdp, answer: lastWhepAnswerSdp }),
    };
    return () => { delete window.__WHEP_HARNESS__; };
  }, [reconnect]);

  return (
    <main className="operations-page">
      <header className="operations-header">
        <div>
          <span className="eyebrow">TEST-ONLY DESCRIPTOR INJECTION</span>
          <h1>WHEP / WebRTC Browser Harness</h1>
          <p>此页面只验证 testsrc2 媒体播放，不代表 Backend live descriptor 或真实 Run E2E。</p>
        </div>
        <div className="run-mode-badges">
          <strong>SIMULATION MODE</strong>
          <span>TESTSRC2 · NVENC</span>
          <small>REAL ROBOT CONTROL DISABLED</small>
        </div>
      </header>
      {mounted && (
        <WhepPlayer
          key={playerKey}
          url={descriptor.whep_url}
          active
          onStateChange={recordState}
          onPeerConnection={recordPeer}
        />
      )}
      <section className="run-facts" id="harness-evidence">
        <div><span>PLAYER</span><strong>{state}</strong></div>
        <div><span>PEER</span><strong>{evidence?.connection_state ?? "new"}</strong></div>
        <div><span>ICE</span><strong>{evidence?.ice_connection_state ?? "new"}</strong></div>
        <div><span>CANDIDATE PAIR</span><strong>{evidence?.selected_candidate_pair ? "SELECTED" : "WAITING"}</strong></div>
        <div><span>REMOTE</span><strong>{evidence?.remote_candidate_address ?? "--"}</strong></div>
        <div><span>TRANSPORT</span><strong>{evidence?.remote_candidate_protocol ?? "--"}</strong></div>
        <div><span>LAST WHEP POST STATUS</span><strong>{evidence?.last_whep_post_status ?? "--"}</strong></div>
        <div><span>LAST WHEP RESPONSE TYPE</span><strong>{evidence?.last_whep_response_type ?? "--"}</strong></div>
        <div><span>WHEP DELETE COUNT</span><strong>{evidence?.whep_delete_count ?? 0}</strong></div>
        <div><span>LAST WHEP DELETE STATUS</span><strong>{evidence?.last_whep_delete_status ?? "--"}</strong></div>
        <div><span>RECONNECT COUNT</span><strong>{evidence?.reconnect_count ?? 0}</strong></div>
        <div><span>RESOLUTION</span><strong>{evidence?.video_width ?? 0} × {evidence?.video_height ?? 0}</strong></div>
        <div><span>CURRENT TIME</span><strong>{(evidence?.current_time ?? 0).toFixed(2)}</strong></div>
        <div><span>RUN STOP REQUESTS</span><strong>{evidence?.stop_requests ?? 0}</strong></div>
      </section>
      <div className="button-row" style={{ margin: 20 }}>
        <button id="reconnect-player" className="secondary" onClick={reconnect}>
          重连测试（可选）
        </button>
      </div>
    </main>
  );
}

const rootElement = document.getElementById("root");
if (!rootElement) throw new Error("WHEP harness root is unavailable");
const root = createRoot(rootElement);
const controller = new AbortController();
window.addEventListener("pagehide", () => controller.abort(), { once: true });

void Promise.resolve(validateManualDescriptor())
  .then(async (manualDescriptor) => {
    if (manualDescriptor) return manualDescriptor;
    const response = await fetch("/api/v1/runs/whep-harness/stream", { signal: controller.signal });
    if (!response.ok) throw new Error("Injected descriptor response failed");
    return response.json() as Promise<StreamDescriptor>;
  })
  .then((descriptor) => root.render(<HarnessApp descriptor={descriptor} />))
  .catch(() => {
    root.render(<p data-video-state="unavailable">测试 descriptor 不可用</p>);
  });
