export const WHEP_SDP_MEDIA_TYPE = "application/sdp";
export const WHEP_RETRY_DELAYS_MS = [1_000, 2_000, 4_000, 8_000] as const;
export const WHEP_NEGOTIATION_RETRY_DELAYS_MS = [
  500,
  500,
  750,
  750,
  1_000,
  1_000,
  1_000,
  1_000,
] as const;
export const WHEP_DELETE_TIMEOUT_MS = 1_500;

export function proxiedWhepUrl(url: string, browserOrigin: string): string {
  const endpoint = new URL(url, browserOrigin);
  if (!/^\/stream-[A-Za-z0-9-]+\/whep$/.test(endpoint.pathname)) return url;
  return new URL(`/media-whep${endpoint.pathname}`, browserOrigin).toString();
}

export interface WhepNegotiationOptions {
  peer: RTCPeerConnection;
  whepUrl: string;
  signal: AbortSignal;
  fetchImpl?: typeof fetch;
  iceGatheringTimeoutMs?: number;
}

export interface WhepNegotiationResult {
  sessionLocation: string;
  responseContentType: string;
}

export class WhepNegotiationSessionError extends Error {
  constructor(message: string, readonly sessionLocation: string) {
    super(message);
    this.name = "WhepNegotiationSessionError";
  }
}

export type WhepDeleteResult =
  | { outcome: "deleted"; status: number }
  | { outcome: "already_absent"; status: 404 }
  | { outcome: "http_error"; status: number }
  | { outcome: "timeout"; status: null }
  | { outcome: "network_error"; status: null };

export interface OwnedWhepCleanup {
  sessionLocation: string | null;
  negotiation: Promise<WhepNegotiationResult> | null;
  peer: RTCPeerConnection;
  stream: MediaStream | null;
  video: HTMLVideoElement | null;
  fetchImpl?: typeof fetch;
  deleteTimeoutMs?: number;
}

export type WhepReconnectReason = "negotiation" | "failed" | "disconnected";

export function whepReconnectDelay(
  reason: WhepReconnectReason,
  completedRetries: number,
  maximumRetries = WHEP_NEGOTIATION_RETRY_DELAYS_MS.length,
): number | null {
  if (completedRetries >= maximumRetries) return null;
  if (reason === "failed") return 0;
  if (reason === "disconnected") return boundedRetryDelay(completedRetries);
  const index = Math.max(
    0,
    Math.min(Math.floor(completedRetries), WHEP_NEGOTIATION_RETRY_DELAYS_MS.length - 1),
  );
  return WHEP_NEGOTIATION_RETRY_DELAYS_MS[index];
}

export function boundedRetryDelay(attempt: number): number {
  const index = Math.max(0, Math.min(Math.floor(attempt), WHEP_RETRY_DELAYS_MS.length - 1));
  return WHEP_RETRY_DELAYS_MS[index];
}

export function mediaType(value: string | null): string {
  return (value ?? "").split(";", 1)[0].trim().toLowerCase();
}

function abortError(): DOMException {
  return new DOMException("The operation was aborted", "AbortError");
}

function requireNotAborted(signal: AbortSignal): void {
  if (signal.aborted) throw abortError();
}

export function resolveWhepSessionLocation(whepUrl: string, location: string | null): string {
  if (!location) throw new Error("WHEP response is missing Location");
  let base: URL;
  let resolved: URL;
  try {
    base = new URL(whepUrl);
    resolved = new URL(location, base);
  } catch {
    throw new Error("WHEP Location is invalid");
  }
  if (!["http:", "https:"].includes(base.protocol) || !["http:", "https:"].includes(resolved.protocol)) {
    throw new Error("WHEP Location scheme is not allowed");
  }
  if (resolved.origin !== base.origin) {
    throw new Error("WHEP Location origin does not match the WHEP endpoint");
  }
  if (resolved.username || resolved.password || resolved.search || resolved.hash) {
    throw new Error("WHEP Location contains forbidden URL components");
  }
  return resolved.toString();
}

export function waitForIceGatheringComplete(
  peer: RTCPeerConnection,
  signal: AbortSignal,
  timeoutMs = 8_000,
): Promise<void> {
  requireNotAborted(signal);
  if (peer.iceGatheringState === "complete") return Promise.resolve();

  return new Promise((resolve, reject) => {
    const timeout = globalThis.setTimeout(() => finish(new Error("ICE gathering timed out")), timeoutMs);
    const onStateChange = () => {
      if (peer.iceGatheringState === "complete") finish();
    };
    const onAbort = () => finish(abortError());
    const finish = (error?: Error) => {
      globalThis.clearTimeout(timeout);
      peer.removeEventListener("icegatheringstatechange", onStateChange);
      signal.removeEventListener("abort", onAbort);
      if (error) reject(error);
      else resolve();
    };
    peer.addEventListener("icegatheringstatechange", onStateChange);
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

export async function negotiateWhep({
  peer,
  whepUrl,
  signal,
  fetchImpl = fetch,
  iceGatheringTimeoutMs = 8_000,
}: WhepNegotiationOptions): Promise<WhepNegotiationResult> {
  requireNotAborted(signal);
  peer.addTransceiver("video", { direction: "recvonly" });
  const offer = await peer.createOffer();
  requireNotAborted(signal);
  await peer.setLocalDescription(offer);
  await waitForIceGatheringComplete(peer, signal, iceGatheringTimeoutMs);
  requireNotAborted(signal);
  const finalSdp = peer.localDescription?.sdp;
  if (!finalSdp) throw new Error("ICE-complete WHEP offer is unavailable");

  const response = await fetchImpl(whepUrl, {
    method: "POST",
    headers: {
      "Content-Type": WHEP_SDP_MEDIA_TYPE,
      Accept: WHEP_SDP_MEDIA_TYPE,
    },
    body: finalSdp,
    credentials: "omit",
    mode: "cors",
    signal,
  });
  if (response.status !== 201) throw new Error("WHEP endpoint did not return HTTP 201");
  const responseContentType = response.headers.get("content-type") ?? "";
  const sessionLocation = resolveWhepSessionLocation(
    whepUrl,
    response.headers.get("location"),
  );
  try {
    if (mediaType(responseContentType) !== WHEP_SDP_MEDIA_TYPE) {
      throw new Error("WHEP response media type is not application/sdp");
    }
    const answerSdp = await response.text();
    requireNotAborted(signal);
    await peer.setRemoteDescription({ type: "answer", sdp: answerSdp });
  } catch (error) {
    throw new WhepNegotiationSessionError(
      error instanceof Error ? error.message : "WHEP negotiation failed",
      sessionLocation,
    );
  }
  return { sessionLocation, responseContentType };
}

export async function deleteWhepSession(
  sessionLocation: string,
  signal: AbortSignal,
  fetchImpl: typeof fetch = fetch,
): Promise<WhepDeleteResult> {
  requireNotAborted(signal);
  try {
    const response = await fetchImpl(sessionLocation, {
      method: "DELETE",
      credentials: "omit",
      keepalive: true,
      mode: "cors",
      signal,
    });
    if (response.status === 404) return { outcome: "already_absent", status: 404 };
    if (response.ok) return { outcome: "deleted", status: response.status };
    return { outcome: "http_error", status: response.status };
  } catch {
    return { outcome: signal.aborted ? "timeout" : "network_error", status: null };
  }
}

async function resolveOwnedSessionLocation(
  sessionLocation: string | null,
  negotiation: Promise<WhepNegotiationResult> | null,
): Promise<string | null> {
  if (sessionLocation) return sessionLocation;
  if (!negotiation) return null;
  try {
    return (await negotiation).sessionLocation;
  } catch (error) {
    return error instanceof WhepNegotiationSessionError ? error.sessionLocation : null;
  }
}

async function cleanupOwnedWhepSession({
  sessionLocation,
  negotiation,
  peer,
  stream,
  video,
  fetchImpl = fetch,
  deleteTimeoutMs = WHEP_DELETE_TIMEOUT_MS,
}: OwnedWhepCleanup): Promise<WhepDeleteResult | null> {
  try {
    const ownedLocation = await resolveOwnedSessionLocation(sessionLocation, negotiation);
    if (!ownedLocation) return null;

    const deleteController = new AbortController();
    const timeout = globalThis.setTimeout(() => deleteController.abort(), deleteTimeoutMs);
    try {
      return await deleteWhepSession(ownedLocation, deleteController.signal, fetchImpl);
    } finally {
      globalThis.clearTimeout(timeout);
    }
  } finally {
    stopPeerMedia(peer, stream, video);
  }
}

export function createWhepEffectCleanup(): (
  owned: OwnedWhepCleanup,
) => Promise<WhepDeleteResult | null> {
  let cleanupPromise: Promise<WhepDeleteResult | null> | null = null;
  return (owned) => {
    cleanupPromise ??= cleanupOwnedWhepSession(owned);
    return cleanupPromise;
  };
}

export function stopPeerMedia(
  peer: RTCPeerConnection,
  stream: MediaStream | null,
  video: HTMLVideoElement | null,
): void {
  const tracks = new Set<MediaStreamTrack>();
  for (const receiver of peer.getReceivers()) {
    if (receiver.track) tracks.add(receiver.track);
  }
  for (const track of stream?.getTracks() ?? []) tracks.add(track);
  for (const track of tracks) track.stop();
  peer.close();
  if (video) video.srcObject = null;
}
