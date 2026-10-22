import { afterEach, describe, expect, it, vi } from "vitest";
import {
  boundedRetryDelay,
  createWhepEffectCleanup,
  deleteWhepSession,
  mediaType,
  negotiateWhep,
  resolveWhepSessionLocation,
  stopPeerMedia,
  WhepNegotiationSessionError,
  whepReconnectDelay,
} from "./whep";

class FakePeer extends EventTarget {
  iceGatheringState: RTCIceGatheringState = "gathering";
  iceConnectionState: RTCIceConnectionState = "new";
  connectionState: RTCPeerConnectionState = "new";
  localDescription: RTCSessionDescription | null = null;
  remoteDescription: RTCSessionDescription | null = null;
  addTransceiver = vi.fn();
  createOffer = vi.fn(async () => ({ type: "offer" as const, sdp: "initial-offer" }));
  setLocalDescription = vi.fn(async (description: RTCSessionDescriptionInit) => {
    this.localDescription = { type: "offer", sdp: description.sdp ?? "" } as RTCSessionDescription;
  });
  setRemoteDescription = vi.fn(async (description: RTCSessionDescriptionInit) => {
    this.remoteDescription = description as RTCSessionDescription;
  });
  close = vi.fn();
  getReceivers = vi.fn(() => [] as RTCRtpReceiver[]);

  completeIce() {
    this.iceGatheringState = "complete";
    this.localDescription = { type: "offer", sdp: "final-ice-offer" } as RTCSessionDescription;
    this.dispatchEvent(new Event("icegatheringstatechange"));
  }
}

function response(
  status = 201,
  contentType = "application/sdp; charset=utf-8",
  location = "/stream-1/whep/session-1",
) {
  return new Response("answer-sdp", {
    status,
    headers: { "Content-Type": contentType, Location: location },
  });
}

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("WHEP contract", () => {
  it("uses capped 1, 2, 4, 8 second retry delays", () => {
    expect([0, 1, 2, 3, 9].map(boundedRetryDelay)).toEqual([1_000, 2_000, 4_000, 8_000, 8_000]);
  });

  it("retries startup negotiation frequently within a bounded window", () => {
    expect([0, 1, 2, 3, 4, 7].map((attempt) => whepReconnectDelay("negotiation", attempt)))
      .toEqual([500, 500, 750, 750, 1_000, 1_000]);
    expect(whepReconnectDelay("negotiation", 8)).toBeNull();
  });

  it("rebuilds failed peers immediately and disconnected peers after the caller grace", () => {
    expect(whepReconnectDelay("failed", 0)).toBe(0);
    expect(whepReconnectDelay("disconnected", 0)).toBe(1_000);
  });

  it.each([
    ["application/sdp", "application/sdp"],
    ["Application/SDP; Charset=UTF-8", "application/sdp"],
    ["text/plain; charset=utf-8", "text/plain"],
  ])("parses the Content-Type media type", (value, expected) => {
    expect(mediaType(value)).toBe(expected);
  });

  it("waits for ICE completion and posts the final local SDP", async () => {
    const peer = new FakePeer();
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => response());
    const controller = new AbortController();
    const resultPromise = negotiateWhep({
      peer: peer as unknown as RTCPeerConnection,
      whepUrl: "http://192.0.2.10:18889/stream-1/whep",
      signal: controller.signal,
      fetchImpl: fetchMock as typeof fetch,
    });
    await Promise.resolve();
    await Promise.resolve();
    expect(fetchMock).not.toHaveBeenCalled();

    peer.completeIce();
    const result = await resultPromise;

    expect(fetchMock).toHaveBeenCalledOnce();
    expect(fetchMock.mock.calls[0][0]).toBe("http://192.0.2.10:18889/stream-1/whep");
    expect(fetchMock.mock.calls[0][1]).toMatchObject({
      method: "POST",
      body: "final-ice-offer",
      credentials: "omit",
      signal: controller.signal,
    });
    expect(peer.addTransceiver).toHaveBeenCalledWith("video", { direction: "recvonly" });
    expect(peer.addTransceiver).toHaveBeenCalledTimes(1);
    expect(peer.setRemoteDescription).toHaveBeenCalledWith({ type: "answer", sdp: "answer-sdp" });
    expect(result.sessionLocation).toBe("http://192.0.2.10:18889/stream-1/whep/session-1");
  });

  it("aborts before an ICE-incomplete offer can be posted", async () => {
    const peer = new FakePeer();
    const fetchMock = vi.fn();
    const controller = new AbortController();
    const resultPromise = negotiateWhep({
      peer: peer as unknown as RTCPeerConnection,
      whepUrl: "http://192.0.2.10:18889/stream-1/whep",
      signal: controller.signal,
      fetchImpl: fetchMock as typeof fetch,
    });
    await Promise.resolve();
    controller.abort();

    await expect(resultPromise).rejects.toMatchObject({ name: "AbortError" });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("times out ICE gathering without posting", async () => {
    vi.useFakeTimers();
    const peer = new FakePeer();
    const fetchMock = vi.fn();
    const resultPromise = negotiateWhep({
      peer: peer as unknown as RTCPeerConnection,
      whepUrl: "http://192.0.2.10:18889/stream-1/whep",
      signal: new AbortController().signal,
      fetchImpl: fetchMock as typeof fetch,
      iceGatheringTimeoutMs: 50,
    });
    const rejection = expect(resultPromise).rejects.toThrow("ICE gathering timed out");
    await vi.advanceTimersByTimeAsync(50);

    await rejection;
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it.each([
    [200, "application/sdp", "HTTP 201"],
    [201, "text/plain", "media type"],
  ])("rejects status or media type contract violations", async (status, contentType, message) => {
    const peer = new FakePeer();
    peer.iceGatheringState = "complete";
    const fetchMock = vi.fn(async () => response(status, contentType));
    await expect(negotiateWhep({
      peer: peer as unknown as RTCPeerConnection,
      whepUrl: "http://192.0.2.10:18889/stream-1/whep",
      signal: new AbortController().signal,
      fetchImpl: fetchMock as typeof fetch,
    })).rejects.toThrow(message);
  });

  it("preserves a safely located session for coordinated cleanup after a 201 media-type failure", async () => {
    const peer = new FakePeer();
    peer.iceGatheringState = "complete";
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => (
      response(201, "text/plain")
    ));
    const failure = negotiateWhep({
      peer: peer as unknown as RTCPeerConnection,
      whepUrl: "http://192.0.2.10:18889/stream-1/whep",
      signal: new AbortController().signal,
      fetchImpl: fetchMock as typeof fetch,
    });
    await expect(failure).rejects.toMatchObject({
      message: expect.stringContaining("media type"),
      sessionLocation: "http://192.0.2.10:18889/stream-1/whep/session-1",
    });
    await expect(failure).rejects.toBeInstanceOf(WhepNegotiationSessionError);
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  it("keeps a WHEP failure isolated from every Run stop endpoint", async () => {
    const peer = new FakePeer();
    peer.iceGatheringState = "complete";
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => (
      new Response(null, { status: 503 })
    ));
    await expect(negotiateWhep({
      peer: peer as unknown as RTCPeerConnection,
      whepUrl: "http://192.0.2.10:18889/stream-1/whep",
      signal: new AbortController().signal,
      fetchImpl: fetchMock as typeof fetch,
    })).rejects.toThrow("HTTP 201");
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(fetchMock.mock.calls.some(([url]) => /\/runs\/[^/]+\/stop/.test(String(url)))).toBe(false);
  });

  it("resolves a relative Location against the WHEP origin", () => {
    expect(resolveWhepSessionLocation(
      "http://192.0.2.10:18889/stream-1/whep",
      "session-1",
    )).toBe("http://192.0.2.10:18889/stream-1/session-1");
  });

  it.each([
    "https://192.0.2.10:18889/session-1",
    "http://example.invalid:18889/session-1",
    "ftp://192.0.2.10:18889/session-1",
    "http://user:password@192.0.2.10:18889/session-1",
    "http://192.0.2.10:18889/session-1?token=hidden",
    "http://192.0.2.10:18889/session-1#fragment",
  ])("rejects an unsafe WHEP Location: %s", (location) => {
    expect(() => resolveWhepSessionLocation(
      "http://192.0.2.10:18889/stream-1/whep",
      location,
    )).toThrow();
  });

  it("sends an abortable best-effort DELETE only to the saved session", async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => (
      new Response(null, { status: 200 })
    ));
    const controller = new AbortController();
    await expect(deleteWhepSession(
      "http://192.0.2.10:18889/stream-1/whep/session-1",
      controller.signal,
      fetchMock as typeof fetch,
    )).resolves.toEqual({ outcome: "deleted", status: 200 });
    expect(fetchMock).toHaveBeenCalledWith(
      "http://192.0.2.10:18889/stream-1/whep/session-1",
      expect.objectContaining({ method: "DELETE", signal: controller.signal }),
    );
    expect(fetchMock.mock.calls.some(([url]) => String(url).includes("/stop"))).toBe(false);
  });

  it("waits for DELETE before closing the peer", async () => {
    let resolveDelete!: (response: Response) => void;
    const pendingDelete = new Promise<Response>((resolve) => { resolveDelete = resolve; });
    const fetchMock = vi.fn(async () => pendingDelete);
    const peer = new FakePeer();
    const video = { srcObject: null } as unknown as HTMLVideoElement;
    const cleanup = createWhepEffectCleanup();

    const result = cleanup({
      sessionLocation: "http://192.0.2.10:18889/old/whep/old-session",
      negotiation: null,
      peer: peer as unknown as RTCPeerConnection,
      stream: null,
      video,
      fetchImpl: fetchMock as typeof fetch,
    });
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledOnce());
    expect(peer.close).not.toHaveBeenCalled();

    resolveDelete(new Response(null, { status: 200 }));
    await expect(result).resolves.toEqual({ outcome: "deleted", status: 200 });
    expect(peer.close).toHaveBeenCalledOnce();
  });

  it("runs a single DELETE and local close when cleanup is requested more than once", async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => (
      new Response(null, { status: 200 })
    ));
    const peer = new FakePeer();
    const cleanup = createWhepEffectCleanup();
    const owned = {
      sessionLocation: "http://192.0.2.10:18889/old/whep/old-session",
      negotiation: null,
      peer: peer as unknown as RTCPeerConnection,
      stream: null,
      video: null,
      fetchImpl: fetchMock as typeof fetch,
    };

    const first = cleanup(owned);
    const second = cleanup(owned);
    expect(first).toBe(second);
    await first;
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(peer.close).toHaveBeenCalledOnce();
  });

  it("cleans a late negotiation result exactly once", async () => {
    let resolveNegotiation!: (result: { sessionLocation: string; responseContentType: string }) => void;
    const negotiation = new Promise<{
      sessionLocation: string;
      responseContentType: string;
    }>((resolve) => { resolveNegotiation = resolve; });
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => (
      new Response(null, { status: 200 })
    ));
    const peer = new FakePeer();
    const cleanup = createWhepEffectCleanup();
    const owned = {
      sessionLocation: null,
      negotiation,
      peer: peer as unknown as RTCPeerConnection,
      stream: null,
      video: null,
      fetchImpl: fetchMock as typeof fetch,
    };

    const first = cleanup(owned);
    const second = cleanup(owned);
    expect(fetchMock).not.toHaveBeenCalled();
    resolveNegotiation({
      sessionLocation: "http://192.0.2.10:18889/late/whep/late-session",
      responseContentType: "application/sdp",
    });

    await Promise.all([first, second]);
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(fetchMock.mock.calls[0][0]).toBe(
      "http://192.0.2.10:18889/late/whep/late-session",
    );
    expect(peer.close).toHaveBeenCalledOnce();
  });

  it("cleans a negotiation failure with an owned Location exactly once", async () => {
    const location = "http://192.0.2.10:18889/failed/whep/failed-session";
    const negotiation = Promise.reject(new WhepNegotiationSessionError("invalid answer", location));
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => (
      new Response(null, { status: 200 })
    ));
    const peer = new FakePeer();
    const cleanup = createWhepEffectCleanup();
    const owned = {
      sessionLocation: null,
      negotiation,
      peer: peer as unknown as RTCPeerConnection,
      stream: null,
      video: null,
      fetchImpl: fetchMock as typeof fetch,
    };

    const first = cleanup(owned);
    const second = cleanup(owned);
    await Promise.all([first, second]);

    expect(fetchMock).toHaveBeenCalledOnce();
    expect(fetchMock.mock.calls[0][0]).toBe(location);
    expect(peer.close).toHaveBeenCalledOnce();
  });

  it("treats 404 as an idempotent terminal result without retrying and still closes locally", async () => {
    const fetchMock = vi.fn(async () => new Response(null, { status: 404 }));
    const peer = new FakePeer();
    const video = { srcObject: {} } as unknown as HTMLVideoElement;
    const cleanup = createWhepEffectCleanup();

    await expect(cleanup({
      sessionLocation: "http://192.0.2.10:18889/gone/whep/gone-session",
      negotiation: null,
      peer: peer as unknown as RTCPeerConnection,
      stream: null,
      video,
      fetchImpl: fetchMock as typeof fetch,
    })).resolves.toEqual({ outcome: "already_absent", status: 404 });
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(peer.close).toHaveBeenCalledOnce();
    expect(video.srcObject).toBeNull();
  });

  it("closes local resources after a bounded DELETE timeout without retrying", async () => {
    vi.useFakeTimers();
    const fetchMock = vi.fn((_input: RequestInfo | URL, init?: RequestInit) => (
      new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
      })
    ));
    const peer = new FakePeer();
    const cleanup = createWhepEffectCleanup();
    const result = cleanup({
      sessionLocation: "http://192.0.2.10:18889/timeout/whep/timeout-session",
      negotiation: null,
      peer: peer as unknown as RTCPeerConnection,
      stream: null,
      video: null,
      fetchImpl: fetchMock as typeof fetch,
      deleteTimeoutMs: 50,
    });

    await vi.advanceTimersByTimeAsync(50);
    await expect(result).resolves.toEqual({ outcome: "timeout", status: null });
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(peer.close).toHaveBeenCalledOnce();
  });

  it("keeps old and new effect cleanup locations isolated and never calls Run stop", async () => {
    const requestedUrls: string[] = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      requestedUrls.push(String(input));
      return new Response(null, { status: 200 });
    });
    const oldPeer = new FakePeer();
    const newPeer = new FakePeer();
    const oldCleanup = createWhepEffectCleanup();
    const newCleanup = createWhepEffectCleanup();

    await Promise.all([
      oldCleanup({
        sessionLocation: "http://192.0.2.10:18889/stream/whep/old-session",
        negotiation: null,
        peer: oldPeer as unknown as RTCPeerConnection,
        stream: null,
        video: null,
        fetchImpl: fetchMock as typeof fetch,
      }),
      newCleanup({
        sessionLocation: "http://192.0.2.10:18889/stream/whep/new-session",
        negotiation: null,
        peer: newPeer as unknown as RTCPeerConnection,
        stream: null,
        video: null,
        fetchImpl: fetchMock as typeof fetch,
      }),
    ]);

    expect(requestedUrls).toEqual([
      "http://192.0.2.10:18889/stream/whep/old-session",
      "http://192.0.2.10:18889/stream/whep/new-session",
    ]);
    expect(requestedUrls.some((url) => /\/runs\/[^/]+\/stop(?:$|\?)/.test(url))).toBe(false);
    expect(oldPeer.close).toHaveBeenCalledOnce();
    expect(newPeer.close).toHaveBeenCalledOnce();
  });

  it("closes the peer, receiver and MediaStream tracks, and video binding", () => {
    const receiverTrack = { stop: vi.fn() } as unknown as MediaStreamTrack;
    const streamTrack = { stop: vi.fn() } as unknown as MediaStreamTrack;
    const peer = new FakePeer();
    peer.getReceivers.mockReturnValue([{ track: receiverTrack } as RTCRtpReceiver]);
    const stream = { getTracks: () => [streamTrack] } as unknown as MediaStream;
    const video = { srcObject: stream } as unknown as HTMLVideoElement;

    stopPeerMedia(peer as unknown as RTCPeerConnection, stream, video);

    expect(receiverTrack.stop).toHaveBeenCalledOnce();
    expect(streamTrack.stop).toHaveBeenCalledOnce();
    expect(peer.close).toHaveBeenCalledOnce();
    expect(video.srcObject).toBeNull();
  });
});
