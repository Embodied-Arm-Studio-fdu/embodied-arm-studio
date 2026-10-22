import { expect, test } from "@playwright/test";
import { createHash } from "node:crypto";
import { mkdir, rename, writeFile } from "node:fs/promises";
import { get, request } from "node:http";
import { dirname, resolve } from "node:path";

declare const process: { env: Record<string, string | undefined>; pid: number };

async function writeEvidence(path: string, contents: string): Promise<void> {
  const destination = resolve(path);
  const temporary = `${destination}.tmp-${process.pid}-${Date.now()}`;
  await mkdir(dirname(destination), { recursive: true });
  await writeFile(temporary, contents, "utf8");
  await rename(temporary, destination);
}

async function fetchFrontendResource(url: URL): Promise<{
  status: number;
  headers: Record<string, string>;
  body: Buffer;
}> {
  return new Promise((resolveRequest, reject) => {
    const request = get(url, (response) => {
      const chunks: Buffer[] = [];
      response.on("data", (chunk: Buffer) => chunks.push(chunk));
      response.on("end", () => {
        const headers = Object.fromEntries(
          Object.entries(response.headers)
            .filter((entry): entry is [string, string | string[]] => entry[1] !== undefined)
            .map(([name, value]) => [name, Array.isArray(value) ? value.join(", ") : value]),
        );
        resolveRequest({ status: response.statusCode ?? 502, headers, body: Buffer.concat(chunks) });
      });
    });
    request.on("error", reject);
  });
}

async function controlApiRequest(
  url: URL,
  method = "GET",
): Promise<{ status: number; body: string }> {
  return new Promise((resolveRequest, reject) => {
    const apiRequest = request(url, { method }, (response) => {
      const chunks: Buffer[] = [];
      response.on("data", (chunk: Buffer) => chunks.push(chunk));
      response.on("end", () => resolveRequest({
        status: response.statusCode ?? 502,
        body: Buffer.concat(chunks).toString("utf8"),
      }));
    });
    apiRequest.on("error", reject);
    apiRequest.end();
  });
}

async function currentWebRtcSessionId(apiOrigin: string, streamId: string): Promise<string> {
  const response = await controlApiRequest(new URL("/v3/webrtcsessions/list", apiOrigin));
  if (response.status !== 200) return "";
  const sessions = JSON.parse(response.body) as {
    items?: Array<{ id?: string; path?: string; peerConnectionEstablished?: boolean }>;
  };
  return sessions.items?.find((session) => (
    session.path === streamId && session.peerConnectionEstablished
  ))?.id ?? "";
}

test("testsrc WHEP player plays, reconnects, and deletes a real MediaMTX session", async ({
  browser,
  page,
}, testInfo) => {
  test.setTimeout(120_000);
  const whepUrl = process.env.WHEP_SMOKE_URL;
  const streamId = process.env.WHEP_SMOKE_STREAM_ID;
  test.skip(!whepUrl || !streamId, "WHEP_SMOKE_URL and WHEP_SMOKE_STREAM_ID are required");
  const runId = process.env.WHEP_EVIDENCE_RUN_ID ?? "testsrc-run";
  const taskId = process.env.WHEP_EVIDENCE_TASK_ID ?? "testsrc-task";
  const cameraId = process.env.WHEP_EVIDENCE_CAMERA_ID ?? "overview";
  const controlApiOrigin = process.env.WHEP_MEDIAMTX_API_ORIGIN ?? "http://127.0.0.1:19997";
  const frontendForwardOrigin = process.env.WHEP_FRONTEND_FORWARD_ORIGIN;

  if (frontendForwardOrigin) {
    const publicOrigin = new URL(testInfo.project.use.baseURL as string).origin;
    await page.route(`${publicOrigin}/**`, async (route) => {
      const publicUrl = new URL(route.request().url());
      if (publicUrl.pathname === "/@vite/client") {
        await route.fulfill({
          status: 200,
          contentType: "application/javascript",
          body: `
export const injectQuery = (url) => url;
export const createHotContext = () => ({
  accept() {}, dispose() {}, decline() {}, invalidate() {}, on() {}, send() {}, prune() {},
});
export const updateStyle = (id, css) => {
  let style = document.querySelector('style[data-vite-dev-id="' + id + '"]');
  if (!style) {
    style = document.createElement('style');
    style.setAttribute('data-vite-dev-id', id);
    document.head.appendChild(style);
  }
  style.textContent = css;
};
export const removeStyle = (id) => document.querySelector('style[data-vite-dev-id="' + id + '"]')?.remove();
`,
        });
        return;
      }
      const forwardedUrl = new URL(`${publicUrl.pathname}${publicUrl.search}`, frontendForwardOrigin);
      const response = await fetchFrontendResource(forwardedUrl);
      if (response.headers["content-type"]?.startsWith("text/html")) {
        response.body = Buffer.from(
          response.body
            .toString("utf8")
            .replace(/<script type="module" src="\/@vite\/client"><\/script>/, ""),
        );
        delete response.headers["content-length"];
        delete response.headers.etag;
      }
      await route.fulfill(response);
    });
  }

  const descriptor = {
    stream_id: streamId,
    run_id: runId,
    task_id: taskId,
    camera_id: cameraId,
    status: "READY",
    webrtc_url: whepUrl?.replace(/\/whep$/, ""),
    whep_url: whepUrl,
    width: 1280,
    height: 720,
    fps: 30,
  };
  let descriptorRequests = 0;
  let offerSdpSha256: string | null = null;
  let offerSdpBytes: number | null = null;
  let answerSdpSha256: string | null = null;
  let answerSdpBytes: number | null = null;

  await page.route("**/api/v1/runs/whep-harness/stream", async (route) => {
    descriptorRequests += 1;
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(descriptor) });
  });

  const harnessQuery = new URLSearchParams({
    run_id: runId,
    task_id: taskId,
    stream_id: streamId as string,
    camera_id: cameraId,
    whep_url: whepUrl as string,
  });
  const manualDescriptor = new URLSearchParams({
    stream_id: streamId as string,
    whep_url: whepUrl as string,
  });
  await page.goto(`/whep-e2e/whep-harness.html?${harnessQuery}#${manualDescriptor}`, {
    waitUntil: "commit",
  });
  const video = page.locator("video");
  await expect(page.locator('[data-video-state="playing"]')).toBeVisible({ timeout: 60_000 });
  await expect.poll(() => video.evaluate((element) => {
    const media = element as HTMLVideoElement;
    return { width: media.videoWidth, height: media.videoHeight };
  })).toEqual({ width: 1280, height: 720 });
  const firstTime = await video.evaluate((element) => (element as HTMLVideoElement).currentTime);
  await expect.poll(() => video.evaluate(
    (element) => (element as HTMLVideoElement).currentTime,
  )).toBeGreaterThan(firstTime + 0.5);
  const advancedTime = await video.evaluate((element) => (element as HTMLVideoElement).currentTime);
  const firstEvidence = await page.evaluate(() => window.__WHEP_HARNESS__?.evidence());
  const sdp = await page.evaluate(() => window.__WHEP_HARNESS__?.sdp());
  if (sdp?.offer) {
    offerSdpSha256 = createHash("sha256").update(sdp.offer).digest("hex");
    offerSdpBytes = Buffer.byteLength(sdp.offer);
  }
  if (sdp?.answer) {
    answerSdpSha256 = createHash("sha256").update(sdp.answer).digest("hex");
    answerSdpBytes = Buffer.byteLength(sdp.answer);
  }

  let initialSessionId = "";
  await expect.poll(async () => {
    initialSessionId = await currentWebRtcSessionId(controlApiOrigin, streamId as string);
    return initialSessionId;
  }).not.toBe("");
  const kickResponse = await controlApiRequest(
    new URL(`/v3/webrtcsessions/kick/${encodeURIComponent(initialSessionId)}`, controlApiOrigin),
    "POST",
  );
  expect(kickResponse.status).toBe(200);
  await expect.poll(async () => (
    await page.evaluate(() => window.__WHEP_HARNESS__?.states.includes("reconnecting"))
  )).toBe(true);
  await expect(page.locator('[data-video-state="playing"]')).toBeVisible({ timeout: 30_000 });
  await expect.poll(async () => (
    await page.evaluate(() => window.__WHEP_HARNESS__?.evidence())
  )?.whep_delete_count).toBeGreaterThanOrEqual(1);
  await expect.poll(async () => (
    await page.evaluate(() => window.__WHEP_HARNESS__?.evidence())
  )?.whep_post_count).toBeGreaterThanOrEqual(2);

  expect(descriptorRequests).toBe(0);
  expect(firstEvidence).toMatchObject({
    schema_version: "whep-browser-evidence/v1",
    identity: {
      run_id: runId,
      task_id: taskId,
      stream_id: streamId,
      camera_id: cameraId,
      whep_url: whepUrl,
    },
    connection_state: "connected",
    ice_connection_state: expect.stringMatching(/connected|completed/),
    selected_candidate_pair: true,
    remote_candidate_address: "192.0.2.10",
    remote_candidate_protocol: "udp",
    last_whep_post_status: 201,
    last_whep_response_type: "application/sdp",
    whep_post_count: 1,
    whep_delete_count: 0,
    last_whep_delete_status: null,
    reconnect_count: 0,
  });
  expect(firstEvidence?.last_whep_session_location).toBeTruthy();
  expect(offerSdpSha256).toMatch(/^[a-f0-9]{64}$/);
  expect(offerSdpBytes).toBeGreaterThan(0);
  expect(answerSdpSha256).toMatch(/^[a-f0-9]{64}$/);
  expect(answerSdpBytes).toBeGreaterThan(0);

  const evidenceAfterNetworkReconnect = await page.evaluate(
    () => window.__WHEP_HARNESS__?.evidence(),
  );
  expect(evidenceAfterNetworkReconnect?.whep_post_count).toBeGreaterThanOrEqual(2);
  expect(evidenceAfterNetworkReconnect?.whep_delete_count).toBeGreaterThanOrEqual(1);
  expect(evidenceAfterNetworkReconnect?.reconnect_count).toBeGreaterThanOrEqual(1);
  expect(await page.evaluate(() => window.__WHEP_HARNESS__?.states))
    .toContain("reconnecting");

  await page.getByRole("button", { name: "重连测试（可选）" }).click();
  await expect(page.locator('[data-video-state="playing"]')).toBeVisible({ timeout: 30_000 });
  await expect.poll(async () => (
    await page.evaluate(() => window.__WHEP_HARNESS__?.evidence())
  )?.whep_delete_count).toBeGreaterThanOrEqual(2);
  const finalEvidence = await page.evaluate(() => window.__WHEP_HARNESS__?.evidence());
  expect(finalEvidence).toMatchObject({
    last_whep_post_status: 201,
    last_whep_response_type: "application/sdp",
  });
  expect(finalEvidence?.last_whep_delete_status).toBeGreaterThanOrEqual(200);
  expect(finalEvidence?.last_whep_delete_status).toBeLessThan(300);
  expect(finalEvidence?.whep_post_count).toBeGreaterThanOrEqual(3);
  expect(finalEvidence?.reconnect_count).toBeGreaterThanOrEqual(2);
  expect(finalEvidence?.selected_candidate_pair).toBe(true);
  expect(finalEvidence?.remote_candidate_protocol).toBe("udp");
  expect(finalEvidence?.video_width).toBe(1280);
  expect(finalEvidence?.video_height).toBe(720);
  expect(finalEvidence?.stop_requests).toBe(0);

  const evidence = {
    schema_version: "whep-browser-evidence/v1",
    evidence_kind: "real-browser-whep",
    source_kind: "ffmpeg-testsrc2",
    formal_run_evidence: false,
    captured_at: finalEvidence?.captured_at ?? new Date().toISOString(),
    browser: {
      name: testInfo.project.name,
      version: browser?.version() ?? null,
      user_agent: await page.evaluate(() => navigator.userAgent),
    },
    identity: finalEvidence?.identity,
    descriptor_injected: true,
    whep_post_status: finalEvidence?.last_whep_post_status,
    whep_response_media_type: "application/sdp",
    cors_passed: true,
    peer_connection: "connected",
    selected_candidate_pair: true,
    remote_candidate_address: "192.0.2.10",
    remote_candidate_protocol: "udp",
    video_width: 1280,
    video_height: 720,
    whep_session_location: finalEvidence?.last_whep_session_location,
    offer_sdp_sha256: offerSdpSha256,
    offer_sdp_bytes: offerSdpBytes,
    answer_sdp_sha256: answerSdpSha256,
    answer_sdp_bytes: answerSdpBytes,
    playback: {
      first_time: firstTime,
      advanced_time: advancedTime,
      progress_seconds: advancedTime - firstTime,
    },
    reconnect_passed: true,
    whep_delete_observed: true,
    last_whep_delete_status: finalEvidence?.last_whep_delete_status,
    reconnect_count: finalEvidence?.reconnect_count,
    stop_requests: finalEvidence?.stop_requests,
  };
  const serializedEvidence = `${JSON.stringify(evidence, null, 2)}\n`;
  await testInfo.attach("whep-browser-evidence", {
    body: serializedEvidence,
    contentType: "application/json",
  });
  if (process.env.WHEP_EVIDENCE_PATH) {
    await writeEvidence(process.env.WHEP_EVIDENCE_PATH, serializedEvidence);
  }
  console.info(JSON.stringify(evidence));
});
