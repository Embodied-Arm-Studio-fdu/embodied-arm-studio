import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  ReplayBlobGeneration,
  ReplayVideoElement,
  autoplayNeedsManualPlay,
  evaluationReplayPlaceholder,
  replayPresentation,
  replayArtifactPath,
  runVideoEnabled,
  runVideoMode,
  selectReplayArtifacts,
} from "./ReplayPlayer";

describe("ReplayPlayer contracts", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("uses the authenticated canonical Artifact download path", () => {
    expect(replayArtifactPath("run/id", "artifact id"))
      .toBe("/api/v1/runs/run%2Fid/artifacts/artifact%20id/download");
  });

  it("labels training video separately from successful evaluation replay", () => {
    expect(replayPresentation("training_video").title).toBe("训练录像");
    const evaluation = replayPresentation("evaluation_video");
    expect(evaluation.title).toBe("成功回合完整场景");
    expect(evaluation.description).toContain("完整机械臂仿真环境");
    expect(replayPresentation("training_video", "en").title).toBe("Training Recording");
    expect(replayPresentation("evaluation_video", "en").description).toContain("complete robot simulation environment");
  });

  it("describes evaluation replay availability from the Run lifecycle", () => {
    expect(evaluationReplayPlaceholder("running", false, false)).toContain("正在训练");
    expect(evaluationReplayPlaceholder("finalizing", false, false)).toContain("正在生成");
    expect(evaluationReplayPlaceholder("completed", true, false)).toContain("正在读取");
    expect(evaluationReplayPlaceholder("completed", false, true)).toContain("读取失败");
    expect(evaluationReplayPlaceholder("completed", false, false)).toContain("训练已完成");
    expect(evaluationReplayPlaceholder("stopped", false, false)).toContain("未正常完成");
    expect(evaluationReplayPlaceholder("running", false, false, "en")).toContain("Training is in progress");
    expect(evaluationReplayPlaceholder("completed", false, false, "en")).toContain("Training is complete");
  });

  it("selects evaluation replay independently when training video is absent", () => {
    const evaluation = {
      artifact_id: "evaluation-id",
      run_id: "run-id",
      artifact_type: "evaluation_video",
      path: "evaluation_video/successful_episode.mp4",
      size_bytes: 10,
      sha256: "a".repeat(64),
      is_primary_model: false,
      created_at: "2026-09-14T00:00:00Z",
    };
    const selected = selectReplayArtifacts([evaluation]);
    expect(selected.trainingVideo).toBeUndefined();
    expect(selected.evaluationVideo).toBe(evaluation);
  });

  it("selects and labels an authenticated Demo MP4 independently", () => {
    const demo = {
      artifact_id: "demo-id",
      run_id: "run-demo",
      artifact_type: "demo_video",
      path: "video/overview.mp4",
      size_bytes: 10,
      sha256: "d".repeat(64),
      is_primary_model: false,
      created_at: "2026-09-22T00:00:00Z",
    };
    const selected = selectReplayArtifacts([demo]);
    expect(selected.demoVideo).toBe(demo);
    expect(selected.trainingVideo).toBeUndefined();
    expect(replayPresentation("demo_video").title).toBe("Demo 录像");
    expect(replayPresentation("demo_video", "en").fallbackFilename).toBe("demo-overview.mp4");
  });

  it("maps the frozen video_enabled and Run/Artifact lifecycle", () => {
    expect(runVideoMode(false, "finalizing", false, true, false)).toBe("disabled");
    expect(runVideoMode(false, "running", false, false, false)).toBe("live");
    expect(runVideoMode(true, "running", false, false, true)).toBe("live");
    expect(runVideoMode(true, "finalizing", false, false, false)).toBe("finalizing");
    expect(runVideoMode(true, "completed", false, true, false)).toBe("preparing");
    expect(runVideoMode(true, "completed", true, true, true)).toBe("replay");
    expect(runVideoMode(true, "stopped", false, false, false)).toBe("unavailable");
    expect(runVideoMode(true, "failed", false, false, true)).toBe("error");
  });

  it("accepts the frozen training config video flag when the snapshot top level omits it", () => {
    expect(runVideoEnabled({ training_config: { video_enabled: true } })).toBe(true);
    expect(runVideoEnabled({ video_enabled: true })).toBe(true);
    expect(runVideoEnabled({ video_enabled: false, training_config: { video_enabled: false } })).toBe(false);
    expect(runVideoMode(runVideoEnabled({ training_config: { video_enabled: true } }), "running", false, false, false)).toBe("live");
  });

  it("renders controls, autoplay, muted, playsInline and default loop", () => {
    const markup = renderToStaticMarkup(
      <ReplayVideoElement src="blob:video" loop onError={() => undefined} />,
    );
    expect(markup).toContain("controls=\"\"");
    expect(markup).toContain("autoplay=\"\"");
    expect(markup).toContain("muted=\"\"");
    expect(markup).toContain("playsinline=\"\"");
    expect(markup).toContain("loop=\"\"");
  });

  it("allows loop to be disabled", () => {
    const markup = renderToStaticMarkup(
      <ReplayVideoElement src="blob:video" loop={false} onError={() => undefined} />,
    );
    expect(markup).not.toContain("loop=\"\"");
  });

  it("treats every autoplay rejection as a manual play prompt", () => {
    expect(autoplayNeedsManualPlay(new Error("NotAllowedError"))).toBe(true);
    expect(autoplayNeedsManualPlay(undefined)).toBe(true);
  });

  it("isolates Blob generations and revokes the prior URL", () => {
    const createObjectURL = vi.fn()
      .mockReturnValueOnce("blob:first")
      .mockReturnValueOnce("blob:second");
    const revokeObjectURL = vi.fn();
    vi.stubGlobal("URL", { createObjectURL, revokeObjectURL });
    const lease = new ReplayBlobGeneration();
    const first = lease.begin();
    expect(lease.install(first, new Blob(["first"]))).toBe("blob:first");
    const second = lease.begin();
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:first");
    expect(lease.install(first, new Blob(["stale"]))).toBeNull();
    expect(lease.install(second, new Blob(["second"]))).toBe("blob:second");
    lease.revoke();
    expect(revokeObjectURL).toHaveBeenLastCalledWith("blob:second");
    expect(createObjectURL).toHaveBeenCalledTimes(2);
  });
});
