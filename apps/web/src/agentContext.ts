export interface AgentPageContext {
  path: string;
  guide_version?: string;
  guide_section_id?: string;
  guide_section_title?: string;
  guide_summary?: string;
  guide_boundaries?: string[];
  guide_steps?: string[];
  guide_principles?: string[];
  project_id?: string;
  run_id?: string;
  scene_id?: string;
  evaluation_id?: string;
}

export function buildAgentPageContext(
  path: string,
  projectId: string,
  currentRunId: string | undefined,
  activeSceneId: string,
  evaluationId?: string,
  guide?: { version: string; id: string; title: string; summary: string; boundaries: string[]; steps: string[]; principles?: string[] },
): AgentPageContext {
  const runId = path.startsWith("/runs/")
    ? decodeURIComponent(path.slice("/runs/".length))
    : currentRunId;
  return {
    path,
    ...(guide ? { guide_version: guide.version, guide_section_id: guide.id, guide_section_title: guide.title, guide_summary: guide.summary, guide_boundaries: guide.boundaries, guide_steps: guide.steps, ...(guide.principles ? { guide_principles: guide.principles } : {}) } : {}),
    ...(projectId ? { project_id: projectId } : {}),
    ...(runId ? { run_id: runId } : {}),
    ...(activeSceneId ? { scene_id: activeSceneId } : {}),
    ...(evaluationId ? { evaluation_id: evaluationId } : {}),
  };
}