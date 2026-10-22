import type { MetricEvent, RunStatus } from "./api";
import type { RuntimeLocale } from "./runtimeMode";

const runStatuses = new Set<RunStatus>([
  "queued",
  "initializing",
  "running",
  "finalizing",
  "stopping",
  "completed",
  "stopped",
  "failed",
]);
const terminalStatuses = new Set<RunStatus>(["completed", "stopped", "failed"]);

const terminalConnectionLabels: Record<"completed" | "stopped" | "failed", string> = {
  completed: "训练已完成，结果已保存",
  stopped: "训练已停止，结果已保存",
  failed: "训练失败，实时连接已关闭",
};

const terminalConnectionLabelsEn: Record<"completed" | "stopped" | "failed", string> = {
  completed: "Training completed; results saved",
  stopped: "Training stopped; results saved",
  failed: "Training failed; live connection closed",
};

export function isTerminalRunStatus(status: RunStatus): boolean {
  return terminalStatuses.has(status);
}

export function runConnectionLabel(status: RunStatus, locale: RuntimeLocale = "zh-CN"): string | null {
  return isTerminalRunStatus(status)
    ? (locale === "en" ? terminalConnectionLabelsEn : terminalConnectionLabels)[status as keyof typeof terminalConnectionLabels]
    : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function runStatus(value: unknown): RunStatus | undefined {
  return typeof value === "string" && runStatuses.has(value as RunStatus)
    ? value as RunStatus
    : undefined;
}

export function normalizeRunEvent(value: unknown): MetricEvent | null {
  if (!isRecord(value) || typeof value.event_id !== "string") return null;
  const sequence = typeof value.sequence === "number" ? value.sequence : undefined;
  const rawPayload = isRecord(value.payload) ? value.payload : {};
  const status = runStatus(
    rawPayload.status ?? rawPayload.state ?? value.status ?? value.state,
  );
  const type = value.type === "state" || value.type === "status"
    ? value.type
    : value.type === "metric" || value.type === "error"
      ? value.type
      : null;
  if (!type) return null;

  return {
    event_id: value.event_id,
    sequence,
    run_id: typeof value.run_id === "string" ? value.run_id : undefined,
    occurred_at: typeof value.occurred_at === "string" ? value.occurred_at : undefined,
    schema_version: typeof value.schema_version === "string" ? value.schema_version : undefined,
    type,
    payload: {
      ...rawPayload,
      ...(status ? { status } : {}),
    } as MetricEvent["payload"],
  };
}

export function normalizeRunEventsResponse(value: unknown): MetricEvent[] {
  const record = isRecord(value) ? value : null;
  const data = record?.data;
  const values = Array.isArray(value)
    ? value
    : Array.isArray(data)
      ? data
      : isRecord(data) && Array.isArray(data.events)
        ? data.events
        : record && Array.isArray(record.events)
          ? record.events
          : [];
  return values.flatMap((item) => {
    const event = normalizeRunEvent(item);
    return event ? [event] : [];
  });
}

export function mergeRunEvents(
  current: MetricEvent[],
  incoming: MetricEvent[],
): MetricEvent[] {
  const known = new Set(current.map((event) => event.event_id));
  return [
    ...current,
    ...incoming.filter((event) => {
      if (known.has(event.event_id)) return false;
      known.add(event.event_id);
      return true;
    }),
  ];
}

export function latestRunEventSequence(events: MetricEvent[], current = 0): number {
  return events.reduce(
    (latest, event) => Math.max(latest, event.sequence ?? latest),
    current,
  );
}

export function reconcileRunStatus(
  currentStatus: RunStatus,
  currentSequence: number,
  event: MetricEvent,
): RunStatus {
  const status = event.payload.status;
  if (!status || (event.sequence ?? 0) < currentSequence) return currentStatus;
  if (terminalStatuses.has(currentStatus) && !terminalStatuses.has(status)) return currentStatus;
  return status;
}