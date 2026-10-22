import { create } from "zustand";
import type { LegacyRunRecord, MetricEvent } from "./api";

interface StudioState {
  activeStep: number;
  completedSteps: string[];
  selectedStrategy: string;
  config: Record<string, number>;
  currentRun: LegacyRunRecord | null;
  metrics: MetricEvent[];
  projectId: string;
  activeSceneId: string;
  setStep: (step: number) => void;
  completeStep: (id: string) => void;
  setStrategy: (strategy: string) => void;
  setConfig: (config: Record<string, number>) => void;
  beginRun: (run: LegacyRunRecord) => void;
  setRunStatus: (status: LegacyRunRecord["status"], error?: string | null) => void;
  appendEvent: (event: MetricEvent) => void;
  setProject: (projectId: string) => void;
  setActiveScene: (sceneId: string) => void;
}

export const useStudio = create<StudioState>((set) => ({
  activeStep: 0,
  completedSteps: [],
  selectedStrategy: "joint-p-control",
  config: { action_scale: 0.35, reward_distance_weight: 1, learning_rate: 0.0003 },
  currentRun: null,
  metrics: [],
  projectId: "",
  activeSceneId: "",
  setStep: (activeStep) => set({ activeStep }),
  completeStep: (id) => set((state) => ({ completedSteps: [...new Set([...state.completedSteps, id])] })),
  setStrategy: (selectedStrategy) => set({ selectedStrategy }),
  setConfig: (config) => set({ config }),
  beginRun: (currentRun) => set({ currentRun, metrics: [] }),
  setRunStatus: (status, error = null) => set((state) => ({
    currentRun: state.currentRun
      ? { ...state.currentRun, status, error }
      : state.currentRun,
  })),
  appendEvent: (event) => set((state) => ({
    metrics: event.type === "metric" ? [...state.metrics, event] : state.metrics,
    currentRun: ["state", "status"].includes(event.type) && state.currentRun
      ? { ...state.currentRun, status: event.payload.status ?? state.currentRun.status }
      : state.currentRun,
  })),
  setProject: (projectId) => set({ projectId }),
  setActiveScene: (activeSceneId) => set({ activeSceneId }),
}));