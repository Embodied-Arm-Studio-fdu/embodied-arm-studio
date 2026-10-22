export type OnboardingStatus = "not_started" | "in_progress" | "completed" | "skipped";

export interface OnboardingRecord {
  firstTimeOnboarding: {
    version: number;
    status: OnboardingStatus;
    completedAt?: string;
  };
}

export interface TourStorageAdapter {
  hasSeenFirstTimeOnboarding(userId?: string): boolean;
  markFirstTimeOnboardingCompleted(userId?: string): void;
  markFirstTimeOnboardingSkipped(userId?: string): void;
  resetOnboardingForDebug(userId?: string): void;
}

export const onboardingStorageKey = (userId?: string) =>
  `embodied-arm:onboarding:${userId || "anonymous"}`;

export function createTourStorageAdapter(storage: Pick<Storage, "getItem" | "setItem" | "removeItem">): TourStorageAdapter {
  const read = (userId?: string): OnboardingRecord | null => {
    try {
      return JSON.parse(storage.getItem(onboardingStorageKey(userId)) || "null") as OnboardingRecord | null;
    } catch {
      return null;
    }
  };
  const mark = (status: "completed" | "skipped", userId?: string) => {
    storage.setItem(onboardingStorageKey(userId), JSON.stringify({
      firstTimeOnboarding: { version: 1, status, completedAt: new Date().toISOString() },
    } satisfies OnboardingRecord));
  };
  return {
    hasSeenFirstTimeOnboarding: (userId) => {
      const status = read(userId)?.firstTimeOnboarding?.status;
      return status === "completed" || status === "skipped";
    },
    markFirstTimeOnboardingCompleted: (userId) => mark("completed", userId),
    markFirstTimeOnboardingSkipped: (userId) => mark("skipped", userId),
    resetOnboardingForDebug: (userId) => storage.removeItem(onboardingStorageKey(userId)),
  };
}

export const tourStorage = createTourStorageAdapter({
  getItem: (key) => localStorage.getItem(key),
  setItem: (key, value) => localStorage.setItem(key, value),
  removeItem: (key) => localStorage.removeItem(key),
});
