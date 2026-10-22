export type SimulationCatalogSource = "server" | "fallback";

export type SimulationTaskMode = "training" | "demo";

export type SimulationParameterType =
  | "integer"
  | "number"
  | "string"
  | "boolean"
  | "array";

export interface SimulationArrayItems {
  type: "integer" | "number" | "string" | "boolean";
}

export interface SimulationTaskParameter {
  name: string;
  type: SimulationParameterType;
  title?: string;
  description?: string;
  default: unknown;
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  step?: number;
  enum?: readonly unknown[];
  items?: SimulationArrayItems;
  fixedLength?: number;
}

export interface SimulationTaskDefinition {
  id: string;
  title: string;
  description: string;
  mode: SimulationTaskMode;
  parameters: readonly SimulationTaskParameter[];
  supportsFormalRun: boolean;
  producesCheckpoint: boolean;
  supportsVideo: boolean;
  formalRunReason?: string;
}

export interface SimulationTaskCatalog {
  schemaVersion: string;
  source: SimulationCatalogSource;
  tasks: readonly SimulationTaskDefinition[];
}

export interface TaskParameterValidation {
  valid: boolean;
  values: Record<string, unknown>;
  errors: Record<string, string>;
}

export interface SimulationTaskGroups {
  training: readonly SimulationTaskDefinition[];
  demo: readonly SimulationTaskDefinition[];
}

export const SIMULATION_TASK_CATALOG_PATH = "/api/simulation/task-catalog";

const SERVER_RESOLVED_PARAMETER_NAMES = new Set([
  "task_type",
  "environment_type",
  "run_mode",
  "device",
  "algorithm",
  "rollout_steps",
  "num_updates",
  "checkpoint_every",
  "episode_steps",
  "physics_steps_per_action",
  "update_epochs",
  "batch_size",
  "object_size",
  "object_start",
  "goal_position",
]);

export function isFrontendTaskParameter(name: string): boolean {
  return !SERVER_RESOLVED_PARAMETER_NAMES.has(name);
}

export class SimulationCatalogError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SimulationCatalogError";
  }
}

export class SimulationCatalogRequestError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "SimulationCatalogRequestError";
    this.status = status;
  }
}

type UnknownRecord = Record<string, unknown>;

const isRecord = (value: unknown): value is UnknownRecord =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const cloneValue = (value: unknown): unknown => {
  if (Array.isArray(value)) {
    return value.map(cloneValue);
  }
  if (isRecord(value)) {
    return Object.fromEntries(
      Object.entries(value).map(([key, nestedValue]) => [key, cloneValue(nestedValue)]),
    );
  }
  return value;
};

const readNumber = (value: unknown): number | undefined =>
  typeof value === "number" && Number.isFinite(value) ? value : undefined;

const readText = (value: unknown): string | undefined =>
  typeof value === "string" && value.trim() ? value : undefined;

const parameterFallback = (type: SimulationParameterType): unknown => {
  if (type === "integer" || type === "number") {
    return 0;
  }
  if (type === "boolean") {
    return false;
  }
  if (type === "array") {
    return [];
  }
  return "";
};

const normalizeParameterType = (
  value: unknown,
  defaultValue?: unknown,
): SimulationParameterType => {
  if (value === "integer" || value === "number" || value === "string" || value === "boolean" || value === "array") {
    return value;
  }
  if (value === "float" || value === "float32" || value === "float64") {
    return "number";
  }
  if (value === "number_pair" || value === "number_triplet") {
    return "array";
  }
  if (value === "enum") {
    return typeof defaultValue === "number"
      ? Number.isInteger(defaultValue) ? "integer" : "number"
      : typeof defaultValue === "boolean"
        ? "boolean"
        : "string";
  }
  throw new SimulationCatalogError(`Unsupported simulation parameter type: ${String(value)}`);
};

const normalizeItemType = (value: unknown): SimulationArrayItems["type"] => {
  const type = normalizeParameterType(value);
  if (type === "array") {
    throw new SimulationCatalogError("Nested simulation parameter arrays are not supported");
  }
  return type;
};

const normalizeParameter = (value: unknown, index: number): SimulationTaskParameter => {
  if (!isRecord(value)) {
    throw new SimulationCatalogError(`Simulation parameter ${index} must be an object`);
  }

  const name = readText(value.name);
  if (!name) {
    throw new SimulationCatalogError(`Simulation parameter ${index} has no name`);
  }

  const type = normalizeParameterType(value.type, value.default);
  const rawItems = isRecord(value.items) ? value.items : undefined;
  const minimum = readNumber(value.minimum);
  const maximum = readNumber(value.maximum);
  const rawExclusiveMinimum = value.exclusive_minimum ?? value.exclusiveMinimum;
  const exclusiveMinimum =
    typeof rawExclusiveMinimum === "boolean"
      ? rawExclusiveMinimum && minimum !== undefined
        ? minimum
        : undefined
      : readNumber(rawExclusiveMinimum);
  const defaultValue = value.default === undefined ? parameterFallback(type) : cloneValue(value.default);
  const enumValues = Array.isArray(value.enum) ? value.enum.map(cloneValue) : undefined;

  return {
    name,
    type,
    title: readText(value.title),
    description: readText(value.description),
    default: defaultValue,
    minimum,
    maximum,
    exclusiveMinimum,
    step: readNumber(value.step ?? value.multipleOf),
    enum: enumValues,
    items:
      value.type === "number_pair" || value.type === "number_triplet"
        ? { type: "number" }
        : type === "array" && rawItems
        ? { type: normalizeItemType(rawItems.type) }
        : type === "array"
          ? { type: "number" }
          : undefined,
    fixedLength: value.type === "number_pair" ? 2 : value.type === "number_triplet" ? 3 : undefined,
  };
};

const readTaskList = (payload: unknown): unknown[] => {
  if (Array.isArray(payload)) {
    return payload;
  }
  if (!isRecord(payload)) {
    throw new SimulationCatalogError("Simulation catalog response must be an object or array");
  }
  if (Array.isArray(payload.tasks)) {
    return payload.tasks;
  }
  if (isRecord(payload.data) && Array.isArray(payload.data.tasks)) {
    return payload.data.tasks;
  }
  throw new SimulationCatalogError("Simulation catalog response has no tasks array");
};

const taskMode = (value: UnknownRecord): SimulationTaskMode => {
  const rawMode = value.mode ?? value.task_type ?? value.taskType ?? value.type;
  if (rawMode === "training" || rawMode === "demo") {
    return rawMode;
  }
  throw new SimulationCatalogError(`Unsupported simulation task mode: ${String(rawMode)}`);
};

const explicitBoolean = (value: unknown): boolean | undefined =>
  typeof value === "boolean" ? value : undefined;

const normalizeTask = (value: unknown, index: number): SimulationTaskDefinition => {
  if (!isRecord(value)) {
    throw new SimulationCatalogError(`Simulation task ${index} must be an object`);
  }

  const id = readText(value.id ?? value.task_id ?? value.taskId);
  if (!id) {
    throw new SimulationCatalogError(`Simulation task ${index} has no id`);
  }

  const mode = taskMode(value);
  const parameters = Array.isArray(value.parameters)
    ? value.parameters.map(normalizeParameter)
    : isRecord(value.parameter_schema) && isRecord(value.parameter_schema.properties)
      ? Object.entries(value.parameter_schema.properties).map(([name, schema]) =>
          normalizeParameter({ ...(isRecord(schema) ? schema : {}), name }, 0),
        )
      : [];
  const supportsFormalRun =
    explicitBoolean(value.supports_formal_run ?? value.supportsFormalRun) ??
    (mode === "training" && explicitBoolean(value.supports_training) === true);
  const producesCheckpoint =
    explicitBoolean(value.produces_checkpoint ?? value.producesCheckpoint) ??
    (mode === "training" && supportsFormalRun);
  const supportsVideo =
    explicitBoolean(value.supports_video ?? value.supportsVideo) ??
    parameters.some((parameter) => parameter.name === "video_enabled");
  const formalRunReason = readText(value.formal_run_reason ?? value.formalRunReason) ??
    (supportsFormalRun
      ? undefined
      : mode === "demo"
        ? "Demo tasks are for preview and do not create training checkpoints."
        : "The current backend does not accept this task for formal Run submission.");

  return {
    id,
    title: readText(value.title ?? value.display_name ?? value.name) ?? id,
    description: readText(value.description) ?? "",
    mode,
    parameters,
    supportsFormalRun,
    producesCheckpoint,
    supportsVideo,
    formalRunReason,
  };
};

export function normalizeSimulationCatalog(
  payload: unknown,
  source: SimulationCatalogSource = "server",
): SimulationTaskCatalog {
  const tasks = readTaskList(payload).map(normalizeTask);
  if (tasks.length === 0) {
    throw new SimulationCatalogError("Simulation catalog contains no tasks");
  }

  const schemaVersion =
    isRecord(payload) && typeof payload.schema_version === "string"
      ? payload.schema_version
      : isRecord(payload) && typeof payload.schemaVersion === "string"
        ? payload.schemaVersion
        : "1.0";

  return { schemaVersion, source, tasks };
}

const integerParameter = (
  name: string,
  defaultValue: number,
  options: Partial<SimulationTaskParameter> = {},
): SimulationTaskParameter => ({ name, type: "integer", default: defaultValue, ...options });

const numberParameter = (
  name: string,
  defaultValue: number,
  options: Partial<SimulationTaskParameter> = {},
): SimulationTaskParameter => ({ name, type: "number", default: defaultValue, ...options });

const stringParameter = (
  name: string,
  defaultValue: string,
  options: Partial<SimulationTaskParameter> = {},
): SimulationTaskParameter => ({ name, type: "string", default: defaultValue, ...options });

const booleanParameter = (
  name: string,
  defaultValue: boolean,
  options: Partial<SimulationTaskParameter> = {},
): SimulationTaskParameter => ({ name, type: "boolean", default: defaultValue, ...options });

const arrayParameter = (
  name: string,
  defaultValue: readonly number[],
  options: Partial<SimulationTaskParameter> = {},
): SimulationTaskParameter => ({
  name,
  type: "array",
  default: [...defaultValue],
  items: { type: "number" },
  ...options,
});

const demoVideoParameters = (): SimulationTaskParameter[] => [
  integerParameter("duration_seconds", 30, { minimum: 5, maximum: 300, step: 1 }),
  integerParameter("seed", 7, { minimum: 0, step: 1 }),
  integerParameter("physics_steps_per_command", 1, { minimum: 1, maximum: 16, step: 1 }),
  booleanParameter("video_enabled", true),
  stringParameter("camera_id", "overview", { enum: ["overview"] }),
  integerParameter("width", 640, { minimum: 160, maximum: 1920, step: 1 }),
  integerParameter("height", 480, { minimum: 120, maximum: 1080, step: 1 }),
  integerParameter("fps", 15, { minimum: 1, maximum: 60, step: 1 }),
];

const trainingParameters = (): SimulationTaskParameter[] => [
  integerParameter("seed", 7, { minimum: 0, step: 1 }),
  stringParameter("run_mode", "training", { enum: ["training", "smoke"] }),
  integerParameter("rollout_steps", 200, { minimum: 1, step: 1 }),
  integerParameter("num_updates", 100, { minimum: 1, step: 1 }),
  integerParameter("checkpoint_every", 10, { minimum: 1, step: 1 }),
  integerParameter("episode_steps", 128, { minimum: 1, step: 1 }),
  integerParameter("physics_steps_per_action", 8, { minimum: 1, step: 1 }),
  integerParameter("update_epochs", 1, { minimum: 1, step: 1 }),
  integerParameter("batch_size", 16, { minimum: 1, step: 1 }),
  booleanParameter("video_enabled", true),
  stringParameter("camera_id", "overview", { enum: ["overview"] }),
  integerParameter("width", 640, { minimum: 160, maximum: 1920, step: 1 }),
  integerParameter("height", 480, { minimum: 120, maximum: 1080, step: 1 }),
  integerParameter("fps", 15, { minimum: 1, maximum: 60, step: 1 }),
];

const fallbackTask = (
  task: Omit<SimulationTaskDefinition, "parameters"> & { parameters: SimulationTaskParameter[] },
): SimulationTaskDefinition => task;

export function defaultSimulationCatalog(
  source: SimulationCatalogSource = "fallback",
): SimulationTaskCatalog {
  const tasks: SimulationTaskDefinition[] = [
    fallbackTask({
      id: "s1_push",
      title: "S1 Push",
      description: "Train a policy to push an object toward a goal.",
      mode: "training",
      parameters: trainingParameters(),
      supportsFormalRun: true,
      producesCheckpoint: true,
      supportsVideo: true,
    }),
    fallbackTask({
      id: "s1_reach",
      title: "S1 Reach",
      description: "Reach a target region with the S1 arm.",
      mode: "training",
      parameters: [
        ...trainingParameters(),
        numberParameter("action_delta_m", 0.005, { minimum: 0.001, maximum: 0.01, step: 0.001 }),
        arrayParameter("target_center_xy_m", [0.36, 0]),
        arrayParameter("target_half_extent_xy_m", [0.03, 0.08], { minimum: 0.001, maximum: 0.2 }),
        numberParameter("success_distance_m", 0.02, { minimum: 0.01, maximum: 0.05, step: 0.001 }),
        integerParameter("success_hold_steps", 3, { minimum: 1, step: 1 }),
      ],
      supportsFormalRun: false,
      producesCheckpoint: false,
      supportsVideo: true,
      formalRunReason: "The current backend catalog does not expose a formal Run adapter for S1 Reach.",
    }),
    fallbackTask({
      id: "s1_grasp",
      title: "S1 Grasp",
      description: "Align, descend and close the gripper to achieve a verified grasp.",
      mode: "training",
      parameters: [
        ...trainingParameters(),
        numberParameter("randomize_xy_m", 0, { minimum: 0, maximum: 0.006, step: 0.001 }),
        numberParameter("action_delta_m", 0.005, { minimum: 0.0025, maximum: 0.005, step: 0.0005 }),
      ],
      supportsFormalRun: true,
      producesCheckpoint: true,
      supportsVideo: true,
    }),
    fallbackTask({
      id: "s1_lift",
      title: "S1 Grasp & Lift",
      description: "Lift a stably grasped cube into the 60 mm target band and hold it.",
      mode: "training",
      parameters: [
        ...trainingParameters(),
        numberParameter("action_delta_m", 0.003, { minimum: 0.0025, maximum: 0.005, step: 0.0005 }),
        numberParameter("hold_seconds", 1, { enum: [1, 5] }),
      ],
      supportsFormalRun: true,
      producesCheckpoint: true,
      supportsVideo: true,
    }),
    fallbackTask({
      id: "s1_lower_release",
      title: "S1 Lower & Release",
      description: "Lower a grasped cube, release it in the target zone and verify settling.",
      mode: "training",
      parameters: [
        ...trainingParameters(),
        numberParameter("action_delta_m", 0.003, { enum: [0.003] }),
        numberParameter("release_trigger_threshold", 0, { minimum: -0.8, maximum: 0.8, step: 0.1 }),
      ],
      supportsFormalRun: true,
      producesCheckpoint: true,
      supportsVideo: true,
    }),
    fallbackTask({
      id: "s1_place",
      title: "S1 Transport & Place",
      description: "Transport a validated grasped-and-lifted cube to a collision-free target candidate and hold it stable for one second.",
      mode: "training",
      parameters: [
        ...trainingParameters(),
        numberParameter("action_delta_m", 0.0045, { enum: [0.0045] }),
        numberParameter("hold_seconds", 1, { enum: [1] }),
      ],
      supportsFormalRun: true,
      producesCheckpoint: true,
      supportsVideo: true,
    }),
    fallbackTask({
      id: "demo_s1_target_pose",
      title: "S1 Target Pose",
      description: "Preview target pose control without creating a checkpoint.",
      mode: "demo",
      parameters: [
        ...demoVideoParameters(),
        arrayParameter("target_offset_xy_m", [0.08, 0.05], { minimum: -0.15, maximum: 0.15 }),
      ],
      supportsFormalRun: false,
      producesCheckpoint: false,
      supportsVideo: true,
    }),
    fallbackTask({
      id: "demo_s1_line_tracking",
      title: "S1 Line Tracking",
      description: "Preview straight-line end-effector tracking.",
      mode: "demo",
      parameters: [
        ...demoVideoParameters(),
        numberParameter("line_length_m", 0.15, { minimum: 0.02, maximum: 0.18, step: 0.01 }),
        numberParameter("period_seconds", 6, { minimum: 2, maximum: 30, step: 1 }),
        stringParameter("direction", "x", { enum: ["x", "y"] }),
      ],
      supportsFormalRun: false,
      producesCheckpoint: false,
      supportsVideo: true,
    }),
    fallbackTask({
      id: "demo_s1_circle_tracking",
      title: "S1 Circle Tracking",
      description: "Preview circular end-effector tracking.",
      mode: "demo",
      parameters: [
        ...demoVideoParameters(),
        numberParameter("radius_m", 0.05, { minimum: 0.01, maximum: 0.08, step: 0.01 }),
        numberParameter("period_seconds", 8, { minimum: 2, maximum: 30, step: 1 }),
      ],
      supportsFormalRun: false,
      producesCheckpoint: false,
      supportsVideo: true,
    }),
    fallbackTask({
      id: "demo_s1_joint_position",
      title: "S1 Joint Position",
      description: "Preview a fixed joint pose.",
      mode: "demo",
      parameters: [
        ...demoVideoParameters(),
        stringParameter("pose", "ready", { enum: ["home", "ready", "side"] }),
      ],
      supportsFormalRun: false,
      producesCheckpoint: false,
      supportsVideo: true,
    }),
    fallbackTask({
      id: "demo_s1_joint_sweep",
      title: "S1 Joint Sweep",
      description: "Preview a bounded joint range sweep.",
      mode: "demo",
      parameters: [
        ...demoVideoParameters(),
        stringParameter("joint_index", "all_arm", { enum: ["all_arm", "0", "1", "2", "3", "4", "5"] }),
        numberParameter("speed_scale", 0.2, { minimum: 0.05, maximum: 0.5, step: 0.05 }),
      ],
      supportsFormalRun: false,
      producesCheckpoint: false,
      supportsVideo: true,
    }),
    fallbackTask({
      id: "demo_s1_scripted_pick_place",
      title: "S1 Scripted Pick & Place",
      description: "Run the fixed rule-based grasp, lift, place and release program.",
      mode: "demo",
      parameters: demoVideoParameters().map((parameter) =>
        parameter.name === "physics_steps_per_command"
          ? integerParameter("physics_steps_per_command", 4, { enum: [4] })
          : parameter.name === "width"
            ? integerParameter("width", 640, { enum: [640] })
            : parameter.name === "height"
              ? integerParameter("height", 480, { enum: [480] })
              : parameter,
      ),
      supportsFormalRun: false,
      producesCheckpoint: false,
      supportsVideo: true,
    }),
  ];

  return { schemaVersion: "1.0", source, tasks };
}

export function getSimulationTask(
  catalog: SimulationTaskCatalog,
  taskId: string,
): SimulationTaskDefinition | undefined {
  return catalog.tasks.find((task) => task.id === taskId);
}

export function groupSimulationTasks(catalog: SimulationTaskCatalog): SimulationTaskGroups {
  return {
    training: catalog.tasks.filter((task) => task.mode === "training"),
    demo: catalog.tasks.filter((task) => task.mode === "demo"),
  };
}

export function getTaskDefaults(task: SimulationTaskDefinition): Record<string, unknown> {
  return Object.fromEntries(
    task.parameters
      .filter((parameter) => isFrontendTaskParameter(parameter.name))
      .map((parameter) => [parameter.name, cloneValue(parameter.default)]),
  );
}

const numericValues = (parameter: SimulationTaskParameter, value: unknown): number[] | undefined => {
  if (parameter.type === "integer" || parameter.type === "number") {
    return typeof value === "number" && Number.isFinite(value) ? [value] : undefined;
  }
  if (parameter.type === "array" && Array.isArray(value)) {
    return value.every((item) => typeof item === "number" && Number.isFinite(item))
      ? value
      : undefined;
  }
  return undefined;
};

const sameValue = (left: unknown, right: unknown): boolean => {
  if (Array.isArray(left) && Array.isArray(right)) {
    return left.length === right.length && left.every((item, index) => sameValue(item, right[index]));
  }
  return Object.is(left, right);
};

const itemMatchesType = (type: SimulationArrayItems["type"], value: unknown): boolean => {
  if (type === "integer") {
    return typeof value === "number" && Number.isInteger(value);
  }
  if (type === "number") {
    return typeof value === "number" && Number.isFinite(value);
  }
  if (type === "string") {
    return typeof value === "string";
  }
  return typeof value === "boolean";
};

const valueMatchesType = (parameter: SimulationTaskParameter, value: unknown): boolean => {
  if (parameter.type === "integer") {
    return typeof value === "number" && Number.isInteger(value) && Number.isFinite(value);
  }
  if (parameter.type === "number") {
    return typeof value === "number" && Number.isFinite(value);
  }
  if (parameter.type === "string") {
    return typeof value === "string";
  }
  if (parameter.type === "boolean") {
    return typeof value === "boolean";
  }
  return Array.isArray(value)
    && (parameter.fixedLength === undefined || value.length === parameter.fixedLength)
    && value.every((item) => itemMatchesType(parameter.items?.type ?? "number", item));
};

const boundaryError = (parameter: SimulationTaskParameter, value: unknown): string | undefined => {
  const numbers = numericValues(parameter, value);
  if (!numbers) {
    return undefined;
  }
  if (parameter.exclusiveMinimum !== undefined && numbers.some((item) => item <= parameter.exclusiveMinimum!)) {
    return `${parameter.name} must be greater than ${parameter.exclusiveMinimum}`;
  }
  if (parameter.minimum !== undefined && parameter.exclusiveMinimum === undefined && numbers.some((item) => item < parameter.minimum!)) {
    return `${parameter.name} must be at least ${parameter.minimum}`;
  }
  if (parameter.maximum !== undefined && numbers.some((item) => item > parameter.maximum!)) {
    return `${parameter.name} must be at most ${parameter.maximum}`;
  }
  if (parameter.step !== undefined && numbers.some((item) => {
    const quotient = item / parameter.step!;
    return Math.abs(quotient - Math.round(quotient)) > 1e-8;
  })) {
    return `${parameter.name} must use increments of ${parameter.step}`;
  }
  return undefined;
};

const validateValue = (parameter: SimulationTaskParameter, value: unknown): string | undefined => {
  if (parameter.enum) {
    return parameter.enum.some((candidate) => sameValue(candidate, value))
      ? boundaryError(parameter, value)
      : `${parameter.name} must be one of the catalog enum values`;
  }
  if (!valueMatchesType(parameter, value)) {
    return `${parameter.name} has an invalid ${parameter.type} value`;
  }
  return boundaryError(parameter, value);
};

export function validateTaskParameters(
  task: SimulationTaskDefinition,
  parameters: Record<string, unknown> | undefined,
): TaskParameterValidation {
  const input = parameters ?? {};
  const values = getTaskDefaults(task);
  const errors: Record<string, string> = {};
  const allowed = new Set(task.parameters.map((parameter) => parameter.name));

  for (const key of Object.keys(input)) {
    if (!allowed.has(key)) {
      errors[key] = "This parameter is not allowed by the selected catalog task";
    }
  }

  for (const parameter of task.parameters) {
    if (Object.prototype.hasOwnProperty.call(input, parameter.name)) {
      const value = input[parameter.name];
      values[parameter.name] = cloneValue(value);
      const error = validateValue(parameter, value);
      if (error) {
        errors[parameter.name] = error;
      }
    }
  }

  return { valid: Object.keys(errors).length === 0, values, errors };
}

export function resolveTaskParameters(
  task: SimulationTaskDefinition,
  initialParameters: Record<string, unknown> | undefined,
): Record<string, unknown> {
  const defaults = getTaskDefaults(task);
  if (!initialParameters) return defaults;
  const candidate = { ...defaults, ...initialParameters };
  const validation = validateTaskParameters(task, candidate);
  return validation.valid ? validation.values : defaults;
}

const clampNumber = (parameter: SimulationTaskParameter, value: number): number => {
  let result = value;
  const minimum = parameter.exclusiveMinimum ?? parameter.minimum;
  if (minimum !== undefined && result < minimum) {
    result = parameter.exclusiveMinimum !== undefined ? minimum + (parameter.step ?? 1) : minimum;
  }
  if (parameter.maximum !== undefined && result > parameter.maximum) {
    result = parameter.maximum;
  }
  return parameter.type === "integer" || parameter.items?.type === "integer" ? Math.round(result) : result;
};

export function clampTaskParameters(
  task: SimulationTaskDefinition,
  parameters: Record<string, unknown> | undefined,
): Record<string, unknown> {
  const values = getTaskDefaults(task);
  const input = parameters ?? {};

  for (const parameter of task.parameters) {
    if (!Object.prototype.hasOwnProperty.call(input, parameter.name)) {
      continue;
    }
    const value = input[parameter.name];
    if (parameter.type === "integer" || parameter.type === "number") {
      values[parameter.name] = typeof value === "number" && Number.isFinite(value)
        ? clampNumber(parameter, value)
        : cloneValue(parameter.default);
    } else if (parameter.type === "array" && Array.isArray(value)) {
      values[parameter.name] = value.map((item) =>
        typeof item === "number" && Number.isFinite(item) && (parameter.items?.type === "integer" || parameter.items?.type === "number")
          ? clampNumber(parameter, item)
          : cloneValue(item),
      );
    } else {
      values[parameter.name] = cloneValue(value);
    }
  }

  return values;
}

export function taskParameterPayload(
  task: SimulationTaskDefinition,
  parameters: Record<string, unknown> | undefined,
): Record<string, unknown> {
  const validation = validateTaskParameters(task, parameters);
  if (!validation.valid) {
    throw new SimulationCatalogError(`Invalid parameters for ${task.id}: ${Object.keys(validation.errors).join(", ")}`);
  }
  return Object.fromEntries(
    Object.entries(validation.values).filter(([name]) => isFrontendTaskParameter(name)),
  );
}

export function serializeTaskParameters(
  task: SimulationTaskDefinition,
  parameters: Record<string, unknown> | undefined,
): string {
  return JSON.stringify(taskParameterPayload(task, parameters));
}

export async function fetchSimulationCatalog(
  fetcher: typeof fetch = fetch,
  path: string = SIMULATION_TASK_CATALOG_PATH,
): Promise<SimulationTaskCatalog> {
  const response = await fetcher(path, { headers: { Accept: "application/json" } });
  if (response.ok) {
    return normalizeSimulationCatalog(await response.json(), "server");
  }
  if (response.status === 404 || response.status === 405) {
    return defaultSimulationCatalog("fallback");
  }
  let message = `Simulation catalog request failed with HTTP ${response.status}`;
  try {
    const body = await response.json();
    if (isRecord(body) && typeof body.message === "string") {
      message = body.message;
    }
  } catch {
    // Keep the status-based error when the server body is not JSON.
  }
  throw new SimulationCatalogRequestError(response.status, message);
}
