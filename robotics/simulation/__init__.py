"""Typed HTTP boundary for the external Simulation Service."""

from .client import (
    SimulationClient,
    SimulationClientClosedError,
    SimulationClientError,
    SimulationConfigurationError,
    SimulationNetworkError,
    SimulationRejectedError,
    SimulationResponseError,
    SimulationServerError,
    SimulationTemporaryError,
    SimulationTimeoutError,
)
from .contracts import (
    CreateStreamSessionRequest,
    CreateTaskRequest,
    InputEventAck,
    InputEventRequest,
    InputEventType,
    SimulationTaskResponse,
    SimulationCatalogParameter,
    SimulationCatalogParameterType,
    SimulationCatalogTask,
    SimulationTaskCatalog,
    SimulationTaskStatus,
    StreamDescriptor,
    StreamSessionStatus,
    resolve_catalog_task_parameters,
)
from .dispatch import (
    SimulationDispatchError,
    SimulationDispatchService,
    SimulationSnapshotMismatch,
    build_simulation_config,
)
from .recovery import SimulationRecoveryResult, SimulationRecoveryService
from .stop import SimulationStopService, SimulationStopUnavailable
from .stream import (
    StreamDescriptorMismatch,
    StreamSessionNotAllowed,
    StreamSessionService,
    StreamSessionUnavailable,
)

__all__ = [
    "CreateTaskRequest",
    "CreateStreamSessionRequest",
    "InputEventAck",
    "InputEventRequest",
    "InputEventType",
    "SimulationClient",
    "SimulationClientClosedError",
    "SimulationClientError",
    "SimulationConfigurationError",
    "SimulationNetworkError",
    "SimulationRejectedError",
    "SimulationResponseError",
    "SimulationServerError",
    "SimulationTaskResponse",
    "SimulationCatalogParameter",
    "SimulationCatalogParameterType",
    "SimulationCatalogTask",
    "SimulationTaskCatalog",
    "SimulationTaskStatus",
    "StreamDescriptor",
    "StreamSessionStatus",
    "resolve_catalog_task_parameters",
    "SimulationTemporaryError",
    "SimulationTimeoutError",
    "SimulationDispatchError",
    "SimulationDispatchService",
    "SimulationSnapshotMismatch",
    "build_simulation_config",
    "SimulationRecoveryResult",
    "SimulationRecoveryService",
    "SimulationStopService",
    "SimulationStopUnavailable",
    "StreamDescriptorMismatch",
    "StreamSessionNotAllowed",
    "StreamSessionService",
    "StreamSessionUnavailable",
]

