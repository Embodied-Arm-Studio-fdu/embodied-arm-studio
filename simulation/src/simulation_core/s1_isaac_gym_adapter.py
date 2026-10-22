"""Minimal, task-independent Isaac Gym adapter for the YHRG S1 robot.

Isaac Gym is imported lazily so importing :mod:`simulation_core` and running the
ordinary unit test suite never initializes CUDA or a simulator.
"""

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
from typing import Any, Callable, Optional, Tuple


DEFAULT_ASSET_ROOT = Path(
    "/home/user/DyWA/dywa/src/data/assets/yhrg_s1_description"
)
DEFAULT_ROBOT_URDF = Path("urdf/yhrg_s1_v2.urdf")
EXPECTED_DOF_NAMES = (
    "1joint",
    "2joint",
    "3joint",
    "4joint",
    "5joint",
    "6joint",
    "7joint",
    "8joint",
)
ARM_DOF_COUNT = 6
GRIPPER_DOF_COUNT = 2
EXPECTED_BODY_NAMES = ("ee_link", "6_Link", "7_Link", "8_Link")
DEFAULT_INITIAL_DOF_POSITIONS = (
    0.0,
    0.785,
    1.57,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)


class S1AdapterError(RuntimeError):
    """Base exception for adapter lifecycle and validation failures."""


class S1AssetValidationError(S1AdapterError):
    """Raised when the configured S1 asset is absent or structurally invalid."""


class S1AdapterStateError(S1AdapterError):
    """Raised when adapter operations are called out of lifecycle order."""


@dataclass(frozen=True)
class S1IsaacGymConfig:
    """Explicit configuration for one headless S1 Isaac Gym environment."""

    asset_root: Path = field(default_factory=lambda: Path(
        os.environ.get("SIMULATION_S1_ASSET_ROOT") or DEFAULT_ASSET_ROOT
    ))
    robot_urdf: Path = DEFAULT_ROBOT_URDF
    compute_device_id: int = 0
    graphics_device_id: int = -1
    physics_dt: float = 1.0 / 60.0
    substeps: int = 2
    headless: bool = True
    initial_dof_positions: Tuple[float, ...] = DEFAULT_INITIAL_DOF_POSITIONS

    def __post_init__(self) -> None:
        object.__setattr__(self, "asset_root", Path(self.asset_root))
        object.__setattr__(self, "robot_urdf", Path(self.robot_urdf))
        object.__setattr__(
            self, "initial_dof_positions", tuple(self.initial_dof_positions)
        )
        if self.robot_urdf.is_absolute():
            raise ValueError("robot_urdf must be relative to asset_root")
        if len(self.initial_dof_positions) != ARM_DOF_COUNT + GRIPPER_DOF_COUNT:
            raise ValueError("initial_dof_positions must contain exactly 8 values")
        if self.compute_device_id < 0:
            raise ValueError("compute_device_id must be non-negative")
        if self.graphics_device_id < -1:
            raise ValueError("graphics_device_id must be -1 or non-negative")
        if self.physics_dt <= 0:
            raise ValueError("physics_dt must be positive")
        if self.substeps <= 0:
            raise ValueError("substeps must be positive")

    @property
    def urdf_path(self) -> Path:
        return self.asset_root / self.robot_urdf

    def validate_asset_path(self) -> Path:
        if not self.asset_root.is_dir():
            raise S1AssetValidationError(
                "S1 asset root is not a directory: %s" % self.asset_root
            )
        if not self.urdf_path.is_file():
            raise S1AssetValidationError(
                "S1 URDF does not exist: %s" % self.urdf_path
            )
        return self.urdf_path


@dataclass(frozen=True)
class S1State:
    """Detached snapshots returned by :meth:`S1IsaacGymAdapter.read_state`."""

    dof_positions: Any
    dof_velocities: Any
    ee_link_state: Any
    arm_ee_jacobian: Any


def _load_isaac_gym_api() -> Tuple[Any, Any]:
    try:
        from isaacgym import gymapi, gymtorch
    except ImportError as exc:
        raise S1AdapterError(
            "Isaac Gym Preview 4 could not be imported; use its configured Python "
            "environment before initializing S1IsaacGymAdapter"
        ) from exc
    return gymapi, gymtorch


class S1IsaacGymAdapter:
    """Own one Isaac Gym sim, one environment, and one S1 actor."""

    ACTOR_NAME = "robot"

    def __init__(
        self,
        config: Optional[S1IsaacGymConfig] = None,
        api_loader: Callable[[], Tuple[Any, Any]] = _load_isaac_gym_api,
    ) -> None:
        self.config = config or S1IsaacGymConfig()
        self._api_loader = api_loader
        self._gymapi = None
        self._gymtorch = None
        self.gym = None
        self.sim = None
        self.env = None
        self.asset = None
        self.actor = None
        self.viewer = None
        self._dof_state = None
        self._rigid_body_state = None
        self._jacobian = None
        self._position_targets = None
        self._body_indices = {}
        self._ee_jacobian_row = None
        self._prepared = False
        self._closed = False

    def __enter__(self) -> "S1IsaacGymAdapter":
        return self.initialize()

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False

    def _require_open(self) -> None:
        if self._closed:
            raise S1AdapterStateError("adapter is closed")

    def initialize(self) -> "S1IsaacGymAdapter":
        """Acquire Isaac Gym and create a PhysX simulator.

        A partially created simulator is destroyed if initialization raises.
        """
        self._require_open()
        if self.sim is not None:
            return self
        try:
            self._gymapi, self._gymtorch = self._api_loader()
            self.gym = self._gymapi.acquire_gym()
            params = self._gymapi.SimParams()
            params.up_axis = self._gymapi.UP_AXIS_Z
            params.gravity = self._gymapi.Vec3(0.0, 0.0, -9.81)
            params.dt = self.config.physics_dt
            params.substeps = self.config.substeps
            params.use_gpu_pipeline = True
            params.physx.solver_type = 1
            params.physx.num_position_iterations = 8
            params.physx.num_velocity_iterations = 1
            params.physx.use_gpu = True
            self.sim = self.gym.create_sim(
                self.config.compute_device_id,
                self.config.graphics_device_id,
                self._gymapi.SIM_PHYSX,
                params,
            )
            if self.sim is None:
                raise S1AdapterError("Isaac Gym failed to create the PhysX simulator")
            return self
        except Exception:
            self.close()
            raise

    def load_asset(self) -> Any:
        """Load S1 v2 directly and validate its public robot structure."""
        self._require_open()
        if self.sim is None:
            raise S1AdapterStateError("initialize() must be called before load_asset()")
        self.config.validate_asset_path()
        try:
            options = self._gymapi.AssetOptions()
            options.fix_base_link = True
            options.disable_gravity = True
            options.flip_visual_attachments = False
            options.enable_gyroscopic_forces = True
            options.vhacd_enabled = True
            options.convex_decomposition_from_submeshes = True
            self.asset = self.gym.load_urdf(
                self.sim,
                str(self.config.asset_root),
                str(self.config.robot_urdf),
                options,
            )
            if self.asset is None:
                raise S1AssetValidationError(
                    "Isaac Gym failed to load S1 URDF: %s" % self.config.urdf_path
                )
            self._validate_loaded_asset()
            return self.asset
        except Exception:
            self.close()
            raise

    def _validate_loaded_asset(self) -> None:
        dof_count = self.gym.get_asset_dof_count(self.asset)
        if dof_count != len(EXPECTED_DOF_NAMES):
            raise S1AssetValidationError(
                "S1 must have 8 DOFs, but Isaac Gym loaded %d" % dof_count
            )
        actual_dofs = tuple(self.gym.get_asset_dof_names(self.asset))
        if actual_dofs != EXPECTED_DOF_NAMES:
            raise S1AssetValidationError(
                "unexpected S1 DOF order: expected %r, got %r"
                % (EXPECTED_DOF_NAMES, actual_dofs)
            )
        body_names = tuple(self.gym.get_asset_rigid_body_names(self.asset))
        missing = tuple(name for name in EXPECTED_BODY_NAMES if name not in body_names)
        if missing:
            raise S1AssetValidationError(
                "S1 asset is missing required rigid bodies: %s" % ", ".join(missing)
            )

    def create_env(
        self,
        actor_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        actor_yaw: float = 0.0,
    ) -> Any:
        """Create exactly one environment and one position-controlled S1 actor."""
        self._require_open()
        if self.asset is None:
            raise S1AdapterStateError("load_asset() must be called before create_env()")
        if (
            len(actor_position) != 3
            or not all(isinstance(value, (int, float)) for value in actor_position)
        ):
            raise ValueError("actor_position must contain three numeric values")
        if not isinstance(actor_yaw, (int, float)) or not math.isfinite(actor_yaw):
            raise ValueError("actor_yaw must be a finite numeric value")
        try:
            self.env = self.gym.create_env(
                self.sim,
                self._gymapi.Vec3(-1.0, -1.0, 0.0),
                self._gymapi.Vec3(1.0, 1.0, 1.0),
                1,
            )
            if self.env is None:
                raise S1AdapterError("Isaac Gym failed to create the S1 environment")
            actor_pose = self._gymapi.Transform()
            actor_pose.p = self._gymapi.Vec3(*actor_position)
            actor_pose.r = self._gymapi.Quat(
                0.0, 0.0, math.sin(actor_yaw / 2.0), math.cos(actor_yaw / 2.0)
            )
            self.actor = self.gym.create_actor(
                self.env,
                self.asset,
                actor_pose,
                self.ACTOR_NAME,
                0,
                0,
            )
            if self.actor is None or self.actor < 0:
                raise S1AdapterError("Isaac Gym failed to create the S1 actor")
            self._configure_dofs()
            return self.env
        except Exception:
            self.close()
            raise

    def _configure_dofs(self) -> None:
        properties = self.gym.get_actor_dof_properties(self.env, self.actor)
        properties["driveMode"][:ARM_DOF_COUNT] = self._gymapi.DOF_MODE_POS
        properties["stiffness"][:ARM_DOF_COUNT] = 300.0
        properties["damping"][:ARM_DOF_COUNT] = 40.0
        properties["driveMode"][ARM_DOF_COUNT:] = self._gymapi.DOF_MODE_POS
        properties["stiffness"][ARM_DOF_COUNT:] = 1000.0
        properties["damping"][ARM_DOF_COUNT:] = 100.0
        self.gym.set_actor_dof_properties(self.env, self.actor, properties)

        self.reset_dofs()

    def reset_dofs(self) -> None:
        """Restore the configured eight-DOF pose and zero joint velocities."""
        self._require_open()
        if self.actor is None:
            raise S1AdapterStateError("create_env() must be called before reset_dofs()")
        properties = self.gym.get_actor_dof_properties(self.env, self.actor)

        lower = tuple(properties["lower"])
        upper = tuple(properties["upper"])
        for index, value in enumerate(self.config.initial_dof_positions):
            if value < lower[index] or value > upper[index]:
                raise S1AssetValidationError(
                    "initial DOF %d (%s) value %s is outside [%s, %s]"
                    % (index, EXPECTED_DOF_NAMES[index], value, lower[index], upper[index])
                )

        if self._prepared:
            self._dof_state[0, :, 0] = self._dof_state.new_tensor(
                self.config.initial_dof_positions
            )
            self._dof_state[0, :, 1] = 0.0
            self._position_targets = self._dof_state[:, :, 0].contiguous()
            self.gym.set_dof_state_tensor(
                self.sim, self._gymtorch.unwrap_tensor(self._dof_state)
            )
            self.gym.set_dof_position_target_tensor(
                self.sim, self._gymtorch.unwrap_tensor(self._position_targets)
            )
            self._refresh_tensors()
            return

        states = self.gym.get_actor_dof_states(
            self.env, self.actor, self._gymapi.STATE_ALL
        )
        states["pos"][:] = self.config.initial_dof_positions
        states["vel"][:] = 0.0
        self.gym.set_actor_dof_states(
            self.env, self.actor, states, self._gymapi.STATE_ALL
        )
        position_targets = states["pos"].copy()
        self.gym.set_actor_dof_position_targets(
            self.env, self.actor, position_targets
        )

    def prepare_and_acquire(self) -> None:
        """Prepare simulation and acquire validated Gym-backed Torch tensors."""
        self._require_open()
        if self.actor is None:
            raise S1AdapterStateError(
                "create_env() must be called before prepare_and_acquire()"
            )
        if self._prepared:
            return
        try:
            self.gym.prepare_sim(self.sim)
            dof_descriptor = self.gym.acquire_dof_state_tensor(self.sim)
            body_descriptor = self.gym.acquire_rigid_body_state_tensor(self.sim)
            jacobian_descriptor = self.gym.acquire_jacobian_tensor(
                self.sim, self.ACTOR_NAME
            )
            self._dof_state = self._gymtorch.wrap_tensor(dof_descriptor).view(1, 8, 2)
            self._rigid_body_state = self._gymtorch.wrap_tensor(body_descriptor)
            self._jacobian = self._gymtorch.wrap_tensor(jacobian_descriptor)

            self._resolve_body_indices()
            self._validate_tensor_shapes()
            self._prepared = True
            self._refresh_tensors()
        except Exception:
            self.close()
            raise

    def _resolve_body_indices(self) -> None:
        for name in EXPECTED_BODY_NAMES:
            index = self.gym.find_actor_rigid_body_index(
                self.env,
                self.actor,
                name,
                self._gymapi.IndexDomain.DOMAIN_SIM,
            )
            if index < 0:
                raise S1AssetValidationError(
                    "could not resolve SIM-domain rigid body: %s" % name
                )
            self._body_indices[name] = index
        asset_body_index = self.gym.find_asset_rigid_body_index(self.asset, "ee_link")
        self._ee_jacobian_row = asset_body_index - 1
        if self._ee_jacobian_row < 0:
            raise S1AssetValidationError("ee_link cannot be mapped to a Jacobian row")

    def _validate_tensor_shapes(self) -> None:
        if tuple(self._dof_state.shape) != (1, 8, 2):
            raise S1AssetValidationError(
                "unexpected DOF state tensor shape: %r" % (tuple(self._dof_state.shape),)
            )
        if self._rigid_body_state.ndim != 2 or self._rigid_body_state.shape[1] != 13:
            raise S1AssetValidationError(
                "unexpected rigid-body state tensor shape: %r"
                % (tuple(self._rigid_body_state.shape),)
            )
        if self._body_indices["ee_link"] >= self._rigid_body_state.shape[0]:
            raise S1AssetValidationError("ee_link state index is outside body tensor")
        expected_jacobian_tail = (6, 8)
        if (
            self._jacobian.ndim != 4
            or self._jacobian.shape[0] != 1
            or tuple(self._jacobian.shape[-2:]) != expected_jacobian_tail
            or self._ee_jacobian_row >= self._jacobian.shape[1]
        ):
            raise S1AssetValidationError(
                "unexpected Jacobian tensor shape %r for ee_link row %s"
                % (tuple(self._jacobian.shape), self._ee_jacobian_row)
            )

    def _refresh_tensors(self) -> None:
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

    def read_state(self) -> S1State:
        """Refresh and return detached snapshots, never the internal tensor table."""
        self._require_open()
        if not self._prepared:
            raise S1AdapterStateError(
                "prepare_and_acquire() must be called before read_state()"
            )
        self._refresh_tensors()
        return S1State(
            dof_positions=self._dof_state[0, :, 0].detach().clone(),
            dof_velocities=self._dof_state[0, :, 1].detach().clone(),
            ee_link_state=self._rigid_body_state[
                self._body_indices["ee_link"]
            ].detach().clone(),
            arm_ee_jacobian=self._jacobian[
                0, self._ee_jacobian_row, :, :ARM_DOF_COUNT
            ].detach().clone(),
        )

    def set_joint_position_targets(self, positions: Any) -> None:
        """Apply one explicit eight-DOF position target through the GPU pipeline."""
        self._require_open()
        if not self._prepared:
            raise S1AdapterStateError(
                "prepare_and_acquire() must be called before setting targets"
            )
        if hasattr(positions, "detach"):
            targets = positions.detach().to(
                device=self._dof_state.device, dtype=self._dof_state.dtype
            ).clone().reshape(-1)
        else:
            targets = self._dof_state.new_tensor(positions).reshape(-1)
        if targets.numel() != 8:
            raise ValueError("joint position target must contain exactly 8 values")
        if not targets.isfinite().all().item():
            raise ValueError("joint position target must contain finite values")
        properties = self.gym.get_actor_dof_properties(self.env, self.actor)
        lower = targets.new_tensor(properties["lower"])
        upper = targets.new_tensor(properties["upper"])
        if ((targets < lower) | (targets > upper)).any().item():
            raise ValueError("joint position target is outside S1 DOF limits")
        self._position_targets = targets.view(1, 8).contiguous()
        self.gym.set_dof_position_target_tensor(
            self.sim, self._gymtorch.unwrap_tensor(self._position_targets)
        )

    @property
    def body_indices(self) -> dict:
        """Return a copy of resolved SIM-domain indices for diagnostics."""
        return dict(self._body_indices)

    def step(self, n: int = 1) -> None:
        """Advance physics without policy, reward, observation, or task logic."""
        self._require_open()
        if not self._prepared:
            raise S1AdapterStateError(
                "prepare_and_acquire() must be called before step()"
            )
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError("n must be a non-negative integer")
        for _ in range(n):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self._refresh_tensors()

    def setup(self) -> "S1IsaacGymAdapter":
        """Convenience composition for the complete minimal setup sequence."""
        try:
            self.initialize()
            self.load_asset()
            self.create_env()
            self.prepare_and_acquire()
            return self
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """Idempotently release viewer/simulator and all borrowed references."""
        if self._closed:
            return
        gym = self.gym
        sim = self.sim
        viewer = self.viewer
        try:
            if gym is not None and viewer is not None:
                gym.destroy_viewer(viewer)
            if gym is not None and sim is not None:
                gym.destroy_sim(sim)
        finally:
            self._dof_state = None
            self._rigid_body_state = None
            self._jacobian = None
            self._position_targets = None
            self._body_indices = {}
            self._ee_jacobian_row = None
            self._prepared = False
            self.viewer = None
            self.actor = None
            self.env = None
            self.asset = None
            self.sim = None
            self.gym = None
            self._gymapi = None
            self._gymtorch = None
            self._closed = True


def main() -> None:
    """Run the explicit real-GPU smoke test; ordinary tests never call this."""
    import argparse

    parser = argparse.ArgumentParser(description="Minimal S1 Isaac Gym smoke test")
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    adapter = S1IsaacGymAdapter()
    try:
        adapter.setup()
        adapter.step(args.steps)
        state = adapter.read_state()
        print("DOF positions:", state.dof_positions)
        print("DOF velocities:", state.dof_velocities)
        print("ee_link state:", state.ee_link_state)
        print("arm EE Jacobian shape:", tuple(state.arm_ee_jacobian.shape))
        print("rigid body indices:", adapter.body_indices)
    finally:
        adapter.close()
        print("S1 Isaac Gym simulator destroyed")


if __name__ == "__main__":
    main()
