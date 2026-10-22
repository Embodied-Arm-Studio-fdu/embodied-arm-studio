import tempfile
import unittest
import xml.etree.ElementTree as ElementTree
from pathlib import Path

from simulation_core.s1_isaac_gym_adapter import (
    ARM_DOF_COUNT,
    DEFAULT_ASSET_ROOT,
    DEFAULT_INITIAL_DOF_POSITIONS,
    DEFAULT_ROBOT_URDF,
    EXPECTED_BODY_NAMES,
    EXPECTED_DOF_NAMES,
    GRIPPER_DOF_COUNT,
    S1AdapterError,
    S1AssetValidationError,
    S1IsaacGymAdapter,
    S1IsaacGymConfig,
)


class FakePhysxParams:
    pass


class FakeSimParams:
    def __init__(self):
        self.physx = FakePhysxParams()


class FakeGymApi:
    UP_AXIS_Z = "z"
    SIM_PHYSX = "physx"

    class Vec3:
        def __init__(self, *values):
            self.values = values

    SimParams = FakeSimParams


class FakeGym:
    def __init__(self, fail_create=False):
        self.fail_create = fail_create
        self.destroyed = []

    def create_sim(self, *args):
        if self.fail_create:
            raise RuntimeError("create failed")
        return "fake-sim"

    def destroy_sim(self, sim):
        self.destroyed.append(sim)


def fake_loader(gym):
    class BoundFakeGymApi(FakeGymApi):
        @staticmethod
        def acquire_gym():
            return gym

    return lambda: (BoundFakeGymApi, object())


class S1ConfigTestCase(unittest.TestCase):
    def test_defaults_are_explicit_and_safe(self):
        config = S1IsaacGymConfig()
        self.assertEqual(DEFAULT_ASSET_ROOT, config.asset_root)
        self.assertEqual(DEFAULT_ROBOT_URDF, config.robot_urdf)
        self.assertEqual(0, config.compute_device_id)
        self.assertEqual(-1, config.graphics_device_id)
        self.assertTrue(config.headless)
        self.assertEqual(8, len(config.initial_dof_positions))
        self.assertEqual(DEFAULT_INITIAL_DOF_POSITIONS, config.initial_dof_positions)

    def test_expected_dof_groups_are_explicit(self):
        self.assertEqual(6, ARM_DOF_COUNT)
        self.assertEqual(2, GRIPPER_DOF_COUNT)
        self.assertEqual(
            ("1joint", "2joint", "3joint", "4joint", "5joint", "6joint"),
            EXPECTED_DOF_NAMES[:ARM_DOF_COUNT],
        )
        self.assertEqual(("7joint", "8joint"), EXPECTED_DOF_NAMES[ARM_DOF_COUNT:])

    def test_invalid_configuration_is_rejected_without_isaac_gym(self):
        with self.assertRaisesRegex(ValueError, "exactly 8"):
            S1IsaacGymConfig(initial_dof_positions=(0.0,) * 7)
        with self.assertRaisesRegex(ValueError, "must be relative"):
            S1IsaacGymConfig(robot_urdf=Path("/absolute/robot.urdf"))
        with self.assertRaisesRegex(ValueError, "physics_dt"):
            S1IsaacGymConfig(physics_dt=0.0)

    def test_missing_asset_path_has_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            config = S1IsaacGymConfig(asset_root=Path(directory) / "missing")
            with self.assertRaisesRegex(S1AssetValidationError, "asset root"):
                config.validate_asset_path()

    def test_default_urdf_static_structure_matches_contract(self):
        config = S1IsaacGymConfig()
        urdf_path = config.validate_asset_path()
        root = ElementTree.parse(str(urdf_path)).getroot()
        active_joints = tuple(
            joint.get("name")
            for joint in root.findall("joint")
            if joint.get("type") != "fixed"
        )
        links = {link.get("name") for link in root.findall("link")}
        # Isaac Gym orders sibling joints by its imported articulation layout,
        # not by their textual order in this URDF. Arm order must remain exact;
        # the two gripper names form the final group in either representation.
        self.assertEqual(EXPECTED_DOF_NAMES[:ARM_DOF_COUNT], active_joints[:6])
        self.assertEqual(
            set(EXPECTED_DOF_NAMES[ARM_DOF_COUNT:]), set(active_joints[6:])
        )
        self.assertTrue(set(EXPECTED_BODY_NAMES).issubset(links))


class S1LifecycleTestCase(unittest.TestCase):
    def test_context_manager_initializes_and_closes_once(self):
        gym = FakeGym()
        with S1IsaacGymAdapter(api_loader=fake_loader(gym)) as adapter:
            self.assertEqual("fake-sim", adapter.sim)
        self.assertEqual(["fake-sim"], gym.destroyed)
        adapter.close()
        self.assertEqual(["fake-sim"], gym.destroyed)

    def test_context_manager_closes_on_body_exception(self):
        gym = FakeGym()
        with self.assertRaisesRegex(RuntimeError, "body failed"):
            with S1IsaacGymAdapter(api_loader=fake_loader(gym)):
                raise RuntimeError("body failed")
        self.assertEqual(["fake-sim"], gym.destroyed)

    def test_initialize_failure_leaves_adapter_closed(self):
        gym = FakeGym(fail_create=True)
        adapter = S1IsaacGymAdapter(api_loader=fake_loader(gym))
        with self.assertRaisesRegex(RuntimeError, "create failed"):
            adapter.initialize()
        adapter.close()
        self.assertIsNone(adapter.sim)
        self.assertEqual([], gym.destroyed)

    def test_import_loader_failure_is_wrapped_and_safe(self):
        def failing_loader():
            raise S1AdapterError("Isaac Gym unavailable")

        adapter = S1IsaacGymAdapter(api_loader=failing_loader)
        with self.assertRaisesRegex(S1AdapterError, "unavailable"):
            adapter.initialize()
        adapter.close()


if __name__ == "__main__":
    unittest.main()
