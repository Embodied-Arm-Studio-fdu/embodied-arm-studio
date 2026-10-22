"""Diagnostic-only physics trace and contact geometry helpers for P2-R3.

Nothing in this module participates in policy observation, action, reward,
candidate, verification, or success.  ``PhysicsTraceTap`` temporarily wraps
an already-created experimental adapter so every physics step can be observed
without changing the number or order of simulation steps.
"""

from contextlib import AbstractContextManager
import math
from statistics import median


DIAGNOSTIC_SCHEMA = "p2-r3-grasp-physics-trace/v1"


def norm(values):
    return math.sqrt(sum(float(value) ** 2 for value in values))


def sub(left, right):
    return [float(a) - float(b) for a, b in zip(left, right)]


def dot(left, right):
    return sum(float(a) * float(b) for a, b in zip(left, right))


def quaternion_conjugate(value):
    return [-float(value[0]), -float(value[1]), -float(value[2]), float(value[3])]


def quaternion_multiply(left, right):
    ax, ay, az, aw = (float(v) for v in left)
    bx, by, bz, bw = (float(v) for v in right)
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def quaternion_rotate(quaternion, vector):
    rotated = quaternion_multiply(
        quaternion_multiply(quaternion, [float(v) for v in vector] + [0.0]),
        quaternion_conjugate(quaternion),
    )
    return rotated[:3]


def world_to_local(point, pose):
    return quaternion_rotate(quaternion_conjugate(pose[3:7]), sub(point, pose[:3]))


def local_to_world(point, pose):
    rotated = quaternion_rotate(pose[3:7], point)
    return [float(pose[i]) + rotated[i] for i in range(3)]


def relative_pose(body_pose, reference_pose):
    return {
        "position": world_to_local(body_pose[:3], reference_pose),
        "quaternion_xyzw": quaternion_multiply(
            quaternion_conjugate(reference_pose[3:7]), body_pose[3:7]),
    }


def classify_cube_contact(local_point, cube_size=0.045):
    """Classify a cube-local contact point without assuming a world axis."""
    half = float(cube_size) / 2.0
    point = [float(v) for v in local_point]
    height = max(0.0, min(1.0, (point[2] + half) / float(cube_size)))
    if height <= 0.16:
        region = "TABLE_NEAR" if abs(point[2] + half) <= 0.004 else "BOTTOM_EDGE"
    elif height >= 0.84:
        region = "TOP"
    else:
        axis = max(range(3), key=lambda index: abs(point[index]))
        if axis == 2:
            region = "TOP" if point[2] >= 0.0 else "BOTTOM_EDGE"
        else:
            region = ("X" if axis == 0 else "Y") + ("_POS_SIDE" if point[axis] >= 0 else "_NEG_SIDE")
    return {"region": region, "normalized_height": height, "local_point": point}


def _vec(value):
    if value is None:
        return [0.0, 0.0, 0.0]
    if all(hasattr(value, name) for name in ("x", "y", "z")):
        return [float(value.x), float(value.y), float(value.z)]
    if hasattr(value, "dtype") and getattr(value.dtype, "names", None):
        names = value.dtype.names
        if all(name in names for name in ("x", "y", "z")):
            return [float(value["x"]), float(value["y"]), float(value["z"])]
    values = value.tolist() if hasattr(value, "tolist") else value
    if isinstance(values, (int, float)):
        return [0.0, 0.0, 0.0]
    return [float(v) for v in values][:3]


def _field(contact, name, default=None):
    if hasattr(contact, "dtype") and getattr(contact.dtype, "names", None):
        if name in contact.dtype.names:
            return contact[name]
    return getattr(contact, name, default)


def _float_field(contact, name, default=0.0):
    value = _field(contact, name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        values = value.tolist() if hasattr(value, "tolist") else value
        return float(sum(abs(float(v)) for v in values))


class PhysicsTraceTap(AbstractContextManager):
    """Observe each physics step of an existing experimental scene."""

    def __init__(self, scene, stage_provider, control_step_provider):
        self.scene = scene
        self.stage_provider = stage_provider
        self.control_step_provider = control_step_provider
        self.rows = []
        # Preview 4 explicitly rejects get_env_rigid_contacts after a GPU
        # pipeline has started, and exposes no per-contact tensor.  Do not
        # repeatedly call the unsupported API; use the documented proxy below.
        self.contact_api_error = "GPU_PIPELINE_GET_ENV_RIGID_CONTACTS_UNAVAILABLE_AFTER_START"
        self._target = None
        self._original_step = None
        self._original_set = None
        self._physics_step = 0
        self._stage_physics_step = 0
        self._last_stage = None
        self._initial_cube = None
        self._body_names = self._resolve_env_body_names()

    def _resolve_env_body_names(self):
        gym, env, api = self.scene.robot.gym, self.scene.robot.env, self.scene.robot._gymapi
        result = {}
        for name in ("ee_link", "8_Link", "7_Link"):
            index = gym.find_actor_rigid_body_index(
                env, self.scene.robot.actor, name, api.IndexDomain.DOMAIN_ENV)
            result[int(index)] = name
        for actor, name in ((self.scene.table_actor, "table"),
                            (self.scene.cube_actor, "cube")):
            if actor is not None:
                index = gym.get_actor_rigid_body_index(
                    env, actor, 0, api.IndexDomain.DOMAIN_ENV)
                result[int(index)] = name
        return result

    def __enter__(self):
        robot = self.scene.robot
        self._original_step = robot.step
        self._original_set = robot.set_joint_position_targets
        self._initial_cube = list(self.scene.snapshot()["cube_state"][:3])

        def traced_set(positions):
            values = positions.detach().cpu().reshape(-1).tolist() if hasattr(positions, "detach") else list(positions)
            self._target = [float(v) for v in values]
            return self._original_set(positions)

        def traced_step(count=1):
            for _ in range(count):
                self._original_step(1)
                self._record()

        robot.set_joint_position_targets = traced_set
        robot.step = traced_step
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._original_step is not None:
            self.scene.robot.step = self._original_step
        if self._original_set is not None:
            self.scene.robot.set_joint_position_targets = self._original_set
        return False

    def _contacts(self, snapshot):
        return self._proxy_contacts(snapshot)

    def _proxy_contacts(self, snapshot):
        """Build auditable OBB/pose/force proxies when contact points are unavailable."""
        cube_pose = snapshot["cube_state"][:7]
        half = self.scene.config.cube_size / 2.0
        result = []
        for finger in ("8_Link", "7_Link"):
            body = snapshot["bodies"][finger]
            force = [float(value) for value in body["force"]]
            magnitude = norm(force)
            if magnitude < 0.2:
                continue
            center_local = world_to_local(body["state"][:3], cube_pose)
            closest = [max(-half, min(half, value)) for value in center_local]
            distance_to_obb = norm([center_local[i] - closest[i] for i in range(3)])
            # Net body force alone cannot identify its counterpart.  Reject
            # finger/table or self contacts when the finger center is too far
            # from the cube OBB to plausibly touch the 45 mm object.
            if distance_to_obb > 0.030:
                continue
            if all(abs(value) < half for value in center_local):
                axis = max(range(3), key=lambda index: abs(center_local[index]) / half)
                closest[axis] = half if center_local[axis] >= 0 else -half
            direction = sub(cube_pose[:3], body["state"][:3])
            direction_norm = norm(direction)
            normal = ([value / direction_norm for value in direction]
                      if direction_norm > 1.0e-9 else [0.0, 0.0, 0.0])
            # Finger tensor is force on the finger.  Negating it gives the
            # Newton-pair estimate of force applied to the contacted object.
            force_on_cube = [-value for value in force]
            entry = {
                "source": "OBB_FORCE_PROXY", "body0": finger, "body1": "cube",
                "body0_index": None, "body1_index": None,
                "local_pos0": [0.0, 0.0, 0.0], "local_pos1": closest,
                "surface_local_pos0": [0.0, 0.0, 0.0], "surface_local_pos1": closest,
                "normal_body0_to_body1_world": normal,
                "normal_force_magnitude": magnitude,
                "friction_force_magnitude": None,
                "initial_overlap": None, "min_dist": None,
                "effective_friction": self.scene.config.finger_friction,
                "world_point0": body["state"][:3],
                "world_point1": local_to_world(closest, cube_pose),
                "finger_center_to_cube_obb_distance": distance_to_obb,
                "cube_contact": classify_cube_contact(closest, self.scene.config.cube_size),
                "normal_force_on_cube_world": force_on_cube,
            }
            result.append(entry)
        # The table tensor contains its reaction to cube plus any finger
        # loading.  Its downward Z reaction is an intentionally conservative
        # measure of how much the table still supports/loads the grasp system.
        support = max(0.0, -float(snapshot["bodies"]["table"]["force"][2]))
        clearance = cube_pose[2] - half - self.scene.config.table_top
        # The table body tensor aggregates every table contact, including a
        # robot/finger touching the table.  Attribute it to cube support only
        # while the cube is geometrically close enough to the tabletop.
        if support >= 0.02 and clearance <= 0.004:
            bottom = [0.0, 0.0, -half]
            result.append({
                "source": "TABLE_NET_FORCE_PROXY", "body0": "table", "body1": "cube",
                "body0_index": None, "body1_index": None,
                "local_pos0": [0.0, 0.0, 0.0], "local_pos1": bottom,
                "surface_local_pos0": [0.0, 0.0, 0.0], "surface_local_pos1": bottom,
                "normal_body0_to_body1_world": [0.0, 0.0, 1.0],
                "normal_force_magnitude": support, "friction_force_magnitude": None,
                "initial_overlap": None, "min_dist": None,
                "effective_friction": self.scene.config.table_friction,
                "world_point0": [cube_pose[0], cube_pose[1], self.scene.config.table_top],
                "world_point1": local_to_world(bottom, cube_pose),
                "cube_contact": classify_cube_contact(bottom, self.scene.config.cube_size),
                "normal_force_on_cube_world": [0.0, 0.0, support],
            })
        return result

    def _record(self):
        snapshot = self.scene.snapshot()
        stage = str(self.stage_provider())
        if stage != self._last_stage:
            self._stage_physics_step = 0
            self._last_stage = stage
            if stage == "RESET_SETTLE":
                self._initial_cube = list(snapshot["cube_state"][:3])
        contacts = self._contacts(snapshot)
        cube = snapshot["cube_state"]; ee = snapshot["ee_state"]
        left = snapshot["bodies"]["8_Link"]["state"]
        right = snapshot["bodies"]["7_Link"]["state"]
        target = list(self._target or snapshot["q"])
        jaw_target = [float(target[6]), float(target[7])]
        jaw_actual = [float(snapshot["q"][6]), float(snapshot["q"][7])]
        cube_contacts = [item for item in contacts if "cube_contact" in item]
        finger_contacts = {
            name: [item for item in cube_contacts if name in (item["body0"], item["body1"])]
            for name in ("8_Link", "7_Link")
        }
        table_contacts = [item for item in cube_contacts if "table" in (item["body0"], item["body1"])]
        support_z = sum(max(0.0, item["normal_force_on_cube_world"][2]) for item in table_contacts)
        mass = float(self.scene.property_dump["cube_bodies"][0]["mass"])
        row = {
            "schema": DIAGNOSTIC_SCHEMA,
            "time": self._physics_step * self.scene.robot.config.physics_dt,
            "physics_step": self._physics_step,
            "physics_step_in_stage": self._stage_physics_step,
            "control_step": int(self.control_step_provider()),
            "physics_step_in_control": self._stage_physics_step % 4,
            "stage": stage,
            "q": snapshot["q"], "dq": snapshot["dq"], "target_8d": target,
            "jaw_command": jaw_target, "jaw_actual": jaw_actual,
            "jaw_actual_minus_command": [jaw_actual[i] - jaw_target[i] for i in range(2)],
            "estimated_aperture": snapshot["estimated_aperture"],
            "ee_pose": ee[:7], "ee_velocity": ee[7:13],
            "cube_pose": cube[:7], "cube_velocity": cube[7:13],
            "cube_bottom_height": cube[2] - self.scene.config.cube_size / 2.0,
            "cube_table_clearance": cube[2] - self.scene.config.cube_size / 2.0 - self.scene.config.table_top,
            "cube_rise_from_reset": cube[2] - self._initial_cube[2],
            "ee_cube_relative_translation": sub(cube[:3], ee[:3]),
            "left_finger_pose": left[:7], "right_finger_pose": right[:7],
            "left_finger_to_cube": relative_pose(left[:7], cube[:7]),
            "right_finger_to_cube": relative_pose(right[:7], cube[:7]),
            "net_contact_force": {name: list(body["force"]) for name, body in snapshot["bodies"].items()},
            "contacts": contacts,
            "contact_api_available": False,
            "contact_geometry_source": "OBB_FORCE_PROXY",
            "left_cube_contact_count": len(finger_contacts["8_Link"]),
            "right_cube_contact_count": len(finger_contacts["7_Link"]),
            "cube_table_contact_count": len(table_contacts),
            "table_support_vertical": support_z,
            "table_support_ratio": support_z / max(1.0e-9, mass * 9.81),
            "cube_obb": {"center": cube[:3], "quaternion_xyzw": cube[3:7],
                         "half_extents": [self.scene.config.cube_size / 2.0] * 3},
        }
        self.rows.append(row)
        self._physics_step += 1
        self._stage_physics_step += 1


def _has_pair(row, body):
    for item in row.get("contacts", ()):
        if "cube" not in (item["body0"], item["body1"]) or body not in (
                item["body0"], item["body1"]):
            continue
        if item.get("source") == "OBB_FORCE_PROXY":
            force = item.get("normal_force_on_cube_world", (0.0, 0.0, 0.0))
            # Reject vertical-dominant finger/table loading.  A side pinch on
            # this fixed-orientation setup has a measurable horizontal pair.
            if math.hypot(float(force[0]), float(force[1])) < 0.15:
                continue
        return True
    return False


def build_event_timeline(rows, candidate_window=6):
    """Align a physics trace around the candidate-equivalent T4 event."""
    if not rows:
        raise ValueError("trace is empty")
    events = {}
    for index, row in enumerate(rows):
        left = _has_pair(row, "8_Link"); right = _has_pair(row, "7_Link")
        if "T0" not in events and (left or right): events["T0"] = index
        if "T1" not in events and left and right: events["T1"] = index
        actual = sum(row["jaw_actual"]) / 2.0; command = sum(row["jaw_command"]) / 2.0
        if "T2" not in events and actual >= command + 0.002: events["T2"] = index
    verify_indices = [i for i, row in enumerate(rows)
                      if row["stage"] in ("VERIFYING", "VERIFICATION_LIFT")]
    if verify_indices:
        events["T5"] = verify_indices[0]
        events["T4"] = max(0, verify_indices[0] - 1)
        control_endpoints = [i for i in range(events["T4"] + 1)
                             if i == events["T4"] or rows[i]["control_step"] != rows[min(i + 1, events["T4"])]["control_step"]]
        events["T3"] = control_endpoints[max(0, len(control_endpoints) - candidate_window)]
        reference_z = rows[events["T4"]]["ee_pose"][2]
        for label, millimetres in (("T6", 5), ("T7", 10), ("T8", 15), ("T9", 25)):
            found = next((i for i in verify_indices
                          if rows[i]["ee_pose"][2] - reference_z >= millimetres / 1000.0), None)
            events[label] = found if found is not None else verify_indices[-1]
    return {
        name: {"row_index": index, "physics_step": rows[index]["physics_step"],
               "time": rows[index]["time"], "relative_to_T4": (
                   rows[index]["time"] - rows[events["T4"]]["time"] if "T4" in events else None)}
        for name, index in events.items()
    }


def contact_summary_at(row, finger):
    contacts = [item for item in row.get("contacts", ())
                if finger in (item["body0"], item["body1"])
                and "cube" in (item["body0"], item["body1"])]
    if not contacts:
        return {"count": 0, "regions": [], "mean_height": None,
                "force_vector": [0.0, 0.0, 0.0], "force_magnitude": 0.0}
    forces = [item["normal_force_on_cube_world"] for item in contacts]
    force = [sum(value[axis] for value in forces) for axis in range(3)]
    heights = [item["cube_contact"]["normalized_height"] for item in contacts]
    return {"count": len(contacts),
            "regions": sorted({item["cube_contact"]["region"] for item in contacts}),
            "mean_height": sum(heights) / len(heights),
            "force_vector": force, "force_magnitude": norm(force)}


def grasp_snapshot(rows, timeline):
    row = rows[timeline["T4"]["row_index"]]
    left = contact_summary_at(row, "8_Link"); right = contact_summary_at(row, "7_Link")
    opposition = None
    if left["force_magnitude"] > 1.0e-9 and right["force_magnitude"] > 1.0e-9:
        cosine = dot(left["force_vector"], right["force_vector"]) / (
            left["force_magnitude"] * right["force_magnitude"])
        opposition = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    return {
        "physics_step": row["physics_step"], "time": row["time"],
        "ee_pose": row["ee_pose"], "cube_pose": row["cube_pose"],
        "left_finger_pose": row["left_finger_pose"],
        "right_finger_pose": row["right_finger_pose"],
        "jaw_actual": row["jaw_actual"], "jaw_command": row["jaw_command"],
        "jaw_blockage": [row["jaw_actual"][i] - row["jaw_command"][i] for i in range(2)],
        "estimated_aperture": row["estimated_aperture"], "cube_obb": row["cube_obb"],
        "left_contact": left, "right_contact": right,
        "force_opposition_angle_degrees": opposition,
        "table_support_vertical": row["table_support_vertical"],
        "table_support_ratio": row["table_support_ratio"],
        "cube_table_clearance": row["cube_table_clearance"],
        "ee_cube_relative_translation": row["ee_cube_relative_translation"],
    }


def verifier_failure(rows, timeline, initial_ee_z):
    verify = [row for row in rows if row["stage"] in ("VERIFYING", "VERIFICATION_LIFT")]
    if not verify:
        return {"first_failure": "OTHER", "reason": "verification_not_entered"}
    start = rows[timeline["T4"]["row_index"]]
    terminal = verify[-1]
    cube_rise = terminal["cube_rise_from_reset"]
    local_ee_rise = terminal["ee_pose"][2] - start["ee_pose"][2]
    formal_ee_rise = terminal["ee_pose"][2] - float(initial_ee_z)
    relative_start = start["ee_cube_relative_translation"]
    translation_drift = norm(sub(terminal["ee_cube_relative_translation"], relative_start))
    from .evidence import quaternion_distance
    rotation_drift = quaternion_distance(terminal["cube_pose"][3:7], start["cube_pose"][3:7])
    actual_jaw = sum(terminal["jaw_actual"]) / 2.0
    finite = all(math.isfinite(float(value)) for key in ("q", "dq", "ee_pose", "cube_pose")
                 for value in terminal[key])
    checks = [
        ("CUBE_NOT_CLEAR_OF_TABLE", terminal["cube_table_clearance"] >= 0.008),
        ("CUBE_RISE_TOO_LOW", cube_rise >= 0.012),
        ("LIFT_RATIO_TOO_LOW", formal_ee_rise > 0.0 and cube_rise >= 0.55 * formal_ee_rise),
        ("TRANSLATION_DRIFT", translation_drift <= 0.012),
        ("ROTATION_DRIFT", rotation_drift <= 0.25),
        ("JAW_UNSAFE", actual_jaw >= 0.01),
        ("NON_FINITE", finite),
    ]
    first = next((name for name, passed in checks if not passed), "OTHER")
    contacts = [(_has_pair(row, "8_Link"), _has_pair(row, "7_Link")) for row in verify]
    last_bilateral = max((i for i, pair in enumerate(contacts) if all(pair)), default=None)
    return {
        "first_failure": first, "failure_physics_step": terminal["physics_step"],
        "ee_rise_at_failure": local_ee_rise, "formal_reset_relative_ee_rise": formal_ee_rise,
        "cube_rise_at_failure": cube_rise,
        "cube_table_clearance": terminal["cube_table_clearance"],
        "table_support_ratio": terminal["table_support_ratio"],
        "left_force": terminal["net_contact_force"]["8_Link"],
        "right_force": terminal["net_contact_force"]["7_Link"],
        "relative_translation_drift": translation_drift,
        "relative_rotation_drift": rotation_drift,
        "last_bilateral_contact_verify_index": last_bilateral,
        "verification_physics_steps": len(verify),
        "all_failed_checks": [name for name, passed in checks if not passed],
    }


def distribution(values):
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"count": len(clean), "mean": sum(clean) / len(clean), "median": median(clean),
            "min": min(clean), "max": max(clean)}
