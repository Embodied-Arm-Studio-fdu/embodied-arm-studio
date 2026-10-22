"""Deterministic P1 grasp/lift/place-style scripted state machine."""

from dataclasses import dataclass
import math
import os
from pathlib import Path
import subprocess
from typing import Optional, Tuple

import cv2
import numpy as np

from .evidence import GraspEvidenceConfig, evaluate_grasp, quaternion_distance
from .scene import ManipulationScene
from .target_compositor import ExperimentalTargetCompositor, Workspace3D


def _norm(values):
    return math.sqrt(sum(float(v) ** 2 for v in values))


def _sub(left, right):
    return tuple(float(a) - float(b) for a, b in zip(left, right))


@dataclass(frozen=True)
class ScriptedRuntimeConfig:
    open_safe: float = 0.05
    close_safe: float = 0.012
    jaw_rate_limit: float = 0.0015
    force_abort_threshold: float = 150.0
    physics_steps_per_control: int = 4
    target_tolerance: float = 0.006
    # World-frame cube-center offset from ee_link for the fixed quaternion.
    # It comes from the v2 finger collision centers (mostly local -Z): fingers
    # are above ee_link in world Z, not below it.
    grasp_frame_object_offset: Tuple[float, float, float] = (-0.008, 0.009, 0.012)
    pregrasp_clearance: float = 0.080
    verification_lift: float = 0.025
    hold_seconds: float = 5.0
    settle_seconds: float = 2.0
    workspace: Workspace3D = Workspace3D()
    fixed_orientation_xyzw: Tuple[float, float, float, float] = (
        0.6040455102920532, -0.7851132154464722,
        0.12926433980464935, -0.044911183416843414,
    )

    def __post_init__(self):
        if not 0.01 <= self.close_safe < self.open_safe <= 0.05:
            raise ValueError("jaw safety values must satisfy 0.01 <= close < open <= 0.05")
        if self.physics_steps_per_control <= 0:
            raise ValueError("physics_steps_per_control must be positive")
        if self.force_abort_threshold <= 0 or self.hold_seconds <= 0 or self.settle_seconds <= 0:
            raise ValueError("force/time limits must be positive")


class ExperimentalMp4Recorder:
    """Standalone Camera RGBA -> NVENC MP4 recorder, outside Production lifecycle."""

    def __init__(self, path: Path, ffmpeg=None, fps=15):
        self.path = Path(path)
        self.ffmpeg = Path(ffmpeg or os.environ.get("SIMULATION_FFMPEG_PATH") or "/home/user/opt/ffmpeg-nvenc/bin/ffmpeg")
        self.fps = int(fps)
        self.process = None; self.frames = 0; self.error = None

    def start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pixel_format", "yuv420p", "-video_size", "640x480",
            "-framerate", str(self.fps), "-i", "pipe:0", "-an",
            "-c:v", "h264_nvenc", "-gpu", "0", "-preset", "p4",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return self

    def write(self, frame):
        if self.process is None or self.process.stdin is None:
            return
        try:
            # This FFmpeg/NVENC build corrupts packed RGB(A) scanlines during
            # conversion.  Convert explicitly to planar I420, which NVENC
            # consumes natively and reproducibly on this host.
            rgb = np.ascontiguousarray(frame.data[:, :, :3], dtype=np.uint8)
            yuv420 = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV_I420)
            payload = memoryview(yuv420.tobytes(order="C"))
            offset = 0
            while offset < len(payload):
                written = self.process.stdin.write(payload[offset:])
                if not written:
                    raise BrokenPipeError("FFmpeg accepted a zero-length frame write")
                offset += written
            self.process.stdin.flush()
            self.frames += 1
        except Exception as exc:
            self.error = str(exc)
            raise

    def close(self):
        if self.process is None:
            return True
        if self.process.stdin is not None:
            try: self.process.stdin.close()
            except BrokenPipeError: pass
        stderr = self.process.stderr.read().decode("utf-8", "replace") if self.process.stderr else ""
        code = self.process.wait(timeout=30)
        self.process = None
        if code != 0:
            self.error = stderr.strip() or "ffmpeg exited %d" % code
            return False
        return self.path.is_file() and self.path.stat().st_size > 0


class ScriptedManipulationRuntime:
    STAGE_LIMITS = {
        "RESET_SETTLE": 45, "PRE_GRASP_CLEARANCE": 180, "PRE_GRASP": 240, "DESCEND": 180,
        "CLOSE": 160, "VERIFICATION_LIFT": 180, "HOLD": 120,
        "LOWER": 180, "RELEASE": 160, "RETREAT": 160, "SETTLE": 120,
    }

    def __init__(self, scene: ManipulationScene, config: ScriptedRuntimeConfig = None,
                 frame_callback=None):
        self.scene = scene
        self.config = config or ScriptedRuntimeConfig()
        self.compositor = ExperimentalTargetCompositor(
            workspace=self.config.workspace, jaw_rate_limit=self.config.jaw_rate_limit)
        self.trace = []; self.stage_latency = {}; self._control_step = 0
        self._target_xyz = None; self._last_command = None
        self._failure_reason = ""; self._stage_reached = ""
        self._max_force = 0.0; self._max_relative_drift = 0.0
        self._minimum_aperture = float("inf"); self._verified_once = False
        self._grasp_reference_relative = None; self._grasp_reference_rotation = None
        self._initial_cube = None; self._initial_ee = None
        self._recorder = None; self._resource_cleanup = True
        self.frame_callback = frame_callback

    def _forces(self, snapshot):
        result = {}
        for key, body in snapshot["bodies"].items():
            result[key] = _norm(body["force"])
        return result

    def _object_in_corridor(self, snapshot):
        if snapshot["cube_state"] is None:
            return False
        ee = snapshot["ee_state"][:3]; obj = snapshot["cube_state"][:3]
        expected = tuple(ee[i] + self.config.grasp_frame_object_offset[i] for i in range(3))
        return _norm(_sub(obj, expected)) <= 0.032

    def _jaw_position_stable(self, window=10, tolerance=0.0003):
        rows=[row for row in self.trace if row["stage"] == "CLOSE"][-window:]
        if len(rows) < window:
            return False
        left=[row["q"][6] for row in rows]; right=[row["q"][7] for row in rows]
        return max(left)-min(left) < tolerance and max(right)-min(right) < tolerance

    def _evidence(self, snapshot):
        cube = snapshot["cube_state"]
        if cube is None or self._grasp_reference_relative is None:
            return None
        ee = snapshot["ee_state"]
        relative = _sub(cube[:3], ee[:3])
        drift = _norm(_sub(relative, self._grasp_reference_relative))
        rotation_drift = quaternion_distance(cube[3:7], self._grasp_reference_rotation)
        clearance = cube[2] - self.scene.config.cube_size / 2.0 - self.scene.config.table_top
        forces = self._forces(snapshot)
        config = GraspEvidenceConfig(
            open_safe=self.config.open_safe,
            self_contact_floor=max(0.0095, self.config.close_safe - 0.0025),
        )
        return evaluate_grasp(
            config=config, actual_jaw=0.5*(snapshot["q"][6]+snapshot["q"][7]),
            object_in_corridor=self._object_in_corridor(snapshot),
            object_table_clearance=clearance,
            object_rise=cube[2]-self._initial_cube[2],
            ee_rise=ee[2]-self._initial_ee[2],
            relative_translation_drift=drift,
            relative_rotation_drift=rotation_drift,
            state_values=snapshot["q"]+snapshot["dq"]+ee+cube,
            left_force=forces.get("8_Link", 0.0),
            right_force=forces.get("7_Link", 0.0),
            object_force=forces.get("cube", 0.0),
        )

    def _abort_reason(self, snapshot, stage, command):
        values = snapshot["q"] + snapshot["dq"] + snapshot["ee_state"]
        if snapshot["cube_state"] is not None: values += snapshot["cube_state"]
        if not all(math.isfinite(float(v)) for v in values): return "NON_FINITE"
        if any(q < lo-1e-5 or q > hi+1e-5 for q, lo, hi in zip(
                snapshot["q"], self.compositor.controller.config.lower_limits,
                self.compositor.controller.config.upper_limits)): return "JOINT_HARD_VIOLATION"
        if command.workspace_clamped: return "WORKSPACE_HARD_VIOLATION"
        forces = self._forces(snapshot)
        dynamic_max = max(forces.get("8_Link", 0.0), forces.get("7_Link", 0.0),
                          forces.get("cube", 0.0))
        if dynamic_max > self.config.force_abort_threshold: return "UNSAFE_FORCE"
        if snapshot["cube_state"] is not None:
            cube = snapshot["cube_state"]
            if abs(cube[0]-self.scene.config.table_position[0]) > self.scene.config.table_size[0]/2:
                return "OBJECT_OUT_OF_TABLE"
            if abs(cube[1]-self.scene.config.table_position[1]) > self.scene.config.table_size[1]/2:
                return "OBJECT_OUT_OF_TABLE"
            bottom = cube[2]-self.scene.config.cube_size/2
            if bottom < self.scene.config.table_top-0.004: return "PENETRATION"
            if stage in ("RESET_SETTLE", "PRE_GRASP_CLEARANCE", "PRE_GRASP", "DESCEND"):
                horizontal = math.hypot(cube[0]-self._initial_cube[0], cube[1]-self._initial_cube[1])
                if horizontal > 0.018: return "OBJECT_PREMATURE_DISPLACEMENT"
        return ""

    def _record(self, seed, stage, target_xyz, command):
        snapshot = self.scene.snapshot(); forces = self._forces(snapshot)
        cube = snapshot["cube_state"]
        ee = snapshot["ee_state"]
        relative = _sub(cube[:3], ee[:3]) if cube else (0.0, 0.0, 0.0)
        clearance = (cube[2]-self.scene.config.cube_size/2-self.scene.config.table_top
                     if cube else 0.0)
        rotation_drift = (quaternion_distance(cube[3:7], self._grasp_reference_rotation)
                          if cube and self._grasp_reference_rotation else 0.0)
        evidence = self._evidence(snapshot)
        cart_error = [float(v) for v in command.cartesian_error.detach().cpu().tolist()]
        target8 = [float(v) for v in command.positions.detach().cpu().tolist()]
        record = {
            "time": self._control_step*self.scene.robot.config.physics_dt*self.config.physics_steps_per_control,
            "seed": int(seed), "stage": stage, "control_step": self._control_step,
            "q": snapshot["q"], "dq": snapshot["dq"], "target_8d": target8,
            "ee_pose": ee[:7], "ee_velocity": ee[7:13],
            "left_finger_pose": snapshot["bodies"]["8_Link"]["state"][:7],
            "right_finger_pose": snapshot["bodies"]["7_Link"]["state"][:7],
            "object_pose": cube[:7] if cube else None,
            "object_velocity": cube[7:13] if cube else None,
            "net_contact_force": {k: v["force"] for k, v in snapshot["bodies"].items()},
            "net_contact_force_magnitude": forces,
            "jaw_commanded_q": command.jaw_target,
            "jaw_actual_q": 0.5*(snapshot["q"][6]+snapshot["q"][7]),
            "estimated_aperture": snapshot["estimated_aperture"],
            "cartesian_error": cart_error, "cartesian_error_norm": _norm(cart_error),
            "object_table_clearance": clearance,
            "ee_object_relative_translation": relative,
            "relative_rotation_drift": rotation_drift,
            "workspace_clamp": command.workspace_clamped,
            "joint_delta_clamp": command.joint_delta_clamped,
            "joint_limit_clamp": command.joint_limit_clamped,
            "grasp_evidence": ({"grasp_candidate": evidence.grasp_candidate,
                                "grasp_verified": evidence.grasp_verified,
                                "confidence": evidence.confidence,
                                "flags": dict(evidence.flags),
                                "failure_reason": evidence.failure_reason}
                               if evidence else None),
        }
        self.trace.append(record); self._control_step += 1
        self._max_force = max(self._max_force, *(forces.values() or [0.0]))
        self._minimum_aperture = min(self._minimum_aperture, snapshot["estimated_aperture"])
        if self._grasp_reference_relative is not None and cube and stage in (
                "CLOSE", "VERIFICATION_LIFT", "HOLD", "LOWER"):
            self._max_relative_drift = max(
                self._max_relative_drift,
                _norm(_sub(relative, self._grasp_reference_relative)))
        if evidence and evidence.grasp_verified: self._verified_once = True
        if stage == "HOLD" and evidence and not evidence.grasp_verified and not self._failure_reason:
            self._failure_reason = "GRASP_LOST_DURING_HOLD"
        if self._recorder is not None:
            self._recorder.write(self.scene.capture())
        elif self.frame_callback is not None:
            self.frame_callback(self.scene.capture())
        reason = self._abort_reason(snapshot, stage, command)
        if reason and not self._failure_reason: self._failure_reason = reason
        return record

    def _control(self, seed, stage, xyz, jaw):
        state = self.scene.robot.read_state()
        command = self.compositor.compose(xyz, self.config.fixed_orientation_xyzw, jaw, state)
        self.scene.robot.set_joint_position_targets(command.positions)
        self.scene.robot.step(self.config.physics_steps_per_control)
        self._last_command = command; self._target_xyz = tuple(xyz)
        return self._record(seed, stage, xyz, command)

    def _run_until(self, seed, stage, xyz, jaw, predicate, max_steps=None):
        self._stage_reached = stage; start = self._control_step
        limit = max_steps or self.STAGE_LIMITS[stage]
        completed = False
        for _ in range(limit):
            record = self._control(seed, stage, xyz, jaw)
            if self._failure_reason: break
            if predicate(record): completed = True; break
        self.stage_latency[stage] = self._control_step-start
        if not completed and not self._failure_reason:
            self._failure_reason = "TIMEOUT_%s" % stage
        return completed

    def run_trial(self, seed: int, cube_xy=None, video_path: Optional[Path] = None):
        self.trace=[]; self.stage_latency={}; self._control_step=0; self._failure_reason=""
        self._stage_reached=""; self._max_force=0.; self._max_relative_drift=0.
        self._minimum_aperture=float("inf"); self._verified_once=False
        self._grasp_reference_relative=None; self._grasp_reference_rotation=None
        self._resource_cleanup=True
        self.scene.reset(cube_xy)
        initial = self.scene.snapshot(); self._initial_cube=tuple(initial["cube_state"][:3])
        self._initial_ee=tuple(initial["ee_state"][:3]); self.compositor.reset(initial["q"])
        if video_path is not None:
            self._recorder = ExperimentalMp4Recorder(video_path).start()
        else: self._recorder=None
        cube = self._initial_cube
        grasp=tuple(cube[i]-self.config.grasp_frame_object_offset[i] for i in range(3))
        pre=(grasp[0],grasp[1],grasp[2]+self.config.pregrasp_clearance)
        lift=(grasp[0],grasp[1],grasp[2]+self.config.verification_lift)
        try:
            current=self._initial_ee
            self._run_until(seed,"RESET_SETTLE",current,self.config.open_safe,
                lambda r: max(abs(v) for v in r["dq"])<0.05 and
                          _norm(r["object_velocity"] or (0,)*6)<0.05)
            if not self._failure_reason:
                clearance=(self._initial_ee[0],self._initial_ee[1],pre[2])
                self._run_until(seed,"PRE_GRASP_CLEARANCE",clearance,self.config.open_safe,
                    lambda r: _norm(_sub(r["ee_pose"][:3],clearance))<=self.config.target_tolerance)
            if not self._failure_reason:
                self._run_until(seed,"PRE_GRASP",pre,self.config.open_safe,
                    lambda r: _norm(_sub(r["ee_pose"][:3],pre))<=self.config.target_tolerance)
            if not self._failure_reason:
                self._run_until(seed,"DESCEND",grasp,self.config.open_safe,
                    lambda r: _norm(_sub(r["ee_pose"][:3],grasp))<=self.config.target_tolerance)
            if not self._failure_reason:
                # Seed candidate evidence at the end of descent.  The
                # reference is refreshed after closure for lift-drift tests.
                snap=self.scene.snapshot(); self._grasp_reference_relative=_sub(
                    snap["cube_state"][:3],snap["ee_state"][:3])
                self._grasp_reference_rotation=tuple(snap["cube_state"][3:7])
                self._run_until(seed,"CLOSE",grasp,self.config.close_safe,
                    lambda r: (r["grasp_evidence"] is not None and
                               r["grasp_evidence"]["grasp_candidate"] and
                               abs(r["jaw_commanded_q"]-self.config.close_safe)<1e-6 and
                               self._jaw_position_stable()))
            if not self._failure_reason:
                snap=self.scene.snapshot(); self._grasp_reference_relative=_sub(
                    snap["cube_state"][:3],snap["ee_state"][:3])
                self._grasp_reference_rotation=tuple(snap["cube_state"][3:7])
                self._run_until(seed,"VERIFICATION_LIFT",lift,self.config.close_safe,
                    lambda r: bool(r["grasp_evidence"] and r["grasp_evidence"]["grasp_verified"]))
            if not self._failure_reason:
                hold_steps=math.ceil(self.config.hold_seconds/
                    (self.scene.robot.config.physics_dt*self.config.physics_steps_per_control))
                self._run_until(seed,"HOLD",lift,self.config.close_safe,
                    lambda r, count=[0], needed=hold_steps: (count.__setitem__(0,count[0]+1) or count[0]>=needed),
                    max_steps=hold_steps+2)
            if not self._failure_reason:
                self._run_until(seed,"LOWER",grasp,self.config.close_safe,
                    # Once the cube is table-supported, insisting that the EE
                    # reach the free-space target would command penetration.
                    lambda r: r["object_table_clearance"]<=0.006 and
                              r["net_contact_force_magnitude"].get("table",0.0)>0.2)
            if not self._failure_reason:
                self._run_until(seed,"RELEASE",grasp,self.config.open_safe,
                    lambda r: (r["estimated_aperture"]>=self.scene.config.cube_size+0.025 and
                               r["object_table_clearance"]<=0.008 and
                               min(r["net_contact_force_magnitude"].get("8_Link",0.0),
                                   r["net_contact_force_magnitude"].get("7_Link",0.0))<0.2))
            if not self._failure_reason:
                retreat=(grasp[0]-0.050,grasp[1],grasp[2]+0.080)
                self._run_until(seed,"RETREAT",retreat,self.config.open_safe,
                    lambda r: _norm(_sub(r["ee_pose"][:3],retreat))<=self.config.target_tolerance)
            if not self._failure_reason:
                settle_steps=math.ceil(self.config.settle_seconds/
                    (self.scene.robot.config.physics_dt*self.config.physics_steps_per_control))
                self._run_until(seed,"SETTLE",retreat,self.config.open_safe,
                    lambda r, count=[0], needed=settle_steps: (count.__setitem__(0,count[0]+1) or
                        (count[0]>=needed and _norm(r["object_velocity"] or (0,)*6)<0.06)),
                    max_steps=settle_steps+30)
        finally:
            if self._recorder is not None:
                try: self._resource_cleanup = self._recorder.close()
                except Exception: self._resource_cleanup=False
        final=self.scene.snapshot(); cube_final=final["cube_state"]
        settle_linear=_norm(cube_final[7:10]); settle_angular=_norm(cube_final[10:13])
        released_distance=_norm(_sub(cube_final[:3],final["ee_state"][:3]))
        success=bool(not self._failure_reason and self._verified_once and
                     settle_linear<0.06 and settle_angular<0.2 and
                     released_distance>0.07 and self._resource_cleanup)
        if not success and not self._failure_reason:
            self._failure_reason=("RESOURCE_CLEANUP" if not self._resource_cleanup
                                  else "SETTLE_OR_RELEASE")
        verification_height=max((r["object_table_clearance"] for r in self.trace), default=0.0)
        release_records=[r for r in self.trace if r["stage"]=="RELEASE"]
        release_height=(release_records[-1]["object_table_clearance"] if release_records else None)
        return {
            "seed": int(seed), "cube_xy": list(cube_xy or self.scene.config.cube_xy),
            "success": success, "failure_reason": self._failure_reason,
            "stage_reached": self._stage_reached, "stage_latency": dict(self.stage_latency),
            "verification_lift_height": verification_height,
            "hold_duration": (self.stage_latency.get("HOLD",0)*self.scene.robot.config.physics_dt*
                              self.config.physics_steps_per_control),
            "max_relative_drift": self._max_relative_drift,
            "max_contact_force": self._max_force,
            "minimum_aperture": self._minimum_aperture,
            "actual_final_aperture": final["estimated_aperture"],
            "release_height": release_height,
            "settle_linear_velocity": settle_linear,
            "settle_angular_velocity": settle_angular,
            "released_ee_object_distance": released_distance,
            "grasp_verified_once": self._verified_once,
            "non_finite_count": sum(not all(math.isfinite(float(v)) for v in r["q"]+r["dq"])
                                    for r in self.trace),
            "unsafe_abort": self._failure_reason in ("UNSAFE_FORCE","PENETRATION"),
            "resource_cleanup": self._resource_cleanup,
            "video_path": str(video_path) if video_path else None,
            "trace": self.trace,
        }
