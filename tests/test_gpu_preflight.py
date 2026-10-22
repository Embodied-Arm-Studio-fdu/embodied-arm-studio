"""Native probe contracts with mocked SDKs; these tests never access a GPU."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from deploy import worker_entrypoint as entry


class GPUPreflightTest(unittest.TestCase):
    def test_cuda_computes_and_synchronizes_and_rejects_failure(self):
        torch = MagicMock()
        torch.cuda.current_device.return_value = 0
        result = torch.ones.return_value.square.return_value.sum.return_value
        result.item.return_value = 8.0
        with patch.dict(sys.modules, {'torch': torch, 'isaacgym': MagicMock()}):
            entry.check_cuda()
            torch.cuda.init.assert_called_once_with()
            torch.ones.assert_called_once_with(8, device='cuda:0')
            torch.cuda.synchronize.assert_called_once_with(0)
            result.item.return_value = 0
            with self.assertRaisesRegex(RuntimeError, 'mismatch'):
                entry.check_cuda()
            torch.cuda.is_available.return_value = False
            with self.assertRaisesRegex(RuntimeError, 'CUDA unavailable'):
                entry.check_cuda()

    def camera_api(self):
        module = MagicMock()
        api = module.gymapi
        gym = api.acquire_gym.return_value
        gym.create_actor.return_value = 0
        gym.create_camera_sensor.return_value = 0
        gym.get_camera_image.return_value.size = 64 * 64 * 4
        gym.get_camera_image.return_value.dtype = 'uint8'
        gym.get_actor_rigid_body_states.return_value = {'pose': {'p': {a: [0.5] for a in ('x', 'y', 'z')}}}
        return module, api, gym

    def test_camera_renders_reads_rgba_and_releases_resources(self):
        module, api, gym = self.camera_api()
        with patch.dict(sys.modules, {'isaacgym': module}):
            entry.check_scene(True)
        gym.create_sim.assert_called_once_with(0, 0, api.SIM_PHYSX, api.SimParams.return_value)
        self.assertTrue(api.SimParams.return_value.physx.use_gpu)
        self.assertFalse(api.SimParams.return_value.use_gpu_pipeline)
        gym.render_all_camera_sensors.assert_called_once()
        gym.get_camera_image.assert_called_once()
        gym.destroy_camera_sensor.assert_called_once()
        gym.destroy_sim.assert_called_once()

    def test_camera_failures_destroy_partial_simulator(self):
        for failure in ('sim', 'env', 'actor', 'camera', 'render', 'frame'):
            module, api, gym = self.camera_api()
            if failure == 'sim': gym.create_sim.return_value = None
            elif failure == 'env': gym.create_env.return_value = None
            elif failure == 'actor': gym.create_actor.return_value = -1
            elif failure == 'camera': gym.create_camera_sensor.return_value = -1
            elif failure == 'render': gym.render_all_camera_sensors.side_effect = RuntimeError('Vulkan failed')
            else: gym.get_camera_image.return_value.size = 0
            with self.subTest(failure=failure), patch.dict(sys.modules, {'isaacgym': module}):
                with self.assertRaises(RuntimeError): entry.check_scene(True)
                self.assertEqual(0 if failure == 'sim' else 1, gym.destroy_sim.call_count)
                self.assertEqual(1 if failure in ('render', 'frame') else 0, gym.destroy_camera_sensor.call_count)

    def test_physx_requires_finite_state_without_graphics(self):
        module, api, gym = self.camera_api()
        with patch.dict(sys.modules, {'isaacgym': module}):
            entry.check_scene(False)
            gym.create_sim.assert_called_once_with(0, -1, api.SIM_PHYSX, api.SimParams.return_value)
            gym.get_actor_rigid_body_states.return_value['pose']['p']['x'] = [float('nan')]
            with self.assertRaisesRegex(RuntimeError, 'Non-finite'):
                entry.check_scene(False)
        self.assertEqual(2, gym.destroy_sim.call_count)

    def test_nvenc_requires_output_and_cleans_temp_files(self):
        with tempfile.TemporaryDirectory() as directory:
            ffmpeg = Path(directory) / 'ffmpeg'; ffmpeg.touch()
            outputs = []
            def encode(command, **kwargs):
                self.assertEqual('h264_nvenc', command[command.index('-c:v') + 1])
                self.assertEqual('2', command[command.index('-frames:v') + 1])
                self.assertEqual(30, kwargs['timeout'])
                outputs.append(Path(command[-1])); outputs[-1].write_bytes(b'h264')
                return subprocess.CompletedProcess(command, 0, '', '')
            with patch.dict(os.environ, {'SIMULATION_FFMPEG_PATH': str(ffmpeg)}):
                with patch.object(entry.subprocess, 'run', side_effect=encode): entry.check_nvenc()
                self.assertFalse(outputs[0].parent.exists())
                for result in (subprocess.CompletedProcess([], 0, '', ''), subprocess.CompletedProcess([], 1, '', 'failed')):
                    with patch.object(entry.subprocess, 'run', return_value=result), self.assertRaises(RuntimeError):
                        entry.check_nvenc()
                with patch.object(entry.subprocess, 'run', side_effect=subprocess.TimeoutExpired('ffmpeg', 30)):
                    with self.assertRaises(subprocess.TimeoutExpired): entry.check_nvenc()

    def test_probe_dispatch_includes_vulkan(self):
        with patch.object(entry, 'check_vulkan') as check:
            entry.run_probe('vulkan')
            check.assert_called_once_with()

    def test_check_only_isolates_all_probes_and_stops_on_failure(self):
        with patch.object(entry.sys, 'version_info', (3, 8)), patch.object(entry, 'validate_reference_models'), \
                patch.dict(sys.modules, {'simulation_core.s1_isaac_gym_adapter': MagicMock()}), \
                patch.object(entry.os, 'execv') as execute, \
                patch.object(entry.sys, 'argv', ['worker_entrypoint.py', '--check-only']), \
                patch.object(entry.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as run:
            entry.main()
            self.assertEqual(list(entry.PROBES), [c.args[0][-1] for c in run.call_args_list])
            for call in run.call_args_list:
                self.assertEqual(120, call.kwargs['timeout'])
                self.assertEqual('--probe', call.args[0][-2])
            execute.assert_not_called()
            for failed_index in range(len(entry.PROBES)):
                run.reset_mock()
                run.side_effect = [subprocess.CompletedProcess([], 0)] * failed_index + [subprocess.CompletedProcess([], -11)]
                with self.assertRaisesRegex(RuntimeError, 'signal 11'): entry.main()
                self.assertEqual(failed_index + 1, run.call_count)
                execute.assert_not_called()
            run.side_effect = subprocess.TimeoutExpired('probe', 120)
            with self.assertRaises(subprocess.TimeoutExpired): entry.main()
            execute.assert_not_called()
