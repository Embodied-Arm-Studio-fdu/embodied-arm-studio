import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import self_host


class SelfHostingTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.config = self.root / '.env.selfhost'
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()

    def tearDown(self):
        self.output.__exit__(None, None, None)
        self.directory.cleanup()

    def init(self, **kwargs):
        return self_host.initialize(self.config, **kwargs)

    def test_core_initialization_is_private_and_independent(self):
        values = self.init()
        self.assertEqual(0o600, self.config.stat().st_mode & 0o777)
        self.assertEqual(values, self_host.read_config(self.config))
        tokens = [values[key] for key in ('POSTGRES_PASSWORD', 'USER_JWT_SECRET', 'SIMULATION_SERVICE_TOKEN', 'INTERNAL_RUN_EVENT_SERVICE_TOKEN')]
        self.assertEqual(4, len(set(tokens)))
        self.assertTrue(all(len(token) == 96 for token in tokens))
        self.assertNotIn(tokens[0], self.output._new_target.getvalue())
        self.assertEqual('core', values['STUDIO_DEPLOY_MODE'])
        self.assertEqual('127.0.0.1', values['STUDIO_BIND_ADDRESS'])
        self.assertEqual('8080', values['STUDIO_HTTP_PORT'])

    def test_init_never_replaces_existing_credentials_or_symlinks(self):
        self.init()
        original = self.config.read_bytes()
        original_values = self_host.read_config(self.config)
        self.assertEqual(original_values, self.init())
        self.assertEqual(original, self.config.read_bytes())
        link = self.root / 'link.env'
        link.symlink_to(self.config)
        with self.assertRaises(ValueError):
            self_host.initialize(link)

    def test_environment_parser_never_executes_shell_content(self):
        self.config.write_text("OPENAI_API_KEY='$(touch /tmp/never-execute-selfhost)`nope`$literal'\n")
        self.config.chmod(0o600)
        self.assertEqual('$(touch /tmp/never-execute-selfhost)`nope`$literal', self_host.read_config(self.config)['OPENAI_API_KEY'])
        self.config.write_text('STUDIO_HTTP_PORT=8080\nSTUDIO_HTTP_PORT=9999\n')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self_host.read_config(self.config)

    def test_group_readable_secrets_are_rejected(self):
        self.init()
        self.config.chmod(0o644)
        with self.assertRaisesRegex(ValueError, '0600'):
            self_host.read_config(self.config)

    def test_gpu_init_requires_sdk_assets_and_nonloopback_video_host(self):
        sdk = self.root / 'SDK with spaces'
        native = sdk / 'python/isaacgym/_bindings/linux-x86_64/gym_38.so'
        native.parent.mkdir(parents=True)
        native.touch()
        assets = self.root / 'robot assets'
        urdf = assets / 'urdf/yhrg_s1_v2.urdf'
        urdf.parent.mkdir(parents=True)
        urdf.write_text('<robot/>')
        with patch.object(self_host, 'ROOT', self.root):
            values = self.init(mode='gpu', public_host='192.168.1.50', sdk=str(sdk), assets=str(assets), bind='0.0.0.0')
        self.assertEqual(str(sdk), values['ISAAC_GYM_PATH'])
        self.assertTrue(Path(values['REFERENCE_MODELS_PATH']).is_dir())
        self_host.doctor(self.config, values, offline=True)
        values['STUDIO_PUBLIC_HOST'] = '127.0.0.1'
        with self.assertRaises(ValueError):
            self_host.validate_config(values)

    def test_missing_sdk_fails_before_a_configuration_is_saved(self):
        with patch.object(self_host, 'ROOT', self.root):
            with self.assertRaises(ValueError):
                self.init(mode='gpu', public_host='192.168.1.50', sdk=str(self.root / 'missing'), assets=str(self.root / 'missing'))
        self.assertFalse(self.config.exists())

    def test_ssh_loopback_is_explicit_and_ports_do_not_collide(self):
        values = self.init()
        values.update(STUDIO_DEPLOY_MODE='gpu', STUDIO_PUBLIC_HOST='127.0.0.1',
                      STUDIO_WEB_ORIGIN='http://127.0.0.1:8080',
                      STUDIO_STREAM_ORIGIN='http://127.0.0.1:18889')
        with self.assertRaises(ValueError): self_host.validate_config(values)
        values['STUDIO_ALLOW_LOOPBACK_PLAYBACK'] = '1'
        # Resource validators are mocked; this is a config test, not SDK acceptance.
        with patch('scripts.reference_models.validate_models'), patch.object(Path, 'is_file', return_value=True):
            values.update(ISAAC_GYM_PATH='/opt/test-sdk', S1_ASSETS_PATH='/opt/test-s1')
            self_host.validate_config(values)
            for changes in ({'STUDIO_BIND_ADDRESS': '0.0.0.0'},
                            {'STUDIO_MEDIA_TCP_PORT': '8080'},
                            {'STUDIO_MEDIA_TCP_PORT': '18889'},
                            {'STUDIO_MEDIA_TCP_PORT': '0'},
                            {'STUDIO_ALLOW_LOOPBACK_PLAYBACK': 'true'},
                            {'STUDIO_WEB_ORIGIN': 'http://localhost:8080'}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    self_host.validate_config(dict(values, **changes))
            self_host.validate_config(dict(values, STUDIO_MEDIA_UDP_ADDRESS=''))

    def test_compose_uses_explicit_project_and_files_without_shell(self):
        values = self.init()
        command = self_host.compose_command(self.config, values)
        self.assertIn('--project-name', command)
        self.assertIn('embodied-arm-selfhost', command)
        self.assertEqual(1, command.count('-f'))
        values['STUDIO_DEPLOY_MODE'] = 'gpu'
        self.assertEqual(2, self_host.compose_command(self.config, values).count('-f'))
        with patch.dict(os.environ, {'USER_JWT_SECRET': 'foreign', 'COMPOSE_FILE': 'foreign.yml', 'OPENAI_API_KEY': 'foreign', 'REFERENCE_MODELS_PATH': '/foreign/models'}):
            env = self_host.compose_environment()
        for key in ('USER_JWT_SECRET', 'COMPOSE_FILE', 'OPENAI_API_KEY', 'REFERENCE_MODELS_PATH'):
            self.assertNotIn(key, env)

    def test_down_preserves_volumes_and_no_command_is_shell_executed(self):
        self.init()
        with patch.object(self_host.subprocess, 'run') as run:
            self.assertEqual(0, self_host.main(['--config', str(self.config), 'down']))
        command = run.call_args[0][0]
        self.assertEqual('down', command[-1])
        self.assertNotIn('-v', command)
        self.assertNotIn('shell', run.call_args[1])

    def test_offline_doctor_does_not_connect_or_start_services(self):
        values = self.init()
        with patch.object(self_host.subprocess, 'run') as run:
            self_host.doctor(self.config, values, offline=True)
        run.assert_not_called()

    def test_bootstrap_uses_module_entry_from_arbitrary_host_directory_for_core_and_gpu(self):
        values = self.init()
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            for mode, compose_file_count in (('core', 1), ('gpu', 2)):
                values['STUDIO_DEPLOY_MODE'] = mode
                with patch.object(self_host, 'read_config', return_value=values), \
                        patch.object(self_host.subprocess, 'run') as run:
                    self.assertEqual(
                        0,
                        self_host.main([
                            '--config', str(self.config),
                            'bootstrap', '--namespace', 'studio',
                        ]),
                    )
                command = run.call_args[0][0]
                self.assertEqual(compose_file_count, command.count('-f'))
                self.assertEqual(
                    ['run', '--rm', '--no-deps', 'bootstrap',
                     'python', '-m', 'scripts.bootstrap_self_host',
                     '--namespace', 'studio', '--apply'],
                    command[-10:],
                )
        finally:
            os.chdir(previous)

    def test_bootstrap_module_imports_without_pythonpath_from_image_workdir(self):
        dockerfile = (self_host.ROOT / 'docker/api.Dockerfile').read_text()
        self.assertIn('WORKDIR /app', dockerfile)
        environment = dict(os.environ)
        environment.pop('PYTHONPATH', None)
        completed = subprocess.run(
            [sys.executable, '-m', 'scripts.bootstrap_self_host', '--help'],
            cwd=self_host.ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)

    def test_compose_keeps_internal_services_private_and_artifact_paths_identical(self):
        import yaml
        base = yaml.safe_load((self_host.ROOT / 'deploy/compose.yaml').read_text())
        gpu = yaml.safe_load((self_host.ROOT / 'deploy/compose.gpu.yaml').read_text())
        for name, service in base['services'].items():
            if name != 'web':
                self.assertNotIn('ports', service)
        for name in ('redis', 'simulation', 'callbacks', 'worker'):
            self.assertNotIn('ports', gpu['services'][name])
        for name in ('simulation', 'callbacks', 'worker'):
            self.assertEqual('/app/var/artifacts', gpu['services'][name]['environment']['SIMULATION_ARTIFACTS_ROOT'])
            self.assertIn('studio_data:/app/var', gpu['services'][name]['volumes'])
        self.assertEqual('service:media', gpu['services']['worker']['network_mode'])
        self.assertEqual('rtsp://127.0.0.1:18554', gpu['services']['worker']['environment']['SIMULATION_STREAM_RTSP_BASE'])
        self.assertEqual(1, gpu['services']['worker']['deploy']['replicas'])

    def test_gpu_ports_and_origins_are_configurable_and_old_defaults_remain_valid(self):
        values = self.init()
        values.update(
            STUDIO_DEPLOY_MODE='gpu', STUDIO_PUBLIC_HOST='192.168.1.50',
            STUDIO_HTTP_PORT='18083', STUDIO_WEB_ORIGIN='http://192.168.1.50:18083',
            STUDIO_MEDIA_HTTP_PORT='18890', STUDIO_MEDIA_UDP_PORT='18190',
            STUDIO_STREAM_ORIGIN='http://192.168.1.50:18890',
        )
        native = self.root / 'sdk/python/isaacgym/_bindings/linux-x86_64/gym_38.so'
        urdf = self.root / 'assets/urdf/yhrg_s1_v2.urdf'
        for path in (native, urdf):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        values.update(ISAAC_GYM_PATH=str(self.root / 'sdk'), S1_ASSETS_PATH=str(self.root / 'assets'))
        with patch('scripts.reference_models.validate_models'):
            self_host.validate_config(values)
            for key in ('STUDIO_MEDIA_HTTP_PORT', 'STUDIO_MEDIA_UDP_PORT'):
                for invalid in ('0', '65536', 'abc', ''):
                    with self.subTest(key=key, invalid=invalid), self.assertRaises(ValueError):
                        self_host.validate_config(dict(values, **{key: invalid}))
            for changes in (
                {'STUDIO_MEDIA_HTTP_PORT': '18083'},
                {'STUDIO_STREAM_ORIGIN': 'http://192.168.1.50:18889'},
                {'STUDIO_WEB_ORIGIN': 'http://192.168.1.50:8080'},
            ):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    self_host.validate_config(dict(values, **changes))
            values.pop('STUDIO_MEDIA_HTTP_PORT')
            values.pop('STUDIO_MEDIA_UDP_PORT')
            values['STUDIO_STREAM_ORIGIN'] = 'http://192.168.1.50:18889'
            self_host.validate_config(values)

    def test_gpu_init_cli_wires_all_three_ports(self):
        with patch.object(self_host, 'initialize') as initialize:
            self.assertEqual(0, self_host.main([
                '--config', str(self.config), 'init', '--mode', 'gpu',
                '--port', '18083', '--media-http-port', '18890', '--media-udp-port', '18190',
            ]))
        self.assertEqual((18083, 'embodied-arm-selfhost', 18890, 18190), initialize.call_args[0][-4:])

    def test_gpu_check_validates_before_running_isolated_service(self):
        values = self.init()
        values['STUDIO_DEPLOY_MODE'] = 'gpu'
        with patch.object(self_host, 'read_config', return_value=values), \
                patch.object(self_host, 'validate_config') as validate, \
                patch.object(self_host.subprocess, 'run') as run:
            self.assertEqual(0, self_host.main(['--config', str(self.config), 'gpu-check']))
            validate.assert_called_once_with(values)
            self.assertEqual(['run', '--rm', '--no-deps', 'gpu-check'], run.call_args[0][0][-4:])
            validate.side_effect = ValueError('Invalid GPU config')
            run.reset_mock()
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(1, self_host.main(['--config', str(self.config), 'gpu-check']))
            run.assert_not_called()

    def test_gpu_check_service_does_not_require_media_or_register_a_worker(self):
        import yaml
        gpu = yaml.safe_load((self_host.ROOT / 'deploy/compose.gpu.yaml').read_text())
        service = gpu['services']['gpu-check']
        self.assertEqual('none', service['network_mode'])
        self.assertEqual({}, service['depends_on'])
        self.assertEqual(['tools'], service['profiles'])
        self.assertEqual(['python3', 'deploy/worker_entrypoint.py', '--check-only'], service['command'])
        self.assertEqual(gpu['services']['worker']['deploy'], service['deploy'])
        self.assertNotIn('ports', service)
        media = gpu['services']['media']
        self.assertIn('${STUDIO_MEDIA_UDP_PORT:-18189}', media['environment']['MTX_WEBRTCLOCALUDPADDRESS'])
