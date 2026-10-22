import contextlib
import io
import json
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import self_host, start


class GuidedStartTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = Path(self.temp.name) / 'private.env'
        self.output = contextlib.ExitStack()
        self.output.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.output.enter_context(contextlib.redirect_stderr(io.StringIO()))
        self.addCleanup(self.output.close)

    def test_unpublished_default_stops_before_configuration_or_docker(self):
        pending = Path(self.temp.name) / 'pending-images.json'
        pending.write_text(json.dumps({'status': 'pending-rights-and-registry'}))
        with patch('builtins.input', return_value='core'), patch.object(start.subprocess, 'run') as run:
            self.assertEqual(1, start.main(['--config', str(self.config), '--image-manifest', str(pending)]))
        run.assert_not_called()
        self.assertFalse(self.config.exists())

    def test_fixed_manifest_and_no_latest(self):
        manifest = Path(self.temp.name) / 'images.json'
        for reference, valid in [('registry.example/org/api@sha256:' + 'a' * 64, True),
                                 ('registry.example/org/api:latest', False)]:
            manifest.write_text(json.dumps({'status': 'released', 'images':
                {role: {'reference': reference} for role in ('api', 'web')}}))
            if valid:
                self.assertEqual(2, len(start.image_settings(manifest, 'core')))
            else:
                with self.assertRaises(ValueError): start.image_settings(manifest, 'core')

    def test_fresh_build_and_repeat_keep_credentials(self):
        with patch('builtins.input', side_effect=['core', 'ci-a5', '19095', '', '']), \
             patch.object(start.host.shutil, 'which', return_value='/usr/bin/docker'), \
             patch.object(start, 'ensure_available'), patch.object(start.host, 'main', return_value=0) as main:
            self.assertEqual(0, start.main(['--config', str(self.config), '--source', 'build']))
            self.assertEqual(['up', 'bootstrap'], [call.args[0][-1] for call in main.call_args_list])
        before = self.config.read_bytes()
        with patch('builtins.input', side_effect=AssertionError('must reuse config')), \
             patch.object(start.host, 'main', return_value=0):
            self.assertEqual(0, start.main(['--config', str(self.config)]))
        self.assertEqual(before, self.config.read_bytes())
        self.assertEqual(0o600, self.config.stat().st_mode & 0o777)

    def test_health_failure_never_bootstraps_or_announces_success(self):
        self_host.initialize(self.config)
        with patch.object(start.host, 'main', return_value=1) as main:
            self.assertEqual(1, start.main(['--config', str(self.config)]))
        self.assertEqual(1, main.call_count)

    def test_project_collision_refuses_before_mutation(self):
        with patch.object(start.subprocess, 'check_output', return_value='existing-container'):
            with self.assertRaisesRegex(ValueError, '项目名已存在'):
                start.ensure_available({'STUDIO_PROJECT_NAME': 'occupied'})

    def test_check_never_initializes(self):
        self.assertEqual(1, start.main(['--config', str(self.config), '--check']))
        self.assertFalse(self.config.exists())
        values = self_host.initialize(self.config)
        with patch.object(start.host, 'doctor') as doctor, patch.object(start.host, 'main') as main:
            self.assertEqual(0, start.main(['--config', str(self.config), '--check']))
        doctor.assert_called_once_with(self.config, values)
        main.assert_not_called()

    def test_prebuilt_up_never_builds_and_bootstrap_has_tty(self):
        self_host.initialize(self.config)
        with self.config.open('a') as target:
            target.write("STUDIO_IMAGE_SOURCE='prebuilt'\n")
            for role in ('API', 'WEB'):
                target.write('STUDIO_IMAGE_' + role + "='registry.example/org/image@sha256:" + 'a' * 64 + "'\n")
        with patch.object(self_host, 'doctor'), patch.object(self_host.subprocess, 'run') as run:
            self.assertEqual(0, self_host.main(['--config', str(self.config), 'up']))
        command = run.call_args.args[0]
        self.assertIn('--no-build', command)
        self.assertNotIn('--build', command)
        self.assertEqual('always', command[command.index('--pull') + 1])
        with patch.object(self_host.sys.stdin, 'isatty', return_value=True), patch.object(self_host.subprocess, 'run') as run:
            self.assertEqual(0, self_host.main(['--config', str(self.config), 'bootstrap']))
        bootstrap = run.call_args.args[0]
        self.assertIn('-it', bootstrap)
        self.assertNotIn('--no-build', bootstrap)
        self.assertNotIn('--build', bootstrap)
        self.assertEqual('never', bootstrap[bootstrap.index('--pull') + 1])
        self.assertEqual(['docker', 'image', 'inspect'], run.call_args_list[0].args[0][:3])

    def test_missing_prebuilt_bootstrap_image_stops_before_compose_run(self):
        self_host.initialize(self.config)
        with self.config.open('a') as target:
            target.write("STUDIO_IMAGE_SOURCE='prebuilt'\nSTUDIO_IMAGE_API='registry.example/api@sha256:" + 'a' * 64 + "'\n")
        with patch.object(self_host.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'inspect')) as run:
            self.assertEqual(1, self_host.main(['--config', str(self.config), 'bootstrap']))
        self.assertEqual(1, run.call_count)
        self.assertEqual(['docker', 'image', 'inspect'], run.call_args.args[0][:3])

    def test_existing_source_cannot_silently_switch(self):
        self_host.initialize(self.config)
        with patch.object(start.host, 'main') as main:
            self.assertEqual(1, start.main(['--config', str(self.config), '--source', 'prebuilt']))
        main.assert_not_called()
