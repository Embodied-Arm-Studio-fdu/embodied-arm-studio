import hashlib
import io
from pathlib import Path
import re
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts import reference_models, export_release


class ReferenceModelsTest(unittest.TestCase):
    def test_actual_bundle_contains_only_the_three_verified_models(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = reference_models.prepare_models(Path(directory) / 'models')
            reference_models.validate_models(destination)
            self.assertEqual(3, len(list(destination.rglob('*.pt'))))
            before = {p: p.stat().st_mtime_ns for p in destination.rglob('*.pt')}
            reference_models.prepare_models(destination)
            self.assertEqual(before, {p: p.stat().st_mtime_ns for p in destination.rglob('*.pt')})

    def test_worker_rechecks_reference_models_before_registering_with_rq(self):
        from deploy.worker_entrypoint import validate_reference_models
        with tempfile.TemporaryDirectory() as directory:
            root = reference_models.prepare_models(Path(directory) / 'models')
            validate_reference_models(root, reference_models.MANIFEST)
            first = root / reference_models.required_models()[0]['path']
            first.write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'reference model'):
                validate_reference_models(root, reference_models.MANIFEST)

    def test_tampered_checkpoint_and_archive_traversal_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'bad.tar.gz'
            with tarfile.open(str(archive), 'w:gz') as target:
                entry = tarfile.TarInfo('../escape.pt')
                entry.size = 3
                target.addfile(entry, io.BytesIO(b'bad'))
            with self.assertRaises(ValueError):
                reference_models.prepare_models(root / 'models', archive)
            self.assertFalse((root / 'escape.pt').exists())
            models = reference_models.prepare_models(root / 'models')
            first = models / reference_models.required_models()[0]['path']
            first.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                reference_models.prepare_models(models)
            self.assertEqual(b'changed', first.read_bytes())

    def test_source_export_excludes_private_files_symlinks_and_runtime_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('README.md', 'apps/web/src/App.tsx', 'apps/web/tsconfig.app.tsbuildinfo', 'apps/web/.env.local', 'apps/web/node_modules/package.json', 'apps/api/data.sqlite', 'apps/api/var/runs/output.json', 'robotics/artifacts/__init__.py', 'robotics/artifacts/service.py', 'robotics/artifacts/storage.py', 'robotics/artifacts/__pycache__/service.pyc', 'simulation/src/module.py', 'simulation/artifacts/user/log.txt', 'scripts/test.py', 'deploy/.env.example', 'deploy/private.key', 'deploy/reference-models.tar.gz'):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('test')
            (root / 'scripts/linked.py').symlink_to(root / 'README.md')
            files = [str(path) for path in export_release.release_files(root)]
            self.assertIn('apps/web/src/App.tsx', files)
            self.assertIn('deploy/.env.example', files)
            self.assertIn('deploy/reference-models.tar.gz', files)
            self.assertIn('robotics/artifacts/__init__.py', files)
            self.assertIn('robotics/artifacts/service.py', files)
            self.assertIn('robotics/artifacts/storage.py', files)
            for name in files:
                self.assertNotIn('node_modules', name)
                self.assertFalse(name.endswith('.tsbuildinfo'))
                self.assertNotIn('.env.local', name)
                self.assertNotIn('linked.py', name)
                self.assertNotIn('.sqlite', name)
                self.assertNotIn('/var/', name)
                self.assertNotIn('__pycache__', name)
                self.assertNotIn('simulation/artifacts', name)
                self.assertFalse(name.endswith('.key'))

    def test_api_dockerfile_source_copy_is_covered_by_export(self):
        dockerfile = (export_release.ROOT / 'docker/api.Dockerfile').read_text()
        copied_sources = {
            match.group(1).rstrip('/')
            for match in re.finditer(r'^COPY ([^ ]+) ', dockerfile, re.MULTILINE)
        }
        self.assertTrue(copied_sources)
        for source in copied_sources:
            self.assertTrue(
                any(
                    entry == source or entry.startswith(source + '/')
                    for entry in export_release.SOURCE_PATHS
                ),
                source,
            )

        exported = {str(path) for path in export_release.release_files()}
        for source in copied_sources:
            path = export_release.ROOT / source
            if path.is_file():
                self.assertIn(source, exported)
                continue
            for module in path.rglob('*.py'):
                relative = module.relative_to(export_release.ROOT)
                if not export_release.has_excluded_part(relative):
                    self.assertIn(str(relative), exported)

    def test_asset_override_is_read_at_construction_and_explicit_paths_win(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('portable_s1_config', reference_models.ROOT / 'simulation/src/simulation_core/s1_isaac_gym_adapter.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        S1IsaacGymConfig = module.S1IsaacGymConfig
        with patch.dict('os.environ', {'SIMULATION_S1_ASSET_ROOT': '/portable/robot'}):
            self.assertEqual(Path('/portable/robot'), S1IsaacGymConfig().asset_root)
            self.assertEqual(Path('/explicit/robot'), S1IsaacGymConfig(asset_root='/explicit/robot').asset_root)

    def test_consolidated_runtime_files_are_exported(self):
        files = {str(p) for p in export_release.release_files()}
        for name in ('apps/web/src/WhepPlayer.tsx', 'apps/web/vite.config.ts',
                     'apps/web/vite.config.js', 'docker/web-nginx.conf',
                     'deploy/worker_entrypoint.py', 'deploy/nvidia-egl-icd.json',
                     'simulation/src/simulation_core/media_streaming.py',
                     'deploy/compose.yaml', 'deploy/compose.gpu.yaml',
                     'deploy/mediamtx.yml', 'deploy/.env.example'):
            self.assertIn(name, files)
