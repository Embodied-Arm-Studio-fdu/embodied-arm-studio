import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy import delete

from apps.api.main import create_app
from apps.api.database import create_postgres_engine, create_session_factory
from tests.auth import StaticCurrentUserProvider
from robotics.runs import (
    ExecutionMode,
    InMemoryRunRepository,
    RunArtifact,
    RunKind,
    RunNotFound,
)
from robotics.runs.db_models import RunArtifactRow, RunRow
from tests.http_client import ASGITestClient
from tests.postgres_users import delete_test_user, seed_test_user
from tests.test_terminal_callback_api import SAFE_POSTGRES_URL


SHA256 = "a" * 64


class FakeArtifactMetadataStore:
    def __init__(self):
        self.artifacts = {}

    def add(self, artifact):
        self.artifacts[artifact.artifact_id] = artifact

    def list_artifacts(self, run_id):
        return sorted(
            [item for item in self.artifacts.values() if item.run_id == run_id],
            key=lambda item: item.path,
        )

    def get_artifact(self, run_id, artifact_id):
        artifact = self.artifacts.get(artifact_id)
        if artifact is None or artifact.run_id != run_id:
            raise RunNotFound(run_id)
        return artifact


class ArtifactDownloadApiTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.artifact_root = root / "artifact-store"
        self.artifact_root.mkdir()
        self.metadata = FakeArtifactMetadataStore()
        self.app = create_app(
            root / "runs",
            tick_seconds=0,
            project_root=root / "projects",
            run_repository=InMemoryRunRepository(),
            current_user_provider=StaticCurrentUserProvider("trusted-user"),
            artifact_storage_root=self.artifact_root,
            artifact_metadata_service=self.metadata,
        )
        self.client = ASGITestClient(self.app)
        self.run = self.app.state.run_service.create_run(
            user_id="trusted-user",
            project_id="project-1",
            execution_mode=ExecutionMode.SIMULATION,
            run_kind=RunKind.TRAIN,
            parameters={},
            project_config_snapshot={},
            code="",
        )
        self.run_dir = self.artifact_root / "project-1" / self.run.run_id
        self.run_dir.mkdir(parents=True)

    def tearDown(self):
        self.client.close()
        self.temp_dir.cleanup()

    def artifact(
        self,
        artifact_id="artifact-1",
        path="manifest.json",
        artifact_type="manifest",
        size_bytes=None,
        sha256=None,
        run_id=None,
    ):
        item = RunArtifact(
            artifact_id=artifact_id,
            run_id=run_id or self.run.run_id,
            artifact_type=artifact_type,
            path=path,
            size_bytes=size_bytes,
            sha256=sha256,
            is_primary_model=artifact_type.startswith("checkpoint"),
            created_at=datetime.now(timezone.utc),
        )
        self.metadata.add(item)
        return item

    def write(self, path, content=b"{}"):
        target = self.run_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return target

    def url(self, artifact_id, run_id=None):
        return "/api/v1/runs/%s/artifacts/%s/download" % (
            run_id or self.run.run_id,
            artifact_id,
        )

    def test_fresh_root_is_created_before_storage_and_reinit_preserves_files(self):
        fresh = Path(self.temp_dir.name) / 'fresh-volume' / 'artifacts'
        def initialize():
            return create_app(
                Path(self.temp_dir.name) / 'fresh-runs', tick_seconds=0,
                project_root=Path(self.temp_dir.name) / 'fresh-projects',
                run_repository=InMemoryRunRepository(),
                current_user_provider=StaticCurrentUserProvider("trusted-user"),
                artifact_storage_root=fresh, artifact_metadata_service=self.metadata,
            )
        self.assertFalse(fresh.exists())
        self.assertIsNotNone(initialize().state.artifact_download_service)
        self.assertEqual(0o750, fresh.stat().st_mode & 0o777)
        sentinel = fresh / 'existing.bin'; sentinel.write_bytes(b'preserve me')
        fresh.chmod(0o700)
        self.assertIsNotNone(initialize().state.artifact_download_service)
        self.assertEqual(b'preserve me', sentinel.read_bytes())
        self.assertEqual(0o700, fresh.stat().st_mode & 0o777)

    def test_list_and_download_registered_manifest(self):
        self.write("manifest.json", b'{"ok":true}')
        item = self.artifact(size_bytes=11, sha256=SHA256)
        listed = self.client.get(
            "/api/v1/runs/%s/artifacts" % self.run.run_id
        )
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json()[0]["artifact_id"], item.artifact_id)
        response = self.client.get(self.url(item.artifact_id))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.content, b'{"ok":true}')
        self.assertEqual(response.headers["content-type"], "application/json")
        self.assertEqual(response.headers["etag"], '"%s"' % SHA256)
        self.assertEqual(response.headers["content-length"], "11")
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(
            response.headers["content-disposition"],
            'attachment; filename="manifest.json"',
        )

    def test_registered_artifact_types_have_safe_content_types(self):
        cases = (
            ("training_config", "training_config.json", "application/json"),
            ("manifest", "manifest.json", "application/json"),
            ("training_metrics", "training.jsonl", "application/x-ndjson"),
            ("training_episode_metrics", "episodes.jsonl", "application/x-ndjson"),
            ("evaluation_summary", "summary.json", "application/json"),
            ("evaluation_episodes", "evaluation.jsonl", "application/x-ndjson"),
            ("stdout_log", "stdout.log", "text/plain; charset=utf-8"),
            ("stderr_log", "stderr.log", "text/plain; charset=utf-8"),
            ("task_log", "task.log", "text/plain; charset=utf-8"),
            ("training_video", "overview.mp4", "video/mp4"),
            ("evaluation_video", "evaluation.mp4", "video/mp4"),
            ("checkpoint_best", "model.pt", "application/octet-stream"),
            ("checkpoint_latest", "model-latest.pt", "application/octet-stream"),
            ("html_report", "report.html", "application/octet-stream"),
        )
        for index, (artifact_type, name, expected) in enumerate(cases):
            with self.subTest(artifact_type=artifact_type):
                path = "files/%s" % name
                self.write(path, b"payload")
                item = self.artifact(
                    artifact_id="artifact-%d" % index,
                    path=path,
                    artifact_type=artifact_type,
                )
                response = self.client.get(self.url(item.artifact_id))
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["content-type"], expected)
                self.assertTrue(
                    response.headers["content-disposition"].startswith("attachment;")
                )
                self.assertEqual(response.headers["x-content-type-options"], "nosniff")

    def test_unknown_run_unauthorized_run_and_unknown_artifact_are_hidden(self):
        other = self.app.state.run_service.create_run(
            user_id="another-user",
            project_id="project-1",
            execution_mode=ExecutionMode.SIMULATION,
            run_kind=RunKind.TRAIN,
            parameters={},
            project_config_snapshot={},
            code="",
        )
        for url in (
            self.url("missing"),
            self.url("missing", "missing-run"),
            self.url("missing", other.run_id),
        ):
            with self.subTest(url=url):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 404)
                self.assertNotIn(str(self.artifact_root), response.text)
        self.assertEqual(
            self.client.get("/api/v1/runs/%s/artifacts" % other.run_id).status_code,
            404,
        )

    def test_artifact_id_from_another_run_is_not_downloadable(self):
        other = self.app.state.run_service.create_run(
            user_id="trusted-user",
            project_id="project-1",
            execution_mode=ExecutionMode.SIMULATION,
            run_kind=RunKind.TRAIN,
            parameters={},
            project_config_snapshot={},
            code="",
        )
        item = self.artifact(artifact_id="other-artifact", run_id=other.run_id)
        response = self.client.get(self.url(item.artifact_id))
        self.assertEqual(response.status_code, 404)

    def test_unregistered_disk_file_and_missing_registered_file_are_hidden(self):
        self.write("unregistered.bin", b"secret")
        missing = self.artifact(path="missing.bin", artifact_type="checkpoint_latest")
        for artifact_id in ("unregistered", missing.artifact_id):
            response = self.client.get(self.url(artifact_id))
            self.assertEqual(response.status_code, 404)
            self.assertNotIn(str(self.artifact_root), response.text)

    def test_unsafe_registered_paths_are_rejected_without_path_disclosure(self):
        paths = (
            "/etc/passwd",
            "../secret",
            "logs\\secret.log",
            "logs//secret.log",
            "logs/./secret.log",
            "logs\r\nX-Injected.log",
        )
        for index, path in enumerate(paths):
            with self.subTest(path=path):
                item = self.artifact(
                    artifact_id="unsafe-%d" % index,
                    path=path,
                    artifact_type="log",
                )
                response = self.client.get(self.url(item.artifact_id))
                self.assertEqual(response.status_code, 404)
                self.assertNotIn(str(self.artifact_root), response.text)
                self.assertNotIn("X-Injected", response.headers)

    def test_symlink_file_and_directory_escape_are_rejected(self):
        outside = Path(self.temp_dir.name) / "private"
        outside.mkdir()
        (outside / "secret.bin").write_bytes(b"secret")
        (self.run_dir / "leak.bin").symlink_to(outside / "secret.bin")
        (self.run_dir / "linked").symlink_to(outside, target_is_directory=True)
        first = self.artifact(artifact_id="file-link", path="leak.bin")
        second = self.artifact(artifact_id="dir-link", path="linked/secret.bin")
        for item in (first, second):
            response = self.client.get(self.url(item.artifact_id))
            self.assertEqual(response.status_code, 404)
            self.assertNotIn(str(outside), response.text)

    def test_size_mismatch_is_a_redacted_integrity_conflict(self):
        self.write("model.pt", b"actual")
        item = self.artifact(
            path="model.pt",
            artifact_type="checkpoint_best",
            size_bytes=999,
        )
        response = self.client.get(self.url(item.artifact_id))
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"], "Artifact integrity check failed")
        self.assertNotIn(str(self.artifact_root), response.text)

    def test_invalid_registered_sha256_cannot_enter_etag_header(self):
        self.write("model.pt", b"actual")
        item = self.artifact(
            path="model.pt",
            artifact_type="checkpoint_best",
            sha256="bad\r\nX-Injected:" + "a" * 48,
        )
        response = self.client.get(self.url(item.artifact_id))
        self.assertEqual(response.status_code, 409)
        self.assertNotIn("X-Injected", response.headers)

    def test_safe_filename_removes_non_ascii_and_header_metacharacters(self):
        self.write("reports/résumé (final).json", b"{}")
        item = self.artifact(path="reports/résumé (final).json")
        response = self.client.get(self.url(item.artifact_id))
        self.assertEqual(response.status_code, 200)
        disposition = response.headers["content-disposition"]
        self.assertNotIn("\r", disposition)
        self.assertNotIn("\n", disposition)
        self.assertEqual(disposition, 'attachment; filename="r_sum___final_.json"')

    def test_large_file_is_chunked_by_storage_and_downloaded_intact(self):
        content = b"0123456789abcdef" * 131072
        self.write("checkpoints/large.pt", content)
        item = self.artifact(
            path="checkpoints/large.pt",
            artifact_type="checkpoint_latest",
            size_bytes=len(content),
        )
        response = self.client.get(self.url(item.artifact_id))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, content)
        self.assertEqual(response.headers["content-length"], str(len(content)))

    def test_download_does_not_change_or_stop_run(self):
        self.write("manifest.json", b"{}")
        item = self.artifact(size_bytes=2)
        before = self.app.state.run_service.get_run(self.run.run_id)
        self.assertEqual(self.client.get(self.url(item.artifact_id)).status_code, 200)
        after = self.app.state.run_service.get_run(self.run.run_id)
        self.assertEqual(after, before)

    def test_training_video_is_authenticated_complete_200_without_range_contract(self):
        content = b"complete-mp4-blob"
        self.write("video/overview.mp4", content)
        item = self.artifact(
            path="video/overview.mp4",
            artifact_type="training_video",
            size_bytes=len(content),
        )
        response = self.client.get(
            self.url(item.artifact_id), headers={"Range": "bytes=0-3"}
        )
        self.assertEqual(200, response.status_code)
        self.assertEqual(content, response.content)
        self.assertEqual("video/mp4", response.headers["content-type"])
        self.assertNotIn("content-range", response.headers)


@unittest.skipUnless(
    SAFE_POSTGRES_URL is not None,
    "PostgreSQL test configuration is unavailable",
)
class ArtifactDownloadPostgreSQLRecoveryTest(unittest.TestCase):
    def test_rebuilt_app_downloads_postgresql_registered_temp_artifact(self):
        temp_dir = tempfile.TemporaryDirectory()
        root = Path(temp_dir.name)
        artifact_root = root / "artifact-store"
        artifact_root.mkdir()
        user_id = "p63-%s" % uuid4().hex
        artifact_id = "artifact-%s" % uuid4().hex
        clients = []
        engine = create_postgres_engine(SAFE_POSTGRES_URL)
        factory = create_session_factory(engine)
        seed_test_user(factory, user_id)
        run_id = None
        try:
            first = create_app(
                root / "first-runs",
                tick_seconds=0,
                project_root=root / "first-projects",
                run_repository_mode="postgresql",
                database_url=SAFE_POSTGRES_URL,
                current_user_provider=StaticCurrentUserProvider(user_id),
                artifact_storage_root=artifact_root,
            )
            first_client = ASGITestClient(first)
            clients.append(first_client)
            run = first.state.run_service.create_run(
                user_id=user_id,
                project_id="project-recovery",
                execution_mode=ExecutionMode.SIMULATION,
                run_kind=RunKind.TRAIN,
                parameters={},
                project_config_snapshot={},
                code="",
            )
            run_id = run.run_id
            run_dir = artifact_root / run.project_id / run.run_id
            run_dir.mkdir(parents=True)
            content = b"persistent registered artifact"
            (run_dir / "manifest.json").write_bytes(content)
            with factory() as session:
                session.add(
                    RunArtifactRow(
                        artifact_id=artifact_id,
                        run_id=run.run_id,
                        artifact_type="manifest",
                        path="manifest.json",
                        size_bytes=len(content),
                        sha256=SHA256,
                        is_primary_model=False,
                        created_at=datetime.now(timezone.utc),
                    )
                )
                session.commit()

            second = create_app(
                root / "second-runs",
                tick_seconds=0,
                project_root=root / "second-projects",
                run_repository_mode="postgresql",
                database_url=SAFE_POSTGRES_URL,
                current_user_provider=StaticCurrentUserProvider(user_id),
                artifact_storage_root=artifact_root,
            )
            second_client = ASGITestClient(second)
            clients.append(second_client)
            response = second_client.get(
                "/api/v1/runs/%s/artifacts/%s/download"
                % (run.run_id, artifact_id)
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.content, content)
            self.assertEqual(response.headers["etag"], '"%s"' % SHA256)
        finally:
            for client in reversed(clients):
                client.close()
            if run_id is not None:
                with factory() as session:
                    session.execute(
                        delete(RunArtifactRow).where(RunArtifactRow.run_id == run_id)
                    )
                    session.execute(delete(RunRow).where(RunRow.run_id == run_id))
                    session.commit()
            delete_test_user(factory, user_id)
            engine.dispose()
            temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
