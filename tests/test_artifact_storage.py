import os
import tempfile
import unittest
from pathlib import Path

from robotics.artifacts import (
    ArtifactPathError,
    ArtifactStorageConfigurationError,
    ArtifactStorageNotFound,
    LocalArtifactStorage,
)


class LocalArtifactStorageTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "artifacts"
        self.run_dir = self.root / "project-1" / "run-1"
        self.run_dir.mkdir(parents=True)
        self.storage = LocalArtifactStorage(self.root, chunk_size=4)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write(self, relative_path, content=b"artifact"):
        target = self.run_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return target

    def test_opens_regular_file_in_bounded_chunks(self):
        self.write("checkpoints/model.pt", b"0123456789")
        handle = self.storage.open(
            "project-1", "run-1", "checkpoints/model.pt"
        )
        chunks = list(handle.iter_chunks())
        self.assertEqual(chunks, [b"0123", b"4567", b"89"])
        self.assertTrue(handle.file.closed)

    def test_closing_partial_stream_releases_the_file(self):
        self.write("large.bin", b"0123456789")
        handle = self.storage.open("project-1", "run-1", "large.bin")
        chunks = handle.iter_chunks()
        self.assertEqual(next(chunks), b"0123")
        chunks.close()
        self.assertTrue(handle.file.closed)

    def test_root_must_be_explicit_safe_absolute_existing_directory(self):
        for root in (Path("relative"), Path("/"), self.root / "missing"):
            with self.subTest(root=root):
                with self.assertRaises(ArtifactStorageConfigurationError):
                    LocalArtifactStorage(root)

    def test_rejects_absolute_parent_backslash_and_noncanonical_paths(self):
        paths = (
            "/etc/passwd",
            "../outside",
            "logs/../secret",
            "logs\\output.log",
            "logs//output.log",
            "logs/./output.log",
            "logs/output.log/",
            "logs\r\nheader.log",
        )
        for path in paths:
            with self.subTest(path=path):
                with self.assertRaises(ArtifactPathError):
                    self.storage.open("project-1", "run-1", path)

    def test_missing_file_and_directory_are_unavailable(self):
        (self.run_dir / "directory").mkdir()
        for path in ("missing.bin", "directory"):
            with self.subTest(path=path):
                with self.assertRaises(ArtifactStorageNotFound):
                    self.storage.open("project-1", "run-1", path)

    def test_rejects_file_symlink_escape(self):
        outside = Path(self.temp_dir.name) / "secret.bin"
        outside.write_bytes(b"secret")
        (self.run_dir / "leak.bin").symlink_to(outside)
        with self.assertRaises(ArtifactPathError):
            self.storage.open("project-1", "run-1", "leak.bin")

    def test_rejects_directory_symlink_escape(self):
        outside = Path(self.temp_dir.name) / "outside"
        outside.mkdir()
        (outside / "secret.bin").write_bytes(b"secret")
        (self.run_dir / "linked").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ArtifactPathError):
            self.storage.open("project-1", "run-1", "linked/secret.bin")

    def test_rejects_project_or_run_directory_symlink_escape(self):
        outside = Path(self.temp_dir.name) / "outside-run"
        outside.mkdir()
        (outside / "secret.bin").write_bytes(b"secret")
        escaped = self.root / "project-1" / "run-linked"
        escaped.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ArtifactPathError):
            self.storage.open("project-1", "run-linked", "secret.bin")

    def test_rejects_unsafe_identifier_components(self):
        self.write("file.bin")
        for project_id, run_id in (
            ("../project", "run-1"),
            ("project-1", "../run"),
            ("project/other", "run-1"),
            ("project-1", "run\\other"),
        ):
            with self.subTest(project_id=project_id, run_id=run_id):
                with self.assertRaises(ArtifactPathError):
                    self.storage.open(project_id, run_id, "file.bin")


if __name__ == "__main__":
    unittest.main()
