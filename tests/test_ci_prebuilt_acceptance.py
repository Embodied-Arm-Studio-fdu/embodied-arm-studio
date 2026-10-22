import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import ci_prebuilt_acceptance


class ImageProvenanceTests(unittest.TestCase):
    def verify(self, entry, revision, version):
        reference = 'ghcr.io/team/worker@sha256:' + 'a' * 64
        manifest = {'source_sha': 'old-source', 'version': 'old-version',
                    'images': {'worker': {'reference': reference, **entry}}}
        image = {'Os': 'linux', 'Architecture': 'amd64', 'Id': 'image-id',
                 'RepoDigests': [reference], 'Config': {'Labels': {
                     'org.opencontainers.image.revision': revision,
                     'org.opencontainers.image.version': version}}}
        with patch.object(ci_prebuilt_acceptance.subprocess, 'run', side_effect=[
            SimpleNamespace(returncode=1), SimpleNamespace(returncode=1), SimpleNamespace(returncode=0),
        ]):
            with patch.object(ci_prebuilt_acceptance, 'command', return_value=json.dumps([image])):
                return ci_prebuilt_acceptance.pull('worker', manifest)

    def test_role_specific_source_and_version(self):
        result = self.verify({'source_sha': 'new-source', 'version': 'new-version'}, 'new-source', 'new-version')
        self.assertEqual('new-source', result['source_sha'])
        self.assertEqual('new-version', result['version'])

    def test_existing_uniform_manifest_remains_supported(self):
        result = self.verify({}, 'old-source', 'old-version')
        self.assertEqual('old-source', result['source_sha'])

    def test_wrong_role_revision_is_rejected(self):
        with self.assertRaises(AssertionError):
            self.verify({'source_sha': 'new-source', 'version': 'new-version'}, 'old-source', 'new-version')
