"""Verify and unpack only the three reference models required by canonical tasks."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'deploy/reference-models.json'
BUNDLE = ROOT / 'deploy/reference-models.tar.gz'


def required_models():
    return json.loads(MANIFEST.read_text())['files']


def validate_models(root):
    root = Path(root)
    for item in required_models():
        path = root / item['path']
        if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Missing or mismatched reference model: ' + item['path'])


def prepare_models(destination, bundle=BUNDLE):
    destination = Path(destination)
    expected = {item['path']: item for item in required_models()}
    # Read and verify the whole archive before writing anything. Never unpickle.
    payloads = {}
    with tarfile.open(str(bundle), 'r:gz') as archive:
        for member in archive.getmembers():
            if member.name not in expected or not member.isfile() or member.name in payloads or member.size > 2 * 1024 * 1024:
                raise ValueError('Unexpected reference-model archive entry')
            payload = archive.extractfile(member).read()
            if hashlib.sha256(payload).hexdigest() != expected[member.name]['sha256']:
                raise ValueError('Reference-model checksum mismatch')
            payloads[member.name] = payload
    if payloads.keys() != expected.keys():
        raise ValueError('Incomplete reference-model archive')
    for name, payload in payloads.items():
        target = destination / name
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_file() or target.read_bytes() != payload:
                raise ValueError('Refusing to overwrite existing reference model: ' + name)
    for name, payload in payloads.items():
        target = destination / name
        if target.exists():
            continue
        if destination.resolve() not in target.resolve().parents:
            raise ValueError('Reference-model destination escapes its directory')
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(payload)
    validate_models(destination)
    return destination.resolve()
