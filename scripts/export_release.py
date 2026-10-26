#!/usr/bin/env python3
"""Export a source-only, dependency-free self-hosting archive from an allowlist."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    'README.md', 'README.zh-CN.md', 'LICENSE', 'NOTICE.md',
    'THIRD_PARTY_NOTICES.md',
    '.gitignore', '.dockerignore', '.github', 'Makefile', 'requirements.txt', 'alembic.ini',
    'docker-compose.yml', 'deploy', 'docker', 'apps', 'agent', 'robotics', 'courses',
    'templates', 'migrations', 'tests', 'scripts',
    'docs/quickstart.md', 'docs/quickstart.en.md', 'docs/user-guide.md',
    'docs/deployment.md', 'docs/developer.md', 'docs/images',
    'simulation/src', 'simulation/scripts', 'simulation/tests',
)
EXCLUDED_PARTS = frozenset(('.git', '.venv', '.tools', 'node_modules', 'dist', 'var', '__pycache__', '.pytest_cache', '.vite', 'artifacts', 'snapshots', 'logs', 'test-results', 'playwright-report', 'coverage'))
EXCLUDED_PART_EXCEPTIONS = frozenset(('robotics/artifacts',))
EXCLUDED_SUFFIXES = frozenset(('.pyc', '.pyo', '.tsbuildinfo', '.log', '.sql', '.dump', '.pt', '.pth', '.onnx', '.mp4', '.webm', '.zip', '.tar', '.gz', '.pem', '.key', '.p12', '.db', '.sqlite', '.sqlite3', '.pkl', '.bak'))


def has_excluded_part(relative):
    parts = relative.parts
    for index, part in enumerate(parts):
        if part not in EXCLUDED_PARTS:
            continue
        path = '/'.join(parts[:index + 1])
        if path not in EXCLUDED_PART_EXCEPTIONS:
            return True
    return False


def release_files(root=ROOT):
    root = Path(root)
    result = set()
    for entry in SOURCE_PATHS:
        source = root / entry
        candidates = source.rglob('*') if source.is_dir() and not source.is_symlink() else [source]
        for path in candidates:
            relative = path.relative_to(root)
            if not path.is_file() or path.is_symlink() or has_excluded_part(relative):
                continue
            if path.suffix.lower() in EXCLUDED_SUFFIXES and str(relative) != 'deploy/reference-models.tar.gz':
                continue
            if (path.name.startswith('.env') and path.name != '.env.example') or (path.name.endswith('.env') and not path.name.endswith('.env.example')):
                continue
            result.add(relative)
    return sorted(result)


def export_release(output, root=ROOT):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    files = release_files(root)
    manifest = {str(name): hashlib.sha256((Path(root) / name).read_bytes()).hexdigest() for name in files}
    # Exclusive creation avoids overwriting an earlier release or another file.
    with output.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:gz', dereference=False) as archive:
            for name in files:
                archive.add(str(Path(root) / name), arcname=str(Path('embodied-arm-studio') / name), recursive=False)
    output.with_name(output.name + '.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Exported %d source files to %s' % (len(files), output))
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    export_release(args.output)


if __name__ == '__main__':
    main()
