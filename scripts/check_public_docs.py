#!/usr/bin/env python3
"""Check user documentation links and export selection without building images."""
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

try:
    from scripts.export_release import ROOT, release_files
except ModuleNotFoundError:
    from export_release import ROOT, release_files


def main():
    documents = ['README.md', 'README.zh-CN.md', 'docs/quickstart.md',
                 'docs/quickstart.en.md', 'docs/user-guide.md', 'docs/deployment.md']
    selected = {str(path) for path in release_files()}
    errors = []
    for name in documents:
        path = ROOT / name
        text = path.read_text(encoding='utf-8')
        if name not in selected:
            errors.append(name + ': omitted from delivery files')
        for link in re.findall(r'\]\(([^)]+)\)', text):
            url = urlsplit(link)
            if url.scheme or url.netloc or not url.path:
                continue
            target = (path.parent / unquote(url.path)).resolve()
            if not target.is_relative_to(ROOT) or not target.exists():
                errors.append(name + ': missing local link ' + link)
            elif target.is_file() and str(target.relative_to(ROOT)) not in selected:
                errors.append(name + ': linked file omitted from delivery ' + link)
        if re.search(r'\]\([^)]*(?:docs/)?(?:internal/|developer\.md)', text):
            errors.append(name + ': internal/developer entry in user documentation')
    for name in documents[:2]:
        text = (ROOT / name).read_text()
        if re.search(r'\bTBD\b|待提供|Phase\s+A-\d+', text):
            errors.append(name + ': placeholder or internal phase')
        for user in ['adaa77', 'Kexin333', 'lxlhhh']:
            if text.count('https://github.com/' + user + ')') != 1:
                errors.append(name + ': missing or duplicated team member ' + user)
    if '[简体中文](README.zh-CN.md)' not in (ROOT / 'README.md').read_text():
        errors.append('Missing Chinese README link')
    if '[English](README.md)' not in (ROOT / 'README.zh-CN.md').read_text():
        errors.append('Missing English README link')
    if errors:
        raise SystemExit('\n'.join(errors))
    print('Checked 6 user documents, bilingual team links and delivery file selection; no image build.')


if __name__ == '__main__':
    main()
