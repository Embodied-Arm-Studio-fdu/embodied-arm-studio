#!/usr/bin/env python3
"""Private-image acceptance on disposable GitHub runners; never a local installer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import urllib.request

from scripts import self_host


def command(*args):
    return subprocess.check_output(list(args), text=True, env=self_host.compose_environment())


def pull(role, manifest):
    reference = manifest['images'][role]['reference']
    assert '@sha256:' in reference
    with tempfile.TemporaryDirectory() as anonymous_config:
        anonymous = subprocess.run(['docker', '--config', anonymous_config, 'manifest', 'inspect', reference],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assert anonymous.returncode != 0, 'Private image unexpectedly accessible anonymously'
    assert subprocess.run(['docker', 'image', 'inspect', reference],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0, 'Image already cached'
    subprocess.run(['docker', 'pull', '--platform', 'linux/amd64', reference], check=True)
    image = json.loads(command('docker', 'image', 'inspect', reference))[0]
    assert image['Os'] == 'linux' and image['Architecture'] == 'amd64'
    source_sha = manifest['images'][role].get('source_sha', manifest['source_sha'])
    version = manifest['images'][role].get('version', manifest['version'])
    assert image['Config']['Labels']['org.opencontainers.image.revision'] == source_sha
    assert image['Config']['Labels']['org.opencontainers.image.version'] == version
    assert reference in image['RepoDigests']
    return {'reference': reference, 'image_id': image['Id'], 'source_sha': source_sha,
            'version': version, 'platform': 'linux/amd64', 'anonymous_manifest_denied': True}


def guided_start(config, project, passwords, transcript, fresh):
    import pexpect
    # Real terminal input drives the unmodified default entry point and hidden passwords.
    child = pexpect.spawn(sys.executable, ['scripts/start.py', '--config', str(config)],
                          encoding='utf-8', timeout=1200, env=dict(os.environ, PYTHONUNBUFFERED='1'))
    captured = []

    def expect(pattern):
        child.expect_exact(pattern)
        captured.append(child.before + child.after)

    try:
        if fresh:
            for prompt, answer in [('模式 core / gpu', ''), ('独立安装名称', project),
                                   ('网页端口', '18080'), ('监听 IPv4', ''),
                                   ('浏览器访问的 IPv4 或主机名', '')]:
                expect(prompt)
                child.sendline(answer)
            for username, password in passwords.items():
                for prompt in ['Password for new user ', 'Confirm password for ']:
                    expect(prompt + username + ': ')
                    child.sendline(password)
        child.expect(pexpect.EOF)
        captured.append(child.before)
        child.close()
        assert child.exitstatus == 0 and child.signalstatus is None, 'Guided startup failed'
        if not fresh:
            assert 'Password for new user' not in ''.join(captured), 'Repeat startup requested new passwords'
    finally:
        if child.isalive():
            captured.append(child.before or '')
            child.terminate(force=True)
        text = ''.join(captured)
        for password in passwords.values():
            text = text.replace(password, '[REDACTED]')
        transcript.write_text(text)


def http(base, path, token=None, payload=None):
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = 'Bearer ' + token
    request = urllib.request.Request(base + path,
        data=json.dumps(payload).encode() if payload is not None else None, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        assert response.status == 200
        return json.load(response)


def teaching_context(base, passwords):
    identities = {}
    for username, password in passwords.items():
        token = http(base, '/api/v1/auth/login', payload={'username': username, 'password': password})['access_token']
        me = http(base, '/api/v1/me', token)
        assert me['enabled'] and len(me['organizations']) == 1
        organization = me['organizations'][0]
        assert organization['role'] == username.removeprefix('studio-')
        courses = http(base, '/api/v1/courses?organization_id=' + organization['organization_id'], token)
        assert len(courses) == 1
        projects = http(base, '/api/v1/courses/' + courses[0]['course_id'] + '/projects', token)
        assert len(projects) == 7 and all(project['enabled'] for project in projects)
        identities[username] = {'user_id': me['user_id'], 'role': organization['role'],
            'organization_id': organization['organization_id'], 'course_id': courses[0]['course_id'],
            'project_ids': sorted(project['project_id'] for project in projects)}
    for path in ['/api/health', '/learn/1', '/learn/12']:
        with urllib.request.urlopen(base + path, timeout=30) as response:
            assert response.status == 200
    return identities


def database_state(config, values):
    sql = """SELECT json_build_object(
      'users', (SELECT json_agg(json_build_array(user_id,username,password_hash) ORDER BY username) FROM users),
      'projects', (SELECT json_agg(project_id ORDER BY project_id) FROM teaching_projects),
      'counts', json_build_object('users',(SELECT count(*) FROM users),
        'courses',(SELECT count(*) FROM courses),'projects',(SELECT count(*) FROM teaching_projects),
        'runs',(SELECT count(*) FROM runs)))::text;"""
    raw = subprocess.check_output(self_host.compose_command(config, values) +
        ['exec', '-T', 'postgres', 'psql', '-U', 'studio', '-d', 'studio', '-At', '-c', sql],
        text=True, env=self_host.compose_environment()).strip()
    counts = json.loads(raw)['counts']
    assert counts == {'users': 3, 'courses': 1, 'projects': 7, 'runs': 0}
    # Never store account password hashes in an artifact.
    return {'counts': counts, 'identity_and_password_fingerprint': hashlib.sha256(raw.encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['core', 'gpu-pull'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('GITHUB_ACTIONS') == 'true' and os.environ.get('RUNNER_ENVIRONMENT') == 'github-hosted', 'Only disposable hosted runners allowed'
    assert not command('docker', 'ps', '-aq').strip(), 'Runner already has containers'
    manifest = json.loads(Path('deploy/images.json').read_text())
    assert manifest['status'] == 'released' and manifest['visibility'] == 'private'
    args.output.mkdir(parents=True, exist_ok=False)
    result = {'mode': args.mode, 'configuration_sha': os.environ['GITHUB_SHA'],
              'source_sha': manifest['source_sha'], 'environment': 'fresh GitHub-hosted Ubuntu 22.04 linux/amd64 runner',
              'images': {}, 'worker_started': False, 'gpu_runtime_tested': False}
    for role in (['api', 'web'] if args.mode == 'core' else ['simulation', 'worker']):
        result['images'][role] = pull(role, manifest)
    if args.mode == 'gpu-pull':
        assert not command('docker', 'ps', '-aq').strip(), 'GPU pull must not create containers'
        result['status'] = 'PASS_PULL_ONLY'
    else:
        project = 'ci-a7-' + os.environ['GITHUB_RUN_ID']
        config = Path(os.environ['RUNNER_TEMP']) / 'a7-private.env'
        assert not config.exists()
        assert not command('docker', 'volume', 'ls', '-q', '--filter', 'label=com.docker.compose.project=' + project).strip()
        passwords = {'studio-' + role: secrets.token_urlsafe(32) for role in ['administrator', 'instructor', 'student']}
        try:
            guided_start(config, project, passwords, args.output / 'first-start.log', True)
            values = self_host.read_config(config)
            assert values['STUDIO_IMAGE_SOURCE'] == 'prebuilt'
            assert config.stat().st_mode & 0o777 == 0o600
            config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
            identities = teaching_context(values['STUDIO_WEB_ORIGIN'], passwords)
            state = database_state(config, values)
            volumes = command('docker', 'volume', 'ls', '-q', '--filter', 'label=com.docker.compose.project=' + project).splitlines()
            assert len(volumes) == 2
            guided_start(config, project, passwords, args.output / 'repeat-start.log', False)
            assert hashlib.sha256(config.read_bytes()).hexdigest() == config_hash
            assert teaching_context(values['STUDIO_WEB_ORIGIN'], passwords) == identities
            assert database_state(config, values) == state
            assert command('docker', 'volume', 'ls', '-q', '--filter', 'label=com.docker.compose.project=' + project).splitlines() == volumes
            result.update(status='PASS', new_configuration=True, new_volumes=volumes,
                          three_logins=True, course_and_seven_projects=True,
                          repeat_start_preserves_config_accounts_passwords_and_data=True,
                          runs=0, database=state, identities=identities)
        finally:
            if config.exists():
                subprocess.run([sys.executable, 'scripts/self_host.py', '--config', str(config), 'down'], check=True)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
