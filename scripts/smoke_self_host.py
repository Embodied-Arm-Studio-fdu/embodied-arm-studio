#!/usr/bin/env python3
"""CI-only Core acceptance: migrations, bootstrap, login and SPA routing."""
import argparse
import json
from pathlib import Path
import secrets
import subprocess
import urllib.request

from scripts.self_host import compose_command, compose_environment, read_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    values = read_config(args.config)
    if values['STUDIO_DEPLOY_MODE'] != 'core' or not values['STUDIO_PROJECT_NAME'].startswith('ci-') or values['STUDIO_BIND_ADDRESS'] != '127.0.0.1':
        raise SystemExit('Refusing acceptance writes outside an isolated ci-* Core stack')
    password = secrets.token_urlsafe(32)
    code = """import json,sys
from apps.api.database import create_postgres_engine,create_session_factory
from scripts.bootstrap_self_host import SelfHostBootstrap
secret=json.load(sys.stdin)['password']
engine=create_postgres_engine()
try:
    SelfHostBootstrap(create_session_factory(engine),'studio',password_reader=lambda prompt: secret).run(apply=True)
finally:
    engine.dispose()
"""
    subprocess.run(compose_command(args.config, values) + ['exec', '-T', 'api', 'python', '-c', code], input=json.dumps({'password': password}), text=True, check=True, env=compose_environment())
    base = 'http://127.0.0.1:' + values['STUDIO_HTTP_PORT']
    for route in ('/api/health', '/learn/1', '/learn/12'):
        with urllib.request.urlopen(base + route, timeout=15) as response:
            if response.status != 200:
                raise RuntimeError('Unexpected response during acceptance')
    payload = json.dumps({'username': 'studio-student', 'password': password}).encode()
    request = urllib.request.Request(base + '/api/v1/auth/login', data=payload, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=15) as response:
        token = json.load(response)['access_token']
    request = urllib.request.Request(base + '/api/v1/me', headers={'Authorization': 'Bearer ' + token})
    with urllib.request.urlopen(request, timeout=15) as response:
        context = json.load(response)
    if not context.get('organizations') or not context.get('enabled'):
        raise RuntimeError('New learner lacks its teaching context')
    print('Core acceptance passed: migrations, all-project bootstrap, login, organization and chapter routes.')


if __name__ == '__main__':
    main()
