#!/usr/bin/env python3
"""Guided first install and daily startup; delegates operations to self_host.py."""
import argparse
import json
from pathlib import Path
import re
import socket
import subprocess
import sys

try:
    from scripts import self_host as host
except ModuleNotFoundError:
    import self_host as host


def ask(label, default=''):
    value = input(label + (' [' + default + ']' if default else '') + ': ').strip()
    return value or default


def image_settings(path, mode):
    manifest = json.loads(Path(path).read_text())
    if manifest.get('status') != 'released':
        raise ValueError('固定版本镜像尚未发布。请等待仓库和资源权利确认后的发布包；开发者可显式使用 --source build。')
    result = {}
    roles = ['api', 'web'] + (['simulation', 'worker'] if mode == 'gpu' else [])
    for role in roles:
        ref = manifest['images'][role]['reference']
        # Require immutable content identity; never silently use latest or a mutable tag.
        if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9._:/-]*@sha256:[a-f0-9]{64}', ref):
            raise ValueError('镜像清单必须包含固定 digest: ' + role)
        result['STUDIO_IMAGE_' + role.upper()] = ref
    return result


def ensure_available(values):
    """Read-only guards before a fresh install; Docker remains the final arbiter."""
    names = subprocess.check_output(['docker', 'ps', '-a', '--filter',
        'label=com.docker.compose.project=' + values['STUDIO_PROJECT_NAME'],
        '--format', '{{.ID}}'], text=True, env=host.compose_environment())
    if names.strip():
        raise ValueError('项目名已存在。请使用原私有配置启动，或为新安装选择另一个项目名。')
    ports = [(socket.SOCK_STREAM, int(values['STUDIO_HTTP_PORT']))]
    if values['STUDIO_DEPLOY_MODE'] == 'gpu':
        ports += [(socket.SOCK_STREAM, int(values['STUDIO_MEDIA_HTTP_PORT'])),
                  (socket.SOCK_STREAM, int(values['STUDIO_MEDIA_TCP_PORT'])),
                  (socket.SOCK_DGRAM, int(values['STUDIO_MEDIA_UDP_PORT']))]
    for kind, port in ports:
        with socket.socket(socket.AF_INET, kind) as probe:
            probe.bind((values['STUDIO_BIND_ADDRESS'], port))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=host.DEFAULT_CONFIG)
    parser.add_argument('--source', choices=('prebuilt', 'build'), default=None,
                        help='首次安装默认 prebuilt；build 为开发者源码构建选项')
    parser.add_argument('--image-manifest', type=Path, default=host.ROOT / 'deploy/images.json')
    parser.add_argument('--check', action='store_true', help='只检查已有配置与环境，不启动服务')
    args = parser.parse_args(argv)
    try:
        fresh = not (args.config.exists() or args.config.is_symlink())
        if args.check and fresh:
            raise ValueError('--check 需要已有私有配置；不会创建配置或启动服务。')
        if fresh:
            print('首次安装：Core 提供网页/课程/账号；GPU 另需 Linux NVIDIA、SDK 和模型。')
            mode = ask('模式 core / gpu', 'core').lower()
            if mode not in ('core', 'gpu'):
                raise ValueError('模式必须为 core 或 gpu')
            source = args.source or 'prebuilt'
            images = image_settings(args.image_manifest, mode) if source == 'prebuilt' else {}
            project = ask('独立安装名称', 'embodied-arm-selfhost')
            port = int(ask('网页端口', '8080'))
            bind = ask('监听 IPv4（本机访问保留默认）', '127.0.0.1')
            public = ask('浏览器访问的 IPv4 或主机名', 'localhost' if mode == 'core' else '')
            sdk = assets = ''
            media = [18889, 18189, 18191]
            ssh = False
            if mode == 'gpu':
                print('GPU SDK/模型须自行合法取得；本工具不安装驱动、Toolkit 或 SDK。')
                sdk = ask('Isaac Gym Preview 4 根目录')
                assets = ask('完整 S1 description 根目录')
                ssh = ask('通过 SSH 回环转发访问？ yes / no', 'no') == 'yes'
                media = [int(ask(label, str(default))) for label, default in zip(
                    ('媒体 HTTP 端口', 'ICE UDP 端口', 'ICE TCP 端口'), media)]
            # Check dependency availability before creating private settings.
            if host.shutil.which('docker') is None:
                raise ValueError('请先安装 Docker Engine 和 Compose v2.20+；工具不会修改系统。')
            identity = dict(STUDIO_PROJECT_NAME=project, STUDIO_HTTP_PORT=str(port),
                STUDIO_BIND_ADDRESS=bind, STUDIO_DEPLOY_MODE=mode,
                STUDIO_MEDIA_HTTP_PORT=str(media[0]), STUDIO_MEDIA_UDP_PORT=str(media[1]),
                STUDIO_MEDIA_TCP_PORT=str(media[2]))
            ensure_available(identity)
            values = host.initialize(args.config, mode, public, sdk, assets, bind, port,
                project, *media, allow_loopback_playback=ssh)
            with args.config.open('a') as target:
                for key, value in dict(STUDIO_IMAGE_SOURCE=source, **images).items():
                    target.write(key + "='" + value + "'\n")
            values = host.read_config(args.config)
        else:
            values = host.read_config(args.config)
            source = values.get('STUDIO_IMAGE_SOURCE', 'build')
            if args.source and args.source != source:
                raise ValueError('已有配置的镜像来源不会自动切换；请保留原安装配置。')
            print('复用现有配置、凭据和数据：' + str(args.config))
        if args.check:
            host.doctor(args.config, values)
            return 0
        # up includes doctor and waits for healthy services. Any failure stops here.
        if host.main(['--config', str(args.config), 'up']):
            return 1
        print('初始化账号：为新账号逐一输入并确认密码（无默认密码，不回显）。')
        if host.main(['--config', str(args.config), 'bootstrap']):
            return 1
        print('安装/启动完成，请打开 ' + values['STUDIO_WEB_ORIGIN'])
        print('账号：studio-administrator / studio-instructor / studio-student')
        return 0
    except (OSError, ValueError, KeyError, EOFError, KeyboardInterrupt, subprocess.CalledProcessError) as exc:
        detail = 'Docker 命令失败，请检查 Docker 服务和当前用户访问权限。' if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        print('启动停止：' + detail, file=sys.stderr)
        print('修正提示的问题后用相同命令重试；现有配置和数据会保留。', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
