"""Publish the reviewed release branch and PR; never rewrite remote history."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import urllib.error
import urllib.request

REPO = 'SanMumumu/RoboUse-RL'
BRANCH = 'release/v0.1.0'


def credential():
    value = os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')
    if value:
        return value
    result = subprocess.run(['git', 'credential', 'fill'], input='protocol=https\nhost=github.com\n\n', text=True, capture_output=True, timeout=20, env={**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GIT_ASKPASS': '/bin/false'})
    fields = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    return fields.get('password')


def api(path, token, body=None):
    request = urllib.request.Request('https://api.github.com/' + path, data=json.dumps(body).encode() if body is not None else None, headers={'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json', 'Content-Type': 'application/json', 'User-Agent': 'RoboUse-release-v0.1.0'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Optional JSON with body_file')
    parser.add_argument('--body-file', type=Path, default=Path('docs/release_pr.md'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads(args.config.read_text()) if args.config else {}
    body_file = root / config.get('body_file', str(args.body_file))
    if args.dry_run or not args.execute:
        print(json.dumps({'status': 'DRY_RUN', 'repository': REPO, 'branch': BRANCH, 'base': 'main', 'body_file': str(body_file), 'force_push': False}))
        return
    token = credential()
    if not token:
        raise SystemExit('BLOCKED_AUTH: configure a repository-scoped GitHub token/HTTPS credential; no push or PR performed')
    try:
        user = api('user', token)
        if user.get('login') != 'SanMumumu':
            raise SystemExit('BLOCKED_ACCOUNT: authenticated account must be SanMumumu')
        repository = api('repos/' + REPO, token)
        if not repository.get('permissions', {}).get('push'):
            raise SystemExit('BLOCKED_PERMISSION: repository write permission unavailable')
        if subprocess.check_output(['git', 'branch', '--show-current'], cwd=root, text=True).strip() != BRANCH:
            raise SystemExit('BLOCKED_BRANCH: checkout the reviewed release/v0.1.0 branch')
        if subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip():
            raise SystemExit('BLOCKED_DIRTY: commit the reviewed release first')
        remote = subprocess.check_output(['git', 'remote', 'get-url', 'origin'], cwd=root, text=True).strip()
        if remote != 'https://github.com/' + REPO + '.git':
            raise SystemExit('BLOCKED_REMOTE: origin must match the reviewed HTTPS repository')
        # No credential is embedded in a URL, command, file, manifest or log.
        (root / 'agent/evidence').mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='release-auth-', dir=root / 'agent/evidence') as directory:
            helper = Path(directory) / 'askpass.py'
            helper.write_text('#!/usr/bin/env python3\nimport os,sys\nprint("x-access-token" if "username" in sys.argv[1].lower() else os.environ["ROBOUSE_RELEASE_TOKEN"])\n')
            helper.chmod(0o700)
            env = {**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GIT_ASKPASS': str(helper), 'ROBOUSE_RELEASE_TOKEN': token}
            subprocess.run(['git', '-c', 'credential.helper=', 'fetch', 'origin', 'main'], cwd=root, env=env, check=True)
            subprocess.run(['git', 'merge-base', '--is-ancestor', 'origin/main', 'HEAD'], cwd=root, check=True)
            subprocess.run(['git', '-c', 'credential.helper=', 'push', 'origin', 'HEAD:refs/heads/' + BRANCH], cwd=root, env=env, check=True)
        pulls = api('repos/' + REPO + '/pulls?state=open&head=SanMumumu:release%2Fv0.1.0&base=main', token)
        result = pulls[0] if pulls else api('repos/' + REPO + '/pulls', token, {'title': 'Release RoboUse-RL v0.1.0', 'head': BRANCH, 'base': 'main', 'body': body_file.read_text()})
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        print(json.dumps({'status': 'PUSHED_PR_CREATED', 'repository': repository['html_url'], 'commit': sha, 'pr_url': result['html_url']}))
    except urllib.error.HTTPError as exc:
        raise SystemExit(f'BLOCKED_AUTH_OR_API: HTTP {exc.code}; no credentials printed') from None


if __name__ == '__main__':
    main()
