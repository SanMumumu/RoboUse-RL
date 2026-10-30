"""Install into the selected Python environment; preview is the default."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, help='Optional JSON with extras')
    p.add_argument('--extras', choices=['base', 'test', 'cpu-test', 'train', 'transport'])
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    cfg = json.loads(a.config.read_text()) if a.config else {}
    extras = a.extras or cfg.get('extras', 'base')
    if extras not in ('base', 'test', 'cpu-test', 'train', 'transport'):
        p.error('Unknown install extras')
    root = Path(__file__).resolve().parents[1]
    target = str(root) + (f'[{extras}]' if extras != 'base' else '')
    command = [sys.executable, '-m', 'pip', 'install', '-e', target]
    print(json.dumps({'status': 'EXECUTE' if a.execute and not a.dry_run else 'DRY_RUN', 'command': command}))
    if a.execute and not a.dry_run:
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
