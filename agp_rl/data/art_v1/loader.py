"""Lazy ART reader: seek PNG members in original tar; no second PNG tree."""
from collections import OrderedDict
from copy import deepcopy
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sqlite3

from PIL import Image


class ArtDataset:
    def __init__(self, root, source=None):
        self.root = Path(root).resolve()
        manifest = json.loads((self.root / 'dataset_manifest.json').read_text()) if (self.root / 'dataset_manifest.json').exists() else {}
        self.source = Path(source or manifest.get('source_root', self.root.parent / 'agp_official')).resolve()
        self.assets = sqlite3.connect(f'file:{self.root / "raw/assets.sqlite"}?mode=ro', uri=True)
        self.files = OrderedDict()
        self.current_id, self.current = None, None
        self._system_prompt = None

    def trajectory(self, episode_id):
        if self.current_id != episode_id:
            with gzip.open(self.root / 'canonical/trajectories' / (episode_id + '.json.gz'), 'rt') as f:
                self.current = json.load(f)
            self.current_id = episode_id
        return self.current

    def image(self, key):
        row = self.assets.execute('SELECT record FROM images WHERE key=?', (key,)).fetchone()
        if not row:
            raise FileNotFoundError('ART_IMAGE_MISSING:' + key)
        r = json.loads(row[0]); path = self.source / 'views' / r['shard']
        if path not in self.files:
            self.files[path] = path.open('rb')
            if len(self.files) > 4:
                self.files.popitem(last=False)[1].close()
        self.files.move_to_end(path)
        f = self.files[path]; f.seek(r['offset']); data = f.read(r['bytes'])
        if len(data) != r['bytes'] or hashlib.sha256(data).hexdigest() != r['sha256']:
            raise ValueError('ART_IMAGE_HASH_CHANGED:' + key)
        with Image.open(BytesIO(data)) as image:
            image.load(); result = image.convert('RGB')
        if (result.width, result.height) != (r['width'], r['height']):
            raise ValueError('ART_IMAGE_DIMENSIONS_CHANGED:' + key)
        return result

    def window(self, record, *, pixels=False):
        """Window accepted by the existing profile_window / encode_window path."""
        trajectory = self.trajectory(record['episode_id'])
        decision = trajectory['decisions'][record['timestep']]
        if decision['decision_id'] != record['decision_id']:
            raise ValueError('ART_DECISION_ID_MISMATCH')
        end = decision['visible_history']['message_end_exclusive']
        messages = deepcopy(trajectory['messages'][:end])
        messages.insert(0, {'role': 'system', 'content': [{'type':'text', 'text': self.system_prompt(record['episode_id'])}]})
        messages.append({'role':'assistant', 'content':[{'type':'text', 'text':decision['model_output']['text']}]})
        refs = {b['image']: trajectory['images'][b['image'][12:]] for m in messages for b in m['content'] if b['type']=='image'}
        images = [{'path':uri, **{k:r[k] for k in ('width','height','sha256','rgb_sha256')}} for uri,r in refs.items()]
        if pixels:
            decoded = {}
            for m in messages:
                for b in m['content']:
                    if b['type'] == 'image':
                        uri = b['image']
                        if uri not in decoded:
                            decoded[uri] = self.image(uri[12:])
                        b['image'] = decoded[uri]
        return {'window_id':decision['decision_id'], 'episode_id':record['episode_id'], 'messages':messages,
                'target_message_index':len(messages)-1, 'images':images, 'split':'train', 'fixture_only':False}

    def system_prompt(self, episode_id):
        if self._system_prompt is not None:
            return self._system_prompt
        from agp_rl.pipeline2b.dataset import student_system_prompt
        interface = json.loads((self.source / 'documents/interfaces/interface-76b013ef.json').read_text())
        self._system_prompt = student_system_prompt({t['name']:t['inputSchema'] for t in interface['tools']})
        return self._system_prompt

    def records(self, view='sft'):
        filename = {'sft':'sft_samples.jsonl', 'offline_rl':'offline_rl_transitions.jsonl.gz'}[view]
        path = self.root / 'training' / filename
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt') as f:
            for line in f:
                yield json.loads(line)

    def close(self):
        for f in self.files.values():
            f.close()
        self.files.clear(); self.assets.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
