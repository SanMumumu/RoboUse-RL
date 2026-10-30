"""Read sealed teacher sources into the existing ART message/decision interface.

No collection code, source schema, transcript, image or source split is changed.
"""
import base64
from collections import Counter, defaultdict
from copy import deepcopy
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import sqlite3

from PIL import Image
from agp_rl.io import digest, read_json, sha256
from agp_rl.data.art_v1.canonical import visible_result, text_message
from agp_rl.data.art_v1.loader import ArtDataset
from agp_rl.data.teacher_dataset import runtime_environment_message, strict_json
from agp_rl.pipeline2b.dataset import student_system_prompt, visible_blocks, code_target
from agp_rl.adapters.tool_codec import ToolCodec

import os
ROOT = Path(os.environ.get('DATA_ROOT', '.')).resolve()
TRUNCATION = re.compile(r'\d+ (?:chars|characters|tokens) (?:truncated|omitted)|<image cut>|\[.{0,50}(?:truncated|omitted).{0,50}\]', re.I)


class IntegrityError(ValueError):
    pass


def require(value, reason):
    if not value:
        raise IntegrityError(reason)


class Verifier:
    def __init__(self):
        self.files = {}

    def check(self, ref, *, base=None):
        base = ROOT if base is None else base
        path = (Path(base) / ref['path']).resolve(strict=True)
        require(path.is_relative_to(ROOT), 'SOURCE_REFERENCE_OUTSIDE_PROJECT')
        stat = path.stat()
        if str(path) not in self.files:
            actual = sha256(path)
            require(actual == ref['sha256'], 'SOURCE_SHA256_MISMATCH:' + str(path))
            require((stat.st_size, stat.st_mtime_ns) == (path.stat().st_size, path.stat().st_mtime_ns), 'SOURCE_CHANGED_DURING_READ')
            self.files[str(path)] = {'sha256': actual, 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        require(self.files[str(path)]['sha256'] == ref['sha256'], 'CONFLICTING_SOURCE_HASH')
        return path


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def public_candidates(source, verifier):
    """Revalidate the 4,776 admitted ART windows, independent of later gaps."""
    import pyarrow.parquet as pq
    root, art = Path(source['root']), Path(source['art_root'])
    manifest = read_json(art / 'dataset_manifest.json')
    require(manifest['sft_subset_status'] == 'VALIDATED', 'PUBLIC_SAFE_WINDOW_VIEW_NOT_VALIDATED')
    trusted = defaultdict(dict)
    for row in jsonl(art / 'training/sft_samples.jsonl'):
        require(row['quality']['official_success'] and row['quality']['input_history_complete'], 'UNSAFE_PUBLIC_WINDOW_IN_ADMITTED_VIEW')
        trusted[row['episode_id']][row['decision_id']] = row
    require(sum(map(len, trusted.values())) == manifest['counts']['sft_samples'], 'PUBLIC_ADMITTED_WINDOW_COUNT_CHANGED')
    file_index = {r['path']: r for r in jsonl(art / 'raw/source_files.jsonl')}
    schemas = read_json(root / 'documents/interfaces/interface-76b013ef.json')['tools']
    codec = ToolCodec({t['name']: t['inputSchema'] for t in schemas})
    for name in ('documents/interfaces/interface-76b013ef.json', 'results/seed0/RELEASE_s0.csv', 'views/metadata.csv'):
        verifier.check(file_index[name], base=root)
    episodes = jsonl(art / 'canonical/episodes.jsonl'); parquet = {}
    with ArtDataset(art, root) as reader:
        for original_ep in episodes:
            ep = deepcopy(original_ep)
            admitted = trusted.get(ep['episode_id'], {})
            audit = {'source': source['name'], 'episode_id': ep['episode_id'], 'task': ep['task'],
                     'seed': ep['seed'], 'layout_id': ep['layout_id'], 'native_success': ep['official_success'],
                     'observable_decisions': ep['data_quality']['decisions'], 'reasons': []}
            if ep['official_success'] is not True or not admitted:
                audit['reasons'] = ['NON_SUCCESS_OR_NO_ADMITTED_LOCAL_WINDOW']
                yield audit, None; continue
            try:
                provenance = ep['provenance']
                path = verifier.check(file_index[provenance['parquet']], base=root)
                if str(path) not in parquet: parquet[str(path)] = pq.ParquetFile(path)
                pf = parquet[str(path)]
                names = [n for n in pf.schema_arrow.names if n not in ('frame_first', 'frame_last', 'video_preview')]
                original = pf.read_row_group(provenance['row_group'], columns=names).to_pylist()[0]
                require(original['episode_id'] == ep['episode_id'] and original['outcome'] == 'success', 'PUBLIC_NATIVE_RESULT_MISMATCH')
                require((original['seed'], original['layout_id'], original['task'], original['base_task']) ==
                    (ep['seed'], ep['layout_id'], ep['task_config'], ep['task']), 'PUBLIC_IDENTITY_CHANGED')
                require(original['score'] == ep['official_score'], 'PUBLIC_NATIVE_SCORE_CHANGED')
                trace_path = verifier.check(file_index[provenance['trace']], base=root)
                with gzip.open(trace_path, 'rb') as stream: lines = list(stream)
                events = {i: json.loads(line) for i, line in enumerate(lines, 1)}
                bundle = deepcopy(reader.trajectory(ep['episode_id']))
                selected = [d for d in bundle['decisions'] if d['decision_id'] in admitted]
                require(len(selected) == len(admitted), 'PUBLIC_WINDOW_ID_NOT_IN_ART_TRAJECTORY')
                launch = text_message('user', original['prompt_text'])
                launch['content'].extend({'type': 'image', 'image': 'art-image://' + key}
                                         for key in original['traces'][0]['image_keys'])
                require(bundle['messages'][0] == launch, 'PUBLIC_LAUNCH_PARQUET_MISMATCH')
                # The admitted prefix ends before its own target. Later broken
                # replies are retained for audit but never tested as its input.
                last_end = max(d['visible_history']['message_end_exclusive'] for d in selected)
                for message, origin in zip(bundle['messages'][1:last_end + 1], bundle['message_origins'][1:last_end + 1]):
                    line = lines[origin['line'] - 1]
                    require(hashlib.sha256(line).hexdigest() == origin['line_sha256'], 'PUBLIC_TRACE_LINE_MISMATCH')
                    item = events[origin['line']]['item']
                    if message['role'] == 'user':
                        blocks, issues = visible_result(item.get('result'))
                        require(not issues and message['content'] == blocks, 'PUBLIC_PREFIX_NOT_FULL_RAW_TOOL_REPLY')
                    else:
                        target = item['text'] if item['type'] == 'agent_message' else json.dumps(
                            {'name': item['tool'], 'arguments': item['arguments']}, ensure_ascii=False, separators=(',', ':'))
                        require(message == text_message('assistant', target), 'PUBLIC_ASSISTANT_NOT_RAW_TRACE')
                needed_images = set()
                for d in bundle['decisions']:
                    d['data_quality']['sft_candidate'] = d['decision_id'] in admitted
                    if not d['data_quality']['sft_candidate']: continue
                    row = admitted[d['decision_id']]; origin = d['provenance']
                    require(row['target_assistant_output'] == d['model_output']['text'], 'PUBLIC_TRUSTED_TARGET_CHANGED')
                    require(origin['line_sha256'] == row['source']['provenance']['line_sha256'] ==
                        hashlib.sha256(lines[origin['line'] - 1]).hexdigest(), 'PUBLIC_TARGET_LINE_CHANGED')
                    end = d['visible_history']['message_end_exclusive']
                    require(end == row['prompt']['message_end_exclusive'], 'PUBLIC_TARGET_BOUNDARY_CHANGED')
                    for m in bundle['messages'][:end]:
                        needed_images.update(b['image'][12:] for b in m['content'] if b['type'] == 'image')
                images = {}
                for key in needed_images:
                    row = reader.assets.execute('SELECT record FROM images WHERE key=?', (key,)).fetchone()
                    require(row is not None, 'PUBLIC_WDS_IMAGE_NOT_INDEXED')
                    record = json.loads(row[0])
                    images[key] = {**record, 'storage': 'tar', 'tar_path': str(root / 'views' / record['shard'])}
                ep['episode_history_complete'] = not ep['data_quality']['final_history_issues']
                ep['supervision_scope'] = 'complete_trajectory' if ep['episode_history_complete'] and len(selected) == len(bundle['decisions']) else 'qualified_local_decisions'
                bundle['raw_observable_decision_count'] = len(bundle['decisions'])
                bundle['unadmitted_future_suffix_decisions'] = len(bundle['decisions']) - selected[-1]['timestep'] - 1
                bundle['messages'] = bundle['messages'][:last_end + 1]
                bundle['message_origins'] = bundle['message_origins'][:last_end + 1]
                bundle['decisions'] = bundle['decisions'][:selected[-1]['timestep'] + 1]
                bundle['images'] = {key: bundle['images'][key] for key in needed_images}
                yield audit, {'episode': ep, 'bundle': bundle, 'images': images,
                    'system_prompt': reader.system_prompt(ep['episode_id']), 'schemas': codec.schemas, 'source_kind': 'public_art'}
            except (ValueError, KeyError, OSError, AssertionError) as exc:
                audit['reasons'] = ['PUBLIC_WINDOW_INTEGRITY_REJECTED:' + str(exc)]
                yield audit, None


def convert_native(ep_id, attempt, index, verifier, schemas):
    """Adapt the original response_item stream; outer exec stays one decision."""
    terminal = read_json(attempt / 'terminal.json')
    require(terminal['valid_score'] is True and terminal['native_success'] is True, 'NATIVE_POSITIVE_EPISODE_REQUIRED')
    require(index['identity']['seed'] == terminal['seed'], 'NATIVE_SEED_IDENTITY_MISMATCH')
    artifacts_path = verifier.check(index['artifacts'])
    artifacts = jsonl(artifacts_path)
    bound = {}; source_aliases = {}
    for ref in artifacts:
        path = verifier.check(ref)
        bound[str(path)] = ref
        if ref.get('original_path') and path.suffix == '.png':
            original = str(Path(ref['original_path']).resolve())
            bound[original] = ref; source_aliases[original] = path
    for name in ('native_result', 'native_metrics'):
        verifier.check(terminal[name])
    from agp_rl.reward.official import read_official
    native = read_official(ROOT / terminal['native_result']['path'], terminal['layout_id'])
    require(native.success and native.score == terminal['native_score'], 'NATIVE_SCORE_EVIDENCE_CHANGED')
    traces = list((attempt / 'codex_transcripts').glob('*.jsonl'))
    require(len(traces) == 1 and str(traces[0].resolve()) in bound, 'ONE_SEALED_NATIVE_TRANSCRIPT_REQUIRED')
    trace = traces[0]; trace_hash = bound[str(trace.resolve())]['sha256']
    raw = trace.read_bytes(); require(raw.endswith(b'\n'), 'INCOMPLETE_NATIVE_TRANSCRIPT')
    image_paths = {ref['sha256']: source_aliases.get(path, Path(path)) for path, ref in bound.items() if Path(path).suffix == '.png'}
    images = {}; messages = []; origins = []; decisions = []; pending = None; results = {}; seen_calls = set(); history_issues = []
    codec = ToolCodec(schemas)

    def convert(value, provenance):
        if isinstance(value, list):
            return [convert(v, provenance) for v in value]
        if not isinstance(value, dict):
            return value
        if value.get('type') in ('input_image', 'image', 'imagecontent'):
            url = value.get('image_url', value.get('url'))
            if isinstance(url, dict):
                url = url.get('url')
            if isinstance(url, str) and url.startswith('data:image/'):
                payload = base64.b64decode(url.split(',', 1)[1], validate=True)
            elif value.get('data'):
                payload = base64.b64decode(value['data'], validate=True)
            else:
                path = Path(value.get('path') or url)
                if not path.is_absolute():
                    path = Path(terminal['session']) / path
                require(str(path.resolve()) in bound, 'NATIVE_IMAGE_NOT_SEALED')
                payload = source_aliases.get(str(path.resolve()), path).read_bytes()
            h = hashlib.sha256(payload).hexdigest()
            require(h in image_paths, 'MODEL_VISIBLE_IMAGE_NOT_IN_CAPTURE')
            with Image.open(BytesIO(payload)) as im:
                im.load(); rgb = im.convert('RGB')
            images[h] = {'key': h, 'storage': 'png', 'path': str(image_paths[h]), 'sha256': h,
                'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(), 'width': rgb.width, 'height': rgb.height,
                'bytes': len(payload), 'decode': True, 'source': provenance}
            return {'type': 'image', 'image': 'art-image://' + h}
        return {k: convert(v, provenance) for k, v in value.items()}

    for number, line in enumerate(raw.splitlines(keepends=True), 1):
        try:
            record = strict_json(line); item = record.get('payload', {})
            require(record.get('type') != 'compacted' and item.get('type') not in ('compaction', 'compacted', 'context_compaction'), 'COMPACTED_HISTORY_UNRECONSTRUCTIBLE')
            if record.get('type') != 'response_item':
                continue
            kind = item.get('type')
            if kind == 'reasoning' or (kind == 'message' and (item.get('role') in ('system', 'developer') or runtime_environment_message(item))):
                continue
            provenance = {'trace': str(trace), 'trace_sha256': trace_hash, 'line': number,
                          'line_sha256': hashlib.sha256(line.rstrip(b'\n')).hexdigest()}
            converted = convert(item, provenance)
            if kind in ('function_call_output', 'custom_tool_call_output'):
                require(pending == item.get('call_id'), 'TOOL_OUTPUT_ORDER_OR_CALL_ID_MISMATCH')
                blocks = visible_blocks(converted['output'])
                if any(TRUNCATION.search(b['text']) for b in blocks if b['type'] == 'text'):
                    history_issues.append('MODEL_VISIBLE_TRUNCATED_TOOL_RETURN')
                messages.append({'role': 'user', 'content': blocks}); origins.append(provenance)
                results[pending] = {'content': blocks, 'provenance': provenance}
                pending = None
                continue
            require(pending is None, 'MODEL_OUTPUT_DURING_PENDING_CALL')
            call_id = None; arguments = None
            if kind == 'message':
                blocks = visible_blocks(converted['content'])
                if item['role'] == 'user':
                    require(not messages, 'ADDITIONAL_TEACHER_USER_MESSAGE')
                    messages.append({'role': 'user', 'content': blocks}); origins.append(provenance)
                    continue
                require(item['role'] == 'assistant' and all(b['type'] == 'text' for b in blocks), 'INVALID_ASSISTANT_TARGET')
                target = ''.join(b['text'] for b in blocks)
            elif kind in ('function_call', 'custom_tool_call'):
                call_id = item['call_id']; require(call_id not in seen_calls, 'DUPLICATE_NATIVE_CALL_ID')
                seen_calls.add(call_id); pending = call_id
                if kind == 'custom_tool_call':
                    target = code_target(item); arguments = {'code': item['input']}
                else:
                    require(item.get('namespace') in (None, '', 'mcp__robot'), 'UNKNOWN_TOOL_NAMESPACE')
                    name = item['name']
                    for prefix in ('mcp__robot__', 'robot.'):
                        if name.startswith(prefix): name = name[len(prefix):]
                    arguments = strict_json(item['arguments']) if isinstance(item['arguments'], str) else item['arguments']
                    target = codec.encode(name, arguments)
            else:
                raise IntegrityError('UNSUPPORTED_NATIVE_RESPONSE_ITEM:' + str(kind))
            require(bool(messages) and any(b['type'] == 'image' for m in messages for b in m['content']), 'REAL_INITIAL_RGB_REQUIRED')
            decision = {'episode_id': ep_id, 'decision_id': f'{ep_id}:d{len(decisions):04d}', 'timestep': len(decisions),
                'kind': 'agent_direct_tool_call' if call_id else 'agent_message',
                'visible_history': {'message_end_exclusive': len(messages)}, 'model_output': {'text': target},
                'tool_calls': [{'call_id': call_id}] if call_id else [], 'tool_arguments': arguments,
                'provenance': provenance, 'data_quality': {'input_history_complete': not history_issues, 'sft_candidate': not history_issues, 'exclusion_reasons': list(history_issues)}}
            decisions.append(decision); messages.append(text_message('assistant', target)); origins.append(provenance)
        except (ValueError, KeyError, OSError, AssertionError) as exc:
            history_issues.append('UNRECONSTRUCTIBLE_SUFFIX:' + str(exc))
            break
    metrics = read_json(ROOT / terminal['native_metrics']['path'])
    if pending:
        qualified_terminal = bool(decisions and decisions[-1]['tool_calls'] and
            decisions[-1]['tool_calls'][0]['call_id'] == pending and
            metrics.get('official_ended_before_finalize') is True and metrics.get('counts_match') is True and
            metrics.get('marked_failed') is False and not metrics.get('infrastructure_failure') and not history_issues)
        if not qualified_terminal:
            history_issues.append('UNQUALIFIED_PENDING_SUFFIX')
            for decision in decisions:
                if decision['tool_calls'] and decision['tool_calls'][0]['call_id'] == pending:
                    decision['data_quality'].update(sft_candidate=False, exclusion_reasons=['UNQUALIFIED_PENDING_SUFFIX'])
    for decision in decisions:
        call = (decision['tool_calls'] or [{}])[0].get('call_id')
        decision['tool_results'] = results.get(call)
        if call in results:
            from robouse.feedback import feedback_flags
            # Use the same native feedback classifier as current decision SFT.
            flags = feedback_flags(results[call]['content'])
            if flags:
                decision['data_quality'].update(sft_candidate=False, exclusion_reasons=['ERROR_ACTION_HISTORY_ONLY'])
    identity = index['identity']
    episode = {'episode_id': ep_id, 'task': identity['base_task'], 'task_config': identity['task'],
        'seed': terminal['seed'], 'layout_id': terminal['layout_id'], 'layout_hash': terminal['layout_sha256'],
        'official_success': True, 'official_score': native.score, 'source_id': index.get('source_id','local:agp_astra_baseline_v1'),
        'teacher_model': index.get('teacher_model'), 'teacher_effort': index.get('teacher_effort'),
        'role': index['role'], 'episode_history_complete': not history_issues,
        'supervision_scope': 'complete_trajectory' if not history_issues and all(d['data_quality']['sft_candidate'] for d in decisions) else 'qualified_local_decisions',
        'data_quality': {'scored': True, 'final_history_issues': history_issues, 'decisions': len(decisions)},
        'provenance': {'terminal': {'path': str(attempt / 'terminal.json'), 'sha256': sha256(attempt / 'terminal.json')},
                       'trajectory_index_sha256': sha256(attempt / 'trajectory-index.json')}}
    bundle = {'episode_id': ep_id, 'messages': messages, 'message_origins': origins, 'decisions': decisions, 'images': images}
    return {'episode': episode, 'bundle': bundle, 'images': images, 'system_prompt': student_system_prompt(schemas),
            'schemas': schemas, 'source_kind': 'native'}


def native_candidates(source, verifier, schemas, frozen_records):
    root = Path(source['root'])
    for ref in frozen_records:
        audit = {'source': source['name'], 'episode_id': ref['episode_id'], 'reasons': [], 'native_success': None}
        if not ref.get('valid_score'):
            audit['reasons'] = ['UNSEALED_OR_NATIVE_INFRASTRUCTURE_UNSCORED']
            yield audit, None; continue
        try:
            path = verifier.check(ref)
            require(path.is_relative_to(root / 'episodes'), 'SEALED_EPISODE_OUTSIDE_SOURCE')
            index = read_json(path); terminal = index['terminal']
            require(terminal['episode_id'] == ref['episode_id'] and terminal['seed'] == source['seed'], 'NATIVE_SOURCE_IDENTITY_MISMATCH')
            audit.update(task=index['identity']['base_task'], seed=terminal['seed'], layout_id=terminal['layout_id'], native_success=terminal['native_success'])
            if not terminal['native_success']:
                audit['reasons'] = ['NATIVE_FAILURE_RETAINED_FOR_FUTURE_RECOVERY']
                yield audit, None; continue
            candidate = convert_native(ref['episode_id'], path.parent, index, verifier, schemas)
            audit['observable_decisions'] = len(candidate['bundle']['decisions'])
            yield audit, candidate
        except (ValueError, KeyError, OSError, AssertionError) as exc:
            audit['reasons'] = ['NATIVE_INTEGRITY_OR_HISTORY_REJECTED:' + str(exc)]
            yield audit, None
