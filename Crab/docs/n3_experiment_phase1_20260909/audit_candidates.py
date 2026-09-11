"""Read-only candidate audit. Prints JSON; never modifies source data or trains.

Run from any directory with Crab/.venv/bin/python; stdout can be inspected.
Low-energy fractions are simple amplitude diagnostics, NOT VAD or emotion labels.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import math
import re
import unicodedata

import numpy as np
import soundfile as sf

CRAB = Path(__file__).resolve().parents[2]
BASE = Path('/home/brant/Project/gemini/dataset')
SOURCE = BASE / 'interview_uncertain_500_v2'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold()
                   if c.isalnum())


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def counts(rows, key):
    return dict(sorted(Counter(str(r.get(key, '')) for r in rows).items()))


def quantiles(values):
    if not values:
        return None
    return {k: round(float(v), 6) for k, v in zip(
        ['min', 'p10', 'p25', 'median', 'p75', 'p90', 'p95', 'max'],
        np.quantile(values, [0, .1, .25, .5, .75, .9, .95, 1]))}


def main():
    report = {'audit_utc': datetime.now(timezone.utc).isoformat(),
              'source_root': str(SOURCE), 'read_only': True,
              'human_listening_performed': False, 'training_performed': False,
              'source_versions': [], 'inventory': []}
    for root in sorted(BASE.iterdir()):
        if not root.is_dir():
            continue
        meta = root / 'metadata.jsonl'
        rows = jsonl(meta) if meta.exists() else []
        summary = root / 'summary.json'
        entry = {'name': root.name,
                 'audio_subtrees': {str(p.relative_to(root)): len(list(p.rglob('*.wav')))
                                    for p in sorted(root.rglob('audio')) if p.is_dir()},
                 'metadata_rows': len(rows), 'split_counts': counts(rows, 'split'),
                 'processing_action_counts': counts(rows, 'processing_action'),
                 'needs_review_counts': counts(rows, 'needs_review')}
        if summary.exists():
            entry['declared_summary'] = json.loads(summary.read_text())
        report['source_versions'].append(entry)

    meta = jsonl(SOURCE / 'metadata.jsonl')
    source_hashes = {p.name: digest(p) for p in [SOURCE / 'metadata.jsonl',
                     SOURCE / 'input_manifest.jsonl', SOURCE / 'prompts.json',
                     SOURCE / 'summary.json'] if p.exists()}
    report['source_manifest_sha256'] = source_hashes
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-large',
                                                  local_files_only=True)
        report['tokenizer_audit'] = 'local cached XLM-R; special tokens included; no truncation'
    except Exception as exc:
        report['tokenizer_audit'] = 'unavailable: ' + type(exc).__name__

    hashes, pcm_hashes, texts = defaultdict(list), defaultdict(list), defaultdict(list)
    groups, split_sources = defaultdict(set), defaultdict(set)
    with (CRAB / 'data/nnime_crab_format.csv').open() as stream:
        real = list(csv.DictReader(stream))
    real_by_text = defaultdict(list)
    for row in real:
        real_by_text[normalize(row['Text'])].append((row['Split_Set'], row['FileName']))
    overlaps = []
    for m in meta:
        path = SOURCE / m['audio']
        item = {k: m.get(k, '') for k in ['id', 'prompt_id', 'voice', 'model', 'source_id',
                  'profession', 'source_file', 'source_turn', 'segment_index',
                  'text', 'spoken_text', 'transcript_verified', 'disfluency_expected']}
        item.update({'audio_path': str(path), 'source_split': m.get('split', ''),
                     'label_intended': 'Unconfident', 'label_verified': '',
                     'qc_status': 'pending_human_review', 'exists': path.is_file()})
        if not path.is_file():
            report['inventory'].append(item)
            continue
        try:
            info = sf.info(path)
            data, sr = sf.read(path, dtype='float64', always_2d=True)
            duration = len(data) / sr
            mono = data.mean(axis=1)
            step = max(1, int(.020 * sr))
            rms_frames = np.array([np.sqrt(np.mean(mono[i:i + step] ** 2))
                                   for i in range(0, len(mono), step)])
            low = rms_frames < .01  # -40 dBFS, fixed threshold, not calibrated VAD
            active = np.flatnonzero(~low)
            file_hash = digest(path)
            pcm_hash = hashlib.sha256(str((sr, data.shape)).encode()
                                     + data.astype('<f8').tobytes()).hexdigest()
            item.update({'decode_ok': True, 'sample_rate_hz': sr, 'channels': info.channels,
                         'subtype': info.subtype, 'duration_seconds': round(duration, 6),
                         'loader12_seconds': round(min(duration, 12), 6),
                         'over12_seconds': round(max(0, duration - 12), 6),
                         'sha256': file_hash, 'pcm_sha256': pcm_hash,
                         'metadata_hash_matches': file_hash == m.get('sha256'),
                         'metadata_duration_matches': abs(duration - m.get('actual_seconds', -999)) < .001,
                         'finite_pcm': bool(np.isfinite(data).all()),
                         'peak_dbfs': round(20 * math.log10(max(float(np.max(np.abs(data))), 1e-12)), 5),
                         'rms_dbfs': round(20 * math.log10(max(float(np.sqrt(np.mean(data ** 2))), 1e-12)), 5),
                         'near_full_scale_fraction': float(np.mean(np.abs(data) >= .999)),
                         'low_energy_frame_fraction_minus40dbfs': round(float(low.mean()), 6),
                         'leading_low_energy_seconds': round(float(active[0] * .02) if len(active) else duration, 6),
                         'trailing_low_energy_seconds': round(float((len(low) - 1 - active[-1]) * .02) if len(active) else duration, 6),
                         'text_character_count': len(item['text']),
                         'text_normalized_count': len(normalize(item['text'])),
                         'text_token_count': len(tokenizer(item['text'], truncation=False)['input_ids']) if tokenizer else None})
            hashes[file_hash].append(m['id'])
            pcm_hashes[pcm_hash].append(m['id'])
        except Exception as exc:
            item.update({'decode_ok': False, 'decode_error': type(exc).__name__ + ': ' + str(exc)})
        texts[normalize(m['text'])].append(m['id'])
        groups[m['source_id']].add(m.get('split', ''))
        split_sources[m.get('split', '')].add(m['source_id'])
        if normalize(m['text']) in real_by_text:
            overlaps.append({'id': m['id'], 'matches': real_by_text[normalize(m['text'])]})
        report['inventory'].append(item)

    good = [r for r in report['inventory'] if r.get('decode_ok')]
    durations = [r['duration_seconds'] for r in good]
    split_summary = {}
    for split in sorted({r['source_split'] for r in good}):
        subset = [r for r in good if r['source_split'] == split]
        split_summary[split] = {'count': len(subset),
            'seconds': round(sum(r['duration_seconds'] for r in subset), 6),
            'over12_count': sum(r['duration_seconds'] > 12 for r in subset),
            'prompt_counts': counts(subset, 'prompt_id')}
    report['summary'] = {
        'metadata_rows': len(meta), 'wave_files': len(list((SOURCE / 'audio').rglob('*.wav'))),
        'decode_ok': len(good), 'decode_failed_or_missing': len(meta) - len(good),
        'sample_rate_counts': counts(good, 'sample_rate_hz'), 'channel_counts': counts(good, 'channels'),
        'subtype_counts': counts(good, 'subtype'), 'prompt_counts': counts(good, 'prompt_id'),
        'voice_counts': counts(good, 'voice'), 'model_counts': counts(good, 'model'),
        'profession_count': len({r['profession'] for r in good}),
        'source_id_count': len(groups),
        'source_id_cross_split_count': sum(len(v) > 1 for v in groups.values()),
        'source_id_cross_split_details': {k: sorted(v) for k, v in groups.items() if len(v) > 1},
        'split_summary': split_summary,
        'total_seconds': round(sum(durations), 6),
        'loader12_total_seconds': round(sum(min(x, 12) for x in durations), 6),
        'mean_seconds': round(float(np.mean(durations)), 6), 'duration_quantiles': quantiles(durations),
        'duration_bins': {'under8': sum(d < 8 for d in durations),
                          '8_to_under10': sum(8 <= d < 10 for d in durations),
                          '10_to_12_inclusive': sum(10 <= d <= 12 for d in durations),
                          'over12_to_16': sum(12 < d <= 16 for d in durations)},
        'over12_count': sum(x > 12 for x in durations),
        'lost_seconds_at_12': round(sum(max(x - 12, 0) for x in durations), 6),
        'file_duplicate_groups': [v for v in hashes.values() if len(v) > 1],
        'pcm_duplicate_groups': [v for v in pcm_hashes.values() if len(v) > 1],
        'normalized_text_duplicate_groups': [v for v in texts.values() if len(v) > 1],
        'normalized_exact_nnime_text_overlap': overlaps,
        'metadata_hash_mismatch_ids': [r['id'] for r in good if not r['metadata_hash_matches']],
        'metadata_duration_mismatch_ids': [r['id'] for r in good if not r['metadata_duration_matches']],
        'transcript_verified_counts': counts(good, 'transcript_verified'),
        'spoken_text_nonempty_count': sum(bool(r['spoken_text']) for r in good),
        'disfluency_expected_counts': counts(good, 'disfluency_expected'),
        'text_token_quantiles': quantiles([r['text_token_count'] for r in good if r['text_token_count'] is not None]),
        'text_over128_count': sum((r['text_token_count'] or 0) > 128 for r in good),
        'rms_dbfs_quantiles': quantiles([r['rms_dbfs'] for r in good]),
        'low_energy_fraction_quantiles': quantiles([r['low_energy_frame_fraction_minus40dbfs'] for r in good]),
        'leading_low_energy_seconds_quantiles': quantiles([r['leading_low_energy_seconds'] for r in good]),
        'trailing_low_energy_seconds_quantiles': quantiles([r['trailing_low_energy_seconds'] for r in good]),
        'near_full_scale_file_count': sum(r['near_full_scale_fraction'] > 0 for r in good),
        'nonfinite_file_count': sum(not r['finite_pcm'] for r in good),
    }
    report['audit_script_sha256'] = digest(Path(__file__))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
