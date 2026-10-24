#!/usr/bin/env python3
"""CPU-only preparation; writes exclusively into a NEW child directory.

No torch, model loading, audio synthesis, downloads, inference, or queue writes.
Source datasets/checkpoints/specification are read-only. Generated manifests are
local-only and omit transcripts. NumPy sampling reproduces pandas sample's
RandomState(seed).choice(n, size, replace=False) used by the V1 builder.
"""
import argparse
import array
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import sys
import wave
from collections import Counter
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
CRAB = HERE.parent
REPO = CRAB.parent
LABELS = ['Excited', 'Unconfident', 'Neutral_3Class']
SEVEN = ['neutral', 'happy', 'sad', 'anger', 'surprise', 'fear', 'disgust']
MSP_MAP = dict(N='neutral', H='happy', S='sad', A='anger', U='surprise', F='fear', D='disgust')
REQUIRED_MODEL_FILES = ['final_ser.pt', 'train_norm_stat.pkl', 'mode.json',
    'audio_lora_adapter/adapter_config.json', 'audio_lora_adapter/adapter_model.safetensors',
    'text_lora_adapter/adapter_config.json', 'text_lora_adapter/adapter_model.safetensors']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        yield from csv.DictReader(f)


def write_json(path, obj):
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'Empty output: {path}')
    with path.open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def onehot(row):
    values = [float(row[k]) for k in LABELS]
    if values not in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
        raise ValueError('Non-one-hot label')
    return LABELS[values.index(1)]


def trailing_zero_count(samples):
    count = 0
    for value in reversed(samples):
        if value != 0:
            break
        count += 1
    return count


def quantile_linear(values, q):
    if not values or not 0 <= q <= 1:
        raise ValueError('Invalid quantile input')
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def eligible(nframes, sr, append_seconds):
    # No post-append truncation allowed. Exact comparison on source timebase.
    return nframes <= (12 - append_seconds) * sr


def prior_delta(msp, et, label):
    if any(v <= 0 for v in [msp[label], msp['neutral'], et[label], et['neutral']]):
        return None  # no invented pseudocount; explicitly report undefined
    return math.log(msp[label] / msp['neutral']) - math.log(et[label] / et['neutral'])


def literal_assignment(path, variable):
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == variable for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(variable)


def tails_and_test(out, sources):
    manifest = CRAB / 'experiments/n3_phase_a_long12_20260909/n3a75_train_dev_test.csv'
    sources.append(manifest)
    tts = [r for r in read_csv(manifest) if r.get('Source', '').startswith('TTS')]
    assert len(tts) == 471 and len({r['ParentID'] for r in tts}) == 471
    records = []
    for r in tts:
        assert r['Split_Set'] == 'Train' and onehot(r) == 'Unconfident'
        p = Path(r['FileName'])
        with wave.open(str(p), 'rb') as f:
            sr, channels, width, n = f.getframerate(), f.getnchannels(), f.getsampwidth(), f.getnframes()
            assert (sr, channels, width) == (24000, 1, 2)
            pcm = array.array('h', f.readframes(n))
        if sys.byteorder != 'little':
            pcm.byteswap()
        trailing = trailing_zero_count(pcm)
        records.append(dict(parent_id=r['ParentID'], audio_path=str(p), sample_rate=sr,
            frames=n, tail_zero_frames=trailing, tail_zero_ms=1000 * trailing / sr,
            duration_seconds=n / sr, sha256=sha(p)))
    write_csv(out / 'a75_tail_zero_per_file.csv', records)
    values = [r['tail_zero_ms'] for r in records]
    positive = [v for v in values if v > 0]
    summary = dict(n=471, definition='exact signed-PCM zero suffix; includes zero-length suffixes',
        quantile_method='linear interpolation, index=(n-1)*q', positive_count=len(positive),
        median_ms=quantile_linear(values, .5), p90_ms=quantile_linear(values, .9), max_ms=max(values),
        strict_greater_ms={str(t): sum(r['tail_zero_frames'] * 1000 > t * r['sample_rate'] for r in records)
                           for t in [10, 50, 100]},
        positive_only_median_ms=quantile_linear(positive, .5),
        positive_only_p90_ms=quantile_linear(positive, .9))
    # Fixed doses determined from Train-only TTS, not from model responses.
    summary['proposed_16k_append_samples'] = {
        'median': int(math.floor(summary['median_ms'] * 16 + .5)),
        'p90': int(math.floor(summary['p90_ms'] * 16 + .5)), 'sensitivity': 16000, 'deployment': 32000}
    p = CRAB / 'data/nnime_crab_format.csv'
    sources.append(p)
    tests = []
    for r in read_csv(p):
        if r['Split_Set'] != 'Test':
            continue
        audio = Path(r['FileName'])
        with wave.open(str(audio), 'rb') as f:
            sr, n = f.getframerate(), f.getnframes()
        parts = audio.stem.split('_')
        team = int(parts[1])
        assert team in {4, 20, 21, 22}
        session = REPO / 'datasets/NNIME/Session Level/Session Level' / ('_'.join(parts[:-1]) + '.wav')
        tests.append(dict(utterance_id=audio.stem, audio_path=str(audio), label=onehot(r),
            team=team, speaker_code=f'{team:02d}_{parts[2]}', frames=n, sample_rate=sr,
            duration_seconds=n / sr, eligible_le11s=eligible(n, sr, 1),
            eligible_common_le10s=eligible(n, sr, 2), session_audio=str(session),
            session_audio_exists=session.is_file(), noise_qc='NOT_SELECTED_NOT_REVIEWED',
            audio_sha256=sha(audio)))
    assert len(tests) == 638
    write_csv(out / 'nnime_test_eligibility.csv', tests)
    write_csv(out / 'noise_review_template.csv', [dict(utterance_id=r['utterance_id'],
        session_audio=r['session_audio'], noise_start_seconds='', noise_end_seconds='',
        speech_absent_verified='', listener_id='', notes='', approved='False')
        for r in tests if r['eligible_common_le10s']])
    summary['test_eligibility'] = {'total': len(tests),
        'le11s': sum(r['eligible_le11s'] for r in tests),
        'common_le10s': sum(r['eligible_common_le10s'] for r in tests),
        'common_class_counts': dict(Counter(r['label'] for r in tests if r['eligible_common_le10s'])),
        'common_team_counts': dict(Counter(r['team'] for r in tests if r['eligible_common_le10s'])),
        'missing_session_audio': sum(not r['session_audio_exists'] for r in tests)}
    write_json(out / 'tail_and_eligibility_summary.json', summary)
    return summary


def prior_counts(out, sources):
    import numpy as np
    v1 = CRAB / 'data/bilingual_strategyA.csv'
    msp_scheme = CRAB / 'data/msp2_interview_scheme1.csv'
    raw_msp = REPO / 'datasets/MSP_Podcast_Data/Labels/labels_consensus.csv'
    et_builder = CRAB / 'scripts/build_emotiontalk_crab_csv.py'
    sources.extend([v1, msp_scheme, raw_msp, et_builder, CRAB / 'scripts/build_bilingual_train_csv.py'])
    counts3 = {'EN': Counter(), 'ZH': Counter()}
    v1_msp_ids, v1_et_ids = [], set()
    for r in read_csv(v1):
        if r['Split_Set'] == 'Train':
            counts3[r['Language']][onehot(r)] += 1
            if r['Language'] == 'EN':
                v1_msp_ids.append(Path(r['FileName']).name)
            else:
                v1_et_ids.add(str(Path(r['FileName']).relative_to(REPO / 'datasets/emotiontalk/Audio16k')))
    assert [counts3['EN'][c] for c in LABELS] == [10934, 5183, 13883]
    assert [counts3['ZH'][c] for c in LABELS] == [2807, 1369, 7568]
    msp_three_pool = [r['FileName'] for r in read_csv(msp_scheme) if r['Split_Set'] == 'Train']
    indices = np.random.RandomState(42).choice(len(msp_three_pool), size=30000, replace=False)
    reproduced = [msp_three_pool[i] for i in indices]
    assert reproduced == v1_msp_ids, 'V1 exact sample order does NOT reproduce; stop'
    raw_counts, exclusions = Counter(), Counter()
    seven_pool, raw_five_order = [], []
    for r in read_csv(raw_msp):
        if r['Split_Set'] != 'Train':
            continue
        code = r['EmoClass']
        if code in MSP_MAP:
            label = MSP_MAP[code]
            raw_counts[label] += 1
            seven_pool.append((r['FileName'], label))
            if code not in {'A', 'D'}:
                raw_five_order.append(r['FileName'])
        else:
            exclusions[code] += 1
    assert raw_five_order == msp_three_pool, 'Raw label row order differs from V1 source'
    sampled = [seven_pool[i] for i in np.random.RandomState(42).choice(len(seven_pool), size=30000, replace=False)]
    sample_counts = Counter(label for _, label in sampled)
    write_csv(out / 'msp_seven_train_sample30k_seed42_CANDIDATE.csv',
              [dict(file_name=f, original_class=c, split='Train', status='COUNTERFACTUAL_NOT_TRAINED') for f, c in sampled])
    mapping = literal_assignment(et_builder, 'EMO_NORM')
    et_root = REPO / 'datasets/emotiontalk/Audio/Audio'
    et_counts, et_excluded, et_three_ids = Counter(), Counter(), set()
    et_inventory = []
    for path in sorted((et_root / 'json').rglob('*.json')):
        data = path.read_bytes()
        ann = json.loads(data)
        if not isinstance(ann, dict):
            raise ValueError(f'Unsupported ET record format: {path}')
        fp = ann.get('file_path') or ann.get('file_name') or ann.get('filename') or ann.get('wav') or ann.get('audio')
        group = re.search(r'[Gg](\d+)', str(fp))
        if not group:
            et_excluded['unresolved_group_all_splits'] += 1
            continue
        gid = int(group.group(1))
        if gid in {1, 12, 3, 15}:
            continue
        label = mapping.get(str(ann.get('emotion_result', '')).strip().lower())
        et_inventory.append(dict(annotation_path=str(path), annotation_sha256=hashlib.sha256(data).hexdigest(),
            audio_relative=str(fp), canonical_label=label or 'UNMAPPED', group=gid,
            original_audio_exists=(et_root / 'wav' / str(fp)).is_file()))
        if label not in SEVEN:
            et_excluded['unknown_label_train'] += 1
            continue
        if not (et_root / 'wav' / str(fp)).is_file():
            et_excluded['missing_original_audio_train'] += 1
            continue
        et_counts[label] += 1
        if label not in {'anger', 'disgust'}:
            et_three_ids.add(str(Path(fp).with_suffix('.wav')))
    assert et_three_ids == v1_et_ids, 'EmotionTalk 5-class pool differs from V1 Train'
    write_csv(out / 'emotiontalk_train_annotation_inventory.csv', et_inventory)
    tables = []
    for name, counts in [('MSP_full7_train', raw_counts), ('MSP_sample30k7_train_seed42_CANDIDATE', sample_counts), ('EmotionTalk_full7_train', et_counts)]:
        for label in SEVEN:
            tables.append(dict(pool=name, label=label, count=counts[label], total=sum(counts.values()),
                               proportion=counts[label] / sum(counts.values())))
    write_csv(out / 'seven_class_counts.csv', tables)
    differences = [dict(label=c, msp_full7_minus_et=prior_delta(raw_counts, et_counts, c),
        msp_sample30k7_minus_et=prior_delta(sample_counts, et_counts, c)) for c in SEVEN if c != 'neutral']
    write_csv(out / 'seven_class_logratio_differences.csv', differences)
    dif3 = {c: math.log(counts3['EN'][c] / counts3['EN']['Neutral_3Class']) -
                math.log(counts3['ZH'][c] / counts3['ZH']['Neutral_3Class']) for c in LABELS[:2]}
    assert round(dif3['Excited'], 2) == .75 and round(dif3['Unconfident'], 2) == .72
    summary = dict(v1_three_counts={k: dict(v) for k, v in counts3.items()}, three_logratio_differences=dif3,
        three_direction_degrees=math.degrees(math.atan2(dif3['Unconfident'], dif3['Excited'])),
        v1_sample_method='pandas sample without replacement; random_state=42; NOT stratified; after 3-class filtering',
        v1_msp_exact_sample_order_reproduced=True, v1_et_train_ids_reproduced=True,
        seven_sample_status='counterfactual new seven-class pool, same method/seed; NOT historical V1 data',
        numpy_version=np.__version__, msp_train_exclusions=dict(exclusions), emotiontalk_exclusions=dict(et_excluded),
        seven_counts={'MSP_full': dict(raw_counts), 'MSP_sample30k_candidate': dict(sample_counts), 'EmotionTalk': dict(et_counts)},
        seven_differences=differences, note='Prior contrast is a reference direction, not a mathematical upper bound on model response.')
    write_json(out / 'prior_summary.json', summary)
    return summary


def model_inventory(out, sources):
    rows = []
    for group, directory in [('A', 'n3_phase_a_multiseed_20260911'), ('B', 'n3_phase_b_matchmin1_s42_20260912')]:
        p = CRAB / 'experiments' / directory / 'campaign.json'
        sources.append(p)
        campaign = json.loads(p.read_text())
        cli = campaign['common_cli']
        assert cli['contrastive_weight'] == 2 and cli['epochs'] == 10 and cli['early_stop_patience'] == 0
        for arm in campaign['arms']:
            model = Path(arm['output_dir'])
            result_path = model / 'run_result.json'
            sources.append(result_path)
            result = json.loads(result_path.read_text())
            rows.append(dict(model_id=arm['experiment_id'], group=group, dose=arm['dose'], seed=arm['seed'],
                contrastive_weight=cli['contrastive_weight'], epochs=cli['epochs'], patience=cli['early_stop_patience'],
                checkpoint_policy='best_Dev', model_dir=str(model), training_wandb_url=result['wandb_url'],
                source_config=str(p), missing_files=';'.join(f for f in REQUIRED_MODEL_FILES if not (model / f).is_file()),
                ready_for_inference=False))
    n3 = CRAB / 'experiments/nnime_v1_warmstart_lora'
    sources.append(CRAB / 'bin/run_v1_warmstart_nnime_lora.sh')
    rows.append(dict(model_id='original_n3', group='N3', dose=0, seed=42, contrastive_weight=2,
        epochs=10, patience=5, checkpoint_policy='best_Dev', model_dir=str(n3),
        training_wandb_url='https://wandb.ai/d1249119-feng-chia-university/Crab_Bilingual_ZH/runs/bkndrhih',
        source_config=str(CRAB / 'bin/run_v1_warmstart_nnime_lora.sh'),
        missing_files=';'.join(f for f in REQUIRED_MODEL_FILES if not (n3 / f).is_file()), ready_for_inference=False))
    assert len(rows) == 13
    write_csv(out / 'model_inventory_DRAFT.csv', rows)
    return dict(n=len(rows), groups=dict(Counter(r['group'] for r in rows)),
        missing_model_files=[r['model_id'] for r in rows if r['missing_files']],
        weight_hash_verified=False, gpu_inference_approved=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    dest = args.out_dir.resolve()
    if dest.parent != HERE or not dest.name.startswith('cpu_'):
        parser.error('Output must be a new cpu_* immediate child of this preparation folder')
    if dest.exists():
        parser.error('Refusing to overwrite an existing output directory')
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[key] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    dest.mkdir()
    sources = [Path(__file__), CRAB / '入學前前測規格_給3090.md', CRAB / 'src/data/dataset/dataset.py']
    tails = tails_and_test(dest, sources)
    print('Tail distribution and NNIME eligibility complete', flush=True)
    priors = prior_counts(dest, sources)
    print('Three/seven-class priors complete', flush=True)
    models = model_inventory(dest, sources)
    write_json(dest / 'summary.json', dict(state='CPU_PREPARATION_ONLY_NOT_FROZEN_NOT_AUTHORIZED_FOR_GPU',
        created_utc=datetime.now(timezone.utc).isoformat(), tail=tails, priors=priors, models=models,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256={str(p.resolve()): sha(p) for p in sources},
        output_sha256={p.name: sha(p) for p in sorted(dest.iterdir()) if p.is_file()},
        no_predictions_generated=True, no_queue_job_submitted=True))
    print(json.dumps({'output': str(dest), 'tail_median_ms': tails['median_ms'], 'tail_p90_ms': tails['p90_ms'],
        'test_eligibility': tails['test_eligibility'], 'models': models,
        'three_logratios': priors['three_logratio_differences']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
