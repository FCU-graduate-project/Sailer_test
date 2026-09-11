"""Read-only validation of existing derived manifests and checkpoint files."""
from pathlib import Path
from collections import Counter
import json
import csv
import hashlib
import soundfile as sf

BASE = Path('/home/brant/Project/gemini/dataset')
CRAB = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readcsv(path):
    with path.open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def readjl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main():
    original = {r['id']: r for r in readjl(BASE / 'interview_uncertain_500_v2/metadata.jsonl')}
    real = readcsv(CRAB / 'data/nnime_crab_format.csv')
    cols = list(real[0])
    keys = {r['FileName'] for r in real}

    def canon(rows):
        return sorted(tuple(r.get(k, '') for k in cols) for r in rows)

    out = {'datasets': {}, 'historical_files': {}}
    parent_sets = {}
    for name in ['interview_uncertain_500_v2_max12', 'interview_uncertain_nnime_u_match_v2']:
        root = BASE / name
        meta = readjl(root / 'metadata.jsonl')
        info, bad = {}, []
        by_path = {str(root / r['audio']): r for r in meta}
        for row in meta:
            path = root / row['audio']
            try:
                wav, sr = sf.read(path, dtype='float32')
                header = sf.info(path)
                info[str(path)] = {'seconds': len(wav) / sr, 'sr': sr,
                                  'channels': header.channels, 'sha256': sha(path)}
                if row.get('sha256') and row['sha256'] != info[str(path)]['sha256']:
                    bad.append(row['audio'])
            except Exception as exc:
                bad.append(str(path) + ':' + type(exc).__name__)
        dataset = {
            'metadata_rows': len(meta), 'decoded_files': len(info),
            'decode_or_hash_issues': bad,
            'over12_count': sum(r['seconds'] > 12 for r in info.values()),
            'min_seconds': min(r['seconds'] for r in info.values()),
            'max_seconds': max(r['seconds'] for r in info.values()),
            'review_rows': [
                {k: r.get(k) for k in ['id', 'file_name', 'parent_id', 'audio',
                                      'processing_action', 'repair_status', 'needs_review']}
                for r in meta if r.get('needs_review')],
            'arms': {}}
        out['datasets'][name] = dataset
        parent_sets[name] = {}
        for n in [25, 50, 75]:
            path = root / f'manifests/u{n}.csv'
            rows = readcsv(path)
            parents = [r['ParentID'] for r in rows]
            parent_sets[name][n] = set(parents)
            trainpath = root / f'training_csv/nnime_plus_u{n}.csv'
            merged = readcsv(trainpath)
            realrows = [r for r in merged if r['FileName'] in keys]
            ttsrows = [r for r in merged if r['FileName'] not in keys]
            dataset['arms'][str(n)] = {
                'rows': len(rows), 'unique_parents': len(set(parents)),
                'unique_audio_paths': len({r['FileName'] for r in rows}),
                'seconds': round(sum(info[r['FileName']]['seconds'] for r in rows), 6),
                'original_uncut_seconds_same_parents':
                    round(sum(original[x]['actual_seconds'] for x in parents), 6),
                'original_uncut_over12_count':
                    sum(original[x]['actual_seconds'] > 12 for x in parents),
                'prompt_counts': dict(Counter(r['Prompt'] for r in rows)),
                'original_split_counts': dict(Counter(original[x]['split'] for x in parents)),
                'selected_needs_review_paths': [
                    r['FileName'] for r in rows if by_path[r['FileName']].get('needs_review')],
                'all_onehot_U_train': all(
                    r['Split_Set'] == 'Train' and float(r['Unconfident']) == 1
                    and float(r['Excited']) == 0 and float(r['Neutral_3Class']) == 0 for r in rows),
                'duration_field_matches': all(
                    abs(float(r['DurationSeconds']) - info[r['FileName']]['seconds']) < .001 for r in rows),
                'all_audio_exists': all(Path(r['FileName']).is_file() for r in rows),
                'merged_counts': dict(Counter(r['Split_Set'] for r in merged)),
                'real_rows_unchanged': canon(realrows) == canon(real),
                'merged_tts_rows': len(ttsrows),
                'merged_tts_matches_manifest': canon(ttsrows) == canon(rows),
                'manifest_sha256': sha(path), 'training_csv_sha256': sha(trainpath)}
        dataset['nested_25_50_75'] = parent_sets[name][25] < parent_sets[name][50] < parent_sets[name][75]
    out['same_parents_between_versions'] = {
        str(n): parent_sets['interview_uncertain_500_v2_max12'][n]
        == parent_sets['interview_uncertain_nnime_u_match_v2'][n] for n in [25, 50, 75]}
    historical = [
        'bin/run_v1_warmstart_nnime_lora.sh', 'bin/train_crab_lora.py',
        'src/data/dataset/dataset.py', 'src/models/ser.py', 'data/nnime_crab_format.csv',
        'data/nnime_class_weights.json',
        'experiments/strategyA_xlsr_xlmr_lora/final_ser.pt',
        'experiments/strategyA_xlsr_xlmr_lora/text_lora_adapter/adapter_model.safetensors',
        'experiments/strategyA_xlsr_xlmr_lora/text_lora_adapter/adapter_config.json',
        'experiments/strategyA_xlsr_xlmr_lora/audio_lora_adapter/adapter_model.safetensors',
        'experiments/strategyA_xlsr_xlmr_lora/audio_lora_adapter/adapter_config.json',
        'experiments/nnime_v1_warmstart_lora/logging_20260707_164222.txt']
    for rel in historical:
        path = CRAB / rel
        out['historical_files'][rel] = {
            'exists': path.is_file(), 'sha256': sha(path) if path.is_file() else None,
            'bytes': path.stat().st_size if path.is_file() else None}
    out['audit_script_sha256'] = sha(Path(__file__))
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

