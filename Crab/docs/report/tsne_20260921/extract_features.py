"""Queue-only, read-only inference for report figures; never train or update W&B."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
CRAB = HERE.parents[2]
sys.path.insert(0, str(CRAB / 'bin'))
sys.path.insert(0, str(CRAB))
CLASSES = ['Excited', 'Unconfident', 'Neutral_3Class']
HISTORY = CRAB / 'experiments/n3_phase_a_test_20260910'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def read_rows():
    with (CRAB / 'data/nnime_crab_format.csv').open() as f:
        return list(csv.DictReader(f))


def prepare():
    import pandas as pd
    import importlib.util
    from collections import Counter
    spec = importlib.util.spec_from_file_location('nnime_map', CRAB / 'scripts/build_nnime_crab_csv.py')
    mapping = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mapping)
    annotations = pd.read_excel(mapping.ANNOT_XLSX).set_index('Name')
    rows = read_rows()
    train = [r for r in rows if r['Split_Set'] == 'Train']
    test = [r for r in rows if r['Split_Set'] == 'Test']
    assert len(train) == 3054 and len(test) == 638
    metadata = []
    for row in train:
        name = Path(row['FileName']).stem
        raw = annotations.loc[name]
        raters = [raw[k] for k in ['E1','E2','E3','E4','E5','E6']]
        atoms = Counter(a.strip() for r in raters if not pd.isna(r)
                        for a in str(r).split(':') if a.strip())
        top = sorted(a for a, n in atoms.items() if n == max(atoms.values()))
        label = CLASSES.index(next(c for c in CLASSES if float(row[c]) == 1))
        assert mapping.classify_row(raters, Counter()) == 'EUN'[label]
        metadata.append({'id': name, 'label': label, 'original_top_labels': top,
                         'original_display': top[0] if len(top) == 1 else '最高票並列',
                         'annotation_votes': dict(atoms)})
    history = json.loads((HISTORY / 'evaluation.json').read_text())
    paths = dict(history['frozen_sha256'])
    paths.update(history['test_audio_sha256'])
    # Require historical source/checkpoint compatibility, not just matching names.
    for p, expected in paths.items():
        if digest(p) != expected:
            raise RuntimeError('Historical input changed: ' + p)
    for r in train:
        paths[r['FileName']] = digest(r['FileName'])
    for p in [Path(__file__), mapping.ANNOT_XLSX, CRAB / 'scripts/build_nnime_crab_csv.py']:
        paths[str(p)] = digest(p)
    for arm in history['arms']:
        for name in ['test_result.json', 'predictions.csv']:
            p = HISTORY / arm['id'] / name
            paths[str(p)] = digest(p)
    config = {'created_unix': time.time(), 'arms': history['arms'], 'sha256': paths,
              'training': False, 'new_seed_training': False, 'wandb_writes': False,
              'test_rows': 638, 'mapping_train_rows': 3054,
              'model_features': 'classifier_relu output, 512 dimensions, before final classifier',
              'mapping_features': 'frozen pretrained XLS-R-300m, no LoRA, mean of last hidden frames',
              'normalization': 'saved per-checkpoint Train statistics; mapping uses N3 Train statistics',
              'precision': 'FP32', 'batch_size': 1, 'workers': 0,
              'audio_max_seconds': 12, 'text_max_length': 128,
              'comparison_scope': 'historical patience5 A only; not full10 or B',
              'test_regression': 'exact predicted labels; all probabilities atol=1e-5, rtol=0'}
    if (HERE / 'manifest.json').exists():
        raise RuntimeError('Refusing to replace prepared manifest')
    dump(HERE / 'mapping_labels.json', metadata)
    config['sha256'][str(HERE / 'mapping_labels.json')] = digest(HERE / 'mapping_labels.json')
    dump(HERE / 'manifest.json', config)
    print('Prepared fixed 3054 Train mapping / 638 Test comparison; 4 historical checkpoints.')
    print('Original top-label counts:', dict(Counter(r['original_display'] for r in metadata)))


def verify(config):
    for path, expected in config['sha256'].items():
        if digest(path) != expected:
            raise RuntimeError('Frozen input changed: ' + path)


def worker(index, out, config):
    from crab_epoch_checkpoint import memory_guard
    memory_guard(start=True)
    verify(config)
    import random
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    from peft import PeftModel
    from src.models import MultiModalEmotionClassifierDeep
    from src.data.dataset.dataset import LazyWavSet, TxtSet, load_norm_stat
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.cuda.manual_seed_all(42); torch.set_num_threads(2)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one queue-assigned CUDA GPU required')
    arm = config['arms'][index]
    model = Path(arm['model_dir'])
    mode = json.loads((model / 'mode.json').read_text())
    assert mode['classes'] == CLASSES and mode['ft_mode'] == 'lora'
    mean, std = load_norm_stat(model / 'train_norm_stat.pkl')
    rows = read_rows()
    ssl = AutoModel.from_pretrained(mode['ssl_type'], local_files_only=True).to('cuda').eval()
    base_commit = getattr(ssl.config, '_commit_hash', None)

    def wave_tensor(wavs, i):
        wav, dur = wavs[i]
        if dur < 400:
            raise RuntimeError('Audio too short')
        audio = torch.zeros((1, dur), dtype=torch.float32)
        audio[0, :dur] = torch.tensor(wav[:dur])
        return audio.to('cuda')

    if index == 0:
        train = [r for r in rows if r['Split_Set'] == 'Train']
        wavs = LazyWavSet([r['FileName'] for r in train], wav_mean=mean, wav_std=std)
        feats = []
        with torch.no_grad():
            for i in range(len(train)):
                audio = wave_tensor(wavs, i)
                features = ssl(audio, attention_mask=torch.ones_like(audio)).last_hidden_state
                feats.append(features.mean(dim=1).float().cpu().numpy()[0])
                if (i+1) % 100 == 0 or i+1 == len(train):
                    memory_guard()
                    print(f'MAPPING pretrained audio {i+1}/{len(train)}', flush=True)
        arr = np.stack(feats)
        assert arr.shape == (3054, ssl.config.hidden_size) and np.isfinite(arr).all()
        np.savez_compressed(out / 'mapping_features.npz', features=arr,
                            ids=np.array([Path(r['FileName']).stem for r in train]))
        del feats, arr, features, audio

    ssl = PeftModel.from_pretrained(ssl, model / 'audio_lora_adapter', is_trainable=False).eval()
    text = PeftModel.from_pretrained(
        AutoModel.from_pretrained(mode['text_model_path'], local_files_only=True),
        model / 'text_lora_adapter', is_trainable=False).to('cuda').eval()
    head = MultiModalEmotionClassifierDeep(features1_dim=ssl.config.hidden_size,
        features2_dim=text.config.hidden_size, fusion_hidden_dim=mode['fusion_hidden_dim'],
        num_emotions=3, dropout=0.5).to('cuda')
    head.load_state_dict(torch.load(model / 'final_ser.pt', map_location='cpu', weights_only=True))
    head.eval()
    tokenizer = AutoTokenizer.from_pretrained(mode['text_model_path'], local_files_only=True)
    test = [r for r in rows if r['Split_Set'] == 'Test']
    wavs = LazyWavSet([r['FileName'] for r in test], wav_mean=mean, wav_std=std)
    texts = TxtSet([r['Text'] or '' for r in test], tokenizer, max_len=128)
    capture = []
    handle = head.classifier_relu.register_forward_hook(
        lambda module, inputs, output: capture.append(output.detach().float().cpu().numpy()[0]))
    probs = []
    with torch.no_grad():
        for i in range(len(test)):
            audio = wave_tensor(wavs, i)
            af = ssl(audio, attention_mask=torch.ones_like(audio)).last_hidden_state
            ids, mask = texts[i]
            tf = text(input_ids=ids.unsqueeze(0).to('cuda'),
                      attention_mask=mask.unsqueeze(0).to('cuda')).last_hidden_state
            logits = head(af, tf).float()
            probs.append(logits.softmax(1).cpu().numpy()[0])
            if (i+1) % 100 == 0 or i+1 == len(test):
                memory_guard()
                print(f'{arm["id"]} Test features {i+1}/{len(test)}', flush=True)
    handle.remove()
    probs, features = np.stack(probs), np.stack(capture)
    assert features.shape == (638,512) and np.isfinite(features).all() and np.isfinite(probs).all()
    labels = np.array([[float(r[c]) for c in CLASSES].index(1.0) for r in test])
    preds = probs.argmax(1)
    with (HISTORY / arm['id'] / 'predictions.csv').open() as f:
        old = list(csv.DictReader(f))
    assert [r['FileName'] for r in old] == [r['FileName'] for r in test]
    old_probs = np.array([[float(r['p_'+c]) for c in CLASSES] for r in old])
    max_delta = float(np.max(np.abs(probs-old_probs)))
    regression = np.array_equal(preds, old_probs.argmax(1)) and np.allclose(probs, old_probs, atol=1e-5, rtol=0)
    result = {'arm': arm['id'], 'macro_f1': float(f1_score(labels,preds,average='macro')),
        'accuracy': float(accuracy_score(labels,preds)), 'confusion_matrix': confusion_matrix(labels,preds).tolist(),
        'regression_passed': bool(regression), 'max_probability_delta': max_delta,
        'feature_shape': list(features.shape), 'base_commit': base_commit,
        'torch': torch.__version__, 'numpy': np.__version__, 'gpu': torch.cuda.get_device_name(0),
        'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'normalization': {'mean': float(mean), 'std': float(std)}}
    dump(out / (arm['id']+'_metrics.json'), result)
    if not regression:
        raise RuntimeError('Historical prediction regression failed: '+str(result))
    np.savez_compressed(out / (arm['id']+'_features.npz'), features=features, probabilities=probs,
                        labels=labels, ids=np.array([Path(r['FileName']).stem for r in test]))
    print(json.dumps(result), flush=True)


def run(config):
    from crab_epoch_checkpoint import memory_guard
    from run_n3_phase_a_safe import cgroup_path, snapshot
    memory_guard(start=True)
    if not os.environ.get('TQ_JOB_ID') or not os.environ.get('TQ_OUTPUT_DIR'):
        raise RuntimeError('Must run inside the existing queue')
    out = Path(os.environ['TQ_OUTPUT_DIR']).resolve()
    if any(out.iterdir()):
        raise RuntimeError('Use a new, empty queue output')
    verify(config)
    dump(out / 'manifest.json', config)
    completed = []
    try:
        for i, arm in enumerate(config['arms']):
            memory_guard(start=True)
            dump(out / 'status.json', {'state':'extracting','active':arm['id'],'completed':completed})
            proc = subprocess.Popen([sys.executable,'-B',str(Path(__file__)), '--worker',str(i),'--output',str(out)])
            start = time.monotonic()
            try:
                while proc.poll() is None:
                    memory_guard()
                    if time.monotonic()-start > 1800:
                        raise RuntimeError('Feature child exceeded 30-minute limit')
                    with (out/'resources.jsonl').open('a') as f:
                        f.write(json.dumps(snapshot(cgroup_path()))+'\n')
                    time.sleep(5)
                if proc.returncode:
                    raise RuntimeError(f'{arm["id"]} exit {proc.returncode}')
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try: proc.wait(timeout=10)
                    except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            completed.append(arm['id'])
        verify(config)
        dump(out/'status.json', {'state':'complete','completed':completed,'training':False})
    except BaseException as exc:
        dump(out/'status.json', {'state':'failed','completed':completed,'error':str(exc)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--worker',type=int)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        config = json.loads((HERE/'manifest.json').read_text())
        if args.run: run(config)
        elif args.worker is not None: worker(args.worker,args.output,config)
        else: parser.error('Choose --prepare, --run, or --worker')
