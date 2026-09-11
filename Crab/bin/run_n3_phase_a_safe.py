"""Sequential N3 phase-A runner; refuses to train outside its memory cgroup."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

GIB = 1024 ** 3
MEMORY_MAX = 20 * GIB
MEMORY_HIGH = 16 * GIB
HOST_STOP = 12 * GIB
OWN_STOP = 19 * GIB


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cgroup_path():
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        if line.startswith('0::'):
            return Path('/sys/fs/cgroup' + line[3:])
    raise RuntimeError('cgroup v2 is required')


def verify_memory_limits(path):
    expected = {'memory.max': str(MEMORY_MAX), 'memory.high': str(MEMORY_HIGH),
                'memory.swap.max': '0', 'memory.oom.group': '1'}
    actual = {key: (path / key).read_text().strip() for key in expected}
    if actual != expected:
        raise RuntimeError('Memory isolation mismatch: ' + repr(actual))
    return actual


def host_available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('Cannot read host available RAM')


def verify_files(config):
    if config.get('approval_status') != 'approved_provisional':
        raise RuntimeError('Provisional data and split-use approval is required')
    for path, expected in config['frozen_files_sha256'].items():
        if sha(path) != expected:
            raise RuntimeError('Frozen input changed: ' + path)
    audio_hashes = json.loads(Path(config['tts_audio_sha256_manifest']).read_text())
    for path, expected in audio_hashes.items():
        if sha(path) != expected:
            raise RuntimeError('Frozen TTS audio changed: ' + path)


def stop_child(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
    except ProcessLookupError:
        pass


def write_status(root, value):
    value = dict(value, updated_at=utc())
    temporary = root / 'status.tmp.json'
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(root / 'status.json')


def snapshot(cgroup):
    return {'timestamp': utc(), 'host_available_bytes': host_available(),
            'cgroup_current_bytes': int((cgroup / 'memory.current').read_text()),
            'cgroup_events': (cgroup / 'memory.events').read_text().strip()}


def run_child(command, config, root, cgroup, phase):
    if host_available() < MEMORY_MAX + HOST_STOP:
        raise RuntimeError('Insufficient host reserve; require at least 32 GiB available before a run')
    gpu = subprocess.check_output([
        'nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '--id=0'],
        text=True).strip()
    if int(gpu) > 1024:
        raise RuntimeError('GPU 0 is occupied (>1 GiB); refusing to overlap another job')
    child_env = os.environ.copy()
    child_env.update({
        'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'CUDA_VISIBLE_DEVICES': '0',
        'PYTHONUNBUFFERED': '1', 'WANDB_ENTITY': config['wandb_entity'],
        'WANDB_PROJECT': config['wandb_project'], 'WANDB_MODE': 'online',
        'WANDB_DIR': str(root), 'WANDB_RESUME': 'never',
    })
    child_env.pop('WANDB_RUN_ID', None)
    print(json.dumps({'event': 'start', 'phase': phase, 'command': command,
                      'resources': snapshot(cgroup)}, ensure_ascii=False), flush=True)
    process = subprocess.Popen(command, cwd=config['crab_root'], env=child_env,
                               start_new_session=True)
    last_report = 0.0
    started = time.monotonic()
    try:
        while process.poll() is None:
            state = snapshot(cgroup)
            if state['host_available_bytes'] < HOST_STOP:
                raise RuntimeError('Safety stop: host available RAM below 12 GiB')
            if state['cgroup_current_bytes'] >= OWN_STOP:
                raise RuntimeError('Safety stop: job RAM reached 19 GiB')
            if time.monotonic() - started > 7200:
                raise RuntimeError('Safety stop: a single child exceeded 2 hours')
            if time.monotonic() - last_report >= 30:
                with (root / 'resources.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(dict(state, phase=phase)) + '\n')
                print('RESOURCE ' + json.dumps(dict(state, phase=phase)), flush=True)
                last_report = time.monotonic()
            time.sleep(2)
    except BaseException:
        stop_child(process)
        raise
    if process.returncode:
        raise RuntimeError(f'{phase} exited with code {process.returncode}; queue stopped')
    print(json.dumps({'event': 'completed', 'phase': phase,
                      'elapsed_seconds': round(time.monotonic() - started, 2)}), flush=True)


def command_for(config, arm, model_dir, smoke=False):
    args = [config['python'], str(Path(config['crab_root']) / 'bin/train_crab_lora.py')]
    for key, value in config['common_cli'].items():
        args.append('--' + key)
        if isinstance(value, list):
            args.extend(map(str, value))
        else:
            args.append(str(value))
    args.extend(config['common_flags'])
    args.extend(['--df_path', arm['training_csv'], '--experiment_metadata', arm['metadata_json'],
                 '--model_path', str(model_dir), '--run_name', arm['run_name']])
    if smoke:
        args.append('--smoke_test')
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root = Path(args.campaign).resolve()
    config = json.loads((root / 'campaign.json').read_text())
    cgroup = cgroup_path()
    verify_memory_limits(cgroup)
    verify_files(config)
    if args.check_only:
        print('CHECK PASS: approval, frozen files and memory cgroup verified', flush=True)
        return
    if (root / 'status.json').exists():
        raise RuntimeError('Campaign already has status.json; no automatic resume or overwrite')
    write_status(root, {'state': 'smoke_test', 'completed': []})
    completed = []
    try:
        smoke_dir = root / 'smoke'
        if smoke_dir.exists():
            raise RuntimeError('Smoke output already exists; refusing overwrite')
        run_child(command_for(config, config['arms'][0], smoke_dir, smoke=True),
                  config, root, cgroup, 'smoke')
        for arm in config['arms']:
            verify_files(config)
            model_dir = Path(arm['output_dir'])
            if model_dir.exists():
                raise RuntimeError('Model output already exists: ' + str(model_dir))
            write_status(root, {'state': 'training', 'active': arm['experiment_id'],
                                'completed': completed})
            run_child(command_for(config, arm, model_dir), config, root, cgroup,
                      arm['experiment_id'])
            result_path = model_dir / 'run_result.json'
            result = json.loads(result_path.read_text())
            if result['test_evaluated']:
                raise RuntimeError('Unexpected Test evaluation; queue stopped')
            completed.append(dict(result, experiment_id=arm['experiment_id']))
            write_status(root, {'state': 'between_runs', 'completed': completed})
        write_status(root, {'state': 'complete_dev_only', 'completed': completed,
                            'note': 'No Test evaluation; all results are provisional'})
    except BaseException as exc:
        write_status(root, {'state': 'stopped', 'completed': completed,
                            'error': str(exc), 'error_type': type(exc).__name__})
        raise


if __name__ == '__main__':
    main()
