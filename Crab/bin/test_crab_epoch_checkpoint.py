"""CPU-only checkpoint tests: no forward/backward/optimizer.step, no remote W&B."""
import ast
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

import numpy as np
import torch

import crab_epoch_checkpoint as cp
import run_crab_queue_job as wrapper


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, 2)
        self.register_buffer('persistent_counter', torch.tensor(13))


def objects():
    models = {name: Tiny() for name in cp.MODELS}
    opts = {name: torch.optim.AdamW(model.parameters(), lr=.003)
            for name, model in models.items()}
    scheds = {name: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
              for name, opt in opts.items()}
    # Populate representative Adam state WITHOUT executing any model updates.
    for opt in opts.values():
        for group in opt.param_groups:
            for p in group['params']:
                opt.state[p] = {'step': torch.tensor(7.), 'exp_avg': torch.full_like(p, .2),
                                'exp_avg_sq': torch.full_like(p, .4)}
    for sch in scheds.values():
        sch.last_epoch = 2
        sch._step_count = 3
    return models, opts, scheds


class Checkpoints(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='crab-checkpoint-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for p in ('output', 'checkpoints', 'execution/control'):
            (self.root/p).mkdir(parents=True)
        self.ctx = cp.QueueContext('job1', 'exec1', 'a'*64, self.root/'output',
                                   self.root/'checkpoints', self.root/'execution/control')
        self.store = cp.EpochStore(self.ctx, 'input-fingerprint')
        self.models, self.opts, self.scheds = objects()
        self.gens = {n: torch.Generator().manual_seed(i+12)
                     for i, n in enumerate(('train', 'dev', 'sampler'))}
        self.progress = dict(completed_epoch=0, next_epoch=1, total_epochs=3,
                             optimizer_updates=7, best_epoch=0, best_metric=.6,
                             epochs_no_improve=0, patience=0, logging_step=10,
                             accumulation_pending=0, accumulation_steps=4)

    def save(self, **kwargs):
        return self.store.save(self.models, self.opts, self.scheds, self.gens,
                               self.progress, {'mean': .01, 'std': .1},
                               {'id': 'run1', 'next_step': self.progress['logging_step']},
                               best_changed=kwargs.pop('best_changed', True), **kwargs)

    def pause(self, **overrides):
        request = dict(version=1, job_id='job1', execution_id='exec1',
                       action='pause', reason='manual', request_id='request1')
        request.update(overrides)
        cp.atomic_json(self.ctx.control/'pause.json', request)

    def equal_tree(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for k in left:
                self.equal_tree(left[k], right[k])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.equal_tree(a, b)
        else:
            self.assertEqual(left, right)

    def test_complete_named_state_restore(self):
        manifest = self.save()
        models, opts, scheds = objects()
        # Dict insertion order is intentionally different; semantic names must win.
        opts = dict(reversed(list(opts.items())))
        scheds = dict(reversed(list(scheds.items())))
        resumed = cp.EpochStore(replace(self.ctx, execution_id='exec2'), 'input-fingerprint')
        runtime = resumed.restore(manifest, models, opts, scheds)
        self.assertEqual(runtime['progress'], self.progress)
        self.assertEqual(runtime['normalization'], {'mean': .01, 'std': .1})
        for k in models:
            self.equal_tree(models[k].state_dict(), self.models[k].state_dict())
            self.equal_tree(opts[k].state_dict(), self.opts[k].state_dict())
            self.equal_tree(scheds[k].state_dict(), self.scheds[k].state_dict())

    def test_current_and_best_are_distinct_immutable_versions(self):
        first = self.save()
        hashes = {n: cp.digest(first.parent/n) for n in cp.STATE_FILES}
        with torch.no_grad():
            self.models['ser'].proj.weight.fill_(9)  # fixture mutation, not training
        self.progress.update(completed_epoch=1, next_epoch=2, epochs_no_improve=1)
        second = self.save(previous=first, best_changed=False)
        self.assertNotEqual(cp.digest(second.parent/'ser.pt'), cp.digest(second.parent/'best_ser.pt'))
        self.assertEqual(os.stat(first.parent/'best_ser.pt').st_ino,
                         os.stat(second.parent/'best_ser.pt').st_ino)
        self.assertEqual(hashes, {n: cp.digest(first.parent/n) for n in cp.STATE_FILES})

    def test_rng_and_next_loader_order(self):
        state = self.store.inspect(self.save())['rng']
        def draws():
            shuffled = torch.utils.data.DataLoader(list(range(17)), batch_size=4,
                       shuffle=True, generator=self.gens['train'], num_workers=0)
            weighted = torch.utils.data.WeightedRandomSampler([1.,2.,3.], 12,
                       generator=self.gens['sampler'])
            return (random.random(), np.random.rand(), torch.rand(3),
                    [b.tolist() for b in shuffled], list(weighted))
        expected = draws()
        objects()  # reconstruction consumes RNG; restore must happen afterwards
        cp.restore_rng(state, self.gens)
        self.equal_tree(expected, draws())

    def test_rng_generator_set_change_rejected(self):
        with self.assertRaises(cp.CheckpointError):
            cp.restore_rng(cp.capture_rng(self.gens), {})

    def test_precision_change_rejected(self):
        with self.assertRaises(cp.CheckpointError):
            self.store.restore(self.save(), self.models, self.opts, self.scheds, precision='bfloat16')

    def test_bfloat16_mode_without_scaler_roundtrip(self):
        m = self.save(precision='bfloat16')
        r = self.store.restore(m, self.models, self.opts, self.scheds, precision='bfloat16')
        self.assertIsNone(r['scaler'])

    def test_snapshot_reopens_in_new_python_process(self):
        m = self.save()
        code = """from pathlib import Path
import sys
import crab_epoch_checkpoint as c
root=Path(sys.argv[1])
ctx=c.QueueContext('job1','exec2','a'*64,root/'output',root/'checkpoints',root/'execution/control')
r=c.EpochStore(ctx,'input-fingerprint').inspect(sys.argv[2])
assert r['progress']['next_epoch']==1
assert set(c.load_torch(Path(sys.argv[2]).parent/'optimizers.pt'))=={'audio','text','ser'}
print('new-process-read-ok')
"""
        result = subprocess.run([sys.executable, '-B', '-c', code, str(self.root), str(m)],
                                cwd=Path(__file__).parent, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('new-process-read-ok', result.stdout)

    def test_missing_optimizer_inside_valid_hash_rejected(self):
        m = self.save()
        path = m.parent/'optimizers.pt'
        state = cp.load_torch(path)
        state.pop('text')
        path.unlink()
        cp.save_torch(path, state)
        metadata = cp.read_json(m)
        for entry in metadata['files']:
            if entry['path'] == 'optimizers.pt':
                entry.update(size=path.stat().st_size, sha256=cp.digest(path))
        cp.atomic_json(m, metadata)
        with self.assertRaises(cp.CheckpointError):
            self.store.restore(m, self.models, self.opts, self.scheds)

    def test_environment_requires_dedicated_roots(self):
        env = dict(TQ_JOB_ID='job1', TQ_EXECUTION_ID='exec1', TQ_SPEC_SHA256='a'*64,
                   TQ_OUTPUT_DIR=str(self.ctx.output), TQ_CHECKPOINT_DIR=str(self.ctx.checkpoints),
                   TQ_CONTROL_DIR=str(self.ctx.control), TQ_RESUME_MANIFEST='')
        self.assertEqual(cp.QueueContext.from_env(env), self.ctx)
        env['TQ_CONTROL_DIR'] = str(self.ctx.output)
        with self.assertRaises(cp.CheckpointError):
            cp.QueueContext.from_env(env)

    def test_memory_guard_checks_finite_limits_and_host_reserve(self):
        contents = {'/proc/self/cgroup':'0::/fixture',
                    '/proc/meminfo': 'MemAvailable: 41943040 kB',
                    '/sys/fs/cgroup/fixture/memory.max': str(20*cp.GIB),
                    '/sys/fs/cgroup/fixture/memory.high': str(16*cp.GIB),
                    '/sys/fs/cgroup/fixture/memory.swap.max':'0',
                    '/sys/fs/cgroup/fixture/memory.oom.group':'1',
                    '/sys/fs/cgroup/fixture/memory.current':str(cp.GIB)}
        with patch.object(Path, 'read_text', lambda p, **kw: contents[str(p)]):
            cp.memory_guard(start=True)
            for key, wrong in [('memory.max','max'),('memory.high','max'),
                               ('memory.swap.max','max'),('memory.oom.group','0')]:
                target='/sys/fs/cgroup/fixture/'+key
                original=contents[target]
                contents[target]=wrong
                with self.assertRaises(cp.CheckpointError):
                    cp.memory_guard(start=True)
                contents[target]=original
            contents['/proc/meminfo']='MemAvailable: 1024 kB'
            with self.assertRaises(cp.CheckpointError):
                cp.memory_guard(start=True)

    def test_training_loss_and_optimizer_steps_match_legacy(self):
        directory=Path(__file__).parent
        source=(directory/'train_crab_queue.py').read_text()
        legacy=(directory/'train_crab_lora.py').read_text()
        start='        with _amp_ctx():\n            ssl = ssl_model'
        end='        # (2) compute scalars ONCE'
        # Compare actual training math, not comments or logging wrappers.
        self.assertEqual(source[source.rindex(start):source.index(end)],
                         legacy[legacy.rindex(start):legacy.index(end)])
        self.assertLess(source.index('resume_state = store.restore'), source.index('wandb_identity = init_wandb'))
        self.assertLess(source.index('wandb_identity = init_wandb'), source.index('restore_rng(resume_state'))
        self.assertLess(source.index('restore_rng(resume_state'), source.index('for epoch in range(next_epoch'))
        ast.parse(source)

    def test_parameter_order_change_rejected(self):
        m = self.save()
        self.opts['ser'].param_groups[0]['params'].reverse()
        with self.assertRaises(cp.CheckpointError):
            self.store.restore(m, self.models, self.opts, self.scheds)

    def test_partial_gradients_rejected_before_commit(self):
        self.models['ser'].proj.weight.grad = torch.ones_like(self.models['ser'].proj.weight)
        with self.assertRaises(cp.CheckpointError):
            self.save()
        self.assertEqual(list(self.ctx.checkpoints.iterdir()), [])

    def test_partial_accumulation_rejected(self):
        self.progress['accumulation_pending'] = 1
        with self.assertRaises(cp.CheckpointError):
            self.save()

    def test_missing_named_optimizer_rejected(self):
        self.opts.pop('text')
        with self.assertRaises(cp.CheckpointError):
            self.save()

    def test_checksum_corruption_rejected(self):
        m = self.save()
        with (m.parent/'optimizers.pt').open('ab') as f:
            f.write(b'corruption')
        with self.assertRaises(cp.CheckpointError):
            self.store.restore(m, self.models, self.opts, self.scheds)

    def test_missing_file_rejected(self):
        m = self.save()
        (m.parent/'schedulers.pt').unlink()
        with self.assertRaises((cp.CheckpointError, OSError)):
            self.store.inspect(m)

    def test_manifest_traversal_rejected(self):
        m = self.save()
        data = cp.read_json(m)
        data['files'][0]['path'] = '../audio.pt'
        cp.atomic_json(m, data)
        with self.assertRaises(cp.CheckpointError):
            self.store.inspect(m)

    def test_symlink_rejected(self):
        m = self.save()
        alias = m.parent/'alias.json'
        alias.symlink_to(m)
        with self.assertRaises(cp.CheckpointError):
            self.store.inspect(alias)

    def test_input_fingerprint_change_rejected(self):
        with self.assertRaises(cp.CheckpointError):
            cp.EpochStore(self.ctx, 'changed').inspect(self.save())

    def test_wrong_job_rejected(self):
        with self.assertRaises(cp.CheckpointError):
            cp.EpochStore(replace(self.ctx, job_id='different'), 'input-fingerprint').inspect(self.save())

    def test_torch_save_failure_never_commits_or_acks(self):
        self.pause()
        with patch.object(cp, 'save_torch', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.save()
        self.assertFalse(list(self.ctx.checkpoints.glob('*/manifest.json')))
        self.assertFalse((self.ctx.control/'pause_ack.json').exists())

    def test_manifest_publish_failure_never_acks(self):
        self.pause()
        with patch.object(cp.os, 'replace', side_effect=OSError('rename failed')):
            with self.assertRaises(OSError):
                self.save()
        self.assertFalse(list(self.ctx.checkpoints.glob('*/manifest.json')))
        self.assertFalse((self.ctx.control/'pause_ack.json').exists())

    def test_fsync_failure_never_acks(self):
        self.pause()
        with patch.object(cp.os, 'fsync', side_effect=OSError('I/O error')):
            with self.assertRaises(OSError):
                self.save()
        self.assertFalse(list(self.ctx.checkpoints.glob('*/manifest.json')))
        self.assertFalse((self.ctx.control/'pause_ack.json').exists())

    def test_no_pause_continues(self):
        self.assertIsNone(self.store.boundary(self.save()))

    def test_manual_pause_acks_75(self):
        self.pause()
        m = self.save()
        self.assertEqual(self.store.boundary(m), 75)
        ack = cp.read_json(self.ctx.control/'pause_ack.json')
        self.assertEqual(ack['request_id'], 'request1')
        self.assertEqual(ack['manifest'], str(m))

    def test_preempt_pause_acks_75(self):
        self.pause(reason='preempt')
        self.assertEqual(self.store.boundary(self.save()), 75)

    def test_stale_pause_rejected(self):
        self.pause(execution_id='old')
        with self.assertRaises(cp.CheckpointError):
            self.store.boundary(self.save())
        self.assertFalse((self.ctx.control/'pause_ack.json').exists())

    def test_invalid_request_rejected(self):
        self.pause(request_id='../bad')
        with self.assertRaises(cp.CheckpointError):
            self.store.boundary(self.save())

    def test_old_execution_snapshot_cannot_ack_new_pause(self):
        m = self.save()
        self.pause(execution_id='exec2')
        store = cp.EpochStore(replace(self.ctx, execution_id='exec2'), 'input-fingerprint')
        with self.assertRaises(cp.CheckpointError):
            store.boundary(m)

    def test_last_epoch_wins_even_over_invalid_pause(self):
        self.progress['total_epochs'] = 1
        self.pause(job_id='wrong')
        self.assertEqual(self.store.boundary(self.save()), 0)
        self.assertFalse((self.ctx.control/'pause_ack.json').exists())

    def test_early_stop_wins_over_pause(self):
        self.progress.update(patience=2, epochs_no_improve=2)
        self.pause()
        self.assertEqual(self.store.boundary(self.save()), 0)

    def test_ack_write_failure_does_not_return_75(self):
        self.pause()
        m = self.save()
        with patch.object(cp, 'atomic_json', side_effect=OSError('ack disk full')):
            with self.assertRaises(OSError):
                self.store.boundary(m)

    def test_real_queue_validator_accepts_manifest_and_ack(self):
        source = Path(os.environ.get('TRAIN_QUEUE_ROOT', '/home/brant/Project/train_queue'))/'src/train_queue/protocol.py'
        self.assertTrue(source.is_file(), 'Set TRAIN_QUEUE_ROOT to real queue repository')
        spec = importlib.util.spec_from_file_location('queue_protocol_readonly', source)
        protocol = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(protocol)
        self.pause()
        m = self.save()
        self.assertEqual(self.store.boundary(m), 75)
        validated = protocol.validate_pause(self.ctx.control.parent, self.ctx.checkpoints,
                     'job1', 'exec1', 'a'*64, 'request1')
        self.assertEqual(validated, str(m))
        self.assertEqual(protocol.latest_checkpoint(self.ctx.checkpoints, 'job1', 'a'*64), str(m))

    def test_wandb_same_id_and_explicit_resume(self):
        args = SimpleNamespace(project_name='project', wandb_entity='entity', wandb_group='group', run_name=None)
        fake = SimpleNamespace(init=Mock())
        with patch.dict(os.environ, {'WANDB_MODE': 'online'}, clear=True):
            runid = 'tq-' + cp.hashlib.sha256(b'job1').hexdigest()[:20]
            fake.init.return_value = SimpleNamespace(id=runid, step=0)
            initial = cp.init_wandb(fake, self.ctx, 'input-fingerprint', args, {})
            self.assertEqual(fake.init.call_args.kwargs['resume'], 'never')
            resumed = cp.init_wandb(fake, self.ctx, 'input-fingerprint', args, {},
                                   {'wandb': dict(initial, next_step=10)})
            self.assertEqual(initial, resumed)
            self.assertEqual(fake.init.call_args.kwargs['resume'], 'must')
            fake.init.return_value.step = 11
            with self.assertRaises(cp.CheckpointError):
                cp.init_wandb(fake, self.ctx, 'input-fingerprint', args, {},
                              {'wandb': dict(initial, next_step=10)})

    def test_offline_wandb_rejected(self):
        with patch.dict(os.environ, {'WANDB_MODE':'offline'}, clear=True):
            with self.assertRaises(cp.CheckpointError):
                cp.init_wandb(Mock(), self.ctx, 'x', SimpleNamespace(), {})

    def test_disk_reserve_rejects_insufficient_space(self):
        with patch.object(cp.shutil, 'disk_usage', return_value=SimpleNamespace(free=cp.GIB)):
            with self.assertRaises(cp.CheckpointError):
                cp.require_disk_budget(self.models, self.opts, 3, self.ctx.checkpoints)

    def test_wrapper_exec_keeps_environment_and_arguments(self):
        env = {'CUDA_VISIBLE_DEVICES':'7', 'WANDB_RUN_ID':'keep', 'WANDB_RESUME':'must'}
        with patch.dict(os.environ, env, clear=True), patch.object(wrapper.QueueContext, 'from_env'), \
             patch.object(wrapper, 'memory_guard'), patch.object(wrapper.os, 'execv') as execute:
            wrapper.main(['--', '--seed', '42'])
            self.assertEqual(dict(os.environ), env)
            argv = execute.call_args.args[1]
            self.assertEqual(argv[-2:], ['--seed', '42'])
            self.assertIn('train_crab_queue.py', argv[2])

    def test_wrapper_actual_exec_propagates_75(self):
        code = """import os,sys
import run_crab_queue_job as w
w.QueueContext.from_env=lambda:None
w.memory_guard=lambda **kw:None
original=os.execv
w.os.execv=lambda path,argv:original(path,[path,'-c','raise SystemExit(75)'])
os.environ['CUDA_VISIBLE_DEVICES']='0'
w.main(['--','--seed','42'])
"""
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=Path(__file__).parent,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), timeout=30)
        self.assertEqual(result.returncode, 75)

    def test_lora_config_rebuild_preserves_trainable_layout(self):
        from peft import LoraConfig, PeftModel, get_peft_model
        cfg = LoraConfig(r=2, lora_alpha=4, target_modules=['proj'])
        original = get_peft_model(Tiny(), cfg)
        adapter = self.root/'adapter'
        original.save_pretrained(adapter)
        warm = PeftModel.from_pretrained(Tiny(), adapter, is_trainable=True)
        rebuilt = cp.build_resume_lora(Tiny(), adapter)
        self.assertEqual(cp.model_layout({'model':warm}), cp.model_layout({'model':rebuilt}))
        self.assertGreater(sum(p.numel() for p in rebuilt.parameters() if p.requires_grad), 0)


if __name__ == '__main__':
    torch.set_num_threads(1)
    program = unittest.main(verbosity=2, exit=False)
    if os.environ.get('CRAB_QUEUE_TEST_REPORT'):
        target = Path(os.environ['CRAB_QUEUE_TEST_REPORT'])
        target.parent.mkdir(parents=True, exist_ok=True)
        cp.atomic_json(target, {
            'tests_run': program.result.testsRun,
            'failures': len(program.result.failures), 'errors': len(program.result.errors),
            'skipped': len(program.result.skipped), 'successful': program.result.wasSuccessful(),
            'python': sys.version, 'torch': str(torch.__version__),
            'self_maxrss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'child_maxrss_kib': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            'real_training': False, 'gpu_disabled': os.environ.get('CUDA_VISIBLE_DEVICES') == '',
            'source_sha256': {name:cp.digest(Path(__file__).with_name(name)) for name in (
                'crab_epoch_checkpoint.py','train_crab_queue.py','run_crab_queue_job.py',
                'test_crab_epoch_checkpoint.py')},
        })
    raise SystemExit(0 if program.result.wasSuccessful() else 1)
