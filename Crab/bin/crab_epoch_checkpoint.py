"""CRAB epoch_v1 checkpoint protocol. No model loading or training on import.

Trusted local job files only: hashes provide integrity, not authenticity.
Snapshots are immutable; legacy/frozen experiments are never converted or edited.
"""
import copy
import csv
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile
import time
import uuid

SCHEMA = "crab-epoch-v1-full-state-1"
GIB = 1024 ** 3
MODELS = ("audio", "text", "ser")
STATE_FILES = tuple(n + ".pt" for n in MODELS) + (
    "optimizers.pt", "schedulers.pt", "runtime.pt",
    "best_audio.pt", "best_text.pt", "best_ser.pt",
)


class CheckpointError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise CheckpointError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    require(not path.is_symlink(), "Refusing symlink output")
    # Leave an incomplete .pending file on failure; never advertise it as committed.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".pending-", delete=False) as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
        pending = f.name
    os.replace(pending, path)
    sync_dir(path.parent)


def read_json(path):
    require(Path(path).stat().st_size <= 1024 * 1024, "JSON exceeds 1 MiB")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), "Expected JSON object")
    return value


def safe_path(path):
    p = Path(path)
    require(p.is_absolute(), "Queue paths must be absolute")
    for part in (p, *p.parents):
        require(not part.is_symlink(), "Symlinks are not supported in queue paths")
    return p.resolve()


def contained(path, root):
    p = safe_path(path)
    require(p.is_relative_to(safe_path(root)) and p.is_file(), "File outside checkpoint root")
    return p


@dataclass(frozen=True)
class QueueContext:
    job_id: str
    execution_id: str
    spec_sha256: str
    output: Path
    checkpoints: Path
    control: Path
    resume: str = ""

    @classmethod
    def from_env(cls, environ=None):
        e = os.environ if environ is None else environ
        names = ("TQ_JOB_ID", "TQ_EXECUTION_ID", "TQ_SPEC_SHA256", "TQ_OUTPUT_DIR",
                 "TQ_CHECKPOINT_DIR", "TQ_CONTROL_DIR", "TQ_RESUME_MANIFEST")
        require(all(k in e for k in names), "Missing TQ_* environment; use the queue")
        for key in names[:2]:
            require(bool(re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", e[key])), "Invalid queue ID")
        require(bool(re.fullmatch(r"[0-9a-f]{64}", e["TQ_SPEC_SHA256"])), "Invalid spec hash")
        roots = [safe_path(e[k]) for k in names[3:6]]
        require(all(p.is_dir() for p in roots), "Queue must create dedicated directories")
        require(all(not a.is_relative_to(b) and not b.is_relative_to(a)
                    for i, a in enumerate(roots) for b in roots[i+1:]), "Overlapping queue directories")
        resume = e["TQ_RESUME_MANIFEST"]
        if resume:
            contained(resume, roots[1])
        return cls(e[names[0]], e[names[1]], e[names[2]], *roots, resume)


def memory_guard(start=False):
    """Stay in the queue's cgroup. Never create a nested escaping service."""
    line = next((x[3:] for x in Path("/proc/self/cgroup").read_text().splitlines()
                 if x.startswith("0::")), None)
    require(line is not None, "cgroup v2 required")
    group = Path("/sys/fs/cgroup" + line)
    limits = {n: (group/n).read_text().strip()
              for n in ("memory.max", "memory.high", "memory.swap.max", "memory.oom.group")}
    require(limits["memory.max"].isdigit() and 0 < int(limits["memory.max"]) <= 20*GIB,
            "Queue must set finite MemoryMax <=20 GiB")
    require(limits["memory.high"].isdigit() and
            0 < int(limits["memory.high"]) <= min(16*GIB, int(limits["memory.max"])),
            "Queue must set MemoryHigh <=16 GiB and <=MemoryMax")
    require(limits["memory.swap.max"] == "0" and limits["memory.oom.group"] == "1",
            "Queue must set MemorySwapMax=0 and OOMPolicy=kill")
    available = next(int(x.split()[1])*1024 for x in Path("/proc/meminfo").read_text().splitlines()
                     if x.startswith("MemAvailable:"))
    require(available >= (32 if start else 12)*GIB, "Insufficient host RAM reserve")
    require(int((group/"memory.current").read_text()) < min(19*GIB, int(limits["memory.max"])),
            "Job memory safety stop")
    return limits


def capture_rng(generators):
    import numpy as np
    import torch
    n = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": (n[0], n[1].tolist(), n[2], n[3], n[4]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "generators": {k: v.get_state() for k, v in generators.items()}}


def restore_rng(state, generators):
    import numpy as np
    import torch
    require(set(state["generators"]) == set(generators), "DataLoader generator mismatch")
    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    require(len(state["cuda"]) == cuda_count, "Visible CUDA device count changed")
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    if cuda_count:
        torch.cuda.set_rng_state_all(state["cuda"])
    for key, generator in generators.items():
        generator.set_state(state["generators"][key])


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_tree(x) for x in value)
    if isinstance(value, list):
        return [cpu_tree(x) for x in value]
    return copy.deepcopy(value)


def build_resume_lora(base_model, adapter_dir):
    """Rebuild topology; full CURRENT weights are restored afterwards."""
    from peft import LoraConfig, get_peft_model
    config = LoraConfig.from_pretrained(adapter_dir)
    # PEFT exports inference_mode=True; trainable resume must explicitly undo it.
    config.inference_mode = False
    return get_peft_model(base_model, config)


def save_torch(path, value):
    import torch
    with Path(path).open("xb") as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())


def load_torch(path):
    import torch
    # Only native tensors/primitives are written (NumPy RNG arrays converted to lists).
    return torch.load(path, map_location="cpu", weights_only=True)


def parameter_layout(models, optimizers):
    names = {id(p): f"{name}.{key}" for name, model in models.items()
             for key, p in model.named_parameters()}
    return {name: [[names[id(p)] for p in group["params"]] for group in opt.param_groups]
            for name, opt in optimizers.items()}


def model_layout(models):
    return {name: {"tensors": {k: [list(v.shape), str(v.dtype)]
                              for k, v in model.state_dict().items()},
                   "trainable": [k for k, p in model.named_parameters() if p.requires_grad]}
            for name, model in models.items()}


def object_types(objects):
    return {k: type(v).__module__ + "." + type(v).__qualname__ for k, v in objects.items()}


def is_complete(progress):
    return progress["next_epoch"] >= progress["total_epochs"] or (
        progress["patience"] > 0 and progress["epochs_no_improve"] >= progress["patience"])


def validate_progress(p):
    keys = ("completed_epoch", "next_epoch", "total_epochs", "optimizer_updates",
            "best_epoch", "epochs_no_improve", "patience", "logging_step")
    require(all(type(p.get(k)) is int for k in keys), "Missing/integer progress fields")
    require(0 <= p["completed_epoch"] < p["total_epochs"] and
            p["next_epoch"] == p["completed_epoch"] + 1, "Invalid epoch boundary")
    require(0 <= p["best_epoch"] <= p["completed_epoch"] and
            all(p[k] >= 0 for k in ("optimizer_updates", "epochs_no_improve", "patience", "logging_step")),
            "Invalid best/early-stop progress")
    require(math.isfinite(p["best_metric"]) and 0 <= p["best_metric"] <= 1, "Invalid best metric")
    require(p.get("accumulation_pending") == 0, "Cannot save partial accumulated gradients")
    require(type(p.get("accumulation_steps")) is int and p["accumulation_steps"] > 0,
            "Invalid accumulation configuration")


class EpochStore:
    def __init__(self, context, fingerprint):
        self.ctx = context
        self.fingerprint = fingerprint

    def validate(self, path, current_execution=False):
        path = contained(path, self.ctx.checkpoints)
        require(path.name == "manifest.json", "Expected committed manifest.json")
        m = read_json(path)
        require(m.get("version") == 1 and m.get("job_id") == self.ctx.job_id and
                m.get("spec_sha256") == self.ctx.spec_sha256, "Manifest identity mismatch")
        if current_execution:
            require(m.get("execution_id") == self.ctx.execution_id, "Stale execution snapshot")
        require(type(m.get("completed_epoch")) is int and m["completed_epoch"] >= 0 and
                type(m.get("next_epoch")) is int and m["next_epoch"] == m["completed_epoch"]+1,
                "Manifest epoch mismatch")
        require(isinstance(m.get("files"), list) and len(m["files"]) == len(STATE_FILES),
                "Incomplete snapshot file list")
        seen = set()
        for item in m["files"]:
            name = item["path"]
            require(name in STATE_FILES and name not in seen, "Invalid/duplicate snapshot file")
            seen.add(name)
            f = contained(path.parent/name, path.parent)
            require(type(item["size"]) is int and f.stat().st_size == item["size"]
                    and digest(f) == item["sha256"], "Snapshot checksum mismatch: " + name)
        return m

    def inspect(self, path):
        m = self.validate(path)
        r = load_torch(Path(path).parent/"runtime.pt")
        require(r.get("schema") == SCHEMA and r.get("job_id") == self.ctx.job_id and
                r.get("spec_sha256") == self.ctx.spec_sha256 and
                r.get("fingerprint") == self.fingerprint, "Runtime/config/input fingerprint mismatch")
        require(r.get("execution_id") == m["execution_id"], "Runtime execution identity mismatch")
        validate_progress(r["progress"])
        require(all(r["progress"][k] == m[k] for k in ("completed_epoch", "next_epoch")),
                "Runtime/manifest epoch mismatch")
        require(r["complete"] == is_complete(r["progress"]), "Completion status mismatch")
        require(r["best_reference"]["epoch"] == r["progress"]["best_epoch"], "Best epoch mismatch")
        require(r["precision"] in ("float32", "bfloat16"), "Unsupported precision")
        require(set(r["model_layout"]) == set(MODELS), "Incomplete current models")
        require(set(r["rng"]) == {"python", "numpy", "torch", "cuda", "generators"}, "Incomplete RNG")
        require(set(r["normalization"]) == {"mean", "std"} and
                all(math.isfinite(x) for x in r["normalization"].values()) and
                r["normalization"]["std"] > 0, "Invalid normalization")
        require(isinstance(r["wandb"]["id"], str) and bool(r["wandb"]["id"]) and
                r["wandb"]["next_step"] == r["progress"]["logging_step"], "Invalid W&B state")
        return r

    def save(self, models, optimizers, schedulers, generators, progress, normalization,
             wandb_state, *, precision="float32", scaler=None, previous=None, best_changed=False):
        require(set(models) == set(MODELS), "Must save audio, text, ser together")
        require(set(optimizers) == set(schedulers) and "ser" in optimizers, "Named optimizer mismatch")
        validate_progress(progress)
        require(wandb_state["next_step"] == progress["logging_step"], "Logging progress mismatch")
        for opt in optimizers.values():
            require(all(p.grad is None for g in opt.param_groups for p in g["params"]),
                    "Snapshot requires optimizer zero_grad at epoch boundary")
        previous_state = None
        if previous:
            previous_state = self.inspect(previous)
            require(previous_state["progress"]["next_epoch"] == progress["completed_epoch"],
                    "Snapshot sequence skipped or repeated an epoch")
        else:
            require(progress["completed_epoch"] == 0 and best_changed, "First epoch must establish best")
        if not best_changed:
            require(previous_state is not None and
                    previous_state["progress"]["best_epoch"] == progress["best_epoch"] and
                    previous_state["progress"]["best_metric"] == progress["best_metric"],
                    "Best reference does not match previous snapshot")
        else:
            require(progress["best_epoch"] == progress["completed_epoch"], "Best epoch is not current")
        name = f"epoch-{progress['completed_epoch']:04d}-{self.ctx.execution_id}-{uuid.uuid4().hex[:12]}"
        folder = self.ctx.checkpoints/name
        folder.mkdir(mode=0o700, exist_ok=False)
        sync_dir(self.ctx.checkpoints)
        # Full state_dict includes base/frozen parameters, adapters and every persistent buffer.
        # Serialize a single model at a time to bound CPU staging memory.
        for key in MODELS:
            save_torch(folder/(key+".pt"), cpu_tree(models[key].state_dict()))
        save_torch(folder/"optimizers.pt", cpu_tree({k: v.state_dict() for k,v in optimizers.items()}))
        save_torch(folder/"schedulers.pt", cpu_tree({k: v.state_dict() for k,v in schedulers.items()}))
        for key in MODELS:
            origin = folder/(key+".pt") if best_changed else Path(previous).parent/("best_"+key+".pt")
            # Same checkpoint filesystem, immutable source; hardlinks are not symlinks.
            os.link(origin, folder/("best_"+key+".pt"))
        best_ref = ({"epoch": progress["best_epoch"], "snapshot": name}
                    if best_changed else previous_state["best_reference"])
        runtime = {
            "schema": SCHEMA, "job_id": self.ctx.job_id, "execution_id": self.ctx.execution_id,
            "spec_sha256": self.ctx.spec_sha256, "fingerprint": self.fingerprint,
            "progress": dict(progress), "complete": is_complete(progress),
            "normalization": dict(normalization), "wandb": dict(wandb_state),
            "precision": precision, "scaler": cpu_tree(scaler.state_dict()) if scaler else None,
            "rng": capture_rng(generators), "model_layout": model_layout(models),
            "optimizer_layout": parameter_layout(models, optimizers),
            "optimizer_types": object_types(optimizers), "scheduler_types": object_types(schedulers),
            "best_reference": best_ref,
        }
        save_torch(folder/"runtime.pt", runtime)
        sync_dir(folder)
        manifest = {"version": 1, "job_id": self.ctx.job_id,
                    "execution_id": self.ctx.execution_id, "spec_sha256": self.ctx.spec_sha256,
                    "completed_epoch": progress["completed_epoch"], "next_epoch": progress["next_epoch"],
                    "files": [{"path": n, "size": (folder/n).stat().st_size, "sha256": digest(folder/n)}
                              for n in STATE_FILES]}
        atomic_json(folder/"manifest.json", manifest)  # sole commit point, after all data fsync
        return folder/"manifest.json"

    def restore(self, path, models, optimizers, schedulers, *, precision="float32", scaler=None):
        runtime = self.inspect(path)  # validate ALL checksums before deserializing model weights
        require(runtime["model_layout"] == model_layout(models), "Model/trainable layout changed")
        require(runtime["optimizer_layout"] == parameter_layout(models, optimizers) and
                runtime["optimizer_types"] == object_types(optimizers) and
                runtime["scheduler_types"] == object_types(schedulers), "Optimizer/scheduler layout changed")
        require(runtime["precision"] == precision and
                (runtime["scaler"] is None) == (scaler is None), "Precision/scaler mismatch")
        for key in MODELS:
            models[key].load_state_dict(load_torch(Path(path).parent/(key+".pt")), strict=True)
        opts = load_torch(Path(path).parent/"optimizers.pt")
        scheds = load_torch(Path(path).parent/"schedulers.pt")
        require(set(opts) == set(optimizers) and set(scheds) == set(schedulers), "Missing optimizer/scheduler")
        for key, opt in optimizers.items():
            opt.load_state_dict(opts[key])
        for key, sch in schedulers.items():
            sch.load_state_dict(scheds[key])
        if scaler:
            scaler.load_state_dict(runtime["scaler"])
        # Caller restores RNG LAST, after model construction, external logging, and all setup.
        return runtime

    def boundary(self, manifest):
        runtime = self.inspect(manifest)
        if runtime["complete"]:
            return 0  # Completion wins even over malformed/late pause requests.
        pause = self.ctx.control/"pause.json"
        if not pause.exists():
            return None
        request = read_json(contained(pause, self.ctx.control))
        require(request.get("version") == 1 and request.get("job_id") == self.ctx.job_id and
                request.get("execution_id") == self.ctx.execution_id and request.get("action") == "pause",
                "Invalid or stale pause request")
        require(isinstance(request.get("request_id"), str) and
                bool(re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", request["request_id"])), "Invalid request ID")
        require(request.get("reason") in ("manual", "preempt"), "Unknown pause reason")
        self.validate(manifest, current_execution=True)
        atomic_json(self.ctx.control/"pause_ack.json",
                    {"version": 1, "job_id": self.ctx.job_id, "execution_id": self.ctx.execution_id,
                     "request_id": request["request_id"], "manifest": str(Path(manifest).resolve())})
        return 75


def backend_settings():
    import torch
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "environment": {name: os.environ.get(name) for name in (
            "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS",
            "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TOKENIZERS_PARALLELISM")},
    }


def collect_inputs(args, crab_root):
    """Fingerprint content, not only CLI paths; local HF snapshots, no network."""
    from huggingface_hub import snapshot_download
    crab_root = Path(crab_root)
    bases = {}
    for key in ("ssl_type", "text_model_path"):
        value = getattr(args, key)
        p = Path(value).expanduser()
        bases[key] = str(p.resolve() if p.is_dir()
                         else Path(snapshot_download(value, local_files_only=True)).resolve())
    paths = set(crab_root.joinpath("src").rglob("*.py"))
    paths.update(crab_root/"bin"/n for n in
                 ("train_crab_queue.py", "crab_epoch_checkpoint.py", "run_crab_queue_job.py"))
    paths.add(Path(args.df_path).resolve())
    if args.experiment_metadata:
        paths.add(Path(args.experiment_metadata).resolve())
    for base in bases.values():
        paths.update(p for p in Path(base).rglob("*") if p.is_file())
    warm = Path(args.pre_trained_path).resolve()
    warm_entries = {}
    for name in ("final_ssl.pt", "final_text.pt", "final_ser.pt",
                 "text_lora_adapter", "audio_lora_adapter"):
        p = warm/name
        warm_entries[str(p)] = p.exists()
        if p.is_file():
            paths.add(p)
        elif p.is_dir():
            paths.update(f for f in p.rglob("*") if f.is_file())
    weights = Path(args.weights_json).resolve()
    if weights.exists():
        paths.add(weights)
    with Path(args.df_path).open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row["Split_Set"] in ("Train", "Development"):
            p = Path(row["FileName"])
            paths.add(p if p.is_absolute() else (Path(args.wav_base_dir)/p).resolve())
    config = {k: v for k,v in vars(args).items() if k not in ("resume", "model_path")}
    versions = {n: importlib.metadata.version(n) for n in
                ("torch", "numpy", "pandas", "transformers", "peft", "safetensors", "wandb", "librosa")}
    content = {"config": config, "bases": bases, "warm_start_entries": warm_entries,
               "weights_exists": weights.exists(), "python": sys.version, "packages": versions,
               "backend": backend_settings(),
               "files": {str(p): digest(p) for p in sorted(paths)}}
    return content, json_digest(content), bases


def init_wandb(module, ctx, fingerprint, args, config, restored=None):
    """Explicit id/resume overrides legacy WANDB_RESUME=never; never logs credentials."""
    mode = os.environ.get("WANDB_MODE", "online")
    require(mode in ("online", "disabled"), "Offline W&B resume is not qualified; use online or disabled")
    run_id = "tq-" + hashlib.sha256(ctx.job_id.encode()).hexdigest()[:20]
    expected = {"job_id": ctx.job_id, "spec_sha256": ctx.spec_sha256,
                "fingerprint": fingerprint, "id": run_id, "mode": mode,
                "project": args.project_name, "entity": args.wandb_entity}
    marker = ctx.output/"crab_job_identity.json"
    if restored:
        require(read_json(marker) == expected, "Job/W&B output identity mismatch")
        require(all(restored["wandb"].get(k) == expected[k]
                    for k in ("id", "mode", "project", "entity")), "W&B identity changed")
    else:
        require(not marker.exists(), "Existing logical job requires a resume manifest")
        atomic_json(marker, expected)
    inherited_id = os.environ.get("WANDB_RUN_ID")
    require(not inherited_id or inherited_id == run_id, "Conflicting inherited W&B ID")
    run = module.init(id=run_id, resume="must" if restored else "never", mode=mode,
                      project=args.project_name, entity=args.wandb_entity, group=args.wandb_group,
                      name=args.run_name or ctx.job_id, config=config, dir=str(ctx.output))
    if mode != "disabled":
        require(run.id == run_id, "W&B returned a different logical run")
        if restored:
            require(run.step <= restored["wandb"]["next_step"],
                    "W&B history is ahead of checkpoint; cannot silently rewind logging")
    return {k: expected[k] for k in ("id", "mode", "project", "entity")}


def require_disk_budget(models, optimizers, remaining_epochs, checkpoint_dir):
    model_bytes = sum(t.numel()*t.element_size() for m in models.values() for t in m.state_dict().values())
    trainable = sum(p.numel() for o in optimizers.values() for g in o.param_groups for p in g["params"])
    # FP32 AdamW moment tensors (8 bytes/parameter), metadata/slack; no automatic retention.
    per_epoch = int((model_bytes + trainable*8)*1.20) + 64*1024**2
    required = per_epoch*remaining_epochs + 20*GIB
    free = shutil.disk_usage(checkpoint_dir).free
    require(free >= required, f"Insufficient disk for all remaining snapshots: need {required/GIB:.2f} GiB incl reserve, free {free/GIB:.2f}")
    return {"estimated_snapshot_bytes": per_epoch, "remaining_epochs": remaining_epochs,
            "required_with_reserve_bytes": required, "available_bytes": free}
