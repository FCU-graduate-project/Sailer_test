"""Validated model metadata and artifact contracts for Crab inference.

This module intentionally has no torch/transformers imports so artifact
validation can run quickly before allocating GPU memory.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import pickle
import re
from typing import Any


INTERVIEW_CLASSES = ("Excited", "Unconfident", "Neutral_3Class")
N3_MODEL_ID = "n3-v1-warmstart-nnime-lora-stage2"
SCHEME1_MODEL_ID = "scheme1-wavlm-roberta-fullft"

_KNOWN_MODEL_IDS = {
    "nnime_v1_warmstart_lora": N3_MODEL_ID,
    "interview_scheme1": SCHEME1_MODEL_ID,
}

_N3_REQUIRED_CHECKSUMS = {
    "mode.json",
    "train_norm_stat.pkl",
    "final_ser.pt",
    "audio_lora_adapter/adapter_config.json",
    "audio_lora_adapter/adapter_model.safetensors",
    "text_lora_adapter/adapter_config.json",
    "text_lora_adapter/adapter_model.safetensors",
}


class ModelArtifactError(RuntimeError):
    """Raised when model metadata or checkpoint artifacts are inconsistent."""


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise ModelArtifactError(f"Missing {label}: {path}")
    if path.stat().st_size <= 0:
        raise ModelArtifactError(f"Empty {label}: {path}")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelArtifactError(f"Invalid {label} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ModelArtifactError(f"{label} must contain a JSON object: {path}")
    return value


def _required_string(data: dict[str, Any], key: str, source: Path) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ModelArtifactError(f"{source}: '{key}' must be a non-empty string")
    return value


def _stable_fallback_id(directory_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", directory_name.lower()).strip("-")
    return f"crab-{slug or 'model'}"


def _verify_checksum_manifest(
    directory: Path,
    *,
    required_entries: set[str] | None = None,
    manifest_path: Path | None = None,
) -> None:
    """Verify a sha256sum-compatible manifest without following outside paths."""
    manifest = _require_file(
        manifest_path or directory / "artifact_manifest.sha256",
        "checksum manifest",
    )
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            expected, relative = line.split(maxsplit=1)
        except ValueError as exc:
            raise ModelArtifactError(
                f"{manifest}:{line_number}: expected '<sha256>  <relative path>'"
            ) from exc
        relative = relative.lstrip("*").strip()
        if not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
            raise ModelArtifactError(
                f"{manifest}:{line_number}: invalid SHA-256 digest"
            )
        path = (directory / relative).resolve()
        try:
            path.relative_to(directory)
        except ValueError as exc:
            raise ModelArtifactError(
                f"{manifest}:{line_number}: checksum path escapes model directory"
            ) from exc
        _require_file(path, f"checksummed artifact '{relative}'")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest().lower() != expected.lower():
            raise ModelArtifactError(f"Checksum mismatch for {path}")
        seen.add(relative)
    if required_entries:
        missing = required_entries - seen
        if missing:
            raise ModelArtifactError(
                f"{manifest}: missing required checksum entries {sorted(missing)!r}"
            )


@dataclass(frozen=True)
class ModelProfile:
    """Fully validated runtime description of one Crab checkpoint."""

    model_id: str
    model_dir: Path
    ft_mode: str
    ssl_type: str
    text_model_path: str
    classes: tuple[str, ...]
    fusion_hidden_dim: int
    freeze_audio: bool
    lora_rank: int | None
    lora_alpha: int | None
    metadata_source: str

    @property
    def mode_path(self) -> Path:
        return self.model_dir / "mode.json"

    @property
    def norm_path(self) -> Path:
        return self.model_dir / "train_norm_stat.pkl"

    @property
    def ser_path(self) -> Path:
        return self.model_dir / "final_ser.pt"

    @property
    def audio_adapter_dir(self) -> Path:
        return self.model_dir / "audio_lora_adapter"

    @property
    def text_adapter_dir(self) -> Path:
        return self.model_dir / "text_lora_adapter"

    @property
    def ssl_weights_path(self) -> Path:
        return self.model_dir / "final_ssl.pt"

    @property
    def text_weights_path(self) -> Path:
        return self.model_dir / "final_text.pt"

    def load_norm_stats(self) -> tuple[float, float]:
        """Load and validate the exact waveform statistics used for training."""
        _require_file(self.norm_path, "normalization statistics")
        try:
            with self.norm_path.open("rb") as handle:
                value = pickle.load(handle)
        except Exception as exc:  # trusted local training artifact
            raise ModelArtifactError(
                f"Could not load normalization statistics {self.norm_path}: {exc}"
            ) from exc
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ModelArtifactError(
                f"Normalization statistics must be a (mean, std) pair: {self.norm_path}"
            )
        try:
            mean, std = float(value[0]), float(value[1])
        except (TypeError, ValueError) as exc:
            raise ModelArtifactError(
                f"Normalization mean/std must be numeric: {self.norm_path}"
            ) from exc
        if not math.isfinite(mean) or not math.isfinite(std) or std < 1e-5:
            raise ModelArtifactError(
                f"Invalid normalization statistics mean={mean!r}, std={std!r}: "
                f"{self.norm_path}"
            )
        return mean, std

    def artifact_summary(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "ft_mode": self.ft_mode,
            "ssl_type": self.ssl_type,
            "text_model_path": self.text_model_path,
            "classes": list(self.classes),
            "fusion_hidden_dim": self.fusion_hidden_dim,
            "metadata_source": self.metadata_source,
        }


def _validate_adapter(
    adapter_dir: Path,
    *,
    label: str,
    expected_base_model: str,
    expected_rank: int | None,
    expected_alpha: int | None,
    expected_targets: set[str] | None,
) -> None:
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    config = _read_json(config_path, f"{label} adapter config")
    _require_file(weights_path, f"{label} adapter weights")

    if config.get("peft_type") != "LORA":
        raise ModelArtifactError(
            f"{config_path}: expected peft_type='LORA', got {config.get('peft_type')!r}"
        )
    if config.get("base_model_name_or_path") != expected_base_model:
        raise ModelArtifactError(
            f"{config_path}: base model mismatch; expected {expected_base_model!r}, "
            f"got {config.get('base_model_name_or_path')!r}"
        )
    if expected_rank is not None and config.get("r") != expected_rank:
        raise ModelArtifactError(
            f"{config_path}: LoRA rank mismatch; expected {expected_rank}, "
            f"got {config.get('r')!r}"
        )
    if expected_alpha is not None and config.get("lora_alpha") != expected_alpha:
        raise ModelArtifactError(
            f"{config_path}: LoRA alpha mismatch; expected {expected_alpha}, "
            f"got {config.get('lora_alpha')!r}"
        )
    if expected_targets is not None:
        actual_targets = set(config.get("target_modules") or [])
        if actual_targets != expected_targets:
            raise ModelArtifactError(
                f"{config_path}: target_modules mismatch; expected "
                f"{sorted(expected_targets)!r}, got {sorted(actual_targets)!r}"
            )


def load_model_profile(
    model_dir: str | Path,
    *,
    requested_model_id: str | None = None,
    legacy_ssl_type: str = "microsoft/wavlm-large",
    legacy_text_model_path: str = "roberta-large",
    legacy_fusion_hidden_dim: int = 512,
) -> ModelProfile:
    """Read metadata and fail fast on every required checkpoint artifact.

    New LoRA checkpoints must contain ``mode.json``. The only metadata-less
    layout retained for backwards compatibility is a complete Full-FT
    checkpoint (the original English scheme1 layout).
    """
    directory = Path(model_dir).expanduser().resolve()
    if not directory.is_dir():
        raise ModelArtifactError(f"Model directory does not exist: {directory}")

    mode_path = directory / "mode.json"
    inferred_id = _KNOWN_MODEL_IDS.get(directory.name, _stable_fallback_id(directory.name))

    if mode_path.is_file():
        mode = _read_json(mode_path, "mode metadata")
        metadata_model_id = mode.get("model_id", inferred_id)
        if not isinstance(metadata_model_id, str) or not metadata_model_id.strip():
            raise ModelArtifactError(f"{mode_path}: 'model_id' must be a non-empty string")
        model_id = metadata_model_id
        ft_mode = _required_string(mode, "ft_mode", mode_path)
        ssl_type = _required_string(mode, "ssl_type", mode_path)
        text_model_path = _required_string(mode, "text_model_path", mode_path)
        raw_classes = mode.get("classes")
        if not isinstance(raw_classes, list) or not all(
            isinstance(item, str) and item for item in raw_classes
        ):
            raise ModelArtifactError(f"{mode_path}: 'classes' must be a list of strings")
        classes = tuple(raw_classes)
        fusion_hidden_dim = mode.get("fusion_hidden_dim")
        if not isinstance(fusion_hidden_dim, int) or fusion_hidden_dim <= 0:
            raise ModelArtifactError(
                f"{mode_path}: 'fusion_hidden_dim' must be a positive integer"
            )
        freeze_audio = mode.get("freeze_audio", False)
        if not isinstance(freeze_audio, bool):
            raise ModelArtifactError(f"{mode_path}: 'freeze_audio' must be boolean")
        lora_rank = mode.get("lora_rank")
        lora_alpha = mode.get("lora_alpha")
        metadata_source = "mode.json"
    else:
        if directory.name == "nnime_v1_warmstart_lora":
            raise ModelArtifactError(f"N3 checkpoint requires mode.json: {mode_path}")
        ft_mode = "full_ft"
        ssl_type = legacy_ssl_type
        text_model_path = legacy_text_model_path
        classes = INTERVIEW_CLASSES
        fusion_hidden_dim = legacy_fusion_hidden_dim
        freeze_audio = False
        lora_rank = None
        lora_alpha = None
        metadata_source = "legacy-full-ft-layout"
        model_id = inferred_id

    if requested_model_id is not None and requested_model_id != model_id:
        raise ModelArtifactError(
            f"model_id override {requested_model_id!r} does not match checkpoint "
            f"identity {model_id!r}"
        )

    if ft_mode not in {"lora", "full_ft", "partial_ft"}:
        raise ModelArtifactError(f"Unsupported ft_mode={ft_mode!r} in {mode_path}")
    if ft_mode == "lora":
        if (
            not isinstance(lora_rank, int)
            or isinstance(lora_rank, bool)
            or lora_rank <= 0
        ):
            raise ModelArtifactError(
                f"{mode_path}: 'lora_rank' must be a positive integer"
            )
        if (
            not isinstance(lora_alpha, int)
            or isinstance(lora_alpha, bool)
            or lora_alpha <= 0
        ):
            raise ModelArtifactError(
                f"{mode_path}: 'lora_alpha' must be a positive integer"
            )
    if classes != INTERVIEW_CLASSES:
        raise ModelArtifactError(
            f"Class order mismatch; expected {list(INTERVIEW_CLASSES)!r}, "
            f"got {list(classes)!r}"
        )

    profile = ModelProfile(
        model_id=model_id,
        model_dir=directory,
        ft_mode=ft_mode,
        ssl_type=ssl_type,
        text_model_path=text_model_path,
        classes=classes,
        fusion_hidden_dim=fusion_hidden_dim,
        freeze_audio=freeze_audio,
        lora_rank=lora_rank if isinstance(lora_rank, int) else None,
        lora_alpha=lora_alpha if isinstance(lora_alpha, int) else None,
        metadata_source=metadata_source,
    )

    _require_file(profile.ser_path, "SER head")
    n3_strict = (
        profile.ft_mode == "lora"
        and profile.ssl_type == "facebook/wav2vec2-xls-r-300m"
        and profile.text_model_path == "FacebookAI/xlm-roberta-large"
    )
    if n3_strict:
        # The packaged, version-controlled manifest is the trust anchor.
        # A manifest beside ignored model artifacts is intentionally not used:
        # an accidental/malicious replacement could otherwise bless itself.
        packaged_manifest = (
            Path(__file__).resolve().parent
            / "model_manifests"
            / f"{profile.model_id}.sha256"
        )
        _verify_checksum_manifest(
            profile.model_dir,
            required_entries=_N3_REQUIRED_CHECKSUMS,
            manifest_path=packaged_manifest,
        )
    # The pickle is only opened after its digest has been authenticated for N3.
    profile.load_norm_stats()

    if profile.ft_mode == "lora":
        if profile.freeze_audio:
            raise ModelArtifactError(
                "This API requires bimodal LoRA checkpoints with both audio and text adapters"
            )
        _validate_adapter(
            profile.audio_adapter_dir,
            label="audio",
            expected_base_model=profile.ssl_type,
            expected_rank=profile.lora_rank,
            expected_alpha=profile.lora_alpha,
            expected_targets={"q_proj", "v_proj"} if n3_strict else None,
        )
        _validate_adapter(
            profile.text_adapter_dir,
            label="text",
            expected_base_model=profile.text_model_path,
            expected_rank=profile.lora_rank,
            expected_alpha=profile.lora_alpha,
            expected_targets={"query", "value"} if n3_strict else None,
        )
    else:
        _require_file(profile.ssl_weights_path, "fine-tuned audio encoder weights")
        _require_file(profile.text_weights_path, "fine-tuned text encoder weights")

    return profile
