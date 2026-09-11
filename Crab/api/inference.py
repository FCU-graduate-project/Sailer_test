"""
CrabEmotionPredictor — Inference engine for the Crab Bimodal Emotion Classifier.

Critical implementation notes (bugs fixed from the plan):
─────────────────────────────────────────────────────────
1. Audio normalization: Training normalises raw waveforms with per-dataset
   mean/std stored in train_norm_stat.pkl. Skipping this makes the SSL
   encoder see a completely different input distribution → wrong predictions.
2. MultiModalEmotionClassifierDeep constructor: The plan used wrong kwargs
   (audio_dim, text_dim, num_classes, head_dim). Actual signature is
   (features1_dim, features2_dim, num_emotions, fusion_hidden_dim, dropout).
3. WavLM attention_mask in batch mode: Without it, WavLM's self-attention
   attends to zero-padded regions and *silently* produces degraded features.
   This is the most insidious bug — no error, just quietly wrong outputs.
4. RoBERTa attention_mask: Must be forwarded so pad tokens don't leak.
5. Import path: Plan had `src.model.ser_model`; actual is `src.models.ser`.
6. Audio truncation: Training truncates at 12s (192000 samples), not 15s.
"""

import io
import os
import logging


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


# Default to deterministic offline startup. Set these before importing
# transformers so huggingface_hub also observes offline mode.
DEFAULT_HF_OFFLINE = _env_flag("CRAB_HF_OFFLINE", True)
if DEFAULT_HF_OFFLINE:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn.functional as F
import torchaudio
from transformers import AutoModel, AutoTokenizer

from .model_profile import ModelArtifactError, ModelProfile, load_model_profile
from .config import SETTINGS

logger = logging.getLogger("crab_api.inference")

CLASSES = ["Excited", "Unconfident", "Neutral_3Class"]
TARGET_SR = 16000
MAX_AUDIO_SEC = 12
MAX_AUDIO_LEN = MAX_AUDIO_SEC * TARGET_SR   # 192000 — matches training collate_fn
MIN_AUDIO_SEC = 0.5                          # relaxed from 3s for real-time short windows
TEXT_MAX_LEN = 128                           # matches training default


class InferenceInputError(ValueError):
    """A safe, client-correctable inference input error."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.public_message = message


class CrabEmotionPredictor:
    """Loads the three model components and runs inference."""

    def __init__(
        self,
        model_dir: str,
        ssl_type: str | None = None,
        text_model_path: str | None = None,
        fusion_hidden_dim: int | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        offline: bool | None = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.offline = DEFAULT_HF_OFFLINE if offline is None else offline
        self.profile: ModelProfile = load_model_profile(
            model_dir,
            requested_model_id=model_id,
            legacy_ssl_type=ssl_type or "microsoft/wavlm-large",
            legacy_text_model_path=text_model_path or "roberta-large",
            legacy_fusion_hidden_dim=fusion_hidden_dim or 512,
        )

        # mode.json is authoritative. Explicit overrides may only confirm it.
        if ssl_type is not None and ssl_type != self.profile.ssl_type:
            raise ModelArtifactError(
                f"ssl_type override {ssl_type!r} does not match mode.json "
                f"value {self.profile.ssl_type!r}"
            )
        if text_model_path is not None and text_model_path != self.profile.text_model_path:
            raise ModelArtifactError(
                f"text_model_path override {text_model_path!r} does not match "
                f"mode.json value {self.profile.text_model_path!r}"
            )
        if fusion_hidden_dim is not None and fusion_hidden_dim != self.profile.fusion_hidden_dim:
            raise ModelArtifactError(
                f"fusion_hidden_dim override {fusion_hidden_dim!r} does not match "
                f"mode.json value {self.profile.fusion_hidden_dim!r}"
            )

        self.model_dir = str(self.profile.model_dir)
        self.model_id = self.profile.model_id
        self.ft_mode = self.profile.ft_mode
        self.ssl_type = self.profile.ssl_type
        self.text_model_path = self.profile.text_model_path
        self.fusion_hidden_dim = self.profile.fusion_hidden_dim
        self.classes = list(self.profile.classes)

        # Normalization stats (loaded from training)
        self.wav_mean: float = 0.0
        self.wav_std: float = 1.0

        self._load_norm_stats()
        self._load_models()

    # ───────────────────────────── model loading ─────────────────────────────

    def _load_norm_stats(self):
        """Load the wav normalization stats saved during training.

        Without this, the SSL encoder sees un-normalised audio and produces
        features from a completely different distribution than what the
        classifier head was trained on.
        """
        self.wav_mean, self.wav_std = self.profile.load_norm_stats()
        logger.info(
            "Loaded norm stats: mean=%.6f, std=%.6f",
            self.wav_mean,
            self.wav_std,
        )

    def _load_models(self):
        """Load SSL encoder, text encoder, and classification head."""

        logger.info(
            "Loading model_id=%s ft_mode=%s offline=%s",
            self.model_id,
            self.ft_mode,
            self.offline,
        )

        # 1. SSL audio encoder. mode.json is authoritative for the base model.
        logger.info(f"Loading SSL model: {self.ssl_type}")
        audio_base = AutoModel.from_pretrained(
            self.ssl_type,
            local_files_only=self.offline,
        )

        # 2. Text encoder + tokenizer.
        logger.info(f"Loading text model: {self.text_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.text_model_path,
            local_files_only=self.offline,
        )
        text_base = AutoModel.from_pretrained(
            self.text_model_path,
            local_files_only=self.offline,
        )

        if self.ft_mode == "lora":
            from peft import PeftModel

            logger.info("Attaching audio LoRA: %s", self.profile.audio_adapter_dir)
            self.ssl_model = PeftModel.from_pretrained(
                audio_base,
                str(self.profile.audio_adapter_dir),
                is_trainable=False,
                local_files_only=self.offline,
            )
            logger.info("Attaching text LoRA: %s", self.profile.text_adapter_dir)
            self.text_model = PeftModel.from_pretrained(
                text_base,
                str(self.profile.text_adapter_dir),
                is_trainable=False,
                local_files_only=self.offline,
            )
            if not isinstance(self.ssl_model, PeftModel) or not isinstance(
                self.text_model, PeftModel
            ):
                raise ModelArtifactError("LoRA checkpoint did not produce PeftModel encoders")
        else:
            ssl_state = torch.load(self.profile.ssl_weights_path, map_location="cpu")
            audio_base.load_state_dict(ssl_state, strict=True)
            text_state = torch.load(self.profile.text_weights_path, map_location="cpu")
            text_base.load_state_dict(text_state, strict=True)
            self.ssl_model = audio_base
            self.text_model = text_base
            logger.info("Loaded strict Full-FT audio and text state dicts.")

        self.ssl_model.requires_grad_(False).to(self.device).eval()
        self.text_model.requires_grad_(False).to(self.device).eval()

        audio_feat_dim = self.ssl_model.config.hidden_size
        text_feat_dim = self.text_model.config.hidden_size

        # 3. Classification head — use EXACT same constructor kwargs as training
        #    (see bin/train_crab.py L430-436)
        if __package__ and __package__.startswith("Crab."):
            from ..src.models.ser import MultiModalEmotionClassifierDeep
        else:
            from src.models.ser import MultiModalEmotionClassifierDeep

        self.ser_model = MultiModalEmotionClassifierDeep(
            features1_dim=audio_feat_dim,
            features2_dim=text_feat_dim,
            fusion_hidden_dim=self.fusion_hidden_dim,
            num_emotions=len(self.classes),
            dropout=0.5,
        )
        state = torch.load(self.profile.ser_path, map_location="cpu")
        self.ser_model.load_state_dict(state, strict=True)
        self.ser_model.requires_grad_(False).to(self.device).eval()

        logger.info(
            f"All models loaded. "
            f"audio_dim={audio_feat_dim}, text_dim={text_feat_dim}, "
            f"fusion_dim={self.fusion_hidden_dim}, classes={self.classes}"
        )

    # ───────────────────────────── preprocessing ─────────────────────────────

    def _preprocess_audio(
        self,
        audio_bytes: bytes,
    ) -> tuple[torch.Tensor, int, float]:
        """Convert raw audio bytes → normalised mono 16 kHz tensor.

        Returns (waveform_1d, num_valid_samples, duration_seconds).
        The normalisation (x - mean) / std replicates LazyWavSet.__getitem__.
        """
        try:
            waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
        except Exception as exc:
            raise InferenceInputError(
                "INVALID_AUDIO",
                "Audio could not be decoded as a supported WAV, MP3, or FLAC file",
            ) from exc
            
        if sr != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        # Mono
        waveform = waveform.mean(dim=0)  # [T]
        num_valid = waveform.shape[0]
        if num_valid == 0:
            raise InferenceInputError("EMPTY_AUDIO", "Decoded audio contains no samples")
        duration_sec = num_valid / TARGET_SR
        if duration_sec > MAX_AUDIO_SEC + 1e-6:
            raise InferenceInputError(
                "AUDIO_TOO_LONG",
                f"Single and batch audio must be at most {MAX_AUDIO_SEC} seconds; "
                "use /v1/emotion/classify-long for longer audio",
            )
        if not bool(torch.isfinite(waveform).all()):
            raise InferenceInputError(
                "INVALID_AUDIO_SAMPLES",
                "Decoded audio contains non-finite samples",
            )
        # Normalise — matches training pipeline exactly
        waveform = (waveform - self.wav_mean) / (self.wav_std + 1e-6)
        return waveform.float(), num_valid, duration_sec

    def _validate_text(self, text: str) -> None:
        """Reject semantic truncation instead of silently dropping transcript tokens."""
        tokenized = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        input_ids = tokenized.get("input_ids", [])
        token_count = len(input_ids)
        if token_count > TEXT_MAX_LEN:
            raise InferenceInputError(
                "TEXT_TOO_LONG",
                f"Transcript is {token_count} tokens; maximum is {TEXT_MAX_LEN}",
            )

    def _tokenize_single(self, text: str) -> dict:
        """Tokenize a single text string for RoBERTa."""
        self._validate_text(text)
        return self.tokenizer(
            text,
            return_tensors="pt",
            max_length=TEXT_MAX_LEN,
            padding="max_length",
            truncation=True,
        )

    def _predict_normalized_waveform(
        self,
        waveform: torch.Tensor,
        text: str,
    ) -> dict[str, float]:
        """Run the exact batch-size-one forward used by N3 validation."""
        wav = waveform.unsqueeze(0).to(self.device)
        audio_mask = torch.ones(
            1,
            wav.shape[1],
            dtype=torch.long,
            device=self.device,
        )

        tok = self._tokenize_single(text)
        input_ids = tok["input_ids"].to(self.device)
        text_mask = tok["attention_mask"].to(self.device)

        audio_feat = self.ssl_model(
            wav,
            attention_mask=audio_mask,
        ).last_hidden_state
        text_feat = self.text_model(
            input_ids=input_ids,
            attention_mask=text_mask,
        ).last_hidden_state
        logits = self.ser_model(audio_feat, text_feat)
        probabilities = F.softmax(logits, dim=1)[0].cpu().tolist()
        return dict(zip(self.classes, probabilities))

    # ───────────────────────────── single predict ────────────────────────────

    @torch.no_grad()
    def predict_single(
        self,
        audio_bytes: bytes,
        text: str = "",
        *,
        return_metadata: bool = False,
    ) -> dict:
        """Run inference on a single (audio, text) pair.

        Returns a dict mapping class names to softmax probabilities.
        """
        wav, _, duration_sec = self._preprocess_audio(audio_bytes)
        probabilities = self._predict_normalized_waveform(wav, text)
        if return_metadata:
            return {
                "probabilities": probabilities,
                "audio_duration_sec": duration_sec,
            }
        return probabilities

    # ───────────────────────────── batch predict ─────────────────────────────

    @torch.no_grad()
    def predict_batch(
        self,
        audio_bytes_list: list[bytes],
        texts: list[str],
        *,
        return_metadata: bool = False,
    ) -> list[dict]:
        """Exact batch wrapper with single-item numerical parity.

        The trained SER head does not consume padding masks, and N3's reported
        test score used batch size 1. Sequential single-item forwards therefore
        preserve the validated model behavior until mask-aware pooling is
        implemented and re-evaluated.
        """
        if len(audio_bytes_list) != len(texts):
            raise ValueError("audio_bytes_list and texts must have the same length")
        if not audio_bytes_list:
            return []
        return [
            self.predict_single(
                audio_bytes,
                text,
                return_metadata=return_metadata,
            )
            for audio_bytes, text in zip(audio_bytes_list, texts)
        ]


    # ───────────────────────────── long-audio predict ──────────────────────────

    @torch.no_grad()
    def predict_long(
        self,
        audio_bytes: bytes,
        text: str = "",
        window_sec: float = 12.0,
        stride_sec: float = 6.0,
        max_batch: int = 16,
    ) -> dict:
        """Segment long audio into sliding windows, infer, and merge.

        Returns:
            {
                "final_label": str,
                "final_confidence": float,
                "avg_probabilities": {class: float, ...},
                "timeline": [
                    {"window": "0.0~12.0s", "label": str, "confidence": float,
                     "probabilities": {class: float, ...}},
                    ...
                ],
                "total_windows": int,
                "audio_duration_sec": float,
            }
        """
        # 1. Decode full audio to mono 16 kHz (NO truncation)
        try:
            waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
        except Exception as exc:
            raise InferenceInputError(
                "INVALID_AUDIO",
                "Audio could not be decoded as a supported WAV, MP3, or FLAC file",
            ) from exc

        if sr != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        waveform = waveform.mean(dim=0)  # mono [T]

        total_samples = waveform.shape[0]
        audio_duration_sec = total_samples / TARGET_SR
        if total_samples == 0:
            raise InferenceInputError("EMPTY_AUDIO", "Decoded audio contains no samples")
        if audio_duration_sec > SETTINGS.max_long_audio_sec + 1e-6:
            raise InferenceInputError(
                "AUDIO_TOO_LONG",
                f"Long audio must be at most {SETTINGS.max_long_audio_sec:g} seconds",
            )
        if not bool(torch.isfinite(waveform).all()):
            raise InferenceInputError(
                "INVALID_AUDIO_SAMPLES",
                "Decoded audio contains non-finite samples",
            )
        self._validate_text(text)

        window_samples = int(window_sec * TARGET_SR)
        stride_samples = int(stride_sec * TARGET_SR)
        min_samples = int(MIN_AUDIO_SEC * TARGET_SR)

        # Reject pathological overlap before materializing and normalizing
        # hundreds of tensors. Every accepted start must leave MIN_AUDIO_SEC.
        window_count = 1 + (total_samples - min_samples) // stride_samples
        if window_count > SETTINGS.max_long_windows:
            raise InferenceInputError(
                "TOO_MANY_WINDOWS",
                f"Long-audio request creates {window_count} windows; "
                f"maximum is {SETTINGS.max_long_windows}",
            )

        # 2. Slice into windows
        windows = []
        start = 0
        while start < total_samples:
            end = min(start + window_samples, total_samples)
            segment = waveform[start:end]
            # Skip segments shorter than MIN_AUDIO_SEC
            if segment.shape[0] >= min_samples:
                # Normalise (same as training)
                segment = (segment - self.wav_mean) / (self.wav_std + 1e-6)
                windows.append({
                    "waveform": segment.float(),
                    "start_sec": start / TARGET_SR,
                    "end_sec": end / TARGET_SR,
                })
            start += stride_samples

        if len(windows) == 0:
            raise InferenceInputError(
                "AUDIO_TOO_SHORT",
                f"Audio must be at least {MIN_AUDIO_SEC:g} seconds",
            )

        # If audio fits in a single window, use the exact single-item path.
        if len(windows) == 1:
            w = windows[0]
            prob_dict = self._predict_normalized_waveform(w["waveform"], text)
            label = max(prob_dict, key=prob_dict.get)

            return {
                "final_label": label,
                "final_confidence": round(prob_dict[label], 4),
                "avg_probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
                "timeline": [{
                    "window": f"{w['start_sec']:.1f}~{w['end_sec']:.1f}s",
                    "label": label,
                    "confidence": round(prob_dict[label], 4),
                    "probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
                }],
                "total_windows": 1,
                "audio_duration_sec": round(audio_duration_sec, 2),
            }

        # 3. Preserve the validated batch-size-one behavior for every window.
        all_probs = [
            [probabilities[name] for name in self.classes]
            for window in windows
            for probabilities in [
                self._predict_normalized_waveform(window["waveform"], text)
            ]
        ]

        # 4. Build timeline and compute average probabilities
        timeline = []
        avg_probs = {c: 0.0 for c in self.classes}

        for i, (w, probs) in enumerate(zip(windows, all_probs)):
            prob_dict = dict(zip(self.classes, probs))
            label = max(prob_dict, key=prob_dict.get)
            timeline.append({
                "window": f"{w['start_sec']:.1f}~{w['end_sec']:.1f}s",
                "label": label,
                "confidence": round(prob_dict[label], 4),
                "probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
            })
            for c in self.classes:
                avg_probs[c] += prob_dict[c]

        n = len(windows)
        avg_probs = {k: round(v / n, 4) for k, v in avg_probs.items()}
        final_label = max(avg_probs, key=avg_probs.get)

        return {
            "final_label": final_label,
            "final_confidence": round(avg_probs[final_label], 4),
            "avg_probabilities": avg_probs,
            "timeline": timeline,
            "total_windows": n,
            "audio_duration_sec": round(audio_duration_sec, 2),
        }

    def warmup(self):
        """Run a dummy forward pass to initialise CUDA kernels."""
        import wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(TARGET_SR)
            wf.writeframes(b"\x00\x00" * TARGET_SR)
        logger.info("Running dummy forward pass...")
        self.predict_single(buf.getvalue(), "warm up")
        logger.info("Dummy forward pass completed.")
