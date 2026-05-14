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
import pickle
import logging

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger("crab_api.inference")

CLASSES = ["Excited", "Unconfident", "Neutral_3Class"]
TARGET_SR = 16000
MAX_AUDIO_SEC = 12
MAX_AUDIO_LEN = MAX_AUDIO_SEC * TARGET_SR   # 192000 — matches training collate_fn
MIN_AUDIO_SEC = 0.5                          # relaxed from 3s for real-time short windows
TEXT_MAX_LEN = 128                           # matches training default


class CrabEmotionPredictor:
    """Loads the three model components and runs inference."""

    def __init__(
        self,
        model_dir: str,
        ssl_type: str = "microsoft/wavlm-large",
        text_model_path: str = "roberta-large",
        fusion_hidden_dim: int = 512,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        self.ssl_type = ssl_type
        self.text_model_path = text_model_path
        self.fusion_hidden_dim = fusion_hidden_dim

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
        norm_path = os.path.join(self.model_dir, "train_norm_stat.pkl")
        if os.path.exists(norm_path):
            with open(norm_path, "rb") as f:
                self.wav_mean, self.wav_std = pickle.load(f)
            
            # Sanity check: if std is suspiciously small, fallback to 1.0 to avoid explosion
            if self.wav_std < 1e-5:
                logger.error(f"wav_std={self.wav_std} is suspiciously small, resetting to 1.0")
                self.wav_std = 1.0
                
            logger.info(f"Loaded norm stats: mean={self.wav_mean:.6f}, std={self.wav_std:.6f}")
        else:
            logger.warning(
                f"Norm stats not found at {norm_path}. "
                "Using identity normalisation (mean=0, std=1). "
                "This WILL degrade accuracy."
            )

    def _load_models(self):
        """Load SSL encoder, text encoder, and classification head."""

        # 1. WavLM-Large SSL audio encoder
        logger.info(f"Loading SSL model: {self.ssl_type}")
        self.ssl_model = AutoModel.from_pretrained(self.ssl_type)
        ssl_weights = os.path.join(self.model_dir, "final_ssl.pt")
        if os.path.exists(ssl_weights):
            state = torch.load(ssl_weights, map_location="cpu")
            self.ssl_model.load_state_dict(state)
            logger.info("Loaded fine-tuned SSL weights.")
        self.ssl_model.to(self.device).eval()

        audio_feat_dim = self.ssl_model.config.hidden_size  # 1024 for WavLM-Large

        # 2. RoBERTa-Large text encoder + tokenizer
        logger.info(f"Loading text model: {self.text_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_path)
        self.text_model = AutoModel.from_pretrained(self.text_model_path)
        text_weights = os.path.join(self.model_dir, "final_text.pt")
        if os.path.exists(text_weights):
            state = torch.load(text_weights, map_location="cpu")
            self.text_model.load_state_dict(state)
            logger.info("Loaded fine-tuned text weights.")
        self.text_model.to(self.device).eval()

        text_feat_dim = self.text_model.config.hidden_size  # 1024 for RoBERTa-Large

        # 3. Classification head — use EXACT same constructor kwargs as training
        #    (see bin/train_crab.py L430-436)
        from src.models.ser import MultiModalEmotionClassifierDeep

        self.ser_model = MultiModalEmotionClassifierDeep(
            features1_dim=audio_feat_dim,
            features2_dim=text_feat_dim,
            fusion_hidden_dim=self.fusion_hidden_dim,
            num_emotions=len(CLASSES),
            dropout=0.5,
        )
        ser_weights = os.path.join(self.model_dir, "final_ser.pt")
        state = torch.load(ser_weights, map_location="cpu")
        self.ser_model.load_state_dict(state)
        self.ser_model.to(self.device).eval()

        logger.info(
            f"All models loaded. "
            f"audio_dim={audio_feat_dim}, text_dim={text_feat_dim}, "
            f"fusion_dim={self.fusion_hidden_dim}, classes={CLASSES}"
        )

    # ───────────────────────────── preprocessing ─────────────────────────────

    def _preprocess_audio(self, audio_bytes: bytes) -> tuple[torch.Tensor, int]:
        """Convert raw audio bytes → normalised mono 16 kHz tensor.

        Returns (waveform_1d, num_valid_samples).
        The normalisation (x - mean) / std replicates LazyWavSet.__getitem__.
        """
        try:
            waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
        except Exception as e:
            raise ValueError(f"Invalid audio file format: {str(e)}")
            
        if sr != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        # Mono
        waveform = waveform.mean(dim=0)  # [T]
        # Truncate to training max (12 seconds = 192000 samples)
        waveform = waveform[:MAX_AUDIO_LEN]
        num_valid = waveform.shape[0]
        # Normalise — matches training pipeline exactly
        waveform = (waveform - self.wav_mean) / (self.wav_std + 1e-6)
        return waveform.float(), num_valid

    def _tokenize_single(self, text: str) -> dict:
        """Tokenize a single text string for RoBERTa."""
        return self.tokenizer(
            text,
            return_tensors="pt",
            max_length=TEXT_MAX_LEN,
            padding="max_length",
            truncation=True,
        )

    # ───────────────────────────── single predict ────────────────────────────

    @torch.no_grad()
    def predict_single(self, audio_bytes: bytes, text: str = "") -> dict:
        """Run inference on a single (audio, text) pair.

        Returns a dict mapping class names to softmax probabilities.
        """
        wav, _ = self._preprocess_audio(audio_bytes)
        wav = wav.unsqueeze(0).to(self.device)  # [1, T]
        audio_mask = torch.ones(1, wav.shape[1], dtype=torch.long).to(self.device)

        tok = self._tokenize_single(text)
        input_ids = tok["input_ids"].to(self.device)
        text_mask = tok["attention_mask"].to(self.device)

        # Extract features
        audio_feat = self.ssl_model(
            wav, attention_mask=audio_mask
        ).last_hidden_state                                       # [1, T', 1024]
        text_feat = self.text_model(
            input_ids=input_ids, attention_mask=text_mask
        ).last_hidden_state                                       # [1, L, 1024]

        # Classify
        logits = self.ser_model(audio_feat, text_feat)            # [1, num_classes]
        probs = F.softmax(logits, dim=1)[0].cpu().tolist()

        return dict(zip(CLASSES, probs))

    # ───────────────────────────── batch predict ─────────────────────────────

    @torch.no_grad()
    def predict_batch(
        self,
        audio_bytes_list: list[bytes],
        texts: list[str],
    ) -> list[dict]:
        """GPU-parallel batch inference.

        All audio clips are padded to the longest clip in the batch.
        An attention_mask is constructed so WavLM ignores zero-padded regions
        (without this, WavLM silently produces wrong features — the worst
        kind of bug because there is no error, just degraded accuracy).
        """
        batch_size = len(audio_bytes_list)

        # 1. Pre-process all audio → list of (waveform_1d, valid_len)
        processed = [self._preprocess_audio(ab) for ab in audio_bytes_list]
        wavs = [p[0] for p in processed]
        valid_lens = [p[1] for p in processed]
        max_len = max(valid_lens)

        # 2. Pad audio and build attention mask
        #    Exactly replicates collate_fn_bimodal (train_crab.py L263-268)
        padded_wav = torch.zeros(batch_size, max_len)
        audio_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        for i, (wav, vlen) in enumerate(zip(wavs, valid_lens)):
            padded_wav[i, :vlen] = wav[:vlen]
            audio_mask[i, :vlen] = 1

        padded_wav = padded_wav.to(self.device)
        audio_mask = audio_mask.to(self.device)

        # 3. Batch tokenize text
        tok = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=TEXT_MAX_LEN,
            padding="max_length",
            truncation=True,
        )
        input_ids = tok["input_ids"].to(self.device)
        text_mask = tok["attention_mask"].to(self.device)

        # 4. Extract features with proper masks
        #    ⚠️ CRITICAL: passing attention_mask to WavLM ensures padded
        #    regions are ignored during self-attention computation.
        audio_feat = self.ssl_model(
            padded_wav, attention_mask=audio_mask
        ).last_hidden_state                                       # [B, T', 1024]
        text_feat = self.text_model(
            input_ids=input_ids, attention_mask=text_mask
        ).last_hidden_state                                       # [B, L, 1024]

        # 5. Classify
        logits = self.ser_model(audio_feat, text_feat)            # [B, num_classes]
        probs = F.softmax(logits, dim=1).cpu().tolist()

        return [dict(zip(CLASSES, p)) for p in probs]

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
        """Segment long audio into sliding windows, batch-infer, and merge.

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
        except Exception as e:
            raise ValueError(f"Invalid audio file format: {str(e)}")

        if sr != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        waveform = waveform.mean(dim=0)  # mono [T]

        total_samples = waveform.shape[0]
        audio_duration_sec = total_samples / TARGET_SR

        window_samples = int(window_sec * TARGET_SR)
        stride_samples = int(stride_sec * TARGET_SR)
        min_samples = int(MIN_AUDIO_SEC * TARGET_SR)

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
            raise ValueError(
                f"Audio too short ({audio_duration_sec:.1f}s). "
                f"Minimum is {MIN_AUDIO_SEC}s."
            )

        # If audio fits in a single window, just use predict_single path
        if len(windows) == 1:
            w = windows[0]
            wav_tensor = w["waveform"].unsqueeze(0).to(self.device)
            audio_mask = torch.ones(1, wav_tensor.shape[1], dtype=torch.long, device=self.device)

            tok = self._tokenize_single(text)
            input_ids = tok["input_ids"].to(self.device)
            text_mask = tok["attention_mask"].to(self.device)

            audio_feat = self.ssl_model(wav_tensor, attention_mask=audio_mask).last_hidden_state
            text_feat = self.text_model(input_ids=input_ids, attention_mask=text_mask).last_hidden_state
            logits = self.ser_model(audio_feat, text_feat)
            probs = F.softmax(logits, dim=1)[0].cpu().tolist()
            prob_dict = dict(zip(CLASSES, probs))
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

        # 3. Batch inference (chunked by max_batch)
        all_probs = []
        for batch_start in range(0, len(windows), max_batch):
            batch_windows = windows[batch_start:batch_start + max_batch]
            wavs = [w["waveform"] for w in batch_windows]
            valid_lens = [w["waveform"].shape[0] for w in batch_windows]
            max_len = max(valid_lens)
            bs = len(batch_windows)

            padded_wav = torch.zeros(bs, max_len)
            audio_mask = torch.zeros(bs, max_len, dtype=torch.long)
            for i, (wav, vlen) in enumerate(zip(wavs, valid_lens)):
                padded_wav[i, :vlen] = wav[:vlen]
                audio_mask[i, :vlen] = 1

            padded_wav = padded_wav.to(self.device)
            audio_mask = audio_mask.to(self.device)

            # Tokenize text (same text for all windows)
            texts_batch = [text] * bs
            tok = self.tokenizer(
                texts_batch,
                return_tensors="pt",
                max_length=TEXT_MAX_LEN,
                padding="max_length",
                truncation=True,
            )
            input_ids = tok["input_ids"].to(self.device)
            text_mask = tok["attention_mask"].to(self.device)

            audio_feat = self.ssl_model(padded_wav, attention_mask=audio_mask).last_hidden_state
            text_feat = self.text_model(input_ids=input_ids, attention_mask=text_mask).last_hidden_state
            logits = self.ser_model(audio_feat, text_feat)
            probs = F.softmax(logits, dim=1).cpu().tolist()
            all_probs.extend(probs)

        # 4. Build timeline and compute average probabilities
        timeline = []
        avg_probs = {c: 0.0 for c in CLASSES}

        for i, (w, probs) in enumerate(zip(windows, all_probs)):
            prob_dict = dict(zip(CLASSES, probs))
            label = max(prob_dict, key=prob_dict.get)
            timeline.append({
                "window": f"{w['start_sec']:.1f}~{w['end_sec']:.1f}s",
                "label": label,
                "confidence": round(prob_dict[label], 4),
                "probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
            })
            for c in CLASSES:
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
