from __future__ import annotations

import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

from Crab.api.model_profile import (
    INTERVIEW_CLASSES,
    ModelArtifactError,
    N3_MODEL_ID,
    load_model_profile,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
N3_DIR = REPO_ROOT / "Crab" / "experiments" / "nnime_v1_warmstart_lora"


class N3ProfileTests(unittest.TestCase):
    def test_real_n3_profile_and_artifacts_are_valid(self) -> None:
        profile = load_model_profile(N3_DIR)
        self.assertEqual(profile.model_id, N3_MODEL_ID)
        self.assertEqual(profile.ft_mode, "lora")
        self.assertEqual(profile.ssl_type, "facebook/wav2vec2-xls-r-300m")
        self.assertEqual(profile.text_model_path, "FacebookAI/xlm-roberta-large")
        self.assertEqual(profile.classes, INTERVIEW_CLASSES)
        self.assertEqual(profile.fusion_hidden_dim, 512)
        mean, std = profile.load_norm_stats()
        self.assertLess(abs(mean), 1.0)
        self.assertGreater(std, 1e-5)

    def test_missing_artifact_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "mode.json").write_text(
                json.dumps(
                    {
                        "model_id": "test-lora-model",
                        "ft_mode": "lora",
                        "ssl_type": "test/audio-base",
                        "text_model_path": "test/text-base",
                        "freeze_audio": False,
                        "lora_rank": 16,
                        "lora_alpha": 32,
                        "classes": list(INTERVIEW_CLASSES),
                        "fusion_hidden_dim": 512,
                    }
                ),
                encoding="utf-8",
            )
            (model_dir / "final_ser.pt").write_bytes(b"not-empty")
            with (model_dir / "train_norm_stat.pkl").open("wb") as handle:
                pickle.dump((0.0, 1.0), handle)
            with self.assertRaisesRegex(
                ModelArtifactError,
                "Missing audio adapter config",
            ):
                load_model_profile(
                    model_dir,
                    requested_model_id="test-lora-model",
                )

    def test_checksum_is_verified_before_norm_pickle_is_opened(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "mode.json").write_text(
                json.dumps(
                    {
                        "model_id": N3_MODEL_ID,
                        "ft_mode": "lora",
                        "ssl_type": "facebook/wav2vec2-xls-r-300m",
                        "text_model_path": "FacebookAI/xlm-roberta-large",
                        "freeze_audio": False,
                        "lora_rank": 16,
                        "lora_alpha": 32,
                        "classes": list(INTERVIEW_CLASSES),
                        "fusion_hidden_dim": 512,
                    }
                ),
                encoding="utf-8",
            )
            (model_dir / "final_ser.pt").write_bytes(b"not-empty")
            with (model_dir / "train_norm_stat.pkl").open("wb") as handle:
                pickle.dump((0.0, 1.0), handle)
            with patch("Crab.api.model_profile.pickle.load") as pickle_load:
                with self.assertRaisesRegex(
                    ModelArtifactError,
                    "Checksum mismatch|checksummed artifact",
                ):
                    load_model_profile(model_dir, requested_model_id=N3_MODEL_ID)
                pickle_load.assert_not_called()

    def test_arbitrary_model_id_override_is_rejected(self) -> None:
        with self.assertRaisesRegex(ModelArtifactError, "does not match checkpoint identity"):
            load_model_profile(N3_DIR, requested_model_id="scheme1-wavlm-roberta-fullft")

    def test_bad_class_order_is_rejected(self) -> None:
        mode_path = N3_DIR / "mode.json"
        mode = json.loads(mode_path.read_text(encoding="utf-8"))
        mode["classes"] = list(reversed(mode["classes"]))
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "mode.json").write_text(json.dumps(mode), encoding="utf-8")
            with self.assertRaisesRegex(ModelArtifactError, "Class order mismatch"):
                load_model_profile(model_dir, requested_model_id=N3_MODEL_ID)

    def test_invalid_lora_metadata_is_rejected_before_model_load(self) -> None:
        mode = json.loads((N3_DIR / "mode.json").read_text(encoding="utf-8"))
        mode["lora_rank"] = 0
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "mode.json").write_text(json.dumps(mode), encoding="utf-8")
            with self.assertRaisesRegex(ModelArtifactError, "lora_rank"):
                load_model_profile(model_dir, requested_model_id=N3_MODEL_ID)

    def test_mode_override_mismatch_is_rejected_before_gpu_load(self) -> None:
        from Crab.api.inference import CrabEmotionPredictor

        with self.assertRaisesRegex(ModelArtifactError, "ssl_type override"):
            CrabEmotionPredictor(
                model_dir=str(N3_DIR),
                ssl_type="microsoft/wavlm-large",
                device="cpu",
            )

    def test_offline_missing_base_cache_fails_without_network_fallback(self) -> None:
        from Crab.api.inference import CrabEmotionPredictor

        with patch(
            "Crab.api.inference.AutoModel.from_pretrained",
            side_effect=OSError("not present in local cache"),
        ) as loader:
            with self.assertRaisesRegex(OSError, "local cache"):
                CrabEmotionPredictor(model_dir=str(N3_DIR), device="cpu")
        self.assertTrue(loader.call_args.kwargs["local_files_only"])


if __name__ == "__main__":
    unittest.main()
