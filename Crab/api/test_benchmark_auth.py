from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from .benchmark_api import ApiClient
from .model_profile import INTERVIEW_CLASSES, N3_MODEL_ID
from .soak_test import health_probe


TOKEN = "test-token-with-at-least-24-characters"


class BenchmarkAuthenticationTests(unittest.TestCase):
    def test_http_benchmark_client_sends_bearer_on_get_and_post(self) -> None:
        client = ApiClient(
            transport="http",
            base_url="https://n3.internal",
            timeout_sec=5.0,
            model_dir=Path("."),
            device="cpu",
            token=TOKEN,
        )
        session = Mock()
        with patch.object(client, "_session", return_value=session):
            client.get("/v1/health")
            client.post(
                "/v1/emotion/classify",
                [("audio", ("test.wav", b"data", "audio/wav"))],
            )
        expected = {"Authorization": f"Bearer {TOKEN}"}
        self.assertEqual(session.get.call_args.kwargs["headers"], expected)
        self.assertEqual(session.post.call_args.kwargs["headers"], expected)

    @patch("Crab.api.soak_test.gpu_sample", return_value=None)
    @patch("Crab.api.soak_test.requests.get")
    def test_soak_health_probe_sends_bearer(
        self,
        get: Mock,
        _gpu_sample: Mock,
    ) -> None:
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "status": "ok",
            "api_version": "1.3.0",
            "model": N3_MODEL_ID,
            "model_id": N3_MODEL_ID,
            "ft_mode": "lora",
            "ssl_type": "facebook/wav2vec2-xls-r-300m",
            "text_model_path": "FacebookAI/xlm-roberta-large",
            "hf_offline": True,
            "device": "cuda:0",
            "classes": list(INTERVIEW_CLASSES),
            "gpu_name": "test",
            "vram_mb": 10240,
            "queue_depth": 0,
            "queue_capacity": 8,
            "active_jobs": 0,
            "request_inflight": 0,
            "request_capacity": 9,
            "limits": {
                "max_single_file_bytes": 10485760,
                "max_batch_total_bytes": 67108864,
                "max_long_file_bytes": 104857600,
                "max_text_chars": 4096,
                "max_long_audio_sec": 300.0,
                "max_long_windows": 64,
                "max_batch_size": 16,
            },
        }
        get.return_value = response
        sample = health_probe(
            "https://n3.internal",
            timeout_sec=5.0,
            sample_local_gpu=False,
            token=TOKEN,
        )
        self.assertEqual(
            get.call_args.kwargs["headers"],
            {"Authorization": f"Bearer {TOKEN}"},
        )
        self.assertEqual(sample["api_version"], "1.3.0")
        self.assertEqual(sample["runtime_policy"]["request_capacity"], 9)


if __name__ == "__main__":
    unittest.main()
