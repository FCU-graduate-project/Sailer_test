"""Fast, GPU-free API contract and error-path tests.

These tests replace the model runtime with a deterministic fake predictor.
They exercise FastAPI multipart parsing, endpoint validation, response models,
stable metadata, and application-generated error envelopes without loading
XLS-R or XLM-R.

Run:
    Crab/.venv/bin/python -m unittest Crab.api.test_contract -v
"""

from __future__ import annotations

import asyncio
import threading
import unittest

from fastapi.testclient import TestClient

from . import app as app_module
from .model_profile import INTERVIEW_CLASSES, N3_MODEL_ID
from .inference import InferenceInputError
from .schemas import (
    BatchClassifyResponse,
    HealthResponse,
    LongClassifyResponse,
    SingleClassifyResponse,
)


PROBABILITIES = {
    "Excited": 0.6,
    "Unconfident": 0.25,
    "Neutral_3Class": 0.15,
}


def _validate(model, payload):
    """Support both Pydantic v1 and v2 in local/CI environments."""
    if hasattr(model, "model_validate"):
        return model.model_validate(payload)
    return model.parse_obj(payload)


class FakePredictor:
    model_id = N3_MODEL_ID
    ft_mode = "lora"
    ssl_type = "facebook/wav2vec2-xls-r-300m"
    text_model_path = "FacebookAI/xlm-roberta-large"
    offline = True
    device = "cuda:0"
    classes = list(INTERVIEW_CLASSES)

    def predict_single(
        self,
        audio_bytes: bytes,
        text: str = "",
        *,
        return_metadata: bool = False,
    ) -> dict:
        if audio_bytes == b"invalid":
            raise InferenceInputError("INVALID_AUDIO", "Audio could not be decoded")
        if audio_bytes == b"crash":
            raise RuntimeError("sensitive backend failure")
        if audio_bytes == b"oom":
            raise RuntimeError("CUDA out of memory: sensitive allocator details")
        if audio_bytes == b"malformed":
            return {"audio_duration_sec": 3.0}
        probabilities = dict(PROBABILITIES)
        if return_metadata:
            return {
                "probabilities": probabilities,
                "audio_duration_sec": 3.0,
            }
        return probabilities

    def predict_batch(
        self,
        audio_bytes_list: list[bytes],
        texts: list[str],
        *,
        return_metadata: bool = False,
    ) -> list[dict]:
        if audio_bytes_list and audio_bytes_list[0] == b"short_result":
            return []
        return [
            self.predict_single(
                audio_bytes,
                text,
                return_metadata=return_metadata,
            )
            for audio_bytes, text in zip(audio_bytes_list, texts)
        ]

    def predict_long(
        self,
        audio_bytes: bytes,
        text: str = "",
        window_sec: float = 12.0,
        stride_sec: float = 6.0,
        max_batch: int = 16,
    ) -> dict:
        self.predict_single(audio_bytes, text)
        return {
            "final_label": "Excited",
            "final_confidence": 0.6,
            "avg_probabilities": dict(PROBABILITIES),
            "timeline": [
                {
                    "window": "0.0~3.0s",
                    "label": "Excited",
                    "confidence": 0.6,
                    "probabilities": dict(PROBABILITIES),
                }
            ],
            "total_windows": 1,
            "audio_duration_sec": 3.0,
        }


class ApiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Constructing TestClient without entering it deliberately avoids the
        # production lifespan, so the 870M-parameter runtime is not loaded.
        cls.client = TestClient(app_module.app, raise_server_exceptions=False)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()

    def setUp(self) -> None:
        app_module.predictor = FakePredictor()
        app_module._gpu_outstanding = 0
        app_module._gpu_waiting = 0
        app_module._gpu_active = 0
        app_module._request_inflight = 0

    def tearDown(self) -> None:
        app_module.predictor = None

    def assert_error(
        self,
        response,
        status_code: int,
        detail: str,
        error_code: str | None = None,
    ) -> None:
        self.assertEqual(response.status_code, status_code, response.text)
        payload = response.json()
        self.assertEqual(payload["detail"], detail)
        self.assertIsInstance(payload["error_code"], str)
        self.assertTrue(payload["request_id"])
        if error_code is not None:
            self.assertEqual(payload["error_code"], error_code)

    def assert_probabilities(self, probabilities: dict[str, float]) -> None:
        self.assertEqual(list(probabilities), list(INTERVIEW_CLASSES))
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=4)
        for value in probabilities.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_openapi_exposes_version_paths_and_error_models(self) -> None:
        document = app_module.app.openapi()
        self.assertEqual(document["info"]["version"], "1.3.0")
        self.assertEqual(
            set(document["paths"]),
            {
                "/v1/health",
                "/v1/live",
                "/v1/ready",
                "/v1/metrics",
                "/v1/emotion/classify",
                "/v1/emotion/classify-batch",
                "/v1/emotion/classify-long",
            },
        )
        for path in (
            "/v1/emotion/classify",
            "/v1/emotion/classify-batch",
            "/v1/emotion/classify-long",
        ):
            responses = document["paths"][path]["post"]["responses"]
            self.assertIn("400", responses)
            self.assertIn("401", responses)
            self.assertIn("500", responses)
            self.assertIn("503", responses)
            self.assertEqual(
                document["paths"][path]["post"]["security"],
                [{"BearerAuth": []}],
            )
        self.assertIn("BearerAuth", document["components"]["securitySchemes"])

    def test_health_contract_and_stable_identity(self) -> None:
        response = self.client.get("/v1/health")
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        _validate(HealthResponse, payload)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["model"], N3_MODEL_ID)
        self.assertEqual(payload["model_id"], N3_MODEL_ID)
        self.assertEqual(payload["classes"], list(INTERVIEW_CLASSES))
        self.assertTrue(payload["hf_offline"])

    def test_health_returns_503_before_model_is_ready(self) -> None:
        app_module.predictor = None
        response = self.client.get("/v1/health")
        self.assert_error(response, 503, "Model is not ready", "MODEL_NOT_READY")

    def test_liveness_does_not_depend_on_model_readiness(self) -> None:
        app_module.predictor = None
        response = self.client.get("/v1/live")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"status": "alive", "version": "1.3.0"})

    def test_readiness_and_metrics_contracts(self) -> None:
        ready = self.client.get("/v1/ready")
        self.assertEqual(ready.status_code, 200, ready.text)
        self.assertEqual(ready.json()["model_id"], N3_MODEL_ID)
        metrics = self.client.get("/v1/metrics")
        self.assertEqual(metrics.status_code, 200, metrics.text)
        self.assertIn("timing_ms", metrics.json())
        self.assertIn("queue", metrics.json())

    def test_inference_endpoints_return_503_before_model_is_ready(self) -> None:
        app_module.predictor = None
        requests = [
            self.client.post(
                "/v1/emotion/classify",
                files={"audio": ("sample.wav", b"valid", "audio/wav")},
            ),
            self.client.post(
                "/v1/emotion/classify-batch",
                files=[("files", ("sample.wav", b"valid", "audio/wav"))],
            ),
            self.client.post(
                "/v1/emotion/classify-long",
                files={"audio": ("sample.wav", b"valid", "audio/wav")},
            ),
        ]
        for response in requests:
            self.assert_error(response, 503, "Model is not ready", "MODEL_NOT_READY")

    def test_single_audio_text_contract(self) -> None:
        response = self.client.post(
            "/v1/emotion/classify",
            files={"audio": ("sample.wav", b"valid", "audio/wav")},
            data={"text": "我對這份工作很有信心"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        _validate(SingleClassifyResponse, payload)
        self.assertEqual(payload["model_id"], N3_MODEL_ID)
        self.assertEqual(payload["input_mode"], "audio_text")
        self.assertEqual(payload["primary_label"], "Excited")
        self.assertEqual(payload["primary_confidence"], 0.6)
        self.assertGreaterEqual(payload["latency_ms"], 0.0)
        self.assertEqual(payload["latency_ms"], payload["inference_ms"])
        self.assertGreater(payload["audio_duration_sec"], 0.0)
        self.assertGreaterEqual(payload["queue_wait_ms"], 0.0)
        self.assertGreaterEqual(payload["server_total_ms"], payload["inference_ms"])
        self.assertTrue(response.headers["X-Request-ID"])
        self.assertTrue(response.headers["X-Server-Total-Ms"])
        self.assert_probabilities(payload["probabilities"])

    def test_single_audio_only_contract(self) -> None:
        response = self.client.post(
            "/v1/emotion/classify",
            files={"audio": ("sample.wav", b"valid", "audio/wav")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["input_mode"], "audio_only")

        whitespace = self.client.post(
            "/v1/emotion/classify",
            files={"audio": ("sample.wav", b"valid", "audio/wav")},
            data={"text": "   \t"},
        )
        self.assertEqual(whitespace.status_code, 200, whitespace.text)
        self.assertEqual(whitespace.json()["input_mode"], "audio_only")

    def test_request_id_is_echoed_and_invalid_value_is_replaced(self) -> None:
        valid = self.client.get(
            "/v1/health",
            headers={"X-Request-ID": "a4500-smoke-001"},
        )
        self.assertEqual(valid.headers["X-Request-ID"], "a4500-smoke-001")
        invalid = self.client.get(
            "/v1/health",
            headers={"X-Request-ID": "contains spaces and should be replaced"},
        )
        self.assertNotEqual(
            invalid.headers["X-Request-ID"],
            "contains spaces and should be replaced",
        )

    def test_auth_can_be_required_without_exposing_liveness(self) -> None:
        previous_required = app_module.REQUIRE_AUTH
        previous_token = app_module.API_TOKEN
        app_module.REQUIRE_AUTH = True
        app_module.API_TOKEN = "test-token-with-at-least-24-characters"
        try:
            live = self.client.get("/v1/live")
            self.assertEqual(live.status_code, 200, live.text)
            denied = self.client.get("/v1/health")
            self.assert_error(
                denied,
                401,
                "A valid bearer token is required",
                "UNAUTHORIZED",
            )
            allowed = self.client.get(
                "/v1/health",
                headers={
                    "Authorization":
                        "Bearer test-token-with-at-least-24-characters"
                },
            )
            self.assertEqual(allowed.status_code, 200, allowed.text)
        finally:
            app_module.REQUIRE_AUTH = previous_required
            app_module.API_TOKEN = previous_token

    def test_file_text_and_content_length_limits(self) -> None:
        previous_file_limit = app_module.MAX_SINGLE_FILE_BYTES
        previous_text_limit = app_module.MAX_TEXT_CHARS
        app_module.MAX_SINGLE_FILE_BYTES = 4
        app_module.MAX_TEXT_CHARS = 4
        try:
            file_too_large = self.client.post(
                "/v1/emotion/classify",
                files={"audio": ("sample.wav", b"12345", "audio/wav")},
            )
            self.assert_error(
                file_too_large,
                413,
                "Audio file exceeds the 4-byte limit",
                "FILE_TOO_LARGE",
            )
            text_too_large = self.client.post(
                "/v1/emotion/classify",
                files={"audio": ("sample.wav", b"1234", "audio/wav")},
                data={"text": "12345"},
            )
            self.assert_error(
                text_too_large,
                400,
                "Transcript exceeds the 4-character request limit",
                "TEXT_TOO_LARGE",
            )
            bad_length = self.client.post(
                "/v1/emotion/classify",
                files={"audio": ("sample.wav", b"1234", "audio/wav")},
                headers={"Content-Length": "invalid"},
            )
            self.assert_error(
                bad_length,
                400,
                "Content-Length must be a non-negative integer",
                "INVALID_CONTENT_LENGTH",
            )
        finally:
            app_module.MAX_SINGLE_FILE_BYTES = previous_file_limit
            app_module.MAX_TEXT_CHARS = previous_text_limit

    def test_gpu_oom_is_sanitized_and_next_request_recovers(self) -> None:
        oom = self.client.post(
            "/v1/emotion/classify",
            files={"audio": ("oom.wav", b"oom", "audio/wav")},
        )
        self.assert_error(
            oom,
            503,
            "GPU capacity is temporarily unavailable",
            "GPU_OUT_OF_MEMORY",
        )
        self.assertNotIn("allocator", oom.text)
        recovered = self.client.post(
            "/v1/emotion/classify",
            files={"audio": ("ok.wav", b"valid", "audio/wav")},
        )
        self.assertEqual(recovered.status_code, 200, recovered.text)

    def test_unhandled_response_shaping_error_uses_stable_json_contract(self) -> None:
        response = self.client.post(
            "/v1/emotion/classify",
            files={"audio": ("malformed.wav", b"malformed", "audio/wav")},
            headers={"X-Request-ID": "malformed-result-001"},
        )
        self.assert_error(
            response,
            500,
            "Internal server error",
            "INTERNAL_ERROR",
        )
        self.assertEqual(response.headers["X-Request-ID"], "malformed-result-001")
        self.assertIn("X-Server-Total-Ms", response.headers)
        self.assertEqual(response.headers["content-type"], "application/json")

    def test_pre_body_request_admission_is_bounded(self) -> None:
        previous_limit = app_module.MAX_INFLIGHT_REQUESTS
        app_module.MAX_INFLIGHT_REQUESTS = 0
        try:
            response = self.client.post(
                "/v1/emotion/classify",
                files={"audio": ("sample.wav", b"valid", "audio/wav")},
            )
            self.assert_error(
                response,
                503,
                "Inference request capacity is full",
                "REQUEST_CAPACITY_FULL",
            )
            self.assertEqual(response.headers["Retry-After"], "1")
        finally:
            app_module.MAX_INFLIGHT_REQUESTS = previous_limit

    def test_framework_and_multipart_errors_use_stable_json_contract(self) -> None:
        malformed = self.client.post(
            "/v1/emotion/classify",
            content=b"--broken\r\ninvalid\r\n",
            headers={
                "Content-Type": "multipart/form-data; boundary=broken",
            },
        )
        self.assert_error(
            malformed,
            400,
            "Malformed request body",
            "MALFORMED_REQUEST",
        )
        missing = self.client.get("/v1/does-not-exist")
        self.assert_error(
            missing,
            404,
            "Endpoint not found",
            "NOT_FOUND",
        )

    def test_single_missing_empty_invalid_and_internal_errors(self) -> None:
        missing = self.client.post("/v1/emotion/classify")
        self.assertEqual(missing.status_code, 422, missing.text)
        self.assert_error(
            missing,
            422,
            "Request fields are missing or invalid",
            "VALIDATION_ERROR",
        )

        empty = self.client.post(
            "/v1/emotion/classify",
            files={"audio": ("empty.wav", b"", "audio/wav")},
        )
        self.assert_error(empty, 400, "Audio file is empty", "EMPTY_AUDIO")

        invalid = self.client.post(
            "/v1/emotion/classify",
            files={"audio": ("invalid.wav", b"invalid", "audio/wav")},
        )
        self.assert_error(invalid, 400, "Audio could not be decoded", "INVALID_AUDIO")

        crash = self.client.post(
            "/v1/emotion/classify",
            files={"audio": ("crash.wav", b"crash", "audio/wav")},
        )
        self.assert_error(crash, 500, "Inference failed", "INFERENCE_FAILED")
        self.assertNotIn("sensitive", crash.text)

    def test_batch_audio_text_audio_only_and_mixed_contracts(self) -> None:
        cases = [
            ([], "audio_only"),
            (["第一句", "第二句"], "audio_text"),
            (["第一句", ""], "mixed"),
        ]
        for texts, expected_mode in cases:
            with self.subTest(expected_mode=expected_mode):
                multipart = [
                    ("files", ("a.wav", b"valid-a", "audio/wav")),
                    ("files", ("b.wav", b"valid-b", "audio/wav")),
                ]
                multipart.extend(("texts", (None, text)) for text in texts)
                response = self.client.post(
                    "/v1/emotion/classify-batch",
                    files=multipart,
                )
                self.assertEqual(response.status_code, 200, response.text)
                payload = response.json()
                _validate(BatchClassifyResponse, payload)
                self.assertEqual(payload["model_id"], N3_MODEL_ID)
                self.assertEqual(payload["input_mode"], expected_mode)
                self.assertEqual(payload["batch_size"], 2)
                self.assertEqual([item["filename"] for item in payload["results"]], ["a.wav", "b.wav"])
                self.assertEqual(len(payload["results"]), payload["batch_size"])
                for item in payload["results"]:
                    self.assert_probabilities(item["probabilities"])

    def test_batch_limit_count_empty_file_and_result_count_errors(self) -> None:
        too_many = [
            ("files", (f"{index}.wav", b"valid", "audio/wav"))
            for index in range(17)
        ]
        response = self.client.post("/v1/emotion/classify-batch", files=too_many)
        self.assert_error(
            response,
            400,
            "Batch size exceeds the maximum of 16",
            "BATCH_TOO_LARGE",
        )

        mismatch = self.client.post(
            "/v1/emotion/classify-batch",
            files=[
                ("files", ("a.wav", b"valid-a", "audio/wav")),
                ("files", ("b.wav", b"valid-b", "audio/wav")),
                ("texts", (None, "only one")),
            ],
        )
        self.assertEqual(mismatch.status_code, 400, mismatch.text)
        self.assert_error(
            mismatch,
            400,
            "The number of texts must equal the number of files",
            "BATCH_TEXT_COUNT_MISMATCH",
        )

        empty = self.client.post(
            "/v1/emotion/classify-batch",
            files=[
                ("files", ("a.wav", b"valid-a", "audio/wav")),
                ("files", ("empty.wav", b"", "audio/wav")),
            ],
        )
        self.assert_error(empty, 400, "Audio file is empty", "EMPTY_AUDIO")

        short_result = self.client.post(
            "/v1/emotion/classify-batch",
            files=[("files", ("a.wav", b"short_result", "audio/wav"))],
        )
        self.assert_error(
            short_result,
            500,
            "Inference result count mismatch",
            "RESULT_COUNT_MISMATCH",
        )

        crash = self.client.post(
            "/v1/emotion/classify-batch",
            files=[("files", ("crash.wav", b"crash", "audio/wav"))],
        )
        self.assert_error(crash, 500, "Inference failed", "INFERENCE_FAILED")
        self.assertNotIn("sensitive", crash.text)

    def test_long_contract_and_input_modes(self) -> None:
        for text, expected_mode in [
            ("完整而且對齊的逐字稿", "audio_text"),
            ("", "audio_only"),
        ]:
            with self.subTest(expected_mode=expected_mode):
                response = self.client.post(
                    "/v1/emotion/classify-long",
                    files={"audio": ("long.wav", b"valid", "audio/wav")},
                    data={"text": text, "window_sec": "12", "stride_sec": "6"},
                )
                self.assertEqual(response.status_code, 200, response.text)
                payload = response.json()
                _validate(LongClassifyResponse, payload)
                self.assertEqual(payload["input_mode"], expected_mode)
                self.assertEqual(payload["total_windows"], len(payload["timeline"]))
                self.assert_probabilities(payload["avg_probabilities"])

    def test_long_parameter_empty_and_decode_errors(self) -> None:
        bad_window = self.client.post(
            "/v1/emotion/classify-long",
            files={"audio": ("long.wav", b"valid", "audio/wav")},
            data={"window_sec": "12.1"},
        )
        self.assertEqual(bad_window.status_code, 400, bad_window.text)
        self.assertIn("window_sec", bad_window.json()["detail"])

        bad_stride = self.client.post(
            "/v1/emotion/classify-long",
            files={"audio": ("long.wav", b"valid", "audio/wav")},
            data={"window_sec": "3", "stride_sec": "4"},
        )
        self.assertEqual(bad_stride.status_code, 400, bad_stride.text)
        self.assertIn("stride_sec", bad_stride.json()["detail"])

        invalid_number = self.client.post(
            "/v1/emotion/classify-long",
            files={"audio": ("long.wav", b"valid", "audio/wav")},
            data={"window_sec": "not-a-number"},
        )
        self.assertEqual(invalid_number.status_code, 422, invalid_number.text)
        self.assert_error(
            invalid_number,
            422,
            "Request fields are missing or invalid",
            "VALIDATION_ERROR",
        )

        empty = self.client.post(
            "/v1/emotion/classify-long",
            files={"audio": ("empty.wav", b"", "audio/wav")},
        )
        self.assert_error(empty, 400, "Audio file is empty", "EMPTY_AUDIO")

        invalid_audio = self.client.post(
            "/v1/emotion/classify-long",
            files={"audio": ("invalid.wav", b"invalid", "audio/wav")},
        )
        self.assert_error(
            invalid_audio,
            400,
            "Audio could not be decoded",
            "INVALID_AUDIO",
        )

        crash = self.client.post(
            "/v1/emotion/classify-long",
            files={"audio": ("crash.wav", b"crash", "audio/wav")},
        )
        self.assert_error(crash, 500, "Inference failed", "INFERENCE_FAILED")
        self.assertNotIn("sensitive", crash.text)


class GpuCancellationSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_cancel_cannot_release_or_bypass_gpu_queue(self) -> None:
        previous_lock = app_module._gpu_lock
        app_module._gpu_lock = asyncio.Lock()
        app_module._gpu_outstanding = 0
        app_module._gpu_waiting = 0
        app_module._gpu_active = 0
        started = threading.Event()
        release = threading.Event()
        second_started = threading.Event()

        def first_job() -> str:
            started.set()
            release.wait(timeout=2.0)
            return "first"

        def second_job() -> str:
            second_started.set()
            return "second"

        first = asyncio.create_task(app_module._run_gpu_inference(first_job))
        second = None
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 1.0))
            first.cancel()
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            self.assertEqual(app_module._queue_snapshot()["outstanding_jobs"], 1)

            second = asyncio.create_task(
                app_module._run_gpu_inference(second_job)
            )
            await asyncio.sleep(0.05)
            self.assertFalse(second_started.is_set())
            self.assertEqual(app_module._queue_snapshot()["outstanding_jobs"], 2)

            release.set()
            result, _ = await asyncio.wait_for(second, timeout=1.0)
            self.assertEqual(result, "second")
            await asyncio.sleep(0)
            self.assertEqual(app_module._queue_snapshot()["outstanding_jobs"], 0)
        finally:
            release.set()
            if second is not None and not second.done():
                second.cancel()
                await asyncio.gather(second, return_exceptions=True)
            if app_module._drain_tasks:
                await asyncio.gather(
                    *list(app_module._drain_tasks),
                    return_exceptions=True,
                )
            app_module._gpu_lock = previous_lock
            app_module._gpu_outstanding = 0
            app_module._gpu_waiting = 0
            app_module._gpu_active = 0

    async def test_queue_full_and_queue_timeout_are_bounded(self) -> None:
        previous_lock = app_module._gpu_lock
        previous_depth = app_module.GPU_QUEUE_DEPTH
        previous_timeout = app_module.GPU_QUEUE_TIMEOUT_SEC
        app_module._gpu_lock = asyncio.Lock()
        app_module._gpu_outstanding = 0
        app_module._gpu_waiting = 0
        app_module._gpu_active = 0
        started = threading.Event()
        release = threading.Event()

        def blocking_job() -> str:
            started.set()
            if not release.wait(timeout=2.0):
                raise TimeoutError("test did not release GPU job")
            return "done"

        first = asyncio.create_task(app_module._run_gpu_inference(blocking_job))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 1.0))
            app_module.GPU_QUEUE_DEPTH = 0
            with self.assertRaises(app_module.ApiFailure) as full_context:
                await app_module._run_gpu_inference(lambda: "never")
            self.assertEqual(full_context.exception.error_code, "GPU_QUEUE_FULL")

            app_module.GPU_QUEUE_DEPTH = 1
            app_module.GPU_QUEUE_TIMEOUT_SEC = 0.01
            with self.assertRaises(app_module.ApiFailure) as timeout_context:
                await app_module._run_gpu_inference(lambda: "never")
            self.assertEqual(
                timeout_context.exception.error_code,
                "GPU_QUEUE_TIMEOUT",
            )
        finally:
            release.set()
            await asyncio.gather(first, return_exceptions=True)
            app_module._gpu_lock = previous_lock
            app_module.GPU_QUEUE_DEPTH = previous_depth
            app_module.GPU_QUEUE_TIMEOUT_SEC = previous_timeout
            app_module._gpu_outstanding = 0
            app_module._gpu_waiting = 0
            app_module._gpu_active = 0

    async def test_cancelled_request_cannot_overlap_next_gpu_job(self) -> None:
        previous_lock = app_module._gpu_lock
        app_module._gpu_lock = asyncio.Lock()
        counter_lock = threading.Lock()
        first_started = threading.Event()
        second_started = threading.Event()
        release_first = threading.Event()
        active = 0
        max_active = 0

        def gpu_job(name: str) -> str:
            nonlocal active, max_active
            with counter_lock:
                active += 1
                max_active = max(max_active, active)
            try:
                if name == "first":
                    first_started.set()
                    if not release_first.wait(timeout=2.0):
                        raise TimeoutError("test did not release the first GPU job")
                else:
                    second_started.set()
                return name
            finally:
                with counter_lock:
                    active -= 1

        first = None
        second = None
        try:
            first = asyncio.create_task(
                app_module._run_gpu_inference(gpu_job, "first")
            )
            started = await asyncio.to_thread(first_started.wait, 1.0)
            self.assertTrue(started, "first GPU job did not start")

            first.cancel()
            await asyncio.sleep(0)
            second = asyncio.create_task(
                app_module._run_gpu_inference(gpu_job, "second")
            )
            await asyncio.sleep(0.05)
            self.assertFalse(
                second_started.is_set(),
                "second GPU job started while cancelled job was still running",
            )
            self.assertEqual(max_active, 1)

            release_first.set()
            with self.assertRaises(asyncio.CancelledError):
                await first
            result, timing = await asyncio.wait_for(second, timeout=1.0)
            self.assertEqual(result, "second")
            self.assertGreaterEqual(timing.inference_ms, 0.0)
            self.assertEqual(max_active, 1)
        finally:
            release_first.set()
            pending = [
                task
                for task in (first, second)
                if task is not None and not task.done()
            ]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            app_module._gpu_lock = previous_lock


if __name__ == "__main__":
    unittest.main()
