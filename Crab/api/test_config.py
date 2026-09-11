from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from .config import ApiSettings


class ApiSettingsTests(unittest.TestCase):
    def test_auth_requirement_fails_without_strong_token(self) -> None:
        with patch.dict(
            os.environ,
            {"CRAB_REQUIRE_AUTH": "true", "CRAB_API_TOKEN": "short"},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "at least 24"):
                ApiSettings.from_env()

    def test_invalid_boolean_fails_fast(self) -> None:
        with patch.dict(
            os.environ,
            {"CRAB_REQUIRE_AUTH": "sometimes"},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "true or false"):
                ApiSettings.from_env()

    def test_zero_waiting_queue_is_allowed(self) -> None:
        with patch.dict(
            os.environ,
            {
                "CRAB_REQUIRE_AUTH": "false",
                "CRAB_GPU_QUEUE_DEPTH": "0",
            },
            clear=False,
        ):
            settings = ApiSettings.from_env()
            self.assertEqual(settings.gpu_queue_depth, 0)
            self.assertEqual(settings.max_inflight_requests, 1)


if __name__ == "__main__":
    unittest.main()
