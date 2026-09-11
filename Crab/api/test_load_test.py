from __future__ import annotations

import unittest

from .load_test import (
    Record,
    build_windows,
    saturation_summary,
    select_stable_windows,
    summarize_records,
)


class LoadTestMathTests(unittest.TestCase):
    def test_summary_splits_all_latency_domains(self) -> None:
        records = [
            Record(1.0, 200, 15.0, 2.0, 10.0, 13.0),
            Record(2.0, 200, 17.0, 3.0, 11.0, 15.0),
            Record(3.0, 503, 1.0, error_code="GPU_QUEUE_FULL"),
        ]
        summary = summarize_records(records, 1.0)
        self.assertEqual(summary["successes"], 2)
        self.assertEqual(summary["failures"], 1)
        self.assertEqual(summary["queue"]["count"], 2)
        self.assertEqual(summary["inference"]["count"], 2)
        self.assertEqual(summary["server"]["count"], 2)
        self.assertEqual(summary["client"]["count"], 2)
        self.assertEqual(summary["error_codes"], {"GPU_QUEUE_FULL": 1})

    def test_stability_requires_consecutive_throughput_and_p95_windows(self) -> None:
        windows = [
            {"requests": 10, "throughput_rps": 2.0, "client": {"p95_ms": 100}},
            {"requests": 10, "throughput_rps": 2.1, "client": {"p95_ms": 103}},
            {"requests": 10, "throughput_rps": 2.0, "client": {"p95_ms": 101}},
        ]
        self.assertEqual(select_stable_windows(windows, 3, 0.10), [0, 1, 2])
        windows[2]["client"]["p95_ms"] = 200
        self.assertEqual(select_stable_windows(windows, 3, 0.10), [])

    def test_partial_tail_window_is_not_folded_into_previous_window(self) -> None:
        records = [
            Record(104.0, 200, 10.0),
            Record(121.0, 200, 10.0),
        ]
        windows, buckets = build_windows(
            records,
            run_started=100.0,
            warmup_sec=0.0,
            measurement_sec=22.0,
            window_sec=5.0,
        )
        self.assertEqual(len(windows), 5)
        self.assertEqual(windows[-1]["start_sec"], 20.0)
        self.assertEqual(windows[-1]["end_sec"], 22.0)
        self.assertEqual(windows[-1]["duration_sec"], 2.0)
        self.assertEqual(buckets[-1], [records[-1]])
        self.assertEqual(windows[-1]["throughput_rps"], 0.5)

    def test_unstable_level_cannot_be_last_good(self) -> None:
        class Args:
            min_success_rate = 0.99
            max_p95_ms = 250.0
            min_throughput_gain = 0.05

        results = [
            {
                "level": 5.0,
                "stable": True,
                "headline": {
                    "success_rate": 1.0,
                    "client": {"p95_ms": 50.0},
                    "dropped_by_load_generator": 0,
                    "throughput_rps": 5.0,
                },
            },
            {
                "level": 15.0,
                "stable": False,
                "headline": {
                    "success_rate": 1.0,
                    "client": {"p95_ms": 60.0},
                    "dropped_by_load_generator": 0,
                    "throughput_rps": 15.0,
                },
            },
        ]
        summary = saturation_summary(results, Args())
        self.assertEqual(summary["last_good_level"], 5.0)
        self.assertEqual(summary["first_failure_or_saturation_level"], 15.0)
        self.assertIn("unstable_measurement", results[1]["failure_reasons"])


if __name__ == "__main__":
    unittest.main()
