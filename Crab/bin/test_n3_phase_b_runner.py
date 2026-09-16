"""CPU-only tests for B's pairing, provisional-QC and training-budget gates."""
import copy
import unittest
from unittest.mock import patch

import run_n3_phase_b_safe as b


class PhaseBTests(unittest.TestCase):
    def rows(self):
        real = []
        for split, count in (("Train", 3054), ("Development", 608), ("Test", 638)):
            real.extend(dict(FileName=f"{split}_{i}.wav", Text="real", Split_Set=split,
                             ParentID="", Source="NNIME") for i in range(count))
        a_tts = [dict(FileName=f"A{i}.wav", Text="parent", Split_Set="Train", ParentID=f"p{i}", Source="TTS") for i in range(2)]
        b_tts = [dict(row, FileName=f"B{i}.wav", Text="segment", Excited="0", Unconfident="1", Neutral_3Class="0",
                      HumanEmotionVerified="False", HumanTranscriptVerified="False", TrainingApproved="False", QCStatus="pending_human_review")
                 for i, row in enumerate(a_tts)]
        return real + a_tts, copy.deepcopy(real) + b_tts[::-1]

    def test_duration_boundaries(self):
        self.assertEqual([b.bin_for(s) for s in (1, 1.99, 2, 4, 6, 8, 10, 12)],
                         ["1-<2", "1-<2", "2-<4", "4-<6", "6-<8", "8-<10", "10-12", "10-12"])

    def test_invalid_durations_rejected(self):
        for value in (0.99, 12.01, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                b.bin_for(value)

    def test_pairing_restores_A_order_without_changing_candidates(self):
        a, candidates = self.rows()
        before = copy.deepcopy(candidates)
        result = b.paired_rows(a, candidates, 2)
        self.assertEqual([r["ParentID"] for r in result], [r["ParentID"] for r in a])
        self.assertEqual(result[-1]["Text"], "segment")
        self.assertEqual(result[-1]["HumanEmotionVerified"], "False")
        self.assertEqual(candidates, before)

    def test_changed_real_row_rejected(self):
        a, candidates = self.rows()
        candidates[0]["Text"] = "changed"
        with self.assertRaisesRegex(RuntimeError, "Real rows changed"):
            b.paired_rows(a, candidates, 2)

    def test_synthetic_test_leak_rejected(self):
        a, candidates = self.rows()
        candidates[-1]["Split_Set"] = "Test"
        with self.assertRaisesRegex(RuntimeError, "outside Train"):
            b.paired_rows(a, candidates, 2)

    def test_duplicate_parent_rejected(self):
        a, candidates = self.rows()
        candidates[-1]["ParentID"] = candidates[-2]["ParentID"]
        with self.assertRaisesRegex(RuntimeError, "Parent sets"):
            b.paired_rows(a, candidates, 2)

    def test_missing_parent_rejected(self):
        a, candidates = self.rows()
        candidates[-1]["ParentID"] = "unknown"
        with self.assertRaisesRegex(RuntimeError, "Parent sets"):
            b.paired_rows(a, candidates, 2)

    def test_human_validation_claim_rejected(self):
        a, candidates = self.rows()
        candidates[-1]["HumanEmotionVerified"] = "True"
        with self.assertRaisesRegex(RuntimeError, "QC flags"):
            b.paired_rows(a, candidates, 2)

    def test_empty_transcript_rejected(self):
        a, candidates = self.rows()
        candidates[-1]["Text"] = " "
        with self.assertRaisesRegex(RuntimeError, "Empty"):
            b.paired_rows(a, candidates, 2)

    def test_label_change_rejected(self):
        a, candidates = self.rows()
        candidates[-1]["Unconfident"] = "0"
        with self.assertRaisesRegex(RuntimeError, "label"):
            b.paired_rows(a, candidates, 2)

    def test_only_complete_dev_only_result_accepted(self):
        arm = dict(run_name="B", expected_optimizer_updates=2010)
        good = dict(run_name="B", test_evaluated=False, optimizer_updates=2010)
        b.validate_result(good, arm)
        for change in (dict(test_evaluated=True), dict(optimizer_updates=1809), dict(run_name="A")):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                b.validate_result(dict(good, **change), arm)

    @patch.object(b, "read", return_value={"state": "training", "completed": []})
    def test_predecessor_running_rejected(self, read):
        with self.assertRaisesRegex(RuntimeError, "not complete"):
            b.verify_predecessor()


if __name__ == "__main__":
    unittest.main()
