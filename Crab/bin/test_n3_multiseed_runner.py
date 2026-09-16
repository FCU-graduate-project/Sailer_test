"""CPU-only protocol and failure-gate tests; never import the model trainer."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

BIN = Path(__file__).resolve().parents[1] / "bin"
sys.path.insert(0, str(BIN))
import run_n3_multiseed_safe as runner


class MultiseedTests(unittest.TestCase):
    def setUp(self):
        # Public CPU fixture: no private campaign, transcripts, audio or weights.
        self.original = {
            "python": sys.executable, "crab_root": str(BIN.parent),
            "common_cli": {"seed": 42, "contrastive_weight": 2, "epochs": 10,
                           "early_stop_patience": 5, "wandb_group": "fixture"},
            "common_flags": ["--no_eval_test"],
            "arms": [{"training_csv": f"/fixture/a{dose}.csv",
                      "metadata_json": f"/fixture/a{dose}.json",
                      "run_name": f"fixture-a{dose}"} for dose in (25, 50, 75)],
        }
        def read_fixture(path):
            if Path(path) == runner.ORIGINAL / "campaign.json":
                return copy.deepcopy(self.original)
            raise AssertionError(f"Unexpected non-fixture read: {path}")
        reader = patch.object(runner, "read", side_effect=read_fixture)
        reader.start()
        self.addCleanup(reader.stop)

    def test_seed_override_does_not_mutate_common_config(self):
        config = copy.deepcopy(self.original)
        config["common_cli"].update(early_stop_patience=0, wandb_group=runner.GROUP)
        arm = dict(config["arms"][0], seed=44)
        before = copy.deepcopy(config)
        command = runner.arm_command(config, arm, Path("/tmp/not-created"))
        self.assertEqual(command[command.index("--seed") + 1], "44")
        self.assertEqual(command[command.index("--early_stop_patience") + 1], "0")
        self.assertEqual(command[command.index("--contrastive_weight") + 1], "2")
        self.assertEqual(command[command.index("--epochs") + 1], "10")
        self.assertIn("--no_eval_test", command)
        self.assertEqual(config, before)

    def test_smoke_is_separate_and_requested(self):
        arm = dict(self.original["arms"][2], seed=42)
        command = runner.arm_command(self.original, arm, Path("/tmp/not-created-smoke"), smoke=True)
        self.assertIn("--smoke_test", command)

    def test_validation_rejects_changed_lambda(self):
        config = copy.deepcopy(self.original)
        config["common_cli"].update(early_stop_patience=0, wandb_group=runner.GROUP, contrastive_weight=3)
        with self.assertRaisesRegex(RuntimeError, "CLI/flag"):
            runner.validate_config(config, Path("/tmp/not-created"))

    def test_validation_rejects_early_stopping(self):
        config = copy.deepcopy(self.original)
        config["common_cli"]["wandb_group"] = runner.GROUP
        with self.assertRaisesRegex(RuntimeError, "CLI/flag"):
            runner.validate_config(config, Path("/tmp/not-created"))

    def test_validation_rejects_test_flag(self):
        config = copy.deepcopy(self.original)
        config["common_cli"].update(early_stop_patience=0, wandb_group=runner.GROUP)
        config["common_flags"].remove("--no_eval_test")
        with self.assertRaisesRegex(RuntimeError, "CLI/flag"):
            runner.validate_config(config, Path("/tmp/not-created"))

    def test_aggregate_requires_nine_unique_runs(self):
        with self.assertRaises(RuntimeError):
            runner.aggregate([])
        with self.assertRaises(RuntimeError):
            runner.aggregate([dict(seed=42, dose=25)] * 9)

    def test_aggregate_uses_sample_std_and_paired_differences(self):
        results = [dict(seed=s, dose=d, best_dev_macro_f1=0.5 + (s - 42) * 0.01 + d / 1000)
                   for s in runner.SEEDS for d in runner.DOSES]
        result = runner.aggregate(results)
        self.assertAlmostEqual(result["by_dose"]["25"]["mean"], 0.535)
        self.assertAlmostEqual(result["by_dose"]["25"]["sample_std"], 0.01)
        self.assertAlmostEqual(result["paired_differences"]["A75_minus_A25"]["43"], 0.05)
        self.assertIs(result["test_evaluated"], False)

    @patch.object(runner, "read", return_value={"state": "training"})
    def test_predecessor_running_is_rejected(self, read):
        with self.assertRaisesRegex(RuntimeError, "not successfully complete"):
            runner.verify_predecessor()

    @patch.object(runner, "read", return_value={"state": "stopped"})
    def test_predecessor_failed_is_rejected(self, read):
        with self.assertRaisesRegex(RuntimeError, "not successfully complete"):
            runner.verify_predecessor()

    @patch.object(runner, "read", return_value={"state": "complete_dev_only", "completed": []})
    def test_predecessor_missing_arms_is_rejected(self, read):
        with self.assertRaisesRegex(RuntimeError, "all four"):
            runner.verify_predecessor()

    def test_predecessor_service_must_release_gpu(self):
        results = [dict(contrastive_weight=w, test_evaluated=False) for w in range(4)]
        status = dict(state="complete_dev_only", completed=results)
        with patch.object(runner, "read", side_effect=[status] + [dict(test_evaluated=False)] * 4):
            with patch.object(runner.subprocess, "check_output", return_value="ActiveState=active\nResult=success\n"):
                with self.assertRaisesRegex(RuntimeError, "not inactive/success"):
                    runner.verify_predecessor()

    def test_predecessor_test_evaluation_is_rejected(self):
        results = [dict(contrastive_weight=w, test_evaluated=False) for w in range(4)]
        status = dict(state="complete_dev_only", completed=results)
        with patch.object(runner, "read", side_effect=[status, dict(test_evaluated=True)]):
            with self.assertRaisesRegex(RuntimeError, "unexpected Test"):
                runner.verify_predecessor()


if __name__ == "__main__":
    unittest.main()
