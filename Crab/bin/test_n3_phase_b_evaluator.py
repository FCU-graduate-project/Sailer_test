"""CPU-only provenance/authorization tests; never loads models or writes campaigns."""
import copy
import unittest

import eval_n3_phase_b_safe as target


class PhaseBTestPreparationTests(unittest.TestCase):
    def fixture(self, dose=25):
        arm = {"dose": dose, "seed": 42, "experiment_id": f"n3b{dose}s42",
               "output_dir": str(target.TRAIN / f"n3b{dose}s42"), "run_name": "frozen"}
        result = {"run_name": "frozen",
                  "wandb_url": f"https://wandb.ai/{target.ENTITY}/{target.PROJECT}/runs/{target.RUN_IDS[dose]}",
                  "test_evaluated": False, "best_epoch": 8, "best_dev_macro_f1": 0.54,
                  "optimizer_updates": {25: 2010, 50: 2110, 75: 2210}[dose]}
        completed = dict(result, experiment_id=arm["experiment_id"], actual_epochs=10)
        return arm, result, completed

    def reject(self, which, key, value):
        args = list(copy.deepcopy(self.fixture()))
        args[which][key] = value
        with self.assertRaises(RuntimeError):
            target.validate_arm(*args)

    def test_three_authorized_doses(self):
        for dose in (25, 50, 75):
            with self.subTest(dose=dose):
                target.validate_arm(*self.fixture(dose))

    def test_reject_extra_seed(self):
        self.reject(0, "seed", 43)

    def test_reject_a_model(self):
        self.reject(0, "experiment_id", "n3a25s42")

    def test_reject_wrong_checkpoint_directory(self):
        self.reject(0, "output_dir", str(target.CRAB / "experiments/nnime_v1_warmstart_lora"))

    def test_reject_wrong_wandb_run(self):
        self.reject(1, "wandb_url", "https://wandb.ai/wrong/project/runs/other")

    def test_reject_changed_training_result(self):
        self.reject(2, "best_dev_macro_f1", 0.99)

    def test_reject_incomplete_epochs(self):
        self.reject(2, "actual_epochs", 9)

    def test_reject_latest_outside_best_epoch_range(self):
        self.reject(1, "best_epoch", 10)

    def test_reject_unexpected_training_test(self):
        self.reject(1, "test_evaluated", True)

    def test_reject_nonfinite_score(self):
        self.reject(1, "best_dev_macro_f1", float("nan"))

    def test_reject_incomplete_optimizer_updates(self):
        self.reject(1, "optimizer_updates", 1)


if __name__ == "__main__":
    unittest.main()

