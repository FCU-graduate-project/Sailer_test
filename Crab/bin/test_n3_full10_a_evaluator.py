"""CPU-only authorization/provenance tests. No GPU, network, or experiment writes."""
import copy
import unittest

import eval_n3_full10_a_safe as target


class Full10ATestPreparationTests(unittest.TestCase):
    def fixture(self, dose=25):
        arm = {"dose": dose, "seed": 42, "experiment_id": f"n3a{dose}s42",
               "output_dir": str(target.TRAIN / f"n3a{dose}s42"), "run_name": "frozen"}
        result = {"run_name": "frozen",
                  "wandb_url": f"https://wandb.ai/{target.ENTITY}/{target.PROJECT}/runs/{target.RUN_IDS[dose]}",
                  "test_evaluated": False, "best_epoch": 3, "best_dev_macro_f1": 0.54,
                  "optimizer_updates": {25: 2010, 50: 2110, 75: 2210}[dose]}
        completed = dict(result, experiment_id=arm["experiment_id"], actual_epochs=10,
                         seed=42, dose=dose)
        return arm, result, completed

    def reject(self, which, key, value):
        args = list(copy.deepcopy(self.fixture()))
        args[which][key] = value
        with self.assertRaises(RuntimeError):
            target.validate_arm(*args)

    def test_authorized_doses(self):
        for dose in (25, 50, 75):
            with self.subTest(dose=dose):
                target.validate_arm(*self.fixture(dose))

    def test_other_seed(self):
        self.reject(0, "seed", 43)

    def test_b_model(self):
        self.reject(0, "experiment_id", "n3b25s42")

    def test_historical_a_model(self):
        self.reject(0, "output_dir", str(target.CRAB / "experiments/n3_phase_a_long12_20260909/n3a25"))

    def test_wrong_run(self):
        self.reject(1, "wandb_url", "https://wandb.ai/wrong/project/runs/other")

    def test_changed_result(self):
        self.reject(2, "best_dev_macro_f1", 0.99)

    def test_incomplete_epochs(self):
        self.reject(2, "actual_epochs", 9)

    def test_wrong_completed_seed(self):
        self.reject(2, "seed", 43)

    def test_wrong_completed_dose(self):
        self.reject(2, "dose", 50)

    def test_bad_epoch(self):
        self.reject(1, "best_epoch", 10)

    def test_test_already_in_training_result(self):
        self.reject(1, "test_evaluated", True)

    def test_nonfinite_score(self):
        self.reject(1, "best_dev_macro_f1", float("nan"))

    def test_wrong_updates(self):
        self.reject(1, "optimizer_updates", 1)

    def test_execution_matrix(self):
        arms = [{"id": f"n3a{dose}s42", "model_dir": str(target.TRAIN / f"n3a{dose}s42"),
                 "wandb_run_id": target.RUN_IDS[dose]} for dose in (25, 50, 75)]
        target.validate_selection(arms)
        for bad in (arms[:2], arms[::-1], arms + arms[:1]):
            with self.assertRaises(RuntimeError):
                target.validate_selection(bad)
        for key, value in (("model_dir", str(target.CRAB)), ("wandb_run_id", "old_run")):
            bad = copy.deepcopy(arms)
            bad[0][key] = value
            with self.assertRaises(RuntimeError):
                target.validate_selection(bad)


if __name__ == "__main__":
    unittest.main()
