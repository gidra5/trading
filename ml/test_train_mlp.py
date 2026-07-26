from types import SimpleNamespace
import unittest

from train_mlp import validation_target_reached


class ValidationTargetTest(unittest.TestCase):
    def test_base_kl_target_is_independent_of_conditional_kl(self) -> None:
        args = SimpleNamespace(
            target_validation_base_kl=0.15,
            target_validation_kl=None,
            target_validation_kl_stddev=None,
        )
        self.assertTrue(validation_target_reached({
            "baseKlDivergence": 0.12,
            "klDivergence": 1.0,
        }, args))
        self.assertFalse(validation_target_reached({
            "baseKlDivergence": 0.16,
            "klDivergence": 0.1,
        }, args))


if __name__ == "__main__":
    unittest.main()
