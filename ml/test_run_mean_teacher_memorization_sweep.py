from __future__ import annotations

import unittest

from run_mean_teacher_memorization_sweep import DEFAULTS, cases, suffix


class MeanTeacherSweepTest(unittest.TestCase):
    def test_sweep_is_twelve_unique_one_factor_at_a_time_runs(self) -> None:
        values = cases()
        self.assertEqual(len(values), 12)
        self.assertEqual(len({suffix(value) for value in values}), 12)
        for case in values:
            changed = sum(
                float(case[name]) != float(default)
                for name, default in DEFAULTS.items()
            )
            self.assertLessEqual(changed, 1)


if __name__ == "__main__":
    unittest.main()
