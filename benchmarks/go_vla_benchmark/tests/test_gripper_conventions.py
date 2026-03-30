from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "go_vla_benchmark"))

from go_vla_benchmark.explainability.core import extract_xyzg  # noqa: E402
from go_vla_benchmark.openvla_action_utils import (  # noqa: E402
    canonicalize_action_to_benchmark,
    openvla_action_to_benchmark,
)
from go_vla_benchmark.rlds_preprocessing import remap_gripper_to_openvla  # noqa: E402


class GripperConventionTest(unittest.TestCase):
    def test_openvla_action_to_benchmark_flips_open_and_close(self) -> None:
        action_open = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        action_close = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        np.testing.assert_array_equal(
            openvla_action_to_benchmark(action_open),
            np.asarray([0.0, 0.0, 0.0, -1.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            openvla_action_to_benchmark(action_close),
            np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )

    def test_extract_xyzg_matches_benchmark_conversion_for_openvla_gripper(self) -> None:
        raw_pred_action = np.asarray([0.4, 0.1, -0.2, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        np.testing.assert_array_equal(
            extract_xyzg(raw_pred_action)[0],
            np.asarray([0.4, 0.1, -0.2, -1.0], dtype=np.float32),
        )

    def test_canonicalize_action_to_benchmark_leaves_raw_benchmark_actions_unchanged(self) -> None:
        benchmark_actions = np.asarray(
            [[0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        np.testing.assert_array_equal(
            canonicalize_action_to_benchmark(benchmark_actions),
            benchmark_actions,
        )

    def test_remap_gripper_to_openvla_canonicalizes_openvla_style_actions(self) -> None:
        openvla_actions = np.asarray(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.25]],
            dtype=np.float32,
        )

        np.testing.assert_allclose(
            remap_gripper_to_openvla(openvla_actions),
            np.asarray(
                [[0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.5]],
                dtype=np.float32,
            ),
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
