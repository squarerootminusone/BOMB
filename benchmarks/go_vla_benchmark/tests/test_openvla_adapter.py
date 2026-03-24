from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "go_vla_benchmark"))

from go_vla_benchmark.explainability.openvla_adapter import OpenVLAExplainabilityAdapter  # noqa: E402


class OpenVLAAdapterTest(unittest.TestCase):
    def test_score_from_outputs_detaches_before_numpy(self) -> None:
        import torch

        adapter = OpenVLAExplainabilityAdapter.__new__(OpenVLAExplainabilityAdapter)
        outputs = types.SimpleNamespace(
            logits=torch.tensor(
                [[[0.0, 1.0, 2.0], [2.0, 0.0, 1.0]]],
                dtype=torch.float32,
                requires_grad=True,
            )
        )

        score = adapter._score_from_outputs(
            outputs=outputs,
            query_text_positions=np.asarray([0, 1], dtype=np.int64),
            target_token_ids=np.asarray([2, 0], dtype=np.int64),
            num_patches=0,
        )

        expected = (
            torch.gather(
                torch.softmax(outputs.logits.detach()[0], dim=-1),
                dim=-1,
                index=torch.tensor([[2], [0]], dtype=torch.long),
            )
            .squeeze(-1)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        self.assertEqual(score.target_token_probs.dtype, np.float32)
        np.testing.assert_allclose(score.target_token_probs, expected, rtol=1e-6, atol=1e-6)
        self.assertAlmostEqual(score.sequence_logprob, float(np.log(expected).sum()), places=6)


if __name__ == "__main__":
    unittest.main()
