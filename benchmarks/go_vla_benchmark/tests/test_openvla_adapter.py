from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "go_vla_benchmark"))

from go_vla_benchmark.explainability.openvla_adapter import (  # noqa: E402
    OpenVLAExplainabilityAdapter,
    TextMaskCandidateSpec,
    _collect_text_mask_spans,
    _resolve_text_mask_candidate_index,
)


class OpenVLAAdapterTest(unittest.TestCase):
    def test_collect_text_mask_spans_shortlists_place_template(self) -> None:
        task_text = "place a black stone on the go board at row 4, column 5"
        labels = [label for label, _start, _end in _collect_text_mask_spans(task_text)]

        self.assertEqual(
            labels,
            ["black", "4", "5", "row 4, column 5", "go board", "row 4", "column 5"],
        )

    def test_collect_text_mask_spans_shortlists_position_template(self) -> None:
        task_text = "put a black stone at position (4, 5) on the go board"
        labels = [label for label, _start, _end in _collect_text_mask_spans(task_text)]

        self.assertEqual(
            labels,
            [
                "black",
                "4",
                "5",
                "position (4, 5)",
                "go board",
                "at position (4, 5)",
                "stone at position (4, 5)",
            ],
        )

    def test_collect_text_mask_spans_shortlists_move_template(self) -> None:
        task_text = "move the white stone to row 3, column 1 on the board"
        labels = [label for label, _start, _end in _collect_text_mask_spans(task_text)]

        self.assertEqual(
            labels,
            ["white", "3", "1", "row 3, column 1", "the board", "row 3", "column 1"],
        )

    def test_collect_text_mask_spans_shortlists_set_template(self) -> None:
        task_text = "set a white stone at (2, 4)"
        labels = [label for label, _start, _end in _collect_text_mask_spans(task_text)]

        self.assertEqual(
            labels,
            ["white", "2", "4", "(2, 4)", "at (2, 4)", "stone at (2, 4)", "white stone"],
        )

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

    def test_resolve_text_mask_candidate_index_prefers_full_target_phrase(self) -> None:
        candidates = [
            TextMaskCandidateSpec(
                index=0,
                label="row 3",
                prompt_token_positions=np.asarray([1], dtype=np.int64),
                task_char_start=0,
                task_char_end=5,
            ),
            TextMaskCandidateSpec(
                index=1,
                label="column 4",
                prompt_token_positions=np.asarray([2], dtype=np.int64),
                task_char_start=7,
                task_char_end=15,
            ),
            TextMaskCandidateSpec(
                index=2,
                label="row 3, column 4",
                prompt_token_positions=np.asarray([1, 2], dtype=np.int64),
                task_char_start=0,
                task_char_end=15,
            ),
        ]

        resolved = _resolve_text_mask_candidate_index(
            candidates,
            target_row=3,
            target_col=4,
        )

        self.assertEqual(resolved, 2)


if __name__ == "__main__":
    unittest.main()
