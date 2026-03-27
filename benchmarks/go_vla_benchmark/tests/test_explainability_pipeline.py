from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "go_vla_benchmark"))

from go_vla_benchmark.explainability import (  # noqa: E402
    CausalLocalizationStep,
    CounterfactualEdit,
    EpisodeClip,
    collect_episode_causal_traces,
    collect_episode_intervention_traces,
    LocalExplanationStep,
    InterventionCandidateEffect,
    InterventionScan,
    collect_episode_traces,
    causal_trace_manifest,
    export_causal_localization_report,
    export_intervention_report,
    export_local_explanation_report,
    intervention_trace_manifest,
    load_causal_trace_file,
    load_intervention_trace_file,
    load_trace_file,
    save_causal_trace_file,
    save_intervention_trace_file,
    save_trace_file,
    trace_manifest,
)
from go_vla_benchmark.explainability.go_hdf5 import GoHDF5DatasetAdapter  # noqa: E402
from go_vla_benchmark.explainability.reporting_causal import (  # noqa: E402
    _apply_colormap,
    _normalize_restoration_scores,
)


class _FakeModelAdapter:
    def __init__(self) -> None:
        self._call_idx = 0
        self.prompt_style = "openvla"

    def explain_step(self, image: np.ndarray, instruction: str) -> LocalExplanationStep:
        del image
        token_offset = float(self._call_idx)
        self._call_idx += 1
        return LocalExplanationStep(
            pred_action_xyzg=np.asarray([token_offset, 0.1, -0.2, -1.0], dtype=np.float32),
            raw_pred_action=np.asarray([token_offset, 0.1, -0.2, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
            target_token_ids=np.asarray([11, 12, 13], dtype=np.int64),
            target_token_probs=np.asarray([0.8, 0.6, 0.7], dtype=np.float32),
            image_patch_attributions=np.asarray(
                [
                    [[0.7, 0.3], [0.0, 0.0]],
                    [[0.2, 0.8], [0.0, 0.0]],
                    [[0.5, 0.5], [0.0, 0.0]],
                ],
                dtype=np.float32,
            ),
            text_token_attributions=np.asarray(
                [
                    [0.6, 0.4],
                    [0.5, 0.5],
                    [0.3, 0.7],
                ],
                dtype=np.float32,
            ),
            mean_image_patch_attribution=np.asarray([[0.46666667, 0.53333336], [0.0, 0.0]], dtype=np.float32),
            mean_text_token_attribution=np.asarray([0.46666667, 0.53333336], dtype=np.float32),
            text_token_ids=np.asarray([101, 102], dtype=np.int64),
            text_tokens=["place", "stone"],
            prompt=f"In: {instruction}",
            task_text=instruction.lower(),
            target_token_bin_indices=np.asarray([101, 102, 103], dtype=np.int64),
            target_token_bin_centers=np.asarray([-0.75, 0.0, 0.75], dtype=np.float32),
        )


class _FakeInterventionAdapter(_FakeModelAdapter):
    def patch_occlusion(
        self,
        image: np.ndarray,
        instruction: str,
        baseline_step: LocalExplanationStep,
        top_k: int,
    ) -> InterventionScan:
        del image, instruction, baseline_step, top_k
        return InterventionScan(
            effect_map=np.asarray([[0.30, 0.10], [0.22, 0.05]], dtype=np.float32),
            top_candidates=[
                InterventionCandidateEffect(
                    index=0,
                    label="patch (0, 0)",
                    score=0.30,
                    pred_action_xyzg=np.asarray([0.5, 0.1, -0.2, -1.0], dtype=np.float32),
                    raw_pred_action=np.asarray([0.5, 0.1, -0.2, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
                    target_token_ids=np.asarray([21, 22, 23], dtype=np.int64),
                    target_token_probs=np.asarray([0.7, 0.6, 0.4], dtype=np.float32),
                    target_token_bin_indices=np.asarray([201, 202, 203], dtype=np.int64),
                    target_token_bin_centers=np.asarray([-0.5, 0.1, 0.9], dtype=np.float32),
                )
            ],
        )

    def text_masking(
        self,
        image: np.ndarray,
        instruction: str,
        baseline_step: LocalExplanationStep,
        top_k: int,
    ) -> InterventionScan:
        del image, baseline_step, top_k
        task_text = instruction.lower()
        stone_start = task_text.find("stone")
        stone_end = stone_start + len("stone") if stone_start >= 0 else -1
        return InterventionScan(
            effect_map=np.asarray([0.25, 0.15], dtype=np.float32),
            top_candidates=[
                InterventionCandidateEffect(
                    index=0,
                    label="stone",
                    score=0.25,
                    pred_action_xyzg=np.asarray([0.4, 0.1, -0.2, -1.0], dtype=np.float32),
                    raw_pred_action=np.asarray([0.4, 0.1, -0.2, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
                    target_token_ids=np.asarray([31, 32, 33], dtype=np.int64),
                    target_token_probs=np.asarray([0.6, 0.5, 0.4], dtype=np.float32),
                    target_token_bin_indices=np.asarray([301, 302, 303], dtype=np.int64),
                    target_token_bin_centers=np.asarray([-0.3, 0.2, 0.8], dtype=np.float32),
                    task_char_start=None if stone_start < 0 else stone_start,
                    task_char_end=None if stone_start < 0 else stone_end,
                )
            ],
        )

    def minimal_counterfactual_edits(
        self,
        image: np.ndarray,
        instruction: str,
        baseline_step: LocalExplanationStep,
        patch_occlusion: InterventionScan | None,
        text_masking: InterventionScan | None,
        max_edits: int,
    ) -> tuple[list[CounterfactualEdit], bool]:
        del image, instruction, baseline_step, patch_occlusion, text_masking, max_edits
        return (
            [
                CounterfactualEdit(
                    step_rank=1,
                    edit_type="patch",
                    index=0,
                    label="patch (0, 0)",
                    single_effect_score=0.30,
                    cumulative_sequence_logprob=-1.2,
                    cumulative_sequence_logprob_drop=0.30,
                    changed_prediction=False,
                    pred_action_xyzg=np.asarray([0.5, 0.1, -0.2, -1.0], dtype=np.float32),
                    raw_pred_action=np.asarray([0.5, 0.1, -0.2, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
                    target_token_ids=np.asarray([21, 22, 23], dtype=np.int64),
                    target_token_probs=np.asarray([0.7, 0.6, 0.4], dtype=np.float32),
                    target_token_bin_indices=np.asarray([201, 202, 203], dtype=np.int64),
                    target_token_bin_centers=np.asarray([-0.5, 0.1, 0.9], dtype=np.float32),
                ),
                CounterfactualEdit(
                    step_rank=2,
                    edit_type="text",
                    index=0,
                    label="stone",
                    single_effect_score=0.25,
                    cumulative_sequence_logprob=-1.7,
                    cumulative_sequence_logprob_drop=0.80,
                    changed_prediction=True,
                    pred_action_xyzg=np.asarray([0.9, 0.0, -0.1, 1.0], dtype=np.float32),
                    raw_pred_action=np.asarray([0.9, 0.0, -0.1, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                    target_token_ids=np.asarray([41, 42, 43], dtype=np.int64),
                    target_token_probs=np.asarray([0.3, 0.2, 0.1], dtype=np.float32),
                    target_token_bin_indices=np.asarray([401, 402, 403], dtype=np.int64),
                    target_token_bin_centers=np.asarray([-0.1, 0.4, 0.95], dtype=np.float32),
                ),
            ],
            True,
        )


class _FakeSignedInterventionAdapter(_FakeInterventionAdapter):
    def __init__(self) -> None:
        super().__init__()
        self._patch_call_idx = 0

    def patch_occlusion(
        self,
        image: np.ndarray,
        instruction: str,
        baseline_step: LocalExplanationStep,
        top_k: int,
    ) -> InterventionScan:
        del image, instruction, baseline_step, top_k
        effect_maps = [
            np.asarray([[2.0, -1.0], [0.0, 0.5]], dtype=np.float32),
            np.asarray([[4.0, -2.0], [0.0, 1.0]], dtype=np.float32),
        ]
        effect_map = effect_maps[min(self._patch_call_idx, len(effect_maps) - 1)]
        self._patch_call_idx += 1
        return InterventionScan(
            effect_map=effect_map,
            top_candidates=[
                InterventionCandidateEffect(
                    index=0,
                    label="patch (0, 0)",
                    score=float(effect_map[0, 0]),
                    pred_action_xyzg=np.asarray([0.5, 0.1, -0.2, -1.0], dtype=np.float32),
                    raw_pred_action=np.asarray([0.5, 0.1, -0.2, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
                    target_token_ids=np.asarray([21, 22, 23], dtype=np.int64),
                    target_token_probs=np.asarray([0.7, 0.6, 0.4], dtype=np.float32),
                    target_token_bin_indices=np.asarray([201, 202, 203], dtype=np.int64),
                    target_token_bin_centers=np.asarray([-0.5, 0.1, 0.9], dtype=np.float32),
                )
            ],
        )


class _FakeCausalAdapter(_FakeModelAdapter):
    def causal_localization(
        self,
        image: np.ndarray,
        instruction: str,
        baseline_step: LocalExplanationStep,
        corruption_type: str,
        corruption_index: int | None,
        per_cross_attention: bool,
    ) -> CausalLocalizationStep:
        del image, instruction, per_cross_attention
        return CausalLocalizationStep(
            baseline_pred_action_xyzg=np.asarray(baseline_step.pred_action_xyzg, dtype=np.float32),
            baseline_raw_pred_action=np.asarray(baseline_step.raw_pred_action, dtype=np.float32),
            baseline_target_token_ids=np.asarray(baseline_step.target_token_ids, dtype=np.int64),
            baseline_target_token_probs=np.asarray(baseline_step.target_token_probs, dtype=np.float32),
            clean_sequence_logprob=-1.1,
            corruption_type=corruption_type,
            corruption_index=0 if corruption_index is None else int(corruption_index),
            corruption_label="patch (0, 0)",
            corrupted_pred_action_xyzg=np.asarray([0.2, 0.1, -0.2, -1.0], dtype=np.float32),
            corrupted_raw_pred_action=np.asarray([0.2, 0.1, -0.2, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
            corrupted_target_token_ids=np.asarray([51, 52, 53], dtype=np.int64),
            corrupted_target_token_probs=np.asarray([0.4, 0.3, 0.2], dtype=np.float32),
            corrupted_sequence_logprob=-2.2,
            layer_restoration_scores=np.asarray([0.1, 0.6, 0.2], dtype=np.float32),
            head_restoration_scores=np.asarray(
                [
                    [0.1, 0.2],
                    [0.7, 0.4],
                    [0.3, 0.2],
                ],
                dtype=np.float32,
            ),
            layer_labels=["layer_0", "layer_1", "layer_2"],
            cross_attention_restoration_scores=np.asarray([0.4], dtype=np.float32),
            cross_attention_labels=["cross_block_0"],
            prompt=baseline_step.prompt,
            task_text=baseline_step.task_text,
            text_token_ids=np.asarray(baseline_step.text_token_ids, dtype=np.int64),
            text_tokens=list(baseline_step.text_tokens),
        )


class ExplainabilityPipelineTest(unittest.TestCase):
    def _write_dataset(self, dataset_path: Path) -> None:
        with h5py.File(dataset_path, "w") as handle:
            data = handle.create_group("data")
            demo = data.create_group("demo_0")
            demo.create_dataset(
                "actions",
                data=np.asarray(
                    [
                        [0.0, 0.1, -0.2, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.1, -0.2, 0.0, 0.0, 0.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
            )
            demo.create_dataset(
                "obs/agentview_image",
                data=np.asarray(
                    [
                        np.full((4, 4, 3), 20, dtype=np.uint8),
                        np.full((4, 4, 3), 80, dtype=np.uint8),
                    ]
                ),
                compression="gzip",
            )
            board_state = np.zeros((2, 5, 5, 3), dtype=np.float32)
            board_state[1, 2, 3, 1] = 1.0
            demo.create_dataset("obs/board_state", data=board_state)
            demo.create_dataset("stone_color", data=np.bytes_("black"))

    def test_collect_and_roundtrip_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset_path = tmp_path / "source_go.hdf5"
            trace_path = tmp_path / "local_trace.npz"
            self._write_dataset(dataset_path)

            clips = GoHDF5DatasetAdapter().load_episode_clips(
                dataset_path=dataset_path,
                demos=None,
                start=0,
                num_demos=0,
                stride=1,
                max_steps=0,
            )
            self.assertEqual(len(clips), 1)
            self.assertIsInstance(clips[0], EpisodeClip)

            traces = collect_episode_traces(clips=clips, model_adapter=_FakeModelAdapter())
            self.assertEqual(len(traces), 1)
            self.assertEqual(traces[0].attention_grids.shape, (2, 2, 2))
            self.assertEqual(traces[0].text_token_attributions.shape, (2, 3, 2))

            save_trace_file(
                trace_path=trace_path,
                dataset_path=dataset_path,
                clips=clips,
                traces=traces,
                checkpoint="/tmp/fake-openvla",
                prompt_style="openvla",
                dataset_adapter="go-hdf5",
                model_adapter="openvla",
            )

            bundle = load_trace_file(trace_path)
            self.assertEqual(bundle.dataset_adapter, "go-hdf5")
            self.assertEqual(bundle.model_adapter, "openvla")
            self.assertEqual(bundle.demo_keys, ["demo_0"])
            self.assertEqual(bundle.traces["demo_0"].target_token_probs.shape, (2, 3))
            self.assertEqual(bundle.traces["demo_0"].text_tokens, ["place", "stone"])

            manifest = trace_manifest(
                dataset_path=dataset_path,
                checkpoint="/tmp/fake-openvla",
                prompt_style="openvla",
                dataset_adapter="go-hdf5",
                model_adapter="openvla",
                clips=clips,
                traces=traces,
            )
            self.assertEqual(manifest["trace_format"], "openvla_local_explanations_v1")
            self.assertEqual(manifest["num_episodes"], 1)

    def test_collect_and_roundtrip_intervention_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset_path = tmp_path / "source_go.hdf5"
            trace_path = tmp_path / "intervention_trace.npz"
            self._write_dataset(dataset_path)

            clips = GoHDF5DatasetAdapter().load_episode_clips(
                dataset_path=dataset_path,
                demos=None,
                start=0,
                num_demos=0,
                stride=1,
                max_steps=0,
            )
            traces = collect_episode_intervention_traces(clips=clips, model_adapter=_FakeInterventionAdapter())
            self.assertEqual(len(traces), 1)
            self.assertEqual(traces[0].steps[0].patch_occlusion.effect_map.shape, (2, 2))
            self.assertEqual(len(traces[0].steps[0].counterfactual_edits), 2)

            save_intervention_trace_file(
                trace_path=trace_path,
                dataset_path=dataset_path,
                clips=clips,
                traces=traces,
                checkpoint="/tmp/fake-openvla",
                prompt_style="openvla",
                dataset_adapter="go-hdf5",
                model_adapter="openvla",
            )

            bundle = load_intervention_trace_file(trace_path)
            self.assertEqual(bundle.trace_format, "openvla_intervention_tests_v1")
            self.assertEqual(bundle.traces["demo_0"].steps[0].text_masking.top_candidates[0].label, "stone")
            np.testing.assert_array_equal(
                bundle.traces["demo_0"].steps[0].baseline_target_token_bin_indices,
                np.asarray([101, 102, 103], dtype=np.int64),
            )

            manifest = intervention_trace_manifest(
                dataset_path=dataset_path,
                checkpoint="/tmp/fake-openvla",
                prompt_style="openvla",
                dataset_adapter="go-hdf5",
                model_adapter="openvla",
                clips=clips,
                traces=traces,
            )
            self.assertEqual(manifest["trace_format"], "openvla_intervention_tests_v1")
            self.assertTrue(manifest["episodes"][0]["has_counterfactual"])

    def test_export_intervention_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset_path = tmp_path / "source_go.hdf5"
            report_dir = tmp_path / "intervention_report"
            self._write_dataset(dataset_path)

            clips = GoHDF5DatasetAdapter().load_episode_clips(
                dataset_path=dataset_path,
                demos=None,
                start=0,
                num_demos=0,
                stride=1,
                max_steps=0,
            )
            traces = collect_episode_intervention_traces(clips=clips, model_adapter=_FakeInterventionAdapter())
            manifest = export_intervention_report(output_dir=report_dir, clips=clips, traces=traces)

            episode = manifest["episodes"][0]
            demo_dir = Path(episode["report_dir"])
            step_json = demo_dir / "step_000.json"
            step_md = demo_dir / "step_000.md"
            patch_png = demo_dir / "step_000_intervention_panel.png"
            self.assertTrue(step_json.is_file())
            self.assertTrue(step_md.is_file())
            self.assertTrue(patch_png.is_file())

            payload = json.loads(step_json.read_text())
            self.assertTrue(payload["counterfactual_success"])
            self.assertEqual(payload["patch_occlusion"]["top_candidates"][0]["label"], "patch (0, 0)")
            self.assertEqual(payload["text_masking"]["top_candidates"][0]["label"], "stone")
            self.assertIn("masked_query", payload["text_masking"]["top_candidates"][0])
            self.assertIn("Original query:", step_md.read_text())
            self.assertEqual(payload["counterfactual_edits"][1]["predicted_token_ids"], [41, 42, 43])
            self.assertEqual(payload["baseline_target_token_bin_indices"], [101, 102, 103])
            self.assertEqual(
                payload["patch_occlusion"]["top_candidates"][0]["predicted_token_bin_centers"],
                [-0.5, 0.10000000149011612, 0.8999999761581421],
            )
            self.assertIn("predicted_bin_centers", step_md.read_text())
            self.assertFalse(payload["patch_occlusion_visualization"]["bilinear_interpolation"])

            panel = np.asarray(Image.open(patch_png))
            np.testing.assert_array_equal(panel[13, 14], np.asarray([20, 20, 20], dtype=np.uint8))
            np.testing.assert_array_equal(panel[13, 30], np.asarray([0, 0, 0], dtype=np.uint8))

    def test_export_intervention_report_cross_step_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset_path = tmp_path / "source_go.hdf5"
            report_dir = tmp_path / "intervention_report"
            self._write_dataset(dataset_path)

            clips = GoHDF5DatasetAdapter().load_episode_clips(
                dataset_path=dataset_path,
                demos=None,
                start=0,
                num_demos=0,
                stride=1,
                max_steps=0,
            )
            clips[0].images[:] = 20
            traces = collect_episode_intervention_traces(clips=clips, model_adapter=_FakeSignedInterventionAdapter())
            manifest = export_intervention_report(
                output_dir=report_dir,
                clips=clips,
                traces=traces,
                cross_step_comparison="episode",
                bilinear_interpolation=True,
            )

            episode = manifest["episodes"][0]
            self.assertEqual(manifest["patch_occlusion_visualization"]["mode"], "cross-step")
            self.assertEqual(episode["patch_occlusion_visualization"]["scope"], "episode")
            self.assertAlmostEqual(float(episode["patch_occlusion_visualization"]["scale_peak"]), 4.0)
            self.assertTrue(manifest["patch_occlusion_visualization"]["bilinear_interpolation"])

            demo_dir = Path(episode["report_dir"])
            step_payload = json.loads((demo_dir / "step_000.json").read_text())
            self.assertEqual(step_payload["patch_occlusion_visualization"]["mode"], "cross-step")
            self.assertAlmostEqual(float(step_payload["patch_occlusion_visualization"]["scale_peak"]), 4.0)
            self.assertTrue(step_payload["patch_occlusion_visualization"]["bilinear_interpolation"])

            step0_panel = np.asarray(Image.open(demo_dir / "step_000_intervention_panel.png"))
            step1_panel = np.asarray(Image.open(demo_dir / "step_001_intervention_panel.png"))

            left_x = 12
            right_x = 28
            top_y = 12
            bottom_y = 15

            step0_positive_delta = np.abs(step0_panel[top_y, right_x].astype(np.int16) - step0_panel[top_y, left_x].astype(np.int16)).sum()
            step1_positive_delta = np.abs(step1_panel[top_y, right_x].astype(np.int16) - step1_panel[top_y, left_x].astype(np.int16)).sum()
            self.assertLess(int(step0_positive_delta), int(step1_positive_delta))

            positive_pixel = step0_panel[top_y, right_x].astype(np.int16)
            negative_pixel = step0_panel[top_y, right_x + 3].astype(np.int16)
            zero_overlay = step0_panel[bottom_y, right_x].astype(np.int16)
            zero_original = step0_panel[bottom_y, left_x].astype(np.int16)

            self.assertGreater(int(positive_pixel[0]), int(positive_pixel[2]))
            self.assertGreater(int(negative_pixel[2]), int(negative_pixel[0]))
            np.testing.assert_array_equal(zero_overlay, zero_original)

    def test_export_intervention_report_text_only_still_writes_raw_panels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset_path = tmp_path / "source_go.hdf5"
            report_dir = tmp_path / "intervention_report"
            self._write_dataset(dataset_path)

            clips = GoHDF5DatasetAdapter().load_episode_clips(
                dataset_path=dataset_path,
                demos=None,
                start=0,
                num_demos=0,
                stride=1,
                max_steps=0,
            )
            traces = collect_episode_intervention_traces(
                clips=clips,
                model_adapter=_FakeInterventionAdapter(),
                run_patch_occlusion=False,
                run_text_masking=True,
                run_counterfactual=False,
            )
            manifest = export_intervention_report(output_dir=report_dir, clips=clips, traces=traces)

            demo_dir = Path(manifest["episodes"][0]["report_dir"])
            panel_path = demo_dir / "step_000_intervention_panel.png"
            payload = json.loads((demo_dir / "step_000.json").read_text())

            self.assertTrue(panel_path.is_file())
            self.assertEqual(payload["artifacts"]["intervention_panel"], "step_000_intervention_panel.png")
            self.assertIsNone(payload["artifacts"]["patch_occlusion_panel"])
            np.testing.assert_array_equal(np.asarray(Image.open(panel_path)), clips[0].images[0])

    def test_collect_and_roundtrip_causal_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset_path = tmp_path / "source_go.hdf5"
            trace_path = tmp_path / "causal_trace.npz"
            self._write_dataset(dataset_path)

            clips = GoHDF5DatasetAdapter().load_episode_clips(
                dataset_path=dataset_path,
                demos=None,
                start=0,
                num_demos=0,
                stride=1,
                max_steps=0,
            )
            traces = collect_episode_causal_traces(clips=clips, model_adapter=_FakeCausalAdapter())
            self.assertEqual(len(traces), 1)
            self.assertEqual(traces[0].steps[0].head_restoration_scores.shape, (3, 2))

            save_causal_trace_file(
                trace_path=trace_path,
                dataset_path=dataset_path,
                clips=clips,
                traces=traces,
                checkpoint="/tmp/fake-openvla",
                prompt_style="openvla",
                dataset_adapter="go-hdf5",
                model_adapter="openvla",
            )

            bundle = load_causal_trace_file(trace_path)
            self.assertEqual(bundle.trace_format, "openvla_causal_localization_v1")
            self.assertEqual(bundle.traces["demo_0"].steps[0].cross_attention_labels, ["cross_block_0"])

            manifest = causal_trace_manifest(
                dataset_path=dataset_path,
                checkpoint="/tmp/fake-openvla",
                prompt_style="openvla",
                dataset_adapter="go-hdf5",
                model_adapter="openvla",
                clips=clips,
                traces=traces,
            )
            self.assertEqual(manifest["trace_format"], "openvla_causal_localization_v1")
            self.assertEqual(manifest["episodes"][0]["layer_count"], 3)

    def test_causal_heatmap_colormap_uses_fixed_restoration_scale(self) -> None:
        normalized = _normalize_restoration_scores(
            np.asarray([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=np.float32)
        )
        np.testing.assert_allclose(
            normalized,
            np.asarray([[0.0, 0.0, 0.5, 1.0, 1.0]], dtype=np.float32),
        )

        colors = _apply_colormap(normalized)
        blue = colors[0, 1].astype(np.int16)
        neutral = colors[0, 2].astype(np.int16)
        red = colors[0, 3].astype(np.int16)

        self.assertGreater(int(blue[2]), int(blue[0]))
        self.assertLess(abs(int(neutral[0]) - int(neutral[2])), 20)
        self.assertGreater(int(red[0]), int(red[2]))

    def test_export_causal_localization_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset_path = tmp_path / "source_go.hdf5"
            report_dir = tmp_path / "causal_report"
            self._write_dataset(dataset_path)

            clips = GoHDF5DatasetAdapter().load_episode_clips(
                dataset_path=dataset_path,
                demos=None,
                start=0,
                num_demos=0,
                stride=1,
                max_steps=0,
            )
            traces = collect_episode_causal_traces(clips=clips, model_adapter=_FakeCausalAdapter())
            manifest = export_causal_localization_report(output_dir=report_dir, clips=clips, traces=traces)

            episode = manifest["episodes"][0]
            demo_dir = Path(episode["report_dir"])
            step_json = demo_dir / "step_000.json"
            step_md = demo_dir / "step_000.md"
            heatmap_png = demo_dir / "step_000_restoration_heatmap.png"
            self.assertTrue(step_json.is_file())
            self.assertTrue(step_md.is_file())
            self.assertTrue(heatmap_png.is_file())

            payload = json.loads(step_json.read_text())
            self.assertEqual(payload["best_layer"], 1)
            self.assertEqual(payload["best_head"], {"layer": 1, "head": 0})
            self.assertEqual(payload["cross_attention"][0]["label"], "cross_block_0")

            step_markdown = step_md.read_text()
            self.assertIn("Color scale in PNG: `blue <= -1`, `white = 0`, `red >= +1`", step_markdown)
            self.assertIn("Head grid shape: `3 layers x 2 heads`", step_markdown)

            heatmap = np.asarray(Image.open(heatmap_png))
            upper_right = heatmap[24:70, -200:-10]
            self.assertTrue(np.any(upper_right[..., 2] > upper_right[..., 0]))
            self.assertTrue(np.any(upper_right[..., 0] > upper_right[..., 2]))

    def test_export_local_explanation_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset_path = tmp_path / "source_go.hdf5"
            report_dir = tmp_path / "local_report"
            self._write_dataset(dataset_path)

            clips = GoHDF5DatasetAdapter().load_episode_clips(
                dataset_path=dataset_path,
                demos=None,
                start=0,
                num_demos=0,
                stride=1,
                max_steps=0,
            )
            traces = collect_episode_traces(clips=clips, model_adapter=_FakeModelAdapter())
            manifest = export_local_explanation_report(
                output_dir=report_dir,
                clips=clips,
                traces=traces,
                token_top_k=2,
            )

            self.assertEqual(len(manifest["episodes"]), 1)
            episode = manifest["episodes"][0]
            self.assertTrue((report_dir / "README.md").is_file())
            self.assertTrue(Path(episode["summary_path"]).is_file())
            self.assertTrue(Path(episode["readme_path"]).is_file())

            demo_dir = Path(episode["report_dir"])
            step_json = demo_dir / "step_000.json"
            step_md = demo_dir / "step_000.md"
            patch_png = demo_dir / "step_000_patch_attribution.png"
            self.assertTrue(step_json.is_file())
            self.assertTrue(step_md.is_file())
            self.assertTrue(patch_png.is_file())

            payload = json.loads(step_json.read_text())
            self.assertEqual(payload["target_output_tokens"][0]["token_id"], 11)
            self.assertAlmostEqual(payload["target_output_tokens"][0]["probability"], 0.8, places=6)
            self.assertEqual(payload["mean_text_token_attribution"][0]["token"], "stone")
            self.assertEqual(payload["per_output_token_text_attribution"][0]["top_instruction_tokens"][0]["token"], "place")
            self.assertEqual(payload["artifacts"]["patch_attribution_panel"], "step_000_patch_attribution.png")

    def test_hdf5_adapter_uses_rlds_style_frame_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset_path = tmp_path / "source_go.hdf5"

            with h5py.File(dataset_path, "w") as handle:
                data = handle.create_group("data")
                demo = data.create_group("demo_0")
                demo.create_dataset(
                    "actions",
                    data=np.asarray(
                        [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.2, 0.0, 0.0, 0.0],
                            [0.2, 0.0, 0.0, 0.0],
                            [0.2, 0.0, 0.0, 0.0],
                            [0.5, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                        dtype=np.float32,
                    ),
                )
                demo.create_dataset(
                    "obs/agentview_image",
                    data=np.asarray(
                        [np.full((4, 4, 3), fill_value=10 * idx, dtype=np.uint8) for idx in range(6)]
                    ),
                    compression="gzip",
                )
                board_state = np.zeros((6, 5, 5, 4), dtype=np.float32)
                board_state[-1, 1, 4, 1] = 1.0
                demo.create_dataset("obs/board_state", data=board_state)
                demo.create_dataset("stone_color", data=np.bytes_("black"))

            clips = GoHDF5DatasetAdapter().load_episode_clips(
                dataset_path=dataset_path,
                demos=None,
                start=0,
                num_demos=0,
                stride=1,
                max_steps=0,
            )

            self.assertEqual(len(clips), 1)
            clip = clips[0]
            np.testing.assert_array_equal(clip.frame_indices, np.asarray([0, 4, 5], dtype=np.int32))
            np.testing.assert_allclose(
                clip.gt_actions,
                np.asarray(
                    [
                        [1.0, 0.0, 0.0, -1.0],
                        [0.5, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
            )
            self.assertEqual(clip.images.shape[0], 3)


if __name__ == "__main__":
    unittest.main()
