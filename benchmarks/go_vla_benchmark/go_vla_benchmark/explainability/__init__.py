"""Reusable explainability pipeline components for the Go VLA benchmark."""

from .causal_localization import (
    CAUSAL_SCHEMA_VERSION,
    CAUSAL_TRACE_FORMAT,
    CausalLocalizationModelAdapter,
    CausalLocalizationStep,
    CausalTraceBundle,
    EpisodeCausalTrace,
    causal_trace_manifest,
    collect_episode_causal_traces,
    load_causal_trace_file,
    save_causal_trace_file,
)
from .core import (
    EpisodeClip,
    EpisodeTrace,
    LocalExplanationModelAdapter,
    LocalExplanationStep,
    TraceBundle,
    collect_episode_traces,
    compute_episode_scores,
    extract_xyzg,
    load_trace_file,
    resolve_dataset_path,
    resolve_optional_path,
    resolve_repo_relative_path,
    save_trace_file,
    select_ranked_traces,
    select_top_k_traces_preserving_order,
    trace_manifest,
)
from .go_hdf5 import GoHDF5DatasetAdapter
from .interventions import (
    INTERVENTION_SCHEMA_VERSION,
    INTERVENTION_TRACE_FORMAT,
    CounterfactualEdit,
    EpisodeInterventionTrace,
    InterventionCandidateEffect,
    InterventionModelAdapter,
    InterventionScan,
    InterventionStepTrace,
    InterventionTraceBundle,
    collect_episode_intervention_traces,
    intervention_trace_manifest,
    load_intervention_trace_file,
    match_intervention_trace_to_clips,
    save_intervention_trace_file,
)
from .openvla_adapter import (
    OpenVLACausalLocalizationAdapter,
    OpenVLAExplainabilityAdapter,
    OpenVLAInterventionAdapter,
    OpenVLALocalExplanationAdapter,
)
from .reporting_causal import export_causal_localization_report
from .reporting_interventions import export_intervention_report
from .reporting import export_local_explanation_report

DATASET_ADAPTERS = {
    "go-hdf5": GoHDF5DatasetAdapter,
}

MODEL_ADAPTERS = {
    "openvla": OpenVLALocalExplanationAdapter,
}

INTERVENTION_MODEL_ADAPTERS = {
    "openvla": OpenVLAInterventionAdapter,
}

CAUSAL_MODEL_ADAPTERS = {
    "openvla": OpenVLACausalLocalizationAdapter,
}

__all__ = [
    "CAUSAL_MODEL_ADAPTERS",
    "CAUSAL_SCHEMA_VERSION",
    "CAUSAL_TRACE_FORMAT",
    "CausalLocalizationModelAdapter",
    "CausalLocalizationStep",
    "CausalTraceBundle",
    "EpisodeCausalTrace",
    "DATASET_ADAPTERS",
    "INTERVENTION_MODEL_ADAPTERS",
    "INTERVENTION_SCHEMA_VERSION",
    "INTERVENTION_TRACE_FORMAT",
    "MODEL_ADAPTERS",
    "CounterfactualEdit",
    "EpisodeClip",
    "EpisodeInterventionTrace",
    "EpisodeTrace",
    "GoHDF5DatasetAdapter",
    "InterventionCandidateEffect",
    "InterventionModelAdapter",
    "InterventionScan",
    "InterventionStepTrace",
    "InterventionTraceBundle",
    "LocalExplanationModelAdapter",
    "LocalExplanationStep",
    "OpenVLACausalLocalizationAdapter",
    "OpenVLAExplainabilityAdapter",
    "OpenVLAInterventionAdapter",
    "OpenVLALocalExplanationAdapter",
    "causal_trace_manifest",
    "collect_episode_causal_traces",
    "collect_episode_intervention_traces",
    "TraceBundle",
    "collect_episode_traces",
    "compute_episode_scores",
    "extract_xyzg",
    "export_causal_localization_report",
    "export_intervention_report",
    "export_local_explanation_report",
    "intervention_trace_manifest",
    "load_causal_trace_file",
    "load_intervention_trace_file",
    "load_trace_file",
    "match_intervention_trace_to_clips",
    "resolve_dataset_path",
    "resolve_optional_path",
    "resolve_repo_relative_path",
    "save_causal_trace_file",
    "save_intervention_trace_file",
    "save_trace_file",
    "select_ranked_traces",
    "select_top_k_traces_preserving_order",
    "trace_manifest",
]
