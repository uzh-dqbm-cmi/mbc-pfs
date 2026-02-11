from __future__ import annotations

from typing import List

from .base import ModalityFrame
from .cancer_presence import build_cancer_presence_features

# from .report_presence import build_report_presence_features
from .clinical import build_clinical_features
from .diagnosis import build_diagnosis_features
from .ecog import build_ecog_features
from .pdl1_mmr import build_mmr_features, build_pdl1_features
from .local_treatment import build_local_treatment_features
from .treatment import build_treatment_features
from .tumor_markers import build_tumor_marker_features
from .tumor_sites import build_tumor_site_features
from .genomics import build_genomics_features

# from .history import build_history_features

__all__ = [
    "ModalityFrame",
    "build_all_modalities",
]


def build_all_modalities(ctx) -> List[ModalityFrame]:
    frames: List[ModalityFrame] = []
    modality_builders = [
        ("clinical", build_clinical_features),
        ("diagnosis", build_diagnosis_features),
        ("cancer_presence", build_cancer_presence_features),
        # ("report_presence", build_report_presence_features),
        ("tumor_sites", build_tumor_site_features),
        ("tumor_markers", build_tumor_marker_features),
        ("ecog", build_ecog_features),
        ("pdl1", build_pdl1_features),
        ("mmr", build_mmr_features),
        ("treatment", build_treatment_features),
        ("genomics", build_genomics_features),
        ("local_treatment", build_local_treatment_features),
        # ("history", build_history_features),
    ]
    for name, builder in modality_builders:
        print(f"Building modality: {name}")
        frame = builder(ctx)
        print(f"Built modality: {name} with shape {frame.frame.shape}")
        frames.append(frame)
    return frames
