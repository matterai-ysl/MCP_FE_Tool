"""
CIF structure featurization based on pymatgen and matminer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from matminer.featurizers.structure import DensityFeatures, GlobalSymmetryFeatures, StructuralComplexity
from pymatgen.core import Structure

from .feature_generator import MaterialsFeatureGenerator


DEFAULT_STRUCTURE_FEATURE_TYPES = ["basic", "symmetry", "density", "complexity"]
DEFAULT_CIF_COMPOSITION_FEATURE_TYPES = [
    "element_property",
    "stoichiometry",
    "valence_orbital",
    "element_amount",
]


@dataclass
class CifFeaturizer:
    structure_feature_types: Sequence[str] = tuple(DEFAULT_STRUCTURE_FEATURE_TYPES)
    composition_feature_types: Sequence[str] = tuple(DEFAULT_CIF_COMPOSITION_FEATURE_TYPES)

    def __post_init__(self) -> None:
        supported_structure_types = {"basic", "symmetry", "density", "complexity"}
        unsupported = sorted(set(self.structure_feature_types) - supported_structure_types)
        if unsupported:
            raise ValueError(f"Unsupported CIF structure feature types: {unsupported}")

    def featurize_files(
        self,
        files_by_name: Dict[str, str],
        ordered_filenames: List[str] | None = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
        filenames = ordered_filenames or sorted(files_by_name)
        structures: List[Structure | None] = []
        warnings: List[str] = []

        for filename in filenames:
            try:
                structures.append(Structure.from_file(files_by_name[filename]))
            except Exception as exc:
                structures.append(None)
                warnings.append(f"Failed to parse CIF file '{filename}': {exc}")

        frames: List[pd.DataFrame] = []
        if "basic" in self.structure_feature_types:
            frames.append(self._basic_features(structures))
        if "symmetry" in self.structure_feature_types:
            frames.append(self._matminer_features("symmetry", GlobalSymmetryFeatures(), structures, warnings))
        if "density" in self.structure_feature_types:
            frames.append(self._matminer_features("density", DensityFeatures(), structures, warnings))
        if "complexity" in self.structure_feature_types:
            frames.append(self._matminer_features("complexity", StructuralComplexity(), structures, warnings))
        if self.composition_feature_types:
            frames.append(self._composition_features(structures, filenames, warnings))

        features = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=range(len(filenames)))
        features.index = filenames
        features = features.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

        summary = {
            "input_count": len(filenames),
            "parsed_count": sum(structure is not None for structure in structures),
            "failed_count": sum(structure is None for structure in structures),
            "structure_feature_types": list(self.structure_feature_types),
            "composition_feature_types": list(self.composition_feature_types),
            "feature_count": int(features.shape[1]),
        }
        return features, summary, warnings

    def _basic_features(self, structures: List[Structure | None]) -> pd.DataFrame:
        rows: List[Dict[str, float]] = []
        for structure in structures:
            if structure is None:
                rows.append({
                    "structure__num_sites": 0.0,
                    "structure__density": 0.0,
                    "structure__volume": 0.0,
                    "structure__volume_per_atom": 0.0,
                    "structure__a": 0.0,
                    "structure__b": 0.0,
                    "structure__c": 0.0,
                    "structure__alpha": 0.0,
                    "structure__beta": 0.0,
                    "structure__gamma": 0.0,
                })
                continue

            lattice = structure.lattice
            num_sites = float(len(structure))
            rows.append({
                "structure__num_sites": num_sites,
                "structure__density": float(structure.density),
                "structure__volume": float(structure.volume),
                "structure__volume_per_atom": float(structure.volume / num_sites) if num_sites else 0.0,
                "structure__a": float(lattice.a),
                "structure__b": float(lattice.b),
                "structure__c": float(lattice.c),
                "structure__alpha": float(lattice.alpha),
                "structure__beta": float(lattice.beta),
                "structure__gamma": float(lattice.gamma),
            })
        return pd.DataFrame(rows)

    def _matminer_features(
        self,
        prefix: str,
        featurizer: Any,
        structures: List[Structure | None],
        warnings: List[str],
    ) -> pd.DataFrame:
        labels = [f"{prefix}__{label}" for label in featurizer.feature_labels()]
        rows: List[List[float]] = []
        for index, structure in enumerate(structures):
            if structure is None:
                rows.append([np.nan] * len(labels))
                continue
            try:
                values = featurizer.featurize(structure)
            except Exception as exc:
                warnings.append(f"{prefix} features failed for row {index}: {exc}")
                values = [np.nan] * len(labels)
            rows.append(values)
        return pd.DataFrame(rows, columns=labels)

    def _composition_features(
        self,
        structures: List[Structure | None],
        filenames: List[str],
        warnings: List[str],
    ) -> pd.DataFrame:
        formulas = [
            structure.composition.reduced_formula if structure is not None else None
            for structure in structures
        ]
        formula_df = pd.DataFrame({"composition_formula": formulas})
        generator = MaterialsFeatureGenerator()
        generator.data = formula_df
        try:
            composition_features = generator.generate_composition_features(
                "composition_formula",
                feature_types=list(self.composition_feature_types),
            )
        except Exception as exc:
            warnings.append(f"Composition features failed for CIF-derived formulas: {exc}")
            composition_features = pd.DataFrame(index=range(len(filenames)))
        return composition_features.add_prefix("composition__")
