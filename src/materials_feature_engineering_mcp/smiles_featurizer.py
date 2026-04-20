"""
SMILES featurization utilities based on RDKit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys


RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")


DEFAULT_DESCRIPTOR_NAMES = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
    "RingCount",
    "FractionCSP3",
    "HeavyAtomCount",
    "NumValenceElectrons",
]


@dataclass
class SmilesFeaturizer:
    feature_types: Tuple[str, ...] = ("descriptors", "morgan")
    descriptor_names: Tuple[str, ...] = tuple(DEFAULT_DESCRIPTOR_NAMES)
    morgan_radius: int = 2
    morgan_n_bits: int = 2048

    def __post_init__(self) -> None:
        supported_feature_types = {"descriptors", "morgan", "maccs"}
        unsupported = sorted(set(self.feature_types) - supported_feature_types)
        if unsupported:
            raise ValueError(f"Unsupported SMILES feature types: {unsupported}")

        descriptor_functions = Descriptors.descList
        self._descriptor_map = {name: func for name, func in descriptor_functions}
        unknown_descriptors = sorted(set(self.descriptor_names) - set(self._descriptor_map))
        if unknown_descriptors:
            raise ValueError(f"Unsupported RDKit descriptors: {unknown_descriptors}")

    def featurize_series(self, smiles: pd.Series) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Featurize a SMILES series into deterministic numeric features."""
        mols = smiles.apply(self._to_mol)
        invalid_mask = mols.isna()
        invalid_indices = [int(idx) for idx in mols[invalid_mask].index.tolist()]

        frames: List[pd.DataFrame] = []
        feature_counts: Dict[str, int] = {}

        if "descriptors" in self.feature_types:
            descriptors_df = self._descriptor_features(mols)
            frames.append(descriptors_df)
            feature_counts["descriptors"] = descriptors_df.shape[1]

        if "morgan" in self.feature_types:
            morgan_df = self._morgan_features(mols)
            frames.append(morgan_df)
            feature_counts["morgan"] = morgan_df.shape[1]

        if "maccs" in self.feature_types:
            maccs_df = self._maccs_features(mols)
            frames.append(maccs_df)
            feature_counts["maccs"] = maccs_df.shape[1]

        if frames:
            features = pd.concat(frames, axis=1)
        else:
            features = pd.DataFrame(index=smiles.index)

        metadata = {
            "invalid_count": int(invalid_mask.sum()),
            "invalid_indices": invalid_indices,
            "feature_counts": feature_counts,
        }
        return features, metadata

    def _to_mol(self, value: Any):
        if pd.isna(value):
            return None
        text = str(value).strip()
        if not text:
            return None
        return Chem.MolFromSmiles(text)

    def _descriptor_features(self, mols: pd.Series) -> pd.DataFrame:
        rows: List[Dict[str, float]] = []
        for mol in mols:
            if mol is None:
                rows.append({f"desc__{name}": np.nan for name in self.descriptor_names})
                continue

            row: Dict[str, float] = {}
            for name in self.descriptor_names:
                row[f"desc__{name}"] = float(self._descriptor_map[name](mol))
            rows.append(row)

        return pd.DataFrame(rows, index=mols.index)

    def _morgan_features(self, mols: pd.Series) -> pd.DataFrame:
        columns = [f"morgan__bit_{idx}" for idx in range(self.morgan_n_bits)]
        rows: List[List[int]] = []
        for mol in mols:
            if mol is None:
                rows.append([0] * self.morgan_n_bits)
                continue

            bit_vector = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                self.morgan_radius,
                nBits=self.morgan_n_bits,
            )
            bits = np.zeros((self.morgan_n_bits,), dtype=int)
            DataStructs.ConvertToNumpyArray(bit_vector, bits)
            rows.append(bits.tolist())

        return pd.DataFrame(rows, index=mols.index, columns=columns)

    def _maccs_features(self, mols: pd.Series) -> pd.DataFrame:
        columns = [f"maccs__bit_{idx}" for idx in range(167)]
        rows: List[List[int]] = []
        for mol in mols:
            if mol is None:
                rows.append([0] * 167)
                continue

            bit_vector = MACCSkeys.GenMACCSKeys(mol)
            bits = np.zeros((167,), dtype=int)
            DataStructs.ConvertToNumpyArray(bit_vector, bits)
            rows.append(bits.tolist())

        return pd.DataFrame(rows, index=mols.index, columns=columns)
