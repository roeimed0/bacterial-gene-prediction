import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis

logger = logging.getLogger(__name__)

__all__ = ["OrfGroupClassifier", "HybridGeneFilter"]

"""
Machine learning classifier for ORF groups.

Loads a trained LightGBM model to filter groups of nested ORFs.
Groups share the same stop codon but have different start codons.
Model predicts: Does this GROUP contain a real gene? (yes/no)
"""


class OrfGroupClassifier:
    """
    Binary classifier for ORF groups using LightGBM.
    Predicts whether a group of nested ORFs contains a real gene.
    """

    def __init__(self):
        """
        Initialize empty classifier.
        Call load() or train() before using.
        """
        self.model: Any = None
        self.feature_names: Optional[List[str]] = None

    def load(self, model_path: str = "../models/orf_classifier_lgb.pkl"):
        """
        Load trained model and feature names from disk.
        Model and feature_names.pkl must be in same directory.
        """
        model_path = Path(model_path)

        # Load the model
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = joblib.load(str(model_path))
        logger.info("Loaded model: %s", {model_path})

        # Load feature names — prefer model-specific file, fall back to shared
        specific = model_path.parent / f"{model_path.stem}_feature_names.pkl"
        shared = model_path.parent / "feature_names.pkl"
        feature_path = specific if specific.exists() else shared
        if feature_path.exists():
            self.feature_names = joblib.load(str(feature_path))
            logger.info("Loaded %d features", len(self.feature_names or []))
        else:
            logger.warning("feature_names not found for %s", model_path.name)
            self.feature_names = None

    def _entropy_from_probs(self, arr, base=2):
        """
        Compute entropy (bits by default) for a 1D array of non-negative numbers.
        Will normalize to probabilities.
        """
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return 0.0
        s = arr.sum()
        if s <= 0:
            return 0.0
        p = arr / s
        p = p[p > 0]
        return -np.sum(p * np.log(p) / np.log(base))

    def extract_group_features(
        self, groups: Dict[str, List[Dict]], genome_id: str, weights: Dict = None
    ) -> pd.DataFrame:
        """
        Extract features from ORF groups for prediction.

        Args:
            groups: Dictionary of {group_id: [list of ORFs]}
            genome_id: Genome identifier (for progress bar)
            weights: Optional weights for start selection score

        Returns:
            DataFrame with one row per group, columns are features
        """
        from tqdm import tqdm

        rows = []
        for group_id, orf_group in tqdm(
            groups.items(), total=len(groups), desc=f"Groups {genome_id}"
        ):
            # Accept both DataFrame (new) and List[Dict] (legacy)
            if isinstance(orf_group, list):
                orf_group = pd.DataFrame(orf_group)
            if len(orf_group) == 0:
                continue

            # .values is faster than .to_numpy(float, na_value=...) for small groups
            combined = orf_group["combined_score"].values.astype(np.float64)
            rbs = orf_group["rbs_score"].values.astype(np.float64)
            codon = orf_group["codon_score"].values.astype(np.float64)
            start = orf_group["start_score"].values.astype(np.float64)
            imm = orf_group["imm_score"].values.astype(np.float64)
            strands = orf_group["strand"].tolist()
            lengths = (
                orf_group["length"].values.astype(np.float64)
                if "length" in orf_group.columns
                else np.ones(len(orf_group))
            )

            if weights is not None:
                cols = orf_group.columns
                ss = (
                    (
                        orf_group["codon_score_norm"].values
                        if "codon_score_norm" in cols
                        else np.zeros(len(orf_group))
                    ).astype(np.float64)
                    * weights.get("codon", 0.0)
                    + (
                        orf_group["imm_score_norm"].values
                        if "imm_score_norm" in cols
                        else np.zeros(len(orf_group))
                    ).astype(np.float64)
                    * weights.get("imm", 0.0)
                    + (
                        orf_group["rbs_score_norm"].values
                        if "rbs_score_norm" in cols
                        else np.zeros(len(orf_group))
                    ).astype(np.float64)
                    * weights.get("rbs", 0.0)
                    + (
                        orf_group["length_score_norm"].values
                        if "length_score_norm" in cols
                        else np.zeros(len(orf_group))
                    ).astype(np.float64)
                    * weights.get("length", 0.0)
                    + (
                        orf_group["start_score_norm"].values
                        if "start_score_norm" in cols
                        else np.zeros(len(orf_group))
                    ).astype(np.float64)
                    * weights.get("start", 0.0)
                )
            else:
                ss = np.zeros(len(orf_group))

            n = len(combined)
            max_combined = combined.max()
            max_rbs = rbs.max()
            max_codon = codon.max()
            max_start = start.max()
            max_ss = ss.max()

            group_features = {
                "group_id": group_id,
                "num_orfs": n,
                "combined_max": max_combined,
                "combined_mean": combined.mean(),
                "combined_std": combined.std() if n > 1 else 0.0,
                "combined_entropy": self._entropy_from_probs(np.maximum(combined, 0)),
                "combined_margin_top2": (
                    (lambda s: s[-1] - s[-2])(np.sort(combined)) if n > 1 else combined[0]
                ),
                "frac_top_orfs": (combined >= 0.8 * max_combined).sum() / n,
                "rbs_max": max_rbs,
                "rbs_mean": rbs.mean(),
                "codon_max": max_codon,
                "codon_mean": codon.mean(),
                "start_max": max_start,
                "start_mean": start.mean(),
                "imm_max": imm.max(),
                "imm_mean": imm.mean(),
                "start_select_max": max_ss,
                "start_select_mean": ss.mean(),
                "strand_plus_frac": strands.count("forward") / n,
                "strand_minus_frac": strands.count("reverse") / n,
                # Relative features
                "rel_combined_mean": (
                    combined / max_combined if max_combined > 0 else np.zeros(n)
                ).mean(),
                "rel_combined_max": (
                    combined / max_combined if max_combined > 0 else np.zeros(n)
                ).max(),
                "rel_rbs_mean": (rbs / max_rbs if max_rbs > 0 else np.zeros(n)).mean(),
                "rel_rbs_max": (rbs / max_rbs if max_rbs > 0 else np.zeros(n)).max(),
                "rel_codon_mean": (codon / max_codon if max_codon > 0 else np.zeros(n)).mean(),
                "rel_codon_max": (codon / max_codon if max_codon > 0 else np.zeros(n)).max(),
                "rel_start_mean": (start / max_start if max_start > 0 else np.zeros(n)).mean(),
                "rel_start_max": (start / max_start if max_start > 0 else np.zeros(n)).max(),
                "rel_start_select_mean": (ss / max_ss if max_ss > 0 else np.zeros(n)).mean(),
                "rel_start_select_max": (ss / max_ss if max_ss > 0 else np.zeros(n)).max(),
                "frac_top_combined": (combined >= 0.95 * max_combined).sum() / n,
                "frac_top_start_select": (ss >= 0.95 * max_ss).sum() / n,
                # Length-based features (#123)
                "top_orf_is_longest": int(combined.argmax() == lengths.argmax()),
                "length_ratio_max_min": float(lengths.max() / max(lengths.min(), 1.0)),
            }

            rows.append(group_features)

        return pd.DataFrame(rows).fillna(0.0)

    def predict_groups(
        self,
        groups: Dict[str, List[Dict]],
        genome_id: str = "unknown",
        weights: Dict = None,
        threshold: float = 0.07,
    ) -> tuple:
        """
        Predict which groups contain real genes.
        Returns both predictions and probabilities.

        Args:
            groups: Dictionary of {group_id: [list of ORFs]}
            genome_id: Genome identifier
            weights: Optional weights for start selection
            threshold: Probability threshold for classification (default 0.1)

        Returns:
            Tuple of (predictions, probabilities, group_ids)
            - predictions: Binary array (1 = real gene, 0 = false positive)
            - probabilities: Probability of being a real gene
            - group_ids: List of group IDs in same order
        """
        # Extract features
        df = self.extract_group_features(groups, genome_id, weights)

        # Resolve feature column order.
        # Models trained on numpy arrays get generic names ("Column_0" …).
        # In that case fall back to self.feature_names (loaded from feature_names.pkl).
        model_features = self.model.feature_name_
        if model_features and model_features[0].startswith("Column_"):
            if self.feature_names:
                model_features = self.feature_names
            else:
                raise ValueError(
                    "Model has generic column names and feature_names.pkl was not loaded. "
                    "Call load() with a path whose directory contains feature_names.pkl."
                )

        missing = [f for f in model_features if f not in df.columns]
        if missing:
            raise ValueError(
                f"Feature mismatch: {len(missing)} feature(s) expected by the "
                f"OrfGroupClassifier model are absent from the extracted feature "
                f"DataFrame. Missing: {sorted(missing)}"
            )

        # Extract feature matrix in the order the model was trained on
        X = df[model_features].values

        # Get probabilities — n_jobs=1 avoids a 1.3s loky cpu_count() call
        probabilities = np.asarray(self.model.predict_proba(X, num_threads=1))[:, 1]

        # Apply threshold
        predictions = (probabilities >= threshold).astype(int)

        return predictions, probabilities, df["group_id"].tolist()

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """
        Train LightGBM classifier with scale_pos_weight to handle class imbalance.

        X_train / X_val must be the output of extract_group_features() with
        group_id dropped. y is a binary array (1 = group contains a real gene).
        Saves feature names from X_train columns.
        """
        import lightgbm as lgb

        n_pos = int(y_train.sum())
        n_neg = int(len(y_train) - n_pos)
        if n_pos == 0:
            raise ValueError("Training set has no positive examples.")

        spw = n_neg / n_pos
        logger.info(
            "Training LightGBM: n_train=%d, pos=%d, neg=%d, scale_pos_weight=%.1f",
            len(y_train),
            n_pos,
            n_neg,
            spw,
        )

        callbacks = []
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val.values, y_val)]
            callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=False))
            callbacks.append(lgb.log_evaluation(period=-1))

        self.model = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        self.model.fit(
            X_train.values,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks if callbacks else None,
        )
        self.feature_names = list(X_train.columns)

    def calibrate_threshold(self, X_val: pd.DataFrame, y_val: np.ndarray) -> float:
        """
        Sweep decision thresholds and return the one maximising F1 on the
        validation set. Updates nothing — caller decides whether to adopt.
        """
        from sklearn.metrics import precision_recall_curve

        probs = np.asarray(self.model.predict_proba(X_val.values, num_threads=1))[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_val, probs)
        f1_scores = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )
        best_idx = int(np.argmax(f1_scores[:-1]))
        best_t = float(thresholds[best_idx])
        best_f1 = float(f1_scores[best_idx])
        logger.info("Best threshold: %.3f  (val F1=%.4f)", best_t, best_f1)
        return best_t

    def save(self, model_path: str) -> None:
        """Save model and model-specific feature names (avoids overwriting shared file)."""
        p = Path(model_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, str(p))
        feature_path = p.parent / f"{p.stem}_feature_names.pkl"
        joblib.dump(self.feature_names, str(feature_path))
        logger.info("Saved model -> %s", p)
        logger.info("Saved features -> %s", feature_path)

    def filter_groups(
        self,
        groups: Dict[str, List[Dict]],
        genome_id: str = "unknown",
        weights: Dict = None,
        threshold: float = 0.07,
    ) -> Dict[str, List[Dict]]:
        """
        Filter groups, keeping only those predicted to contain real genes.
        Returns:
            Filtered dictionary with only groups predicted as real genes
        """
        # Get predictions
        predictions, probabilities, group_ids = self.predict_groups(
            groups, genome_id, weights, threshold
        )

        # Create set of kept group IDs (those above threshold)
        kept_group_ids = set(
            gid for gid, prob in zip(group_ids, probabilities) if prob >= threshold
        )

        # Filter groups
        filtered_groups = {gid: orfs for gid, orfs in groups.items() if gid in kept_group_ids}

        return filtered_groups


# ----------------
# --- Hybrid model
# ----------------
class CNNBranch(nn.Module):
    def __init__(self, output_dim=128, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 64, 7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # expects x shape: (batch, seq_len, 4)
        x = x.permute(0, 2, 1)  # -> (batch, 4, seq_len)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        x = F.relu(self.fc(x))
        return x


class DenseBranch(nn.Module):
    def __init__(self, input_dim=25, output_dim=128, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        return x


class HybridGenePredictor(nn.Module):
    def __init__(self, num_traditional_features=25, dropout=0.3):
        super().__init__()
        self.cnn_branch = CNNBranch(128, dropout)
        self.dense_branch = DenseBranch(num_traditional_features, 128, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),  # raw logits
        )

    def forward(self, sequences, features):
        # sequences: (batch, seq_len, 4)
        # features: (batch, num_traditional_features)
        cnn_out = self.cnn_branch(sequences)
        dense_out = self.dense_branch(features)
        combined = torch.cat([cnn_out, dense_out], dim=1)
        logits = self.fusion(combined)
        return logits.squeeze(-1)


class HybridGeneFilter:
    def __init__(self, device: str = None):
        self.model = None
        self.threshold = 0.25
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_names = [
            "codon_score_norm",
            "imm_score_norm",
            "rbs_score_norm",
            "length_score_norm",
            "start_score_norm",
            "combined_score",
            "length_bp",
            "length_codons",
            "length_log",
            "start_codon_type",
            "stop_codon_type",
            "has_kozak_like",
            "gc_content",
            "gc_skew",
            "at_skew",
            "purine_content",
            "effective_num_codons",
            "codon_bias_index",
            "has_hairpin_near_stop",
            "hydrophobicity_mean",
            "hydrophobicity_std",
            "charge_mean",
            "aromatic_fraction",
            "small_fraction",
            "polar_fraction",
            # New features (#124, #126)
            "gc3_content",
            "gc_deviation",
        ]

    def load(self, model_path):
        import pickle

        with open(model_path, "rb") as f:
            data = pickle.load(f)

        model = HybridGenePredictor(num_traditional_features=data["num_traditional_features"])
        model.load_state_dict(data["model_state_dict"])
        model.eval()

        self.model = model
        self.threshold = data["threshold"]

        logger.info("Loaded hybrid model from %s", {model_path})

    @staticmethod
    def _calculate_enc(sequence: str) -> float:
        codons = [sequence[i : i + 3] for i in range(0, len(sequence) - 2, 3)]
        valid_codons = [c for c in codons if len(c) == 3 and "N" not in c]
        if len(valid_codons) == 0:
            return 0.0
        codon_counts = Counter(valid_codons)
        num_unique = len(codon_counts)
        enc_normalized = num_unique / max(len(valid_codons), 1)
        return float(enc_normalized)

    @staticmethod
    def _calculate_cbi(sequence: str) -> float:
        codons = [sequence[i : i + 3] for i in range(0, len(sequence) - 2, 3)]
        valid_codons = [c for c in codons if len(c) == 3 and "N" not in c]
        if len(valid_codons) == 0:
            return 0.0
        codon_counts = Counter(valid_codons)
        frequencies = np.array(list(codon_counts.values())) / len(valid_codons)
        entropy = -np.sum(frequencies * np.log2(frequencies + 1e-10))
        max_entropy = math.log2(61)
        cbi = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
        return float(cbi)

    @staticmethod
    def _calculate_gc3(sequence: str) -> float:
        """GC content at third codon positions — strong coding signal in high-GC organisms."""
        codons = [sequence[i : i + 3] for i in range(0, len(sequence) - 2, 3)]
        valid = [c for c in codons if len(c) == 3 and "N" not in c]
        if not valid:
            return 0.0
        return float(sum(1 for c in valid if c[2] in "GC") / len(valid))

    @staticmethod
    def _detect_hairpin_near_stop(sequence: str, window: int = 30) -> float:
        if len(sequence) < window:
            return 0.0
        tail = sequence[-window:]
        try:
            seq_obj = Seq(tail)
            reverse_comp = str(seq_obj.reverse_complement())
            matches = sum(1 for a, b in zip(tail, reverse_comp) if a == b)
            hairpin_score = matches / len(tail)
            return 1.0 if hairpin_score > 0.6 else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _calculate_amino_acid_properties(sequence: str) -> Dict[str, float]:
        try:
            seq_obj = Seq(sequence)
            protein = str(seq_obj.translate(table=11, to_stop=True))
            if len(protein) == 0:
                return dict(
                    hydro_mean=0.0,
                    hydro_std=0.0,
                    charge_mean=0.0,
                    aromatic_frac=0.0,
                    small_frac=0.0,
                    polar_frac=0.0,
                )
            analysis = ProteinAnalysis(protein)
            aa_percent = analysis.amino_acids_percent
            aromatic = set("FYW")
            small = set("AGSTCV")
            polar = set("STNQY")
            charged = set("DEKR")
            aromatic_frac = sum(aa_percent.get(aa, 0) for aa in aromatic)
            small_frac = sum(aa_percent.get(aa, 0) for aa in small)
            polar_frac = sum(aa_percent.get(aa, 0) for aa in polar)
            charge_mean = sum(aa_percent.get(aa, 0) for aa in charged)
            kd_scale = {
                "A": 1.8,
                "R": -4.5,
                "N": -3.5,
                "D": -3.5,
                "C": 2.5,
                "Q": -3.5,
                "E": -3.5,
                "G": -0.4,
                "H": -3.2,
                "I": 4.5,
                "L": 3.8,
                "K": -3.9,
                "M": 1.9,
                "F": 2.8,
                "P": -1.6,
                "S": -0.8,
                "T": -0.7,
                "W": -0.9,
                "Y": -1.3,
                "V": 4.2,
            }
            hydro_values = [kd_scale.get(aa, 0.0) for aa in protein]
            return {
                "hydro_mean": float(np.mean(hydro_values)) if hydro_values else 0.0,
                "hydro_std": (float(np.std(hydro_values)) if len(hydro_values) > 1 else 0.0),
                "charge_mean": float(charge_mean),
                "aromatic_frac": float(aromatic_frac),
                "small_frac": float(small_frac),
                "polar_frac": float(polar_frac),
            }
        except Exception:
            return dict(
                hydro_mean=0.0,
                hydro_std=0.0,
                charge_mean=0.0,
                aromatic_frac=0.0,
                small_frac=0.0,
                polar_frac=0.0,
            )

    def extract_features(self, candidates: List[Dict], genome_id: str = "unknown") -> pd.DataFrame:
        rows = []
        for candidate in candidates:
            sequence = candidate.get("sequence", "").upper()
            feature_dict = {
                "codon_score_norm": float(candidate.get("codon_score_norm", 0.0)),
                "imm_score_norm": float(candidate.get("imm_score_norm", 0.0)),
                "rbs_score_norm": float(candidate.get("rbs_score_norm", 0.0)),
                "length_score_norm": float(candidate.get("length_score_norm", 0.0)),
                "start_score_norm": float(candidate.get("start_score_norm", 0.0)),
                "combined_score": float(candidate.get("combined_score", 0.0)),
            }
            length = int(candidate.get("length", len(sequence)))
            feature_dict["length_bp"] = float(length)
            feature_dict["length_codons"] = float(length / 3.0)
            feature_dict["length_log"] = math.log(max(length, 1))
            start_codon = candidate.get(
                "start_codon", sequence[:3] if len(sequence) >= 3 else "ATG"
            )
            start_map = {"ATG": 0.0, "GTG": 1.0, "TTG": 2.0}
            feature_dict["start_codon_type"] = float(start_map.get(start_codon, 0.0))
            stop_codon = sequence[-3:] if len(sequence) >= 3 else "TAA"
            stop_map = {"TAA": 0.0, "TAG": 1.0, "TGA": 2.0}
            feature_dict["stop_codon_type"] = float(stop_map.get(stop_codon, 0.0))
            feature_dict["has_kozak_like"] = float(candidate.get("rbs_score", 0) > 3.0)
            counts = Counter(sequence)
            seq_len = len(sequence)
            g = counts.get("G", 0)
            c = counts.get("C", 0)
            a = counts.get("A", 0)
            t = counts.get("T", 0)
            feature_dict["gc_content"] = (g + c) / seq_len if seq_len > 0 else 0.0
            feature_dict["gc_skew"] = (g - c) / (g + c) if (g + c) > 0 else 0.0
            feature_dict["at_skew"] = (a - t) / (a + t) if (a + t) > 0 else 0.0
            feature_dict["purine_content"] = (a + g) / seq_len if seq_len > 0 else 0.0
            feature_dict["effective_num_codons"] = self._calculate_enc(sequence)
            feature_dict["codon_bias_index"] = self._calculate_cbi(sequence)
            feature_dict["has_hairpin_near_stop"] = self._detect_hairpin_near_stop(sequence)
            aa_props = self._calculate_amino_acid_properties(sequence)
            feature_dict.update(
                {
                    "hydrophobicity_mean": aa_props["hydro_mean"],
                    "hydrophobicity_std": aa_props["hydro_std"],
                    "charge_mean": aa_props["charge_mean"],
                    "aromatic_fraction": aa_props["aromatic_frac"],
                    "small_fraction": aa_props["small_frac"],
                    "polar_fraction": aa_props["polar_frac"],
                }
            )
            # GC3: GC content at third codon positions (#126)
            gc3 = self._calculate_gc3(sequence)
            feature_dict["gc3_content"] = gc3
            # GC deviation from genome mean (#124) — uses genome_gc_mean if provided
            genome_gc = candidate.get("genome_gc_mean", feature_dict["gc_content"])
            feature_dict["gc_deviation"] = feature_dict["gc_content"] - genome_gc
            rows.append(feature_dict)
        return pd.DataFrame(rows).fillna(0.0)

    @staticmethod
    def _one_hot_encode_dna(candidates: List[Dict], max_len: int = None):
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
        if max_len is None:
            max_len = max((len(c.get("sequence", "")) for c in candidates), default=0)
        num_candidates = len(candidates)
        one_hot = np.zeros((num_candidates, max_len, 4), dtype=np.float32)
        for idx, candidate in enumerate(candidates):
            seq = candidate.get("sequence", "").upper()
            for i, nt in enumerate(seq):
                if i >= max_len:
                    break
                if nt in mapping:
                    one_hot[idx, i, mapping[nt]] = 1.0
        return torch.from_numpy(one_hot)

    def predict(
        self,
        candidates: List[Dict],
        genome_id: str = "unknown",
        threshold: float = None,
        batch_size: int = 64,  # NEW PARAMETER
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Predict which candidates are real genes using BATCHED processing.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if not candidates:
            return np.array([]), np.array([]), []

        if threshold is None:
            threshold = float(self.threshold)

        # Extract features once (doesn't use much memory)
        df = self.extract_features(candidates, genome_id)
        missing = [f for f in self.feature_names if f not in df.columns]
        if missing:
            raise ValueError(
                f"Feature mismatch: {len(missing)} feature(s) expected by the "
                f"HybridGeneFilter model are absent from the extracted feature "
                f"DataFrame. Missing: {sorted(missing)}"
            )

        X_features = torch.tensor(df[self.feature_names].values, dtype=torch.float32)

        # Determine max sequence length
        max_seq_len = max((len(c.get("sequence", "")) for c in candidates), default=1000)

        # Process in batches to avoid OOM
        all_probs = []
        num_batches = (len(candidates) + batch_size - 1) // batch_size

        self.model.to(self.device)
        self.model.eval()

        logger.debug(
            "Processing %d candidates in %d batches of %d...",
            len(candidates),
            num_batches,
            batch_size,
        )

        with torch.no_grad():
            for i in range(0, len(candidates), batch_size):
                batch_end = min(i + batch_size, len(candidates))
                batch_candidates = candidates[i:batch_end]

                # One-hot encode batch
                X_sequences_batch = self._one_hot_encode_dna(batch_candidates, max_len=max_seq_len)
                X_features_batch = X_features[i:batch_end]

                # Move to device
                X_sequences_batch = X_sequences_batch.to(self.device)
                X_features_batch = X_features_batch.to(self.device)

                # Predict
                outputs = self.model(X_sequences_batch, X_features_batch)
                probs = torch.sigmoid(outputs).cpu().numpy()

                all_probs.append(probs)

                # Clear cache
                del X_sequences_batch, X_features_batch, outputs
                if self.device == "cuda":
                    torch.cuda.empty_cache()

        # Concatenate all batch results
        probs = np.concatenate(all_probs)

        if probs.ndim == 0:
            probs = np.array([float(probs)])

        preds = (probs >= threshold).astype(int)
        gene_ids = [c.get("gene_id", f"gene_{i}") for i, c in enumerate(candidates)]

        return preds, probs, gene_ids

    def filter_candidates(
        self,
        candidates: List[Dict],
        genome_id: str = "unknown",
        threshold: float = None,
        batch_size: int = 64,  # NEW PARAMETER
    ) -> List[Dict]:
        """
        Filter candidates, keeping only those predicted to be real genes.
        Uses batched processing to avoid memory issues.

        Args:
            candidates: List of candidate ORFs
            genome_id: Genome identifier
            threshold: Probability threshold
            batch_size: Batch size for processing (default: 64)

        Returns:
            Filtered list of candidates
        """
        # Accept DataFrame or List[Dict]
        if hasattr(candidates, "to_dict"):
            candidates = candidates.to_dict("records")

        preds, probs, gene_ids = self.predict(candidates, genome_id, threshold, batch_size)

        kept = []
        for cand, p in zip(candidates, probs):
            if p >= (threshold if threshold is not None else self.threshold):
                new = dict(cand)
                new["hybrid_prob"] = float(p)
                kept.append(new)

        return kept


if __name__ == "__main__":
    print("Testing HybridGeneFilter...\n")

    try:
        classifier = HybridGeneFilter()
        model_path = Path(__file__).parent.parent / "models" / "hybrid_best_model.pkl"
        if model_path.exists():
            classifier.load(str(model_path))
        else:
            print("[!] Model not found, skipping ML...")
    except Exception as e:
        print(f"[!] ML error: {e}, skipping...")

    print("Testing OrfGroupClassifier...\n")

    try:
        classifier = OrfGroupClassifier()
        model_path = Path(__file__).parent.parent / "models" / "orf_classifier_lgb.pkl"
        if model_path.exists():
            classifier.load(str(model_path))
        else:
            print("[!] Model not found, skipping ML...")
    except Exception as e:
        print(f"[!] ML error: {e}, skipping...")
