"""
Machine learning classifier for ORF groups.

Loads a trained LightGBM model to filter groups of nested ORFs.
Groups share the same stop codon but have different start codons.
Model predicts: Does this GROUP contain a real gene? (yes/no)
"""

import joblib
import numpy as np
import pandas as pd
import torch
import math
from pathlib import Path
from typing import List, Dict
from collections import Counter
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis

class OrfGroupClassifier:
    """
    Binary classifier for ORF groups using LightGBM.
    Predicts whether a group of nested ORFs contains a real gene.
    """
    
    def __init__(self):
        """
        Initialize empty classifier.
        Call load() to load a trained model before using.
        """
        self.model = None
        self.feature_names = None
    
    def load(self, model_path: str = '../models/orf_classifier_lgb.pkl'):
        """
        Load trained model and feature names from disk.
        Model and feature_names.pkl must be in same directory.
        """
        model_path = Path(model_path)
        
        # Load the model
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = joblib.load(str(model_path))
        print(f"✓ Loaded model: {model_path}")
        
        # Load feature names (should be in same directory)
        feature_path = model_path.parent / 'feature_names.pkl'
        if feature_path.exists():
            self.feature_names = joblib.load(str(feature_path))
            print(f"✓ Loaded {len(self.feature_names)} features")
        else:
            print(f"⚠ Warning: feature_names.pkl not found in {model_path.parent}")
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
        self, 
        groups: Dict[str, List[Dict]], 
        genome_id: str,
        weights: Dict = None
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
        for group_id, orf_list in tqdm(groups.items(), total=len(groups), desc=f"Groups {genome_id}"):
            if len(orf_list) == 0:
                continue

            combined_scores = []
            rbs_scores = []
            codon_scores = []
            start_scores = []
            imm_scores = []
            start_select_scores = []
            strands = []

            for orf in orf_list:
                strands.append(orf.get("strand", "+"))
                combined_scores.append(orf.get("combined_score", 0.0))
                rbs_scores.append(orf.get("rbs_score", 0.0))
                codon_scores.append(orf.get("codon_score", 0.0))
                start_scores.append(orf.get("start_score", 0.0))
                imm_scores.append(orf.get("imm_score", 0.0))

                if weights is not None:
                    orf_score = (
                        orf.get("codon_score_norm", 0.0) * weights.get("codon", 0.0) +
                        orf.get("imm_score_norm", 0.0) * weights.get("imm", 0.0) +
                        orf.get("rbs_score_norm", 0.0) * weights.get("rbs", 0.0) +
                        orf.get("length_score_norm", 0.0) * weights.get("length", 0.0) +
                        orf.get("start_score_norm", 0.0) * weights.get("start", 0.0)
                    )
                else:
                    orf_score = 0.0
                start_select_scores.append(orf_score)

            # Aggregate features
            group_features = {
                "group_id": group_id,
                "num_orfs": len(orf_list),
                "combined_max": max(combined_scores),
                "combined_mean": np.mean(combined_scores),
                "combined_std": np.std(combined_scores),
                "combined_entropy": self._entropy_from_probs(np.maximum(combined_scores, 0)),
                "combined_margin_top2": np.sort(combined_scores)[-1] - np.sort(combined_scores)[-2]
                if len(combined_scores) > 1 else combined_scores[0],
                "frac_top_orfs": np.sum(np.array(combined_scores) >= 0.8 * max(combined_scores)) / len(combined_scores),
                "rbs_max": max(rbs_scores),
                "rbs_mean": np.mean(rbs_scores),
                "codon_max": max(codon_scores),
                "codon_mean": np.mean(codon_scores),
                "start_max": max(start_scores),
                "start_mean": np.mean(start_scores),
                "imm_max": max(imm_scores),
                "imm_mean": np.mean(imm_scores),
                "start_select_max": max(start_select_scores),
                "start_select_mean": np.mean(start_select_scores),
                "strand_plus_frac": strands.count("+") / len(strands),
                "strand_minus_frac": strands.count("-") / len(strands),
            }

            # Relative features
            max_combined = max(combined_scores)
            max_rbs = max(rbs_scores)
            max_codon = max(codon_scores)
            max_start = max(start_scores)
            max_start_select = max(start_select_scores)

            rel_combined = [c / max_combined if max_combined > 0 else 0 for c in combined_scores]
            rel_rbs = [c / max_rbs if max_rbs > 0 else 0 for c in rbs_scores]
            rel_codon = [c / max_codon if max_codon > 0 else 0 for c in codon_scores]
            rel_start = [c / max_start if max_start > 0 else 0 for c in start_scores]
            rel_start_select = [c / max_start_select if max_start_select > 0 else 0 for c in start_select_scores]

            group_features.update({
                "rel_combined_mean": np.mean(rel_combined),
                "rel_combined_max": np.max(rel_combined),
                "rel_rbs_mean": np.mean(rel_rbs),
                "rel_rbs_max": np.max(rel_rbs),
                "rel_codon_mean": np.mean(rel_codon),
                "rel_codon_max": np.max(rel_codon),
                "rel_start_mean": np.mean(rel_start),
                "rel_start_max": np.max(rel_start),
                "rel_start_select_mean": np.mean(rel_start_select),
                "rel_start_select_max": np.max(rel_start_select),
                "frac_top_combined": np.sum(np.array(combined_scores) >= 0.95 * max_combined) / len(combined_scores),
                "frac_top_start_select": np.sum(np.array(start_select_scores) >= 0.95 * max_start_select) / len(start_select_scores),
            })

            rows.append(group_features)

        return pd.DataFrame(rows).fillna(0.0)
    

    def predict_groups(
        self, 
        groups: Dict[str, List[Dict]], 
        genome_id: str = "unknown",
        weights: Dict = None,
        threshold: float = 0.1
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
        
        # Get feature names from model
        model_features = self.model.feature_name_
        
        # Use only features that exist in both model and data
        available_features = [f for f in model_features if f in df.columns]
        
        if len(available_features) < len(model_features):
            missing = set(model_features) - set(available_features)
            print(f"⚠ Warning: {len(missing)} features missing from data: {missing}")
        
        # Extract feature matrix
        X = df[available_features].values
        
        # Get probabilities (probability of class 1 = real gene)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        # Apply threshold
        predictions = (probabilities >= threshold).astype(int)
        
        return predictions, probabilities, df['group_id'].tolist()
    
    def filter_groups(
        self,
        groups: Dict[str, List[Dict]],
        genome_id: str = "unknown", 
        weights: Dict = None,
        threshold: float = 0.1
    ) -> Dict[str, List[Dict]]:
        """
        Filter groups, keeping only those predicted to contain real genes.
        
        Args:
            groups: Dictionary of {group_id: [list of ORFs]}
            genome_id: Genome identifier
            weights: Optional weights for start selection
            threshold: Probability threshold (default 0.1)
        
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
        filtered_groups = {
            gid: orfs for gid, orfs in groups.items() if gid in kept_group_ids
        }
        
        return filtered_groups

# filters.py
import math
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.metrics import confusion_matrix  # kept for potential evaluation prints


# ----------------------------
# --- Hybrid model arch (same as your notebook)
# ----------------------------
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
            nn.Linear(64, 1)  # raw logits
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
        self.threshold = 0.12
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_names = [
            'codon_score_norm', 'imm_score_norm', 'rbs_score_norm',
            'length_score_norm', 'start_score_norm', 'combined_score',
            'length_bp', 'length_codons', 'length_log',
            'start_codon_type', 'stop_codon_type', 'has_kozak_like',
            'gc_content', 'gc_skew', 'at_skew', 'purine_content',
            'effective_num_codons', 'codon_bias_index',
            'has_hairpin_near_stop', 'hydrophobicity_mean', 'hydrophobicity_std',
            'charge_mean', 'aromatic_fraction', 'small_fraction', 'polar_fraction'
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

        print(f"Loaded hybrid model from {model_path}")


    @staticmethod
    def _calculate_enc(sequence: str) -> float:
        codons = [sequence[i:i+3] for i in range(0, len(sequence)-2, 3)]
        valid_codons = [c for c in codons if len(c) == 3 and 'N' not in c]
        if len(valid_codons) == 0:
            return 0.0
        codon_counts = Counter(valid_codons)
        num_unique = len(codon_counts)
        enc_normalized = num_unique / max(len(valid_codons), 1)
        return float(enc_normalized)

    @staticmethod
    def _calculate_cbi(sequence: str) -> float:
        codons = [sequence[i:i+3] for i in range(0, len(sequence)-2, 3)]
        valid_codons = [c for c in codons if len(c) == 3 and 'N' not in c]
        if len(valid_codons) == 0:
            return 0.0
        codon_counts = Counter(valid_codons)
        frequencies = np.array(list(codon_counts.values())) / len(valid_codons)
        entropy = -np.sum(frequencies * np.log2(frequencies + 1e-10))
        max_entropy = math.log2(61)
        cbi = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
        return float(cbi)

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
                return dict(hydro_mean=0.0, hydro_std=0.0, charge_mean=0.0,
                            aromatic_frac=0.0, small_frac=0.0, polar_frac=0.0)
            analysis = ProteinAnalysis(protein)
            aa_percent = analysis.get_amino_acids_percent()
            aromatic = set('FYW')
            small = set('AGSTCV')
            polar = set('STNQY')
            charged = set('DEKR')
            aromatic_frac = sum(aa_percent.get(aa, 0) for aa in aromatic)
            small_frac = sum(aa_percent.get(aa, 0) for aa in small)
            polar_frac = sum(aa_percent.get(aa, 0) for aa in polar)
            charge_mean = sum(aa_percent.get(aa, 0) for aa in charged)
            kd_scale = {
                'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
                'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
                'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
                'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
            }
            hydro_values = [kd_scale.get(aa, 0.0) for aa in protein]
            return {
                'hydro_mean': float(np.mean(hydro_values)) if hydro_values else 0.0,
                'hydro_std': float(np.std(hydro_values)) if len(hydro_values) > 1 else 0.0,
                'charge_mean': float(charge_mean),
                'aromatic_frac': float(aromatic_frac),
                'small_frac': float(small_frac),
                'polar_frac': float(polar_frac),
            }
        except Exception:
            return dict(hydro_mean=0.0, hydro_std=0.0, charge_mean=0.0,
                        aromatic_frac=0.0, small_frac=0.0, polar_frac=0.0)

    def extract_features(self, candidates: List[Dict], genome_id: str = "unknown") -> pd.DataFrame:
        rows = []
        for candidate in candidates:
            sequence = candidate.get('sequence', '').upper()
            feature_dict = {
                'codon_score_norm': float(candidate.get('codon_score_norm', 0.0)),
                'imm_score_norm': float(candidate.get('imm_score_norm', 0.0)),
                'rbs_score_norm': float(candidate.get('rbs_score_norm', 0.0)),
                'length_score_norm': float(candidate.get('length_score_norm', 0.0)),
                'start_score_norm': float(candidate.get('start_score_norm', 0.0)),
                'combined_score': float(candidate.get('combined_score', 0.0)),
            }
            length = int(candidate.get('length', len(sequence)))
            feature_dict['length_bp'] = float(length)
            feature_dict['length_codons'] = float(length / 3.0)
            feature_dict['length_log'] = math.log(max(length, 1))
            start_codon = candidate.get('start_codon', sequence[:3] if len(sequence) >= 3 else 'ATG')
            start_map = {'ATG': 0.0, 'GTG': 1.0, 'TTG': 2.0}
            feature_dict['start_codon_type'] = float(start_map.get(start_codon, 0.0))
            stop_codon = sequence[-3:] if len(sequence) >= 3 else 'TAA'
            stop_map = {'TAA': 0.0, 'TAG': 1.0, 'TGA': 2.0}
            feature_dict['stop_codon_type'] = float(stop_map.get(stop_codon, 0.0))
            feature_dict['has_kozak_like'] = float(candidate.get('rbs_score', 0) > 3.0)
            counts = Counter(sequence)
            seq_len = len(sequence)
            g = counts.get('G', 0)
            c = counts.get('C', 0)
            a = counts.get('A', 0)
            t = counts.get('T', 0)
            feature_dict['gc_content'] = (g + c) / seq_len if seq_len > 0 else 0.0
            feature_dict['gc_skew'] = (g - c) / (g + c) if (g + c) > 0 else 0.0
            feature_dict['at_skew'] = (a - t) / (a + t) if (a + t) > 0 else 0.0
            feature_dict['purine_content'] = (a + g) / seq_len if seq_len > 0 else 0.0
            feature_dict['effective_num_codons'] = self._calculate_enc(sequence)
            feature_dict['codon_bias_index'] = self._calculate_cbi(sequence)
            feature_dict['has_hairpin_near_stop'] = self._detect_hairpin_near_stop(sequence)
            aa_props = self._calculate_amino_acid_properties(sequence)
            feature_dict.update({
                'hydrophobicity_mean': aa_props['hydro_mean'],
                'hydrophobicity_std': aa_props['hydro_std'],
                'charge_mean': aa_props['charge_mean'],
                'aromatic_fraction': aa_props['aromatic_frac'],
                'small_fraction': aa_props['small_frac'],
                'polar_fraction': aa_props['polar_frac'],
            })
            rows.append(feature_dict)
        return pd.DataFrame(rows).fillna(0.0)

    @staticmethod
    def _one_hot_encode_dna(candidates: List[Dict], max_len: int = None):
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        if max_len is None:
            max_len = max((len(c.get('sequence', '')) for c in candidates), default=0)
        num_candidates = len(candidates)
        one_hot = np.zeros((num_candidates, max_len, 4), dtype=np.float32)
        for idx, candidate in enumerate(candidates):
            seq = candidate.get('sequence', '').upper()
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
        batch_size: int = 64  # NEW PARAMETER
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
            print(f"⚠ Warning: missing numeric features: {missing}")

        X_features = torch.tensor(
            df[[f for f in self.feature_names if f in df.columns]].values, 
            dtype=torch.float32
        )
        
        # Determine max sequence length
        max_seq_len = max((len(c.get('sequence', '')) for c in candidates), default=1000)
        
        # Process in batches to avoid OOM
        all_probs = []
        num_batches = (len(candidates) + batch_size - 1) // batch_size
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Processing {len(candidates)} candidates in {num_batches} batches of {batch_size}...")
        
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
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
        
        # Concatenate all batch results
        probs = np.concatenate(all_probs)
        
        if probs.ndim == 0:
            probs = np.array([float(probs)])

        preds = (probs >= threshold).astype(int)
        gene_ids = [c.get('gene_id', f'gene_{i}') for i, c in enumerate(candidates)]
        
        return preds, probs, gene_ids

    def filter_candidates(
        self, 
        candidates: List[Dict], 
        genome_id: str = "unknown", 
        threshold: float = None,
        batch_size: int = 64  # NEW PARAMETER
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
        preds, probs, gene_ids = self.predict(
            candidates, genome_id, threshold, batch_size
        )
        
        kept = []
        for cand, p in zip(candidates, probs):
            if p >= (threshold if threshold is not None else self.threshold):
                new = dict(cand)
                new['hybrid_prob'] = float(p)
                kept.append(new)
        
        return kept

if __name__ == '__main__':
    print("Testing HybridGeneFilter...\n")
    
    try:
        classifier = HybridGeneFilter()
        model_path = Path(__file__).parent.parent/ 'models' / 'hybrid_best_model.pkl'
        if model_path.exists():
            classifier.load(str(model_path))
        else:
            print(f"[!] Model not found, skipping ML...")
    except Exception as e:
         print(f"[!] ML error: {e}, skipping...")

    print("Testing OrfGroupClassifier...\n")
    
    try:
        classifier = OrfGroupClassifier()
        model_path = Path(__file__).parent.parent/ 'models' / 'orf_classifier_lgb.pkl'
        if model_path.exists():
            classifier.load(str(model_path))
        else:
            print(f"[!] Model not found, skipping ML...")
    except Exception as e:
         print(f"[!] ML error: {e}, skipping...")