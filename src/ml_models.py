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



"""
Hybrid gene filter classifier.

This class wraps the hybrid_final_filter.pkl model for final gene filtering.
Uses both numeric features and CNN sequence encoding.
"""
class HybridGeneFilter:
    """
    Final hybrid classifier for gene prediction.
    Uses trained PyTorch model with both features and sequence CNN.
    """
    
    def __init__(self):
        """
        Initialize empty classifier.
        Call load() to load a trained model before using.
        """
        self.model = None
        self.threshold = 0.12
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
    
    def load(self, model_path: str = '../models/hybrid_final_filter.pkl'):
        """
        Load trained PyTorch model from disk.
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load the checkpoint (contains full model + threshold)
        checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=False)
        
        # Extract model and threshold
        self.model = checkpoint['model']
        self.model.eval()
        self.threshold = checkpoint.get('threshold', 0.12)
        
        print(f"✓ Loaded PyTorch model: {model_path}")
        print(f"✓ Expecting {len(self.feature_names)} features + DNA sequence")
        print(f"✓ Best threshold: {self.threshold:.4f}")
    
    def _calculate_enc(self, sequence):
        """Effective Number of Codons (de novo - uses only this sequence)."""
        codons = [sequence[i:i+3] for i in range(0, len(sequence)-2, 3)]
        valid_codons = [c for c in codons if len(c) == 3 and 'N' not in c]
        
        if len(valid_codons) == 0:
            return 0.0
        
        codon_counts = Counter(valid_codons)
        num_unique = len(codon_counts)
        enc_normalized = num_unique / max(len(valid_codons), 1)
        
        return float(enc_normalized)
    
    def _calculate_cbi(self, sequence):
        """Codon Bias Index - entropy-based measure."""
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
    
    def _detect_hairpin_near_stop(self, sequence, window=30):
        """Detect potential hairpin structure near stop codon."""
        if len(sequence) < window:
            return 0.0
        
        tail = sequence[-window:]
        
        try:
            seq_obj = Seq(tail)
            reverse_comp = str(seq_obj.reverse_complement())
            matches = sum(1 for a, b in zip(tail, reverse_comp) if a == b)
            hairpin_score = matches / len(tail)
            return 1.0 if hairpin_score > 0.6 else 0.0
        except:
            return 0.0
    
    def _calculate_amino_acid_properties(self, sequence):
        """Calculate amino acid properties from DNA sequence."""
        try:
            seq_obj = Seq(sequence)
            protein = str(seq_obj.translate(table=11, to_stop=True))
            
            if len(protein) == 0:
                return {
                    'hydro_mean': 0.0, 'hydro_std': 0.0, 'charge_mean': 0.0,
                    'aromatic_frac': 0.0, 'small_frac': 0.0, 'polar_frac': 0.0
                }
            
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
            return {
                'hydro_mean': 0.0, 'hydro_std': 0.0, 'charge_mean': 0.0,
                'aromatic_frac': 0.0, 'small_frac': 0.0, 'polar_frac': 0.0
            }
    
    def extract_features(
        self, 
        candidates: List[Dict], 
        genome_id: str = "unknown"
    ) -> pd.DataFrame:
        """
        Extract all 25 numeric features from candidate genes.
        
        Args:
            candidates: List of candidate gene dictionaries (must have 'sequence' key)
            genome_id: Genome identifier (for progress bar)
        
        Returns:
            DataFrame with one row per candidate, columns are features
        """
        from tqdm import tqdm
        
        rows = []
        for candidate in tqdm(candidates, desc=f"Extracting features {genome_id}"):
            sequence = candidate.get('sequence', '')
            seq_upper = sequence.upper()
            
            # Start with existing scores
            feature_dict = {
                'codon_score_norm': candidate.get('codon_score_norm', 0.0),
                'imm_score_norm': candidate.get('imm_score_norm', 0.0),
                'rbs_score_norm': candidate.get('rbs_score_norm', 0.0),
                'length_score_norm': candidate.get('length_score_norm', 0.0),
                'start_score_norm': candidate.get('start_score_norm', 0.0),
                'combined_score': candidate.get('combined_score', 0.0),
            }
            
            # Length features (3)
            length = candidate.get('length', len(sequence))
            feature_dict['length_bp'] = float(length)
            feature_dict['length_codons'] = float(length / 3)
            feature_dict['length_log'] = math.log(max(length, 1))
            
            # Start/stop codon types (2)
            start_codon = candidate.get('start_codon', seq_upper[:3] if len(seq_upper) >= 3 else 'ATG')
            start_map = {'ATG': 0, 'GTG': 1, 'TTG': 2}
            feature_dict['start_codon_type'] = float(start_map.get(start_codon, 0))
            
            stop_codon = seq_upper[-3:] if len(seq_upper) >= 3 else 'TAA'
            stop_map = {'TAA': 0, 'TAG': 1, 'TGA': 2}
            feature_dict['stop_codon_type'] = float(stop_map.get(stop_codon, 0))
            
            # Kozak-like (1)
            feature_dict['has_kozak_like'] = float(candidate.get('rbs_score', 0) > 3.0)
            
            # Sequence composition (3)
            counts = Counter(seq_upper)
            seq_len = len(seq_upper)
            
            g = counts.get('G', 0)
            c = counts.get('C', 0)
            a = counts.get('A', 0)
            t = counts.get('T', 0)
            
            feature_dict['gc_content'] = (g + c) / seq_len if seq_len > 0 else 0.0
            feature_dict['gc_skew'] = (g - c) / (g + c) if (g + c) > 0 else 0.0
            feature_dict['at_skew'] = (a - t) / (a + t) if (a + t) > 0 else 0.0
            
            # Complex calculations (10 features)
            feature_dict['purine_content'] = (a + g) / seq_len if seq_len > 0 else 0.0
            feature_dict['effective_num_codons'] = self._calculate_enc(seq_upper)
            feature_dict['codon_bias_index'] = self._calculate_cbi(seq_upper)
            feature_dict['has_hairpin_near_stop'] = self._detect_hairpin_near_stop(seq_upper)
            
            # Amino acid properties (6)
            aa_props = self._calculate_amino_acid_properties(seq_upper)
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
    
    def _one_hot_encode_dna(self, candidates: List[Dict], max_len: int = None):
        """
        Convert DNA sequences to one-hot tensor (A,C,G,T).
        
        Returns:
            torch.Tensor of shape (num_candidates, max_len, 4)
        """
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        
        # Determine max length
        if max_len is None:
            max_len = max(len(c.get('sequence', '')) for c in candidates)
        
        # Pre-allocate numpy array instead of list
        num_candidates = len(candidates)
        one_hot = np.zeros((num_candidates, max_len, 4), dtype=np.float32)
        
        for idx, candidate in enumerate(candidates):
            seq = candidate.get('sequence', '').upper()
            
            for i, nt in enumerate(seq):
                if i >= max_len:
                    break
                if nt in mapping:
                    one_hot[idx, i, mapping[nt]] = 1.0
        
        # Convert to tensor once (much faster!)
        return torch.from_numpy(one_hot)
    
    def predict(
        self, 
        candidates: List[Dict], 
        genome_id: str = "unknown",
        threshold: float = None,
        max_seq_len: int = None
    ) -> tuple:
        """
        Predict which candidates are real genes using hybrid model.
        
        Args:
            candidates: List of candidate gene dictionaries (must have 'sequence' key)
            genome_id: Genome identifier
            threshold: Probability threshold for classification (uses saved threshold if None)
            max_seq_len: Maximum sequence length for padding (auto-detected if None)
        
        Returns:
            Tuple of (predictions, probabilities, gene_ids)
            - predictions: Binary array (1 = real gene, 0 = false positive)
            - probabilities: Probability of being a real gene
            - gene_ids: List of gene IDs in same order
        """
        if not candidates:
            return np.array([]), np.array([]), []
        
        # Use saved threshold if not provided
        if threshold is None:
            threshold = self.threshold
        
        # Extract numeric features
        df = self.extract_features(candidates, genome_id)
        X_features = torch.tensor(df[self.feature_names].values, dtype=torch.float32)
        
        # Encode DNA sequences
        X_sequences = self._one_hot_encode_dna(candidates, max_seq_len)
        
        # Run model (NOTE: sequences first, then features!)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_sequences, X_features)
            probabilities = torch.sigmoid(outputs).squeeze().numpy()
        
        # Handle single prediction case
        if probabilities.ndim == 0:
            probabilities = np.array([probabilities])
        
        # Apply threshold
        predictions = (probabilities >= threshold).astype(int)
        
        # Get gene IDs
        gene_ids = [c.get('gene_id', f'gene_{i}') for i, c in enumerate(candidates)]
        
        return predictions, probabilities, gene_ids
    
    def filter_candidates(self, candidates, genome_id="unknown", threshold=None, max_seq_len=None):
        if not candidates:
            return []

        threshold = self.threshold if threshold is None else threshold

        _, probabilities, _ = self.predict(candidates, genome_id, threshold, max_seq_len)

        return [
            {**cand, "hybrid_prob": float(prob)}
            for cand, prob in zip(candidates, probabilities)
            if prob >= threshold
        ]

if __name__ == '__main__':
    print("Testing HybridGeneFilter...\n")
    
    try:
        classifier = HybridGeneFilter()
        classifier.load('../models/hybrid_final_filter.pkl')
        
        print(f"\n✓ Model loaded successfully")
        print(f"  Type: {type(classifier.model).__name__}")
        print(f"\n✓ Expected {len(classifier.feature_names)} features:")
        for i, name in enumerate(classifier.feature_names, 1):
            print(f"  {i:2d}. {name}")
        
    except FileNotFoundError as e:
        print(f"✗ {e}")
        print("\nTo create the model:")
        print("  1. Train the hybrid_final_filter model")
        print("  2. Save with joblib.dump()")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

    print("Testing OrfGroupClassifier...\n")
    
    try:
        classifier = OrfGroupClassifier()
        classifier.load('../models/orf_classifier_lgb.pkl')
        
        print(f"\n✓ Model loaded successfully")
        print(f"  Type: {type(classifier.model).__name__}")
        
        if classifier.feature_names:
            print(f"\nExpected features:")
            for i, name in enumerate(classifier.feature_names, 1):
                print(f"  {i}. {name}")
        
    except FileNotFoundError as e:
        print(f"✗ {e}")
        print("\nTo create the model:")
        print("  1. Run notebook 03_ml_models.ipynb")
        print("  2. Train the model")
        print("  3. Save with joblib.dump()")
