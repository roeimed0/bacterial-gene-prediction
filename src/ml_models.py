"""
Machine learning classifier for ORF groups.

Loads a trained LightGBM model to filter groups of nested ORFs.
Groups share the same stop codon but have different start codons.
Model predicts: Does this GROUP contain a real gene? (yes/no)
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict


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


# Test the loading
if __name__ == '__main__':
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