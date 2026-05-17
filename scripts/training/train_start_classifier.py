"""
Pairwise start-selection classifier.

Uses the same seed-42 catalog split as the LGB model:
  train+val (~84 genomes) -> collect pairwise features -> train classifier
  test (~16 genomes) + holdout (20 genomes) -> evaluate

For each LGB-kept group where the real gene is the top-1 OR top-2 candidate:
  Features: pairwise differences (top1 - top2) + group-level context
  Label:    1 = top-1 is correct (keep), 0 = top-2 is correct (flip)

At inference time: only apply classifier when baseline gap < CONTEST_THRESHOLD.

Models tested: Logistic Regression, Random Forest, LightGBM
Validation:    leave-one-phylum-out on the ~84 train+val genomes
Final test:    catalog test set + 20 holdout genomes

Run from repo root:
    python scripts/train_start_classifier.py
"""

import argparse
import contextlib
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import lightgbm as lgb_sk
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
from scipy.special import logit as sp_logit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import GENOME_CATALOG, START_SELECTION_WEIGHTS, TEST_GENOMES
from src.data_management import get_data_dir, get_gff_path, load_genome_sequence
from src.ml_models import OrfGroupClassifier
from src.traditional_methods import (
    build_all_scoring_models,
    create_intergenic_set,
    create_training_set,
    filter_candidates,
    find_orfs_candidates,
    organize_nested_orfs,
    score_all_orfs,
    score_codon_bias_ratio,
    score_imm_ratio,
)

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
DATA_DIR = get_data_dir("full_dataset")
OUT_DIR = Path(__file__).parent.parent.parent / "lgb_attribution_results"
OUT_DIR.mkdir(exist_ok=True)

# ── CLI args ──────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="Train start-selection classifier.")
_parser.add_argument(
    "--features",
    default="",
    help="Comma-separated v3 feature groups to add to v2 baseline: nterm,gc3,groupnorm  (default: none = v2 baseline)",
)
_parser.add_argument(
    "--out-model",
    default=str(MODELS_DIR / "start_selector.pkl"),
    help="Output model path (default: models/start_selector.pkl)",
)
_parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for train/val/test split and model training (default: 42)",
)
_args = _parser.parse_args()

SEED = _args.seed
CONTEST_T = 1.0  # re-score groups with baseline gap < this
VAL_PER_GROUP = 4
TEST_PER_GROUP = 4
LGB_T = 0.05
EPS = 1e-9
BASES = "ACGT"
_RC = str.maketrans("ACGT", "TGCA")
SD_MOTIFS = ["AAGGAGG", "AAGGAG", "AGGAG", "GGAGG", "GAGG", "AAGG", "AGGA"]
STOPS = {"TAA", "TAG", "TGA"}
W = START_SELECTION_WEIGHTS

lgb_model = OrfGroupClassifier()
lgb_model.load(str(MODELS_DIR / "orf_classifier_lgb.pkl"))

_ENABLED_GROUPS = {f.strip() for f in _args.features.split(",") if f.strip()}
_V3_GROUPS = {
    "nterm": {
        "d_nterm_charged_pos",
        "d_nterm_charged_neg",
        "d_nterm_polar",
        "d_nterm_hydrophobic",
        "t1_nterm_charged_pos",
        "t1_nterm_charged_neg",
        "t1_nterm_polar",
        "t1_nterm_hydrophobic",
    },
    "gc3": {"d_gc3_period", "gc3_period_t1", "gc3_period_t2"},
    "groupnorm": {"d_length_norm", "d_codon_norm", "d_imm_norm"},
}
_V3_ALL = set().union(*_V3_GROUPS.values())
_V3_ENABLED = set().union(*(_V3_GROUPS[g] for g in _ENABLED_GROUPS if g in _V3_GROUPS))
_V3_DROPPED = _V3_ALL - _V3_ENABLED

if _ENABLED_GROUPS:
    print(f"\nAblation mode — enabled v3 groups: {sorted(_ENABLED_GROUPS)}")
    if _V3_DROPPED:
        print(f"  Dropping: {sorted(_V3_DROPPED)}")
else:
    print("\nBaseline mode — all v3 features dropped (v2 feature set only)")
print(f"Output model: {_args.out_model}")

# ── Catalog split (seed=42, mirrors train_lgb.py) ────────────────────────────


def build_splits(catalog, val_per_group, test_per_group, seed):
    by_group = {}
    for g in catalog:
        by_group.setdefault(g["group"], []).append(g["accession"])
    rng = np.random.default_rng(seed)
    train_accs, val_accs, test_accs = [], [], []
    for group, accs in sorted(by_group.items()):
        accs = list(accs)
        rng.shuffle(accs)
        n_held = min(val_per_group + test_per_group, len(accs) // 3)
        n_val = n_held // 2
        n_test = n_held - n_val
        val_accs.extend(accs[:n_val])
        test_accs.extend(accs[n_val : n_val + n_test])
        train_accs.extend(accs[n_val + n_test :])
    return train_accs, val_accs, test_accs


catalog_train, catalog_val, catalog_test = build_splits(
    GENOME_CATALOG, VAL_PER_GROUP, TEST_PER_GROUP, SEED
)
print(
    f"Catalog split (seed={SEED}): "
    f"{len(catalog_train)} train / {len(catalog_val)} val / {len(catalog_test)} test"
)
print(f"Holdout genomes: {len(TEST_GENOMES)}")


# Genomes available on disk
def available(accs):
    return [a for a in accs if (Path(DATA_DIR) / f"{a}.fasta").exists()]


TRAIN_GENOMES = available(catalog_train + catalog_val)  # train classifier on both
TEST_CATALOG = available(catalog_test)
TEST_HOLDOUT = available(TEST_GENOMES)

print(
    f"Available for training: {len(TRAIN_GENOMES)} | "
    f"catalog test: {len(TEST_CATALOG)} | holdout: {len(TEST_HOLDOUT)}"
)

# Phylum lookup
_PHYLUM = {g["accession"]: g["group"] for g in GENOME_CATALOG}
_PHYLUM.update(
    {
        "NC_002947.4": "Proteobacteria",
        "NC_002929.2": "Proteobacteria",
        "NC_003143.1": "Proteobacteria",
        "NC_003116.1": "Proteobacteria",
        "NC_004757.1": "Proteobacteria",
        "NC_008497.1": "Firmicutes",
        "NC_004350.2": "Firmicutes",
        "NC_006270.3": "Firmicutes",
        "NC_006274.1": "Firmicutes",
        "NC_003030.1": "Firmicutes",
        "NC_003155.5": "Actinobacteria",
        "NC_003450.3": "Actinobacteria",
        "NC_002677.1": "Actinobacteria",
        "NC_008268.1": "Actinobacteria",
        "NC_006958.1": "Actinobacteria",
        "NC_008818.1": "Archaea",
        "NC_015948.1": "Archaea",
        "NC_014408.1": "Archaea",
        "NC_019977.1": "Archaea",
        "NC_007644.1": "Archaea",
    }
)


# ── Feature helpers ───────────────────────────────────────────────────────────


def get_upstream(seq, orf, window=25):
    strand = orf.get("strand", "forward")
    gs = int(orf.get("genome_start", orf.get("start", 0)))
    ge = int(orf.get("genome_end", orf.get("end", 0)))
    if gs > ge:
        gs, ge = ge, gs
    if strand == "forward":
        return seq[max(0, gs - window - 1) : gs - 1].upper()
    raw = seq[ge : min(len(seq), ge + window)].upper()
    return raw.translate(_RC)[::-1]


# ── New feature helpers ───────────────────────────────────────────────────────

_CODON_TABLE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}
_CHARGED_POS = set("KRH")
_CHARGED_NEG = set("DE")
_POLAR = set("STNQ")
_HYDROPHOBIC = set("ACGILMFPWVY")


def nterm_features(orf_seq: str, n_codons: int = 5) -> dict:
    """N-terminal amino acid composition of first n_codons codons."""
    aas = []
    for i in range(0, min(n_codons * 3, len(orf_seq) - 2), 3):
        aa = _CODON_TABLE.get(orf_seq[i : i + 3].upper(), "X")
        if aa != "*":
            aas.append(aa)
    n = max(len(aas), 1)
    return {
        "nterm_charged_pos": sum(a in _CHARGED_POS for a in aas) / n,
        "nterm_charged_neg": sum(a in _CHARGED_NEG for a in aas) / n,
        "nterm_polar": sum(a in _POLAR for a in aas) / n,
        "nterm_hydrophobic": sum(a in _HYDROPHOBIC for a in aas) / n,
    }


def gc3_periodicity_strength(seq: str) -> float:
    """
    Coefficient of variation of GC% across codon positions 1/2/3.
    High value = strong position-specific GC bias = coding-like.
    """
    if len(seq) < 9:
        return 0.0
    gc = [0, 0, 0]
    n = 0
    for i in range(0, len(seq) - 2, 3):
        for pos in range(3):
            gc[pos] += seq[i + pos].upper() in "GC"
        n += 1
    if n == 0:
        return 0.0
    means = [g / n for g in gc]
    overall = sum(means) / 3
    if overall < 0.01:
        return 0.0
    variance = sum((m - overall) ** 2 for m in means) / 3
    return (variance**0.5) / overall  # CV


def f4_spacer(up):
    best = 0.0
    for m in SD_MOTIFS:
        idx = up.rfind(m)
        if idx < 0:
            continue
        sp = len(up) - (idx + len(m))
        if 4 <= sp <= 12:
            best = max(best, len(m) / 7.0)
    return best


def f5_gc_bias(up):
    if len(up) < 9:
        return 0.0
    gc1 = gc2 = gc3 = n = 0
    for i in range(0, len(up) - 2, 3):
        gc1 += up[i] in "GC"
        gc2 += up[i + 1] in "GC"
        gc3 += up[i + 2] in "GC"
        n += 1
    if n == 0:
        return 0.0
    r1, r2, r3 = gc1 / n, gc2 / n, gc3 / n
    return abs(r3 - (r1 + r2) / 2)


def upstream_imm(up, models):
    if len(up) < 9:
        return 0.0
    return score_imm_ratio(up, models["coding_imm"], models["noncoding_imm"], models["max_order"])


# ── New feature helpers ───────────────────────────────────────────────────────

# Anti-SD: 3' tail of 16S rRNA (5'->3', RNA). Varies by organism; E. coli consensus.
_ANTI_SD = "GAUCACCUCCUUA"
_PAIRS = {"A": "U", "U": "A", "G": "C", "C": "G"}
_WOBBLE = {("G", "U"), ("U", "G")}


def anti_sd_score(upstream_dna: str, anti_sd: str = _ANTI_SD) -> float:
    """
    ΔG proxy: best Watson-Crick + G:U wobble complementarity of anti-SD
    to mRNA upstream region within spacer 4-14 bp.
    Normalised to [0,1] by anti-SD length.
    """
    mrna = upstream_dna.upper().replace("T", "U")
    n = len(anti_sd)
    best = 0.0
    for i in range(len(mrna) - n + 1):
        spacer = len(mrna) - (i + n)
        if not (4 <= spacer <= 14):
            continue
        sc = 0.0
        for j, ab in enumerate(anti_sd):
            mb = mrna[i + n - 1 - j]  # antiparallel alignment
            if _PAIRS.get(ab) == mb:
                sc += 1.0
            elif (ab, mb) in _WOBBLE or (mb, ab) in _WOBBLE:
                sc += 0.5
        best = max(best, sc / n)
    return best


def ext_codon_score(top1_row, top2_row, genome_seq: str, models: dict) -> float:
    """
    Score the EXTENSION region unique to the longer candidate.
    Returns the codon log-ratio of that region (negative = non-coding-like).
    Returns 0 if both starts are at the same position.
    """
    t1_start = int(top1_row.get("genome_start", top1_row.get("start", 0)))
    t2_start = int(top2_row.get("genome_start", top2_row.get("start", 0)))
    t1_end = int(top1_row.get("genome_end", top1_row.get("end", 0)))
    strand = top1_row.get("strand", "forward")
    if t1_start > t1_end:
        t1_start, t1_end = t1_end, t1_start
    if t1_start == t2_start:
        return 0.0
    # Ensure top1 is the LONGER one (lower genome start in forward strand)
    long_start = min(t1_start, t2_start)
    short_start = max(t1_start, t2_start)
    if strand == "forward":
        ext_seq = genome_seq[long_start - 1 : short_start - 1].upper()
    else:
        raw = genome_seq[short_start:long_start].upper()
        ext_seq = raw.translate(_RC)[::-1]
    if len(ext_seq) < 3:
        return 0.0
    return score_codon_bias_ratio(ext_seq, models["codon_model"], models["background_codon_model"])


def dist_any_frame_stop(genome_seq: str, orf: dict, max_scan: int = 300) -> int:
    """
    Nearest stop codon in ANY reading frame upstream of start.
    Short = start sits right after a stop in some frame (clean boundary).
    """
    strand = orf.get("strand", "forward")
    gs = int(orf.get("genome_start", orf.get("start", 0)))
    ge = int(orf.get("genome_end", orf.get("end", 0)))
    if gs > ge:
        gs, ge = ge, gs
    if strand == "forward":
        region = genome_seq[max(0, gs - max_scan - 1) : gs - 1].upper()
    else:
        raw = genome_seq[ge : min(len(genome_seq), ge + max_scan)].upper()
        region = raw.translate(_RC)[::-1]
    # Scan all 3 frames backward
    min_dist = len(region)
    for frame in range(3):
        for j in range(frame, len(region) - 2, 3):
            codon = region[j : j + 3]
            if codon in STOPS:
                dist = len(region) - j
                min_dist = min(min_dist, dist)
    return min_dist


def build_len_prior(groups: dict, probs: np.ndarray):
    """
    Build length distribution from singleton LGB-kept groups.
    Returns (mean, std) of gene lengths for this genome.
    """
    lengths = []
    for i, (gid, gdf) in enumerate(groups.items()):
        if isinstance(gdf, list):
            gdf = pd.DataFrame(gdf)
        if len(gdf) != 1:
            continue
        if (float(probs[i]) if i < len(probs) else 0.0) < LGB_T:
            continue
        lengths.append(float(gdf.iloc[0].get("length", 0)))
    if len(lengths) < 10:
        return 800.0, 400.0  # fallback
    return float(np.mean(lengths)), float(np.std(lengths))


def post_start_codon_score(orf_seq: str, models: dict, n_codons: int = 5) -> float:
    """
    Codon log-ratio of codons 2 to n_codons+1 (skip the start codon itself).
    Captures whether post-start codon context looks gene-like.
    """
    region = orf_seq[3 : 3 + n_codons * 3]  # skip codon 1 (start)
    if len(region) < 3:
        return 0.0
    return score_codon_bias_ratio(region, models["codon_model"], models["background_codon_model"])


def build_start_context_pwm(
    groups: dict, probs: np.ndarray, genome_seq: str, window: int = 13, min_seqs: int = 50
):
    """
    Per-genome start-context PWM: [-10 .. +3] window centred on the A of ATG.
    Built from singleton LGB-kept groups (unambiguous real starts).
    """
    seqs = []
    for i, (gid, gdf) in enumerate(groups.items()):
        if isinstance(gdf, list):
            gdf = pd.DataFrame(gdf)
        if len(gdf) != 1:
            continue
        if (float(probs[i]) if i < len(probs) else 0.0) < LGB_T:
            continue
        row = gdf.iloc[0]
        strand = row.get("strand", "forward")
        gs = int(row.get("genome_start", row.get("start", 0)))
        ge = int(row.get("genome_end", row.get("end", 0)))
        if gs > ge:
            gs, ge = ge, gs
        if strand == "forward":
            ctx_start = gs - 11  # -10 before A
            ctx_end = gs + 3  # +3 after A
            if ctx_start < 0:
                continue
            ctx = genome_seq[ctx_start:ctx_end].upper()
        else:
            ctx_start = ge - 2
            ctx_end = ge + 11
            if ctx_end > len(genome_seq):
                continue
            ctx = genome_seq[ctx_start:ctx_end].upper().translate(_RC)[::-1]
        if len(ctx) == window:
            seqs.append(ctx)
    if len(seqs) < min_seqs:
        return None
    bi = {b: i for i, b in enumerate(BASES)}
    cnt = np.ones((window, 4))
    for s in seqs:
        for p, b in enumerate(s):
            if b in bi:
                cnt[p, bi[b]] += 1
    return np.log(cnt / cnt.sum(axis=1, keepdims=True) / 0.25)


def score_start_context(genome_seq: str, orf: dict, pwm) -> float:
    if pwm is None:
        return 0.0
    strand = orf.get("strand", "forward")
    gs = int(orf.get("genome_start", orf.get("start", 0)))
    ge = int(orf.get("genome_end", orf.get("end", 0)))
    if gs > ge:
        gs, ge = ge, gs
    window = pwm.shape[0]
    if strand == "forward":
        ctx_start = gs - 11
        ctx_end = gs + 3
        if ctx_start < 0:
            return 0.0
        ctx = genome_seq[ctx_start:ctx_end].upper()
    else:
        ctx_start = ge - 2
        ctx_end = ge + 11
        if ctx_end > len(genome_seq):
            return 0.0
        ctx = genome_seq[ctx_start:ctx_end].upper().translate(_RC)[::-1]
    if len(ctx) != window:
        return 0.0
    bi = {b: i for i, b in enumerate(BASES)}
    return sum(pwm[p, bi[b]] for p, b in enumerate(ctx) if b in bi)


def build_pwm(groups, probs, seq, window=20, min_seqs=50):
    seqs = []
    for i, (gid, gdf) in enumerate(groups.items()):
        if isinstance(gdf, list):
            gdf = pd.DataFrame(gdf)
        if len(gdf) != 1:
            continue
        if (float(probs[i]) if i < len(probs) else 0.0) < LGB_T:
            continue
        up = get_upstream(seq, gdf.iloc[0].to_dict(), window)
        if len(up) >= window:
            seqs.append(up[-window:])
    if len(seqs) < min_seqs:
        return None
    bi = {b: i for i, b in enumerate(BASES)}
    cnt = np.ones((window, 4))
    for s in seqs:
        for p, b in enumerate(s):
            if b in bi:
                cnt[p, bi[b]] += 1
    return np.log(cnt / cnt.sum(axis=1, keepdims=True) / 0.25)


def score_pwm(up, pwm):
    if pwm is None or len(up) < pwm.shape[0]:
        return 0.0
    bi = {b: i for i, b in enumerate(BASES)}
    return sum(pwm[p, bi[b]] for p, b in enumerate(up[-pwm.shape[0] :]) if b in bi)


def baseline_score(row):
    return (
        row.get("codon_score_norm", 0) * W["codon"]
        + row.get("imm_score_norm", 0) * W["imm"]
        + row.get("rbs_score_norm", 0) * W["rbs"]
        + row.get("length_score_norm", 0) * W["length"]
        + row.get("start_score_norm", 0) * W["start"]
    )


# ── Feature extraction for a single genome ───────────────────────────────────


def extract_features(
    acc, seq, ref_set, models, groups, probs, pwm, gc_pct, len_mean, len_std, ctx_pwm
):
    """
    For each LGB-kept group where real gene is top-1 or top-2:
    compute pairwise + group features and label.
    """
    rows = []
    for i, (gid, grp_df) in enumerate(groups.items()):
        if isinstance(grp_df, list):
            grp_df = pd.DataFrame(grp_df)
        if len(grp_df) == 0:
            continue
        if (float(probs[i]) if i < len(probs) else 0.0) < LGB_T:
            continue

        real = {
            (
                int(r.get("genome_start", r.get("start", 0))),
                int(r.get("genome_end", r.get("end", 0))),
            )
            for _, r in grp_df.iterrows()
            if (
                int(r.get("genome_start", r.get("start", 0))),
                int(r.get("genome_end", r.get("end", 0))),
            )
            in ref_set
        }
        if not real:
            continue

        grp_df = grp_df.copy()
        grp_df["_base"] = grp_df.apply(baseline_score, axis=1)
        sorted_idx = grp_df["_base"].sort_values(ascending=False).index

        top1 = grp_df.loc[sorted_idx[0]]
        if len(sorted_idx) < 2:
            continue
        top2 = grp_df.loc[sorted_idx[1]]

        t1_coord = (
            int(top1.get("genome_start", top1.get("start", 0))),
            int(top1.get("genome_end", top1.get("end", 0))),
        )
        t2_coord = (
            int(top2.get("genome_start", top2.get("start", 0))),
            int(top2.get("genome_end", top2.get("end", 0))),
        )

        # Only use groups where real gene is top-1 or top-2
        if t1_coord not in real and t2_coord not in real:
            continue

        label = 1 if t1_coord in real else 0  # 1=keep top-1, 0=flip to top-2

        scores = grp_df["_base"]
        s_range = scores.max() - scores.min()
        gap = float(top1["_base"]) - float(top2["_base"])
        n_near = int((scores >= float(top1["_base"]) - 0.5).sum()) - 1
        lengths = grp_df["length"].values
        l_rank_t1 = float((lengths < float(top1.get("length", 0))).sum()) / (
            max(len(lengths) - 1, 1)
        )
        l_cv = float(np.std(lengths) / max(np.mean(lengths), 1))
        frac_long = float((lengths > float(top1.get("length", 0))).sum() / max(len(lengths), 1))
        rbs_vals = grp_df.get(
            "rbs_score_norm", pd.Series(np.zeros(len(grp_df)), index=grp_df.index)
        ).values
        rbs_range = float(rbs_vals.max() - rbs_vals.min())
        grp_rbs_mean = float(rbs_vals.mean())
        frac_atg = float(
            (grp_df.get("start_codon", pd.Series(["ATG"] * len(grp_df))) == "ATG").mean()
        )
        t1_rbs_rank = float((rbs_vals < float(top1.get("rbs_score_norm", 0))).sum()) / max(
            len(rbs_vals) - 1, 1
        )

        # Pairwise upstream features
        up1 = get_upstream(seq, top1.to_dict(), 25)
        up2 = get_upstream(seq, top2.to_dict(), 25)

        # New features
        t1seq = top1.get("sequence", "")
        t2seq = top2.get("sequence", "")
        t1_len = float(top1.get("length", 0))
        t2_len = float(top2.get("length", 0))
        t1_zscore = (t1_len - len_mean) / max(len_std, 1.0)
        t2_zscore = (t2_len - len_mean) / max(len_std, 1.0)

        feat = {
            "acc": acc,
            "phylum": _PHYLUM.get(acc, "?"),
            "label": label,
            # ── Existing pairwise differences ──────────────────────────────
            "d_baseline": gap,
            "d_rbs": float(top1.get("rbs_score_norm", 0)) - float(top2.get("rbs_score_norm", 0)),
            "d_start": float(top1.get("start_score_norm", 0))
            - float(top2.get("start_score_norm", 0)),
            "d_codon": float(top1.get("codon_score_norm", 0))
            - float(top2.get("codon_score_norm", 0)),
            "d_imm": float(top1.get("imm_score_norm", 0)) - float(top2.get("imm_score_norm", 0)),
            "d_length": t1_len - t2_len,
            "d_f4": f4_spacer(up1) - f4_spacer(up2),
            "d_f5": f5_gc_bias(up1) - f5_gc_bias(up2),
            "d_up_imm": upstream_imm(up1, models) - upstream_imm(up2, models),
            "d_genome_rbs": score_pwm(up1, pwm) - score_pwm(up2, pwm),
            # ── New pairwise features ───────────────────────────────────────
            # F6: Anti-SD score (ΔG proxy)
            "d_anti_sd": anti_sd_score(up1) - anti_sd_score(up2),
            "anti_sd_top1": anti_sd_score(up1),
            "anti_sd_top2": anti_sd_score(up2),
            # F7: Extension region codon score (unique portion of longer ORF)
            "ext_codon": ext_codon_score(top1.to_dict(), top2.to_dict(), seq, models),
            # F8: Distance to nearest any-frame stop upstream
            "d_any_stop_dist": dist_any_frame_stop(seq, top1.to_dict())
            - dist_any_frame_stop(seq, top2.to_dict()),
            "any_stop_top1": dist_any_frame_stop(seq, top1.to_dict()),
            "any_stop_top2": dist_any_frame_stop(seq, top2.to_dict()),
            # F9: Gene length z-score vs genome prior
            "d_len_zscore": t1_zscore - t2_zscore,
            "len_zscore_top1": t1_zscore,
            "len_zscore_top2": t2_zscore,
            # F10: Post-start codon score (codons 2-6)
            "d_post_start": post_start_codon_score(t1seq, models)
            - post_start_codon_score(t2seq, models),
            "post_start_top1": post_start_codon_score(t1seq, models),
            "post_start_top2": post_start_codon_score(t2seq, models),
            # F11: Start codon context PWM
            "d_ctx_pwm": score_start_context(seq, top1.to_dict(), ctx_pwm)
            - score_start_context(seq, top2.to_dict(), ctx_pwm),
            "ctx_pwm_top1": score_start_context(seq, top1.to_dict(), ctx_pwm),
            "ctx_pwm_top2": score_start_context(seq, top2.to_dict(), ctx_pwm),
            # ── Group context ───────────────────────────────────────────────
            "gap": gap,
            "score_range": float(s_range),
            "rel_gap": float(gap / max(s_range, 1e-6)),
            "n_near_ties": n_near,
            "n_orfs": len(grp_df),
            "top1_len_rank": l_rank_t1,
            "group_len_cv": l_cv,
            "frac_longer": frac_long,
            "grp_rbs_mean": grp_rbs_mean,
            "grp_rbs_range": rbs_range,
            "frac_atg": frac_atg,
            "top1_rbs_rank": t1_rbs_rank,
            "gc_pct": gc_pct,
            "both_atg": int(
                top1.get("start_codon", "") == "ATG" and top2.get("start_codon", "") == "ATG"
            ),
            # ── N-terminal AA composition (pairwise differences) ────────────
            **{
                f"d_{k}": nterm_features(str(top1.get("sequence", "")))[k]
                - nterm_features(str(top2.get("sequence", "")))[k]
                for k in [
                    "nterm_charged_pos",
                    "nterm_charged_neg",
                    "nterm_polar",
                    "nterm_hydrophobic",
                ]
            },
            **{
                f"t1_{k}": nterm_features(str(top1.get("sequence", "")))[k]
                for k in [
                    "nterm_charged_pos",
                    "nterm_charged_neg",
                    "nterm_polar",
                    "nterm_hydrophobic",
                ]
            },
            # ── GC3 periodicity strength upstream ──────────────────────────
            "d_gc3_period": gc3_periodicity_strength(up1) - gc3_periodicity_strength(up2),
            "gc3_period_t1": gc3_periodicity_strength(up1),
            "gc3_period_t2": gc3_periodicity_strength(up2),
            # ── Group-normalised features ───────────────────────────────────
            # d_X / group_score_range makes the feature scale-invariant
            "d_length_norm": (float(top1.get("length", 0)) - float(top2.get("length", 0)))
            / max(float(grp_df["length"].max() - grp_df["length"].min()), 1.0),
            "d_codon_norm": (
                float(top1.get("codon_score_norm", 0)) - float(top2.get("codon_score_norm", 0))
            )
            / max(
                float(
                    grp_df.get("codon_score_norm", pd.Series([0])).max()
                    - grp_df.get("codon_score_norm", pd.Series([0])).min()
                ),
                EPS,
            ),
            "d_imm_norm": (
                float(top1.get("imm_score_norm", 0)) - float(top2.get("imm_score_norm", 0))
            )
            / max(
                float(
                    grp_df.get("imm_score_norm", pd.Series([0])).max()
                    - grp_df.get("imm_score_norm", pd.Series([0])).min()
                ),
                EPS,
            ),
        }
        rows.append(feat)
    return rows


# ── Data collection ───────────────────────────────────────────────────────────

FEATURES_CSV = OUT_DIR / "pairwise_features_v3.csv"
FLIP_THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

# LGB hyperparameter grid — focused on the most impactful parameters
LGB_PARAM_GRID = list(
    ParameterGrid(
        {
            "num_leaves": [63, 127],
            "min_child_samples": [20, 100],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
        }
    )
)

if FEATURES_CSV.exists():
    print(f"\nLoading cached features from {FEATURES_CSV}")
    all_feats = pd.read_csv(FEATURES_CSV)
else:
    all_rows = []
    print(f"\nCollecting pairwise features from {len(TRAIN_GENOMES)} training genomes...")
    for idx, acc in enumerate(TRAIN_GENOMES, 1):
        phylum = _PHYLUM.get(acc, "?")
        genome = load_genome_sequence(f"{DATA_DIR}/{acc}.fasta")
        seq = genome["sequence"]
        gc_pct = (seq.count("G") + seq.count("C")) / len(seq)
        ref = pd.read_csv(get_gff_path(acc), sep="\t", comment="#", header=None)
        ref_set = set(zip(ref[ref[2] == "CDS"][3].astype(int), ref[ref[2] == "CDS"][4].astype(int)))
        print(f"  [{idx:>3}/{len(TRAIN_GENOMES)}] {acc} [{phylum}]", flush=True)
        with contextlib.redirect_stdout(io.StringIO()):
            orfs = find_orfs_candidates(seq, min_length=100)
            train = create_training_set(sequence=seq, all_orfs=orfs)
            interg = create_intergenic_set(sequence=seq, all_orfs=orfs)
            models_ = build_all_scoring_models(train, interg)
            scored = score_all_orfs(orfs, models_)
            f1 = filter_candidates(
                scored,
                **__import__(
                    "src.config", fromlist=["FIRST_FILTER_THRESHOLD"]
                ).FIRST_FILTER_THRESHOLD,
            )
            groups = organize_nested_orfs(f1)
        df_f = lgb_model.extract_group_features(groups, acc, weights=START_SELECTION_WEIGHTS)
        mf = lgb_model.model.feature_name_
        if mf and mf[0].startswith("Column_"):
            mf = lgb_model.feature_names or mf
        probs = lgb_model.model.predict_proba(df_f[mf].values, num_threads=1)[:, 1]
        pwm = build_pwm(groups, probs, seq)
        len_mean, len_std = build_len_prior(groups, probs)
        ctx_pwm = build_start_context_pwm(groups, probs, seq)
        rows = extract_features(
            acc, seq, ref_set, models_, groups, probs, pwm, gc_pct, len_mean, len_std, ctx_pwm
        )
        all_rows.extend(rows)
        print(f"           pairs collected: {len(rows)}")
    all_feats = pd.DataFrame(all_rows)
    all_feats.to_csv(FEATURES_CSV, index=False)
    print(f"Saved {len(all_feats):,} pairs to {FEATURES_CSV}")

# Feature columns (exclude metadata and any dropped v3 groups)
META = ["acc", "phylum", "label"]
FCOLS = [c for c in all_feats.columns if c not in META and c not in _V3_DROPPED]

# ── Filter to contested pairs only (same distribution as inference) ───────────
print(f"\nFull dataset: {len(all_feats):,} pairs")
all_feats = all_feats[all_feats["gap"] < CONTEST_T].reset_index(drop=True)
print(
    f"After filtering gap < {CONTEST_T}: {len(all_feats):,} contested pairs  "
    f"(label=1: {all_feats['label'].sum():,}  label=0: {(~all_feats['label'].astype(bool)).sum():,})"
)
print(f"Naive baseline (always keep top-1): {100*all_feats['label'].mean():.1f}%")
for ph in sorted(all_feats["phylum"].unique()):
    g = all_feats[all_feats["phylum"] == ph]
    print(
        f"  {ph:<18} {len(g):6,} pairs  "
        f"flip={100*(g['label']==0).mean():.1f}%  "
        f"naive_acc={100*g['label'].mean():.1f}%"
    )


# ── Leave-one-phylum-out cross-validation ─────────────────────────────────────

PHYLA = ["Proteobacteria", "Firmicutes", "Actinobacteria", "Archaea"]

naive_base = 100 * all_feats["label"].mean()


def temperature_scale(proba_val: np.ndarray, y_val: np.ndarray) -> float:
    """Find temperature T that minimises cross-entropy on validation probabilities."""

    def nll(T):
        p = expit(sp_logit(np.clip(proba_val, 1e-7, 1 - 1e-7)) / T)
        return -np.mean(y_val * np.log(p + 1e-7) + (1 - y_val) * np.log(1 - p + 1e-7))

    return minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded").x


def apply_temperature(proba: np.ndarray, T: float) -> np.ndarray:
    return expit(sp_logit(np.clip(proba, 1e-7, 1 - 1e-7)) / T)


def lopo_cv(make_clf, calibrate=False):
    """
    Leave-one-phylum-out CV.
    Returns (mean_auroc, best_flip_threshold, best_acc_at_that_t).
    """
    fold_aurocs = []
    fold_accs_by_t = {t: [] for t in FLIP_THRESHOLDS}
    for held_ph in PHYLA:
        train_mask = all_feats["phylum"] != held_ph
        test_mask = all_feats["phylum"] == held_ph
        X_train = all_feats.loc[train_mask, FCOLS].fillna(0).values
        y_train = all_feats.loc[train_mask, "label"].values
        X_test = all_feats.loc[test_mask, FCOLS].fillna(0).values
        y_test = all_feats.loc[test_mask, "label"].values
        if len(y_test) == 0:
            for t in FLIP_THRESHOLDS:
                fold_accs_by_t[t].append(np.nan)
            fold_aurocs.append(np.nan)
            continue
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        clf = make_clf()
        clf.fit(X_train, y_train)
        raw_proba = clf.predict_proba(X_test)[:, 1]
        if calibrate == "temperature":
            T = temperature_scale(clf.predict_proba(X_train)[:, 1], y_train)
            proba = apply_temperature(raw_proba, T)
        elif calibrate == "isotonic":
            cal = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
            cal.fit(X_train, y_train)
            proba = cal.predict_proba(X_test)[:, 1]
        else:
            proba = raw_proba
        auc_ = roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else 0.5
        fold_aurocs.append(auc_)
        for t in FLIP_THRESHOLDS:
            flip_mask = proba < (1 - t)
            correct = (~flip_mask & (y_test == 1)) | (flip_mask & (y_test == 0))
            fold_accs_by_t[t].append(correct.mean())
    mean_auc = np.nanmean(fold_aurocs)
    best_t = max(FLIP_THRESHOLDS, key=lambda t: np.nanmean(fold_accs_by_t[t]))
    best_acc = 100 * np.nanmean(fold_accs_by_t[best_t])
    return mean_auc, best_t, best_acc, fold_accs_by_t


# ── Step 1: LGB hyperparameter search ────────────────────────────────────────
print("\n=== LGB HYPERPARAMETER SEARCH (LOPO-CV) ===")
print(f"  Naive baseline: {naive_base:.1f}%")
print(f"  Searching {len(LGB_PARAM_GRID)} configurations...")

best_lgb_params = None
best_lgb_acc = 0.0
best_lgb_t = 0.80

for params in LGB_PARAM_GRID:

    def make_lgb(p=params):
        return lgb_sk.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
            **p,
        )

    auc, bt, bacc, _ = lopo_cv(make_lgb, calibrate=False)
    marker = " *" if bacc > best_lgb_acc else ""
    print(f"  {params}  AUROC={auc:.3f}  best_t={bt}  acc={bacc:.1f}%{marker}")
    if bacc > best_lgb_acc:
        best_lgb_acc = bacc
        best_lgb_params = params
        best_lgb_t = bt

print(f"\n  Best LGB params: {best_lgb_params}")
print(f"  Best LGB acc:    {best_lgb_acc:.1f}%  (t={best_lgb_t})")


# ── Step 2: Compare uncalibrated vs calibrated best LGB ──────────────────────
print("\n=== CALIBRATION COMPARISON ===")
print(f"  {'Config':<28}  {'AUROC':>7}", end="")
for t in FLIP_THRESHOLDS:
    print(f"  {f't={t}':>8}", end="")
print(f"  {'BestT':>7}  {'BestAcc':>8}")
print("  " + "-" * 100)

cv_results = {}


def make_best_lgb():
    return lgb_sk.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
        **best_lgb_params,
    )


for label, make_fn, calibrate in [
    ("LGB_tuned", make_best_lgb, False),
    ("LGB+temp_scale", make_best_lgb, "temperature"),
    ("LGB+isotonic", make_best_lgb, "isotonic"),
    (
        "LogReg+temp_scale",
        lambda: LogisticRegression(
            max_iter=1000, class_weight="balanced", C=1.0, random_state=SEED
        ),
        "temperature",
    ),
]:
    auc, bt, bacc, fold_accs = lopo_cv(make_fn, calibrate=calibrate)
    cv_results[label] = {
        "auroc": auc,
        "best_t": bt,
        "best_acc": bacc,
        "fold_accs": fold_accs,
        "calibrate": calibrate,
        "make_fn": make_fn,
    }
    print(f"  {label:<28}  {auc:>7.3f}", end="")
    for t in FLIP_THRESHOLDS:
        a = 100 * np.nanmean(fold_accs[t])
        flag = "*" if a > naive_base else " "
        print(f"  {a:>7.1f}%{flag}", end="")
    print(f"  {bt:>7.2f}  {bacc:>7.1f}%")


# ── Train final model on all training data, test on held-out genomes ──────────

best_model_name = max(cv_results, key=lambda m: cv_results[m]["best_acc"])
best_flip_t = cv_results[best_model_name]["best_t"]
use_calibration = cv_results[best_model_name]["calibrate"]
print(
    f"\nBest config: {best_model_name}  "
    f"AUROC={cv_results[best_model_name]['auroc']:.3f}  "
    f"flip_threshold={best_flip_t}  calibrate={use_calibration}  "
    f"CV_acc={cv_results[best_model_name]['best_acc']:.1f}%  "
    f"(naive={naive_base:.1f}%)"
)

# Train on all available training data
X_all = all_feats[FCOLS].fillna(0).values
y_all = all_feats["label"].values
scaler_f = StandardScaler()
X_all_sc = scaler_f.fit_transform(X_all)
final_clf = cv_results[best_model_name]["make_fn"]()
final_clf.fit(X_all_sc, y_all)
temp_T = None
if use_calibration == "temperature":
    temp_T = temperature_scale(final_clf.predict_proba(X_all_sc)[:, 1], y_all)
    print(f"  Temperature scaling T={temp_T:.4f}")
elif use_calibration == "isotonic":
    cal_clf = CalibratedClassifierCV(final_clf, method="isotonic", cv="prefit")
    cal_clf.fit(X_all_sc, y_all)
    final_clf = cal_clf

# Feature importances
if hasattr(final_clf, "coef_"):
    imps = np.abs(final_clf.coef_[0])
elif hasattr(final_clf, "feature_importances_"):
    imps = final_clf.feature_importances_
else:
    imps = np.zeros(len(FCOLS))

print("\n=== TOP FEATURES ===")
for feat, imp in sorted(zip(FCOLS, imps), key=lambda x: -x[1])[:10]:
    print(f"  {feat:<22}  {imp:.4f}")


# ── Evaluate on test genomes ──────────────────────────────────────────────────

from src.config import FIRST_FILTER_THRESHOLD


def evaluate_on_genomes(accessions, label):
    """Run two-stage start selection and report accuracy vs baseline."""
    records = []
    for idx, acc in enumerate(accessions, 1):
        phylum = _PHYLUM.get(acc, "?")
        genome = load_genome_sequence(f"{DATA_DIR}/{acc}.fasta")
        seq = genome["sequence"]
        gc_pct = (seq.count("G") + seq.count("C")) / len(seq)
        ref = pd.read_csv(get_gff_path(acc), sep="\t", comment="#", header=None)
        ref_set = set(zip(ref[ref[2] == "CDS"][3].astype(int), ref[ref[2] == "CDS"][4].astype(int)))
        print(f"  [{idx:>2}/{len(accessions)}] {acc} [{phylum}]", flush=True)
        with contextlib.redirect_stdout(io.StringIO()):
            orfs = find_orfs_candidates(seq, min_length=100)
            train = create_training_set(sequence=seq, all_orfs=orfs)
            interg = create_intergenic_set(sequence=seq, all_orfs=orfs)
            models_ = build_all_scoring_models(train, interg)
            scored = score_all_orfs(orfs, models_)
            f1 = filter_candidates(scored, **FIRST_FILTER_THRESHOLD)
            groups = organize_nested_orfs(f1)
        df_f = lgb_model.extract_group_features(groups, acc, weights=START_SELECTION_WEIGHTS)
        mf = lgb_model.model.feature_name_
        if mf and mf[0].startswith("Column_"):
            mf = lgb_model.feature_names or mf
        probs = lgb_model.model.predict_proba(df_f[mf].values, num_threads=1)[:, 1]
        pwm = build_pwm(groups, probs, seq)
        len_mean_, len_std_ = build_len_prior(groups, probs)
        ctx_pwm_ = build_start_context_pwm(groups, probs, seq)

        base_correct = new_correct = n_total = 0
        for i, (gid, grp_df) in enumerate(groups.items()):
            if isinstance(grp_df, list):
                grp_df = pd.DataFrame(grp_df)
            if len(grp_df) == 0:
                continue
            if (float(probs[i]) if i < len(probs) else 0.0) < LGB_T:
                continue
            real = {
                (
                    int(r.get("genome_start", r.get("start", 0))),
                    int(r.get("genome_end", r.get("end", 0))),
                )
                for _, r in grp_df.iterrows()
                if (
                    int(r.get("genome_start", r.get("start", 0))),
                    int(r.get("genome_end", r.get("end", 0))),
                )
                in ref_set
            }
            if not real:
                continue
            n_total += 1
            grp_df = grp_df.copy()
            grp_df["_base"] = grp_df.apply(baseline_score, axis=1)
            sorted_idx = grp_df["_base"].sort_values(ascending=False).index
            t1 = grp_df.loc[sorted_idx[0]]
            t1c = (
                int(t1.get("genome_start", t1.get("start", 0))),
                int(t1.get("genome_end", t1.get("end", 0))),
            )
            if t1c in real:
                base_correct += 1

            gap = float(t1["_base"]) - (
                float(grp_df.loc[sorted_idx[1], "_base"]) if len(sorted_idx) > 1 else -999
            )
            if gap >= CONTEST_T or len(sorted_idx) < 2:
                if t1c in real:
                    new_correct += 1
                continue

            # Contested: extract features and classify
            t2 = grp_df.loc[sorted_idx[1]]
            up1 = get_upstream(seq, t1.to_dict(), 25)
            up2 = get_upstream(seq, t2.to_dict(), 25)
            lengths = grp_df["length"].values
            scores = grp_df["_base"]
            s_range = scores.max() - scores.min()
            rbs_vals = grp_df.get(
                "rbs_score_norm", pd.Series(np.zeros(len(grp_df)), index=grp_df.index)
            ).values
            t1s = t1.get("sequence", "")
            t2s = t2.get("sequence", "")
            t1l = float(t1.get("length", 0))
            t2l = float(t2.get("length", 0))
            t1z = (t1l - len_mean_) / max(len_std_, 1.0)
            t2z = (t2l - len_mean_) / max(len_std_, 1.0)
            fv = {
                "d_baseline": gap,
                "d_rbs": float(t1.get("rbs_score_norm", 0)) - float(t2.get("rbs_score_norm", 0)),
                "d_start": float(t1.get("start_score_norm", 0))
                - float(t2.get("start_score_norm", 0)),
                "d_codon": float(t1.get("codon_score_norm", 0))
                - float(t2.get("codon_score_norm", 0)),
                "d_imm": float(t1.get("imm_score_norm", 0)) - float(t2.get("imm_score_norm", 0)),
                "d_length": t1l - t2l,
                "d_f4": f4_spacer(up1) - f4_spacer(up2),
                "d_f5": f5_gc_bias(up1) - f5_gc_bias(up2),
                "d_up_imm": upstream_imm(up1, models_) - upstream_imm(up2, models_),
                "d_genome_rbs": score_pwm(up1, pwm) - score_pwm(up2, pwm),
                "d_anti_sd": anti_sd_score(up1) - anti_sd_score(up2),
                "anti_sd_top1": anti_sd_score(up1),
                "anti_sd_top2": anti_sd_score(up2),
                "ext_codon": ext_codon_score(t1.to_dict(), t2.to_dict(), seq, models_),
                "d_any_stop_dist": dist_any_frame_stop(seq, t1.to_dict())
                - dist_any_frame_stop(seq, t2.to_dict()),
                "any_stop_top1": dist_any_frame_stop(seq, t1.to_dict()),
                "any_stop_top2": dist_any_frame_stop(seq, t2.to_dict()),
                "d_len_zscore": t1z - t2z,
                "len_zscore_top1": t1z,
                "len_zscore_top2": t2z,
                "d_post_start": post_start_codon_score(t1s, models_)
                - post_start_codon_score(t2s, models_),
                "post_start_top1": post_start_codon_score(t1s, models_),
                "post_start_top2": post_start_codon_score(t2s, models_),
                "d_ctx_pwm": score_start_context(seq, t1.to_dict(), ctx_pwm_)
                - score_start_context(seq, t2.to_dict(), ctx_pwm_),
                "ctx_pwm_top1": score_start_context(seq, t1.to_dict(), ctx_pwm_),
                "ctx_pwm_top2": score_start_context(seq, t2.to_dict(), ctx_pwm_),
                "gap": gap,
                "score_range": float(s_range),
                "rel_gap": float(gap / max(s_range, 1e-6)),
                "n_near_ties": int((scores >= float(t1["_base"]) - 0.5).sum()) - 1,
                "n_orfs": len(grp_df),
                "top1_len_rank": float((lengths < t1l).sum()) / max(len(lengths) - 1, 1),
                "group_len_cv": float(np.std(lengths) / max(np.mean(lengths), 1)),
                "frac_longer": float((lengths > t1l).sum() / max(len(lengths), 1)),
                "grp_rbs_mean": float(rbs_vals.mean()),
                "grp_rbs_range": float(rbs_vals.max() - rbs_vals.min()),
                "frac_atg": float(
                    (grp_df.get("start_codon", pd.Series(["ATG"] * len(grp_df))) == "ATG").mean()
                ),
                "top1_rbs_rank": float((rbs_vals < float(t1.get("rbs_score_norm", 0))).sum())
                / max(len(rbs_vals) - 1, 1),
                "gc_pct": gc_pct,
                "both_atg": int(
                    t1.get("start_codon", "") == "ATG" and t2.get("start_codon", "") == "ATG"
                ),
                # N-terminal AA composition
                **{
                    f"d_{k}": nterm_features(str(t1.get("sequence", "")))[k]
                    - nterm_features(str(t2.get("sequence", "")))[k]
                    for k in [
                        "nterm_charged_pos",
                        "nterm_charged_neg",
                        "nterm_polar",
                        "nterm_hydrophobic",
                    ]
                },
                **{
                    f"t1_{k}": nterm_features(str(t1.get("sequence", "")))[k]
                    for k in [
                        "nterm_charged_pos",
                        "nterm_charged_neg",
                        "nterm_polar",
                        "nterm_hydrophobic",
                    ]
                },
                # GC3 periodicity strength upstream
                "d_gc3_period": gc3_periodicity_strength(up1) - gc3_periodicity_strength(up2),
                "gc3_period_t1": gc3_periodicity_strength(up1),
                "gc3_period_t2": gc3_periodicity_strength(up2),
                # Group-normalised features
                "d_length_norm": (t1l - t2l) / max(float(lengths.max() - lengths.min()), 1.0),
                "d_codon_norm": (
                    float(t1.get("codon_score_norm", 0)) - float(t2.get("codon_score_norm", 0))
                )
                / max(
                    float(
                        grp_df.get("codon_score_norm", pd.Series([0])).max()
                        - grp_df.get("codon_score_norm", pd.Series([0])).min()
                    ),
                    EPS,
                ),
                "d_imm_norm": (
                    float(t1.get("imm_score_norm", 0)) - float(t2.get("imm_score_norm", 0))
                )
                / max(
                    float(
                        grp_df.get("imm_score_norm", pd.Series([0])).max()
                        - grp_df.get("imm_score_norm", pd.Series([0])).min()
                    ),
                    EPS,
                ),
            }
            X_fv = scaler_f.transform(np.array([[fv[c] for c in FCOLS]]))
            prob_keep = final_clf.predict_proba(X_fv)[0, 1]
            if temp_T is not None:
                prob_keep = float(apply_temperature(np.array([prob_keep]), temp_T)[0])
            flip = prob_keep < (1 - best_flip_t)
            winner_idx = sorted_idx[1] if flip else sorted_idx[0]
            wc = (
                int(
                    grp_df.loc[winner_idx].get(
                        "genome_start", grp_df.loc[winner_idx].get("start", 0)
                    )
                ),
                int(grp_df.loc[winner_idx].get("genome_end", grp_df.loc[winner_idx].get("end", 0))),
            )
            if wc in real:
                new_correct += 1

        records.append(
            {
                "acc": acc,
                "phylum": phylum,
                "baseline": base_correct / max(n_total, 1),
                "new": new_correct / max(n_total, 1),
                "n": n_total,
            }
        )
        print(
            f"     baseline={100*base_correct/max(n_total,1):.1f}%  "
            f"new={100*new_correct/max(n_total,1):.1f}%  "
            f"delta={100*(new_correct-base_correct)/max(n_total,1):+.1f}pp"
        )
    return pd.DataFrame(records)


print(f"\n=== EVALUATION ON CATALOG TEST GENOMES ({len(TEST_CATALOG)}) ===")
df_cat = evaluate_on_genomes(TEST_CATALOG, "catalog_test")

print(f"\n=== EVALUATION ON HOLDOUT GENOMES ({len(TEST_HOLDOUT)}) ===")
df_hld = evaluate_on_genomes(TEST_HOLDOUT, "holdout")

df_all_test = pd.concat([df_cat, df_hld])

print("\n=== FINAL RESULTS BY PHYLUM ===")
print(f"  {'Phylum':<18}  {'N':>6}  {'Baseline':>10}  {'NewModel':>10}  {'Delta':>8}")
print("  " + "-" * 58)
for ph in PHYLA:
    g = df_all_test[df_all_test["phylum"] == ph]
    if len(g) == 0:
        continue
    b = 100 * g["baseline"].mean()
    n = 100 * g["new"].mean()
    flag = " *" if n > b + 0.3 else ""
    print(f"  {ph:<18}  {g['n'].sum():>6,}  {b:>9.1f}%  {n:>9.1f}%  {n-b:>+7.1f}pp{flag}")
b_all = 100 * df_all_test["baseline"].mean()
n_all = 100 * df_all_test["new"].mean()
print(
    f"  {'OVERALL':<18}  {df_all_test['n'].sum():>6,}  {b_all:>9.1f}%  {n_all:>9.1f}%  {n_all-b_all:>+7.1f}pp"
)

df_all_test.to_csv(OUT_DIR / "start_classifier_results.csv", index=False)

# ── Save model for integration ────────────────────────────────────────────────
import pickle

model_bundle = {
    "clf": final_clf,
    "scaler": scaler_f,
    "features": FCOLS,
    "contest_t": CONTEST_T,
    "flip_t": best_flip_t,
    "calibrated": use_calibration,
    "temperature_T": temp_T,  # None if not using temperature scaling
}
model_path = Path(_args.out_model)
with open(model_path, "wb") as f:
    pickle.dump(model_bundle, f)
print("Saved: start_classifier_results.csv")
print(f"Saved: {model_path}")
