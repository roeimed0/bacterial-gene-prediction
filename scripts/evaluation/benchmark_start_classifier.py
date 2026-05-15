"""
Benchmark: full pipeline WITH vs WITHOUT the new start-selection classifier.

Runs the complete gene prediction pipeline on all 20 holdout genomes twice:
  A) Current pipeline  — existing select_best_starts() weighted sum
  B) New pipeline      — same + pairwise LightGBM classifier for contested groups

Reports per genome and per phylum:
  Sensitivity, Precision, F1 (at the gene coordinate level)
  Plus start selection accuracy (what % of predicted genes match reference exactly)

Run from repo root:
    python scripts/evaluation/benchmark_start_classifier.py
"""

import contextlib
import io
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.comparative_analysis import compare_orfs_to_reference
from src.config import FIRST_FILTER_THRESHOLD, START_SELECTION_WEIGHTS, TEST_GENOMES
from src.data_management import get_data_dir, get_gff_path, load_genome_sequence
from src.ml_models import HybridGeneFilter, OrfGroupClassifier
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
    select_best_starts,
)

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
DATA_DIR = get_data_dir("full_dataset")
OUT_DIR = Path(__file__).parent.parent.parent / "lgb_attribution_results"
OUT_DIR.mkdir(exist_ok=True)

_PHYLA = {
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

# Load models
lgb_clf = OrfGroupClassifier()
lgb_clf.load(str(MODELS_DIR / "orf_classifier_lgb.pkl"))

hf = HybridGeneFilter()
with contextlib.redirect_stdout(io.StringIO()):
    hf.load(str(MODELS_DIR / "hybrid_best_model.pkl"))

with open(MODELS_DIR / "start_selector.pkl", "rb") as f:
    ss_bundle = pickle.load(f)
SS_CLF = ss_bundle["clf"]
SS_SCALER = ss_bundle["scaler"]
SS_FCOLS = ss_bundle["features"]
CONTEST_T = ss_bundle["contest_t"]
FLIP_T = ss_bundle["flip_t"]

from json import load as jload

with open(MODELS_DIR / "thresholds.json") as f:
    thr = jload(f)
LGB_T = thr.get("orf_classifier_lgb", {}).get("threshold", 0.05)
HF_T = thr.get("hybrid_best_model", {}).get("threshold", 0.471)

genomes = [a for a in TEST_GENOMES if (Path(DATA_DIR) / f"{a}.fasta").exists()]

_RC = str.maketrans("ACGT", "TGCA")
SD_MOTIFS = ["AAGGAGG", "AAGGAG", "AGGAG", "GGAGG", "GAGG", "AAGG", "AGGA"]
STOPS = {"TAA", "TAG", "TGA"}
BASES = "ACGT"
EPS = 1e-9

W = START_SELECTION_WEIGHTS


# ── Feature helpers (identical to train_start_classifier.py) ─────────────────


def get_upstream(seq, orf, window=25):
    strand = orf.get("strand", "forward")
    gs = int(orf.get("genome_start", orf.get("start", 0)))
    ge = int(orf.get("genome_end", orf.get("end", 0)))
    if gs > ge:
        gs, ge = ge, gs
    if strand == "forward":
        return seq[max(0, gs - window - 1) : gs - 1].upper()
    return seq[ge : min(len(seq), ge + window)].upper().translate(_RC)[::-1]


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


_ANTI_SD = "GAUCACCUCCUUA"
_PAIRS = {"A": "U", "U": "A", "G": "C", "C": "G"}
_WOBBLE = {("G", "U"), ("U", "G")}


def anti_sd_score(up):
    mrna = up.upper().replace("T", "U")
    n = len(_ANTI_SD)
    best = 0.0
    for i in range(len(mrna) - n + 1):
        sp = len(mrna) - (i + n)
        if not (4 <= sp <= 14):
            continue
        sc = sum(
            (
                1.0
                if _PAIRS.get(ab) == mrna[i + n - 1 - j]
                else (
                    0.5
                    if (ab, mrna[i + n - 1 - j]) in _WOBBLE or (mrna[i + n - 1 - j], ab) in _WOBBLE
                    else 0
                )
            )
            for j, ab in enumerate(_ANTI_SD)
        )
        best = max(best, sc / n)
    return best


def ext_codon_score(t1, t2, seq, models):
    gs1 = int(t1.get("genome_start", t1.get("start", 0)))
    ge1 = int(t1.get("genome_end", t1.get("end", 0)))
    gs2 = int(t2.get("genome_start", t2.get("start", 0)))
    strand = t1.get("strand", "forward")
    if gs1 > ge1:
        gs1, ge1 = ge1, gs1
    ls = min(gs1, gs2)
    ss = max(gs1, gs2)
    if ls == ss:
        return 0.0
    ext = (
        seq[ls - 1 : ss - 1].upper()
        if strand == "forward"
        else seq[ss:ls].upper().translate(_RC)[::-1]
    )
    return (
        score_codon_bias_ratio(ext, models["codon_model"], models["background_codon_model"])
        if len(ext) >= 3
        else 0.0
    )


def dist_any_stop(seq, orf, max_scan=300):
    strand = orf.get("strand", "forward")
    gs = int(orf.get("genome_start", orf.get("start", 0)))
    ge = int(orf.get("genome_end", orf.get("end", 0)))
    if gs > ge:
        gs, ge = ge, gs
    if strand == "forward":
        region = seq[max(0, gs - max_scan - 1) : gs - 1].upper()
    else:
        region = seq[ge : min(len(seq), ge + max_scan)].upper().translate(_RC)[::-1]
    mn = len(region)
    for fr in range(3):
        for j in range(fr, len(region) - 2, 3):
            if region[j : j + 3] in STOPS:
                mn = min(mn, len(region) - j)
    return mn


def build_len_prior(groups, probs):
    ls = []
    for i, (gid, gdf) in enumerate(groups.items()):
        if isinstance(gdf, list):
            gdf = pd.DataFrame(gdf)
        if len(gdf) != 1:
            continue
        if (float(probs[i]) if i < len(probs) else 0.0) < LGB_T:
            continue
        ls.append(float(gdf.iloc[0].get("length", 0)))
    return (float(np.mean(ls)), float(np.std(ls))) if len(ls) >= 10 else (800.0, 400.0)


def post_start_score(seq_str, models, n=5):
    r = seq_str[3 : 3 + n * 3]
    return (
        score_codon_bias_ratio(r, models["codon_model"], models["background_codon_model"])
        if len(r) >= 3
        else 0.0
    )


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


def build_ctx_pwm(groups, probs, seq, window=13, min_seqs=50):
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
            if gs - 11 < 0:
                continue
            ctx = seq[gs - 11 : gs + 3].upper()
        else:
            if ge + 11 > len(seq):
                continue
            ctx = seq[ge - 2 : ge + 11].upper().translate(_RC)[::-1]
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


def score_ctx_pwm(seq, orf, pwm):
    if pwm is None:
        return 0.0
    strand = orf.get("strand", "forward")
    window = pwm.shape[0]
    gs = int(orf.get("genome_start", orf.get("start", 0)))
    ge = int(orf.get("genome_end", orf.get("end", 0)))
    if gs > ge:
        gs, ge = ge, gs
    if strand == "forward":
        if gs - 11 < 0:
            return 0.0
        ctx = seq[gs - 11 : gs + 3].upper()
    else:
        if ge + 11 > len(seq):
            return 0.0
        ctx = seq[ge - 2 : ge + 11].upper().translate(_RC)[::-1]
    if len(ctx) != window:
        return 0.0
    bi = {b: i for i, b in enumerate(BASES)}
    return sum(pwm[p, bi[b]] for p, b in enumerate(ctx) if b in bi)


def select_best_starts_new(groups, seq, scoring_models, probs_lgb, pwm, len_mean, len_std, ctx_pwm):
    """
    Select best start codon per group using the new classifier for contested groups.
    Falls back to baseline weighted sum when gap >= CONTEST_T.
    Returns a DataFrame (same format as select_best_starts).
    """
    selected = []
    for i, (gid, grp_df) in enumerate(groups.items()):
        if isinstance(grp_df, list):
            grp_df = pd.DataFrame(grp_df)
        if len(grp_df) == 0:
            continue

        # Baseline score
        grp_df = grp_df.copy()
        grp_df["_base"] = (
            grp_df.get("codon_score_norm", 0) * W["codon"]
            + grp_df.get("imm_score_norm", 0) * W["imm"]
            + grp_df.get("rbs_score_norm", 0) * W["rbs"]
            + grp_df.get("length_score_norm", 0) * W["length"]
            + grp_df.get("start_score_norm", 0) * W["start"]
        )
        sorted_idx = grp_df["_base"].sort_values(ascending=False).index
        t1 = grp_df.loc[sorted_idx[0]]
        gap = (
            float(t1["_base"]) - float(grp_df.loc[sorted_idx[1], "_base"])
            if len(sorted_idx) > 1
            else 999.0
        )

        winner_idx = sorted_idx[0]

        if gap < CONTEST_T and len(sorted_idx) >= 2:
            t2 = grp_df.loc[sorted_idx[1]]
            up1 = get_upstream(seq, t1.to_dict(), 25)
            up2 = get_upstream(seq, t2.to_dict(), 25)
            lengths = grp_df["length"].values
            scores = grp_df["_base"]
            s_range = scores.max() - scores.min()
            rbs_v = grp_df.get(
                "rbs_score_norm", pd.Series(np.zeros(len(grp_df)), index=grp_df.index)
            ).values
            t1l = float(t1.get("length", 0))
            t2l = float(t2.get("length", 0))
            t1z = (t1l - len_mean) / max(len_std, 1.0)
            t2z = (t2l - len_mean) / max(len_std, 1.0)
            t1s = t1.get("sequence", "")
            t2s = t2.get("sequence", "")
            fv = {c: 0.0 for c in SS_FCOLS}
            fv.update(
                {
                    "d_baseline": gap,
                    "d_rbs": float(t1.get("rbs_score_norm", 0))
                    - float(t2.get("rbs_score_norm", 0)),
                    "d_start": float(t1.get("start_score_norm", 0))
                    - float(t2.get("start_score_norm", 0)),
                    "d_codon": float(t1.get("codon_score_norm", 0))
                    - float(t2.get("codon_score_norm", 0)),
                    "d_imm": float(t1.get("imm_score_norm", 0))
                    - float(t2.get("imm_score_norm", 0)),
                    "d_length": t1l - t2l,
                    "d_f4": f4_spacer(up1) - f4_spacer(up2),
                    "d_f5": f5_gc_bias(up1) - f5_gc_bias(up2),
                    "d_up_imm": score_imm_ratio(
                        up1,
                        scoring_models["coding_imm"],
                        scoring_models["noncoding_imm"],
                        scoring_models["max_order"],
                    )
                    - score_imm_ratio(
                        up2,
                        scoring_models["coding_imm"],
                        scoring_models["noncoding_imm"],
                        scoring_models["max_order"],
                    ),
                    "d_genome_rbs": score_pwm(up1, pwm) - score_pwm(up2, pwm),
                    "d_anti_sd": anti_sd_score(up1) - anti_sd_score(up2),
                    "anti_sd_top1": anti_sd_score(up1),
                    "anti_sd_top2": anti_sd_score(up2),
                    "ext_codon": ext_codon_score(t1.to_dict(), t2.to_dict(), seq, scoring_models),
                    "d_any_stop_dist": dist_any_stop(seq, t1.to_dict())
                    - dist_any_stop(seq, t2.to_dict()),
                    "any_stop_top1": dist_any_stop(seq, t1.to_dict()),
                    "any_stop_top2": dist_any_stop(seq, t2.to_dict()),
                    "d_len_zscore": t1z - t2z,
                    "len_zscore_top1": t1z,
                    "len_zscore_top2": t2z,
                    "d_post_start": post_start_score(t1s, scoring_models)
                    - post_start_score(t2s, scoring_models),
                    "post_start_top1": post_start_score(t1s, scoring_models),
                    "post_start_top2": post_start_score(t2s, scoring_models),
                    "d_ctx_pwm": score_ctx_pwm(seq, t1.to_dict(), ctx_pwm)
                    - score_ctx_pwm(seq, t2.to_dict(), ctx_pwm),
                    "ctx_pwm_top1": score_ctx_pwm(seq, t1.to_dict(), ctx_pwm),
                    "ctx_pwm_top2": score_ctx_pwm(seq, t2.to_dict(), ctx_pwm),
                    "gap": gap,
                    "score_range": float(s_range),
                    "rel_gap": float(gap / max(s_range, EPS)),
                    "n_near_ties": int((scores >= float(t1["_base"]) - 0.5).sum()) - 1,
                    "n_orfs": len(grp_df),
                    "top1_len_rank": float((lengths < t1l).sum()) / max(len(lengths) - 1, 1),
                    "group_len_cv": float(np.std(lengths) / max(np.mean(lengths), 1)),
                    "frac_longer": float((lengths > t1l).sum() / max(len(lengths), 1)),
                    "grp_rbs_mean": float(rbs_v.mean()),
                    "grp_rbs_range": float(rbs_v.max() - rbs_v.min()),
                    "frac_atg": float(
                        (
                            grp_df.get("start_codon", pd.Series(["ATG"] * len(grp_df))) == "ATG"
                        ).mean()
                    ),
                    "top1_rbs_rank": float((rbs_v < float(t1.get("rbs_score_norm", 0))).sum())
                    / max(len(rbs_v) - 1, 1),
                    "gc_pct": float((sum(seq.count(b) for b in "GC") / len(seq))),
                    "both_atg": int(
                        t1.get("start_codon", "") == "ATG" and t2.get("start_codon", "") == "ATG"
                    ),
                }
            )
            X = SS_SCALER.transform(np.array([[fv[c] for c in SS_FCOLS]]))
            prob_keep = SS_CLF.predict_proba(X)[0, 1]
            if prob_keep < (1 - FLIP_T):
                winner_idx = sorted_idx[1]

        selected.append(grp_df.loc[winner_idx])

    return pd.DataFrame(selected).reset_index(drop=True) if selected else pd.DataFrame()


# ── Benchmark loop ────────────────────────────────────────────────────────────

from src.config import SECOND_FILTER_THRESHOLD

records = []

for idx, acc in enumerate(genomes, 1):
    phylum = _PHYLA.get(acc, "?")
    genome = load_genome_sequence(f"{DATA_DIR}/{acc}.fasta")
    seq = genome["sequence"]
    print(f"[{idx:02d}/20] {acc}  [{phylum}]", flush=True)

    with contextlib.redirect_stdout(io.StringIO()):
        orfs = find_orfs_candidates(seq, min_length=100)
        train = create_training_set(sequence=seq, all_orfs=orfs)
        interg = create_intergenic_set(sequence=seq, all_orfs=orfs)
        models_ = build_all_scoring_models(train, interg)
        scored = score_all_orfs(orfs, models_)
        f1 = filter_candidates(scored, **FIRST_FILTER_THRESHOLD)
        groups = organize_nested_orfs(f1)

    df_f = lgb_clf.extract_group_features(groups, acc, weights=START_SELECTION_WEIGHTS)
    mf = lgb_clf.model.feature_name_
    if mf and mf[0].startswith("Column_"):
        mf = lgb_clf.feature_names or mf
    probs = lgb_clf.model.predict_proba(df_f[mf].values, num_threads=1)[:, 1]

    # Filter groups by LGB threshold
    kept_groups = {
        gid: gdf
        for i, (gid, gdf) in enumerate(groups.items())
        if (float(probs[i]) if i < len(probs) else 0.0) >= LGB_T
    }

    # Pre-compute per-genome objects for new selector
    pwm = build_pwm(groups, probs, seq)
    lm, ls = build_len_prior(groups, probs)
    ctx_pwm = build_ctx_pwm(groups, probs, seq)

    def run_pipeline(use_new_selector):
        if use_new_selector:
            top = select_best_starts_new(kept_groups, seq, models_, probs, pwm, lm, ls, ctx_pwm)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                top = select_best_starts(kept_groups, START_SELECTION_WEIGHTS)
        with contextlib.redirect_stdout(io.StringIO()):
            final = filter_candidates(top, **SECOND_FILTER_THRESHOLD)
        if hf is not None:
            with contextlib.redirect_stdout(io.StringIO()):
                final = hf.filter_candidates(
                    candidates=final, genome_id=acc, threshold=HF_T, batch_size=32
                )
        return final

    res_old = run_pipeline(False)
    res_new = run_pipeline(True)

    with contextlib.redirect_stdout(io.StringIO()):
        m_old = compare_orfs_to_reference(res_old, acc)
        m_new = compare_orfs_to_reference(res_new, acc)

    print(
        f"         OLD: sens={m_old['sensitivity']:.3f}  prec={m_old['precision']:.3f}  F1={m_old['f1_score']:.3f}  n={m_old.get('predicted',0)}"
    )
    print(
        f"         NEW: sens={m_new['sensitivity']:.3f}  prec={m_new['precision']:.3f}  F1={m_new['f1_score']:.3f}  n={m_new.get('predicted',0)}"
    )
    d_f1 = m_new["f1_score"] - m_old["f1_score"]
    print(f"         dF1={d_f1:+.4f}")

    records.append(
        {
            "acc": acc,
            "phylum": phylum,
            "old_sens": m_old["sensitivity"],
            "old_prec": m_old["precision"],
            "old_f1": m_old["f1_score"],
            "new_sens": m_new["sensitivity"],
            "new_prec": m_new["precision"],
            "new_f1": m_new["f1_score"],
            "d_sens": m_new["sensitivity"] - m_old["sensitivity"],
            "d_prec": m_new["precision"] - m_old["precision"],
            "d_f1": m_new["f1_score"] - m_old["f1_score"],
        }
    )

df = pd.DataFrame(records)
df.to_csv(OUT_DIR / "benchmark_start_classifier.csv", index=False)

print("\n=== BENCHMARK: OLD vs NEW START SELECTION ===")
print(
    f"  {'Phylum':<16}  {'OldSens':>8}  {'NewSens':>8}  {'dSens':>7}  "
    f"{'OldPrec':>8}  {'NewPrec':>8}  {'dPrec':>7}  "
    f"{'OldF1':>7}  {'NewF1':>7}  {'dF1':>7}"
)
print("  " + "-" * 90)

for ph in ["Proteobacteria", "Firmicutes", "Actinobacteria", "Archaea"]:
    g = df[df["phylum"] == ph]
    if len(g) == 0:
        continue
    flag = " *" if g["d_f1"].mean() > 0.001 else ""
    print(
        f"  {ph:<16}  {g['old_sens'].mean():>7.3f}   {g['new_sens'].mean():>7.3f}  "
        f"{g['d_sens'].mean():>+6.3f}   {g['old_prec'].mean():>7.3f}   "
        f"{g['new_prec'].mean():>7.3f}  {g['d_prec'].mean():>+6.3f}   "
        f"{g['old_f1'].mean():>6.3f}   {g['new_f1'].mean():>6.3f}  "
        f"{g['d_f1'].mean():>+6.3f}{flag}"
    )

print(
    f"  {'OVERALL':<16}  {df['old_sens'].mean():>7.3f}   {df['new_sens'].mean():>7.3f}  "
    f"{df['d_sens'].mean():>+6.3f}   {df['old_prec'].mean():>7.3f}   "
    f"{df['new_prec'].mean():>7.3f}  {df['d_prec'].mean():>+6.3f}   "
    f"{df['old_f1'].mean():>6.3f}   {df['new_f1'].mean():>6.3f}  "
    f"{df['d_f1'].mean():>+6.3f}"
)
print(f"\nSaved: benchmark_start_classifier.csv")
