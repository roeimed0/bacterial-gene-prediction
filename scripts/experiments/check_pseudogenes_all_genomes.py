"""Quick check: pseudogene counts in all 20 holdout genome GFF files."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import TEST_GENOMES
from src.data_management import get_gff_path

print(f"\n  {'Acc':<15} {'Total CDS':>10} {'Pseudogenes':>12} {'Real genes':>10} {'Pseudo%':>8}")
print(f"  {'-'*15} {'-'*10} {'-'*12} {'-'*10} {'-'*8}")
total_real = total_pseudo = 0
for acc in TEST_GENOMES:
    try:
        r = pd.read_csv(get_gff_path(acc), sep="\t", comment="#", header=None)
        cds = r[r[2] == "CDS"]
        n_total = len(cds)
        if 8 in cds.columns:
            n_pseudo = int(
                cds[8].str.contains("pseudo=true|pseudogene", case=False, na=False).sum()
            )
        else:
            n_pseudo = 0
        n_real = n_total - n_pseudo
        total_real += n_real
        total_pseudo += n_pseudo
        flag = "  <-- HIGH" if n_pseudo > 100 else ""
        print(
            f"  {acc:<15} {n_total:>10} {n_pseudo:>12} {n_real:>10} "
            f"{n_pseudo/max(n_total,1)*100:>7.1f}%{flag}"
        )
    except Exception as e:
        print(f"  {acc:<15} ERROR: {e}")
print(
    f"  {'TOTAL':<15} {total_real+total_pseudo:>10} {total_pseudo:>12} "
    f"{total_real:>10} {total_pseudo/(total_real+total_pseudo)*100:>7.1f}%"
)
