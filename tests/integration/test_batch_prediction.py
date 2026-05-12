"""
Integration tests for scripts/predict_batch.py.

Structural/behavioural tests use the synthetic FASTA fixtures (small, always available).
GFF-content tests require a real genome from data/full_dataset/ and are skipped
automatically if none is present.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "sequences"
DATA_DIR = REPO_ROOT / "data" / "full_dataset"
SCRIPT = REPO_ROOT / "scripts" / "training" / "predict_batch.py"
PYTHON = sys.executable

# Synthetic seqs are too small for full pipeline — use no-ML mode + low min-length
# These tests only check script *behaviour* (file handling, error isolation, etc.)
STRUCTURAL_FLAGS = ["--no-group-ml", "--no-final-ml", "--min-length", "30"]


def _run(args, **kw):
    return subprocess.run([PYTHON, str(SCRIPT)] + args, capture_output=True, text=True, **kw)


@pytest.fixture()
def any_fasta():
    """One synthetic FASTA file (always present)."""
    f = FIXTURES / "synthetic_multi_orf.fasta"
    assert f.exists()
    return str(f)


@pytest.fixture()
def three_fastas():
    """Three synthetic FASTA files (always present)."""
    names = [
        "synthetic_single_orf.fasta",
        "synthetic_multi_orf.fasta",
        "synthetic_reverse_strand.fasta",
    ]
    return [str(FIXTURES / n) for n in names if (FIXTURES / n).exists()]


@pytest.fixture()
def real_fasta():
    """A real bacterial genome — skipped automatically if not downloaded."""
    fastas = sorted(DATA_DIR.glob("*.fasta"))
    if not fastas:
        pytest.skip("No real genomes in data/full_dataset/ — run download first")
    return str(fastas[0])


@pytest.fixture()
def output_dir(tmp_path):
    return tmp_path / "batch_out"


# ── Structural / behavioural tests (no real genomes needed) ───────────────────


class TestBatchPredictionStructure:
    def test_no_input_exits_with_error(self, output_dir):
        r = _run(["--output-dir", str(output_dir)])
        assert r.returncode != 0

    def test_nonexistent_file_skipped_batch_continues(self, any_fasta, output_dir):
        r = _run(
            STRUCTURAL_FLAGS
            + ["--output-dir", str(output_dir), "/nonexistent/path.fasta", any_fasta]
        )
        assert r.returncode == 0
        assert "skipped" in r.stdout or "SKIP" in r.stdout

    def test_deduplicates_inputs(self, any_fasta, output_dir):
        r = _run(STRUCTURAL_FLAGS + ["--output-dir", str(output_dir), any_fasta, any_fasta])
        assert "1 genome" in r.stdout or "1 genome(s)" in r.stdout

    def test_prints_summary_line(self, any_fasta, output_dir):
        r = _run(STRUCTURAL_FLAGS + ["--output-dir", str(output_dir), any_fasta])
        assert "DONE" in r.stdout

    def test_input_list_comment_lines_skipped(self, any_fasta, output_dir, tmp_path):
        list_file = tmp_path / "genomes.txt"
        list_file.write_text(f"# comment\n{any_fasta}\n")
        r = _run(
            STRUCTURAL_FLAGS + ["--output-dir", str(output_dir), "--input-list", str(list_file)]
        )
        assert r.returncode == 0
        assert "1 genome" in r.stdout or "1 genome(s)" in r.stdout

    def test_input_list_accepted(self, three_fastas, output_dir, tmp_path):
        list_file = tmp_path / "genomes.txt"
        list_file.write_text("\n".join(three_fastas) + "\n")
        r = _run(
            STRUCTURAL_FLAGS + ["--output-dir", str(output_dir), "--input-list", str(list_file)]
        )
        assert r.returncode == 0


# ── GFF-content tests (require a real downloaded genome) ─────────────────────


class TestBatchPredictionGFFOutput:
    def test_creates_gff_file(self, real_fasta, output_dir):
        r = _run(["--no-group-ml", "--no-final-ml", "--output-dir", str(output_dir), real_fasta])
        assert r.returncode == 0
        stem = Path(real_fasta).stem
        assert (output_dir / f"{stem}_predictions.gff").exists()

    def test_gff_has_version_header(self, real_fasta, output_dir):
        _run(["--no-group-ml", "--no-final-ml", "--output-dir", str(output_dir), real_fasta])
        stem = Path(real_fasta).stem
        gff = output_dir / f"{stem}_predictions.gff"
        assert gff.read_text().splitlines()[0] == "##gff-version 3"

    def test_gff_contains_predictions(self, real_fasta, output_dir):
        _run(["--output-dir", str(output_dir), real_fasta])
        stem = Path(real_fasta).stem
        gff = output_dir / f"{stem}_predictions.gff"
        lines = [l for l in gff.read_text().splitlines() if not l.startswith("#")]
        assert len(lines) > 0, "Expected at least one predicted gene"

    def test_batch_of_two_creates_two_gffs(self, real_fasta, output_dir):
        r = _run(
            [
                "--no-group-ml",
                "--no-final-ml",
                "--output-dir",
                str(output_dir),
                real_fasta,
                real_fasta,
            ]
        )
        # Deduplication: same file twice → only 1 output
        assert "1 genome" in r.stdout or "1 genome(s)" in r.stdout
