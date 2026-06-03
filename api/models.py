"""
API Models - Request and Response schemas
"""

import json
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

# Load calibrated thresholds so API defaults always match production models.
# Falls back to safe values if thresholds.json is absent (e.g. test environments).
_thresholds: dict = {}
_thr_path = Path(__file__).parent.parent / "models" / "thresholds.json"
if _thr_path.exists():
    with open(_thr_path) as _f:
        _thresholds = json.load(_f)

_LGB_DEFAULT: float = _thresholds.get("orf_classifier_lgb", {}).get("threshold", 0.07)
_HF_DEFAULT: float = _thresholds.get("hybrid_best_model", {}).get("threshold", 0.471)


class PredictionRequest(BaseModel):
    """Request body for gene prediction"""

    sequence: str = Field(..., description="Genome sequence in FASTA format or raw DNA sequence")
    filename: Optional[str] = Field(
        None, description="Optional filename for output (without extension)"
    )
    use_group_ml: bool = Field(True, description="Use ML for group filtering")
    group_threshold: float = Field(
        _LGB_DEFAULT, description="LGB group-filter threshold (calibrated from thresholds.json)"
    )
    use_final_ml: bool = Field(True, description="Use final hybrid ML filtration")
    final_threshold: float = Field(
        _HF_DEFAULT, description="Hybrid filter threshold (calibrated from thresholds.json)"
    )


class NcbiPredictionRequest(BaseModel):
    """Request body for NCBI genome prediction"""

    accession: str = Field(..., description="NCBI accession number (e.g., NC_000913.3)")
    email: str = Field(..., description="Email address (required by NCBI)")
    use_group_ml: bool = Field(True, description="Use ML for group filtering")
    group_threshold: float = Field(
        _LGB_DEFAULT, description="LGB group-filter threshold (calibrated from thresholds.json)"
    )
    use_final_ml: bool = Field(True, description="Use final hybrid ML filtration")
    final_threshold: float = Field(
        _HF_DEFAULT, description="Hybrid filter threshold (calibrated from thresholds.json)"
    )


class GenePrediction(BaseModel):
    """Single gene prediction"""

    gene_id: str
    start: int
    end: int
    strand: str  # 'forward' or 'reverse'
    length: int
    combined_score: float
    rbs_score: Optional[float] = None


class PredictionResponse(BaseModel):
    """Response containing all predictions"""

    genome_id: str
    sequence_length: int
    total_genes: int
    predictions: List[GenePrediction]
    ml_settings: dict


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    models_loaded: dict


class ValidationRequest(BaseModel):
    """Request body for validation"""

    genome_id: str = Field(..., description="Genome ID (NCBI accession) to validate")


class ValidationResponse(BaseModel):
    """Response containing validation metrics"""

    genome_id: str
    reference_count: int
    predicted_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    sensitivity: float
    precision: float
    f1_score: float
    reference_file: str
    results_file: str


class FileInfo(BaseModel):
    """Information about a file"""

    filename: str
    path: str
    size: int
    created: float
    type: str  # 'genome' or 'result'
    can_delete: bool


class FileListResponse(BaseModel):
    """Response containing file list"""

    files: List[FileInfo]
    total_size: int


class DeleteFilesRequest(BaseModel):
    """Request to delete files"""

    paths: List[str] = Field(..., description="List of file paths to delete")


class DeleteFilesResponse(BaseModel):
    """Response after deleting files"""

    deleted: int
    failed: int
    errors: List[str]
