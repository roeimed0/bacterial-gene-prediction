"""
API Models - Request and Response schemas
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class PredictionRequest(BaseModel):
    """Request body for gene prediction"""
    sequence: str = Field(..., description="Genome sequence in FASTA format or raw DNA sequence")
    use_group_ml: bool = Field(True, description="Use ML for group filtering")
    group_threshold: float = Field(0.1, description="ML group filtering threshold")
    use_final_ml: bool = Field(True, description="Use final hybrid ML filtration")
    final_threshold: float = Field(0.12, description="Final ML filtration threshold")


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