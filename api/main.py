"""
FastAPI wrapper for Bacterial Gene Prediction
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import tempfile
import sys
from typing import List, Dict
import traceback

# Add parent directory to path to import from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.models import (
    PredictionRequest,
    NcbiPredictionRequest,
    PredictionResponse, 
    GenePrediction,
    HealthResponse
)

# Create FastAPI app
app = FastAPI(
    title="Bacterial Gene Predictor API",
    description="Hybrid gene prediction combining traditional bioinformatics with ML",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def check_models_exist() -> dict:
    """Check if ML models are available"""
    models_dir = project_root / 'models'
    
    return {
        "group_classifier": (models_dir / 'orf_classifier_lgb.pkl').exists(),
        "hybrid_filter": (models_dir / 'hybrid_best_model.pkl').exists()
    }


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Bacterial Gene Predictor API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "predict_file": "/predict/file (POST)",
            "predict_ncbi": "/predict/ncbi (POST)"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and model availability"""
    models_status = check_models_exist()
    
    return HealthResponse(
        status="healthy",
        models_loaded=models_status
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_genes(request: PredictionRequest):
    """
    Predict genes from a genome sequence
    
    Accepts raw DNA sequence or FASTA format string
    """
    try:
        # Use a meaningful filename instead of random temp name
        filename = request.filename if request.filename else "pasted_sequence"
        
        # Create temp directory for our files
        temp_dir = Path(tempfile.gettempdir()) / 'gene_predictions'
        temp_dir.mkdir(exist_ok=True)
        tmp_file_path = temp_dir / f"{filename}.fasta"
        
        # Write sequence to file with proper name
        with open(tmp_file_path, 'w') as f:
            if not request.sequence.strip().startswith('>'):
                f.write(f">{filename}\n")
            f.write(request.sequence)
        
        # Import prediction function
        from hybrid_predictor import predict_fasta_file
        
        # Run prediction - backend will use filename automatically
        predictions = predict_fasta_file(
            fasta_path=str(tmp_file_path),
            use_ml=request.use_group_ml,
            ml_threshold=request.group_threshold,
            use_final_filtration_ml=request.use_final_ml,
            final_ml_threshold=request.final_threshold
        )
        
        # Clean up temp file
        tmp_file_path.unlink(missing_ok=True)
        
        # Convert predictions to response format
        gene_predictions = []
        for i, pred in enumerate(predictions, 1):
            gene_predictions.append(GenePrediction(
                gene_id=f"gene_{i}",
                start=pred.get('genome_start', pred.get('start')),
                end=pred.get('genome_end', pred.get('end')),
                strand='forward' if pred.get('strand') == 'forward' else 'reverse',
                length=pred.get('genome_end', pred.get('end')) - pred.get('genome_start', pred.get('start')) + 1,
                combined_score=pred.get('combined_score', 0.0),
                rbs_score=pred.get('rbs_score')
            ))
        
        # Calculate sequence length
        sequence = request.sequence.replace('>', '').replace('\n', '').replace(' ', '')
        sequence = ''.join(c for c in sequence if c in 'ATGCatgc')
        
        return PredictionResponse(
            genome_id=filename,
            sequence_length=len(sequence),
            total_genes=len(gene_predictions),
            predictions=gene_predictions,
            ml_settings={
                "group_ml_enabled": request.use_group_ml,
                "group_threshold": request.group_threshold,
                "final_ml_enabled": request.use_final_ml,
                "final_threshold": request.final_threshold
            }
        )
        
    except Exception as e:
        # Clean up temp file on error
        try:
            tmp_file_path.unlink(missing_ok=True)
        except:
            pass
        
        print(f"Prediction error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/file", response_model=PredictionResponse, tags=["Prediction"])
async def predict_genes_from_file(
    file: UploadFile = File(...),
    use_group_ml: bool = True,
    group_threshold: float = 0.1,
    use_final_ml: bool = True,
    final_threshold: float = 0.12
):
    """
    Predict genes from an uploaded FASTA file
    """
    # Validate file type
    if not file.filename.endswith(('.fasta', '.fa', '.fna')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Must be .fasta, .fa, or .fna"
        )
    
    try:
        # Save with original filename in temp directory
        temp_dir = Path(tempfile.gettempdir()) / 'gene_predictions'
        temp_dir.mkdir(exist_ok=True)
        tmp_file_path = temp_dir / file.filename
        
        # Write uploaded content
        with open(tmp_file_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        # Import prediction function
        from hybrid_predictor import predict_fasta_file
        
        # Run prediction - backend will use filename automatically
        predictions = predict_fasta_file(
            fasta_path=str(tmp_file_path),
            use_ml=use_group_ml,
            ml_threshold=group_threshold,
            use_final_filtration_ml=use_final_ml,
            final_ml_threshold=final_threshold
        )
        
        # Clean up temp file
        tmp_file_path.unlink(missing_ok=True)
        
        # Convert predictions to response format
        gene_predictions = []
        for i, pred in enumerate(predictions, 1):
            gene_predictions.append(GenePrediction(
                gene_id=f"gene_{i}",
                start=pred.get('genome_start', pred.get('start')),
                end=pred.get('genome_end', pred.get('end')),
                strand='forward' if pred.get('strand') == 'forward' else 'reverse',
                length=pred.get('genome_end', pred.get('end')) - pred.get('genome_start', pred.get('start')) + 1,
                combined_score=pred.get('combined_score', 0.0),
                rbs_score=pred.get('rbs_score')
            ))
        
        # Get sequence length from predictions
        sequence_length = max(p.end for p in gene_predictions) if gene_predictions else 0
        
        # Extract genome ID from filename (without extension)
        genome_id = Path(file.filename).stem
        
        return PredictionResponse(
            genome_id=genome_id,
            sequence_length=sequence_length,
            total_genes=len(gene_predictions),
            predictions=gene_predictions,
            ml_settings={
                "group_ml_enabled": use_group_ml,
                "group_threshold": group_threshold,
                "final_ml_enabled": use_final_ml,
                "final_threshold": final_threshold
            }
        )
        
    except Exception as e:
        # Clean up temp file on error
        try:
            tmp_file_path.unlink(missing_ok=True)
        except:
            pass
        
        print(f"Prediction error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/ncbi", response_model=PredictionResponse, tags=["Prediction"])
async def predict_ncbi(request: NcbiPredictionRequest):
    """
    Download genome from NCBI and predict genes
    
    Requires NCBI accession number and email address
    """
    try:
        # Import prediction function
        from hybrid_predictor import predict_ncbi_genome
        
        # Run NCBI download and prediction
        predictions = predict_ncbi_genome(
            accession=request.accession,
            email=request.email,
            use_ml=request.use_group_ml,
            ml_threshold=request.group_threshold,
            use_final_filtration_ml=request.use_final_ml,
            final_ml_threshold=request.final_threshold
        )
        
        # Convert predictions to response format
        gene_predictions = []
        for i, pred in enumerate(predictions, 1):
            gene_predictions.append(GenePrediction(
                gene_id=f"gene_{i}",
                start=pred.get('genome_start', pred.get('start')),
                end=pred.get('genome_end', pred.get('end')),
                strand='forward' if pred.get('strand') == 'forward' else 'reverse',
                length=pred.get('genome_end', pred.get('end')) - pred.get('genome_start', pred.get('start')) + 1,
                combined_score=pred.get('combined_score', 0.0),
                rbs_score=pred.get('rbs_score')
            ))
        
        # Get sequence length from predictions
        sequence_length = max(p.end for p in gene_predictions) if gene_predictions else 0
        
        return PredictionResponse(
            genome_id=request.accession,
            sequence_length=sequence_length,
            total_genes=len(gene_predictions),
            predictions=gene_predictions,
            ml_settings={
                "group_ml_enabled": request.use_group_ml,
                "group_threshold": request.group_threshold,
                "final_ml_enabled": request.use_final_ml,
                "final_threshold": request.final_threshold
            }
        )
        
    except Exception as e:
        print(f"NCBI prediction error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"NCBI prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)