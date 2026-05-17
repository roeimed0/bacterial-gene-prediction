"""
FastAPI wrapper for Bacterial Gene Prediction
"""

import os
import re
import sys
import tempfile
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path to import from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline import load_models, predict_genome_from_file  # noqa: E402

# Module-level model cache — populated once at startup by the lifespan handler
_api_lgb = None
_api_hf = None

from api.models import (  # noqa: E402
    DeleteFilesRequest,
    DeleteFilesResponse,
    FileInfo,
    FileListResponse,
    GenePrediction,
    HealthResponse,
    NcbiPredictionRequest,
    PredictionRequest,
    PredictionResponse,
    ValidationRequest,
    ValidationResponse,
)

# Accepts only safe NCBI-style accession characters; blocks path traversal.
_VALID_GENOME_ID = re.compile(r"^[A-Za-z0-9._-]+$")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload ML models once at startup so prediction endpoints pay no load cost."""
    global _api_lgb, _api_hf
    _api_lgb, _api_hf = load_models()
    yield


# Create FastAPI app
app = FastAPI(
    title="Bacterial Gene Predictor API",
    description="Hybrid gene prediction combining traditional bioinformatics with ML",
    version="1.2.0",
    lifespan=lifespan,
)

# Enable CORS — override origin via CORS_ORIGIN env var for non-dev deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("CORS_ORIGIN", "http://localhost:5173")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def check_models_exist() -> dict:
    """Check if ML models are available"""
    models_dir = project_root / "models"

    return {
        "group_classifier": (models_dir / "orf_classifier_lgb.pkl").exists(),
        "hybrid_filter": (models_dir / "hybrid_best_model.pkl").exists(),
    }


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Bacterial Gene Predictor API",
        "version": "1.2.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "predict_file": "/predict/file (POST)",
            "predict_ncbi": "/predict/ncbi (POST)",
            "catalog": "/catalog (GET)",
            "results": "/results (GET)",
            "validate": "/validate (POST)",
            "files": "/files (GET)",
            "delete": "/files/delete (POST)",
            "cleanup": "/files/cleanup (POST)",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and model availability"""
    models_status = check_models_exist()

    return HealthResponse(status="healthy", models_loaded=models_status)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_genes(request: PredictionRequest):
    """
    Predict genes from a genome sequence

    Accepts raw DNA sequence or FASTA format string
    """
    if not request.sequence or not request.sequence.strip():
        raise HTTPException(
            status_code=400,
            detail="Sequence is empty. Provide a raw DNA sequence or a FASTA-formatted string.",
        )

    # Strip FASTA header lines and whitespace, then validate nucleotide alphabet.
    _seq_body = (
        "".join(line for line in request.sequence.splitlines() if not line.startswith(">"))
        .replace(" ", "")
        .replace("\t", "")
    )
    _invalid = set(_seq_body.upper()) - set("ACGTNRYWSKMBDHV")
    if _invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid nucleotide characters in sequence: {sorted(_invalid)}. "
            "Only IUPAC nucleotide characters (A/C/G/T/N and ambiguity codes) are accepted.",
        )

    tmp_file_path = None
    try:
        filename = request.filename if request.filename else "pasted_sequence"

        temp_dir = Path(tempfile.gettempdir()) / "gene_predictions"
        temp_dir.mkdir(exist_ok=True)
        tmp_file_path = temp_dir / f"{filename}.fasta"

        with open(tmp_file_path, "w") as f:
            if not request.sequence.strip().startswith(">"):
                f.write(f">{filename}\n")
            f.write(request.sequence)

        raw = predict_genome_from_file(
            fasta_path=str(tmp_file_path),
            genome_id=filename,
            lgb=_api_lgb if request.use_group_ml else None,
            lgb_threshold=request.group_threshold,
            hf=_api_hf if request.use_final_ml else None,
            hf_threshold=request.final_threshold,
        )
        tmp_file_path.unlink(missing_ok=True)

        predictions = raw.to_dict("records") if hasattr(raw, "to_dict") else list(raw)
        gene_predictions = [
            GenePrediction(
                gene_id=f"gene_{i}",
                start=p.get("genome_start", p.get("start")),
                end=p.get("genome_end", p.get("end")),
                strand="forward" if p.get("strand") == "forward" else "reverse",
                length=p.get("genome_end", p.get("end"))
                - p.get("genome_start", p.get("start"))
                + 1,
                combined_score=p.get("combined_score", 0.0),
                rbs_score=p.get("rbs_score"),
            )
            for i, p in enumerate(predictions, 1)
        ]

        sequence = "".join(c for c in request.sequence if c in "ATGCatgcNn")
        return PredictionResponse(
            genome_id=filename,
            sequence_length=len(sequence),
            total_genes=len(gene_predictions),
            predictions=gene_predictions,
            ml_settings={
                "group_ml_enabled": request.use_group_ml,
                "group_threshold": request.group_threshold,
                "final_ml_enabled": request.use_final_ml,
                "final_threshold": request.final_threshold,
            },
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        if tmp_file_path:
            tmp_file_path.unlink(missing_ok=True)
        print(f"Prediction error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/file", response_model=PredictionResponse, tags=["Prediction"])
async def predict_genes_from_file(
    file: UploadFile = File(...),
    use_group_ml: bool = True,
    group_threshold: float = 0.07,
    use_final_ml: bool = True,
    final_threshold: float = 0.25,
):
    """
    Predict genes from an uploaded FASTA file
    """
    if not file.filename.endswith((".fasta", ".fa", ".fna")):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Must be .fasta, .fa, or .fna"
        )

    tmp_file_path = None
    try:
        temp_dir = Path(tempfile.gettempdir()) / "gene_predictions"
        temp_dir.mkdir(exist_ok=True)
        tmp_file_path = temp_dir / file.filename

        with open(tmp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        genome_id = Path(file.filename).stem
        raw = predict_genome_from_file(
            fasta_path=str(tmp_file_path),
            genome_id=genome_id,
            lgb=_api_lgb if use_group_ml else None,
            lgb_threshold=group_threshold,
            hf=_api_hf if use_final_ml else None,
            hf_threshold=final_threshold,
        )
        tmp_file_path.unlink(missing_ok=True)

        predictions = raw.to_dict("records") if hasattr(raw, "to_dict") else list(raw)
        gene_predictions = [
            GenePrediction(
                gene_id=f"gene_{i}",
                start=p.get("genome_start", p.get("start")),
                end=p.get("genome_end", p.get("end")),
                strand="forward" if p.get("strand") == "forward" else "reverse",
                length=p.get("genome_end", p.get("end"))
                - p.get("genome_start", p.get("start"))
                + 1,
                combined_score=p.get("combined_score", 0.0),
                rbs_score=p.get("rbs_score"),
            )
            for i, p in enumerate(predictions, 1)
        ]

        sequence_length = max(p.end for p in gene_predictions) if gene_predictions else 0
        return PredictionResponse(
            genome_id=genome_id,
            sequence_length=sequence_length,
            total_genes=len(gene_predictions),
            predictions=gene_predictions,
            ml_settings={
                "group_ml_enabled": use_group_ml,
                "group_threshold": group_threshold,
                "final_ml_enabled": use_final_ml,
                "final_threshold": final_threshold,
            },
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        if tmp_file_path:
            tmp_file_path.unlink(missing_ok=True)
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
        from hybrid_predictor import predict_ncbi_genome

        predictions = predict_ncbi_genome(
            accession=request.accession,
            email=request.email,
            use_ml=request.use_group_ml,
            ml_threshold=request.group_threshold,
            use_final_filtration_ml=request.use_final_ml,
            final_ml_threshold=request.final_threshold,
        )

        gene_predictions = []
        for i, pred in enumerate(predictions, 1):
            gene_predictions.append(
                GenePrediction(
                    gene_id=f"gene_{i}",
                    start=pred.get("genome_start", pred.get("start")),
                    end=pred.get("genome_end", pred.get("end")),
                    strand="forward" if pred.get("strand") == "forward" else "reverse",
                    length=pred.get("genome_end", pred.get("end"))
                    - pred.get("genome_start", pred.get("start"))
                    + 1,
                    combined_score=pred.get("combined_score", 0.0),
                    rbs_score=pred.get("rbs_score"),
                )
            )

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
                "final_threshold": request.final_threshold,
            },
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (OSError, IOError) as e:
        raise HTTPException(status_code=502, detail=f"NCBI download failed: {str(e)}")
    except Exception as e:
        print(f"NCBI prediction error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"NCBI prediction failed: {str(e)}")


@app.get("/catalog", tags=["Catalog"])
async def get_genome_catalog():
    """
    Get the list of 100 well-studied genomes from the catalog
    """
    try:
        from src import config

        return {"total": len(config.GENOME_CATALOG), "genomes": config.GENOME_CATALOG}

    except Exception as e:
        print(f"Catalog error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to load catalog: {str(e)}")


@app.get("/results", tags=["Results"])
async def get_results():
    """
    Get list of available prediction results for validation
    """
    try:
        results_dir = project_root / "results"

        if not results_dir.exists():
            return {"results": []}

        all_gff = sorted(results_dir.glob("*_predictions.gff"))
        results = []

        import re

        for file in all_gff:
            genome_id = file.stem.replace("_predictions", "")
            is_ncbi = bool(re.match(r"^[A-Z]{2}_\d{6,}\.\d+$", genome_id))

            results.append(
                {
                    "genome_id": genome_id,
                    "filename": file.name,
                    "size": file.stat().st_size,
                    "created": file.stat().st_mtime,
                    "can_validate": is_ncbi,
                }
            )

        return {"results": results}

    except Exception as e:
        print(f"Results listing error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to list results: {str(e)}")


@app.post("/validate", response_model=ValidationResponse, tags=["Validation"])
async def validate_predictions(request: ValidationRequest):
    """
    Validate predictions against reference annotations

    Only works for NCBI genomes with available reference annotations
    """
    if not request.genome_id or not _VALID_GENOME_ID.match(request.genome_id):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid genome ID '{request.genome_id}'. "
                "Expected an NCBI accession such as NC_000913.3 "
                "(letters, digits, underscores, hyphens, and dots only)."
            ),
        )

    try:
        from src.validation import validate_from_results_directory

        metrics = validate_from_results_directory(request.genome_id)

        # Save validation report
        results_dir = project_root / "results"
        report_path = results_dir / f"{request.genome_id}_validation_report.txt"

        if report_path.exists():
            version = 2
            while True:
                versioned_path = (
                    results_dir / f"{request.genome_id}_validation_report_v{version}.txt"
                )
                if not versioned_path.exists():
                    report_path = versioned_path
                    break
                version += 1

        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("VALIDATION REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Genome ID:   {request.genome_id}\n")
            f.write(f"Prediction:  {metrics['results_file']}\n")
            f.write(f"Reference:   {metrics['reference_file']}\n")
            f.write("\n")
            f.write(f"Reference genes:       {metrics['reference_count']:,}\n")
            f.write(f"Predicted genes:       {metrics['predicted_count']:,}\n")
            f.write("\n")
            f.write(f"True Positives (TP):   {metrics['true_positives']:,}\n")
            f.write(f"False Positives (FP):  {metrics['false_positives']:,}\n")
            f.write(f"False Negatives (FN):  {metrics['false_negatives']:,}\n")
            f.write("\n")
            f.write(f"Sensitivity (Recall):  {metrics['sensitivity']:.4f}\n")
            f.write(f"Precision:             {metrics['precision']:.4f}\n")
            f.write(f"F1 Score:              {metrics['f1_score']:.4f}\n")
            f.write("=" * 80 + "\n")

        print(f"[+] Validation report saved to: {report_path}")

        return ValidationResponse(
            genome_id=request.genome_id,
            reference_count=metrics["reference_count"],
            predicted_count=metrics["predicted_count"],
            true_positives=metrics["true_positives"],
            false_positives=metrics["false_positives"],
            false_negatives=metrics["false_negatives"],
            sensitivity=metrics["sensitivity"],
            precision=metrics["precision"],
            f1_score=metrics["f1_score"],
            reference_file=metrics["reference_file"],
            results_file=metrics["results_file"],
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")
    except Exception as e:
        print(f"Validation error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@app.get("/files", response_model=FileListResponse, tags=["Files"])
async def list_files():
    """
    List all downloaded genomes and prediction results
    """
    try:
        files = []
        total_size = 0

        # List FASTA genomes from data/full_dataset/
        genomes_dir = project_root / "data" / "full_dataset"
        if genomes_dir.exists():
            for file in genomes_dir.glob("*.fasta"):
                stat = file.stat()
                files.append(
                    FileInfo(
                        filename=file.name,
                        path=str(file.relative_to(project_root)),
                        size=stat.st_size,
                        created=stat.st_mtime,
                        type="genome",
                        can_delete=True,
                    )
                )
                total_size += stat.st_size

            # List GFF reference annotations from data/full_dataset/
            for file in genomes_dir.glob("*.gff"):
                stat = file.stat()
                files.append(
                    FileInfo(
                        filename=file.name,
                        path=str(file.relative_to(project_root)),
                        size=stat.st_size,
                        created=stat.st_mtime,
                        type="reference",
                        can_delete=True,
                    )
                )
                total_size += stat.st_size

        # List GFF prediction results from results/
        results_dir = project_root / "results"
        if results_dir.exists():
            for file in results_dir.glob("*_predictions.gff"):
                stat = file.stat()
                files.append(
                    FileInfo(
                        filename=file.name,
                        path=str(file.relative_to(project_root)),
                        size=stat.st_size,
                        created=stat.st_mtime,
                        type="result",
                        can_delete=True,
                    )
                )
                total_size += stat.st_size

            # List validation reports from results/
            for file in results_dir.glob("*_validation_report*.txt"):
                stat = file.stat()
                files.append(
                    FileInfo(
                        filename=file.name,
                        path=str(file.relative_to(project_root)),
                        size=stat.st_size,
                        created=stat.st_mtime,
                        type="report",
                        can_delete=True,
                    )
                )
                total_size += stat.st_size

        files.sort(key=lambda x: x.created, reverse=True)

        return FileListResponse(files=files, total_size=total_size)

    except Exception as e:
        print(f"File listing error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")


@app.post("/files/delete", response_model=DeleteFilesResponse, tags=["Files"])
async def delete_files(request: DeleteFilesRequest):
    """
    Delete specified files
    """
    try:
        deleted = 0
        failed = 0
        errors = []

        for path_str in request.paths:
            try:
                file_path = project_root / path_str

                # Security check: only allow deleting from data/ and results/
                if not (
                    str(file_path).startswith(str(project_root / "data"))
                    or str(file_path).startswith(str(project_root / "results"))
                ):
                    errors.append(f"Cannot delete {path_str}: Outside allowed directories")
                    failed += 1
                    continue

                if file_path.exists():
                    file_path.unlink()
                    deleted += 1
                else:
                    errors.append(f"File not found: {path_str}")
                    failed += 1

            except Exception as e:
                errors.append(f"Failed to delete {path_str}: {str(e)}")
                failed += 1

        return DeleteFilesResponse(deleted=deleted, failed=failed, errors=errors)

    except Exception as e:
        print(f"File deletion error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to delete files: {str(e)}")


@app.post("/files/cleanup", response_model=DeleteFilesResponse, tags=["Files"])
async def cleanup_all_files():
    """
    Delete all downloaded genomes and results
    """
    try:
        deleted = 0
        failed = 0
        errors = []

        # Delete all genomes and reference annotations
        genomes_dir = project_root / "data" / "full_dataset"
        if genomes_dir.exists():
            # Delete FASTA files
            for file in genomes_dir.glob("*.fasta"):
                try:
                    file.unlink()
                    deleted += 1
                except Exception as e:
                    errors.append(f"Failed to delete {file.name}: {str(e)}")
                    failed += 1

            # Delete GFF reference files
            for file in genomes_dir.glob("*.gff"):
                try:
                    file.unlink()
                    deleted += 1
                except Exception as e:
                    errors.append(f"Failed to delete {file.name}: {str(e)}")
                    failed += 1

        # Delete all results
        results_dir = project_root / "results"
        if results_dir.exists():
            for file in results_dir.glob("*"):
                if file.is_file():
                    try:
                        file.unlink()
                        deleted += 1
                    except Exception as e:
                        errors.append(f"Failed to delete {file.name}: {str(e)}")
                        failed += 1

        return DeleteFilesResponse(deleted=deleted, failed=failed, errors=errors)

    except Exception as e:
        print(f"Cleanup error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
