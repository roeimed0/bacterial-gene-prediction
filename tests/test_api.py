"""
Unit tests for api/main.py.

Uses httpx.AsyncClient with ASGITransport (httpx ≥ 0.20) to avoid the
starlette TestClient / httpx 0.28 incompatibility.  Tests are marked
asyncio and run via pytest-asyncio.

The full prediction pipeline (POST /predict, POST /predict/ncbi) is not
exercised here — those belong in integration tests.  These tests cover:
- Lightweight read-only endpoints
- Request-body validation (422 on missing required fields)
- Security constraints on file deletion
- 404 / error responses for missing resources
"""

import httpx
import pytest
import pytest_asyncio

from api.main import app

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def ac():
    """Async ASGI test client."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


class TestRoot:
    async def test_returns_200(self, ac):
        r = await ac.get("/")
        assert r.status_code == 200

    async def test_body_contains_api_name(self, ac):
        r = await ac.get("/")
        assert "Bacterial Gene Predictor" in r.json()["message"]

    async def test_body_contains_endpoints_map(self, ac):
        r = await ac.get("/")
        assert "endpoints" in r.json()


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    async def test_returns_200(self, ac):
        r = await ac.get("/health")
        assert r.status_code == 200

    async def test_status_is_healthy(self, ac):
        r = await ac.get("/health")
        assert r.json()["status"] == "healthy"

    async def test_models_loaded_key_present(self, ac):
        r = await ac.get("/health")
        assert "models_loaded" in r.json()

    async def test_models_loaded_is_dict(self, ac):
        r = await ac.get("/health")
        assert isinstance(r.json()["models_loaded"], dict)


# ---------------------------------------------------------------------------
# GET /catalog
# ---------------------------------------------------------------------------


class TestCatalog:
    async def test_returns_200(self, ac):
        r = await ac.get("/catalog")
        assert r.status_code == 200

    async def test_total_is_100(self, ac):
        r = await ac.get("/catalog")
        assert r.json()["total"] == 100

    async def test_genomes_list_has_100_items(self, ac):
        r = await ac.get("/catalog")
        assert len(r.json()["genomes"]) == 100

    async def test_each_genome_has_id_and_accession(self, ac):
        r = await ac.get("/catalog")
        for genome in r.json()["genomes"]:
            assert "id" in genome
            assert "accession" in genome


# ---------------------------------------------------------------------------
# GET /results
# ---------------------------------------------------------------------------


class TestResults:
    async def test_returns_200(self, ac):
        r = await ac.get("/results")
        assert r.status_code == 200

    async def test_body_has_results_key(self, ac):
        r = await ac.get("/results")
        assert "results" in r.json()

    async def test_results_is_a_list(self, ac):
        r = await ac.get("/results")
        assert isinstance(r.json()["results"], list)


# ---------------------------------------------------------------------------
# POST /predict — request body validation only
# ---------------------------------------------------------------------------


class TestPredictValidation:
    async def test_empty_body_returns_422(self, ac):
        r = await ac.post("/predict", json={})
        assert r.status_code == 422

    async def test_missing_sequence_returns_422(self, ac):
        r = await ac.post("/predict", json={"use_group_ml": False})
        assert r.status_code == 422

    async def test_non_json_body_returns_422(self, ac):
        r = await ac.post(
            "/predict", content=b"not json", headers={"Content-Type": "application/json"}
        )
        assert r.status_code == 422

    @pytest.mark.xfail(
        reason="issue #151: invalid nucleotides return 500 until issue #135 is fixed"
    )
    async def test_invalid_nucleotides_returns_400_issue_151(self, ac):
        # Regression for #151: a sequence with non-nucleotide characters must
        # return 400 with a descriptive message, not a generic 500.
        r = await ac.post("/predict", json={"sequence": "AAA@#$INVALID"})
        assert r.status_code == 400
        detail = r.json()["detail"].lower()
        assert "invalid" in detail or "nucleotide" in detail or "fasta" in detail


# ---------------------------------------------------------------------------
# POST /predict/ncbi — request body validation only
# ---------------------------------------------------------------------------


class TestPredictNcbiValidation:
    async def test_empty_body_returns_422(self, ac):
        r = await ac.post("/predict/ncbi", json={})
        assert r.status_code == 422

    async def test_missing_accession_returns_422(self, ac):
        r = await ac.post("/predict/ncbi", json={"use_group_ml": False})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# POST /validate — request validation and missing-file error
# ---------------------------------------------------------------------------


class TestValidate:
    async def test_empty_body_returns_422(self, ac):
        r = await ac.post("/validate", json={})
        assert r.status_code == 422

    async def test_nonexistent_genome_returns_404(self, ac):
        r = await ac.post("/validate", json={"genome_id": "NC_NONEXISTENT_FAKE"})
        assert r.status_code == 404

    async def test_empty_genome_id_returns_400_issue_174(self, ac):
        """
        Regression for issue #174: /validate accepted empty genome_id and
        returned a confusing 500 from a downstream FileNotFoundError.
        After the fix it must return 400 with an actionable message.
        """
        r = await ac.post("/validate", json={"genome_id": ""})
        assert r.status_code == 400
        assert "genome ID" in r.json()["detail"]

    async def test_genome_id_with_path_separator_returns_400_issue_174(self, ac):
        """
        Regression for issue #174: genome_id containing '/' could cause
        path-traversal; must be rejected with 400 before hitting the filesystem.
        """
        r = await ac.post("/validate", json={"genome_id": "../etc/passwd"})
        assert r.status_code == 400
        assert "genome ID" in r.json()["detail"]

    async def test_genome_id_with_spaces_returns_400_issue_174(self, ac):
        """Spaces in genome_id are not valid NCBI accession characters → 400."""
        r = await ac.post("/validate", json={"genome_id": "NC 000913.3"})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# POST /files/delete — security constraint
# ---------------------------------------------------------------------------


class TestFilesDelete:
    async def test_delete_outside_allowed_dirs_is_rejected(self, ac):
        r = await ac.post("/files/delete", json={"paths": ["src/config.py"]})
        assert r.status_code == 200
        body = r.json()
        assert body["failed"] == 1
        assert body["deleted"] == 0
        assert any("Outside allowed directories" in e for e in body["errors"])

    async def test_delete_nonexistent_results_file_counts_as_failed(self, ac):
        r = await ac.post(
            "/files/delete",
            json={"paths": ["results/NC_FAKE_NONEXISTENT_predictions.gff"]},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["failed"] == 1
        assert body["deleted"] == 0

    async def test_empty_paths_list_returns_zero_counts(self, ac):
        r = await ac.post("/files/delete", json={"paths": []})
        assert r.status_code == 200
        body = r.json()
        assert body["deleted"] == 0
        assert body["failed"] == 0

    async def test_missing_paths_key_returns_422(self, ac):
        r = await ac.post("/files/delete", json={})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# GET /files
# ---------------------------------------------------------------------------


class TestFiles:
    async def test_returns_200(self, ac):
        r = await ac.get("/files")
        assert r.status_code == 200

    async def test_body_is_dict(self, ac):
        r = await ac.get("/files")
        assert isinstance(r.json(), dict)
