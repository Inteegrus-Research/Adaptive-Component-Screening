"""FastAPI application boundary for Adaptive Component Screening.

Scientific logic remains under ``src/``. The API layer exposes stable contracts
for the React frontend and a deterministic demo-results mode for presentation.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.schemas import BatchSummary, ContractResponse, ScreenResponse, SystemStatus
from app.services.results import ResultsRepository
from src.pipeline import PipelineError, screen_file

API_VERSION = "1.0.0"

app = FastAPI(
    title="Adaptive Component Screening",
    version=API_VERSION,
    description="Operational API for component burn-in screening, forecasting, safety gating, and engineering explanation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _repo() -> ResultsRepository:
    return ResultsRepository.from_environment()


def _run_payload(run, report_dir: Path) -> ScreenResponse:
    # Persist the complete in-memory run to the same temporary result shape used
    # by demo mode, then use the shared assembler. This keeps upload-time and
    # persisted-result semantics identical.
    report_dir.mkdir(parents=True, exist_ok=True)
    run.screening.to_csv(report_dir / "screening.csv", index=False)
    run.explanations.to_csv(report_dir / "explanations.csv", index=False)
    run.canonical.to_csv(report_dir / "canonical.csv", index=False)
    run.forecast.to_csv(report_dir / "forecast.csv", index=False)
    run.anomaly.to_csv(report_dir / "anomaly.csv", index=False)
    repo = ResultsRepository(report_dir)
    components = repo.components()
    intelligence = [repo.intelligence(c.part_id) for c in components]
    return ScreenResponse(
        manifest=run.manifest,
        summary=repo.summary(),
        decisions=json.loads(run.screening.to_json(orient="records")),
        explanations=json.loads(run.explanations.to_json(orient="records")),
        components=components,
        component_intelligence=intelligence,
    )


@app.get("/health", response_model=SystemStatus)
def health() -> SystemStatus:
    repo = _repo()
    return SystemStatus(
        status="ok",
        pipeline="READY",
        models="AVAILABLE" if repo.available() else "MODEL_ARTIFACTS_REQUIRED",
        demo_mode=repo.report_dir.name == "demo_screen",
    )


@app.get("/api/health", response_model=SystemStatus)
def api_health() -> SystemStatus:
    return health()


@app.get("/api/contract", response_model=ContractResponse)
def contract() -> ContractResponse:
    return ContractResponse(
        api_version=API_VERSION,
        resources={
            "health": "GET /api/health",
            "summary": "GET /api/summary",
            "components": "GET /api/components",
            "component": "GET /api/components/{component_id}",
            "explanation": "GET /api/components/{component_id}/explanation",
            "screen": "POST /api/screen",
            "benchmark": "GET /api/audit/benchmark",
        },
    )


@app.get("/api/summary", response_model=BatchSummary)
def summary() -> BatchSummary:
    repo = _repo()
    if not repo.available():
        raise HTTPException(status_code=404, detail=f"No persisted screening report found at {repo.report_dir}")
    return repo.summary()


@app.get("/api/components")
def components(
    decision: str | None = Query(default=None),
    search: str | None = Query(default=None),
    limit: int = Query(default=250, ge=1, le=5000),
) -> dict:
    repo = _repo()
    if not repo.available():
        raise HTTPException(status_code=404, detail=f"No persisted screening report found at {repo.report_dir}")
    rows = repo.components()
    if decision:
        wanted = decision.upper()
        rows = [r for r in rows if r.decision == wanted]
    if search:
        needle = search.lower()
        rows = [r for r in rows if needle in r.part_id.lower() or needle in (r.component_family or "").lower() or needle in (r.parameter or "").lower()]
    return {"items": [r.model_dump() for r in rows[:limit]], "count": len(rows)}


@app.get("/api/components/{component_id}")
def component(component_id: str) -> dict:
    repo = _repo()
    if not repo.available():
        raise HTTPException(status_code=404, detail=f"No persisted screening report found at {repo.report_dir}")
    try:
        return repo.intelligence(component_id).model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Component '{component_id}' not found") from exc


@app.get("/api/components/{component_id}/explanation")
def component_explanation(component_id: str) -> dict:
    repo = _repo()
    try:
        payload = repo.intelligence(component_id).explanation
        if not payload:
            return {"part_id": component_id, "available": False}
        return {"part_id": component_id, "available": True, "explanation": payload}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Component '{component_id}' not found") from exc


@app.get("/api/components/{component_id}/progressive")
def component_progressive(component_id: str) -> dict:
    """Expose persisted progressive screening decisions when benchmark artifacts exist.

    This endpoint does not recompute any scientific logic; it only reads the existing
    origin_* screening artifacts generated by the benchmark pipeline.
    """
    repo = _repo()
    roots = [repo.report_dir.parent / "benchmark_kaggle", Path("reports/benchmark_kaggle")]
    points = []
    import pandas as pd
    for root in roots:
        if not root.exists():
            continue
        for origin_dir in sorted(root.glob("seed_*/progressive_metrics.csv")):
            pass
        for screen_path in root.glob("seed_*/origin_*h/screening.csv"):
            name = screen_path.parent.name
            try:
                origin = float(name.replace("origin_", "").replace("h", ""))
            except ValueError:
                continue
            try:
                d = pd.read_csv(screen_path, low_memory=False)
            except Exception:
                continue
            rows = d[d.get("part_id", pd.Series(dtype=str)).astype(str).eq(str(component_id))] if "part_id" in d.columns else d.iloc[0:0]
            if rows.empty:
                continue
            row = rows.iloc[0]
            points.append({"origin_h":origin,"decision":str(row.get("decision","UNKNOWN")),"risk_score":None if pd.isna(row.get("risk_score")) else float(row.get("risk_score"))})
        if points:
            break
    unique = {}
    for x in sorted(points, key=lambda z:(z["origin_h"], z["decision"])):
        unique[x["origin_h"]]=x
    return {"part_id":component_id,"items":list(unique.values()),"available":bool(unique)}


@app.get("/api/audit/benchmark")
def audit_benchmark() -> dict:
    root = _repo().report_dir
    candidates = [root / "benchmark_kaggle", root.parent / "benchmark_kaggle", Path("reports/benchmark_kaggle")]
    for directory in candidates:
        if directory.exists():
            result: dict[str, object] = {"available": True, "source": str(directory), "files": {}}
            for name in [
                "system_metrics_mean_std.csv",
                "lead_time_mean_std.csv",
                "latent_escape_mean_std.csv",
                "mechanism_metrics_mean.csv",
                "forecast_metrics_mean.csv",
            ]:
                path = directory / name
                if path.exists():
                    result["files"][name] = json.loads(pd.read_csv(path).to_json(orient="records"))
            return result
    return {"available": False, "message": "Benchmark artifacts are optional in Phase 1 and will be connected in the audit screen phase."}


@app.post("/screen")
@app.post("/api/screen", response_model=ScreenResponse)
async def screen(
    file: UploadFile = File(...),
    as_of_h: float = Query(24.0, ge=0.0),
    target_horizon: float = Query(168.0, ge=0.0),
) -> ScreenResponse:
    suffix = Path(file.filename or "input.csv").suffix or ".csv"
    content = await file.read()
    if not content:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    with tempfile.TemporaryDirectory(prefix="screen_api_") as td:
        root = Path(td)
        input_path = root / f"input{suffix}"
        output_dir = root / "output"
        input_path.write_bytes(content)
        try:
            run = screen_file(
                input_path,
                output_dir,
                as_of_h=as_of_h,
                target_horizon=target_horizon,
                auto_train_missing=False,
            )
        except (PipelineError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _run_payload(run, output_dir)


# Optional production-style serving: after `npm run build`, FastAPI can serve the
# compiled frontend from the same process. Development still uses Vite on :5173.
_FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if _FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
