"""Thin application layer.

No scientific logic lives here. It only exposes the operational pipeline.
FastAPI is deliberately kept as a dependency of the application boundary, not
of the scientific modules.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from src.pipeline import PipelineError, screen_file

app = FastAPI(title="Adaptive Component Screening", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/screen")
async def screen(
    file: UploadFile = File(...),
    as_of_h: float = Query(24.0, ge=0.0),
) -> dict:
    suffix = Path(file.filename or "input.csv").suffix or ".csv"
    with tempfile.TemporaryDirectory(prefix="screen_api_") as td:
        root = Path(td)
        input_path = root / f"input{suffix}"
        output_dir = root / "output"
        content = await file.read()
        input_path.write_bytes(content)
        try:
            run = screen_file(input_path, output_dir, as_of_h=as_of_h, auto_train_missing=False)
        except (PipelineError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return {
            "manifest": run.manifest,
            "decisions": json.loads(run.screening.to_json(orient="records")),
            "explanations": json.loads(run.explanations.to_json(orient="records")),
        }
