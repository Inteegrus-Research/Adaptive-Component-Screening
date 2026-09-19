"""Final reproducibility utilities."""
from __future__ import annotations
import hashlib, json, platform, random, subprocess, sys
from pathlib import Path
from typing import Any
import numpy as np

PROJECT_ROOT=Path(__file__).resolve().parents[1]

def seed_everything(seed:int):
    random.seed(seed); np.random.seed(seed)

def file_sha256(path:Path)->str:
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def config_sha256(path:Path)->str:
    return file_sha256(path)

def git_revision()->str:
    try: return subprocess.check_output(["git","rev-parse","HEAD"],cwd=PROJECT_ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:return "UNAVAILABLE"

def build_run_manifest(seed:int,config_path:Path,dataset_path:Path,artifact_versions:dict[str,Any]|None=None)->dict[str,Any]:
    return {"seed":seed,"config_sha256":file_sha256(config_path) if config_path.exists() else None,"dataset_sha256":file_sha256(dataset_path) if dataset_path.exists() else None,"code_revision":git_revision(),"python_version":platform.python_version(),"platform":platform.platform(),"numpy":np.__version__,"artifact_versions":artifact_versions or {}}
