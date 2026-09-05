#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
EXPECTED_PYTHON_MINOR="3.11"; EXPECTED_SKLEARN="1.9.0"; EXPECTED_NUMPY="2.3.5"; EXPECTED_PANDAS="2.2.3"; EXPECTED_SCIPY="1.17.0"; EXPECTED_JOBLIB="1.5.3"; EXPECTED_PYYAML="6.0.3"
python_minor="$(python -c 'import platform; print(".".join(platform.python_version().split(".")[:2]))')"
[[ "$python_minor" == "$EXPECTED_PYTHON_MINOR" ]] || { echo "ERROR: Python $EXPECTED_PYTHON_MINOR.x required; found $(python --version)" >&2; exit 2; }
python - <<PY
import sys, sklearn, numpy, pandas, scipy, joblib, yaml
expected={"sklearn":"$EXPECTED_SKLEARN","numpy":"$EXPECTED_NUMPY","pandas":"$EXPECTED_PANDAS","scipy":"$EXPECTED_SCIPY","joblib":"$EXPECTED_JOBLIB","yaml":"$EXPECTED_PYYAML"}
actual={"sklearn":sklearn.__version__,"numpy":numpy.__version__,"pandas":pandas.__version__,"scipy":scipy.__version__,"joblib":joblib.__version__,"yaml":yaml.__version__}
errors=[f"{k}: expected {expected[k]}, found {actual[k]}" for k in expected if actual[k]!=expected[k]]
if errors: print('ERROR: runtime lock mismatch\n'+'\n'.join(errors),file=sys.stderr); sys.exit(2)
print('Runtime lock verified:')
for k,v in actual.items(): print(f'  {k}={v}')
PY
mkdir -p models/anomaly models/forecast models/calibration reports
backup="models/backup_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$backup"
cp -a models/anomaly/. "$backup/" 2>/dev/null || true
cp -a models/forecast/. "$backup/" 2>/dev/null || true
cp -a models/calibration/. "$backup/" 2>/dev/null || true
rm -f models/calibration/safety_policy.json
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
python -m src.anomaly train data/processed/module_A_dataset.csv "$TMP/anomaly.joblib" --as-of 24
# One universal model is trained over all target horizons used by the generator.
python -m src.forecast train data/processed/module_B_drift_full.csv "$TMP/forecast.joblib" --horizon 168 --as-of 24 --target-horizons 168
export SIH_OOD_TMP="$TMP"
python - <<'PY'
from pathlib import Path
import os
import pandas as pd
src=Path('data/processed/module_A_dataset.csv')
df=pd.read_csv(src, low_memory=False)
if 'split' not in df.columns:
    raise SystemExit('module_A_dataset.csv must contain train/val/test split labels for leakage-safe OOD calibration')
train=df[df['split'].astype(str).eq('train')].copy()
out=Path(os.environ['SIH_OOD_TMP'])/'ood_train_reference.csv'
train.to_csv(out,index=False)
print(f'OOD TRAIN-ONLY reference: rows={len(train)} lots={train.lot_id.nunique() if "lot_id" in train.columns else "NA"}')
PY
python -m src.safety fit-ood "$TMP/ood_train_reference.csv" "$TMP/ood_profile.joblib"
mv "$TMP/anomaly.joblib" models/anomaly/model.joblib
mv "$TMP/anomaly.joblib.manifest.json" models/anomaly/model.joblib.manifest.json
mv "$TMP/forecast.joblib" models/forecast/model.joblib
mv "$TMP/forecast.joblib.manifest.json" models/forecast/model.joblib.manifest.json
mv "$TMP/ood_profile.joblib" models/calibration/ood_profile.joblib
mv "$TMP/ood_profile.joblib.manifest.json" models/calibration/ood_profile.joblib.manifest.json
# Validation-only safety threshold calibration. Test lots never enter calibration.
python - <<'PY'
from pathlib import Path
import pandas as pd
from src.pipeline import screen_dataframe
from src.evaluation import calibrate_safety_policy
from src.utils import PROJECT_ROOT
src=PROJECT_ROOT/'data'/'processed'/'module_A_dataset.csv'
d=pd.read_csv(src)
if 'split' not in d.columns:
    raise SystemExit('module_A_dataset.csv must contain train/val/test lot splits')
val=d[d['split'].astype(str).eq('val')].copy()
run=screen_dataframe(val, PROJECT_ROOT/'reports'/'calibration_screen', as_of_h=24.0, target_horizon=168.0, auto_train_missing=False)
s=run.screening.copy()
if 'future_defective_168h' in val.columns:
    labels=val[['part_id','future_defective_168h']].drop_duplicates('part_id').copy()
    labels['future_defective']=pd.to_numeric(labels['future_defective_168h'],errors='coerce').fillna(0).astype(int)
elif 'defect_state' in val.columns:
    labels=val[['part_id','defect_state']].drop_duplicates('part_id').copy()
    labels['future_defective']=labels['defect_state'].astype(str).str.lower().isin({'latent','hard','defective','failed'}).astype(int)
else:
    raise SystemExit('No future_defective_168h or legacy defect_state label in validation data')
s=s.merge(labels[['part_id','future_defective']],on='part_id',how='left')
if 'absolute_fail_168h' in s.columns:
    s['latent_escape_target']=((pd.to_numeric(s['absolute_fail_168h'],errors='coerce').fillna(0).astype(int)==0) & (s['future_defective']==1)).astype(int)
payload=calibrate_safety_policy(s, output_path=PROJECT_ROOT/'models'/'calibration'/'safety_policy.json', fn_cost=100.0, fp_cost=1.0, max_reject_rate=0.25, target_metric="escape_recall")
print('SAFETY POLICY CALIBRATED ON VALIDATION LOTS ONLY')
print(payload['optimization'])
PY
python -m src.utils validate-config
python - <<'PY'
from pathlib import Path
from src.utils import validate_artifact_compatibility
checks=[(Path('models/anomaly/model.joblib'),'anomaly_model'),(Path('models/forecast/model.joblib'),'forecast_model'),(Path('models/calibration/ood_profile.joblib'),'ood_profile')]
for p,k in checks:
    m=validate_artifact_compatibility(p,artifact_kind=k)
    print(f'ARTIFACT OK: {p} [{k}] runtime={m["runtime"]["scikit_learn"]}')
print('SAFETY POLICY:', Path('models/calibration/safety_policy.json').exists())
PY
echo "RELEASE-RETRAIN COMPLETE"
echo "Previous artifacts backed up under: $backup"


