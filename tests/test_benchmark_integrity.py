import pandas as pd
from pathlib import Path

def test_split_integrity():
    p=Path('data/processed/module_A_dataset.csv')
    if not p.exists(): return
    d=pd.read_csv(p); sets={s:set(d.loc[d.split.astype(str).eq(s),'lot_id']) for s in ['train','val','test']}
    assert not sets['train']&sets['val']; assert not sets['train']&sets['test']; assert not sets['val']&sets['test']
