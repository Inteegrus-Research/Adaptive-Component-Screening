from pathlib import Path

def test_no_fake_progressive_metrics():
    p=Path('src/evaluation.py').read_text().lower(); assert 'metric_values={' not in p
