from pathlib import Path
import yaml


def test_policy_has_safety_states_and_cost_asymmetry():
    p = Path(__file__).parents[1] / 'configs' / 'policy.yaml'
    cfg = yaml.safe_load(p.read_text())
    assert set(cfg['decision_states']['allowed']) == {'SAFE','REVIEW','REJECT','UNKNOWN'}
    assert cfg['costs']['false_negative'] > cfg['costs']['false_positive']
    assert cfg['hard_limit_policy']['absolute_violation_action'] == 'REJECT'
    assert cfg['objective']['forbid_trivial_all_reject_solution'] is True
