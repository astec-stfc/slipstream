"""Import safety must identify changed models without changing legacy maps."""
import json

import pytest

from slipstream import BeamlineStudy, parse_elegant_lattice


def lattice(tmp_path, text):
    path = tmp_path / 'audit.lte'
    path.write_text(text)
    return path


def test_strict_import_rejects_used_unknown_physics(tmp_path):
    path = lattice(tmp_path, 'S: SOLENOID, L=0.3, KS=2\nR: LINE=(S)\n')
    with pytest.raises(ValueError, match='S.*SOLENOID.*drift'):
        parse_elegant_lattice(path, 'R', strict=True)


def test_permissive_import_preserves_drift_and_records_each_occurrence(tmp_path):
    path = lattice(tmp_path, 'S: SOLENOID, L=0.3, KS=2\nR: LINE=(S,S)\n')
    audit = []
    with pytest.warns(UserWarning, match='SOLENOID') as emitted:
        specs = parse_elegant_lattice(path, 'R', audit=audit)
    assert len(emitted) == 1
    assert [(s.kind, s.length, s.occurrence) for s in specs] == [('drift', .3, 1), ('drift', .3, 2)]
    assert [r['occurrence'] for r in audit] == [1, 2]
    assert audit[0]['ignored_attributes'] == {'KS': '2'}
    assert audit[0]['replacement'] == 'drift'
    assert json.loads(json.dumps(audit)) == audit


def test_strict_import_checks_only_expanded_elements(tmp_path):
    path = lattice(tmp_path, 'UNUSED: SOLENOID, L=1\nD: DRIFT, L=2\nM: MARK\nR: LINE=(M,D,D)\n')
    audit = []
    specs = parse_elegant_lattice(path, 'r', strict=True, audit=audit)
    assert [s.length for s in specs] == [2, 2]
    assert audit == []


@pytest.mark.parametrize('definition, token, message', [
    ('Q: KQUAD, L=0.1, K1=3, DX=0.01', 'Q', 'DX'),
    ('K: UNKNOWNKICK, KICK=0.01', 'K', 'UNKNOWNKICK'),
    ('D: DRIFT, L=1', '2*D', 'unresolved'),
    ('C: RFCA, L=0.2, BODY_FOCUS_MODEL="typo"', 'C', 'BODY_FOCUS_MODEL'),
])
def test_strict_import_rejects_ignored_physics_and_unresolved_tokens(tmp_path, definition, token, message):
    path = lattice(tmp_path, definition + '\nR: LINE=(' + token + ')\n')
    with pytest.raises(ValueError, match=message):
        parse_elegant_lattice(path, 'R', strict=True)


def test_missing_root_is_audited_and_strictly_rejected(tmp_path):
    path = lattice(tmp_path, 'D: DRIFT, L=1\n')
    with pytest.raises(ValueError, match='MISSING.*unresolved'):
        parse_elegant_lattice(path, 'MISSING', strict=True)
    audit = []
    with pytest.warns(UserWarning, match='MISSING'):
        assert parse_elegant_lattice(path, 'MISSING', audit=audit) == []
    assert audit[0]['replacement'] == 'dropped'


def test_beamline_facade_forwards_import_policy(tmp_path):
    path = lattice(tmp_path, 'S: SOLENOID, L=0.3, KS=2\nR: LINE=(S)\n')
    with pytest.raises(ValueError, match='SOLENOID'):
        BeamlineStudy.from_lattice_file(path, 'R', 5e6, strict_import=True)
    audit = []
    with pytest.warns(UserWarning):
        study = BeamlineStudy.from_lattice_file(path, 'R', 5e6, device='cpu', import_audit=audit)
    assert study.specs[0].kind == 'drift'
    assert audit[0]['element'] == 'S'
