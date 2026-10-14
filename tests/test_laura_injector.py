"""LAURA typed importer smoke checks use optional installed CLARA data, no generated field files."""
import math
from pathlib import Path

import pytest
import torch

from slipstream.tracking.native.laura_injector import from_laura

LAURA = pytest.importorskip('laura.laura').LAURA
CLARA = pytest.importorskip('laura_lattices.CLARA')


def machine():
    return LAURA(lattice=CLARA, layout={'default_layout': 'INJECTOR',
                                      'layouts': {'INJECTOR': ['HRG1', 'S01']}})


def test_from_machine_module_uses_scoped_sections_and_original_field_maps():
    m = from_laura(CLARA, ('HRG1', 'S01'), 1.02)
    assert len(m.cavities) == len(m.solenoids) == 1
    assert m.cavities[0].z_m == 0.0
    assert m.cavities[0].crest_deg is None
    assert m.cavities[0].phase_deg == -machine()["CLA-HRG1-GUN-CAV-01"].cavity.phase
    assert m.cavities[0].e_max_V_per_m == pytest.approx(machine()['CLA-HRG1-GUN-CAV-01'].simulation.field_amplitude)
    assert m.solenoids[0].z_m == pytest.approx(0.00241)
    assert m.solenoids[0].b_max_T == pytest.approx(0.345)
    assert m.screens == [0.9803]
    assert len(m.apertures) == 4
    for el in (*m.cavities, *m.solenoids):
        assert Path(el.file).is_file()
        assert Path(el.file).suffix.lower() == '.hdf5'
        assert Path(el.file).is_relative_to(Path(CLARA.__file__).parent)
    assert m._solenoid_field(m.solenoids[0], device='cpu', dtype=torch.float64).map.n == 4001
    assert m._cavity_field(m.cavities[0], device='cpu', dtype=torch.float64).map.n == 4001


def test_prebuilt_machine_preserves_controls_and_solenoid_placement():
    lattice = machine()
    gun = lattice['CLA-HRG1-GUN-CAV-01']
    sol = lattice['EBT-HRG1-MAG-SOL-01']
    gun.cavity.phase = 7.5
    gun.simulation.field_amplitude = 83e6
    sol.magnetic.field_amplitude = -0.23
    sol.physical.error.position.x = 2e-4
    sol.physical.error.position.y = -1e-4
    sol.physical.error.rotation.theta = 2e-3
    sol.physical.error.rotation.phi = -3e-3
    m = from_laura(lattice, ('HRG1', 'S01'), 1.02)
    assert m.cavities[0].phase_deg == -7.5
    assert m.cavities[0].e_max_V_per_m == 83e6
    assert m.solenoids[0].b_max_T == pytest.approx(-0.23)
    assert m.solenoids[0].x_m == 2e-4
    assert m.solenoids[0].y_m == -1e-4
    assert m.solenoids[0].xrot_rad == -2e-3
    assert m.solenoids[0].yrot_rad == 3e-3
    assert isinstance(gun.simulation.field_definition, str)
    assert isinstance(sol.simulation.field_definition, str)


def test_imported_values_match_laura_astra_translator_without_converted_map_writes(monkeypatch, tmp_path):
    from laura.translator.converters.base import BaseElementTranslator
    from laura.translator.converters.converter import translate_elements
    from slipstream.tracking.native.astra_io import read_astra_namelist

    lattice = machine()
    lattice['CLA-HRG1-GUN-CAV-01'].cavity.phase = 9.25
    sol = lattice['EBT-HRG1-MAG-SOL-01']
    sol.physical.error.position.x = 1e-4
    sol.physical.error.position.y = -2e-4
    sol.physical.error.rotation.theta = 2e-3
    sol.physical.error.rotation.phi = -1e-3
    section = lattice.sections['HRG1']
    elements = translate_elements([section.elements[n] for n in section.order],
                                  master_lattice=section.master_lattice)
    monkeypatch.setattr(BaseElementTranslator, 'generate_field_file_name',
                        lambda self, param, **kwargs: param.filename)
    text = '&CAVITY\n' + elements['CLA-HRG1-GUN-CAV-01'].to_astra(n=1) + '/\n'
    text += '&SOLENOID\n' + elements['EBT-HRG1-MAG-SOL-01'].to_astra(n=1) + '/\n'
    path = tmp_path / 'reference.in'
    path.write_text(text)
    nml = read_astra_namelist(path)
    m = from_laura(lattice, ('HRG1', 'S01'), 1.02)
    cav, sol = m.cavities[0], m.solenoids[0]
    assert cav.z_m == pytest.approx(nml['cavity']['c_pos'][1], abs=1e-8)
    assert cav.phase_deg == pytest.approx(nml['cavity']['phi'][1])
    assert cav.e_max_V_per_m == pytest.approx(nml['cavity']['maxe'][1] * 1e6)
    for attribute, key in [('z_m', 's_pos'), ('b_max_T', 'maxb'), ('x_m', 's_xoff'),
                           ('y_m', 's_yoff'), ('xrot_rad', 's_xrot'), ('yrot_rad', 's_yrot')]:
        assert getattr(sol, attribute) == pytest.approx(nml['solenoid'][key][1], abs=1e-8)


def test_absolute_phase_fields_match_laura_astra_sine_convention(monkeypatch, tmp_path):
    from laura.translator.converters.base import BaseElementTranslator
    from laura.translator.converters.converter import translate_elements
    from slipstream.tracking.native.astra_io import read_astra_namelist

    lattice = machine()
    gun = lattice['CLA-HRG1-GUN-CAV-01']
    gun.cavity.crest = 137.0
    gun.cavity.phase = 12.5
    translated = translate_elements([gun], master_lattice=lattice.master_lattice)[gun.name]
    monkeypatch.setattr(BaseElementTranslator, 'generate_field_file_name',
                        lambda self, param, **kwargs: param.filename)
    path = tmp_path / 'absolute.in'
    path.write_text('&CAVITY\n' + translated.to_astra(n=1, auto_phase=False) + '/\n')
    phi = read_astra_namelist(path)['cavity']['phi'][1]
    m = from_laura(lattice, ('HRG1', 'S01'), 1.02, absolute_phases=True)
    field = m.fields(device='cpu', dtype=torch.float64)[0][0]
    assert math.degrees(field.phase_rad) == pytest.approx(phi - 90.0)
    index = int(field.map.f.abs().argmax())
    z = torch.full((3,), field.z_offset + field.map.z0 + index * field.map.dz, dtype=torch.float64)
    t = torch.tensor([0.0, 3e-11, 8e-11], dtype=torch.float64)
    zero = torch.zeros_like(t)
    ez = field(zero, zero, z, t)[2]
    profile = field.map(z - field.z_offset)[0] * field.scale
    expected = profile * torch.sin(field.omega * t + math.radians(phi))
    torch.testing.assert_close(ez, expected, atol=1e-7, rtol=1e-13)
