"""``Injector.from_astra`` reads solenoid x and y offsets and tilts (ASTRA ``S_xoff``, ``S_yoff``, ``S_xrot``, ``S_yrot``)."""
from slipstream.tracking.native.injector import Injector

DECK = """&NEWRUN
Head = 'trial', RUN = 1, Distribution = 'beam.astra', Auto_Phase = False, H_max = 0.001, H_min = 0.001
/
&OUTPUT
ZSTART = 0.0, ZSTOP = 0.5
/
&CHARGE
LSPCH = False
/
&CAVITY
LEField = False
/
&SOLENOID
LBField = True
S_pos(1) = 0.01, FILE_BFieLD(1) = 'sol.astra', MaxB(1) = 0.2, S_xoff(1) = 1.0e-3, S_yoff(1) = -2.0e-3, S_xrot(1) = 3.0e-3, S_yrot(1) = -4.0e-3
/
"""


def test_solenoid_placement_is_read_in_both_planes(tmp_path):
    deck = tmp_path / "injector.in"
    deck.write_text(DECK)
    sol, = Injector.from_astra(deck).solenoids
    assert (sol.x_m, sol.y_m, sol.xrot_rad, sol.yrot_rad) == (1.0e-3, -2.0e-3, 3.0e-3, -4.0e-3)
    assert sol.placed
