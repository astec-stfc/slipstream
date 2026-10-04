"""Shared constants include the speed of light, elementary charge, electron rest energy, classical
electron radius, vacuum permittivity and permeability. See docs/modules/constants.md."""

C_LIGHT = 299792458.0  # m/s
ELECTRON_MASS_EV = 0.51099895e6  # eV/c^2
PROTON_MASS_EV = 938.27208816e6  # eV/c^2 (CODATA; antiproton mass is identical)
E_CHARGE_C = 1.602176634e-19  # C, elementary charge (CODATA, exact under SI 2019)
EPS0_F_PER_M = 8.8541878128e-12  # F/m, vacuum permittivity (CODATA 2018)
R_E_M = 2.8179403262e-15  # m, classical electron radius (CODATA)
R_P_M = R_E_M * ELECTRON_MASS_EV / PROTON_MASS_EV  # m, classical proton radius
