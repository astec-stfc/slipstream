"""Independent fixed-energy sector-bend transverse transport regression."""
import numpy as np
from scipy.integrate import solve_ivp
import torch

from slipstream.tracking.native.elements import track_element


def test_sector_bend_against_hamiltonian_ode():
    initial = np.array([[0, 0, 0, 0, 0, 0], [0.008, 0.02, 0.001, -0.01, 0, 0.01],
                        [-0.008, -0.02, -0.001, 0.01, 0, -0.01]])
    length, angle = 1.2, 0.25
    h = angle / length
    delta = initial[:, 5]

    def rhs(_, flat):
        x, px, _, py = flat.reshape(-1, 4).T
        pz = np.sqrt((1 + delta) ** 2 - px ** 2 - py ** 2)
        return np.stack(((1 + h*x)*px/pz, h*(pz - 1 - h*x),
                         (1 + h*x)*py/pz, np.zeros_like(py)), axis=1).ravel()

    ode = solve_ivp(rhs, (0, length), initial[:, :4].ravel(), method="DOP853",
                    rtol=3e-14, atol=3e-16)
    assert ode.success
    expected = ode.y[:, -1].reshape(-1, 4)
    for exact in (True, False):
        actual = track_element(torch.tensor(initial), length, angle, 0, 0, 0, 0,
                               629e6, 510998.95, n_kicks=32, integration_order=6,
                               exact_dipole=exact).numpy()
        np.testing.assert_allclose(actual[:, :4], expected, atol=3e-11, rtol=0)
        np.testing.assert_array_equal(actual[:, 5], delta)
