"""Cylindrical PIC deposits/gathers with CIC and combines longitudinal open-boundary FFT treatment with
radial finite differences. See docs/modules/tracking/native/space_charge.md."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ...constants import EPS0_F_PER_M


def smooth_rho(rho: torch.Tensor, passes_r: int, passes_z: int) -> torch.Tensor:
    """Compute smooth rho; see the module documentation for details."""
    for _ in range(passes_r):
        # F.pad avoids the compiled CUDA alignment bug on odd grids.
        p = F.pad(rho, (0, 0, 1, 1))
        axis_ghost = F.pad(rho[1:2], (0, 0, 0, rho.shape[0] - 1))
        rho = 0.25 * p[:-2] + 0.5 * p[1:-1] + 0.25 * p[2:] + 0.25 * axis_ghost
    for _ in range(passes_z):
        p = F.pad(rho, (1, 1))
        rho = 0.25 * p[:, :-2] + 0.5 * p[:, 1:-1] + 0.25 * p[:, 2:]
    return rho


def _default_device_dtype(
    device: str | torch.device | None, dtype: torch.dtype | None
) -> tuple[torch.device, torch.dtype]:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if dtype is None:
        dtype = torch.float32 if device.type == "cuda" else torch.float64
    return device, dtype


def _radial_laplacian_matrix(nr: int, dr: float, device, dtype) -> torch.Tensor:
    """``(nr, nr)`` matrix ``L`` such that ``(L @ phi)[i]`` approximates ``(1/r) d/dr(r dphi/dr)`` at
    node ``r_i = i*dr``."""
    L = torch.zeros((nr, nr), device=device, dtype=dtype)
    L[0, 0] = -4.0 / dr**2
    if nr > 1:
        L[0, 1] = 4.0 / dr**2
    for i in range(1, nr):
        r_i = i * dr
        r_plus = (i + 0.5) * dr
        r_minus = (i - 0.5) * dr
        L[i, i] = -(r_plus + r_minus) / (r_i * dr**2)
        if i + 1 < nr:
            L[i, i + 1] = r_plus / (r_i * dr**2)
        L[i, i - 1] = r_minus / (r_i * dr**2)
    return L


class CylindricalSpaceCharge:
    """Axisymmetric (r, z) electrostatic space-charge solver on a fixed grid."""

    def __init__(
        self,
        nr: int,
        nz: int,
        r_max: float,
        z_min: float,
        z_max: float,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device, dtype = _default_device_dtype(device, dtype)
        self.nr, self.nz = nr, nz
        self.device, self.dtype = device, dtype
        # Hockney zero-padding: at least double the z-domain so the periodic FFT convolution reproduces an open boundary in z.
        self.nz_pad = 1 << (2 * nz - 1).bit_length()
        L_unit = _radial_laplacian_matrix(nr, 1.0, "cpu", torch.float64)
        v_unit = 2 * math.pi * torch.arange(nr, dtype=torch.float64)
        v_unit[0] = math.pi / 4  # half-disk axis cell, pi*(1/2)^2
        sv = torch.sqrt(v_unit)
        lam, W = torch.linalg.eigh(sv[:, None] * L_unit / sv[None, :])
        self._lam = lam.to(device=device, dtype=dtype)
        self.P_left = (W / sv[:, None]).to(device=device, dtype=dtype)
        self.P_right = (W.mT * sv[None, :]).to(device=device, dtype=dtype)
        self._set_grid(r_max, z_min, z_max)

    def _set_grid(self, r_max: float, z_min: float, z_max: float) -> None:
        """Derive every grid-dependent quantity (spacings, node volumes,
        the per-frequency radial-solve operators) from a physical box."""
        self.dr = r_max / self.nr
        self.dz = (z_max - z_min) / (self.nz - 1)
        self.z_min = z_min
        r_nodes = torch.arange(self.nr, device=self.device, dtype=self.dtype) * self.dr
        self.r_nodes = r_nodes
        node_volume = 2 * math.pi * r_nodes * self.dr * self.dz
        node_volume[0] = math.pi * (self.dr / 2) ** 2 * self.dz
        self.node_volume = node_volume
        # dtype passed in, not cast after: rfftfreq defaults to float32,
        # which silently put ~1e-7 relative error on every k^2.
        k = 2 * math.pi * torch.fft.rfftfreq(
            self.nz_pad, d=self.dz, device=self.device, dtype=self.dtype
        )
        # A^-1 = P_left @ diag(inv_diag) @ P_right, per z-frequency (see __init__).
        self.inv_diag = 1.0 / (self._lam[:, None] / self.dr**2 - k[None, :] ** 2)  # (nr, n_freq)

    def coefficients(self, dr, dz):
        """Compute coefficients; see the module documentation for details."""
        v = 2 * math.pi * torch.arange(self.nr, device=self.device, dtype=self.dtype)
        v[0] = math.pi / 4  # half-disk axis cell, as in __init__
        k1 = 2 * math.pi * torch.fft.rfftfreq(self.nz_pad, d=1.0, device=self.device, dtype=self.dtype)
        return v * dr**2 * dz, 1.0 / (self._lam[:, None] / dr**2 - (k1[None, :] / dz) ** 2)

    def rescale(self, r_max: float, z_min: float, z_max: float) -> None:
        """Move/resize the grid's physical box in place, keeping nr, nz (hence the deposited-grid
        shape, the FFT length, and every ``torch.compile`` graph that captured them) unchanged."""
        self._set_grid(r_max, z_min, z_max)

    def _cic_weights(self, r: torch.Tensor, z: torch.Tensor):
        r_f = r / self.dr
        r0 = torch.clamp(r_f.floor().long(), 0, self.nr - 2)
        wr = torch.clamp(r_f - r0, 0.0, 1.0)
        z_f = (z - self.z_min) / self.dz
        z0 = torch.clamp(z_f.floor().long(), 0, self.nz - 2)
        wz = torch.clamp(z_f - z0, 0.0, 1.0)
        return r0, wr, z0, wz

    def deposit(self, r: torch.Tensor, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Compute deposit; see the module documentation for details."""
        r0, wr, z0, wz = self._cic_weights(r, z)
        charge = torch.zeros(self.nr * self.nz, device=self.device, dtype=self.dtype)
        for dr_i, wr_i in ((0, 1 - wr), (1, wr)):
            for dz_i, wz_i in ((0, 1 - wz), (1, wz)):
                idx = (r0 + dr_i) * self.nz + (z0 + dz_i)
                charge.scatter_add_(0, idx, q * wr_i * wz_i)
        charge = charge.view(self.nr, self.nz)
        return charge / self.node_volume.unsqueeze(1)

    def solve(
        self, rho: torch.Tensor, smooth_r: int = 0, smooth_z: int = 0,
        gamma: float = 1.0, smooth_gamma_ref: float = math.inf,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Poisson solve: charge-density grid ``(nr, nz)`` -> ``(Er, Ez)`` field grids ``(nr, nz)``."""
        if smooth_r or smooth_z:
            w = 1.0 / (1.0 + (gamma / smooth_gamma_ref) ** 2)
            if w > 0:
                rho = rho + w * (smooth_rho(rho, smooth_r, smooth_z) - rho)
        rho_padded = torch.zeros(self.nr, self.nz_pad, device=self.device, dtype=self.dtype)
        rho_padded[:, : self.nz] = self.P_right @ rho
        rho_k = torch.fft.rfft(rho_padded, dim=1)  # (nr, n_freq), complex
        phi_k = rho_k * (-self.inv_diag / EPS0_F_PER_M)
        phi_padded = torch.fft.irfft(phi_k, n=self.nz_pad, dim=1)  # real, (nr, nz_pad)
        phi = self.P_left @ phi_padded[:, : self.nz]

        Ez = torch.zeros_like(phi)
        Ez[:, 1:-1] = -(phi[:, 2:] - phi[:, :-2]) / (2 * self.dz)
        Ez[:, 0] = -(phi[:, 1] - phi[:, 0]) / self.dz
        Ez[:, -1] = -(phi[:, -1] - phi[:, -2]) / self.dz

        Er = torch.zeros_like(phi)
        Er[1:-1, :] = -(phi[2:, :] - phi[:-2, :]) / (2 * self.dr)
        Er[0, :] = 0.0  # axisymmetry: Er(r=0) = 0 exactly
        Er[-1, :] = phi[-2, :] / (2 * self.dr)  # ghost node at the Dirichlet r_max boundary, phi=0

        return Er, Ez

    def gather(self, grid: torch.Tensor, r: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Bilinear interpolation of a ``(nr, nz)`` grid back to particle
        positions -- the reverse of ``deposit``."""
        r0, wr, z0, wz = self._cic_weights(r, z)
        return (
            grid[r0, z0] * (1 - wr) * (1 - wz)
            + grid[r0 + 1, z0] * wr * (1 - wz)
            + grid[r0, z0 + 1] * (1 - wr) * wz
            + grid[r0 + 1, z0 + 1] * wr * wz
        )

    def field_at(
        self, r: torch.Tensor, z: torch.Tensor, q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convenience: ``deposit(r,z,q)`` -> ``solve`` -> ``gather`` back
        at the same ``(r, z)`` -- the full self-field of this
        macroparticle set, evaluated at its own particles."""
        rho = self.deposit(r, z, q)
        Er_grid, Ez_grid = self.solve(rho)
        return self.gather(Er_grid, r, z), self.gather(Ez_grid, r, z)


def cartesian_field_at(
    solver: CylindricalSpaceCharge, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, q: torch.Tensor,
    smooth_r: int = 0, smooth_z: int = 0, gamma: float = 1.0, smooth_gamma_ref: float = math.inf,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``CylindricalSpaceCharge.field_at`` in Cartesian ``(x, y, z)``, with
    ``solve``'s optional density smoothing (see there)."""
    r = torch.sqrt(x * x + y * y)
    rho = solver.deposit(r, z, q)
    Er_grid, Ez_grid = solver.solve(rho, smooth_r, smooth_z, gamma, smooth_gamma_ref)
    Er, Ez = solver.gather(Er_grid, r, z), solver.gather(Ez_grid, r, z)
    r_safe = torch.clamp(r, min=1e-12)
    return Er * x / r_safe, Er * y / r_safe, Ez


def field_with_cathode_image(
    solver: CylindricalSpaceCharge,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    q: torch.Tensor,
    cathode_z: float,
    smooth_r: int = 0, smooth_z: int = 0, gamma: float = 1.0, smooth_gamma_ref: float = math.inf,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute field with cathode image; see the module documentation for details."""
    r = torch.sqrt(x * x + y * y)
    z_image = 2 * cathode_z - z
    r_all = torch.cat([r, r])
    z_all = torch.cat([z, z_image])
    q_all = torch.cat([q, -q])

    rho = solver.deposit(r_all, z_all, q_all)
    Er_grid, Ez_grid = solver.solve(rho, smooth_r, smooth_z, gamma, smooth_gamma_ref)
    Er = solver.gather(Er_grid, r, z)
    Ez = solver.gather(Ez_grid, r, z)

    r_safe = torch.clamp(r, min=1e-12)
    return Er * x / r_safe, Er * y / r_safe, Ez



def _igf_antiderivative(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """``F`` with ``d3F/dxdydz = 1/r`` (Qiang et al., PRSTAB 9, 044204 (2006)). ``ln(u + r)``
    for ``u < 0`` is taken in its cancellation-free form ``ln((r^2 - u^2) / (r - u))``."""
    r = torch.sqrt(x * x + y * y + z * z)
    tiny = torch.finfo(x.dtype).tiny

    def safe_div(a, b):
        return a / torch.where(b == 0, torch.full_like(b, tiny), b)

    def log_plus_r(u, v, w):
        return torch.where(u >= 0, torch.log(u + r),
                           torch.log(torch.clamp(v * v + w * w, min=tiny) / (r - u)))

    return (-0.5 * z * z * torch.atan(safe_div(x * y, z * r))
            - 0.5 * y * y * torch.atan(safe_div(x * z, y * r))
            - 0.5 * x * x * torch.atan(safe_div(y * z, x * r))
            + y * z * log_plus_r(x, y, z) + x * z * log_plus_r(y, x, z) + x * y * log_plus_r(z, x, y))


def integrated_green(ox: torch.Tensor, oy: torch.Tensor, oz: torch.Tensor,
                     hx: float, hy: float, hz: float, far_cells: float = 20.0) -> torch.Tensor:
    """Compute integrated green; see the module documentation for details."""
    total = 0.0
    for sx in (-0.5, 0.5):
        for sy in (-0.5, 0.5):
            for sz in (-0.5, 0.5):
                sign = 1.0 if sx * sy * sz > 0 else -1.0
                total = total + sign * _igf_antiderivative(ox + sx * hx, oy + sy * hy, oz + sz * hz)
    igf = total / (hx * hy * hz)
    r = torch.sqrt(ox * ox + oy * oy + oz * oz)
    far = r > far_cells * max(hx, hy, hz)
    return torch.where(far, 1.0 / torch.where(far, r, torch.ones_like(r)), igf)


def _green_fft(nx, ny, nz, hx, hy, hz, z_shift, device):
    """rFFT of the zero-padded (Hockney, ``2n`` per axis) Green's function ``1/(4 pi eps0 r)``, cell-
    integrated, with every ``z`` offset displaced by ``z_shift`` (``0`` for the direct kernel)."""
    f64 = torch.float64

    def offsets(n, h):
        m = torch.arange(2 * n, device=device, dtype=f64)
        return torch.where(m < n, m, m - 2 * n) * h

    ox = offsets(nx, hx)[:, None, None]
    oy = offsets(ny, hy)[None, :, None]
    oz = offsets(nz, hz)[None, None, :] + z_shift
    g = integrated_green(ox, oy, oz, hx, hy, hz) / (4 * math.pi * EPS0_F_PER_M)
    return torch.fft.rfftn(g)


# Cap the padded Hockney volume, not the smaller requested mesh: kernel
# construction uses float64/complex buffers at this size before any solve.
SC3D_MAX_GRID_CELLS = 1024**3


class SpaceCharge3D:
    """Cartesian open-boundary particle-in-cell space-charge solver."""

    def __init__(
        self,
        nx: int, ny: int, nz: int,
        hx: float, hy: float, hz: float,
        x_lo: float, y_lo: float, z_lo: float,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        far_cells: float = 20.0,
        max_grid_cells: int = SC3D_MAX_GRID_CELLS,
    ):
        device, dtype = _default_device_dtype(device, dtype)
        if min(nx, ny, nz) < 2:
            raise ValueError("SpaceCharge3D needs at least two nodes per axis")
        if not all(math.isfinite(h) and h > 0 for h in (hx, hy, hz)):
            raise ValueError("SpaceCharge3D cell sizes must be finite and positive")
        # Power-of-two Hockney padding is the established cuFFT discipline
        # in this module: at least 2n on every axis, never an awkward FFT.
        px = 1 << (2 * nx - 1).bit_length()
        py = 1 << (2 * ny - 1).bit_length()
        pz = 1 << (2 * nz - 1).bit_length()
        padded_grid_cells = px * py * pz
        if max_grid_cells <= 0 or padded_grid_cells > max_grid_cells:
            raise ValueError(
                f"SpaceCharge3D padded grid {px}x{py}x{pz} ({padded_grid_cells} cells; "
                f"requested {nx}x{ny}x{nz}) exceeds max_grid_cells={max_grid_cells}; "
                "reduce n_xyz or raise the explicit memory cap"
            )
        self.nx, self.ny, self.nz = nx, ny, nz
        self.hx, self.hy, self.hz = hx, hy, hz
        self.device, self.dtype = device, dtype
        self.px, self.py, self.pz = px, py, pz

        # G is even in every axis. Evaluate its expensive IGF expression on
        # one positive octant, then reflect into the periodic Hockney grid.
        ox = torch.arange(self.px // 2 + 1, device=device, dtype=torch.float64)[:, None, None] * hx
        oy = torch.arange(self.py // 2 + 1, device=device, dtype=torch.float64)[None, :, None] * hy
        oz = torch.arange(self.pz // 2 + 1, device=device, dtype=torch.float64)[None, None, :] * hz
        octant = integrated_green(ox, oy, oz, hx, hy, hz, far_cells) / (4 * math.pi * EPS0_F_PER_M)

        def reflected_indices(n: int) -> torch.Tensor:
            i = torch.arange(n, device=device)
            return torch.minimum(i, n - i)

        g = octant.index_select(0, reflected_indices(self.px))
        g = g.index_select(1, reflected_indices(self.py))
        g = g.index_select(2, reflected_indices(self.pz))
        # Match the stateless solver: transform in float64, store in the
        # solve precision.
        self.G_k = torch.fft.rfftn(g).to(torch.complex128 if dtype == torch.float64 else torch.complex64)
        self.rescale(x_lo, y_lo, z_lo)

    def rescale(self, x_lo: float, y_lo: float, z_lo: float) -> None:
        """Move the box origin; ``G_k`` stays valid because it depends only
        on the fixed grid shape and cell sizes, not absolute coordinates."""
        if not all(math.isfinite(v) for v in (x_lo, y_lo, z_lo)):
            raise ValueError("SpaceCharge3D origins must be finite")
        self.x_lo, self.y_lo, self.z_lo = x_lo, y_lo, z_lo

    @property
    def x_hi(self) -> float:
        return self.x_lo + (self.nx - 1) * self.hx

    @property
    def y_hi(self) -> float:
        return self.y_lo + (self.ny - 1) * self.hy

    @property
    def z_hi(self) -> float:
        return self.z_lo + (self.nz - 1) * self.hz

    def _cic(self, u: torch.Tensor, lo: float, h: float, n: int):
        f = (u - lo) / h
        i0 = torch.clamp(f.floor().long(), 0, n - 2)
        return i0, torch.clamp(f - i0, 0.0, 1.0)

    def deposit(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Cloud-in-Cell deposit of macroparticle charges onto the ``(nx, ny, nz)`` node grid, returned
        as CHARGE per node (see the class docstring)."""
        ix, wx = self._cic(x, self.x_lo, self.hx, self.nx)
        iy, wy = self._cic(y, self.y_lo, self.hy, self.ny)
        iz, wz = self._cic(z, self.z_lo, self.hz, self.nz)
        charge = torch.zeros(self.nx * self.ny * self.nz, device=self.device, dtype=self.dtype)
        for a, wa in ((0, 1 - wx), (1, wx)):
            for b, wb in ((0, 1 - wy), (1, wy)):
                for c, wc in ((0, 1 - wz), (1, wz)):
                    idx = ((ix + a) * self.ny + (iy + b)) * self.nz + (iz + c)
                    charge.scatter_add_(0, idx, q * wa * wb * wc)
        return charge.view(self.nx, self.ny, self.nz)

    def solve(self, charge: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Charge grid ``(nx, ny, nz)`` -> ``(Ex, Ey, Ez)`` field grids (V/m): Hockney convolution with
        the cached cell-averaged kernel, central-difference gradients (one-sided at edges, matching
        ``torch.gradient(edge_order=1)``)."""
        rho_k = torch.fft.rfftn(charge, s=(self.px, self.py, self.pz))
        phi = torch.fft.irfftn(rho_k * self.G_k, s=(self.px, self.py, self.pz))
        phi = phi[: self.nx, : self.ny, : self.nz]

        def neg_grad(axis: int, h: float) -> torch.Tensor:
            E = torch.zeros_like(phi)
            if axis == 0:
                E[1:-1, :, :] = -(phi[2:, :, :] - phi[:-2, :, :]) / (2 * h)
                E[0, :, :] = -(phi[1, :, :] - phi[0, :, :]) / h
                E[-1, :, :] = -(phi[-1, :, :] - phi[-2, :, :]) / h
            elif axis == 1:
                E[:, 1:-1, :] = -(phi[:, 2:, :] - phi[:, :-2, :]) / (2 * h)
                E[:, 0, :] = -(phi[:, 1, :] - phi[:, 0, :]) / h
                E[:, -1, :] = -(phi[:, -1, :] - phi[:, -2, :]) / h
            else:
                E[:, :, 1:-1] = -(phi[:, :, 2:] - phi[:, :, :-2]) / (2 * h)
                E[:, :, 0] = -(phi[:, :, 1] - phi[:, :, 0]) / h
                E[:, :, -1] = -(phi[:, :, -1] - phi[:, :, -2]) / h
            return E

        return neg_grad(0, self.hx), neg_grad(1, self.hy), neg_grad(2, self.hz)

    def gather(self, grid: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Trilinear interpolation of a field grid back to particle
        positions -- the reverse of ``deposit``."""
        ix, wx = self._cic(x, self.x_lo, self.hx, self.nx)
        iy, wy = self._cic(y, self.y_lo, self.hy, self.ny)
        iz, wz = self._cic(z, self.z_lo, self.hz, self.nz)
        acc = torch.zeros_like(x)
        for a, wa in ((0, 1 - wx), (1, wx)):
            for b, wb in ((0, 1 - wy), (1, wy)):
                for c, wc in ((0, 1 - wz), (1, wz)):
                    acc = acc + grid[ix + a, iy + b, iz + c] * wa * wb * wc
        return acc

    def field_at(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convenience: ``deposit`` -> ``solve`` -> ``gather`` at the
        particles' own positions, matching ``CylindricalSpaceCharge``."""
        charge = self.deposit(x, y, z, q)
        grids = self.solve(charge)
        return tuple(self.gather(g, x, y, z) for g in grids)


def space_charge_3d(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, q: torch.Tensor,
    n_cells: tuple[int, int, int], box: tuple[float, float, float, float, float, float],
    mirror_z: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute space charge 3d; see the module documentation for details."""
    nx, ny, nz = n_cells
    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = box
    hx, hy, hz = (x_hi - x_lo) / (nx - 1), (y_hi - y_lo) / (ny - 1), (z_hi - z_lo) / (nz - 1)
    dev, dt = x.device, x.dtype

    def cic(u, lo, h, n):
        f = (u - lo) / h
        i0 = torch.clamp(f.floor().long(), 0, n - 2)
        return i0, torch.clamp(f - i0, 0.0, 1.0)

    ix, wx = cic(x, x_lo, hx, nx)
    iy, wy = cic(y, y_lo, hy, ny)
    iz, wz = cic(z, z_lo, hz, nz)
    corners = [(a, b, c, (wx if a else 1 - wx) * (wy if b else 1 - wy) * (wz if c else 1 - wz))
               for a in (0, 1) for b in (0, 1) for c in (0, 1)]

    charge = torch.zeros(nx * ny * nz, device=dev, dtype=dt)
    for a, b, c, w in corners:
        charge.scatter_add_(0, ((ix + a) * ny + (iy + b)) * nz + (iz + c), q * w)
    charge = charge.view(nx, ny, nz)

    pad = (2 * nx, 2 * ny, 2 * nz)
    phi_k = torch.fft.rfftn(charge, s=pad) * _green_fft(nx, ny, nz, hx, hy, hz, 0.0, dev).to(
        torch.complex128 if dt == torch.float64 else torch.complex64)
    if mirror_z is not None:
        shift = 2.0 * (z_lo - mirror_z) + (nz - 1) * hz
        g_img = _green_fft(nx, ny, nz, hx, hy, hz, shift, dev).to(phi_k.dtype)
        phi_k = phi_k - torch.fft.rfftn(charge.flip(2), s=pad) * g_img
    phi = torch.fft.irfftn(phi_k, s=pad)[:nx, :ny, :nz]

    def neg_grad(axis, h):
        return -torch.gradient(phi, spacing=h, dim=axis, edge_order=1)[0]

    grids = (neg_grad(0, hx), neg_grad(1, hy), neg_grad(2, hz))
    out = []
    for g in grids:
        acc = torch.zeros_like(x)
        for a, b, c, w in corners:
            acc = acc + g[ix + a, iy + b, iz + c] * w
        out.append(acc)
    return tuple(out)

