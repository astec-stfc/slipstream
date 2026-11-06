"""Optional float64 CUDA pair kernel; the model/reference lives in injector_ibs."""
from __future__ import annotations

import torch

from ...constants import C_LIGHT
from . import warp_kernels


if warp_kernels._HAVE_WARP:
    import warp as wp

    # Match the reference's operation order around small thermal increments.
    wp.set_module_options({"enable_backward": False, "enable_fp_fusion": False})
    _C = wp.constant(wp.float64(C_LIGHT))

    @wp.func
    def _expm1(x: wp.float64):
        if wp.abs(x) < wp.float64(0.01):
            return x * (wp.float64(1.0) + x * (wp.float64(0.5) + x * (wp.float64(1.0)/wp.float64(6.0)
                + x * (wp.float64(1.0)/wp.float64(24.0) + x * (wp.float64(1.0)/wp.float64(120.0)
                + x * (wp.float64(1.0)/wp.float64(720.0) + x * wp.float64(1.0)/wp.float64(5040.0)))))))
        return wp.exp(x) - wp.float64(1.0)

    @wp.func
    def _log1p(x: wp.float64):
        if wp.abs(x) < wp.float64(0.01):
            return x * (wp.float64(1.0) + x * (wp.float64(-0.5) + x * (wp.float64(1.0)/wp.float64(3.0)
                + x * (wp.float64(-0.25) + x * (wp.float64(0.2) + x * (wp.float64(-1.0)/wp.float64(6.0)
                + x * (wp.float64(1.0)/wp.float64(7.0) - x * wp.float64(0.125))))))))
        return wp.log(wp.float64(1.0) + x)

    @wp.func
    def _nanbu_parameter(variance: wp.float64):
        one = wp.float64(1.0)
        two = wp.float64(2.0)
        if variance > wp.float64(20.0):
            return wp.float64(3.0) * wp.exp(-two * variance)
        high = one / wp.max(-_expm1(-two * variance), wp.float64(2.2250738585072014e-308))
        if variance <= wp.float64(0.02):
            return high
        low = wp.float64(0.0)
        target = wp.exp(-two * variance)
        for iteration in range(48):
            mid = (low + high) / two
            mean = wp.float64(0.0)
            if mid < wp.float64(0.1):
                a2 = mid * mid
                mean = mid * (wp.float64(1.0)/wp.float64(3.0) - a2/wp.float64(45.0)
                    + two*a2*a2/wp.float64(945.0) - a2*a2*a2/wp.float64(4725.0)
                    + two*a2*a2*a2*a2/wp.float64(93555.0))
            else:
                mean = one/wp.tanh(mid) - one/mid
            if mean < target:
                low = mid
            else:
                high = mid
        return (low + high) / two

    @wp.kernel
    def _collide(
        velocity: wp.array2d(dtype=wp.float64),
        indices: wp.array(dtype=wp.int64), a: wp.array(dtype=wp.int64), b: wp.array(dtype=wp.int64),
        counts: wp.array(dtype=wp.int64), pair_cells: wp.array(dtype=wp.int64),
        cell_gamma: wp.array(dtype=wp.float64),
        azimuth: wp.array(dtype=wp.float64), random: wp.array(dtype=wp.float64),
        coefficient: wp.float64, density_scale: wp.float64, dt: wp.float64,
        fixed_log: wp.float64, log_cutoff: wp.float64, max_beta: wp.float64, model: wp.int32,
        result: wp.array2d(dtype=wp.float64), stats: wp.array2d(dtype=wp.float64),
    ):
        i = wp.tid()
        ia, ib = indices[a[i]], indices[b[i]]
        one, two = wp.float64(1.0), wp.float64(2.0)
        ua = wp.vec3d(velocity[ia, 0], velocity[ia, 1], velocity[ia, 2])
        ub = wp.vec3d(velocity[ib, 0], velocity[ib, 1], velocity[ib, 2])
        ga = wp.sqrt(one + wp.dot(ua/_C, ua/_C))
        gb = wp.sqrt(one + wp.dot(ub/_C, ub/_C))
        beta = (ua + ub) / (_C * (ga + gb))
        gamma = one / wp.sqrt(one - wp.dot(beta, beta))
        difference = (ua - ub) / two
        rest = difference - (gamma/(gamma+one) * wp.dot(difference, beta)) * beta
        norm = wp.length(rest)
        rest_gamma = wp.sqrt(one + (norm/_C)*(norm/_C))
        thermal_beta = norm / (_C * rest_gamma)
        stats[i, 0] = wp.float64(0.0)
        stats[i, 1] = thermal_beta
        if thermal_beta > max_beta:
            return
        speed = two * norm / rest_gamma
        logarithm = fixed_log
        if fixed_log < wp.float64(0.0):
            logarithm = wp.float64(0.0)
            if speed > wp.float64(0.0):
                logarithm = wp.max(log_cutoff + two*wp.log(speed), wp.float64(0.0))
        if speed == wp.float64(0.0) or logarithm == wp.float64(0.0):
            return
        occupancy = wp.float64(counts[pair_cells[i]])
        correction = one
        if counts[pair_cells[i]] % wp.int64(2) == wp.int64(1):
            correction = occupancy / (occupancy-one)
        density = occupancy*density_scale/cell_gamma[i]
        rest_time = dt/cell_gamma[i]
        variance = coefficient*density*rest_time*logarithm/(speed*speed*speed)*correction
        stats[i, 0] = variance
        if not wp.isfinite(variance) or variance <= wp.float64(0.0):
            return
        sine, loss = wp.float64(0.0), wp.float64(0.0)
        if model == 0:
            delta = random[i]*wp.sqrt(variance)
            sine = two*delta/(one+delta*delta)
            loss = two*delta*delta/(one+delta*delta)
        else:
            parameter = _nanbu_parameter(variance)
            uniform = random[i]
            if parameter < wp.float64(1.0e-8):
                loss = two*(one-uniform) - two*parameter*uniform*(one-uniform)
            elif parameter < one:
                loss = two - _log1p(uniform*_expm1(two*parameter))/parameter
            else:
                loss = -wp.log(uniform + (one-uniform)*wp.exp(-two*parameter))/parameter
            loss = wp.clamp(loss, wp.float64(0.0), two)
            sine = wp.sqrt(loss*(two-loss))
        unit = rest/norm
        axis = wp.vec3d(0.0, 0.0, 1.0)
        if wp.abs(unit[0]) <= wp.abs(unit[1]) and wp.abs(unit[0]) <= wp.abs(unit[2]):
            axis = wp.vec3d(1.0, 0.0, 0.0)
        elif wp.abs(unit[1]) <= wp.abs(unit[2]):
            axis = wp.vec3d(0.0, 1.0, 0.0)
        tangent1 = wp.normalize(wp.cross(unit, axis))
        tangent2 = wp.cross(unit, tangent1)
        tangent = wp.cos(azimuth[i])*tangent1 + wp.sin(azimuth[i])*tangent2
        delta_rest = -loss*rest + (sine*norm)*tangent
        increment = delta_rest + (gamma*gamma/(gamma+one)*wp.dot(delta_rest, beta))*beta
        for j in range(3):
            result[ia, j] = ua[j] + increment[j]
            result[ib, j] = ub[j] - increment[j]


def collide(velocity, indices, a, b, counts, pair_cells, cell_gamma, azimuth, random,
            coefficient, density_scale, dt, fixed_log, log_cutoff, max_beta, model):
    """Apply disjoint pairs using caller-owned Torch RNG and the active Torch stream."""
    warp_kernels._ensure_init()
    result = velocity.clone()
    stats = torch.empty((len(a), 2), device=velocity.device, dtype=torch.float64)
    arrays = (velocity, indices, a, b, counts, pair_cells, cell_gamma, azimuth, random)
    wp.launch(_collide, dim=len(a),
              inputs=[warp_kernels._wp_arg(t) for t in arrays]
              + [wp.float64(v) for v in (coefficient, density_scale, dt, fixed_log, log_cutoff, max_beta)]
              + [wp.int32(model == "nanbu"), warp_kernels._wp_arg(result), warp_kernels._wp_arg(stats)],
              stream=warp_kernels._launch_stream(velocity))
    maximum_variance, maximum_beta = stats.amax(dim=0).tolist()
    return result, maximum_variance, maximum_beta
