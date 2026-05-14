# file: mms_coupled_solver_discrete_mms.py

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import vtk
from scipy.fft import dctn, fft2, idctn, ifft2
from vtk.util.numpy_support import numpy_to_vtk

Array = np.ndarray


@dataclass
class Params:
    nx: int = 64
    ny: int = 64
    lx: float = 1.0
    ly: float = 1.0

    dt_max: float = 5.0e-4
    dt_min: float = 2.0e-5
    cfl: float = 0.10
    t_final: float = 5.0
    save_every: int = 50

    alphau: float = 10
    alphaphi: float = 10
    alphapsi: float = 10

    h: float = 0.005
    tau: float = 1.0e-4
    eps: float = 5.0e-3
    re: float = 100.0

    etaf: float = 5 * 1.0e-3
    etap: float = 5 * 1.0e-2
    lam: float = 1.0
    lambdae: float = 0.5
    lambdab: float = 1.0e-5
    gamma: float = 0.0025
    rho: float = 1.0
    kper_t: float = 1.0e-1
    kper_b: float = 1.0e-5

    capillary_coeff: float = 0.05
    extra_damping: float = 0.0
    force_clip: float = 2.0
    elastic_force_clip: float = 2.0
    nudge_clip: float = 0.5
    spectral_filter_power: int = 8
    spectral_filter_cutoff: float = 1.0

    fourier_cutoff_u_x: int = 4
    fourier_cutoff_u_y: int = 4
    cosine_cutoff_phi_x: int = 8
    cosine_cutoff_phi_y: int = 8
    cosine_cutoff_psi_x: int = 8
    cosine_cutoff_psi_y: int = 8

    use_mms: bool = False
    use_discrete_mms: bool = False
    disable_velocity_stabilization_for_mms: bool = False
    disable_filter_for_mms: bool = False

    mms_a: float = 0.10
    mms_b: float = 0.10
    mms_init_perturbation: float = 0.0
    mms_velocity_mode: Literal["periodic_2pi", "requested_pi"] = "periodic_2pi"

    outdir: str = "spectral_output_2"


def dct2(a: Array) -> Array:
    return dctn(a, type=2, norm="ortho")


def idct2(a: Array) -> Array:
    return idctn(a, type=2, norm="ortho")


def make_grid(p: Params) -> Tuple[Array, Array, float, float]:
    x = np.linspace(0.0, p.lx, p.nx, endpoint=False)
    y = np.linspace(0.0, p.ly, p.ny, endpoint=False)
    dx = p.lx / p.nx
    dy = p.ly / p.ny
    X, Y = np.meshgrid(x, y, indexing="xy")
    return X, Y, dx, dy


def fourier_wavenumbers(n: int, L: float) -> Array:
    return 2.0 * np.pi * np.fft.fftfreq(n, d=L / n)


def neumann_wavenumbers(n: int, L: float) -> Array:
    return np.pi * np.arange(n) / L


def fourier_operators(p: Params) -> Tuple[Array, Array, Array]:
    kx = fourier_wavenumbers(p.nx, p.lx)
    ky = fourier_wavenumbers(p.ny, p.ly)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    K2 = KX**2 + KY**2
    return KX, KY, K2


def neumann_operators(p: Params) -> Tuple[Array, Array, Array]:
    kx = neumann_wavenumbers(p.nx, p.lx)
    ky = neumann_wavenumbers(p.ny, p.ly)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    K2 = KX**2 + KY**2
    return KX, KY, K2


def fourier_spectral_filter(p: Params) -> Array:
    if p.spectral_filter_cutoff >= 1.0:
        return np.ones((p.ny, p.nx), dtype=float)

    kx = np.fft.fftfreq(p.nx) * p.nx
    ky = np.fft.fftfreq(p.ny) * p.ny
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    rx = np.abs(KX) / max(1.0, 0.5 * p.nx)
    ry = np.abs(KY) / max(1.0, 0.5 * p.ny)
    r = np.sqrt(rx**2 + ry**2)

    filt = np.ones_like(r)
    cutoff = p.spectral_filter_cutoff
    mask = r > cutoff
    z = np.zeros_like(r)
    z[mask] = (r[mask] - cutoff) / max(1.0e-12, 1.0 - cutoff)
    filt[mask] = np.exp(-36.0 * z[mask] ** p.spectral_filter_power)
    return filt


def grad_fd(f: Array, dx: float, dy: float) -> Tuple[Array, Array]:
    pad = np.pad(f, ((1, 1), (1, 1)), mode="edge")
    fx = (pad[1:-1, 2:] - pad[1:-1, :-2]) / (2.0 * dx)
    fy = (pad[2:, 1:-1] - pad[:-2, 1:-1]) / (2.0 * dy)
    return fx, fy


def div_fd(fx: Array, fy: Array, dx: float, dy: float) -> Array:
    fxp = np.pad(fx, ((0, 0), (1, 1)), mode="edge")
    fyp = np.pad(fy, ((1, 1), (0, 0)), mode="edge")
    dfx = (fxp[:, 2:] - fxp[:, :-2]) / (2.0 * dx)
    dfy = (fyp[2:, :] - fyp[:-2, :]) / (2.0 * dy)
    return dfx + dfy


def dct_grad_fd(f: Array, dx: float, dy: float) -> Tuple[Array, Array]:
    return grad_fd(f, dx, dy)


def dct_div_fd(fx: Array, fy: Array, dx: float, dy: float) -> Array:
    return div_fd(fx, fy, dx, dy)


def divergence_fft(u1: Array, u2: Array, KX: Array, KY: Array) -> Array:
    return ifft2(1j * KX * fft2(u1) + 1j * KY * fft2(u2)).real


def clip_phi(phi: Array) -> Array:
    return np.clip(phi, 1.0e-6, 1.0 - 1.0e-6)


def clip_field(f: Array, bound: float) -> Array:
    return np.clip(f, -bound, bound)


def project_fft_low_modes(f: Array, mx: int, my: int) -> Array:
    fh = fft2(f)
    ny, nx = f.shape
    ix = np.fft.fftfreq(nx) * nx
    iy = np.fft.fftfreq(ny) * ny
    IX, IY = np.meshgrid(ix, iy, indexing="xy")
    mask = (np.abs(IX) <= mx) & (np.abs(IY) <= my)
    return ifft2(fh * mask).real


def project_dct_low_modes(f: Array, mx: int, my: int) -> Array:
    ch = dct2(f)
    mask = np.zeros_like(ch, dtype=bool)
    mask[: my + 1, : mx + 1] = True
    return idct2(ch * mask)


def compute_dt(u1: Array, u2: Array, dx: float, dy: float, p: Params) -> float:
    umax = max(1.0e-8, float(np.max(np.sqrt(u1**2 + u2**2))))
    adv_dt = p.cfl * min(dx, dy) / umax
    return max(p.dt_min, min(p.dt_max, adv_dt))


def neumann_apply_minus_laplacian(f: Array, K2n: Array) -> Array:
    return idct2(K2n * dct2(f))


def exact_mms_velocity(X: Array, Y: Array, p: Params) -> Tuple[Array, Array]:
    if p.mms_velocity_mode == "requested_pi":
        return np.sin(np.pi * Y / p.ly), np.sin(np.pi * X / p.lx)
    return np.sin(2.0 * np.pi * Y / p.ly), np.sin(2.0 * np.pi * X / p.lx)


def exact_mms_fields(X: Array, Y: Array, t: float, p: Params) -> Dict[str, Array]:
    decay = np.exp(-t)
    C = np.cos(np.pi * X / p.lx) * np.cos(np.pi * Y / p.ly)

    phi = clip_phi(0.5 + p.mms_b * decay * C)
    psi1 = p.mms_a * decay * C
    psi2 = p.mms_a * decay * C
    u1, u2 = exact_mms_velocity(X, Y, p)

    return {
        "phi": phi,
        "psi1": psi1,
        "psi2": psi2,
        "u1": u1,
        "u2": u2,
        "p": np.zeros_like(X),
    }


def initialize_fields(p: Params, X: Array, Y: Array) -> Dict[str, Array]:
    if p.use_mms:
        exact0 = exact_mms_fields(X, Y, 0.0, p)
        perturb = p.mms_init_perturbation
        phi = clip_phi(
            exact0["phi"]
            + perturb * np.cos(2.0 * np.pi * X / p.lx) * np.cos(2.0 * np.pi * Y / p.ly)
        )
        return {
            "phi": phi,
            "psi1": exact0["psi1"].copy(),
            "psi2": exact0["psi2"].copy(),
            "u1": exact0["u1"].copy(),
            "u2": exact0["u2"].copy(),
            "p": exact0["p"].copy(),
            "phida": exact0["phi"].copy(),
            "psi1da": exact0["psi1"].copy(),
            "psi2da": exact0["psi2"].copy(),
            "u1da": exact0["u1"].copy(),
            "u2da": exact0["u2"].copy(),
            "pda": exact0["p"].copy(),
        }

    x0 = 0.5 * p.lx
    y0 = 0.5 * p.ly
    r0 = 0.18 * min(p.lx, p.ly)

    phi = 0.5 + 0.5 * np.tanh(
        (r0 - np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)) / np.sqrt(2.0 * p.gamma)
    )
    phi = clip_phi(phi)

    psi1 = -Y.copy()
    psi2 = X.copy()
    u1 = np.sin(2 * np.pi * Y)
    u2 = np.sin(2 * np.pi * X)
    p_ref = np.zeros_like(X)
    
    x0da = 0.12 * p.lx
    y0da = 0.3 * p.ly
    r0da = 0.21 * min(p.lx, p.ly)

    phida = 0.5 + 0.5 * np.tanh(
        (r0da - np.sqrt((X - x0da) ** 2 + (Y - y0da) ** 2)) / np.sqrt(2.0 * p.gamma)
    )
    phida = clip_phi(phida)
    p_da = np.zeros_like(X)
    theta = 0.45
    psi1da = -(np.cos(theta) * Y )
    psi2da = np.cos(theta) * X 
    u1da = 0.01 * np.sin(2.0 * np.pi * Y / p.ly)
    u2da = np.zeros_like(X)
    '''
    u1da = np.zeros_like(X)
    u2da = np.zeros_like(X)
    phida = np.zeros_like(X)
    psi1da = np.zeros_like(X)
    psi2da = np.zeros_like(X)
    p_da = np.zeros_like(X)
    '''
    return {
        "phi": phi,
        "psi1": psi1,
        "psi2": psi2,
        "u1": u1,
        "u2": u2,
        "p": p_ref,
        "phida": phida,
        "psi1da": psi1da,
        "psi2da": psi2da,
        "u1da": u1da,
        "u2da": u2da,
        "pda": p_da,
    }


def compute_mu_source(
    phi: Array,
    psi1: Array,
    psi2: Array,
    p: Params,
    dx: float,
    dy: float,
) -> Array:
    psi1x, psi1y = dct_grad_fd(psi1, dx, dy)
    psi2x, psi2y = dct_grad_fd(psi2, dx, dy)
    elastic = psi1x**2 + psi1y**2 + psi2x**2 + psi2y**2
    return (
        p.lam * p.gamma * phi**2 * (1.0 - phi) ** 2 / (4.0 * p.h)
        + (p.lambdae * (1.0 - phi) + p.lambdab * phi) * elastic
    )


def compute_exact_mu(
    phi: Array,
    psi1: Array,
    psi2: Array,
    p: Params,
    dx: float,
    dy: float,
    K2n: Array,
) -> Array:
    src = compute_mu_source(phi, psi1, psi2, p, dx, dy)
    return idct2(p.lam * K2n * dct2(phi) + dct2(src))


def kappa_of_phi(phi: Array, p: Params) -> Array:
    return p.kper_t * phi + p.kper_b * (1.0 - phi)


def capillary_force_screened(
    phi: Array,
    mu: Array,
    p: Params,
    dx: float,
    dy: float,
) -> Tuple[Array, Array]:
    phix, phiy = grad_fd(phi, dx, dy)
    fx = p.capillary_coeff * mu * phix
    fy = p.capillary_coeff * mu * phiy
    return clip_field(fx, p.force_clip), clip_field(fy, p.force_clip)


def elastic_force_screened(
    phi: Array,
    psi1: Array,
    psi2: Array,
    p: Params,
    dx: float,
    dy: float,
) -> Tuple[Array, Array]:
    psi1x, psi1y = grad_fd(psi1, dx, dy)
    psi2x, psi2y = grad_fd(psi2, dx, dy)

    coeff = p.lambdae * (1.0 - phi) + p.lambdab * phi
    q1 = div_fd(coeff * psi1x, coeff * psi1y, dx, dy)
    q2 = div_fd(coeff * psi2x, coeff * psi2y, dx, dy)

    fx = -(psi1x * q1 + psi2x * q2)
    fy = -(psi1y * q1 + psi2y * q2)
    return clip_field(fx, p.elastic_force_clip), clip_field(fy, p.elastic_force_clip)


def viscosity_diffusion(u: Array, eta: Array, dx: float, dy: float) -> Array:
    ux, uy = grad_fd(u, dx, dy)
    return div_fd(eta * ux, eta * uy, dx, dy)


def advection_term(u: Array, v: Array, f: Array, dx: float, dy: float) -> Array:
    fx, fy = grad_fd(f, dx, dy)
    return u * fx + v * fy


def compute_ch_step_mu_from_exact_transition(
    phi_old: Array,
    phi_new: Array,
    psi1_old: Array,
    psi2_old: Array,
    p: Params,
    dx: float,
    dy: float,
    K2n: Array,
) -> Array:
    src_old = compute_mu_source(phi_old, psi1_old, psi2_old, p, dx, dy)
    return idct2(p.lam * K2n * dct2(phi_new) + dct2(src_old))


def discrete_mms_sources(
    exact_old: Dict[str, Array],
    exact_new: Dict[str, Array],
    p: Params,
    dt: float,
    dx: float,
    dy: float,
    KX: Array,
    KY: Array,
    K2: Array,
    K2n: Array,
) -> Dict[str, Array]:
    phi_old = exact_old["phi"]
    psi1_old = exact_old["psi1"]
    psi2_old = exact_old["psi2"]
    u1_old = exact_old["u1"]
    u2_old = exact_old["u2"]

    phi_new = exact_new["phi"]
    psi1_new = exact_new["psi1"]
    psi2_new = exact_new["psi2"]
    u1_new = exact_new["u1"]
    u2_new = exact_new["u2"]

    phix_old, phiy_old = dct_grad_fd(phi_old, dx, dy)
    adv_phi_old = u1_old * phix_old + u2_old * phiy_old
    src_old = compute_mu_source(phi_old, psi1_old, psi2_old, p, dx, dy)
    sh_old = dct2(src_old)
    phi_new_h = dct2(phi_new)
    denom_phi = 1.0 / dt + p.tau * p.lam * K2n**2
    rhs_phi_h = dt * (denom_phi * phi_new_h + p.tau * K2n * sh_old)
    rhs_phi = idct2(rhs_phi_h)
    phi_source = (rhs_phi - phi_old + dt * adv_phi_old) / dt

    psi1x_old, psi1y_old = dct_grad_fd(psi1_old, dx, dy)
    psi2x_old, psi2y_old = dct_grad_fd(psi2_old, dx, dy)
    adv1_old = u1_old * psi1x_old + u2_old * psi1y_old
    adv2_old = u1_old * psi2x_old + u2_old * psi2y_old

    coeff_old = p.lambdae * (1.0 - phi_old) + p.lambdab * phi_old
    diff1_old = dct_div_fd(coeff_old * psi1x_old, coeff_old * psi1y_old, dx, dy)
    diff2_old = dct_div_fd(coeff_old * psi2x_old, coeff_old * psi2y_old, dx, dy)

    denom_psi = 1.0 + dt * p.eps * 0.2 * K2n
    rhs1 = idct2(denom_psi * dct2(psi1_new))
    rhs2 = idct2(denom_psi * dct2(psi2_new))

    psi1_source = (rhs1 - psi1_old + dt * adv1_old - dt * p.eps * diff1_old) / dt
    psi2_source = (rhs2 - psi2_old + dt * adv2_old - dt * p.eps * diff2_old) / dt

    mu_for_u = compute_ch_step_mu_from_exact_transition(
        phi_old, phi_new, psi1_old, psi2_old, p, dx, dy, K2n
    )

    rho_re = p.rho * p.re
    eta_new = phi_new * p.etap + (1.0 - phi_new) * p.etaf

    advu1_old = advection_term(u1_old, u2_old, u1_old, dx, dy)
    advu2_old = advection_term(u1_old, u2_old, u2_old, dx, dy)

    visc1_new = viscosity_diffusion(u1_old, eta_new, dx, dy)
    visc2_new = viscosity_diffusion(u2_old, eta_new, dx, dy)

    fcap1_new, fcap2_new = capillary_force_screened(phi_new, mu_for_u, p, dx, dy)
    fel1_new, fel2_new = elastic_force_screened(phi_new, psi1_new, psi2_new, p, dx, dy)

    u1_source = (u1_new - u1_old) / dt + advu1_old - visc1_new / rho_re - (fcap1_new + fel1_new) / rho_re
    u2_source = (u2_new - u2_old) / dt + advu2_old - visc2_new / rho_re - (fcap2_new + fel2_new) / rho_re

    return {
        "phi": phi_source,
        "psi1": psi1_source,
        "psi2": psi2_source,
        "u1": u1_source,
        "u2": u2_source,
    }


def ch_step_neumann(
    phi_old: Array,
    u1: Array,
    u2: Array,
    psi1: Array,
    psi2: Array,
    p: Params,
    dt: float,
    dx: float,
    dy: float,
    K2n: Array,
    phi_nudge: Optional[Array] = None,
    phi_source: Optional[Array] = None,
) -> Tuple[Array, Array]:
    phix, phiy = dct_grad_fd(phi_old, dx, dy)
    adv = u1 * phix + u2 * phiy

    rhs = phi_old - dt * adv
    if phi_source is not None:
        rhs = rhs + dt * phi_source
    if phi_nudge is not None:
        rhs = rhs + dt * phi_nudge

    src = compute_mu_source(phi_old, psi1, psi2, p, dx, dy)
    rh = dct2(rhs)
    sh = dct2(src)

    denom = 1.0 / dt + p.tau * p.lam * K2n**2
    phi_h = (rh / dt - p.tau * K2n * sh) / denom

    phi_new = clip_phi(idct2(phi_h))
    mu_new = idct2(p.lam * K2n * dct2(phi_new) + sh)
    return phi_new, mu_new


def psi_step_neumann_coupled(
    psi1_old: Array,
    psi2_old: Array,
    u1: Array,
    u2: Array,
    phi: Array,
    p: Params,
    dt: float,
    dx: float,
    dy: float,
    K2n: Array,
    psi1_nudge: Optional[Array] = None,
    psi2_nudge: Optional[Array] = None,
    psi1_source: Optional[Array] = None,
    psi2_source: Optional[Array] = None,
) -> Tuple[Array, Array]:
    psi1x, psi1y = dct_grad_fd(psi1_old, dx, dy)
    psi2x, psi2y = dct_grad_fd(psi2_old, dx, dy)

    adv1 = u1 * psi1x + u2 * psi1y
    adv2 = u1 * psi2x + u2 * psi2y

    coeff = p.lambdae * (1.0 - phi) + p.lambdab * phi
    diff1 = dct_div_fd(coeff * psi1x, coeff * psi1y, dx, dy)
    diff2 = dct_div_fd(coeff * psi2x, coeff * psi2y, dx, dy)

    rhs1 = psi1_old - dt * adv1 + dt * p.eps * diff1
    rhs2 = psi2_old - dt * adv2 + dt * p.eps * diff2

    if psi1_source is not None:
        rhs1 = rhs1 + dt * psi1_source
    if psi2_source is not None:
        rhs2 = rhs2 + dt * psi2_source

    if psi1_nudge is not None:
        rhs1 = rhs1 + dt * clip_field(psi1_nudge, p.nudge_clip)
    if psi2_nudge is not None:
        rhs2 = rhs2 + dt * clip_field(psi2_nudge, p.nudge_clip)

    rh1 = dct2(rhs1)
    rh2 = dct2(rhs2)
    denom = 1.0 + dt * p.eps * 0.2 * K2n

    psi1_new = idct2(rh1 / denom)
    psi2_new = idct2(rh2 / denom)
    return psi1_new, psi2_new


def ns_step_fourier(
    u1_old: Array,
    u2_old: Array,
    phi: Array,
    mu: Array,
    psi1: Array,
    psi2: Array,
    p: Params,
    dt: float,
    dx: float,
    dy: float,
    KX: Array,
    KY: Array,
    K2: Array,
    filt: Array,
    u1_nudge: Optional[Array] = None,
    u2_nudge: Optional[Array] = None,
    u1_source: Optional[Array] = None,
    u2_source: Optional[Array] = None,
) -> Tuple[Array, Array, Array, Array]:
    rho_re = p.rho * p.re
    eta = phi * p.etap + (1.0 - phi) * p.etaf
    eta_bar = float(np.mean(eta))
    kappa = np.maximum(kappa_of_phi(phi, p), 1.0e-12)
    drag = eta * (1.0 - phi) / kappa
    drag_bar = float(np.mean(drag))

    adv1 = advection_term(u1_old, u2_old, u1_old, dx, dy)
    adv2 = advection_term(u1_old, u2_old, u2_old, dx, dy)

    visc1 = viscosity_diffusion(u1_old, eta, dx, dy)
    visc2 = viscosity_diffusion(u2_old, eta, dx, dy)

    fcap1, fcap2 = capillary_force_screened(phi, mu, p, dx, dy)
    fel1, fel2 = elastic_force_screened(phi, psi1, psi2, p, dx, dy)

    nudge1 = np.zeros_like(u1_old) if u1_nudge is None else clip_field(u1_nudge, p.nudge_clip)
    nudge2 = np.zeros_like(u2_old) if u2_nudge is None else clip_field(u2_nudge, p.nudge_clip)
    source1 = np.zeros_like(u1_old) if u1_source is None else u1_source
    source2 = np.zeros_like(u2_old) if u2_source is None else u2_source

    rhs1 = u1_old + dt * (
        -adv1
        + visc1 / rho_re
        + (fcap1 + fel1) / rho_re
        + source1
        + nudge1
    )
    rhs2 = u2_old + dt * (
        -adv2
        + visc2 / rho_re
        + (fcap2 + fel2) / rho_re
        + source2
        + nudge2
    )

    rhs1 = clip_field(rhs1, 1.0e3)
    rhs2 = clip_field(rhs2, 1.0e3)

    if not np.all(np.isfinite(rhs1)) or not np.all(np.isfinite(rhs2)):
        raise FloatingPointError("Non-finite velocity RHS before FFT")

    use_clean_mms_step = p.use_mms and p.disable_velocity_stabilization_for_mms
    if use_clean_mms_step:
        denom = np.ones_like(K2, dtype=float)
    else:
        denom = 1.0 + dt * (
            4.0 * eta_bar * K2 / rho_re
            + 2.0 * drag_bar / rho_re
            + p.extra_damping
        )

    u1h_star = fft2(rhs1) / denom
    u2h_star = fft2(rhs2) / denom

    divh_star = 1j * KX * u1h_star + 1j * KY * u2h_star
    div_star = ifft2(divh_star).real

    ph = np.zeros_like(u1h_star, dtype=np.complex128)
    mask = K2 > 0.0
    ph[mask] = -divh_star[mask] / (dt * K2[mask])
    ph[~mask] = 0.0

    u1h = u1h_star - dt * 1j * KX * ph
    u2h = u2h_star - dt * 1j * KY * ph

    apply_filter = not (p.use_mms and p.disable_filter_for_mms)
    if apply_filter and filt is not None:
        u1h *= filt
        u2h *= filt

    return ifft2(u1h).real, ifft2(u2h).real, ifft2(ph).real, div_star


def energy_and_errors(
    fields: Dict[str, Array],
    p: Params,
    dx: float,
    dy: float,
    KX: Array,
    KY: Array,
    div_star_u: Array,
    div_star_uda: Array,
    X: Array,
    Y: Array,
    time: float,
) -> Dict[str, float]:
    phi = fields["phi"]
    psi1 = fields["psi1"]
    psi2 = fields["psi2"]
    u1 = fields["u1"]
    u2 = fields["u2"]

    if p.use_mms:
        exact = exact_mms_fields(X, Y, time, p)
        phi_ref = exact["phi"]
        psi1_ref = exact["psi1"]
        psi2_ref = exact["psi2"]
        u1_ref = exact["u1"]
        u2_ref = exact["u2"]
    else:
        phi_ref = fields["phida"]
        psi1_ref = fields["psi1da"]
        psi2_ref = fields["psi2da"]
        u1_ref = fields["u1da"]
        u2_ref = fields["u2da"]

    phix, phiy = dct_grad_fd(phi, dx, dy)
    psi1x, psi1y = dct_grad_fd(psi1, dx, dy)
    psi2x, psi2y = dct_grad_fd(psi2, dx, dy)

    cell = dx * dy
    kinetic = 0.5 * p.rho * np.sum(u1**2 + u2**2) * cell
    grad_psi_sq = psi1x**2 + psi1y**2 + psi2x**2 + psi2y**2
    elastic_modulus = p.lambdae * (1.0 - phi) + p.lambdab * phi
    elastic = 0.5 * np.sum(elastic_modulus * grad_psi_sq) * cell
    mixing = np.sum(
        0.5 * p.lam * (phix**2 + phiy**2)
        + 4.0 * p.lam * p.gamma * phi**2 * (1.0 - phi) ** 2
    ) * cell

    e_phi = np.sqrt(np.sum((phi - phi_ref) ** 2) * cell)
    e_psi = np.sqrt(np.sum((psi1 - psi1_ref) ** 2 + (psi2 - psi2_ref) ** 2) * cell)
    e_u = np.sqrt(np.sum((u1 - u1_ref) ** 2 + (u2 - u2_ref) ** 2) * cell)

    div_u = divergence_fft(u1, u2, KX, KY)
    div_ref = divergence_fft(u1_ref, u2_ref, KX, KY)

    return {
        "kinetic": float(kinetic),
        "elastic": float(elastic),
        "mixing": float(mixing),
        "total": float(kinetic + elastic + mixing),
        "e_phi": float(e_phi),
        "e_psi": float(e_psi),
        "e_u": float(e_u),
        "div_star_u_l2": float(np.sqrt(np.sum(div_star_u**2) * cell)),
        "div_star_ref_l2": float(np.sqrt(np.sum(div_star_uda**2) * cell)),
        "div_u_l2": float(np.sqrt(np.sum(div_u**2) * cell)),
        "div_ref_l2": float(np.sqrt(np.sum(div_ref**2) * cell)),
        "max_div_star_u": float(np.max(np.abs(div_star_u))),
        "max_div_star_ref": float(np.max(np.abs(div_star_uda))),
        "max_div_u": float(np.max(np.abs(div_u))),
        "max_div_ref": float(np.max(np.abs(div_ref))),
    }


def write_vti_snapshot(
    outdir: Path,
    step: int,
    time: float,
    fields: Dict[str, Array],
    lx: float,
    ly: float,
) -> None:
    phi = np.asarray(fields["phi"], dtype=np.float64)
    ny, nx = phi.shape

    image = vtk.vtkImageData()
    image.SetDimensions(nx + 1, ny + 1, 1)
    image.SetOrigin(0.0, 0.0, 0.0)
    image.SetSpacing(lx / nx, ly / ny, 1.0)

    cell_data = image.GetCellData()

    for name, arr in fields.items():
        arr = np.asarray(arr)
        if arr.ndim != 2 or arr.shape != (ny, nx):
            continue
        vtk_arr = numpy_to_vtk(np.ascontiguousarray(arr.T).ravel(), deep=True)
        vtk_arr.SetName(name)
        cell_data.AddArray(vtk_arr)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(str(outdir / f"snapshot_{time:010.6f}.vti"))
    writer.SetInputData(image)
    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write VTI for step={step}, time={time:.6f}")


def save_snapshot(
    outdir: Path,
    step: int,
    time: float,
    dt: float,
    fields: Dict[str, Array],
    metrics: Dict[str, float],
    lx: float,
    ly: float,
) -> None:
    np.savez_compressed(
        outdir / f"snapshot_{time:010.6f}.npz",
        step=step,
        time=time,
        dt=dt,
        **fields,
        **metrics,
    )
    write_vti_snapshot(outdir, step, time, fields, lx, ly)


def run(p: Params) -> None:
    outdir = Path(p.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X, Y, dx, dy = make_grid(p)
    KX, KY, K2 = fourier_operators(p)
    _, _, K2n = neumann_operators(p)
    filt = fourier_spectral_filter(p)

    fields = initialize_fields(p, X, Y)
    time = 0.0
    step = 0

    with (outdir / "energy.csv").open("w", encoding="utf-8") as fe, (
        outdir / "error.csv"
    ).open("w", encoding="utf-8") as ff:
        fe.write("step,time,dt,kinetic,mixing,elastic,total\n")
        ff.write(
            "step,time,dt,e_phi,e_psi,e_u,"
            "div_star_u_l2,div_star_ref_l2,div_u_l2,div_ref_l2,"
            "max_div_star_u,max_div_star_ref,max_div_u,max_div_ref\n"
        )

        while time < p.t_final:
            dt = compute_dt(fields["u1"], fields["u2"], dx, dy, p)
            if time + dt > p.t_final:
                dt = p.t_final - time

            new_time = time + dt

            if p.use_mms and p.use_discrete_mms:
                exact_old = exact_mms_fields(X, Y, time, p)
                exact_new = exact_mms_fields(X, Y, new_time, p)
                sources = discrete_mms_sources(
                    exact_old, exact_new, p, dt, dx, dy, KX, KY, K2, K2n
                )
            elif p.use_mms:
                raise ValueError("This solver expects use_discrete_mms=True for MMS runs.")
            else:
                zeros = np.zeros_like(fields["phi"])
                sources = {"phi": zeros, "psi1": zeros, "psi2": zeros, "u1": zeros, "u2": zeros}

            phi = fields["phi"]
            psi1 = fields["psi1"]
            psi2 = fields["psi2"]
            u1 = fields["u1"]
            u2 = fields["u2"]

            phida = fields["phida"]
            psi1da = fields["psi1da"]
            psi2da = fields["psi2da"]
            u1da = fields["u1da"]
            u2da = fields["u2da"]

            phi_new, mu = ch_step_neumann(
                phi, u1, u2, psi1, psi2, p, dt, dx, dy, K2n, phi_source=sources["phi"]
            )

            phi_ref_proj = project_dct_low_modes(
                phi_new, p.cosine_cutoff_phi_x, p.cosine_cutoff_phi_y
            )
            phi_da_proj = project_dct_low_modes(
                phida, p.cosine_cutoff_phi_x, p.cosine_cutoff_phi_y
            )

            phida_new, muda = ch_step_neumann(
                phida,
                u1da,
                u2da,
                psi1da,
                psi2da,
                p,
                dt,
                dx,
                dy,
                K2n,
                phi_nudge=p.alphaphi * (phi_ref_proj - phi_da_proj),
                phi_source=sources["phi"] if p.use_mms else None,
            )

            psi1_new, psi2_new = psi_step_neumann_coupled(
                psi1,
                psi2,
                u1,
                u2,
                phi_new,
                p,
                dt,
                dx,
                dy,
                K2n,
                psi1_source=sources["psi1"],
                psi2_source=sources["psi2"],
            )

            psi1_ref_proj = project_dct_low_modes(
                psi1_new, p.cosine_cutoff_psi_x, p.cosine_cutoff_psi_y
            )
            psi2_ref_proj = project_dct_low_modes(
                psi2_new, p.cosine_cutoff_psi_x, p.cosine_cutoff_psi_y
            )
            psi1_da_proj = project_dct_low_modes(
                psi1da, p.cosine_cutoff_psi_x, p.cosine_cutoff_psi_y
            )
            psi2_da_proj = project_dct_low_modes(
                psi2da, p.cosine_cutoff_psi_x, p.cosine_cutoff_psi_y
            )

            psi1da_new, psi2da_new = psi_step_neumann_coupled(
                psi1da,
                psi2da,
                u1da,
                u2da,
                phida_new,
                p,
                dt,
                dx,
                dy,
                K2n,
                psi1_nudge=p.alphapsi * (psi1_ref_proj - psi1_da_proj),
                psi2_nudge=p.alphapsi * (psi2_ref_proj - psi2_da_proj),
                psi1_source=sources["psi1"] if p.use_mms else None,
                psi2_source=sources["psi2"] if p.use_mms else None,
            )

            u1_new, u2_new, p_new, div_star_u = ns_step_fourier(
                u1,
                u2,
                phi_new,
                mu,
                psi1_new,
                psi2_new,
                p,
                dt,
                dx,
                dy,
                KX,
                KY,
                K2,
                filt,
                u1_source=sources["u1"],
                u2_source=sources["u2"],
            )

            u1_ref_proj = project_fft_low_modes(
                u1_new, p.fourier_cutoff_u_x, p.fourier_cutoff_u_y
            )
            u2_ref_proj = project_fft_low_modes(
                u2_new, p.fourier_cutoff_u_x, p.fourier_cutoff_u_y
            )
            u1_da_proj = project_fft_low_modes(
                u1da, p.fourier_cutoff_u_x, p.fourier_cutoff_u_y
            )
            u2_da_proj = project_fft_low_modes(
                u2da, p.fourier_cutoff_u_x, p.fourier_cutoff_u_y
            )

            u1da_new, u2da_new, pda_new, div_star_uda = ns_step_fourier(
                u1da,
                u2da,
                phida_new,
                muda,
                psi1da_new,
                psi2da_new,
                p,
                dt,
                dx,
                dy,
                KX,
                KY,
                K2,
                filt,
                u1_nudge=p.alphau * (u1_ref_proj - u1_da_proj),
                u2_nudge=p.alphau * (u2_ref_proj - u2_da_proj),
                u1_source=sources["u1"] if p.use_mms else None,
                u2_source=sources["u2"] if p.use_mms else None,
            )

            fields["phi"] = phi_new
            fields["psi1"] = psi1_new
            fields["psi2"] = psi2_new
            fields["u1"] = u1_new
            fields["u2"] = u2_new
            fields["p"] = p_new

            fields["phida"] = phida_new
            fields["psi1da"] = psi1da_new
            fields["psi2da"] = psi2da_new
            fields["u1da"] = u1da_new
            fields["u2da"] = u2da_new
            fields["pda"] = pda_new

            metrics = energy_and_errors(
                fields, p, dx, dy, KX, KY, div_star_u, div_star_uda, X, Y, new_time
            )

            print(
                f"step={step:6d} t={new_time:10.6f} dt={dt:10.3e} "
                f"e_phi={metrics['e_phi']:.12e} e_psi={metrics['e_psi']:.12e} "
                f"e_u={metrics['e_u']:.12e} div_u={metrics['div_u_l2']:.12e}"
            )

            fe.write(
                f"{step},{new_time:.12e},{dt:.12e},"
                f"{metrics['kinetic']:.16e},{metrics['mixing']:.16e},"
                f"{metrics['elastic']:.16e},{metrics['total']:.16e}\n"
            )
            ff.write(
                f"{step},{new_time:.12e},{dt:.12e},"
                f"{metrics['e_phi']:.16e},{metrics['e_psi']:.16e},{metrics['e_u']:.16e},"
                f"{metrics['div_star_u_l2']:.16e},{metrics['div_star_ref_l2']:.16e},"
                f"{metrics['div_u_l2']:.16e},{metrics['div_ref_l2']:.16e},"
                f"{metrics['max_div_star_u']:.16e},{metrics['max_div_star_ref']:.16e},"
                f"{metrics['max_div_u']:.16e},{metrics['max_div_ref']:.16e}\n"
            )

            if step % p.save_every == 0:
                save_snapshot(outdir, step, new_time, dt, fields, metrics, p.lx, p.ly)

            if not all(np.isfinite(v) for v in metrics.values()):
                raise FloatingPointError(
                    f"Non-finite values detected at step {step}, t={new_time:.6e}"
                )

            time = new_time
            step += 1


def main() -> None:
    run(Params())


if __name__ == "__main__":
    main()