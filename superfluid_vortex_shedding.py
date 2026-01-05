
# gpe_2d_q1.py
# Audit-pure GPE 2D channel flow with Gaussian bluff obstacle, hard-wall channel (true mask),
# Split-step Fourier (Strang), JAX float32/complex64
# Upgrades:
# - Thomas-Fermi (TF) initial state option
# - True hard-wall mask (Dirichlet projection at each step)
# - Systematic convergence table with linear h-extrapolation (h -> 0)
# - Strouhal comparison vs reference range (e.g., 0.18–0.22)

import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import json

# dtypes
f32 = jnp.float32
c64 = jnp.complex64

@dataclass
class Params:
    g: float
    dt: float
    Lx: float
    Ly: float
    Nx: int
    Ny: int
    omega: float = 0.0
    # Obstacle (Gaussian barrier)
    obs_height: float = 1.5
    obs_sigma: float = 0.30
    obs_x0: float = 2.0
    obs_y0: float = 0.0
    # Channel walls
    wall_height: float = 8.0
    wall_thickness: float = 2.0
    # Audit cadence
    sample_stride: int = 80
    target_norm: float = 1.0
    # Initial condition
    init_profile: str = "TF"  # "TF" or "Gaussian"
    tf_mu: float = 1.0        # TF chemical potential baseline
    # Snapshots and animation
    snapshot_stride: int = 800
    max_snapshots: int = 6

def make_grid(params: Params):
    dx = jnp.asarray(params.Lx / params.Nx, dtype=f32)
    dy = jnp.asarray(params.Ly / params.Ny, dtype=f32)
    x = jnp.linspace(-params.Lx/2, params.Lx/2 - float(dx), params.Nx, dtype=f32)
    y = jnp.linspace(-params.Ly/2, params.Ly/2 - float(dy), params.Ny, dtype=f32)
    X, Y = jnp.meshgrid(x, y, indexing="xy")
    kx = (2.0 * jnp.pi * jnp.fft.fftfreq(params.Nx, d=float(dx))).astype(f32)
    ky = (2.0 * jnp.pi * jnp.fft.fftfreq(params.Ny, d=float(dy))).astype(f32)
    KX, KY = jnp.meshgrid(kx, ky, indexing="xy")
    K2 = KX**2 + KY**2
    return X, Y, KX, KY, K2, dx, dy

def potential_trap(X, Y, omega):
    omega = jnp.asarray(omega, dtype=f32)
    return jnp.asarray(0.5, dtype=f32) * (omega**2) * (X**2 + Y**2)

def potential_obstacle(X, Y, A, sigma, x0, y0):
    A = jnp.asarray(A, dtype=f32)
    sigma = jnp.asarray(sigma, dtype=f32)
    x0 = jnp.asarray(x0, dtype=f32)
    y0 = jnp.asarray(y0, dtype=f32)
    R2 = ((X - x0)**2 + (Y - y0)**2) / (jnp.asarray(2.0, dtype=f32) * sigma**2)
    return A * jnp.exp(-R2)

def potential_walls(Y, Ly, height, thickness):
    t = jnp.asarray(thickness, dtype=f32)
    h = jnp.asarray(height, dtype=f32)
    y_top = jnp.asarray(Ly/2, dtype=f32)
    y_bot = jnp.asarray(-Ly/2, dtype=f32)
    mask_top = (jnp.abs(Y - y_top) < t).astype(f32)
    mask_bot = (jnp.abs(Y - y_bot) < t).astype(f32)
    return h * (mask_top + mask_bot), (mask_top + mask_bot)

def total_potential(params: Params, X, Y):
    V = jnp.zeros_like(X, dtype=f32)
    if params.omega != 0.0:
        V = V + potential_trap(X, Y, params.omega)
    V = V + potential_obstacle(X, Y, params.obs_height, params.obs_sigma, params.obs_x0, params.obs_y0)
    Vwalls, _ = potential_walls(Y, params.Ly, params.wall_height, params.wall_thickness)
    V = V + Vwalls
    return V

def build_step(params: Params, X, Y, K2, fluid_mask):
    g = jnp.asarray(params.g, dtype=f32)
    dt = jnp.asarray(params.dt, dtype=f32)
    Vxy = total_potential(params, X, Y)
    half = jnp.asarray(0.5, dtype=f32)

    def phase_nl(psi): return jnp.exp(-1j * g * jnp.abs(psi)**2 * (dt * half)) * psi
    def phase_V(psi): return jnp.exp(-1j * Vxy * (dt * half)) * psi
    def kinetic(psi):
        psi_hat = jnp.fft.fftn(psi, axes=(0,1))
        return jnp.fft.ifftn(psi_hat * jnp.exp(-1j * K2 * (dt * half)), axes=(0,1))

    @jax.jit
    def step(psi):
        psi = phase_V(psi)
        psi = phase_nl(psi)
        psi = kinetic(psi)
        psi = phase_nl(psi)
        psi = phase_V(psi)
        # True hard-wall mask: project psi to fluid region (Dirichlet psi=0 at walls)
        psi = psi * fluid_mask
        return psi

    return step, Vxy

def norm(psi, dx, dy): return jnp.sum(jnp.abs(psi)**2).astype(f32) * dx * dy

def grad_spectral(psi, KX, KY):
    psi_hat = jnp.fft.fftn(psi, axes=(0,1))
    dpsi_dx = jnp.fft.ifftn(1j * KX * psi_hat, axes=(0,1))
    dpsi_dy = jnp.fft.ifftn(1j * KY * psi_hat, axes=(0,1))
    return dpsi_dx, dpsi_dy

def current_density(psi, KX, KY):
    dpsi_dx, dpsi_dy = grad_spectral(psi, KX, KY)
    jx = jnp.imag(jnp.conj(psi) * dpsi_dx).astype(f32)
    jy = jnp.imag(jnp.conj(psi) * dpsi_dy).astype(f32)
    return jx, jy

def vorticity_spectral(jx, jy, KX, KY):
    jy_hat = jnp.fft.fftn(jy, axes=(0,1))
    jx_hat = jnp.fft.fftn(jx, axes=(0,1))
    d_jy_dx = jnp.fft.ifftn(1j * KX * jy_hat, axes=(0,1))
    d_jx_dy = jnp.fft.ifftn(1j * KY * jx_hat, axes=(0,1))
    return (d_jy_dx - d_jx_dy).astype(f32)

def obstacle_force_analytic(psi, X, Y, dx, dy, params: Params):
    Vobs = potential_obstacle(X, Y, params.obs_height, params.obs_sigma, params.obs_x0, params.obs_y0)
    dVdx = -((X - jnp.asarray(params.obs_x0, dtype=f32)) / (jnp.asarray(params.obs_sigma, dtype=f32)**2)) * Vobs
    rho = jnp.abs(psi)**2
    Fx = -jnp.sum(rho * dVdx).astype(f32) * dx * dy
    return Fx

def transmission_fraction_x(psi, X, dx, dy, x0):
    mask_right = (X > jnp.asarray(x0, dtype=f32)).astype(f32)
    mass_right = jnp.sum(jnp.abs(psi)**2 * mask_right).astype(f32) * dx * dy
    mass_total = jnp.sum(jnp.abs(psi)**2).astype(f32) * dx * dy
    return jnp.where(mass_total > 0, mass_right / mass_total, jnp.asarray(0.0, dtype=f32))

def phase_winding_count(psi):
    phase = jnp.angle(psi)
    dphi_x = phase[:,1:] - phase[:,:-1]
    dphi_y = phase[1:,:] - phase[:-1,:]
    dphi_x = jnp.arctan2(jnp.sin(dphi_x), jnp.cos(dphi_x))
    dphi_y = jnp.arctan2(jnp.sin(dphi_y), jnp.cos(dphi_y))
    w = dphi_x[:-1,:] + dphi_y[:,1:] - dphi_x[1:,:] - dphi_y[:,:-1]
    wn = jnp.round(w / (2*jnp.pi))
    return float(jnp.sum(jnp.abs(wn)))

def energy_total(params: Params, psi, X, Y, KX, KY, dx, dy, Vxy):
    dpsi_dx, dpsi_dy = grad_spectral(psi, KX, KY)
    kinetic = jnp.sum(jnp.asarray(0.5, dtype=f32) * (jnp.abs(dpsi_dx)**2 + jnp.abs(dpsi_dy)**2)).astype(f32) * dx * dy
    interaction = jnp.sum(jnp.asarray(0.5, dtype=f32) * jnp.asarray(params.g, dtype=f32) * (jnp.abs(psi)**4)).astype(f32) * dx * dy
    pot = jnp.sum(Vxy * (jnp.abs(psi)**2)).astype(f32) * dx * dy
    return kinetic + interaction + pot

def hann_window(n):
    n = int(n)
    return jnp.asarray(0.5, dtype=f32) - jnp.asarray(0.5, dtype=f32) * jnp.cos(2.0 * jnp.pi * jnp.arange(n, dtype=f32) / jnp.asarray(n, dtype=f32))

def fft_peak_with_fwhm(Fx_hist, dt, pad_factor=8):
    Fx_hist = jnp.asarray(Fx_hist)
    if Fx_hist.size < 32:
        return 0.0, 0.0, 0.0, 0.0
    y = Fx_hist - jnp.mean(Fx_hist)
    n = int(y.shape[0])
    w = hann_window(n)
    yw = y * w
    Npad = int(n * pad_factor)
    yw_pad = jnp.pad(yw, (0, max(Npad - n, 0)))
    Y = jnp.fft.fft(yw_pad)
    power = jnp.abs(Y)**2
    f = jnp.fft.fftfreq(yw_pad.shape[0], d=float(dt))
    pos = f > 0
    fpos = f[pos]; ppos = power[pos]
    idx = int(jnp.argmax(ppos))
    f0 = float(fpos[idx]); p0 = float(ppos[idx])
    half = p0/2.0
    li = idx
    while li > 0 and float(ppos[li]) > half: li -= 1
    ri = idx
    while ri < (ppos.shape[0]-1) and float(ppos[ri]) > half: ri += 1
    fwhm = float(fpos[ri] - fpos[li]) if (ri > li) else 0.0
    Q_fwhm = (f0 / fwhm) if fwhm > 1e-9 else 0.0
    Q_med = float(p0 / (float(jnp.median(ppos)) + 1e-12))
    return f0, p0, Q_fwhm, Q_med

def initial_field(params: Params, X, Y, k0x, fluid_mask):
    if params.init_profile == "TF":
        # Thomas-Fermi using walls potential as background; obstacle excluded to avoid deep depletion
        Vwalls, _ = potential_walls(Y, params.Ly, params.wall_height, params.wall_thickness)
        mu = jnp.asarray(params.tf_mu, dtype=f32)
        g = jnp.asarray(params.g, dtype=f32)
        n_tf = jnp.maximum((mu - Vwalls) / jnp.maximum(g, jnp.asarray(1e-6, dtype=f32)), jnp.asarray(0.0, dtype=f32))
        psi0 = jnp.sqrt(n_tf).astype(c64)
    else:
        sigma_y = jnp.asarray(params.Ly / 4.0, dtype=f32)
        psi0 = jnp.exp(-jnp.asarray(0.5, dtype=f32) * ((Y - jnp.asarray(params.obs_y0, dtype=f32))**2) / sigma_y**2).astype(c64)
    psi0 = psi0 * jnp.exp(1j * (jnp.asarray(k0x, dtype=f32) * X))
    # Apply hard-wall mask to initial state
    psi0 = psi0 * fluid_mask
    return psi0

def make_hard_wall_mask(params: Params, Y):
    _, wall_mask = potential_walls(Y, params.Ly, params.wall_height, params.wall_thickness)
    # fluid region: 1 inside channel away from walls; 0 on walls
    fluid_mask = (1.0 - jnp.clip(wall_mask, 0.0, 1.0)).astype(f32)
    return fluid_mask

def simulate_once(params: Params, k0x: float, A: float, n_steps: int):
    X, Y, KX, KY, K2, dx, dy = make_grid(params)
    fluid_mask = make_hard_wall_mask(params, Y)
    local = Params(**{**params.__dict__, "obs_height": float(A)})
    step, Vxy = build_step(local, X, Y, K2, fluid_mask)

    psi = initial_field(local, X, Y, k0x, fluid_mask)
    psi = psi * jnp.sqrt(jnp.asarray(local.target_norm, dtype=f32) / jnp.maximum(norm(psi, dx, dy), jnp.asarray(1e-12, dtype=f32)))

    for _ in range(200):
        psi = step(psi)

    Fx_hist, T_hist, vort_hist, Etot_hist, N_hist = [], [], [], [], []
    snapshots = []
    snap_count = 0

    for t in range(n_steps):
        psi = step(psi)
        if (t % params.sample_stride) == 0:
            jx, jy = current_density(psi, KX, KY)
            Fx = obstacle_force_analytic(psi, X, Y, dx, dy, local)
            Tfrac = transmission_fraction_x(psi, X, dx, dy, local.obs_x0)
            vort_count = phase_winding_count(psi)
            Etot = energy_total(local, psi, X, Y, KX, KY, dx, dy, Vxy)
            Fx_hist.append(float(Fx)); T_hist.append(float(Tfrac)); vort_hist.append(float(vort_count))
            Etot_hist.append(float(Etot)); N_hist.append(float(norm(psi, dx, dy)))

        if (t % params.snapshot_stride) == 0 and snap_count < params.max_snapshots:
            jx, jy = current_density(psi, KX, KY)
            vort = vorticity_spectral(jx, jy, KX, KY)
            dens = jnp.abs(psi)**2
            phase = jnp.angle(psi)
            snapshots.append({"step": t, "dens": np.array(dens), "phase": np.array(phase), "vort": np.array(vort)})
            snap_count += 1

    Fx_hist = np.array(Fx_hist, dtype=np.float32)
    T_hist = np.array(T_hist, dtype=np.float32)
    vort_hist = np.array(vort_hist, dtype=np.float32)
    Etot_hist = np.array(Etot_hist, dtype=np.float32)
    N_hist = np.array(N_hist, dtype=np.float32)

    dt_samp = float(params.dt) * params.sample_stride
    f_peak, p_peak, Q_fwhm, Q_med = fft_peak_with_fwhm(jnp.asarray(Fx_hist), dt_samp, pad_factor=8)

    D_eff = 2.0 * float(params.obs_sigma)
    U = float(k0x)
    St = (f_peak * D_eff / U) if (U > 1e-12 and f_peak > 0) else 0.0

    return {
        "k0x": k0x, "A": A,
        "mean_Fx": float(np.mean(Fx_hist)) if Fx_hist.size else 0.0,
        "mean_Tx": float(np.mean(T_hist)) if T_hist.size else 0.0,
        "f_peak": f_peak, "p_peak": p_peak, "Q_fwhm": Q_fwhm, "Q_med": Q_med, "St": St,
        "vortex_count_est": float(np.mean(vort_hist[-5:])) if vort_hist.size >= 5 else float(np.mean(vort_hist)) if vort_hist.size else 0.0,
        "Fx_hist": Fx_hist.tolist(),
        "T_hist": T_hist.tolist(),
        "vort_hist": vort_hist.tolist(),
        "energy_total_hist": Etot_hist.tolist(),
        "norm_hist": N_hist.tolist(),
        "snapshots": snapshots
    }

def convergence_extrapolation(h_vals, st_vals):
    # Linear least-squares fit: St(h) = a*h + b ; extrapolate b = St(h->0)
    h = np.asarray(h_vals, dtype=np.float64)
    st = np.asarray(st_vals, dtype=np.float64)
    A = np.vstack([h, np.ones_like(h)]).T
    a, b = np.linalg.lstsq(A, st, rcond=None)[0]
    # simple metrics
    residuals = st - (a*h + b)
    rmse = float(np.sqrt(np.mean(residuals**2)))
    return {"a": float(a), "St_extrap": float(b), "rmse": rmse}

def run_convergence_table(base_params: Params, k0x: float, A: float, grids_dt, n_steps, st_ref_range=(0.18, 0.22)):
    table = []
    h_vals = []
    st_vals = []

    for (Nx, Ny, dt) in grids_dt:
        p = Params(
            g=base_params.g, dt=float(dt), Lx=base_params.Lx, Ly=base_params.Ly,
            Nx=int(Nx), Ny=int(Ny), omega=base_params.omega,
            obs_height=A, obs_sigma=base_params.obs_sigma, obs_x0=base_params.obs_x0, obs_y0=base_params.obs_y0,
            wall_height=base_params.wall_height, wall_thickness=base_params.wall_thickness,
            sample_stride=base_params.sample_stride, target_norm=base_params.target_norm,
            init_profile=base_params.init_profile, tf_mu=base_params.tf_mu,
            snapshot_stride=base_params.snapshot_stride, max_snapshots=base_params.max_snapshots
        )
        X, Y, _, _, _, dx, dy = make_grid(p)
        h = float(max(dx, dy))
        res = simulate_once(p, k0x=k0x, A=A, n_steps=n_steps)
        table.append({"Nx": Nx, "Ny": Ny, "dt": dt, "h": h, "f_peak": res["f_peak"], "St": res["St"], "Q_fwhm": res["Q_fwhm"]})
        h_vals.append(h); st_vals.append(res["St"])

    extr = convergence_extrapolation(h_vals, st_vals)
    St_ex = extr["St_extrap"]
    ref_low, ref_high = st_ref_range
    # comparison to literature range
    within = (St_ex >= ref_low) and (St_ex <= ref_high)
    err_to_mid = abs(St_ex - (0.5*(ref_low+ref_high)))
    summary = {
        "extrapolated_St": St_ex,
        "rmse": extr["rmse"],
        "slope_a": extr["a"],
        "ref_range": list(st_ref_range),
        "within_ref_range": bool(within),
        "abs_error_to_mid": float(err_to_mid)
    }
    return table, summary

# --------------------
# Main
# --------------------
if __name__ == "__main__":
    # Baseline parameters (TF initial, true hard-wall)
    params = Params(
        g=1.0, dt=2e-4, Lx=40.0, Ly=20.0, Nx=512, Ny=384,
        omega=0.0,
        obs_height=1.5, obs_sigma=0.30, obs_x0=2.0, obs_y0=0.0,
        wall_height=8.0, wall_thickness=2.0,
        sample_stride=80, target_norm=1.0,
        init_profile="TF", tf_mu=1.0,
        snapshot_stride=800, max_snapshots=6
    )

    # Target shedding case (supercritical window)
    K_VALUES = [0.6, 0.7, 0.8]
    A_VALUES = [1.3, 1.5, 1.7]
    N_STEPS = 24000

    # Sweep and compact summary
    results = []
    for A in A_VALUES:
        for k0 in K_VALUES:
            print(f"Running k0x={k0}, A={A} ...")
            r = simulate_once(params, k0x=k0, A=A, n_steps=N_STEPS)
            results.append(r)
            print(f"  Fx={r['mean_Fx']:.4g}, Tx={r['mean_Tx']:.3f}, f={r['f_peak']:.4f}, Q={r['Q_fwhm']:.2f}, St={r['St']:.3f}, vort={r['vortex_count_est']:.2f}")

    # Convergence table and extrapolation on the strongest shedding case (k0x=0.7, A=1.7)
    grids_dt = [
        (384, 288, 2e-4),
        (512, 384, 2e-4),
        (768, 512, 1e-4),
    ]
    conv_table, conv_summary = run_convergence_table(params, k0x=0.7, A=1.7, grids_dt=grids_dt, n_steps=16000, st_ref_range=(0.18, 0.22))

    # Print convergence results
    print("\nConvergence table (St vs h):")
    for row in conv_table:
        print(f"  Nx={row['Nx']} Ny={row['Ny']} dt={row['dt']} h={row['h']:.4f} | f={row['f_peak']:.4f} St={row['St']:.4f} Q={row['Q_fwhm']:.1f}")

    print("\nExtrapolation summary (linear in h):")
    print(json.dumps(conv_summary, indent=2))

    # Optional: quick convergence plot
    hs = [row["h"] for row in conv_table]
    sts = [row["St"] for row in conv_table]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(hs, sts, "o-", label="St(h)")
    ax.axhspan(conv_summary["ref_range"][0], conv_summary["ref_range"][1], color="orange", alpha=0.2, label="Ref range")
    ax.axhline(conv_summary["extrapolated_St"], color="tab:red", linestyle="--", label=f"Extrapolated St={conv_summary['extrapolated_St']:.3f}")
    ax.set_xlabel("Grid spacing h=max(dx,dy)")
    ax.set_ylabel("Strouhal St")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

    # JSON dump for repo audits
    print("\nJSON summary (compact):")
    print(json.dumps({
        "params": {
            "g": params.g, "dt": params.dt, "Lx": params.Lx, "Ly": params.Ly,
            "Nx": params.Nx, "Ny": params.Ny, "obs_sigma": params.obs_sigma,
            "wall_thickness": params.wall_thickness, "init_profile": params.init_profile
        },
        "results": [
            {k: v for k, v in r.items() if k in ("k0x","A","mean_Fx","mean_Tx","f_peak","Q_fwhm","St","vortex_count_est")}
            for r in results
        ],
        "convergence_table": conv_table,
        "convergence_summary": conv_summary
    }, indent=2))
