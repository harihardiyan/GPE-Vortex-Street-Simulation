
# gpe_2d_q1_monolith_jax.py
# Monolithic JAX pipeline: simulation (soft-wall + sponge, ITP relaxation, adaptive sampling),
# upstream-thresholded vortex counting, robust FFT, norm audit;
# plus full post-processing (St vs Mach from measured upstream density, Fx(t) + spectrum,
# 2D density/phase snapshots, and energy audit).
#
# JAX float32/complex64, split-step Fourier (Strang). Matplotlib for plotting.

import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
from dataclasses import dataclass
import numpy as np
import json
import matplotlib.pyplot as plt

f32 = jnp.float32
c64 = jnp.complex64

# =========================
# Parameters
# =========================
@dataclass
class Params:
    # Physics
    g: float
    dt: float
    Lx: float
    Ly: float
    Nx: int
    Ny: int
    # Obstacle
    obs_height: float = 5.0
    obs_sigma: float = 0.50
    obs_x0: float = 2.0
    obs_y0: float = 0.0
    # Soft walls
    wall_V0: float = 10.0
    wall_width: float = 2.2
    # Sponge layers (absorbing zones) near x-edges
    sponge_V0: float = 3.0
    sponge_width: float = 4.0
    sponge_power: float = 2.0
    # Sampling
    sample_stride: int = 10
    target_norm: float = 1.0
    # Initial condition
    init_profile: str = "TF"
    tf_mu: float = 1.2
    # ITP relaxation
    itp_steps: int = 4000
    itp_dt: float = 2e-4
    # FFT options
    pad_factor: int = 16
    window_type: str = "tukey"  # "tukey" or "hann"
    tukey_alpha: float = 0.25
    # Vortex counting threshold (relative to upstream density)
    vort_density_frac: float = 0.10
    # Snapshots (post-processing)
    snap_stride: int = 2000
    max_snaps: int = 6

# =========================
# Grid and potentials
# =========================
def make_grid(p: Params):
    dx = jnp.asarray(p.Lx / p.Nx, dtype=f32)
    dy = jnp.asarray(p.Ly / p.Ny, dtype=f32)
    x = jnp.linspace(-p.Lx/2, p.Lx/2 - float(dx), p.Nx, dtype=f32)
    y = jnp.linspace(-p.Ly/2, p.Ly/2 - float(dy), p.Ny, dtype=f32)
    X, Y = jnp.meshgrid(x, y, indexing="xy")
    kx = (2.0 * jnp.pi * jnp.fft.fftfreq(p.Nx, d=float(dx))).astype(f32)
    ky = (2.0 * jnp.pi * jnp.fft.fftfreq(p.Ny, d=float(dy))).astype(f32)
    KX, KY = jnp.meshgrid(kx, ky, indexing="xy")
    K2 = KX**2 + KY**2
    return X, Y, KX, KY, K2, dx, dy

def potential_obstacle(X, Y, A, sigma, x0, y0):
    A = jnp.asarray(A, dtype=f32); sigma = jnp.asarray(sigma, dtype=f32)
    x0 = jnp.asarray(x0, dtype=f32); y0 = jnp.asarray(y0, dtype=f32)
    R2 = ((X - x0)**2 + (Y - y0)**2) / (jnp.asarray(2.0, dtype=f32) * sigma**2)
    return A * jnp.exp(-R2)

def potential_soft_walls(Y, Ly, V0, w):
    y_top = jnp.asarray(Ly/2, dtype=f32)
    y_bot = jnp.asarray(-Ly/2, dtype=f32)
    w = jnp.asarray(w, dtype=f32)
    return jnp.asarray(V0, dtype=f32) * (jnp.exp(-((Y - y_top)**2) / (w**2)) + jnp.exp(-((Y - y_bot)**2) / (w**2)))

def potential_sponge(X, Lx, V0, w, power):
    w = jnp.asarray(w, dtype=f32); power = jnp.asarray(power, dtype=f32)
    x_left = jnp.asarray(-Lx/2, dtype=f32); x_right = jnp.asarray(Lx/2, dtype=f32)
    left = jnp.clip((X - x_left) / w, 0.0, 1.0)
    right = jnp.clip((x_right - X) / w, 0.0, 1.0)
    sponge = (left**power + right**power)
    return jnp.asarray(V0, dtype=f32) * sponge

def total_potential(p: Params, X, Y, A_override=None):
    Ause = p.obs_height if A_override is None else A_override
    Vobs = potential_obstacle(X, Y, Ause, p.obs_sigma, p.obs_x0, p.obs_y0)
    Vwalls = potential_soft_walls(Y, p.Ly, p.wall_V0, p.wall_width)
    Vsponge = potential_sponge(X, p.Lx, p.sponge_V0, p.sponge_width, p.sponge_power)
    return Vobs + Vwalls + Vsponge

# =========================
# Split-step operators
# =========================
def build_stepper(p: Params, X, Y, K2, A_override=None):
    g = jnp.asarray(p.g, dtype=f32)
    dt = jnp.asarray(p.dt, dtype=f32)
    V = total_potential(p, X, Y, A_override)
    half = jnp.asarray(0.5, dtype=f32)

    def phase_nl(psi): return jnp.exp(-1j * g * jnp.abs(psi)**2 * (dt * half)) * psi
    def phase_V(psi):  return jnp.exp(-1j * V * (dt * half)) * psi
    def kinetic(psi):
        psi_hat = jnp.fft.fftn(psi, axes=(0,1))
        return jnp.fft.ifftn(psi_hat * jnp.exp(-1j * K2 * (dt * half)), axes=(0,1))

    @jax.jit
    def step(psi):
        psi = phase_V(psi); psi = phase_nl(psi)
        psi = kinetic(psi)
        psi = phase_nl(psi); psi = phase_V(psi)
        return psi
    return step

def build_itp_stepper(p: Params, X, Y, K2):
    g = jnp.asarray(p.g, dtype=f32)
    dt_itp = jnp.asarray(p.itp_dt, dtype=f32)
    V = total_potential(p, X, Y)
    half = jnp.asarray(0.5, dtype=f32)

    def damp_nl(psi): return jnp.exp(- g * jnp.abs(psi)**2 * (dt_itp * half)) * psi
    def damp_V(psi):  return jnp.exp(- V * (dt_itp * half)) * psi
    def kinetic_damp(psi):
        psi_hat = jnp.fft.fftn(psi, axes=(0,1))
        return jnp.fft.ifftn(psi_hat * jnp.exp(- K2 * (dt_itp * half)), axes=(0,1))

    @jax.jit
    def step_itp(psi):
        psi = damp_V(psi); psi = damp_nl(psi)
        psi = kinetic_damp(psi)
        psi = damp_nl(psi); psi = damp_V(psi)
        return psi
    return step_itp

# =========================
# Diagnostics and FFT
# =========================
def norm(psi, dx, dy): return jnp.sum(jnp.abs(psi)**2).astype(f32) * dx * dy

def obstacle_force_x(psi, X, Y, dx, dy, p: Params, A_override=None):
    Ause = p.obs_height if A_override is None else A_override
    Vobs = potential_obstacle(X, Y, Ause, p.obs_sigma, p.obs_x0, p.obs_y0)
    dVdx = -((X - jnp.asarray(p.obs_x0, dtype=f32)) / (jnp.asarray(p.obs_sigma, dtype=f32)**2)) * Vobs
    rho = jnp.abs(psi)**2
    return -jnp.sum(rho * dVdx).astype(f32) * dx * dy

def transmission_fraction_x(psi, X, dx, dy, x0):
    mask_right = (X > jnp.asarray(x0, dtype=f32)).astype(f32)
    mass_right = jnp.sum(jnp.abs(psi)**2 * mask_right).astype(f32) * dx * dy
    mass_total = jnp.sum(jnp.abs(psi)**2).astype(f32) * dx * dy
    return jnp.where(mass_total > 0, mass_right / mass_total, jnp.asarray(0.0, dtype=f32))

def tukey_window(n, alpha):
    n = int(n); alpha = float(alpha)
    k = jnp.arange(n, dtype=f32)
    w = jnp.ones(n, dtype=f32)
    idx_rise = k < (alpha*(n-1)/2)
    w = w.at[idx_rise].set(0.5 * (1 + jnp.cos(jnp.pi * ((2*k[idx_rise]/(alpha*(n-1))) - 1))))
    idx_fall = k > ((n-1)*(1 - alpha/2))
    w = w.at[idx_fall].set(0.5 * (1 + jnp.cos(jnp.pi * ((2*k[idx_fall]/(alpha*(n-1))) - (2/alpha) + 1))))
    return w

def make_window(n, kind, alpha):
    if kind == "hann":
        return jnp.hanning(int(n)).astype(f32)
    return tukey_window(int(n), alpha)

def fft_drag(Fx_hist, dt, pad_factor=16, window_type="tukey", alpha=0.25):
    y = Fx_hist - jnp.mean(Fx_hist)
    n = int(y.shape[0])
    if n < 64:
        return 0.0, 0.0
    w = make_window(n, window_type, alpha)
    yw = y * w
    pad_n = int(n * pad_factor)
    Y = jnp.fft.fft(yw, pad_n)
    f = jnp.fft.fftfreq(pad_n, d=float(dt))
    pos = f > 0
    fpos = f[pos]; ppos = jnp.abs(Y[pos])**2
    idx = int(jnp.argmax(ppos))
    return float(fpos[idx]), float(ppos[idx])

# =========================
# Initial state and vortex counting
# =========================
def initial_field_TF(p: Params, X, Y):
    Vwalls = potential_soft_walls(Y, p.Ly, p.wall_V0, p.wall_width)
    mu = jnp.asarray(p.tf_mu, dtype=f32)
    g = jnp.asarray(p.g, dtype=f32)
    n_tf = jnp.maximum((mu - Vwalls) / jnp.maximum(g, jnp.asarray(1e-6, dtype=f32)), jnp.asarray(0.0, dtype=f32))
    return jnp.sqrt(n_tf).astype(c64)

def imprint_flow(psi, X, k0x):
    return psi * jnp.exp(1j * (jnp.asarray(k0x, dtype=f32) * X))

def phase_winding_count_thresholded(psi, density_threshold):
    dens = jnp.abs(psi)**2
    mask = (dens >= density_threshold).astype(f32)
    phase = jnp.angle(psi)
    m_x = mask[:,1:] * mask[:,:-1]
    m_y = mask[1:,:] * mask[:-1,:]
    dphi_x = phase[:,1:] - phase[:,:-1]
    dphi_y = phase[1:,:] - phase[:-1,:]
    dphi_x = jnp.arctan2(jnp.sin(dphi_x), jnp.cos(dphi_x))
    dphi_y = jnp.arctan2(jnp.sin(dphi_y), jnp.cos(dphi_y))
    m_plaq = m_x[:-1,:] * m_y[:,1:] * m_x[1:,:] * m_y[:,:-1]
    w = dphi_x[:-1,:] + dphi_y[:,1:] - dphi_x[1:,:] - dphi_y[:,:-1]
    wn = jnp.round(w / (2*jnp.pi))
    return float(jnp.sum(jnp.abs(wn) * m_plaq))

# =========================
# Adaptive controls for speed (optional)
# =========================
def adaptive_params_for_speed(p: Params, k0x: float):
    p_mod = Params(**p.__dict__)
    if k0x >= 1.5:
        p_mod.sample_stride = 2
        p_mod.dt = 5e-5
        p_mod.pad_factor = 32
        p_mod.tukey_alpha = 0.20
    return p_mod

def upstream_density(dens, X, x_ref, sigma_ref):
    return float(jnp.median(dens[X < (x_ref - 2.0 * sigma_ref)]))

# =========================
# Energy audit (JAX)
# =========================
def energy_total(g, psi, X, Y, dx, dy, V, KX, KY):
    psi_hat = jnp.fft.fftn(psi, axes=(0,1))
    dpsi_dx = jnp.fft.ifftn(1j * KX * psi_hat, axes=(0,1))
    dpsi_dy = jnp.fft.ifftn(1j * KY * psi_hat, axes=(0,1))
    kinetic = jnp.asarray(0.5, dtype=f32) * jnp.sum(jnp.abs(dpsi_dx)**2 + jnp.abs(dpsi_dy)**2).astype(f32) * dx * dy
    interaction = jnp.asarray(0.5, dtype=f32) * jnp.asarray(g, dtype=f32) * jnp.sum(jnp.abs(psi)**4).astype(f32) * dx * dy
    pot = jnp.sum(V * jnp.abs(psi)**2).astype(f32) * dx * dy
    return float(kinetic + interaction + pot)

# =========================
# Single simulation with audits (+optional snapshots and time series)
# =========================
def simulate_once(p: Params, k0x: float, A: float, n_steps: int, record_timeseries=False):
    p_use = adaptive_params_for_speed(p, k0x)
    X, Y, KX, KY, K2, dx, dy = make_grid(p_use)

    # ITP relaxation
    psi = initial_field_TF(p_use, X, Y)
    step_itp = build_itp_stepper(p_use, X, Y, K2)
    for s in range(p_use.itp_steps):
        psi = step_itp(psi)
        if (s % 100) == 0:
            psi = psi * jnp.sqrt(jnp.asarray(p_use.target_norm, dtype=f32) /
                                 jnp.maximum(norm(psi, dx, dy), jnp.asarray(1e-12, dtype=f32)))

    # Flow imprint and normalization
    psi = imprint_flow(psi, X, k0x)
    psi = psi * jnp.sqrt(jnp.asarray(p_use.target_norm, dtype=f32) /
                         jnp.maximum(norm(psi, dx, dy), jnp.asarray(1e-12, dtype=f32)))

    step_rt = build_stepper(p_use, X, Y, K2, A_override=A)

    # Warm-up compile
    for _ in range(200): psi = step_rt(psi)

    Fx_hist = []
    T_hist = []
    vort_hist = []
    N_hist = []
    Et_hist = []
    snapshots = []

    x_ref = float(p_use.obs_x0); sigma_ref = float(p_use.obs_sigma)
    dt_samp = float(p_use.dt) * p_use.sample_stride

    for t in range(n_steps):
        psi = step_rt(psi)

        # Optional snapshots
        if record_timeseries and (t % p_use.snap_stride == 0) and (len(snapshots) < p_use.max_snaps):
            dens_np = np.array(jnp.abs(psi)**2, dtype=np.float32)
            phase_np = np.array(jnp.angle(psi), dtype=np.float32)
            snapshots.append({"t": t, "dens": dens_np, "phase": phase_np})

        if (t % p_use.sample_stride) == 0:
            # Norm and energy
            N = float(norm(psi, dx, dy)); N_hist.append(N)
            V = total_potential(p_use, X, Y, A_override=A)
            Et = energy_total(p_use.g, psi, X, Y, dx, dy, V, KX, KY); Et_hist.append(Et)
            if abs(N - p_use.target_norm) > 0.01 * p_use.target_norm:
                raise RuntimeError(f"Norm drift detected: N={N:.6f} at step {t}")

            Fx = obstacle_force_x(psi, X, Y, dx, dy, p_use, A_override=A)
            Tx = transmission_fraction_x(psi, X, dx, dy, jnp.asarray(p_use.obs_x0, dtype=f32))

            dens = jnp.abs(psi)**2
            n_up = upstream_density(dens, X, x_ref, sigma_ref)
            thr = jnp.asarray(p_use.vort_density_frac * n_up, dtype=f32)
            vort = phase_winding_count_thresholded(psi, thr)

            Fx_hist.append(float(Fx)); T_hist.append(float(Tx)); vort_hist.append(float(vort))

    Fx_hist_j = jnp.asarray(Fx_hist, dtype=f32)
    f_peak, _ = fft_drag(Fx_hist_j, dt_samp, pad_factor=p_use.pad_factor, window_type=p_use.window_type, alpha=p_use.tukey_alpha)

    D_eff = 2.0 * float(p_use.obs_sigma)
    U = float(k0x)
    St = (f_peak * D_eff / U) if (U > 1e-12 and f_peak > 0) else 0.0

    res = {
        "k0x": k0x, "A": A,
        "mean_Fx": float(jnp.mean(Fx_hist_j)) if Fx_hist_j.size else 0.0,
        "mean_Tx": float(jnp.mean(jnp.asarray(T_hist))) if T_hist else 0.0,
        "f_peak": float(f_peak),
        "St": float(St),
        "vortex_count_est": float(jnp.mean(jnp.asarray(vort_hist[-5:]))) if len(vort_hist) >= 5 else (float(jnp.mean(jnp.asarray(vort_hist))) if vort_hist else 0.0),
        "norm_max_dev_pct": float(100.0 * max(abs(n - p_use.target_norm) for n in N_hist) / p_use.target_norm) if N_hist else 0.0,
        "n_upstream_est": float(upstream_density(jnp.abs(psi)**2, X, x_ref, sigma_ref))
    }
    if record_timeseries:
        res["timeseries"] = {
            "Fx_hist": Fx_hist,
            "Et_hist": Et_hist,
            "N_hist": N_hist,
            "dt_samp": dt_samp
        }
        res["snapshots"] = snapshots
    return res

# =========================
# Sweep and critical velocity
# =========================
def run_sweep(p: Params, K_VALUES, A_VALUES, n_steps, record_timeseries=False):
    results = []
    for A in A_VALUES:
        for k0 in K_VALUES:
            print(f"Running k0x={k0:.2f}, A={A} ...")
            r = simulate_once(p, k0x=k0, A=A, n_steps=n_steps, record_timeseries=record_timeseries and (k0==K_VALUES[-1]))
            results.append(r)
            print(f"  Fx={r['mean_Fx']:.4g}, Tx={r['mean_Tx']:.3f}, f={r['f_peak']:.4f}, St={r['St']:.3f}, vort={r['vortex_count_est']:.2f}, N_dev%={r['norm_max_dev_pct']:.2f}")
    return results

def detect_critical_velocity(results, min_vort=1.0):
    sorted_r = sorted(results, key=lambda r: r["k0x"])
    for r in sorted_r:
        if r["vortex_count_est"] >= min_vort:
            return r["k0x"], r
    return None, None

# =========================
# Post-processing: St vs Mach, Fx(t) + spectrum, snapshots, energy/norm audit
# =========================
def compute_mach(U, g, n_upstream):
    cs = np.sqrt(g * max(n_upstream, 1e-12))
    return U / cs if cs > 0 else 0.0

def plot_st_vs_m(results, g):
    Ms, Sts, kxs = [], [], []
    for r in results:
        M = compute_mach(U=float(r["k0x"]), g=g, n_upstream=float(r["n_upstream_est"]))
        Ms.append(M); Sts.append(float(r["St"])); kxs.append(float(r["k0x"]))
    fig, ax = plt.subplots(figsize=(6,4))
    sc = ax.scatter(Ms, Sts, c=kxs, cmap="viridis", s=60)
    ax.set_xlabel("Mach M = U / c_s (measured upstream density)"); ax.set_ylabel("Strouhal St")
    cb = plt.colorbar(sc); cb.set_label("k0x")
    ax.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

def plot_time_series_and_spectrum(Fx_hist, dt_samp, pad_factor=16, window="tukey", alpha=0.25):
    Fx_hist = np.asarray(Fx_hist, dtype=np.float64)
    t = np.arange(Fx_hist.size) * dt_samp
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    ax[0].plot(t, Fx_hist, lw=1)
    ax[0].set_xlabel("t"); ax[0].set_ylabel("Fx"); ax[0].set_title("Drag time series"); ax[0].grid(True, alpha=0.3)
    y = Fx_hist - np.mean(Fx_hist); n = y.size
    if window == "hann":
        w = np.hanning(n)
    else:
        k = np.arange(n); w = np.ones(n)
        rise = k < (alpha*(n-1)/2)
        w[rise] = 0.5 * (1 + np.cos(np.pi * ((2*k[rise]/(alpha*(n-1))) - 1)))
        fall = k > ((n-1)*(1 - alpha/2))
        w[fall] = 0.5 * (1 + np.cos(np.pi * ((2*k[fall]/(alpha*(n-1))) - (2/alpha) + 1)))
    yw = y * w
    Y = np.fft.fft(yw, n*pad_factor)
    f = np.fft.fftfreq(n*pad_factor, d=dt_samp)
    pos = f > 0
    ax[1].plot(f[pos], np.abs(Y[pos])**2, lw=1)
    ax[1].set_xlabel("f"); ax[1].set_ylabel("Power"); ax[1].set_title("Power spectrum"); ax[1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

def visualize_vortex_core(snapshot, title="Snapshot"):
    dens = snapshot["dens"]; phase = snapshot["phase"]
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    im0 = ax[0].imshow(dens, origin="lower", cmap="magma")
    ax[0].set_title(f"{title} density"); plt.colorbar(im0, ax=ax[0])
    im1 = ax[1].imshow(phase, origin="lower", cmap="twilight")
    ax[1].set_title(f"{title} phase"); plt.colorbar(im1, ax=ax[1])
    plt.tight_layout(); plt.show()
    print(f"Min density in snapshot: {dens.min():.3e}")

def plot_energy_and_norm(Et_hist, N_hist):
    Et_hist = np.asarray(Et_hist, dtype=np.float64)
    N_hist = np.asarray(N_hist, dtype=np.float64)
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    ax[0].plot(Et_hist, lw=1); ax[0].set_xlabel("sample index"); ax[0].set_ylabel("Energy"); ax[0].set_title("Energy audit"); ax[0].grid(True, alpha=0.3)
    ax[1].plot(N_hist, lw=1); ax[1].set_xlabel("sample index"); ax[1].set_ylabel("Norm"); ax[1].set_title("Norm audit"); ax[1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

# =========================
# Main
# =========================
if __name__ == "__main__":
    # Domain and conservative dt
    params = Params(
        g=1.0, dt=1e-4, Lx=80.0, Ly=24.0, Nx=1536, Ny=864,
        obs_height=5.0, obs_sigma=0.50, obs_x0=2.0, obs_y0=0.0,
        wall_V0=10.0, wall_width=2.2,
        sponge_V0=3.0, sponge_width=4.0, sponge_power=2.0,
        sample_stride=10, target_norm=1.0,
        init_profile="TF", tf_mu=1.2,
        itp_steps=4000, itp_dt=2e-4,
        pad_factor=16, window_type="tukey", tukey_alpha=0.25,
        vort_density_frac=0.10,
        snap_stride=2000, max_snaps=6
    )

    # Critical velocity sweep: k0x 0.50..0.80 with strong obstacle
    K_VALUES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    A_VALUES = [5.0]
    N_STEPS_SWEEP = 30000

    print("Running critical velocity sweep...")
    sweep_results = run_sweep(params, K_VALUES, A_VALUES, N_STEPS_SWEEP, record_timeseries=True)

    # Detect onset of vortex shedding
    kcrit, rcrit = detect_critical_velocity(sweep_results, min_vort=1.0)
    if kcrit is not None:
        print(f"\nCritical velocity detected at k0x={kcrit:.2f}: vort={rcrit['vortex_count_est']:.2f}, f={rcrit['f_peak']:.4f}, St={rcrit['St']:.3f}")
    else:
        print("\nNo vortex shedding detected in 0.50..0.80 with current settings. Consider increasing A or reducing dt/sample_stride.")

    # Save compact JSON summary
    summary = {
        "params": {
            "g": float(params.g), "dt": float(params.dt),
            "Lx": float(params.Lx), "Ly": float(params.Ly),
            "Nx": int(params.Nx), "Ny": int(params.Ny),
            "obs_sigma": float(params.obs_sigma),
            "wall_width": float(params.wall_width),
            "sponge_width": float(params.sponge_width),
            "init_profile": params.init_profile,
            "tf_mu": float(params.tf_mu)
        },
        "sweep_results": [
            {k: r[k] for k in ("k0x","A","mean_Fx","mean_Tx","f_peak","St","vortex_count_est","norm_max_dev_pct","n_upstream_est")}
            for r in sweep_results
        ],
        "critical_velocity": {"k0x": float(kcrit) if kcrit is not None else None, "case": rcrit}
    }
    with open("summary_monolith.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved summary to summary_monolith.json")

    # Post-processing: St vs Mach (using measured upstream density)
    print("\nPlotting St vs Mach from measured upstream density...")
    plot_st_vs_m(sweep_results, g=float(params.g))

    # Choose one representative shedding case (last in sweep or critical) for deep plots
    case_for_plots = rcrit if rcrit is not None else sweep_results[-1]
    if "timeseries" in case_for_plots:
        ts = case_for_plots["timeseries"]
        print("\nPlotting Fx time series and spectrum...")
        plot_time_series_and_spectrum(ts["Fx_hist"], ts["dt_samp"],
                                      pad_factor=params.pad_factor,
                                      window="tukey",
                                      alpha=params.tukey_alpha)

        print("Plotting energy and norm audits...")
        plot_energy_and_norm(ts["Et_hist"], ts["N_hist"])

    if "snapshots" in case_for_plots:
        print("\nVisualizing density and phase snapshots...")
        for snap in case_for_plots["snapshots"]:
            visualize_vortex_core(snap, title=f"t={snap['t']}")
