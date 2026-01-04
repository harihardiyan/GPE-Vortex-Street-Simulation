
# GPE 2D channel flow (TPU/GPU-friendly) with bluff obstacle, robust vortex counting,
# drag-FFT, Strouhal comparison, and snapshot visualizations (phase, density, vorticity).
# Split-step Fourier (Strang), JAX float32/complex64 (x64 disabled for TPU).
# Ready for Colab/GPU/TPU: tune Nx, Ny, N_STEPS for runtime/RAM.

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
    # Obstacle
    obs_height: float = 1.4
    obs_sigma: float = 0.30
    obs_x0: float = 2.0
    obs_y0: float = 0.0
    # Channel walls
    wall_height: float = 6.0
    wall_thickness: float = 2.0
    # Audit cadence
    sample_stride: int = 80
    target_norm: float = 1.0
    # Snapshots
    snapshot_stride: int = 800   # every N steps, save phase/density/vorticity snapshots
    max_snapshots: int = 6       # limit total snapshots to avoid RAM blowup

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
    return h * (mask_top + mask_bot)

def total_potential(params: Params, X, Y):
    V = jnp.zeros_like(X, dtype=f32)
    if params.omega != 0.0:
        V = V + potential_trap(X, Y, params.omega)
    V = V + potential_obstacle(X, Y, params.obs_height, params.obs_sigma, params.obs_x0, params.obs_y0)
    V = V + potential_walls(Y, params.Ly, params.wall_height, params.wall_thickness)
    return V

def build_step_real(params: Params, X, Y, K2):
    g = jnp.asarray(params.g, dtype=f32)
    dt = jnp.asarray(params.dt, dtype=f32)
    Vxy = total_potential(params, X, Y)
    half = jnp.asarray(0.5, dtype=f32)

    def phase_nl(psi):
        return jnp.exp(-1j * g * jnp.abs(psi)**2 * (dt * half)) * psi

    def phase_V(psi):
        return jnp.exp(-1j * Vxy * (dt * half)) * psi

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
        return psi

    return step, Vxy

def norm(psi, dx, dy):
    return jnp.sum(jnp.abs(psi)**2).astype(f32) * dx * dy

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

def obstacle_force_analytic(psi, X, Y, dx, dy, params: Params):
    Vobs = potential_obstacle(X, Y, params.obs_height, params.obs_sigma, params.obs_x0, params.obs_y0)
    dVdx = -((X - jnp.asarray(params.obs_x0, dtype=f32)) / (jnp.asarray(params.obs_sigma, dtype=f32)**2)) * Vobs
    dVdy = -((Y - jnp.asarray(params.obs_y0, dtype=f32)) / (jnp.asarray(params.obs_sigma, dtype=f32)**2)) * Vobs
    rho = jnp.abs(psi)**2
    Fx = -jnp.sum(rho * dVdx).astype(f32) * dx * dy
    Fy = -jnp.sum(rho * dVdy).astype(f32) * dx * dy
    return Fx, Fy

def transmission_fraction_x(psi, X, dx, dy, x0):
    mask_right = (X > jnp.asarray(x0, dtype=f32)).astype(f32)
    mass_right = jnp.sum(jnp.abs(psi)**2 * mask_right).astype(f32) * dx * dy
    mass_total = jnp.sum(jnp.abs(psi)**2).astype(f32) * dx * dy
    return jnp.where(mass_total > 0, mass_right / mass_total, jnp.asarray(0.0, dtype=f32))

def vorticity_from_current(jx, jy, dx, dy):
    d_jy_dx = (jnp.roll(jy, -1, axis=1) - jnp.roll(jy, 1, axis=1)) / (jnp.asarray(2.0, dtype=f32) * dx)
    d_jx_dy = (jnp.roll(jx, -1, axis=0) - jnp.roll(jx, 1, axis=0)) / (jnp.asarray(2.0, dtype=f32) * dy)
    return d_jy_dx - d_jx_dy

def phase_winding_count(psi):
    # robust ±2π winding on plaquettes
    phase = jnp.angle(psi)
    dphi_x = phase[:,1:] - phase[:,:-1]
    dphi_y = phase[1:,:] - phase[:-1,:]
    dphi_x = jnp.arctan2(jnp.sin(dphi_x), jnp.cos(dphi_x))
    dphi_y = jnp.arctan2(jnp.sin(dphi_y), jnp.cos(dphi_y))
    w = dphi_x[:-1,:] + dphi_y[:,1:] - dphi_x[1:,:] - dphi_y[:,:-1]
    wn = jnp.round(w / (2*jnp.pi))
    return float(jnp.sum(jnp.abs(wn)))

def hann_window(n):
    n = int(n)
    return jnp.asarray(0.5, dtype=f32) - jnp.asarray(0.5, dtype=f32) * jnp.cos(2.0 * jnp.pi * jnp.arange(n, dtype=f32) / jnp.asarray(n, dtype=f32))

def drag_fft_peak(Fx_hist, dt, pad_factor=8):
    if Fx_hist.size < 16:
        return 0.0, 0.0, 0.0
    y = Fx_hist - jnp.mean(Fx_hist)
    n = int(y.shape[0])
    w = hann_window(n)
    yw = y * w
    Npad = int(n * pad_factor)
    pad_len = max(Npad - n, 0)
    yw_pad = jnp.pad(yw, (0, pad_len))
    Y = jnp.fft.fft(yw_pad)
    power = jnp.abs(Y)**2
    f = jnp.fft.fftfreq(yw_pad.shape[0], d=float(dt))
    pos = f > 0
    fpos = f[pos]
    ppos = power[pos]
    idx = int(jnp.argmax(ppos))
    p_peak = float(ppos[idx])
    f_peak = float(fpos[idx])
    q_factor = float(p_peak / (float(jnp.median(ppos)) + 1e-9))
    return f_peak, p_peak, q_factor

def energy_total(params: Params, psi, X, Y, KX, KY, dx, dy, Vxy):
    dpsi_dx, dpsi_dy = grad_spectral(psi, KX, KY)
    kinetic = jnp.sum(jnp.asarray(0.5, dtype=f32) * (jnp.abs(dpsi_dx)**2 + jnp.abs(dpsi_dy)**2)).astype(f32) * dx * dy
    interaction = jnp.sum(jnp.asarray(0.5, dtype=f32) * jnp.asarray(params.g, dtype=f32) * (jnp.abs(psi)**4)).astype(f32) * dx * dy
    pot = jnp.sum(Vxy * (jnp.abs(psi)**2)).astype(f32) * dx * dy
    return kinetic + interaction + pot

def take_snapshots(psi, KX, KY, dx, dy):
    # Return numpy arrays for visualization: density, phase, vorticity
    jx, jy = current_density(psi, KX, KY)
    vort = vorticity_from_current(jx, jy, dx, dy)
    dens = jnp.abs(psi)**2
    phase = jnp.angle(psi)
    return np.array(dens), np.array(phase), np.array(vort)

def simulate_once(params: Params, k0x: float, A: float, n_steps: int):
    X, Y, KX, KY, K2, dx, dy = make_grid(params)
    local_params = Params(
        params.g, params.dt, params.Lx, params.Ly, params.Nx, params.Ny,
        params.omega, float(A), params.obs_sigma, params.obs_x0, params.obs_y0,
        params.wall_height, params.wall_thickness, params.sample_stride, params.target_norm,
        params.snapshot_stride, params.max_snapshots
    )
    step, Vxy = build_step_real(local_params, X, Y, K2)

    sigma_y = jnp.asarray(params.Ly / 4.0, dtype=f32)
    psi = jnp.exp(-jnp.asarray(0.5, dtype=f32) * ((Y - jnp.asarray(params.obs_y0, dtype=f32))**2) / sigma_y**2).astype(c64)
    psi = psi * jnp.exp(1j * (jnp.asarray(k0x, dtype=f32) * X))
    psi = psi * jnp.sqrt(jnp.asarray(params.target_norm, dtype=f32) / jnp.maximum(norm(psi, dx, dy), jnp.asarray(1e-12, dtype=f32)))

    # warm-up compile
    for _ in range(200):
        psi = step(psi)

    # histories
    Fx_hist, T_hist, vort_hist, jx_mean_hist, Etot_hist, N_hist = [], [], [], [], [], []
    snapshots = []  # list of dicts: {"step": t, "dens":..., "phase":..., "vort":...}
    snap_count = 0

    for t in range(n_steps):
        psi = step(psi)
        if (t % params.sample_stride) == 0:
            jx, jy = current_density(psi, KX, KY)
            Fx, Fy = obstacle_force_analytic(psi, X, Y, dx, dy, local_params)
            Tfrac = transmission_fraction_x(psi, X, dx, dy, local_params.obs_x0)
            vort_count = phase_winding_count(psi)
            Etot = energy_total(local_params, psi, X, Y, KX, KY, dx, dy, Vxy)
            Fx_hist.append(float(Fx)); T_hist.append(float(Tfrac)); vort_hist.append(float(vort_count))
            jx_mean_hist.append(float(jnp.mean(jx))); Etot_hist.append(float(Etot)); N_hist.append(float(norm(psi, dx, dy)))

        # snapshot cadence
        if (t % params.snapshot_stride) == 0 and snap_count < params.max_snapshots:
            dens, phase, vort = take_snapshots(psi, KX, KY, dx, dy)
            snapshots.append({"step": t, "dens": dens, "phase": phase, "vort": vort})
            snap_count += 1

    Fx_hist = np.array(Fx_hist, dtype=np.float32)
    T_hist = np.array(T_hist, dtype=np.float32)
    vort_hist = np.array(vort_hist, dtype=np.float32)
    jx_mean_hist = np.array(jx_mean_hist, dtype=np.float32)
    Etot_hist = np.array(Etot_hist, dtype=np.float32)
    N_hist = np.array(N_hist, dtype=np.float32)

    dt_samp = float(params.dt) * params.sample_stride
    f_peak, p_peak, q_factor = drag_fft_peak(jnp.asarray(Fx_hist), dt_samp, pad_factor=8)

    return {
        "k0x": k0x,
        "A": A,
        "mean_Fx": float(np.mean(Fx_hist)) if Fx_hist.size else 0.0,
        "mean_Tx": float(np.mean(T_hist)) if T_hist.size else 0.5,
        "f_peak": f_peak,
        "p_peak": p_peak,
        "q_factor": q_factor,
        "vortex_count_est": float(np.mean(vort_hist[-5:])) if vort_hist.size >= 5 else float(np.mean(vort_hist)) if vort_hist.size else 0.0,
        "Fx_hist": Fx_hist.tolist(),
        "T_hist": T_hist.tolist(),
        "vort_hist": vort_hist.tolist(),
        "jx_mean_hist": jx_mean_hist.tolist(),
        "energy_total_hist": Etot_hist.tolist(),
        "norm_hist": N_hist.tolist(),
        "snapshots": snapshots  # contains numpy arrays for visualization
    }

# Runner: sweep k0x × A, visualize snapshots, Strouhal, and summaries
if __name__ == "__main__":
    # Baseline tuned toward shedding; increase Nx,Ny,N_STEPS if GPU/TPU allows
    params = Params(
        g=1.0, dt=2e-4, Lx=40.0, Ly=20.0, Nx=512, Ny=384,   # upscaled grid for better vortex visibility
        omega=0.0,
        obs_height=1.4, obs_sigma=0.30, obs_x0=2.0, obs_y0=0.0,
        wall_height=6.0, wall_thickness=2.0,
        sample_stride=80, target_norm=1.0,
        snapshot_stride=800, max_snapshots=6
    )

    K_VALUES = [0.6, 0.7, 0.8]     # push toward supercritical regime
    A_VALUES = [1.3, 1.5, 1.7]
    N_STEPS = 24000                # increase if time allows (30k–50k for sharper FFT)

    results = []
    for A in A_VALUES:
        for k0 in K_VALUES:
            print(f"Running k0x={k0}, A={A} ...")
            r = simulate_once(params, k0x=k0, A=A, n_steps=N_STEPS)
            results.append(r)
            print(f"  mean_Fx={r['mean_Fx']:.4g}, Tx={r['mean_Tx']:.3f}, f_peak={r['f_peak']:.4f}, Q={r['q_factor']:.2f}, vort={r['vortex_count_est']:.2f}")

    # organize matrices
    K_vals = np.array(K_VALUES, dtype=np.float32)
    A_vals = np.array(A_VALUES, dtype=np.float32)
    f_peak_mat = np.zeros((len(A_vals), len(K_vals)), dtype=np.float32)
    vort_mat = np.zeros_like(f_peak_mat)
    drag_mat = np.zeros_like(f_peak_mat)

    for i, A in enumerate(A_vals):
        for j, k0 in enumerate(K_vals):
            entry = next(r for r in results if abs(r["A"] - float(A)) < 1e-6 and abs(r["k0x"] - float(k0)) < 1e-6)
            f_peak_mat[i, j] = entry["f_peak"]
            vort_mat[i, j] = entry["vortex_count_est"]
            drag_mat[i, j] = entry["mean_Fx"]

    # Strouhal number: St = f * D / U ; effective diameter D ~ 2*sigma
    D_eff = 2.0 * float(params.obs_sigma)
    St_mat = np.zeros_like(f_peak_mat)
    for i in range(f_peak_mat.shape[0]):
        for j in range(f_peak_mat.shape[1]):
            U = float(K_vals[j])
            f = float(f_peak_mat[i, j])
            St_mat[i, j] = (f * D_eff / U) if (U > 1e-12 and f > 0) else 0.0

    # Visualizations
    # 1) Snapshot panels: pick the run with highest q_factor to showcase patterns
    showcase = max(results, key=lambda r: r["q_factor"])
    snaps = showcase["snapshots"]
    if len(snaps) > 0:
        n_show = len(snaps)
        fig, axes = plt.subplots(n_show, 3, figsize=(12, 4*n_show))
        for idx, s in enumerate(snaps):
            dens = s["dens"]; phase = s["phase"]; vort = s["vort"]
            axes[idx, 0].imshow(dens, origin="lower", cmap="viridis")
            axes[idx, 0].set_title(f"Density |psi|^2 (step={s['step']})")
            axes[idx, 0].set_xlabel("x"); axes[idx, 0].set_ylabel("y")

            im = axes[idx, 1].imshow(phase, origin="lower", cmap="twilight")
            axes[idx, 1].set_title("Phase angle")
            axes[idx, 1].set_xlabel("x"); axes[idx, 1].set_ylabel("y")
            plt.colorbar(im, ax=axes[idx, 1], fraction=0.046, pad=0.04)

            im2 = axes[idx, 2].imshow(vort, origin="lower", cmap="plasma")
            axes[idx, 2].set_title("Vorticity (∂x jy − ∂y jx)")
            axes[idx, 2].set_xlabel("x"); axes[idx, 2].set_ylabel("y")
            plt.colorbar(im2, ax=axes[idx, 2], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()

    # 2) Line plots for first A
    A0 = A_vals[0]
    sel = [r for r in results if abs(r["A"] - float(A0)) < 1e-6]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(K_vals, [s["f_peak"] for s in sel], marker="o")
    axes[0].set_title(f"Shedding frequency vs k0x (A={float(A0)})")
    axes[0].set_xlabel("k0x"); axes[0].set_ylabel("f_peak")

    axes[1].plot(K_vals, [s["vortex_count_est"] for s in sel], marker="o", color="tab:red")
    axes[1].set_title(f"Vortex count vs k0x (A={float(A0)})")
    axes[1].set_xlabel("k0x"); axes[1].set_ylabel("vortex count")

    axes[2].plot(K_vals, [s["mean_Fx"] for s in sel], marker="o", color="tab:green")
    axes[2].set_title(f"Mean drag vs k0x (A={float(A0)})")
    axes[2].set_xlabel("k0x"); axes[2].set_ylabel("mean Fx")
    plt.tight_layout(); plt.show()

    # 3) Heatmaps: f_peak, vortex count, Strouhal
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    im0 = axes[0].imshow(f_peak_mat, origin="lower", aspect="auto",
                         extent=[K_vals.min(), K_vals.max(), A_vals.min(), A_vals.max()], cmap="magma")
    axes[0].set_title("Heatmap f_peak (k0x × A)"); axes[0].set_xlabel("k0x"); axes[0].set_ylabel("A")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(vort_mat, origin="lower", aspect="auto",
                         extent=[K_vals.min(), K_vals.max(), A_vals.min(), A_vals.max()], cmap="plasma")
    axes[1].set_title("Heatmap vortex count (k0x × A)"); axes[1].set_xlabel("k0x"); axes[1].set_ylabel("A")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(St_mat, origin="lower", aspect="auto",
                         extent=[K_vals.min(), K_vals.max(), A_vals.min(), A_vals.max()], cmap="inferno")
    axes[2].set_title("Heatmap Strouhal number St = f D / U"); axes[2].set_xlabel("k0x"); axes[2].set_ylabel("A")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.show()

    # JSON summary (compact)
    summary = {
        "params": {
            "g": params.g, "dt": params.dt, "Lx": params.Lx, "Ly": params.Ly,
            "Nx": params.Nx, "Ny": params.Ny, "obs_sigma": params.obs_sigma,
            "wall_thickness": params.wall_thickness
        },
        "K_VALUES": K_VALUES, "A_VALUES": A_VALUES,
        "f_peak_mat": f_peak_mat.tolist(),
        "vort_mat": vort_mat.tolist(),
        "drag_mat": drag_mat.tolist(),
        "St_mat": St_mat.tolist(),
        "showcase": {"k0x": showcase["k0x"], "A": showcase["A"], "q_factor": showcase["q_factor"]},
        "results": [{k: v for k, v in r.items() if k in ("k0x","A","mean_Fx","mean_Tx","f_peak","q_factor","vortex_count_est")} for r in results]
    }
    print(json.dumps(summary, indent=2))
