import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)


@dataclass
class Params:
    nu: float = 0.02
    dt: float = 0.01
    t_final: float = 1.0
    N_int_v: int = 44
    N_int_eta: int = 36
    lam_pde: float = 1e-8
    lam_bc: float = 1e-10
    jitter: float = 1e-10
    ls_v: float = 0.2
    sigma_v: float = 1.0
    ls_eta: float = 0.2
    sigma_eta: float = 1.0
    beta_warp: float = 1.2
    rho: float = 0.2
    eps: float = 1e-6
    alpha: float = 1.0
    gn_max_iter: int = 3
    gn_tol: float = 2e-4
    grid_N: int = 801
    warp_fd_order: int = 2
    n_modes_ref: int = 2500
    n_quad_ref: int = 20001


def make_interior_points(n, rng, pad=0.02):
    m = max(6, n // 8)
    core_n = n - 2 * m
    core = rng.uniform(-1.0 + pad, 1.0 - pad, size=core_n)
    left = -1.0 + pad + rng.uniform(0.0, 0.05, size=m)
    right = 1.0 - pad - rng.uniform(0.0, 0.05, size=m)
    pts = np.concatenate([core, left, right])
    return np.sort(pts)


def sample_eta_points(s_grid, monitor, n_eta, rho, alpha, eps, rng):
    weights = eps + np.power(np.maximum(monitor, 0.0), alpha)
    weights /= np.trapezoid(weights, s_grid)
    pmass = weights / np.sum(weights)
    unif = np.full_like(pmass, 1.0 / pmass.size)
    mix = (1.0 - rho) * pmass + rho * unif
    mix /= np.sum(mix)
    idx = rng.choice(np.arange(s_grid.size), size=n_eta, p=mix, replace=True)
    pts = s_grid[idx]
    m = max(2, n_eta // 10)
    b_left = -1.0 + 0.01 + rng.uniform(0, 0.03, size=m)
    b_right = 1.0 - 0.01 - rng.uniform(0, 0.03, size=m)
    pts = np.concatenate([pts, b_left, b_right])
    return np.sort(np.clip(pts, -1.0, 1.0))


def rbf_1d_deriv(y, yp, ls, sigma2, a, b):
    r = y - yp
    k = sigma2 * jnp.exp(-(r**2) / (2.0 * ls**2))
    l2 = ls**2
    l4 = l2**2
    l6 = l2**3
    l8 = l2**4
    if a + b == 0:
        dn = k
    elif a + b == 1:
        dn = -(r / l2) * k
    elif a + b == 2:
        dn = ((r**2) / l4 - 1.0 / l2) * k
    elif a + b == 3:
        dn = (-(r**3) / l6 + 3.0 * r / l4) * k
    elif a + b == 4:
        dn = ((r**4) / l8 - 6.0 * (r**2) / l6 + 3.0 / l4) * k
    else:
        raise ValueError("Only derivatives up to order 4 are supported.")
    return ((-1) ** b) * dn


def side_terms(order, wp, wpp):
    if order == 0:
        return [(0, wp * 0.0 + 1.0)]
    if order == 1:
        return [(1, wp)]
    if order == 2:
        return [(1, wpp), (2, wp**2)]
    raise ValueError("Jet order must be 0, 1, or 2")


def kernel_entry(sL, pL, sR, pR, ls, sigma, wrap=None):
    if wrap is None:
        yL, yR = sL, sR
        wpL, wppL, wpR, wppR = 1.0, 0.0, 1.0, 0.0
    else:
        yL = wrap["w"](sL)
        yR = wrap["w"](sR)
        wpL = wrap["wp"](sL)
        wppL = wrap["wpp"](sL)
        wpR = wrap["wp"](sR)
        wppR = wrap["wpp"](sR)
    left = side_terms(pL, wpL, wppL)
    right = side_terms(pR, wpR, wppR)
    out = 0.0
    for a, ca in left:
        for b, cb in right:
            out = out + ca * cb * rbf_1d_deriv(yL, yR, ls, sigma**2, a, b)
    return out


def build_functional_set(s_int):
    n = s_int.size
    s_fun = np.zeros(3 * n + 2)
    p_fun = np.zeros(3 * n + 2, dtype=int)
    for i in range(n):
        base = 3 * i
        s_fun[base : base + 3] = s_int[i]
        p_fun[base : base + 3] = np.array([0, 1, 2])
    s_fun[3 * n] = -1.0
    s_fun[3 * n + 1] = 1.0
    p_fun[3 * n] = 0
    p_fun[3 * n + 1] = 0
    return s_fun, p_fun


def assemble_gram(s_fun, p_fun, ls, sigma, jitter, wrap=None):
    M = s_fun.size
    K = np.zeros((M, M), dtype=np.float64)
    for i in range(M):
        for j in range(i, M):
            kij = float(kernel_entry(s_fun[i], int(p_fun[i]), s_fun[j], int(p_fun[j]), ls, sigma, wrap))
            K[i, j] = kij
            K[j, i] = kij
    K += jitter * np.eye(M)
    return K


def assemble_H_y(s_int, c0, c1, c2, b):
    n = s_int.size
    M = 3 * n + 2
    H = np.zeros((n + 2, M), dtype=np.float64)
    y = np.zeros(n + 2, dtype=np.float64)
    for i in range(n):
        H[i, 3 * i + 0] = c0[i]
        H[i, 3 * i + 1] = c1[i]
        H[i, 3 * i + 2] = c2
        y[i] = b[i]
    H[n, 3 * n] = 1.0
    H[n + 1, 3 * n + 1] = 1.0
    y[n] = 0.0
    y[n + 1] = 0.0
    return H, y


def solve_min_norm(K, H, y, lam_pde, lam_bc):
    n_rows = H.shape[0]
    Lambda = np.diag(np.concatenate([lam_pde * np.ones(n_rows - 2), [lam_bc, lam_bc]]))
    A = H @ K @ H.T + Lambda
    rhs = y
    beta = np.linalg.solve(A, rhs)
    alpha = H.T @ beta
    return alpha


def evaluate_from_alpha(alpha, s_query, s_fun, p_fun, ls, sigma, wrap=None, d_query=0):
    out = np.zeros_like(s_query, dtype=np.float64)
    for m, sq in enumerate(s_query):
        val = 0.0
        for a in range(s_fun.size):
            val += alpha[a] * float(kernel_entry(sq, d_query, s_fun[a], int(p_fun[a]), ls, sigma, wrap))
        out[m] = val
    return out


def build_warp(s_grid, monitor_abs, beta, eps):
    q = np.power(eps + np.maximum(monitor_abs, 0.0), beta)
    q /= np.trapezoid(q, s_grid)
    ds = np.diff(s_grid)
    cdf = np.zeros_like(s_grid)
    cdf[1:] = np.cumsum(0.5 * (q[:-1] + q[1:]) * ds)
    cdf /= cdf[-1]
    wp = np.gradient(cdf, s_grid, edge_order=2)
    wpp = np.gradient(wp, s_grid, edge_order=2)

    def interp(arr):
        return lambda x: np.interp(np.asarray(x), s_grid, arr)

    return {"w": interp(cdf), "wp": interp(wp), "wpp": interp(wpp), "q": q, "w_grid": cdf}


def cole_hopf_reference(s, t, nu, n_modes=2500, n_quad=20001):
    sq = np.linspace(-1.0, 1.0, n_quad)
    yq = (sq + 1.0) / 2.0
    phi0 = np.exp(-(np.cos(np.pi * sq) + 1.0) / (2.0 * nu * np.pi))
    c0 = np.trapezoid(phi0, yq)
    k = np.arange(1, n_modes + 1)
    cos_mat = np.cos(np.pi * np.outer(k, yq))
    ck = 2.0 * np.trapezoid(phi0[None, :] * cos_mat, yq, axis=1)

    y = (s + 1.0) / 2.0
    cos_eval = np.cos(np.pi * np.outer(k, y))
    sin_eval = np.sin(np.pi * np.outer(k, y))
    decay = np.exp(-nu * (k * np.pi / 2.0) ** 2 * t)

    phi = c0 + np.sum((ck * decay)[:, None] * cos_eval, axis=0)
    phi_s = np.sum((ck * decay * (-k * np.pi / 2.0))[:, None] * sin_eval, axis=0)
    u = -2.0 * nu * (phi_s / phi)
    return u


def run_solver(params: Params):
    rng = np.random.default_rng(42)
    s_grid = np.linspace(-1.0, 1.0, params.grid_N)
    nt = int(round(params.t_final / params.dt))
    times = np.linspace(0.0, params.t_final, nt + 1)

    u = -np.sin(np.pi * s_grid)
    u[0], u[-1] = 0.0, 0.0

    snapshots = {0.0: u.copy()}
    monitor_records = []

    for k in range(nt):
        t_next = times[k + 1]
        u_prev = u.copy()
        u_n = u_prev.copy()

        for _ in range(params.gn_max_iter):
            u_n_s = np.gradient(u_n, s_grid, edge_order=2)

            s_v_int = make_interior_points(params.N_int_v, rng)
            c0 = 1.0 / params.dt + np.interp(s_v_int, s_grid, u_n_s)
            c1 = np.interp(s_v_int, s_grid, u_n)
            b = (1.0 / params.dt) * np.interp(s_v_int, s_grid, u_prev)

            s_fun_v, p_fun_v = build_functional_set(s_v_int)
            K_v = assemble_gram(s_fun_v, p_fun_v, params.ls_v, params.sigma_v, params.jitter, wrap=None)
            H_v, y_v = assemble_H_y(s_v_int, c0, c1, -params.nu, b)
            alpha_v = solve_min_norm(K_v, H_v, y_v, params.lam_pde, params.lam_bc)

            v = evaluate_from_alpha(alpha_v, s_grid, s_fun_v, p_fun_v, params.ls_v, params.sigma_v, wrap=None, d_query=0)
            v_s = evaluate_from_alpha(alpha_v, s_grid, s_fun_v, p_fun_v, params.ls_v, params.sigma_v, wrap=None, d_query=1)
            v_ss = evaluate_from_alpha(alpha_v, s_grid, s_fun_v, p_fun_v, params.ls_v, params.sigma_v, wrap=None, d_query=2)
            v[0], v[-1] = 0.0, 0.0

            monitor = np.abs((v - u_prev) / params.dt + v * v_s - params.nu * v_ss)
            s_eta_int = sample_eta_points(
                s_grid, monitor, params.N_int_eta, params.rho, params.alpha, params.eps, rng
            )

            c0e = 1.0 / params.dt + np.interp(s_eta_int, s_grid, u_n_s)
            c1e = np.interp(s_eta_int, s_grid, u_n)
            be = (1.0 / params.dt) * np.interp(s_eta_int, s_grid, u_prev)
            vv = np.interp(s_eta_int, s_grid, v)
            vv_s = np.interp(s_eta_int, s_grid, v_s)
            vv_ss = np.interp(s_eta_int, s_grid, v_ss)
            r_lin = c0e * vv + c1e * vv_s - params.nu * vv_ss - be

            warp = build_warp(s_grid, monitor, params.beta_warp, params.eps)

            s_fun_e, p_fun_e = build_functional_set(s_eta_int)
            K_e = assemble_gram(s_fun_e, p_fun_e, params.ls_eta, params.sigma_eta, params.jitter, wrap=warp)
            H_e, y_e = assemble_H_y(s_eta_int, c0e, c1e, -params.nu, -r_lin)
            alpha_e = solve_min_norm(K_e, H_e, y_e, params.lam_pde, params.lam_bc)

            eta = evaluate_from_alpha(alpha_e, s_grid, s_fun_e, p_fun_e, params.ls_eta, params.sigma_eta, wrap=warp, d_query=0)
            u_next = v + eta
            u_next[0], u_next[-1] = 0.0, 0.0

            if np.max(np.abs(u_next - u_n)) < params.gn_tol:
                u_n = u_next
                break
            u_n = u_next

        u = u_n
        monitor_records.append((t_next, monitor.copy(), s_eta_int.copy()))
        for ts in (0.5, 0.75, 0.95):
            if abs(t_next - ts) < 0.5 * params.dt:
                snapshots[ts] = u.copy()

    return s_grid, times, snapshots, monitor_records


def main():
    params = Params()
    out_dir = "outputs_test2"
    os.makedirs(out_dir, exist_ok=True)

    s_grid, times, snapshots, monitor_records = run_solver(params)

    target_times = [0.5, 0.75, 0.95]
    refs = {
        t: cole_hopf_reference(
            s_grid, t, params.nu, n_modes=params.n_modes_ref, n_quad=params.n_quad_ref
        )
        for t in target_times
    }

    plt.figure(figsize=(10, 6))
    for t in target_times:
        u_num = snapshots[t]
        u_ref = refs[t]
        err = np.linalg.norm(u_num - u_ref) / np.sqrt(u_num.size)
        print(f"t={t:.2f}: L2 pointwise RMS error={err:.6e}, Linf={np.max(np.abs(u_num-u_ref)):.6e}")
        plt.plot(s_grid, u_num, label=f"num t={t:.2f}")
        plt.plot(s_grid, u_ref, "--", label=f"ref t={t:.2f}")
    plt.title("Burgers solution slices: RKHS GN (v+eta) vs Cole-Hopf")
    plt.xlabel("s")
    plt.ylabel("u(s,t)")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "solution_slices.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    for t in target_times:
        err = np.abs(snapshots[t] - refs[t])
        plt.plot(s_grid, err, label=f"|err| t={t:.2f}")
    plt.yscale("log")
    plt.title("Pointwise absolute error vs Cole-Hopf")
    plt.xlabel("s")
    plt.ylabel("|u_num-u_ref|")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pointwise_errors.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(10, 8))
    picks = [0, len(monitor_records) // 2, len(monitor_records) - 1]
    for i, idx in enumerate(picks, 1):
        t, mon, s_eta = monitor_records[idx]
        ax = plt.subplot(3, 1, i)
        ax.plot(s_grid, mon, lw=1.2, label=f"monitor @ t={t:.2f}")
        ax.scatter(s_eta, np.interp(s_eta, s_grid, mon), s=8, c="r", alpha=0.6, label="eta points")
        ax.set_ylabel("monitor")
        ax.legend(fontsize=8)
    plt.xlabel("s")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "monitor_and_eta_points.png"), dpi=180)
    plt.close()

    print(f"Saved plots in: {out_dir}")


if __name__ == "__main__":
    main()
