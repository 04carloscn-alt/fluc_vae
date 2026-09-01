#!/usr/bin/env python3
"""
pt_moments.py — Momentos de la distribución evento a evento de ⟨pT⟩
               por clase de centralidad.

Observable principal (Ec. 2.1 del protocolo)
─────────────────────────────────────────────
Para cada evento i con Ni partículas filtradas se calcula:

    <pT>_i = (1/Ni) * sum_{j=1}^{Ni} pT_j

Este escalar es el observable de entrada. Los momentos se calculan
sobre la COLECCIÓN de valores {<pT>_i} dentro de cada clase de centralidad,
no sobre las partículas individuales de un evento.

Momentos de la distribución P(<pT>) (Ecs. 2.2–2.4 del protocolo)
──────────────────────────────────────────────────────────────────
    mu1    = <<pT>>          = (1/Nev) * sum_i  <pT>_i
    mu2    = sigma^2         = <<pT>^2> - <<pT>>^2
    mu3    = tercer momento  = <<(<pT> - mu1)^3>
    S      = skewness        = mu3 / sigma^3        (Ec. 2.3)

Parametros de la Gamma estimados desde mu1 y mu2 (Ec. 5 de la tesis)
──────────────────────────────────────────────────────────────────────
    alpha = mu1^2 / mu2        (parametro de forma)
    beta  = mu2  / mu1         (parametro de escala)

    mu3_pred   = mu2^2 / mu1   (prediccion cerrada de la Gamma)
    S_pred     = 2/sqrt(alpha) (skewness estandarizado predicho)

Incertidumbre
─────────────
Se reporta el error estandar de la media (SEM = sigma/sqrt(Nev)) para
mu1, y para mu2/mu3/S se propaga via bootstrap (500 remuestreos) para
no asumir normalidad.

Filtro de particulas:
    ncl > 0
    --charged  -> chg != 0  (cargadas)
    default    -> chg == 0  (neutras)

Graficas producidas:
    Fig 1 — pt_moments_*:  mu1 | mu2 | mu3 vs centralidad
    Fig 2 — pt_gamma_*:    alpha | beta | S medido vs S predicho

Uso:
    python3 analysis/pt_moments.py output/Bi_11GeV_MB --charged
    python3 analysis/pt_moments.py output/Bi_11GeV_MB* --charged
    python3 analysis/pt_moments.py output/Bi_11GeV_MB output/Bi_11GeV_MB1 \\
        output/Bi_11GeV_MB2 output/Bi_11GeV_MB3 output/Bi_11GeV_MB4 \\
        output/Bi_11GeV_MB5 --charged --outdir output/moments_plots
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import sem

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_f14 import iter_all


# ── Centralidad (mismos cortes que fluc_observables.py) ──────────────────────
B_CUTS = [
    (0.0,   4.0,  "0-10%"),
    (4.0,   6.0,  "10-20%"),
    (6.0,   7.0,  "20-30%"),
    (7.0,   8.0,  "30-40%"),
    (8.0,   9.0,  "40-50%"),
    (9.0,  10.0,  "50-60%"),
    (10.0, 11.0,  "60-70%"),
    (11.0, 12.0,  "70-80%"),
    (12.0, 13.5,  "80-90%"),
    (13.5, np.inf,"90-100%"),
]
LABELS = [label for _, _, label in B_CUTS]

COLORS = [
    "#d62728", "#ff7f0e", "#f7c617", "#2ca02c",
    "#17becf", "#1f77b4", "#9467bd", "#8c564b", "#bcbd22", "#7f7f7f",
]


def assign_centrality(b):
    for b_lo, b_hi, label in B_CUTS:
        if b_lo <= b < b_hi:
            return label
    return "90-100%"


# ── Observable por evento ────────────────────────────────────────────────────

def pt_mean_event(pt_vals):
    """
    Calcula el observable <pT>_i para un evento dado.
    Devuelve nan si el evento no tiene particulas filtradas (N=0).
    Nota: se permite N=1 porque <pT> con una sola particula es valido;
    la varianza de la *distribucion entre eventos* aun esta bien definida.
    """
    if len(pt_vals) == 0:
        return np.nan
    return pt_vals.mean()


# ── Lectura de eventos ────────────────────────────────────────────────────────

def collect_moments(output_dirs, charged=False, max_events=None):
    """
    Itera sobre todos los eventos y guarda el observable <pT>_i de cada
    evento en la clase de centralidad que le corresponde.

    Retorna:
        { label: {"pt_mean": [<pT>_1, <pT>_2, ...]} }

    Los momentos de la distribucion P(<pT>) se calculan DESPUES en
    class_stats(), sobre esta coleccion de escalares.
    """
    if isinstance(output_dirs, (str, Path)):
        output_dirs = [output_dirs]

    groups = {label: {"pt_mean": []} for label in LABELS}
    n_total = 0

    for output_dir in output_dirs:
        output_dir = Path(output_dir)
        remaining = None if max_events is None else max_events - n_total
        if remaining is not None and remaining <= 0:
            break

        print(f"  Leyendo: {output_dir} ...", flush=True)
        for ev in iter_all(output_dir, max_events=remaining):
            if charged:
                mask = (ev["ncl"] > 0) & (ev["chg"] != 0)
            else:
                mask = (ev["ncl"] > 0) & (ev["chg"] == 0)

            pt_i = pt_mean_event(ev["pT"][mask])

            if np.isfinite(pt_i):
                label = assign_centrality(ev["b"])
                groups[label]["pt_mean"].append(pt_i)

            n_total += 1
            if n_total % 500 == 0:
                print(f"  {n_total} eventos procesados...", flush=True)
            if max_events is not None and n_total >= max_events:
                break

    return groups


# ── Estadistica por clase ─────────────────────────────────────────────────────

# ── Estadistica por clase ─────────────────────────────────────────────────────

def _bootstrap_err(arr, stat_fn, n_boot=500, rng=None):
    """
    Estima el error de una estadistica via bootstrap no-parametrico.
    stat_fn debe aceptar un array y devolver un escalar.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(arr)
    boot = np.array([stat_fn(rng.choice(arr, size=n, replace=True))
                     for _ in range(n_boot)])
    return float(np.std(boot, ddof=1))


def class_stats(groups):
    """
    Calcula los momentos de la distribucion P(<pT>) para cada clase de
    centralidad, tal como lo definen las Ecs. 2.2-2.4 del protocolo de tesis.

    Para la clase con Nev eventos con observable {y_i = <pT>_i}:

        mu1  = (1/Nev) * sum(y_i)                       media global
        mu2  = (1/Nev) * sum((y_i - mu1)^2)             varianza (Ec. 2.2)
        mu3  = (1/Nev) * sum((y_i - mu1)^3)             tercer momento central
        S    = mu3 / mu2^(3/2)                          skewness (Ec. 2.3)

    Parametros Gamma (Ec. 5 de la tesis):
        alpha = mu1^2 / mu2
        beta  = mu2  / mu1
        mu3_pred  = mu2^2 / mu1
        S_pred    = 2 / sqrt(alpha)

    Incertidumbres:
        mu1  -> SEM analitico (sigma/sqrt(N))
        mu2, mu3, S, alpha, beta -> bootstrap (500 remuestreos)

    Retorna:
        stats   : dict con sub-keys "mean", "err", "n" para cada cantidad
        x_valid : lista de indices de clases con Nev >= 10
    """
    rng = np.random.default_rng(42)
    quantities = ("mu1", "mu2", "mu3", "S",
                  "alpha", "beta", "mu3_pred", "S_pred")
    stats = {k: {"mean": [], "err": [], "n": []} for k in quantities}
    x_valid = []

    for idx, label in enumerate(LABELS):
        y = np.array(groups[label]["pt_mean"])
        n = len(y)

        if n < 10:          # minimo estadistico razonable
            for k in quantities:
                stats[k]["mean"].append(np.nan)
                stats[k]["err"].append(np.nan)
                stats[k]["n"].append(n)
            continue

        x_valid.append(idx)

        # ── Momentos medidos ─────────────────────────────────────────────────
        mu1 = y.mean()
        dy  = y - mu1
        mu2 = np.mean(dy**2)
        mu3 = np.mean(dy**3)
        S   = mu3 / mu2**1.5 if mu2 > 1e-30 else 0.0

        # ── Parametros Gamma ─────────────────────────────────────────────────
        alpha     = mu1**2 / mu2      if mu2 > 0 else np.nan
        beta      = mu2    / mu1      if mu1 > 0 else np.nan
        mu3_pred  = mu2**2 / mu1      if mu1 > 0 else np.nan
        S_pred    = 2.0 / np.sqrt(alpha) if (alpha and np.isfinite(alpha) and alpha > 0) else np.nan

        # ── Incertidumbres ───────────────────────────────────────────────────
        mu1_err = float(np.std(y, ddof=1) / np.sqrt(n))    # SEM analitico

        def _mu2(s):
            m = s.mean(); return np.mean((s - m)**2)
        def _mu3(s):
            m = s.mean(); return np.mean((s - m)**3)
        def _S(s):
            m = s.mean(); d = s - m; v = np.mean(d**2)
            return np.mean(d**3) / v**1.5 if v > 1e-30 else 0.0
        def _alpha(s):
            m = s.mean(); v = np.mean((s-m)**2)
            return m**2/v if v > 0 else np.nan
        def _beta(s):
            m = s.mean(); v = np.mean((s-m)**2)
            return v/m if m > 0 else np.nan
        def _mu3p(s):
            m = s.mean(); v = np.mean((s-m)**2)
            return v**2/m if m > 0 else np.nan
        def _Sp(s):
            m = s.mean(); v = np.mean((s-m)**2)
            a = m**2/v if v > 0 else np.nan
            return 2.0/np.sqrt(a) if (a and np.isfinite(a) and a > 0) else np.nan

        mu2_err   = _bootstrap_err(y, _mu2,  rng=rng)
        mu3_err   = _bootstrap_err(y, _mu3,  rng=rng)
        S_err     = _bootstrap_err(y, _S,    rng=rng)
        alpha_err = _bootstrap_err(y, _alpha, rng=rng)
        beta_err  = _bootstrap_err(y, _beta,  rng=rng)
        mu3p_err  = _bootstrap_err(y, _mu3p,  rng=rng)
        Sp_err    = _bootstrap_err(y, _Sp,    rng=rng)

        for k, m, e in [
            ("mu1",      mu1,      mu1_err),
            ("mu2",      mu2,      mu2_err),
            ("mu3",      mu3,      mu3_err),
            ("S",        S,        S_err),
            ("alpha",    alpha,    alpha_err),
            ("beta",     beta,     beta_err),
            ("mu3_pred", mu3_pred, mu3p_err),
            ("S_pred",   S_pred,   Sp_err),
        ]:
            stats[k]["mean"].append(m)
            stats[k]["err"].append(e)
            stats[k]["n"].append(n)

    return stats, x_valid


# ── Graficas ──────────────────────────────────────────────────────────────────

def _errorbar_series(ax, x, means, errs, ns, colors, labels,
                     ref_line=None, ls="-", marker="o", zorder_base=3):
    """Dibuja puntos con barras de error y linea de conexion."""
    means = np.asarray(means, dtype=float)
    errs  = np.asarray(errs,  dtype=float)
    for i in range(len(x)):
        if not np.isfinite(means[i]):
            continue
        ax.errorbar(
            x[i], means[i], yerr=errs[i],
            fmt=marker, color=colors[i], markersize=6,
            elinewidth=1.5, capsize=4, capthick=1.5,
            linestyle="none",
            label=f"{labels[i]} (N={ns[i]})" if ns is not None else labels[i],
            zorder=zorder_base,
        )
    # linea de conexion
    ok = [i for i in range(len(x)) if np.isfinite(means[i])]
    if ok:
        ax.plot([x[i] for i in ok], [means[i] for i in ok],
                color="gray", linewidth=0.9, linestyle=ls, alpha=0.6, zorder=zorder_base - 1)
    if ref_line is not None:
        ax.axhline(ref_line, color="gray", linewidth=0.8, linestyle="--", zorder=1)


def plot_moments(groups, tag, outdir, charged):
    """
    Figura 1: los 3 momentos medidos (mu1, mu2, mu3 crudo).
    """
    stats, _ = class_stats(groups)
    particle_type = "cargadas" if charged else "neutras (ncl>0)"
    x = np.arange(len(LABELS))

    PANELS = [
        ("mu1",
         r"$\mu_1 = \langle\langle p_T \rangle\rangle$ [GeV/c]",
         r"Momento 1 — Media $\mu_1 = \langle\langle p_T \rangle\rangle$",
         None),
        ("mu2",
         r"$\mu_2 = \sigma^2_{\langle p_T \rangle}$ [GeV$^2$/c$^2$]",
         r"Momento 2 — Varianza $\mu_2 = \sigma^2$",
         None),
        ("mu3",
         r"$\mu_3$ [GeV$^3$/c$^3$]",
         r"Momento 3 — Tercer momento central $\mu_3$",
         0.0),
    ]

    fig, axs = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    fig.suptitle(
        f"Momentos de $P(\\langle p_T \\rangle)$ por centralidad — {tag}\n"
        f"partículas: {particle_type}  |  "
        r"momentos de la distribución evento a evento  |  err: SEM / bootstrap",
        fontsize=10
    )

    for ax, (key, ylabel, title, ref) in zip(axs, PANELS):
        means = stats[key]["mean"]
        errs  = stats[key]["err"]
        ns    = stats[key]["n"]
        _errorbar_series(ax, x, means, errs, ns, COLORS, LABELS, ref_line=ref)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9, pad=4)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
        ax.legend(fontsize=6.5, ncol=2, loc="best", framealpha=0.7, borderpad=0.4)

    axs[-1].set_xticks(x)
    axs[-1].set_xticklabels(LABELS, rotation=30, ha="right", fontsize=8)
    axs[-1].set_xlabel("Clase de centralidad", fontsize=9)

    fig.tight_layout()
    suffix = "charged" if charged else "neutral"
    path = outdir / f"pt_moments_{tag}_{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Guardado: {path}")


def plot_gamma(groups, tag, outdir, charged):
    """
    Figura 2: solo parámetros Gamma — alpha y beta vs centralidad.
    (El panel de skewness se eliminó; se muestra en pt_dist_* superpuesto.)
    """
    stats, _ = class_stats(groups)
    particle_type = "cargadas" if charged else "neutras (ncl>0)"
    x = np.arange(len(LABELS))

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Parámetros de la distribución Gamma — $P(\\langle p_T \\rangle)$ — {tag}\n"
        f"partículas: {particle_type}  |  "
        r"$\alpha = \mu_1^2/\mu_2$,  $\beta = \mu_2/\mu_1$  |  err: bootstrap",
        fontsize=10
    )

    # ── Panel izquierdo: alpha ────────────────────────────────────────────────
    ax = axs[0]
    _errorbar_series(ax, x,
                     stats["alpha"]["mean"], stats["alpha"]["err"],
                     stats["alpha"]["n"], COLORS, LABELS)
    ax.set_ylabel(r"$\alpha = \mu_1^2\,/\,\mu_2$", fontsize=10)
    ax.set_title(r"Parámetro de forma $\alpha$"
                 "\n(fuentes efectivas independientes)", fontsize=9, pad=4)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("Clase de centralidad", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(fontsize=6.5, ncol=2, loc="best", framealpha=0.7, borderpad=0.4)

    # ── Panel derecho: beta ───────────────────────────────────────────────────
    ax = axs[1]
    _errorbar_series(ax, x,
                     stats["beta"]["mean"], stats["beta"]["err"],
                     stats["beta"]["n"], COLORS, LABELS)
    ax.set_ylabel(r"$\beta = \mu_2\,/\,\mu_1$ [GeV/c]", fontsize=10)
    ax.set_title(r"Parámetro de escala $\beta$"
                 "\n(temperatura efectiva)", fontsize=9, pad=4)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("Clase de centralidad", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(fontsize=6.5, ncol=2, loc="best", framealpha=0.7, borderpad=0.4)

    fig.tight_layout()
    suffix = "charged" if charged else "neutral"
    path = outdir / f"pt_gamma_{tag}_{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Guardado: {path}")


def plot_pt_distributions(groups, tag, outdir, charged):
    """
    Figura 3 — pt_dist_*:
    Cuadrícula 2×5 (una celda por clase de centralidad).
    Cada celda muestra:
      · Histograma de {<pT>_i} en escala log (barras azules)
      · Función Gamma superpuesta (curva roja) con alpha y beta
        estimados a partir de mu1 y mu2 de la distribución
      · Recuadro de parámetros (estilo ROOT): chi2/ndf, alpha, beta, mu1
    """
    from scipy.stats import gamma as gamma_dist
    from scipy.special import gammaln

    stats, _ = class_stats(groups)
    particle_type = "cargadas" if charged else "neutras (ncl>0)"

    n_cols = 5
    n_rows = 2
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.6, n_rows * 3.8))
    fig.suptitle(
        f"Distribución de $\\langle p_T \\rangle$ por centralidad — {tag}\n"
        f"partículas: {particle_type}  |  "
        r"ajuste Gamma: $f(y;\alpha,\beta)=y^{\alpha-1}e^{-y/\beta}"
        r"/[\Gamma(\alpha)\beta^\alpha]$",
        fontsize=10
    )

    plot_idx = 0
    for ci, label in enumerate(LABELS):
        y_vals = np.array(groups[label]["pt_mean"])
        n_ev   = len(y_vals)
        row, col = divmod(plot_idx, n_cols)
        ax = axs[row][col]
        plot_idx += 1

        # ── Recuperar alpha y beta estimados ─────────────────────────────────
        alpha_m = stats["alpha"]["mean"][ci]
        beta_m  = stats["beta"]["mean"][ci]
        mu1_m   = stats["mu1"]["mean"][ci]

        color = COLORS[ci]

        if n_ev < 10 or not np.isfinite(alpha_m) or not np.isfinite(beta_m):
            ax.text(0.5, 0.5, f"{label}\nN = {n_ev}\n(sin datos)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.set_visible(True)
            continue

        # ── Histograma ────────────────────────────────────────────────────────
        n_bins = min(80, max(20, n_ev // 20))
        y_min, y_max = y_vals.min(), y_vals.max()
        margin = (y_max - y_min) * 0.05
        bin_edges = np.linspace(y_min - margin, y_max + margin, n_bins + 1)
        bin_w = bin_edges[1] - bin_edges[0]
        counts, _ = np.histogram(y_vals, bins=bin_edges)

        # Escala log: evitar log(0) poniendo floor en 0.5 para el plot
        ax.bar(bin_edges[:-1], np.where(counts > 0, counts, np.nan),
               width=bin_w, align="edge",
               color=color, alpha=0.35, edgecolor=color, linewidth=0.4,
               zorder=2, label="Datos")
        ax.set_yscale("log")

        # ── Curva Gamma ajustada ──────────────────────────────────────────────
        y_curve = np.linspace(y_min * 0.85, y_max * 1.15, 500)
        # scipy.stats.gamma usa (a=alpha, scale=beta)
        pdf_vals = gamma_dist.pdf(y_curve, a=alpha_m, scale=beta_m)
        # Normalizar al número de eventos × ancho de bin
        curve_scaled = pdf_vals * n_ev * bin_w
        ax.plot(y_curve, curve_scaled, color="red", linewidth=1.8,
                zorder=3, label="Gamma")

        # ── Chi² / ndf sobre los bins con counts > 0 ─────────────────────────
        bin_centers  = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        expected     = gamma_dist.pdf(bin_centers, a=alpha_m, scale=beta_m) \
                       * n_ev * bin_w
        mask_fit     = (counts > 0) & (expected > 0)
        if mask_fit.sum() > 2:
            chi2 = float(np.sum((counts[mask_fit] - expected[mask_fit])**2
                                / expected[mask_fit]))
            ndf  = int(mask_fit.sum()) - 2   # 2 parámetros libres
            chi2_str = f"{chi2:.2f} / {ndf}"
        else:
            chi2_str = "—"

        # ── Recuadro de parámetros (estilo ROOT) ─────────────────────────────
        textstr = (
            f"$\\chi^2$/ndf  {chi2_str}\n"
            f"$\\alpha$       {alpha_m:.1f} ± {stats['alpha']['err'][ci]:.1f}\n"
            f"$\\beta$        {beta_m:.5f} ± {stats['beta']['err'][ci]:.5f}\n"
            f"$\\mu_1$        {mu1_m:.5f}"
        )
        ax.text(0.97, 0.97, textstr,
                transform=ax.transAxes,
                fontsize=6.2, verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="gray", alpha=0.85),
                family="monospace", zorder=5)

        # ── Estética ──────────────────────────────────────────────────────────
        ax.set_title(f"{label}  (N = {n_ev})", fontsize=8, pad=3)
        ax.set_xlabel(r"$\langle p_T \rangle$ [GeV/c]", fontsize=7.5)
        ax.set_ylabel("Eventos", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.legend(fontsize=6.5, loc="upper left", framealpha=0.7)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5, which="both")
        # Limitar eje y inferior a 0.5 para que la escala log sea limpia
        y_top = ax.get_ylim()[1]
        ax.set_ylim(0.5, y_top * 3)

    fig.tight_layout()
    suffix = "charged" if charged else "neutral"
    path = outdir / f"pt_dist_{tag}_{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Guardado: {path}")



    stats, _ = class_stats(groups)
    hdr = (f"  {'Clase':10s} | {'Nev':6s} | "
           f"{'mu1 [GeV/c]':>16s} | "
           f"{'mu2 [GeV2/c2]':>16s} | "
           f"{'mu3 [GeV3/c3]':>16s} | "
           f"{'S (medido)':>13s} | "
           f"{'alpha':>12s} | "
           f"{'beta [GeV/c]':>14s} | "
           f"{'S (Gamma)':>13s}")
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    for idx, label in enumerate(LABELS):
        n = stats["mu1"]["n"][idx]
        if n < 10:
            continue
        def fv(k, fmt=".5f"):
            m, e = stats[k]["mean"][idx], stats[k]["err"][idx]
            return f"{m:{fmt}}±{e:{fmt}}" if np.isfinite(m) else "     —     "
        print(f"  {label:10s} | {n:6d} | "
              f"{fv('mu1'):>16s} | {fv('mu2'):>16s} | {fv('mu3'):>16s} | "
              f"{fv('S', '.4f'):>13s} | {fv('alpha', '.3f'):>12s} | "
              f"{fv('beta', '.5f'):>14s} | {fv('S_pred', '.4f'):>13s}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def print_summary(groups):
    stats, _ = class_stats(groups)
    hdr = (f"  {'Clase':10s} | {'Nev':6s} | "
           f"{'mu1 [GeV/c]':>16s} | "
           f"{'mu2 [GeV2/c2]':>16s} | "
           f"{'mu3 [GeV3/c3]':>16s} | "
           f"{'S (medido)':>13s} | "
           f"{'alpha':>12s} | "
           f"{'beta [GeV/c]':>14s} | "
           f"{'S (Gamma)':>13s}")
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    for idx, label in enumerate(LABELS):
        n = stats["mu1"]["n"][idx]
        if n < 10:
            continue
        def fv(k, fmt=".5f"):
            m, e = stats[k]["mean"][idx], stats[k]["err"][idx]
            return f"{m:{fmt}}±{e:{fmt}}" if np.isfinite(m) else "     —     "
        print(f"  {label:10s} | {n:6d} | "
              f"{fv('mu1'):>16s} | {fv('mu2'):>16s} | {fv('mu3'):>16s} | "
              f"{fv('S', '.4f'):>13s} | {fv('alpha', '.3f'):>12s} | "
              f"{fv('beta', '.5f'):>14s} | {fv('S_pred', '.4f'):>13s}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Primeros 3 momentos de pT por clase de centralidad"
    )
    parser.add_argument(
        "run_dirs", nargs="+",
        help="Uno o mas directorios con subdirectorios run_NNNN/ "
             "(ej. output/Bi_11GeV_MB* o varios paths)"
    )
    parser.add_argument("--charged", action="store_true",
                        help="Particulas cargadas (chg!=0); default: neutras (ncl>0, chg==0)")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Limite total de eventos entre todas las carpetas")
    parser.add_argument("--outdir", default=None,
                        help="Directorio de salida para la grafica")
    parser.add_argument("--tag", default=None,
                        help="Etiqueta para el nombre de archivo de salida")
    args = parser.parse_args()

    run_dirs = [Path(d) for d in args.run_dirs]
    for d in run_dirs:
        if not d.exists():
            sys.exit(f"ERROR: no existe {d}")

    if args.tag:
        tag = args.tag
    elif len(run_dirs) == 1:
        tag = run_dirs[0].name
    else:
        tag = f"{run_dirs[0].name}_combined"

    outdir = Path(args.outdir) if args.outdir else run_dirs[0] / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    if len(run_dirs) == 1:
        print(f"Leyendo eventos de {run_dirs[0]} ...")
    else:
        print(f"Leyendo eventos de {len(run_dirs)} carpetas:")
        for d in run_dirs:
            print(f"  - {d}")

    groups = collect_moments(run_dirs, charged=args.charged,
                             max_events=args.max_events)

    total = sum(len(g["pt_mean"]) for g in groups.values())
    if total == 0:
        sys.exit("ERROR: no se encontraron eventos con particulas filtradas")
    print(f"Total de eventos con <pT> definido: {total}")

    print_summary(groups)
    print("Generando graficas...")
    plot_moments(groups, tag, outdir, args.charged)
    plot_gamma(groups, tag, outdir, args.charged)
    plot_pt_distributions(groups, tag, outdir, args.charged)
    print("Listo.")


if __name__ == "__main__":
    main()