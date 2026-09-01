#!/usr/bin/env python3
"""
pt_correlator_norm.py — v2: reproduce la Figura 4.12 (observable
    normalizado, Ec. 4.8) separando partículas por ncl en PRIMARIAS
    (ncl==1) y SECUNDARIAS (ncl>1), en vez de usar cortes de |eta|.

    *** Los cortes de |eta| < 1.0 se retiraron a petición explícita —
    no se usan en esta versión. Si más adelante se necesitan de vuelta,
    es un cambio acotado (reintroducir CUT_VARIANTS de la v1). ***

Clasificación según ncl (número de colisiones sufridas):
    ncl == 0   ->  espectadoras   (excluidas, como siempre)
    ncl == 1   ->  primarias
    ncl >  1   ->  secundarias

Observable (Ec. 4.8):
    sqrt(<N> * <Delta p_t,i Delta p_t,j>) / <<pt>>

donde <Delta p_t,i Delta p_t,j> se calcula igual que en pt_correlator.py
(Ec. A.16), y <N> es la multiplicidad promedio de la clase de
centralidad, bajo la misma selección de partículas (primarias o
secundarias).

Uso:
    python3 analysis/pt_correlator_norm.py output/Bi_11GeV_MB* --charged \\
        --n-min 10 --n-boot 500
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_f14 import iter_all


# ═══════════════════════════════════════════════════════════════════════════
#  0. CENTRALIDAD
# ═══════════════════════════════════════════════════════════════════════════

B_CUTS = [
    (0.0,   4.0,  "0-10%",   0),
    (4.0,   6.0,  "10-20%",  1),
    (6.0,   7.0,  "20-30%",  2),
    (7.0,   8.0,  "30-40%",  3),
    (8.0,   9.0,  "40-50%",  4),
    (9.0,  10.0,  "50-60%",  5),
    (10.0, 11.0,  "60-70%",  6),
    (11.0, 12.0,  "70-80%",  7),
    (12.0, 13.5,  "80-90%",  8),
    (13.5, np.inf,"90-100%", 9),
]
CENT_LABELS = [c[2] for c in B_CUTS]
CENT_X      = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

def assign_centrality(b):
    for b_lo, b_hi, label, idx in B_CUTS:
        if b_lo <= b < b_hi:
            return idx
    return 9


# ═══════════════════════════════════════════════════════════════════════════
#  1. CLASES DE PARTÍCULA POR ncl (reemplaza las variantes de |eta|)
# ═══════════════════════════════════════════════════════════════════════════

PARTICLE_CLASSES = [
    {"key": "primary",   "label": r"UrQMD Bi+Bi $\sqrt{s_{NN}}$=11 GeV, primarias (ncl=1)",
     "color": "tab:blue"},
    {"key": "secondary", "label": r"UrQMD Bi+Bi $\sqrt{s_{NN}}$=11 GeV, secundarias (ncl>1)",
     "color": "tab:orange"},
]

def ncl_mask(ncl, key):
    if key == "primary":
        return ncl == 1
    elif key == "secondary":
        return ncl > 1
    raise ValueError(key)


# ═══════════════════════════════════════════════════════════════════════════
#  2. CARGA — una sola pasada, estadísticas por evento para cada clase
# ═══════════════════════════════════════════════════════════════════════════

def load_event_stats_by_particle_class(output_dirs, charged=False, n_min=10,
                                        max_events=None):
    if isinstance(output_dirs, (str, Path)):
        output_dirs = [output_dirs]

    data = {c["key"]: {ci: {"mean": [], "var": [], "n": []}
                        for ci in range(len(CENT_LABELS))}
            for c in PARTICLE_CLASSES}

    n_total = 0

    for output_dir in output_dirs:
        output_dir = Path(output_dir)
        remaining  = None if max_events is None else max_events - n_total
        if remaining is not None and remaining <= 0:
            break

        print(f"  Leyendo: {output_dir} ...", flush=True)
        for ev in iter_all(output_dir, max_events=remaining):
            chg_mask = (ev["chg"] != 0) if charged else (ev["chg"] == 0)
            ncl = ev["ncl"]
            pt_all = ev["pT"]
            n_total += 1
            ci = assign_centrality(float(ev["b"]))

            for c in PARTICLE_CLASSES:
                m = chg_mask & ncl_mask(ncl, c["key"])
                pt = pt_all[m]
                N = len(pt)
                if N >= n_min and np.all(np.isfinite(pt)):
                    mean_k = pt.mean()
                    var_k  = np.mean((pt - mean_k) ** 2)
                    bucket = data[c["key"]][ci]
                    bucket["mean"].append(mean_k)
                    bucket["var"].append(var_k)
                    bucket["n"].append(N)

            if n_total % 10000 == 0:
                print(f"    {n_total} eventos procesados...", flush=True)
            if max_events is not None and n_total >= max_events:
                break

    for c in PARTICLE_CLASSES:
        for ci in range(len(CENT_LABELS)):
            b = data[c["key"]][ci]
            data[c["key"]][ci] = {k: np.array(v, dtype=np.float64)
                                   for k, v in b.items()}

    print(f"  Total eventos leídos: {n_total}")
    return data


# ═══════════════════════════════════════════════════════════════════════════
#  3. OBSERVABLES (Ec. A.16 + Ec. 4.8)
# ═══════════════════════════════════════════════════════════════════════════

def class_observables(mean_arr, var_arr, n_arr):
    mu1 = mean_arr.mean()
    n_mean = n_arr.mean()
    sigma2_ens = mean_arr.var()
    sigma2_over_N = np.mean(var_arr / n_arr)
    corr = sigma2_ens - sigma2_over_N

    with np.errstate(invalid="ignore"):
        norm_obs = np.sqrt(n_mean * corr) / mu1 if corr > 0 else np.nan

    return mu1, n_mean, corr, norm_obs


def jackknife_error(mean_arr, var_arr, n_arr):
    """
    Error por jackknife delete-1, vectorizado. Se prefiere sobre
    bootstrap para este observable por la misma razón que en
    pt_correlator.py: sqrt(<N> * corr)/mu1 es una razón que amplifica
    el ruido cuando corr es pequeño/ruidoso, y jackknife es más estable
    para ese caso que el bootstrap (ver Adams et al. 2005, STAR).
    """
    M = len(mean_arr)
    if M < 5:
        return np.nan
    M1 = M - 1

    S1  = mean_arr.sum()
    S2  = (mean_arr ** 2).sum()
    Svn = (var_arr / n_arr).sum()
    Sn  = n_arr.sum()

    mu1_loo   = (S1 - mean_arr) / M1
    sig2_loo  = (S2 - mean_arr ** 2) / M1 - mu1_loo ** 2
    sigoN_loo = (Svn - var_arr / n_arr) / M1
    corr_loo  = sig2_loo - sigoN_loo
    nmean_loo = (Sn - n_arr) / M1

    with np.errstate(invalid="ignore"):
        norm_loo = np.where(corr_loo > 0,
                             np.sqrt(nmean_loo * np.abs(corr_loo)) / mu1_loo,
                             np.nan)

    theta_bar = np.nanmean(norm_loo)
    return np.sqrt((M1 / M) * np.nansum((norm_loo - theta_bar) ** 2))


def compute_class_results(data_c, n_boot=None, seed=None):
    results = []
    for ci, label_ci in enumerate(CENT_LABELS):
        d = data_c[ci]
        if len(d["mean"]) < 5:
            results.append(None)
            continue
        mu1, n_mean, corr, norm_obs = class_observables(d["mean"], d["var"], d["n"])
        err = jackknife_error(d["mean"], d["var"], d["n"])
        results.append({"x": CENT_X[ci], "y": norm_obs, "yerr": err,
                         "n_events": len(d["mean"])})
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  4. FIGURA 4.12 (primarias vs secundarias, sin cortes de eta)
# ═══════════════════════════════════════════════════════════════════════════

def plot_figure(all_results, tag, outdir):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for c in PARTICLE_CLASSES:
        res = all_results[c["key"]]
        x    = [r["x"]    for r in res if r is not None]
        y    = [r["y"]    for r in res if r is not None]
        yerr = [r["yerr"] for r in res if r is not None]
        ax.errorbar(x, y, yerr=yerr, fmt="o-", color=c["color"],
                    label=c["label"], capsize=3, markersize=6)

    ax.set_xlabel("Centrality(%)")
    ax.set_ylabel(r"$\sqrt{\langle N\rangle\langle\Delta p_{t,i}\Delta p_{t,j}"
                   r"\rangle}/\langle\langle p_t\rangle\rangle$ (%)")
    ax.set_title("Figura 4.12 (reproducción) — primarias vs. secundarias",
                 fontsize=10)
    ax.grid(ls=":", alpha=0.4)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    path = outdir / f"figura_4_12_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Guardado: {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  5. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Figura 4.12 (solo UrQMD Bi+Bi 11 GeV), "
                     "separando primarias (ncl=1) y secundarias (ncl>1). "
                     "Los cortes de |eta| NO se usan en esta versión."
    )
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--charged",    action="store_true")
    parser.add_argument("--n-min",      type=int, default=10)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--n-boot",     type=int, default=None,
                        help="[Obsoleto] Ya no se usa bootstrap; los errores "
                             "se calculan por jackknife (determinista).")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--outdir",     default=None)
    parser.add_argument("--tag",        default=None)
    args = parser.parse_args()

    run_dirs = [Path(d) for d in args.run_dirs]
    for d in run_dirs:
        if not d.exists():
            sys.exit(f"ERROR: no existe {d}")

    tag = args.tag or (run_dirs[0].name if len(run_dirs) == 1
                        else f"{run_dirs[0].name}_combined")
    tag += f"_nmin{args.n_min}"

    outdir = Path(args.outdir) if args.outdir else run_dirs[0] / "correlator_plots"
    outdir.mkdir(parents=True, exist_ok=True)

    particle_type = "cargadas" if args.charged else "neutras"
    print("=" * 65)
    print(f"Observable normalizado sqrt(<N><dp dp>)/<<pt>>, por ncl  |  "
          f"partículas: {particle_type}")
    print(f"n_min={args.n_min}  n_boot={args.n_boot}")
    print("=" * 65)

    print("\n[1] Cargando eventos (una sola pasada, primarias + secundarias)...")
    data = load_event_stats_by_particle_class(
        run_dirs, charged=args.charged, n_min=args.n_min,
        max_events=args.max_events)

    print("\n[2] Calculando observable normalizado por clase de partícula...")
    all_results = {}
    for c in PARTICLE_CLASSES:
        print(f"  Clase: {c['label']}")
        res = compute_class_results(data[c["key"]])
        all_results[c["key"]] = res
        for label_ci, r in zip(CENT_LABELS, res):
            if r is None:
                print(f"    {label_ci:<9} (insuficientes eventos)")
            else:
                print(f"    {label_ci:<9} N_ev={r['n_events']:>6}  "
                      f"obs={r['y']:.3f} ± {r['yerr']:.3f}")

    print("\n[3] Generando Figura 4.12 (primarias vs. secundarias)...")
    plot_figure(all_results, tag, outdir)

    print(f"\nListo. Resultados en: {outdir}")


if __name__ == "__main__":
    main()
