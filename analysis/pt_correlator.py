#!/usr/bin/env python3
"""
pt_correlator.py — v2: separa partículas por número de colisiones (ncl)
                   en PRIMARIAS (ncl==1) y SECUNDARIAS (ncl>1), y genera
                   las Figuras 5.1 y 5.2 (Apéndice A de la tesis de
                   referencia) por separado para cada clase de partícula.

Clasificación según ncl (número de colisiones sufridas):
    ncl == 0   ->  espectadoras   (excluidas del análisis, como siempre
                                    se ha hecho con el corte ncl>0)
    ncl == 1   ->  primarias      (participaron en exactamente 1 colisión)
    ncl >  1   ->  secundarias    (productos de colisiones/reacciones
                                    múltiples)

El resto de la física no cambia respecto a v1:

    <Delta p_t,i Delta p_t,j> = sigma^2_<pt>  -  < sigma^2_pt / N >   (Ec. A.16)
    S  = <(<pt>_k - <<pt>>)^3> / sigma_<pt>^3                        (Ec. 2.3)
    C_V ~ <<pt>> / sqrt(<Delta p_t,i Delta p_t,j>)                    (Ec. 4.7)

pero calculado por separado para el subconjunto de partículas primarias
y para el subconjunto de secundarias, en una sola pasada por los datos
(no se relee el archivo dos veces).

Uso:
    python3 analysis/pt_correlator.py output/Bi_11GeV_MB* --charged \\
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
#  1. CLASES DE PARTÍCULA POR ncl
# ═══════════════════════════════════════════════════════════════════════════

PARTICLE_CLASSES = [
    {"key": "primary",   "label": "Primarias (ncl=1)",  "color": "tab:blue"},
    {"key": "secondary", "label": "Secundarias (ncl>1)", "color": "tab:orange"},
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
    """
    Retorna: data[class_key][ci] = dict con arrays mean, var, n
    """
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
        print(f"  Clase: {c['label']}")
        for ci, label in enumerate(CENT_LABELS):
            b = data[c["key"]][ci]
            data[c["key"]][ci] = {k: np.array(v, dtype=np.float64)
                                   for k, v in b.items()}
            print(f"    {label}: {len(b['mean'])} eventos (N_i >= {n_min})")

    print(f"  Total eventos leídos: {n_total}")
    return data


# ═══════════════════════════════════════════════════════════════════════════
#  3. OBSERVABLES POR CLASE DE CENTRALIDAD (Ec. A.16, 2.3, 4.7)
# ═══════════════════════════════════════════════════════════════════════════

def class_observables(mean_arr, var_arr, n_arr):
    mu1 = mean_arr.mean()
    sigma2_ens = mean_arr.var()
    sigma2_over_N = np.mean(var_arr / n_arr)
    corr = sigma2_ens - sigma2_over_N

    if sigma2_ens > 0:
        sigma_ens = np.sqrt(sigma2_ens)
        skew = np.mean((mean_arr - mu1) ** 3) / sigma_ens ** 3
    else:
        skew = np.nan

    with np.errstate(invalid="ignore"):
        cv = mu1 / np.sqrt(corr) if corr > 0 else np.nan

    return mu1, sigma2_ens, corr, skew, cv


def jackknife_errors(mean_arr, var_arr, n_arr):
    """
    Errores por jackknife delete-1, vectorizado (sin recomputar sumas
    desde cero por cada evento excluido -> O(M), no O(M^2)).

    Se prefiere sobre bootstrap para este observable específico: la
    literatura de <Delta p_t,i Delta p_t,j> (p.ej. Adams et al. 2005,
    STAR) usa jackknife precisamente porque el bootstrap tiende a ser
    inestable/sobreestimar el error cuando la cantidad de interés es
    una razón con un denominador pequeño o ruidoso (como C_V, que
    divide entre la raíz de una diferencia de dos términos similares).
    """
    M = len(mean_arr)
    if M < 5:
        return (np.nan,) * 4
    M1 = M - 1

    S1  = mean_arr.sum()
    S2  = (mean_arr ** 2).sum()
    S3c = (mean_arr ** 3).sum()
    Svn = (var_arr / n_arr).sum()

    mu1_loo    = (S1 - mean_arr) / M1
    sig2_loo   = (S2 - mean_arr ** 2) / M1 - mu1_loo ** 2
    sigoN_loo  = (Svn - var_arr / n_arr) / M1
    corr_loo   = sig2_loo - sigoN_loo

    m3_loo = ((S3c - mean_arr ** 3) / M1
              - 3 * mu1_loo * (S2 - mean_arr ** 2) / M1
              + 2 * mu1_loo ** 3)
    with np.errstate(invalid="ignore"):
        sigma_loo = np.sqrt(np.where(sig2_loo > 0, sig2_loo, np.nan))
        skew_loo  = m3_loo / sigma_loo ** 3
        cv_loo    = np.where(corr_loo > 0, mu1_loo / np.sqrt(np.abs(corr_loo)), np.nan)

    def jack_se(theta_loo):
        theta_bar = np.nanmean(theta_loo)
        return np.sqrt((M1 / M) * np.nansum((theta_loo - theta_bar) ** 2))

    return (jack_se(mu1_loo), jack_se(corr_loo),
            jack_se(skew_loo), jack_se(cv_loo))


def compute_results(data_c, n_boot=None, seed=None):
    results = []
    for ci, label_ci in enumerate(CENT_LABELS):
        d = data_c[ci]
        if len(d["mean"]) < 5:
            continue
        mu1, sigma2_ens, corr, skew, cv = class_observables(
            d["mean"], d["var"], d["n"])
        mu1_e, corr_e, skew_e, cv_e = jackknife_errors(
            d["mean"], d["var"], d["n"])
        results.append({
            "label": label_ci, "x": CENT_X[ci], "n_events": len(d["mean"]),
            "mu1": mu1, "mu1_e": mu1_e,
            "corr": corr, "corr_e": corr_e,
            "skew": skew, "skew_e": skew_e,
            "cv": cv, "cv_e": cv_e,
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  4. FIGURAS 5.1 y 5.2 (por clase de partícula)
# ═══════════════════════════════════════════════════════════════════════════

def make_figures(results, label, tag, outdir):
    x      = [r["x"]      for r in results]
    mu1    = [r["mu1"]    for r in results]
    mu1_e  = [r["mu1_e"]  for r in results]
    corr   = [r["corr"]   for r in results]
    corr_e = [r["corr_e"] for r in results]
    skew   = [r["skew"]   for r in results]
    skew_e = [r["skew_e"] for r in results]
    cv     = [r["cv"]     for r in results]
    cv_e   = [r["cv_e"]   for r in results]

    fig, axs = plt.subplots(3, 1, figsize=(6.5, 11))

    axs[0].errorbar(x, mu1, yerr=mu1_e, fmt="o-", color="tab:blue",
                     label=label, capsize=3)
    axs[0].set_ylabel(r"$\langle\langle p_t\rangle\rangle$ (GeV/c)")
    axs[0].set_title("(a) Distribución de " r"$\langle\langle p_t\rangle\rangle$"
                      " para distintos rangos de centralidad.", fontsize=9)

    axs[1].errorbar(x, corr, yerr=corr_e, fmt="o-", color="tab:blue",
                     label=label, capsize=3)
    axs[1].set_ylabel(r"$\langle\Delta p_{t,i}\Delta p_{t,j}\rangle$ (GeV/c)$^2$")
    axs[1].set_title("(b) Segundo momento correspondiente a las fluctuaciones "
                      r"dinámicas de $\langle p_t\rangle$.", fontsize=9)

    axs[2].errorbar(x, skew, yerr=skew_e, fmt="o-", color="tab:blue",
                     label=label, capsize=3)
    axs[2].axhline(0, color="black", lw=0.8)
    axs[2].set_ylabel("Skewness")
    axs[2].set_title(r"(c) Asimetría de las distribuciones de $\langle p_t\rangle$ "
                      "para distintos rangos de centralidad.", fontsize=9)

    for ax in axs:
        ax.set_xlabel("Centrality(%)")
        ax.grid(ls=":", alpha=0.4)
        ax.legend(fontsize=8)

    fig.suptitle(f"Figura 5.1 (reproducción) — {label}", fontsize=10)
    fig.tight_layout()
    path = outdir / f"figura_5_1_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Guardado: {path}")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(x, cv, yerr=cv_e, fmt="o-", color="tab:blue",
                label=label, capsize=3)
    ax.set_xlabel("Centrality(%)")
    ax.set_ylabel(r"$C_V \propto \langle\langle p_t\rangle\rangle "
                   r"/\sqrt{\langle\Delta p_{t,i}\Delta p_{t,j}\rangle}$")
    ax.set_title(f"Figura 5.2 (reproducción) — {label}", fontsize=10)
    ax.grid(ls=":", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = outdir / f"figura_5_2_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Guardado: {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  5. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Figuras 5.1 y 5.2 por separado para partículas "
                     "primarias (ncl=1) y secundarias (ncl>1)."
    )
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--charged",    action="store_true")
    parser.add_argument("--n-min",      type=int, default=10)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--n-boot",     type=int, default=None,
                        help="[Obsoleto] Ya no se usa bootstrap; los errores "
                             "se calculan por jackknife (determinista). Se "
                             "mantiene el flag solo para no romper comandos "
                             "anteriores.")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--label-base", default="UrQMD Bi+Bi 11 GeV",
                        help="Prefijo de etiqueta para las leyendas.")
    parser.add_argument("--outdir",     default=None)
    parser.add_argument("--tag",        default=None)
    args = parser.parse_args()

    run_dirs = [Path(d) for d in args.run_dirs]
    for d in run_dirs:
        if not d.exists():
            sys.exit(f"ERROR: no existe {d}")

    tag_base = args.tag or (run_dirs[0].name if len(run_dirs) == 1
                             else f"{run_dirs[0].name}_combined")
    tag_base += f"_nmin{args.n_min}"

    outdir = Path(args.outdir) if args.outdir else run_dirs[0] / "correlator_plots"
    outdir.mkdir(parents=True, exist_ok=True)

    particle_type = "cargadas" if args.charged else "neutras"
    print("=" * 65)
    print(f"Correlador <Delta p_t,i Delta p_t,j>, separado por ncl  |  "
          f"partículas: {particle_type}")
    print(f"n_min={args.n_min}  n_boot={args.n_boot}")
    print("=" * 65)

    print("\n[1] Cargando eventos (una sola pasada, primarias + secundarias)...")
    data = load_event_stats_by_particle_class(
        run_dirs, charged=args.charged, n_min=args.n_min,
        max_events=args.max_events)

    for c in PARTICLE_CLASSES:
        print(f"\n[2] Calculando observables — {c['label']}...")
        results = compute_results(data[c["key"]])
        if len(results) == 0:
            print(f"  Sin datos suficientes para {c['label']}; se omite.")
            continue
        for r in results:
            print(f"  {r['label']:<9} N_ev={r['n_events']:>6}  "
                  f"<<pt>>={r['mu1']:.4f}±{r['mu1_e']:.4f}  "
                  f"<dpdp>={r['corr']:.3e}±{r['corr_e']:.3e}  "
                  f"S={r['skew']:.3f}±{r['skew_e']:.3f}  "
                  f"C_V={r['cv']:.3f}±{r['cv_e']:.3f}")

        label = f"{args.label_base}, {c['label']}"
        tag = f"{tag_base}_{c['key']}"
        make_figures(results, label, tag, outdir)

    print(f"\nListo. Resultados en: {outdir}")


if __name__ == "__main__":
    main()