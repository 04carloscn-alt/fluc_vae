#!/usr/bin/env python3
"""
fluc_observables.py — Fluctuaciones de multiplicidad y pT por clase de centralidad.

Adapta flucK.ipynb para leer archivos f14 directamente en lugar de un CSV.
Centralidad asignada por parametro de impacto b (cortes del notebook).

Filtro de particulas (igual al notebook):
    ncl > 0  (particulas primarias, al menos una colision)
    chg == 0 (particulas neutras; cambiar a chg != 0 para cargadas)

Graficas producidas:
    1. Distribucion de n_particulas por centralidad
    2. Media de n_particulas por evento y centralidad
    3. Distribucion de pT por centralidad (normalizada a n_eventos)
    4. pT medio por evento y centralidad
    + tabla resumen por centralidad

Uso:
    python3 fluc/fluc_observables.py output/Bi_11GeV_MB
    python3 fluc/fluc_observables.py output/Bi_11GeV_MB --charged
    python3 fluc/fluc_observables.py output/Bi_11GeV_MB --max-events 500

    # Combinar varias carpetas (ej. Bi_11GeV_MB, Bi_11GeV_MB1, ... Bi_11GeV_MB5):
    python3 analysis/fluc_observables.py output/Bi_11GeV_MB output/Bi_11GeV_MB1 \\
        output/Bi_11GeV_MB2 output/Bi_11GeV_MB3 output/Bi_11GeV_MB4 \\
        output/Bi_11GeV_MB5 --charged --outdir output/Bi_11GeV_MB_combined/plots

    # O usando un glob (la shell expande el patron):
    python3 analysis/fluc_observables.py output/Bi_11GeV_MB* --charged
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from read_f14 import iter_all


# Cortes en b [fm] -> etiqueta de centralidad (del notebook)
B_CUTS = [
    (0.0,  4.0,  "0-10%"),
    (4.0,  6.0,  "10-20%"),
    (6.0,  7.0,  "20-30%"),
    (7.0,  8.0,  "30-40%"),
    (8.0,  9.0,  "40-50%"),
    (9.0,  10.0, "50-60%"),
    (10.0, 11.0, "60-70%"),
    (11.0, 12.0, "70-80%"),
    (12.0, 13.5, "80-90%"),
    (13.5, np.inf, "90-100%"),
]

COLORS = [
    "#d62728", "#ff7f0e", "#f7c617", "#2ca02c",
    "#17becf", "#1f77b4", "#9467bd", "#8c564b", "#bcbd22", "#7f7f7f",
]


def assign_centrality(b):
    for b_lo, b_hi, label in B_CUTS:
        if b_lo <= b < b_hi:
            return label
    return "90-100%"


def collect_event_data(output_dirs, charged=False, max_events=None):
    """
    Lee todos los f14 de una o varias carpetas y devuelve una lista de dicts
    por evento (combinando todas las carpetas):
        label     : clase de centralidad (str)
        b         : parametro de impacto (fm)
        n_part    : numero de particulas que pasan el filtro
        pt_values : array de pT de esas particulas
        pt_mean   : pT medio del evento

    output_dirs : Path o lista de Path, cada uno con subdirectorios run_NNNN/
    max_events  : limite TOTAL de eventos combinados entre todas las carpetas
                  (None = sin limite, lee todo lo disponible)
    """
    if isinstance(output_dirs, (str, Path)):
        output_dirs = [output_dirs]

    events = []
    n = 0
    for output_dir in output_dirs:
        output_dir = Path(output_dir)
        remaining = None if max_events is None else max_events - n
        if remaining is not None and remaining <= 0:
            break

        print(f"  Leyendo carpeta: {output_dir} ...", flush=True)
        for ev in iter_all(output_dir, max_events=remaining):
            if charged:
                mask = (ev["ncl"] > 0) & (ev["chg"] != 0)
            else:
                mask = (ev["ncl"] > 0) & (ev["chg"] == 0)

            pt_vals = ev["pT"][mask]
            events.append({
                "label":    assign_centrality(ev["b"]),
                "b":        ev["b"],
                "n_part":   int(mask.sum()),
                "pt_values": pt_vals,
                "pt_mean":  float(pt_vals.mean()) if len(pt_vals) > 0 else np.nan,
            })
            n += 1
            if n % 500 == 0:
                print(f"  {n} eventos procesados...", flush=True)

            if max_events is not None and n >= max_events:
                break
    return events


def group_by_centrality(events):
    """Agrupa la lista de eventos por etiqueta de centralidad."""
    groups = {label: [] for _, _, label in B_CUTS}
    for ev in events:
        groups[ev["label"]].append(ev)
    return groups


def print_summary(groups):
    print(f"\n  {'Clase':10s} | {'N eventos':10s} | {'<n_part>':10s} | {'<pT> [GeV/c]':12s}")
    print("  " + "-" * 50)
    for _, _, label in B_CUTS:
        evs = groups[label]
        if not evs:
            continue
        n_ev = len(evs)
        mean_n = np.mean([e["n_part"] for e in evs])
        mean_pt = np.nanmean([e["pt_mean"] for e in evs])
        print(f"  {label:10s} | {n_ev:10d} | {mean_n:10.1f} | {mean_pt:12.4f}")
    print()


def plot_panels(groups, tag, outdir, charged):
    labels = [label for _, _, label in B_CUTS]
    fig, axs = plt.subplots(2, 2, figsize=(13, 10))
    particle_type = "cargadas" if charged else "neutras (ncl>0)"
    fig.suptitle(f"Fluctuaciones — {tag}  |  particulas: {particle_type}", fontsize=11)

    for idx, label in enumerate(labels):
        evs = groups[label]
        if not evs:
            continue
        color = COLORS[idx]
        n_ev = len(evs)
        n_parts = np.array([e["n_part"] for e in evs])
        pt_means = np.array([e["pt_mean"] for e in evs])
        pt_all = np.concatenate([e["pt_values"] for e in evs])

        # 1. Distribucion de n_particulas (excluyendo n_part == 0 para evitar
        #    el pico artificial en cero; ese 0 no representa una fluctuacion
        #    fisica del observable, sino eventos sin particulas que pasan el filtro)
        n_parts_nz = n_parts[n_parts > 0]
        if len(n_parts_nz) > 0:
            w = np.ones(len(n_parts_nz)) / n_ev
            axs[0, 0].hist(n_parts_nz, bins=30, histtype="step", color=color,
                           label=label, weights=w)

        # 2. Vela japonesa de n_particulas por clase de centralidad
        #    Mecha inferior/superior : min / max
        #    Cuerpo (rect)           : Q1 / Q3
        #    Linea interior          : mediana
        #    Cruz (x)                : media
        q1, med, q3 = np.percentile(n_parts, [25, 50, 75])
        n_min, n_max = n_parts.min(), n_parts.max()
        mean_n = n_parts.mean()
        w_candle = 0.4  # ancho de la vela en unidades del eje x

        # Mecha completa (min -> max)
        axs[0, 1].plot([idx, idx], [n_min, n_max],
                       color=color, linewidth=1.2, zorder=2)
        # Cuerpo Q1-Q3
        rect = plt.Rectangle(
            (idx - w_candle / 2, q1), w_candle, q3 - q1,
            facecolor=color, edgecolor=color, alpha=0.75, zorder=3, label=label
        )
        axs[0, 1].add_patch(rect)
        # Linea de mediana
        axs[0, 1].plot([idx - w_candle / 2, idx + w_candle / 2], [med, med],
                       color="white", linewidth=1.5, zorder=4)
        # Marca de media
        axs[0, 1].plot(idx, mean_n, marker="x", color="white",
                       markersize=5, markeredgewidth=1.5, zorder=5)

        # 3. Distribucion de pT (normalizada a n_eventos)
        if len(pt_all) > 0:
            w_pt = np.ones(len(pt_all)) / n_ev
            pt_max = np.percentile(pt_all, 99.5)
            axs[1, 0].hist(pt_all, bins=80, range=(0, pt_max), histtype="step",
                           color=color, label=label, weights=w_pt)

        # 4. pT medio por evento
        valid = pt_means[np.isfinite(pt_means)]
        if len(valid) > 0:
            w_pm = np.ones(len(valid)) / n_ev
            axs[1, 1].hist(valid, bins=40, histtype="step", color=color,
                           label=label, weights=w_pm)

    axs[0, 0].set_xlabel("Numero de particulas por evento (n_part > 0)")
    axs[0, 0].set_ylabel("Frecuencia (norm. a N eventos)")
    axs[0, 0].set_title("Distribucion de n_particulas")
    axs[0, 0].legend(title="Centralidad", fontsize=7, ncol=2)

    axs[0, 1].set_xlabel("Clase de centralidad")
    axs[0, 1].set_ylabel(r"$n_{\rm part}$")
    axs[0, 1].set_title(
        r"$n_{\rm part}$ por clase  —  vela: min/Q1/med/Q3/max,  $\times$=media"
    )
    axs[0, 1].set_xticks(range(len(labels)))
    axs[0, 1].set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    axs[0, 1].set_xlim(-0.7, len(labels) - 0.3)
    # Ajustar ylim para que las mechas no queden cortadas
    all_n = [e["n_part"] for evs in groups.values() for e in evs]
    if all_n:
        axs[0, 1].set_ylim(0, max(all_n) * 1.05)

    axs[1, 0].set_xlabel(r"$p_T$ [GeV/c]")
    axs[1, 0].set_ylabel("Frecuencia (norm. a N eventos)")
    axs[1, 0].set_title(r"Distribucion de $p_T$")
    axs[1, 0].legend(title="Centralidad", fontsize=7, ncol=2)

    axs[1, 1].set_xlabel(r"$\langle p_T \rangle$ por evento [GeV/c]")
    axs[1, 1].set_ylabel("Frecuencia (norm. a N eventos)")
    axs[1, 1].set_title(r"$\langle p_T \rangle$ medio por evento")
    axs[1, 1].legend(title="Centralidad", fontsize=7, ncol=2)

    fig.tight_layout()
    suffix = "charged" if charged else "neutral"
    path = outdir / f"fluc_{tag}_{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Guardado: {path}")


def plot_mean_pt_vs_centrality(groups, tag, outdir, charged):
    labels = [label for _, _, label in B_CUTS]
    mean_pts, x = [], []
    for idx, label in enumerate(labels):
        evs = groups[label]
        if not evs:
            continue
        vals = [e["pt_mean"] for e in evs if np.isfinite(e["pt_mean"])]
        if vals:
            mean_pts.append(np.mean(vals))
            x.append(idx)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, mean_pts, marker="o", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[i] for i in x], rotation=30, ha="right")
    ax.set_ylabel(r"$\langle p_T \rangle$ medio [GeV/c]")
    ax.set_title(f"pT medio vs centralidad — {tag}")
    fig.tight_layout()
    suffix = "charged" if charged else "neutral"
    path = outdir / f"mean_pt_vs_centrality_{tag}_{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Guardado: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fluctuaciones de multiplicidad y pT por centralidad"
    )
    parser.add_argument(
        "run_dirs", nargs="+",
        help="Uno o mas directorios con subdirectorios run_NNNN/ "
             "(ej. output/Bi_11GeV_MB output/Bi_11GeV_MB1 ... o output/Bi_11GeV_MB*)"
    )
    parser.add_argument("--charged", action="store_true",
                        help="Usar particulas cargadas (chg!=0) en lugar de neutras")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Limite TOTAL de eventos combinados entre todas las carpetas")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--tag", default=None,
                        help="Etiqueta para nombrar archivos de salida "
                             "(default: nombre de la primera carpeta, "
                             "+ '_combined' si hay varias)")
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

    events = collect_event_data(run_dirs, charged=args.charged,
                                max_events=args.max_events)
    if not events:
        sys.exit("ERROR: no se encontraron eventos")
    print(f"Total: {len(events)} eventos")

    groups = group_by_centrality(events)
    print_summary(groups)

    print("Generando graficas...")
    plot_panels(groups, tag, outdir, args.charged)
    plot_mean_pt_vs_centrality(groups, tag, outdir, args.charged)
    print("Listo.")


if __name__ == "__main__":
    main()