#!/usr/bin/env python3
"""
Genera histogramas de pT, eta y phi a partir de archivos .f14/.f15 de UrQMD.

A diferencia de la versión anterior, en lugar de una gráfica por especie
individual, esta versión JUNTA varias especies en una sola gráfica por
observable:

    - "Todas"     : todas las partículas (participantes vs espectadores)
    - "Cargadas"  : π⁺, π⁻, K⁺, K⁻, p  -> cada especie con su curva de
                    participantes (línea sólida) y espectadores (línea
                    punteada), todas en el mismo plot.
    - "No cargadas": π⁰, K⁰, K̄⁰, n     -> mismo esquema.

Esto da 3 gráficas por observable (pT, eta, phi) = 9 gráficas en total,
en vez de una por cada una de las 11 especies.

Si además quieres las gráficas individuales por especie (comportamiento
del script original), usa --by-species.

Soporta múltiples carpetas de datos (por ejemplo, cuando los 100,000
eventos están repartidos en 5 sub-lotes de simulación), leyendo los
eventos en un solo recorrido (streaming) con `iter_all` de read_f14.py,
igual que hace pt_moments.py.

Uso:
    python3 analysis/plot_observables.py output/Bi_11GeV_MB/
    python3 analysis/plot_observables.py output/Bi_11GeV_MB1 output/Bi_11GeV_MB2 \\
        output/Bi_11GeV_MB3 output/Bi_11GeV_MB4 output/Bi_11GeV_MB5
    python3 analysis/plot_observables.py output/Bi_11GeV_MB* --outdir output/plots_combinados
    python3 analysis/plot_observables.py output/Bi_11GeV_MB/ --max-events 500
    python3 analysis/plot_observables.py output/Bi_11GeV_MB/ --export
    python3 analysis/plot_observables.py output/Bi_11GeV_MB/ --by-species
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Verificar dependencia read_f14.py
# ─────────────────────────────────────────────
_analysis_dir = Path(__file__).parent
_read_f14_path = _analysis_dir / "read_f14.py"

if not _read_f14_path.exists():
    sys.exit(
        f"ERROR: No se encontró 'read_f14.py' en {_analysis_dir}\n"
        f"Asegúrate de que el archivo exista en el mismo directorio que este script."
    )

sys.path.insert(0, str(_analysis_dir))
try:
    from read_f14 import iter_all
except ImportError as e:
    sys.exit(
        f"ERROR al importar 'iter_all' de read_f14: {e}\n"
        f"Esta versión del script necesita que read_f14.py exponga una función\n"
        f"generadora `iter_all(run_dir, max_events=None)` que vaya entregando\n"
        f"eventos uno por uno (igual que la usa analysis/pt_moments.py).\n"
        f"Si tu read_f14.py solo tiene `read_all`, puedes envolverla así:\n\n"
        f"    def iter_all(run_dir, max_events=None):\n"
        f"        for ev in read_all(run_dir, max_events=max_events):\n"
        f"            yield ev\n"
    )


# ─────────────────────────────────────────────────────────────
# Catálogo de especies
# Cada entrada: (nombre_legible, tag_archivo, pid, charged_only, charge_sign)
# ─────────────────────────────────────────────────────────────
SPECIES = [
    # ── Todas / Cargadas / No cargadas (agregados, sin desglosar especie) ──
    ("Todas",                "all",           None, False,        None),
    ("Cargadas",             "charged",       None, True,         None),
    ("No cargadas",          "neutral_all",   None, False,        "neutral"),

    # ── Piones ──────────────────────────────────────────────────────────────
    ("Pión +  (π⁺)",        "pi_plus",       101,  False,        "positive"),
    ("Pión -  (π⁻)",        "pi_minus",      101,  False,        "negative"),
    ("Pión 0  (π⁰)",        "pi_zero",       101,  False,        "neutral"),

    # ── Kaones ──────────────────────────────────────────────────────────────
    ("Kaón +  (K⁺)",        "k_plus",        106,  False,        "positive"),
    ("Kaón -  (K⁻)",        "k_minus",       -106, False,        "negative"),
    ("Kaón 0  (K⁰)",        "k_zero",        106,  False,        "neutral"),
    ("Kaón 0b (K̄⁰)",       "k_zero_bar",    -106, False,        "neutral"),

    # ── Nucleones ────────────────────────────────────────────────────────────
    ("Protón  (p)",          "proton",        1,    False,        "positive"),
    ("Neutrón (n)",          "neutron",       1,    False,        "neutral"),
]

# Tags de especies "cargadas físicas" y "neutras físicas" a agrupar.
CHARGED_SPECIES_TAGS = ["pi_plus", "pi_minus", "k_plus", "k_minus", "proton"]
NEUTRAL_SPECIES_TAGS = ["pi_zero", "k_zero", "k_zero_bar", "neutron"]

_LABEL_BY_TAG = {tag: label for label, tag, *_ in SPECIES}

# Categorías de gráficas agrupadas: (título, tag_archivo, [(label, file_tag), ...])
CATEGORIES = [
    ("Todas las partículas", "all_particles",
        [(_LABEL_BY_TAG["all"], "all")]),
    ("Cargadas: π⁺, π⁻, K⁺, K⁻, p", "charged_species",
        [(_LABEL_BY_TAG[t], t) for t in CHARGED_SPECIES_TAGS]),
    ("No cargadas: π⁰, K⁰, K̄⁰, n", "neutral_species",
        [(_LABEL_BY_TAG[t], t) for t in NEUTRAL_SPECIES_TAGS]),
]

# ─────────────────────────────────────────────────────────────
# Configuración de cada observable
# ─────────────────────────────────────────────────────────────
ETA_BINS = np.linspace(-8, 8, 81)
PHI_BINS = np.linspace(-np.pi, np.pi, 73)

OBS_CONFIG = {
    "pT": dict(
        key="pT",
        xlabel=r"$p_T$ [GeV/c]",
        ylabel=r"$\frac{1}{N_\mathrm{ev}}\frac{dN}{dp_T}$ [GeV/c]$^{-1}$",
        title_prefix="Distribución $p_T$",
        log=True,
    ),
    "eta": dict(
        key="eta",
        xlabel=r"$\eta$ (pseudorapidez)",
        ylabel=r"$\frac{1}{N_\mathrm{ev}}\frac{dN}{d\eta}$",
        title_prefix="Distribución $\\eta$",
        log=False,
    ),
    "phi": dict(
        key="phi",
        xlabel=r"$\phi$ [rad]",
        ylabel=r"$\frac{1}{N_\mathrm{ev}}\frac{dN}{d\phi}$",
        title_prefix="Distribución $\\phi$",
        log=False,
    ),
}


# ─────────────────────────────────────────────────────────────
# Lectura en un solo recorrido (streaming) sobre 1 o varias carpetas
# ─────────────────────────────────────────────────────────────
def collect_all_species(run_dirs, max_events=None):
    """
    Recorre los eventos de todas las carpetas indicadas UNA SOLA VEZ.
    Para cada evento aplica los filtros de TODAS las especies de SPECIES
    y de ambos grupos (participantes ncl>0 / espectadores ncl==0),
    acumulando pT, eta, phi por (file_tag, select).

    Esto evita releer los datos 22 veces (11 especies x 2 grupos) como
    ocurría filtrando después de cargar todo en memoria, algo importante
    cuando se combinan varias carpetas con decenas de miles de eventos.

    Retorna:
        data[(file_tag, select)] = {"pT": array, "eta": array, "phi": array}
        n_ev: número total de eventos leídos (para normalizar dN/n_ev)
    """
    data = {}
    for _label, file_tag, *_ in SPECIES:
        for select in ("participants", "spectators"):
            data[(file_tag, select)] = {"pT": [], "eta": [], "phi": []}

    n_ev = 0

    for run_dir in run_dirs:
        remaining = None if max_events is None else max_events - n_ev
        if remaining is not None and remaining <= 0:
            break

        print(f"  Leyendo: {run_dir} ...", flush=True)
        for ev in iter_all(run_dir, max_events=remaining):
            n_ev += 1
            eta_finite = np.isfinite(ev["eta"])

            for select in ("participants", "spectators"):
                if select == "participants":
                    sel_mask = eta_finite & (ev["ncl"] > 0)
                else:
                    sel_mask = eta_finite & (ev["ncl"] == 0)

                for _label, file_tag, pid, charged_only, charge_sign in SPECIES:
                    mask = sel_mask
                    if charged_only:
                        mask = mask & (ev["chg"] != 0)
                    if charge_sign == "positive":
                        mask = mask & (ev["chg"] > 0)
                    elif charge_sign == "negative":
                        mask = mask & (ev["chg"] < 0)
                    elif charge_sign == "neutral":
                        mask = mask & (ev["chg"] == 0)
                    if pid is not None:
                        mask = mask & (ev["ityp"] == pid)

                    if mask.any():
                        d = data[(file_tag, select)]
                        d["pT"].append(ev["pT"][mask])
                        d["eta"].append(ev["eta"][mask])
                        d["phi"].append(ev["phi"][mask])

            if n_ev % 2000 == 0:
                print(f"    {n_ev} eventos procesados...", flush=True)

    for d in data.values():
        for obs in ("pT", "eta", "phi"):
            d[obs] = np.concatenate(d[obs]) if d[obs] else np.array([])

    return data, n_ev


# ─────────────────────────────────────────────────────────────
# Helpers para histogramas normalizados
# ─────────────────────────────────────────────────────────────
def _make_dN(data_arr, bins, n_ev):
    """Densidad normalizada: counts / (N_ev * bin_width)."""
    counts, _ = np.histogram(data_arr, bins=bins)
    return counts / (n_ev * (bins[1] - bins[0]))


def _bin_centers(bins):
    return 0.5 * (bins[:-1] + bins[1:])


def _dynamic_pt_bins(arrays):
    """Rango dinámico de pT: percentil 99.5 del conjunto con más datos."""
    candidates = [a for a in arrays if len(a) > 0]
    if not candidates:
        return None
    pt_max = max(max(np.percentile(a, 99.5) for a in candidates), 0.5)
    return np.linspace(0, pt_max, 61)


# ─────────────────────────────────────────────────────────────
# Plot genérico: N especies, 2 curvas (participantes/espectadores) c/u
# ─────────────────────────────────────────────────────────────
def plot_group(obs_key, entries, group_label, group_tag, data, n_ev, run_tag, outdir):
    """
    entries: lista [(label_legible, file_tag), ...] de las especies a
    superponer en una misma gráfica. Cada especie aporta 2 curvas:
    participantes (línea sólida) y espectadores (línea punteada), con
    el mismo color para distinguir que son la misma especie.
    """
    cfg = OBS_CONFIG[obs_key]

    if obs_key == "pT":
        all_arrays = [
            data[(file_tag, select)]["pT"]
            for _label, file_tag in entries
            for select in ("participants", "spectators")
        ]
        bins = _dynamic_pt_bins(all_arrays)
        if bins is None:
            print(f"  AVISO [{group_label}]: sin datos para pT, se omite.")
            return
    elif obs_key == "eta":
        bins = ETA_BINS
    else:
        bins = PHI_BINS
    bc = _bin_centers(bins)

    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(8.5, 6))
    any_data = False

    for i, (label, file_tag) in enumerate(entries):
        color = colors[i % len(colors)]
        for select, ls, lw in (("participants", "-", 1.7), ("spectators", "--", 1.3)):
            arr = data[(file_tag, select)][cfg["key"]]
            if len(arr) == 0:
                continue
            any_data = True
            dN = _make_dN(arr, bins, n_ev)
            tag = "Part." if select == "participants" else "Esp."
            ax.step(bc, dN, where="mid", color=color, ls=ls, lw=lw,
                    label=f"{label} ({tag})")

    if not any_data:
        print(f"  AVISO [{group_label}]: sin datos, se omite.")
        plt.close(fig)
        return

    ax.set_xlabel(cfg["xlabel"])
    ax.set_ylabel(cfg["ylabel"])
    ax.set_title(f"{cfg['title_prefix']} — {group_label}\n{run_tag}  ({n_ev} eventos)")
    if cfg["log"]:
        ax.set_yscale("log")
    if obs_key == "pT":
        ax.set_xlim(0, bins[-1])
    if obs_key == "phi":
        ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax.legend(framealpha=0.85, fontsize=8, ncol=2 if len(entries) > 1 else 1)
    fig.tight_layout()

    path = outdir / f"{obs_key}_{group_tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Guardado: {path.name}")


# ─────────────────────────────────────────────────────────────
# Exportación opcional
# ─────────────────────────────────────────────────────────────
def export_arrays(pT, eta, phi, tag, outdir):
    npy_path = outdir / f"data_{tag}.npy"
    csv_path = outdir / f"data_{tag}.csv"
    np.save(npy_path, {"pT": pT, "eta": eta, "phi": phi})
    np.savetxt(csv_path, np.column_stack([pT, eta, phi]),
               delimiter=",", header="pT_GeV,eta,phi_rad", comments="")
    print(f"    Exportado: {npy_path.name}  |  {csv_path.name}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Genera plots pT/eta/phi combinando varias especies por gráfica "
            "(Todas / Cargadas: π+,π-,K+,K-,p / No cargadas: π0,K0,K0b,n), "
            "leyendo una o varias carpetas de simulación en un solo recorrido."
        )
    )
    parser.add_argument(
        "run_dirs", nargs="+",
        help=(
            "Uno o varios directorios con subdirectorios run_NNNN/ "
            "(ej. output/Bi_11GeV_MB1 ... output/Bi_11GeV_MB5, o un glob "
            "como output/Bi_11GeV_MB*)"
        ),
    )
    parser.add_argument("--max-events", type=int, default=None,
                        help="Límite total de eventos a leer, sumado entre todas las carpetas")
    parser.add_argument("--export", action="store_true",
                        help="Exportar arrays de participantes a .npy y .csv por especie")
    parser.add_argument("--outdir", default=None,
                        help="Directorio de salida (default: <primera_carpeta>/plots)")
    parser.add_argument("--tag", default=None,
                        help="Etiqueta para títulos/nombres (default: nombre de la 1a carpeta, o '..._combined')")
    parser.add_argument("--by-species", action="store_true",
                        help="Además de las gráficas agrupadas, generar 1 gráfica por cada especie individual")
    args = parser.parse_args()

    run_dirs = [Path(d) for d in args.run_dirs]
    for d in run_dirs:
        if not d.exists():
            sys.exit(f"ERROR: no existe el directorio '{d}'")

    if args.tag:
        run_tag = args.tag
    elif len(run_dirs) == 1:
        run_tag = run_dirs[0].name
    else:
        run_tag = f"{run_dirs[0].name}_combined"

    outdir = Path(args.outdir) if args.outdir else run_dirs[0] / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    if len(run_dirs) == 1:
        print(f"\nLeyendo eventos de: {run_dirs[0]}")
    else:
        print(f"\nLeyendo eventos de {len(run_dirs)} carpetas:")
        for d in run_dirs:
            print(f"  - {d}")

    data, n_ev = collect_all_species(run_dirs, max_events=args.max_events)
    if n_ev == 0:
        sys.exit("ERROR: no se encontraron eventos.")
    print(f"\nTotal de eventos leídos: {n_ev}\n")

    # ── Gráficas agrupadas (Todas / Cargadas por especie / No cargadas por especie) ──
    print("── Generando gráficas agrupadas ──")
    for obs_key in ("pT", "eta", "phi"):
        for group_label, group_tag, entries in CATEGORIES:
            plot_group(obs_key, entries, group_label, group_tag, data, n_ev, run_tag, outdir)
    print()

    # ── Gráficas individuales por especie (opcional) ──────────────────────
    if args.by_species:
        print("── Generando gráficas por especie individual ──")
        for label, file_tag, *_ in SPECIES:
            for obs_key in ("pT", "eta", "phi"):
                plot_group(obs_key, [(label, file_tag)], label, file_tag,
                          data, n_ev, run_tag, outdir)
        print()

    # ── Exportación opcional (participantes, por especie) ─────────────────
    if args.export:
        print("── Exportando arrays (participantes) ──")
        for label, file_tag, *_ in SPECIES:
            arr = data[(file_tag, "participants")]
            if len(arr["pT"]) > 0:
                export_arrays(arr["pT"], arr["eta"], arr["phi"], file_tag, outdir)
        print()

    print(f"Listo. Gráficas guardadas en: {outdir}\n")


if __name__ == "__main__":
    main()