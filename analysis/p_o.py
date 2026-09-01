#!/usr/bin/env python3
"""
Genera histogramas de pT, eta y phi a partir de archivos .f14/.f15 de UrQMD.

Correcciones aplicadas:
  1. Verificación de dependencia read_f14.py antes de importar.
  2. Histogramas normalizados por número de eventos (dN/dpT por evento).
  3. Rango de pT dinámico (basado en los datos reales, no hardcodeado).
  4. Exportación de arrays procesados a .npy y .csv para reutilización.

Uso:
    python3 analysis/plot_observables.py output/Bi_11GeV_0-20
    python3 analysis/plot_observables.py output/Bi_11GeV_0-20 --charged-only
    python3 analysis/plot_observables.py output/Bi_11GeV_0-20 --pid 101 --charged-only
    python3 analysis/plot_observables.py output/Bi_11GeV_0-20 --select participants
    python3 analysis/plot_observables.py output/Bi_11GeV_0-20 --select spectators --charged-only
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# CORRECCIÓN 1: Verificar que read_f14.py existe
# antes de intentar importarlo, con mensaje claro.
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
    from read_f14 import read_all
except ImportError as e:
    sys.exit(f"ERROR al importar read_f14: {e}")


# ─────────────────────────────────────────────────────────────
# Recolección de arrays con filtros
# ─────────────────────────────────────────────────────────────
def collect_arrays(events, charged_only=False, pid=None, select="all"):
    """
    Concatena pT, eta, phi de todos los eventos aplicando filtros.

    select : "all" | "participants" | "spectators"
        participants -> ncl > 0  (nucleones que colisionaron al menos una vez)
        spectators   -> ncl == 0 (nucleones que no colisionaron)
        all          -> sin filtro por ncl
    """
    pT_list, eta_list, phi_list = [], [], []

    for ev in events:
        mask = np.ones(len(ev["pT"]), dtype=bool)

        if select == "participants":
            mask &= (ev["ncl"] > 0)
        elif select == "spectators":
            mask &= (ev["ncl"] == 0)

        if charged_only:
            mask &= (ev["chg"] != 0)

        if pid is not None:
            mask &= (ev["ityp"] == pid)

        mask &= np.isfinite(ev["eta"])

        pT_list.append(ev["pT"][mask])
        eta_list.append(ev["eta"][mask])
        phi_list.append(ev["phi"][mask])

    return (
        np.concatenate(pT_list),
        np.concatenate(eta_list),
        np.concatenate(phi_list),
    )


# ─────────────────────────────────────────────────────────────
# CORRECCIÓN 4: Exportar arrays a .npy y .csv
# ─────────────────────────────────────────────────────────────
def export_arrays(pT, eta, phi, tag, outdir):
    """
    Guarda los arrays procesados en formato .npy (rápido) y .csv (portable).
    Así no es necesario releer todos los eventos en análisis posteriores.
    """
    npy_path = outdir / f"data_{tag}.npy"
    csv_path = outdir / f"data_{tag}.csv"

    # Formato .npy (eficiente, recargable con np.load)
    np.save(npy_path, {"pT": pT, "eta": eta, "phi": phi})

    # Formato .csv (legible por cualquier herramienta)
    header = "pT_GeV,eta,phi_rad"
    data_matrix = np.column_stack([pT, eta, phi])
    np.savetxt(csv_path, data_matrix, delimiter=",", header=header, comments="")

    print(f"  Datos exportados: {npy_path.name}  |  {csv_path.name}")


# ─────────────────────────────────────────────────────────────
# CORRECCIÓN 2 y 3: Normalización por evento + rango dinámico de pT
# ─────────────────────────────────────────────────────────────
def plot_pt(pT, n_ev, tag, outdir):
    """
    Histograma de pT normalizado como dN/dpT por evento.

    CORRECCIÓN 2: Se divide cada bin entre n_ev (número de eventos)
                  y el ancho del bin (dp_T) → resultado en [GeV/c]^-1 por evento.
    CORRECCIÓN 3: El rango superior del eje X se fija al percentil 99.5
                  de los datos reales, en lugar del valor hardcodeado de 3 GeV/c.
    """
    if len(pT) == 0:
        print("  AVISO: no hay partículas para el histograma de pT.")
        return

    pt_max = np.percentile(pT, 99.5)          # rango dinámico
    pt_max = max(pt_max, 0.5)                  # mínimo razonable
    bins = np.linspace(0, pt_max, 61)
    bin_width = bins[1] - bins[0]

    counts, _ = np.histogram(pT, bins=bins)
    dNdpT = counts / (n_ev * bin_width)        # normalización

    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    fig, ax = plt.subplots()
    ax.step(bin_centers, dNdpT, where="mid", linewidth=1.5, color="steelblue")
    ax.fill_between(bin_centers, dNdpT, step="mid", alpha=0.15, color="steelblue")
    ax.set_xlabel(r"$p_T$ [GeV/c]")
    ax.set_ylabel(r"$\frac{1}{N_\mathrm{ev}}\frac{dN}{dp_T}$ [GeV/c]$^{-1}$")
    ax.set_title(f"Distribución $p_T$ — {tag}\n({n_ev} eventos)")
    ax.set_yscale("log")
    ax.set_xlim(0, pt_max)
    fig.tight_layout()

    path = outdir / f"pt_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Guardado: {path}")


def plot_eta(eta, n_ev, tag, outdir):
    """
    Histograma de eta normalizado como (1/N_ev) dN/deta.
    """
    if len(eta) == 0:
        print("  AVISO: no hay partículas para el histograma de eta.")
        return

    bins = np.linspace(-8, 8, 81)
    bin_width = bins[1] - bins[0]

    counts, _ = np.histogram(eta, bins=bins)
    dNdeta = counts / (n_ev * bin_width)

    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    fig, ax = plt.subplots()
    ax.step(bin_centers, dNdeta, where="mid", linewidth=1.5, color="firebrick")
    ax.fill_between(bin_centers, dNdeta, step="mid", alpha=0.15, color="firebrick")
    ax.set_xlabel(r"$\eta$ (pseudorapidez)")
    ax.set_ylabel(r"$\frac{1}{N_\mathrm{ev}}\frac{dN}{d\eta}$")
    ax.set_title(f"Distribución $\\eta$ — {tag}\n({n_ev} eventos)")
    fig.tight_layout()

    path = outdir / f"eta_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Guardado: {path}")


def plot_phi(phi, n_ev, tag, outdir):
    """
    Histograma de phi normalizado como (1/N_ev) dN/dphi.
    Una distribución plana indica isotropía azimutal (esperado en colisiones centrales).
    """
    if len(phi) == 0:
        print("  AVISO: no hay partículas para el histograma de phi.")
        return

    bins = np.linspace(-np.pi, np.pi, 73)
    bin_width = bins[1] - bins[0]

    counts, _ = np.histogram(phi, bins=bins)
    dNdphi = counts / (n_ev * bin_width)

    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    fig, ax = plt.subplots()
    ax.step(bin_centers, dNdphi, where="mid", linewidth=1.5, color="seagreen")
    ax.fill_between(bin_centers, dNdphi, step="mid", alpha=0.15, color="seagreen")
    ax.set_xlabel(r"$\phi$ [rad]")
    ax.set_ylabel(r"$\frac{1}{N_\mathrm{ev}}\frac{dN}{d\phi}$")
    ax.set_title(f"Distribución $\\phi$ — {tag}\n({n_ev} eventos)")
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    fig.tight_layout()

    path = outdir / f"phi_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Guardado: {path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Plots pT, eta, phi de UrQMD (normalizado por evento)"
    )
    parser.add_argument("run_dir", help="Directorio con subdirectorios run_NNNN/")
    parser.add_argument("--charged-only", action="store_true",
                        help="Solo partículas cargadas")
    parser.add_argument("--pid", type=int, default=None,
                        help="Filtrar por ityp (ID interno de UrQMD, ej. 101 = pión)")
    parser.add_argument("--select", choices=["all", "participants", "spectators"],
                        default="all",
                        help="all: todas las partículas (default); "
                             "participants: ncl>0; spectators: ncl==0")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Límite de eventos a leer")
    parser.add_argument("--export", action="store_true",
                        help="Exportar arrays procesados a .npy y .csv")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        sys.exit(f"ERROR: no existe el directorio '{run_dir}'")

    # Construir etiqueta descriptiva para nombres de archivo
    tag = run_dir.name
    if args.select != "all":
        tag += f"_{args.select}"
    if args.charged_only:
        tag += "_cargadas"
    if args.pid is not None:
        tag += f"_pid{args.pid}"

    outdir = run_dir / "plots"
    outdir.mkdir(exist_ok=True)

    print(f"\nLeyendo eventos de: {run_dir}")
    events = read_all(run_dir, max_events=args.max_events)

    if not events:
        sys.exit("ERROR: no se encontraron eventos en el directorio indicado.")

    pT, eta, phi = collect_arrays(
        events,
        charged_only=args.charged_only,
        pid=args.pid,
        select=args.select,
    )

    n_ev   = len(events)
    n_part = len(pT)
    print(f"Eventos leídos      : {n_ev}")
    print(f"Partículas (filtro) : {n_part}")

    if n_part == 0:
        sys.exit("ERROR: ninguna partícula sobrevivió los filtros aplicados.")

    # CORRECCIÓN 4: Exportación opcional
    if args.export:
        print("\nExportando datos...")
        export_arrays(pT, eta, phi, tag, outdir)

    print("\nGenerando gráficas...")
    plot_pt(pT,  n_ev, tag, outdir)
    plot_eta(eta, n_ev, tag, outdir)
    plot_phi(phi, n_ev, tag, outdir)

    print(f"\nListo. Gráficas guardadas en: {outdir}\n")


if __name__ == "__main__":
    main()