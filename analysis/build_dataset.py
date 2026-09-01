#!/usr/bin/env python3
"""
Construye un dataset HDF5 para ML a partir de archivos .f14 de UrQMD.

Por cada evento almacena un vector de features (histogramas de pT, eta, phi)
y el parámetro de impacto b. Los eventos se procesan uno a la vez (generador)
y se escriben al disco en batches, por lo que el uso de RAM es constante
independientemente del número de eventos.

Uso:
    python3 analysis/build_dataset.py output/Bi_11GeV_0-20 features_0-20.h5
    python3 analysis/build_dataset.py output/Bi_11GeV_0-20 features_0-20.h5 --select participants
    python3 analysis/build_dataset.py output/Bi_11GeV_0-20 features_0-20.h5 --select spectators --charged-only
    python3 analysis/build_dataset.py output/Bi_11GeV_0-20 features_0-20.h5 --pid 101

Estructura del archivo HDF5 resultante:
    /pt_hist   shape (N_events, n_pt_bins)   histograma de pT por evento
    /eta_hist  shape (N_events, n_eta_bins)  histograma de eta por evento
    /phi_hist  shape (N_events, n_phi_bins)  histograma de phi por evento
    /b         shape (N_events,)             parametro de impacto [fm]
    /attrs     select, charged_only, pid, bin edges guardados como metadata

Leer el dataset para ML:
    import h5py, numpy as np
    with h5py.File("features_0-20.h5", "r") as h5:
        X = np.hstack([h5["pt_hist"][:], h5["eta_hist"][:], h5["phi_hist"][:]])
        b = h5["b"][:]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from read_f14 import iter_all

try:
    import h5py
except ImportError:
    sys.exit("ERROR: instala h5py con:  pip install h5py")


# ── Bins por defecto ──────────────────────────────────────────────────────────
PT_BINS  = np.linspace(0.0,  3.0,  61)   # 60 bins, 0-3 GeV/c
ETA_BINS = np.linspace(-8.0, 8.0,  81)   # 80 bins
PHI_BINS = np.linspace(-np.pi, np.pi, 73) # 72 bins


def _apply_mask(ev, select, charged_only, pid):
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
    return mask


def build_dataset(run_dir, outfile, select="all", charged_only=False,
                  pid=None, batch=500, max_events=None, verbose=True):
    n_pt  = len(PT_BINS)  - 1
    n_eta = len(ETA_BINS) - 1
    n_phi = len(PHI_BINS) - 1

    buf_pt, buf_eta, buf_phi, buf_b = [], [], [], []
    n_total = 0

    with h5py.File(outfile, "w") as h5:
        ds_pt  = h5.create_dataset("pt_hist",  shape=(0, n_pt),
                                   maxshape=(None, n_pt),  dtype="f4",
                                   chunks=(batch, n_pt),   compression="gzip")
        ds_eta = h5.create_dataset("eta_hist", shape=(0, n_eta),
                                   maxshape=(None, n_eta), dtype="f4",
                                   chunks=(batch, n_eta),  compression="gzip")
        ds_phi = h5.create_dataset("phi_hist", shape=(0, n_phi),
                                   maxshape=(None, n_phi), dtype="f4",
                                   chunks=(batch, n_phi),  compression="gzip")
        ds_b   = h5.create_dataset("b",        shape=(0,),
                                   maxshape=(None,),       dtype="f4",
                                   chunks=(batch,),        compression="gzip")

        # guardar bin edges como metadata
        h5.create_dataset("pt_edges",  data=PT_BINS.astype("f4"))
        h5.create_dataset("eta_edges", data=ETA_BINS.astype("f4"))
        h5.create_dataset("phi_edges", data=PHI_BINS.astype("f4"))
        h5.attrs["select"]       = select
        h5.attrs["charged_only"] = charged_only
        h5.attrs["pid"]          = pid if pid is not None else -1
        h5.attrs["run_dir"]      = str(run_dir)

        def _flush():
            n = len(buf_pt)
            for ds, buf in [(ds_pt, buf_pt), (ds_eta, buf_eta), (ds_phi, buf_phi)]:
                ds.resize(ds.shape[0] + n, axis=0)
                ds[-n:] = np.array(buf, dtype="f4")
            ds_b.resize(ds_b.shape[0] + n, axis=0)
            ds_b[-n:] = np.array(buf_b, dtype="f4")
            buf_pt.clear(); buf_eta.clear(); buf_phi.clear(); buf_b.clear()

        for ev in iter_all(run_dir, max_events=max_events):
            mask = _apply_mask(ev, select, charged_only, pid)
            buf_pt.append( np.histogram(ev["pT"][mask],  bins=PT_BINS)[0])
            buf_eta.append(np.histogram(ev["eta"][mask], bins=ETA_BINS)[0])
            buf_phi.append(np.histogram(ev["phi"][mask], bins=PHI_BINS)[0])
            buf_b.append(float(ev["b"]))
            n_total += 1

            if len(buf_pt) >= batch:
                _flush()
                if verbose:
                    print(f"\r  {n_total} eventos procesados...", end="", flush=True)

        if buf_pt:
            _flush()

    if verbose:
        print(f"\r  {n_total} eventos procesados.        ")
    return n_total


def main():
    parser = argparse.ArgumentParser(
        description="Construye dataset HDF5 de features para ML desde archivos f14 de UrQMD")
    parser.add_argument("run_dir",  help="Directorio con subdirectorios run_NNNN/")
    parser.add_argument("outfile",  help="Archivo HDF5 de salida (ej. features_0-20.h5)")
    parser.add_argument("--select", choices=["all", "participants", "spectators"],
                        default="all",
                        help="all: todas; participants: ncl>0; spectators: ncl==0")
    parser.add_argument("--charged-only", action="store_true",
                        help="Solo particulas cargadas")
    parser.add_argument("--pid", type=int, default=None,
                        help="Filtrar por ityp (ID interno de UrQMD)")
    parser.add_argument("--batch", type=int, default=500,
                        help="Eventos por flush a disco (default 500)")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Limite de eventos a procesar")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        sys.exit(f"ERROR: no existe {run_dir}")

    print(f"Fuente  : {run_dir}")
    print(f"Salida  : {args.outfile}")
    print(f"Select  : {args.select}  |  charged_only={args.charged_only}  |  pid={args.pid}")
    print(f"Bins    : pT={len(PT_BINS)-1}  eta={len(ETA_BINS)-1}  phi={len(PHI_BINS)-1}")
    print(f"Batch   : {args.batch} eventos por flush")
    print()

    n = build_dataset(run_dir, args.outfile,
                      select=args.select,
                      charged_only=args.charged_only,
                      pid=args.pid,
                      batch=args.batch,
                      max_events=args.max_events)

    import h5py
    with h5py.File(args.outfile, "r") as h5:
        shape_pt = h5["pt_hist"].shape
    print(f"Guardado: {args.outfile}")
    print(f"  Eventos : {n}")
    print(f"  Shape X : {shape_pt}  (pt_hist) + eta + phi -> {shape_pt[1] + len(ETA_BINS)-1 + len(PHI_BINS)-1} features/evento")
    print()
    print("Para ML:")
    print("  import h5py, numpy as np")
    print(f"  with h5py.File('{args.outfile}', 'r') as h5:")
    print("      X = np.hstack([h5['pt_hist'][:], h5['eta_hist'][:], h5['phi_hist'][:]])")
    print("      b = h5['b'][:]")


if __name__ == "__main__":
    main()
