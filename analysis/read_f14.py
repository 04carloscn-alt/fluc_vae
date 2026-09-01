"""
Lector de archivos .f14 / .f15 de UrQMD.

Formato de línea de partícula (23 columnas):
  t  X  Y  Z  E  Px  Py  Pz  m  ityp  2i3  chg  lcl#  ncl  or
  frezeT  frezeX  frezeY  frezeZ  frezeE  frezePx  frezePy  frezePz  counter

Observables calculados:
  pT  = sqrt(Px^2 + Py^2)
  P   = sqrt(Px^2 + Py^2 + Pz^2)
  eta = 0.5 * ln((P + Pz) / (P - Pz))
  phi = atan2(Py, Px)
"""

import numpy as np
from pathlib import Path


# UrQMD 3.4 f14 format: 19 columnas
# r0 rx ry rz p0 px py pz m ityp 2i3 chg lcl# ncl or  frezetime frezeX step counter
_COLS = [
    "t", "X", "Y", "Z", "E", "Px", "Py", "Pz", "m",
    "ityp", "iso", "chg", "lcl", "ncl", "hist",
    "frezeT", "frezeX", "step", "counter",
]
_NCOLS_MIN = 15   # mínimo para garantizar E, Px, Py, Pz, chg, ncl


def _derived(particles):
    """Añade pT, P, eta, phi a un dict de arrays numpy."""
    px, py, pz = particles["Px"], particles["Py"], particles["Pz"]
    pT = np.sqrt(px**2 + py**2)
    P  = np.sqrt(px**2 + py**2 + pz**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(np.abs(P - pz) > 1e-10,
                       0.5 * np.log((P + pz) / (P - pz)),
                       np.nan)
    phi = np.arctan2(py, px)
    particles.update({"pT": pT, "P": P, "eta": eta, "phi": phi})
    return particles


def iter_events(filepath, max_events=None):
    """
    Generator: yields one event dict at a time from a .f14 file.
    Only one event lives in RAM at a time — use this for large datasets.

    Each yielded dict contains numpy arrays for all 19 columns plus
    derived quantities pT, P, eta, phi, and scalar 'b'.

    Parameters
    ----------
    filepath   : str | Path
    max_events : int | None — stop after this many events (None = all)

    Yields
    ------
    dict
    """
    filepath = Path(filepath)
    current_particles = {c: [] for c in _COLS}
    b_val = np.nan
    in_event = False
    n_yielded = 0

    with open(filepath) as f:
        for line in f:
            c = line[0] if line else ""

            if c == "U":   # new event header
                if in_event and current_particles["E"]:
                    p = {k: np.array(v, dtype=np.float64)
                         for k, v in current_particles.items()}
                    p["b"] = b_val
                    yield _derived(p)
                    n_yielded += 1
                    if max_events and n_yielded >= max_events:
                        return
                current_particles = {c: [] for c in _COLS}
                b_val = np.nan
                in_event = True

            elif c == "i":
                try:
                    b_val = float(line[37:42])
                except (ValueError, IndexError):
                    pass

            elif c in ("p", "t", "e", "o"):
                continue

            elif in_event and line.strip() and c not in ("#", "\n"):
                parts = line.split()
                if len(parts) >= _NCOLS_MIN:
                    try:
                        for col, val in zip(_COLS, parts[:len(_COLS)]):
                            current_particles[col].append(float(val))
                    except ValueError:
                        pass

    # last event
    if in_event and current_particles["E"]:
        if not max_events or n_yielded < max_events:
            p = {k: np.array(v, dtype=np.float64)
                 for k, v in current_particles.items()}
            p["b"] = b_val
            yield _derived(p)


def read_events(filepath, max_events=None):
    """
    Load all events from a .f14 file into a list.
    Convenience wrapper around iter_events — loads everything into RAM.
    For large files use iter_events or iter_all directly.
    """
    return list(iter_events(filepath, max_events=max_events))


def iter_all(output_dir, pattern="*/urqmd.f14", max_events=None):
    """
    Generator: yields events one at a time across all f14 files in output_dir.
    Memory usage is constant regardless of the number of files or events.
    """
    files = sorted(Path(output_dir).glob(pattern))
    if not files:
        files = sorted(Path(output_dir).glob("*/urqmd.f14"))
    if not files:
        raise FileNotFoundError(f"No f14 files found in {output_dir}")

    n_yielded = 0
    for fp in files:
        for ev in iter_events(fp):
            yield ev
            n_yielded += 1
            if max_events and n_yielded >= max_events:
                return


def read_all(output_dir, pattern="*/urqmd.f14", max_events=None):
    """
    Load all events from output_dir into a list.
    Uses iter_all internally. For large datasets use iter_all directly.
    """
    files = sorted(Path(output_dir).glob(pattern))
    if not files:
        files = sorted(Path(output_dir).glob("*/urqmd.f14"))
    if not files:
        raise FileNotFoundError(f"No f14 files found in {output_dir}")

    all_events = list(iter_all(output_dir, pattern=pattern, max_events=max_events))
    print(f"Leídos {len(all_events)} eventos de {len(files)} archivos")
    return all_events
