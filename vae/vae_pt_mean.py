#!/usr/bin/env python3
"""
vae_pt_mean.py — VAE sobre histogramas bootstrap de <pT>, por evento,
                  separado por clase de centralidad.

Objeto de estudio (protocolo, Ecs. 1.1, 1.2, 2.1-2.4):
    <pT>_i = (1/N_i) * sum_j pT_j        (un escalar por evento)
    P(<pT>)                              (distribución evento a evento)
    sigma^2_pT = <<pT>^2> - <<pT>>^2      (varianza del ENSEMBLE)

Cómo se construye el input del VAE (vector de dimensión d por evento,
tal como pide el protocolo: "Histograma normalizado de <pT>"):

    <pT>_i es un solo número, así que un "histograma por evento" solo
    puede construirse a partir de algo DERIVADO de ese evento. Aquí se
    usa bootstrap: se remuestrean (con reemplazo) las propias N_i
    partículas del evento K veces, se calcula <pT> en cada remuestreo,
    y el histograma normalizado (d bins) de esas K estimaciones es el
    vector de entrada x_i para ese evento.

    Esto tiene una lectura física directa: el ANCHO de ese histograma
    es la incertidumbre estadística de <pT>_i dado un N_i finito
    (~ sigma_partícula / sqrt(N_i)). Eventos con N_i grande dan
    histogramas angostos (estimación precisa); eventos con N_i chico
    dan histogramas anchos (estimación ruidosa). El VAE ve tanto la
    posición como la FORMA de esa incertidumbre.

n_min (mínima multiplicidad aceptada) ya NO es arbitrario: se estima
con un estudio de bootstrap independiente sobre un pool de partículas,
midiendo el error estándar (SE) de <pT> como función de N, y eligiendo
el N a partir del cual ese ruido estadístico cae por debajo de un
umbral relativo respecto a la dispersión intrínseca partícula-a-partícula.
Ver estimate_n_min().

Todo el análisis (carga, n_min, entrenamiento del VAE, clustering,
gráficas) se realiza POR SEPARADO para cada una de las 10 clases de
centralidad — no se mezclan ni se resta ninguna media entre clases,
porque cada clase entrena y evalúa su propio modelo.

Uso:
    python3 analysis/vae_pt_mean.py output/Bi_11GeV_MB* --charged \\
        --n-boot 200 --d-bins 25 --z-dim 2 --beta 1.0 --epochs 150

    # Forzar n_min manual en vez del recomendado por bootstrap:
    python3 analysis/vae_pt_mean.py output/Bi_11GeV_MB* --charged \\
        --n-min 12 --z-dim 2 --beta 1.0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

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
CENT_COLORS = [
    "#d62728","#ff7f0e","#f7c617","#2ca02c",
    "#17becf","#1f77b4","#9467bd","#8c564b","#bcbd22","#7f7f7f",
]

def assign_centrality(b):
    for b_lo, b_hi, label, idx in B_CUTS:
        if b_lo <= b < b_hi:
            return idx
    return 9


# ═══════════════════════════════════════════════════════════════════════════
#  1. CARGA — bucketizar eventos (arreglos crudos de pT) por clase
# ═══════════════════════════════════════════════════════════════════════════

def load_events_by_class(output_dirs, charged=False, max_events=None,
                          max_per_class=None):
    """
    Lee eventos y los agrupa por clase de centralidad SIN aplicar todavía
    ningún corte de multiplicidad (eso se decide después, con bootstrap).

    Retorna:
        events[ci] = lista de dicts {"pt": array, "b": float, "n": int}
    """
    if isinstance(output_dirs, (str, Path)):
        output_dirs = [output_dirs]

    events = {ci: [] for ci in range(len(CENT_LABELS))}
    n_total = 0

    for output_dir in output_dirs:
        output_dir = Path(output_dir)
        remaining  = None if max_events is None else max_events - n_total
        if remaining is not None and remaining <= 0:
            break

        print(f"  Leyendo: {output_dir} ...", flush=True)
        for ev in iter_all(output_dir, max_events=remaining):
            if charged:
                mask = (ev["ncl"] > 0) & (ev["chg"] != 0)
            else:
                mask = (ev["ncl"] > 0) & (ev["chg"] == 0)

            pt_vals = ev["pT"][mask]
            n_total += 1

            if len(pt_vals) >= 1 and np.all(np.isfinite(pt_vals)):
                ci = assign_centrality(float(ev["b"]))
                if max_per_class is None or len(events[ci]) < max_per_class:
                    events[ci].append({
                        "pt": pt_vals.astype(np.float64),
                        "b":  float(ev["b"]),
                        "n":  len(pt_vals),
                    })

            if n_total % 10000 == 0:
                print(f"    {n_total} eventos procesados...", flush=True)
            if max_events is not None and n_total >= max_events:
                break

    for ci, label in enumerate(CENT_LABELS):
        print(f"  {label}: {len(events[ci])} eventos disponibles")

    return events


# ═══════════════════════════════════════════════════════════════════════════
#  2. BOOTSTRAP — n_min y construcción de histogramas por evento
# ═══════════════════════════════════════════════════════════════════════════

def bootstrap_se_vs_N(pt_pool, n_grid, n_boot=1000, seed=42):
    """
    Remuestrea (con reemplazo) el pool de partículas para cada N en
    n_grid, y mide el error estándar (SE) de <pT> = std entre réplicas
    del promedio bootstrap. Esto aísla el ruido puramente ESTADÍSTICO
    (tipo Poisson / muestreo finito), independiente de cualquier
    fluctuación física real.
    """
    rng = np.random.default_rng(seed)
    se = np.empty(len(n_grid))
    for k, N in enumerate(n_grid):
        idx  = rng.integers(0, len(pt_pool), size=(n_boot, N))
        means = pt_pool[idx].mean(axis=1)
        se[k] = means.std()
    return se


def estimate_n_min(pt_pool, n_grid=None, n_boot=1000, rel_threshold=0.05,
                    seed=42):
    """
    Recomienda n_min: el menor N de la malla para el cual el ruido
    estadístico puro SE(N) cae por debajo de rel_threshold * sigma_pool
    (sigma_pool = dispersión partícula-a-partícula del pool, una
    cantidad fija y siempre disponible). rel_threshold=0.05 exige que
    la incertidumbre de <pT> sea <=5% de la dispersión intrínseca de
    una sola partícula.
    """
    if n_grid is None:
        n_grid = [3, 5, 8, 10, 15, 20, 30, 50, 80, 120, 200, 300]
    sigma_pool = pt_pool.std()
    se = bootstrap_se_vs_N(pt_pool, n_grid, n_boot=n_boot, seed=seed)
    rel = se / sigma_pool
    n_min_rec = None
    for N, r in zip(n_grid, rel):
        if r <= rel_threshold:
            n_min_rec = N
            break
    if n_min_rec is None:
        n_min_rec = n_grid[-1]
        print(f"  [AVISO] SE no baja de {rel_threshold*100:.0f}% ni en "
              f"N={n_grid[-1]}. Se usa el mayor N de la malla.")
    return n_min_rec, n_grid, se, sigma_pool


def plot_bootstrap_nmin(n_grid, se, sigma_pool, n_min_rec, rel_threshold,
                         tag, outdir):
    rel = np.array(se) / sigma_pool
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))

    axs[0].plot(n_grid, se, "o-", color="#1f77b4")
    axs[0].axvline(n_min_rec, color="crimson", ls="--",
                    label=f"$n_{{min}}$ recomendado = {n_min_rec}")
    axs[0].set_xscale("log"); axs[0].set_yscale("log")
    axs[0].set_xlabel("N (partículas por remuestreo)")
    axs[0].set_ylabel(r"SE bootstrap de $\langle p_T\rangle$ [GeV/c]")
    axs[0].set_title("Ruido estadístico puro vs. N")
    axs[0].grid(ls=":", alpha=0.6); axs[0].legend(fontsize=8)

    axs[1].plot(n_grid, rel * 100, "o-", color="#2ca02c")
    axs[1].axhline(rel_threshold * 100, color="gray", ls=":",
                    label=f"umbral = {rel_threshold*100:.0f}%")
    axs[1].axvline(n_min_rec, color="crimson", ls="--")
    axs[1].set_xscale("log")
    axs[1].set_xlabel("N (partículas por remuestreo)")
    axs[1].set_ylabel(r"SE / $\sigma_{\rm partícula}$  [%]")
    axs[1].set_title("Criterio de selección de $n_{min}$")
    axs[1].grid(ls=":", alpha=0.6); axs[1].legend(fontsize=8)

    fig.suptitle(f"Estimación bootstrap de $n_{{min}}$ — {tag}", fontsize=10)
    fig.tight_layout()
    path = outdir / f"bootstrap_nmin_{tag}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Guardado: {path}")


def build_event_histograms(events_ci, n_min, n_boot=200, d_bins=25,
                            seed=42, edge_percentiles=(0.5, 99.5)):
    """
    Para cada evento con N_i >= n_min: remuestrea sus propias N_i
    partículas n_boot veces (con reemplazo), calcula <pT> en cada
    remuestreo, y arma un histograma normalizado (d_bins) de esas
    estimaciones. El binning es COMPARTIDO dentro de la clase (definido
    por percentiles del pool de todas las estimaciones bootstrap), para
    que los histogramas de distintos eventos sean comparables entre sí.

    Retorna:
        X          : (N_ev_validos, d_bins) float32 — histogramas normalizados
        pt_mean    : (N_ev_validos,) — <pT>_i (promedio de las réplicas bootstrap)
        boot_se    : (N_ev_validos,) — std de las réplicas bootstrap (incertidumbre)
        n_part     : (N_ev_validos,) — N_i
        b_arr      : (N_ev_validos,) — parámetro de impacto
        bin_edges  : (d_bins+1,) — bordes de los bins compartidos
    """
    rng = np.random.default_rng(seed)
    valid = [e for e in events_ci if e["n"] >= n_min]
    if len(valid) == 0:
        return None

    all_boot_means = []
    pt_mean, boot_se, n_part, b_arr = [], [], [], []
    for e in valid:
        pt = e["pt"]
        N  = e["n"]
        idx = rng.integers(0, N, size=(n_boot, N))
        means = pt[idx].mean(axis=1)
        all_boot_means.append(means)
        pt_mean.append(means.mean())
        boot_se.append(means.std())
        n_part.append(N)
        b_arr.append(e["b"])

    pooled = np.concatenate(all_boot_means)
    lo, hi = np.percentile(pooled, edge_percentiles)
    if hi <= lo:
        hi = lo + 1e-6
    bin_edges = np.linspace(lo, hi, d_bins + 1)

    X = np.empty((len(valid), d_bins), dtype=np.float32)
    for i, means in enumerate(all_boot_means):
        counts, _ = np.histogram(means, bins=bin_edges)
        total = counts.sum()
        X[i] = counts / total if total > 0 else 1.0 / d_bins

    return {
        "X":         X,
        "pt_mean":   np.array(pt_mean,  dtype=np.float64),
        "boot_se":   np.array(boot_se,  dtype=np.float64),
        "n_part":    np.array(n_part,   dtype=np.int32),
        "b":         np.array(b_arr,    dtype=np.float32),
        "bin_edges": bin_edges,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  3. ARQUITECTURA VAE  (salida Softmax: el input es una distribución)
# ═══════════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, z_dim):
        super().__init__()
        layers, in_d = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.LayerNorm(h), nn.LeakyReLU(0.1)]
            in_d = h
        self.net    = nn.Sequential(*layers)
        self.mu     = nn.Linear(in_d, z_dim)
        self.logvar = nn.Linear(in_d, z_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dims, output_dim):
        super().__init__()
        layers, in_d = [], z_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.LayerNorm(h), nn.LeakyReLU(0.1)]
            in_d = h
        layers.append(nn.Linear(in_d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        logits = self.net(z)
        return F.softmax(logits, dim=-1)   # el output es una distribución


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, z_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dims, z_dim)
        self.decoder = Decoder(z_dim, list(reversed(hidden_dims)), input_dim)
        self.z_dim   = z_dim

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z          = self.reparameterize(mu, logvar)
        x_rec      = self.decoder(z)
        return x_rec, mu, logvar, z

    def loss(self, x, x_rec, mu, logvar, beta=1.0):
        l_rec = F.mse_loss(x_rec, x, reduction="mean")
        d_kl  = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return l_rec + beta * d_kl, l_rec, d_kl


def train_vae(X, input_dim, z_dim=2, hidden_dims=(16, 8), beta=1.0,
              epochs=150, batch_size=256, lr=1e-3, val_frac=0.15,
              device=None, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xt      = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(Xt)
    n_val   = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )
    bs = min(batch_size, max(1, n_train // 2))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              drop_last=(n_train > bs))
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False)

    model = VAE(input_dim, hidden_dims, z_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                        eta_min=lr / 20)

    history = {k: [] for k in
               ("train_loss","val_loss","train_rec","val_rec","train_kl","val_kl")}
    best_val, best_state = np.inf, None

    for epoch in range(1, epochs + 1):
        model.train()
        tl = tr = tk = 0.
        for (xb,) in train_loader:
            xb = xb.to(device)
            opt.zero_grad()
            x_rec, mu, logvar, _ = model(xb)
            loss, lrec, lkl = model.loss(xb, x_rec, mu, logvar, beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            b_ = xb.size(0)
            tl += loss.item()*b_; tr += lrec.item()*b_; tk += lkl.item()*b_
        sched.step()

        model.eval()
        vl = vr = vk = 0.
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                x_rec, mu, logvar, _ = model(xb)
                loss, lrec, lkl = model.loss(xb, x_rec, mu, logvar, beta)
                b_ = xb.size(0)
                vl += loss.item()*b_; vr += lrec.item()*b_; vk += lkl.item()*b_

        for key, val in [("train_loss", tl/n_train), ("val_loss", vl/n_val),
                          ("train_rec",  tr/n_train), ("val_rec",  vr/n_val),
                          ("train_kl",   tk/n_train), ("val_kl",   vk/n_val)]:
            history[key].append(val)

        if vl/n_val < best_val:
            best_val   = vl/n_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 25 == 0 or epoch == 1:
            print(f"    Época {epoch:4d}/{epochs}  loss={tl/n_train:.5f}  "
                  f"val={vl/n_val:.5f}  rec={tr/n_train:.5f}  "
                  f"kl={tk/n_train:.5f}", flush=True)

    model.load_state_dict(best_state)
    return model, history, best_val


@torch.no_grad()
def encode_all(model, X, device=None, batch_size=2048):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xt), batch_size=batch_size, shuffle=False)
    zs, recs = [], []
    for (xb,) in loader:
        xb = xb.to(device)
        mu, _ = model.encoder(xb)
        x_rec, _, _, _ = model(xb)
        zs.append(mu.cpu().numpy()); recs.append(x_rec.cpu().numpy())
    return np.vstack(zs), np.vstack(recs)


# ═══════════════════════════════════════════════════════════════════════════
#  4. CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════

def clustering_analysis(Z, k_range=(2, 7)):
    Zs = StandardScaler().fit_transform(Z)
    km_scores = {}
    for k in range(*k_range):
        if k >= len(Z):
            continue
        km   = KMeans(n_clusters=k, random_state=42, n_init=10)
        labs = km.fit_predict(Zs)
        if len(np.unique(labs)) > 1:
            km_scores[k] = (silhouette_score(Zs, labs), labs)
    if not km_scores:
        return None
    k_best  = max(km_scores, key=lambda k: km_scores[k][0])
    km_labs, km_sil = km_scores[k_best][1], km_scores[k_best][0]

    nn_  = NearestNeighbors(n_neighbors=min(5, len(Z)-1)).fit(Zs)
    d, _ = nn_.kneighbors(Zs)
    eps  = max(float(np.percentile(d[:, -1], 5)), 0.3)
    db   = DBSCAN(eps=eps, min_samples=5)
    db_labs = db.fit_predict(Zs)
    n_db    = len(set(db_labs) - {-1})
    db_sil  = silhouette_score(Zs, db_labs) if n_db > 1 else np.nan

    return dict(km_labs=km_labs, k_best=k_best, km_sil=km_sil,
                db_labs=db_labs, n_db=n_db, db_sil=db_sil, km_scores=km_scores)


# ═══════════════════════════════════════════════════════════════════════════
#  5. GRÁFICAS (todas etiquetadas por clase de centralidad individual)
# ═══════════════════════════════════════════════════════════════════════════

def plot_loss_curves(history, tag, outdir):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
    ep = range(1, len(history["train_loss"]) + 1)
    pairs = [("train_loss","val_loss", r"Pérdida total $\mathcal{L}$"),
             ("train_rec", "val_rec",  r"Reconstrucción"),
             ("train_kl",  "val_kl",   r"$\beta \cdot D_{KL}$")]
    for ax, (tr, vl, title) in zip(axs, pairs):
        ax.plot(ep, history[tr], label="Train", lw=1.4)
        ax.plot(ep, history[vl], label="Val", lw=1.4, ls="--")
        ax.set_title(title, fontsize=9); ax.set_xlabel("Época")
        ax.legend(fontsize=7); ax.grid(ls=":", alpha=0.5)
    fig.suptitle(f"Curvas de pérdida — {tag}", fontsize=10)
    fig.tight_layout()
    path = outdir / f"loss_{tag}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"    Guardado: {path}")


def plot_latent_and_clustering(Z, hd, cl, tag, outdir):
    z_dim = Z.shape[1]
    if z_dim < 2:
        z1, z2 = Z[:, 0], np.zeros_like(Z[:, 0])
    else:
        z1, z2 = Z[:, 0], Z[:, 1]

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))

    sc0 = axs[0].scatter(z1, z2, c=hd["n_part"], cmap="viridis",
                          s=6, alpha=0.5, rasterized=True)
    axs[0].set_title(r"Coloreado por $N_i$ (multiplicidad)", fontsize=9)
    plt.colorbar(sc0, ax=axs[0], label=r"$N_i$")

    sc1 = axs[1].scatter(z1, z2, c=hd["boot_se"], cmap="magma",
                          s=6, alpha=0.5, rasterized=True)
    axs[1].set_title(r"Coloreado por SE bootstrap de $\langle p_T\rangle_i$", fontsize=9)
    plt.colorbar(sc1, ax=axs[1], label="SE [GeV/c]")

    if cl is not None:
        sc2 = axs[2].scatter(z1, z2, c=cl["km_labs"], cmap="tab10",
                              s=6, alpha=0.6, rasterized=True)
        axs[2].set_title(f"K-means k={cl['k_best']}  "
                          f"Silhouette={cl['km_sil']:.3f}", fontsize=9)
    else:
        axs[2].text(0.5, 0.5, "Muy pocos eventos\npara clustering",
                    ha="center", va="center", transform=axs[2].transAxes)

    for ax in axs:
        ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$")
        ax.grid(ls=":", alpha=0.4)
    fig.suptitle(f"Espacio latente — {tag}", fontsize=10)
    fig.tight_layout()
    path = outdir / f"latent_{tag}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"    Guardado: {path}")


def plot_reconstruction_examples(X, recs, hd, tag, outdir, n_examples=6):
    edges = hd["bin_edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    order = np.argsort(hd["n_part"])
    idxs = order[np.linspace(0, len(order) - 1, n_examples, dtype=int)]

    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    for ax, i in zip(axs.ravel(), idxs):
        ax.step(centers, X[i],    where="mid", color="black", lw=1.4,
                label="Bootstrap (real)")
        ax.step(centers, recs[i], where="mid", color="crimson", lw=1.2,
                ls="--", label="VAE (reconstruido)")
        ax.set_title(f"$N_i$={hd['n_part'][i]}  "
                      f"SE={hd['boot_se'][i]:.3f} GeV/c", fontsize=8)
        ax.set_xlabel(r"$\langle p_T\rangle$ [GeV/c]", fontsize=8)
        ax.grid(ls=":", alpha=0.4)
    axs.ravel()[0].legend(fontsize=7)
    fig.suptitle(f"Reconstrucción de histogramas bootstrap — {tag}", fontsize=10)
    fig.tight_layout()
    path = outdir / f"reconstruction_{tag}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"    Guardado: {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  6. MAIN — procesa cada clase de centralidad por separado
# ═══════════════════════════════════════════════════════════════════════════

def process_class(ci, events_ci, args, tag_base, outdir):
    label = CENT_LABELS[ci]
    tag   = f"{tag_base}_{label.replace('%','pct')}"
    print(f"\n{'='*65}\nClase de centralidad: {label}  ({len(events_ci)} eventos)\n{'='*65}")

    if len(events_ci) < 50:
        print("  Muy pocos eventos en esta clase; se omite.")
        return None

    # ── n_min por bootstrap (pool de partículas de ESTA clase) ─────────────
    pt_pool = np.concatenate([e["pt"] for e in events_ci])
    n_min_rec, n_grid, se, sigma_pool = estimate_n_min(
        pt_pool, n_boot=args.n_boot_nmin, rel_threshold=args.rel_threshold)
    n_min = args.n_min if args.n_min is not None else n_min_rec
    print(f"  sigma_partícula={sigma_pool:.4f} GeV/c   "
          f"n_min recomendado={n_min_rec}   n_min usado={n_min}")
    if args.n_min is not None and args.n_min < n_min_rec:
        print(f"  [AVISO] n_min manual ({args.n_min}) es menor que el "
              f"recomendado por bootstrap ({n_min_rec}).")
    plot_bootstrap_nmin(n_grid, se, sigma_pool, n_min_rec,
                         args.rel_threshold, tag, outdir)

    # ── Histogramas bootstrap por evento ────────────────────────────────────
    hd = build_event_histograms(events_ci, n_min, n_boot=args.n_boot,
                                 d_bins=args.d_bins, seed=args.seed)
    if hd is None or len(hd["X"]) < 50:
        print(f"  Menos de 50 eventos sobreviven al corte n_min={n_min}; se omite.")
        return None
    print(f"  Eventos usados (N_i >= {n_min}): {len(hd['X'])}")
    print(f"  <pT> (ensemble): mean={hd['pt_mean'].mean():.4f}  "
          f"std={hd['pt_mean'].std():.4f} GeV/c   "
          f"[cf. sigma^2_pT del protocolo]")

    # ── Entrenar VAE ─────────────────────────────────────────────────────
    print(f"  Entrenando VAE (input_dim={args.d_bins}, z_dim={args.z_dim})...")
    model, history, best_val = train_vae(
        hd["X"], input_dim=args.d_bins, z_dim=args.z_dim,
        hidden_dims=(16, 8), beta=args.beta, epochs=args.epochs,
        batch_size=args.batch, lr=args.lr, seed=args.seed)
    print(f"    Mejor val_loss: {best_val:.6f}")

    device = next(model.parameters()).device
    Z, recs = encode_all(model, hd["X"], device=device)

    cl = clustering_analysis(Z)
    if cl is not None:
        print(f"  K-means: k={cl['k_best']}  Silhouette={cl['km_sil']:.3f}  |  "
              f"DBSCAN: {cl['n_db']} clusters  Silhouette={cl['db_sil']:.3f}")

    plot_loss_curves(history, tag, outdir)
    plot_latent_and_clustering(Z, hd, cl, tag, outdir)
    plot_reconstruction_examples(hd["X"], recs, hd, tag, outdir)

    return {
        "label": label, "n_events": len(hd["X"]), "n_min": n_min,
        "n_min_rec": n_min_rec, "sigma_pool": sigma_pool,
        "pt_mean": hd["pt_mean"].mean(), "pt_std": hd["pt_mean"].std(),
        "best_val": best_val,
        "km_sil": cl["km_sil"] if cl else np.nan,
        "db_sil": cl["db_sil"] if cl else np.nan,
    }


def main():
    parser = argparse.ArgumentParser(
        description="VAE sobre histogramas bootstrap de <pT> por evento, "
                     "separado por clase de centralidad."
    )
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--charged",       action="store_true")
    parser.add_argument("--max-events",    type=int, default=None)
    parser.add_argument("--max-per-class", type=int, default=6000,
                        help="Máximo de eventos por clase (control de memoria/"
                             "tiempo de cómputo). Default 6000.")
    parser.add_argument("--n-min",         type=int, default=None,
                        help="Forzar n_min manual. Si no se da, se usa el "
                             "recomendado por bootstrap para cada clase.")
    parser.add_argument("--rel-threshold", type=float, default=0.05,
                        help="Umbral relativo SE/sigma_partícula para elegir "
                             "n_min (default 0.05 = 5%%).")
    parser.add_argument("--n-boot-nmin",   type=int, default=1000,
                        help="Réplicas bootstrap para el estudio de n_min.")
    parser.add_argument("--n-boot",        type=int, default=200,
                        help="Réplicas bootstrap por evento para construir "
                             "su histograma de <pT>.")
    parser.add_argument("--d-bins",        type=int, default=25,
                        help="Bins del histograma de entrada al VAE.")
    parser.add_argument("--z-dim",         type=int, default=2)
    parser.add_argument("--beta",          type=float, default=1.0)
    parser.add_argument("--epochs",        type=int, default=150)
    parser.add_argument("--batch",         type=int, default=256)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--outdir",        default=None)
    parser.add_argument("--tag",           default=None)
    args = parser.parse_args()

    run_dirs = [Path(d) for d in args.run_dirs]
    for d in run_dirs:
        if not d.exists():
            sys.exit(f"ERROR: no existe {d}")

    tag_base = args.tag or (run_dirs[0].name if len(run_dirs) == 1
                             else f"{run_dirs[0].name}_combined")
    tag_base += f"_ptmean_d{args.d_bins}_z{args.z_dim}_b{args.beta}"

    outdir = Path(args.outdir) if args.outdir else run_dirs[0] / "vae_pt_mean_plots"
    outdir.mkdir(parents=True, exist_ok=True)

    particle_type = "cargadas" if args.charged else "neutras"
    print("=" * 65)
    print(f"VAE — input: histograma bootstrap de <pT>_i  |  partículas: {particle_type}")
    print(f"d_bins={args.d_bins}  z_dim={args.z_dim}  β={args.beta}  "
          f"n_boot={args.n_boot}  epochs={args.epochs}")
    print("=" * 65)

    print("\n[1] Cargando eventos y clasificando por centralidad...")
    events = load_events_by_class(run_dirs, charged=args.charged,
                                   max_events=args.max_events,
                                   max_per_class=args.max_per_class)

    print("\n[2] Procesando cada clase de centralidad por separado...")
    summary = []
    for ci in range(len(CENT_LABELS)):
        res = process_class(ci, events[ci], args, tag_base, outdir)
        if res is not None:
            summary.append(res)

    print(f"\n{'='*65}\nResumen por clase de centralidad\n{'='*65}")
    hdr = f"{'Clase':<10}{'N_ev':>7}{'n_min':>7}{'n_min_rec':>11}{'<pT>':>9}{'std<pT>':>10}{'val_loss':>10}{'KM_sil':>8}{'DB_sil':>8}"
    print(hdr)
    for r in summary:
        print(f"{r['label']:<10}{r['n_events']:>7}{r['n_min']:>7}"
              f"{r['n_min_rec']:>11}{r['pt_mean']:>9.4f}{r['pt_std']:>10.4f}"
              f"{r['best_val']:>10.5f}{r['km_sil']:>8.3f}{r['db_sil']:>8.3f}")

    print(f"\nListo. Resultados en: {outdir}")


if __name__ == "__main__":
    main()
