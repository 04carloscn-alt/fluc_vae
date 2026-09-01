#!/usr/bin/env python3
"""
vae_pt.py — Red Neuronal Variacional (VAE) para análisis de fluctuaciones
            evento a evento del espectro de momento transverso ⟨pT⟩.

Descripción
───────────
Cada evento se representa como un histograma normalizado de pT (60 bins,
0–3 GeV/c). El VAE aprende una representación latente de baja dimensión
de la FORMA del espectro, sin supervisión.

El espacio latente se analiza en dos niveles:
  1. Correlación con observables físicos conocidos:
       b (parámetro de impacto), clase de centralidad, ⟨pT⟩, multiplicidad N.
  2. Clustering no supervisado (K-means + DBSCAN) para identificar
       estructuras latentes y evaluarlas con Silhouette score.

Arquitectura VAE
────────────────
  Encoder: 60 → 256 → 128 → (μ_z, log σ²_z)  [z_dim neuronas cada uno]
  Decoder: z_dim → 128 → 256 → 60  (salida Softmax: reconstruye distribución)

  Función de pérdida (Ec. 1.3 del protocolo):
      L = L_rec + β · D_KL
      L_rec = MSE(x_rec, x)   [o BCE si --bce]
      D_KL  = -½ Σ (1 + log σ² - μ² - σ²)

Gráficas producidas
───────────────────
  loss_curves_*.png       : L_total, L_rec, β·D_KL vs época (train/val)
  latent_centrality_*.png : scatter z₁ vs z₂ coloreado por centralidad
  latent_b_*.png          : scatter z₁ vs z₂ coloreado por b continuo
  latent_pt_*.png         : scatter z₁ vs z₂ coloreado por ⟨pT⟩
  correlation_*.png       : heatmap ρ(zₖ, observable) Pearson
  clustering_*.png        : K-means y DBSCAN sobre el espacio latente
  reconstruction_*.png    : espectros originales vs reconstruidos por centralidad

Uso
───
  # Una carpeta
  python3 analysis/vae_pt.py output/Bi_11GeV_MB --charged

  # Varias carpetas (100 k eventos)
  python3 analysis/vae_pt.py output/Bi_11GeV_MB* --charged

  # Opciones avanzadas
  python3 analysis/vae_pt.py output/Bi_11GeV_MB* --charged \\
      --z-dim 4 --beta 1.0 --epochs 100 --batch 256 --lr 3e-4 \\
      --outdir output/vae_results
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_f14 import iter_all


# ── Bins (mismos que build_dataset.py) ───────────────────────────────────────
PT_BINS  = np.linspace(0.0, 3.0, 61)   # 60 bins
N_BINS   = len(PT_BINS) - 1            # 60

# ── Centralidad ───────────────────────────────────────────────────────────────
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


# ═══════════════════════════════════════════════════════════════════════════════
#  1. CARGA DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset(output_dirs, charged=False, max_events=None):
    """
    Lee eventos de uno o varios directorios run_NNNN/.
    Por cada evento construye:
        hist  : histograma de pT normalizado (60 bins)  float32
        b     : parámetro de impacto [fm]               float32
        cent  : índice de centralidad 0-9               int
        pt_mean : ⟨pT⟩ del evento [GeV/c]              float32
        n_part  : multiplicidad filtrada                int

    Solo se incluyen eventos con N >= 2 partículas filtradas.
    """
    if isinstance(output_dirs, (str, Path)):
        output_dirs = [output_dirs]

    hists, bs, cents, pt_means, n_parts = [], [], [], [], []
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
            n = len(pt_vals)
            if n < 2:
                n_total += 1
                continue

            # Histograma normalizado (área = 1)
            h, _ = np.histogram(pt_vals, bins=PT_BINS)
            h    = h.astype(np.float32)
            s    = h.sum()
            if s > 0:
                h /= s

            hists.append(h)
            bs.append(float(ev["b"]))
            cents.append(assign_centrality(ev["b"]))
            pt_means.append(float(pt_vals.mean()))
            n_parts.append(n)

            n_total += 1
            if n_total % 2000 == 0:
                print(f"    {n_total} eventos procesados...", flush=True)
            if max_events is not None and n_total >= max_events:
                break

    print(f"  Total eventos cargados: {len(hists)}", flush=True)
    return {
        "hist":    np.array(hists,    dtype=np.float32),
        "b":       np.array(bs,       dtype=np.float32),
        "cent":    np.array(cents,    dtype=np.int32),
        "pt_mean": np.array(pt_means, dtype=np.float32),
        "n_part":  np.array(n_parts,  dtype=np.int32),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  2. ARQUITECTURA VAE
# ═══════════════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, z_dim):
        super().__init__()
        layers = []
        in_d = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.1)]
            in_d = h
        self.net   = nn.Sequential(*layers)
        self.mu    = nn.Linear(in_d, z_dim)
        self.logvar = nn.Linear(in_d, z_dim)

    def forward(self, x):
        h      = self.net(x)
        mu     = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        in_d = z_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.1)]
            in_d = h
        layers += [nn.Linear(in_d, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        # Softmax para reconstruir una distribución de probabilidad normalizada
        return F.softmax(self.net(z), dim=-1)


class VAE(nn.Module):
    def __init__(self, input_dim=60, hidden_dims=(256, 128), z_dim=2):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dims, z_dim)
        self.decoder = Decoder(z_dim, list(reversed(hidden_dims)), input_dim)
        self.z_dim   = z_dim

    def reparameterize(self, mu, logvar):
        """Truco de reparametrización: z = μ + ε·σ,  ε ~ N(0,I)"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu   # en evaluación: usar la media directamente

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z          = self.reparameterize(mu, logvar)
        x_rec      = self.decoder(z)
        return x_rec, mu, logvar, z

    def loss(self, x, x_rec, mu, logvar, beta=1.0, use_bce=False):
        """
        L = L_rec + β · D_KL
        L_rec : MSE (default) o BCE sobre el histograma normalizado
        D_KL  : divergencia KL analítica (Ec. 1.3 del protocolo)
        """
        if use_bce:
            # Clamp para estabilidad numérica
            x_rec_c = x_rec.clamp(1e-8, 1 - 1e-8)
            x_c     = x.clamp(1e-8, 1 - 1e-8)
            l_rec   = F.binary_cross_entropy(x_rec_c, x_c, reduction="mean")
        else:
            l_rec = F.mse_loss(x_rec, x, reduction="mean")

        # D_KL = -½ Σ_j (1 + log σ²_j - μ²_j - σ²_j)  promediado sobre el batch
        d_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return l_rec + beta * d_kl, l_rec, d_kl


# ═══════════════════════════════════════════════════════════════════════════════
#  3. ENTRENAMIENTO
# ═══════════════════════════════════════════════════════════════════════════════

def train_vae(data, z_dim=2, hidden_dims=(256, 128), beta=1.0,
              epochs=80, batch_size=256, lr=3e-4, val_frac=0.15,
              use_bce=False, device=None, seed=42):
    """
    Entrena el VAE y devuelve (model, history).
    history : dict con listas "train_loss", "val_loss", "train_rec",
              "val_rec", "train_kl", "val_kl" por época.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Dispositivo: {device}")

    X = torch.tensor(data["hist"], dtype=torch.float32)
    dataset    = TensorDataset(X)
    n_val      = max(1, int(len(dataset) * val_frac))
    n_train    = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model = VAE(input_dim=N_BINS, hidden_dims=hidden_dims, z_dim=z_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr/20)

    history = {k: [] for k in
               ("train_loss","val_loss","train_rec","val_rec","train_kl","val_kl")}

    best_val  = np.inf
    best_state = None

    for epoch in range(1, epochs + 1):
        # ── Entrenamiento ─────────────────────────────────────────────────────
        model.train()
        tl, tr, tk = 0., 0., 0.
        for (xb,) in train_loader:
            xb = xb.to(device)
            opt.zero_grad()
            x_rec, mu, logvar, _ = model(xb)
            loss, lrec, lkl = model.loss(xb, x_rec, mu, logvar, beta, use_bce)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs_ = xb.size(0)
            tl += loss.item() * bs_
            tr += lrec.item() * bs_
            tk += lkl.item()  * bs_
        scheduler.step()

        # ── Validación ────────────────────────────────────────────────────────
        model.eval()
        vl, vr, vk = 0., 0., 0.
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                x_rec, mu, logvar, _ = model(xb)
                loss, lrec, lkl = model.loss(xb, x_rec, mu, logvar, beta, use_bce)
                bs_ = xb.size(0)
                vl += loss.item() * bs_
                vr += lrec.item() * bs_
                vk += lkl.item()  * bs_

        for key, val in [("train_loss", tl/n_train), ("val_loss", vl/n_val),
                         ("train_rec",  tr/n_train), ("val_rec",  vr/n_val),
                         ("train_kl",   tk/n_train), ("val_kl",   vk/n_val)]:
            history[key].append(val)

        if vl / n_val < best_val:
            best_val   = vl / n_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Época {epoch:4d}/{epochs}  "
                  f"loss={tl/n_train:.5f}  val={vl/n_val:.5f}  "
                  f"rec={tr/n_train:.5f}  kl={tk/n_train:.5f}",
                  flush=True)

    model.load_state_dict(best_state)
    print(f"  Mejor val_loss: {best_val:.5f}")
    return model, history


# ═══════════════════════════════════════════════════════════════════════════════
#  4. INFERENCIA: OBTENER REPRESENTACIONES LATENTES
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def encode_all(model, data, batch_size=512, device=None):
    """
    Pasa todos los histogramas por el encoder y devuelve μ_z (media latente)
    y las reconstrucciones para cada evento.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    X       = torch.tensor(data["hist"], dtype=torch.float32)
    loader  = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)
    zs, recs = [], []

    for (xb,) in loader:
        xb = xb.to(device)
        mu, _ = model.encoder(xb)
        x_rec, _, _, _ = model(xb)
        zs.append(mu.cpu().numpy())
        recs.append(x_rec.cpu().numpy())

    return np.vstack(zs), np.vstack(recs)


# ═══════════════════════════════════════════════════════════════════════════════
#  5. ANÁLISIS DEL ESPACIO LATENTE
# ═══════════════════════════════════════════════════════════════════════════════

def correlation_analysis(Z, data):
    """
    Calcula ρ de Pearson entre cada dimensión latente zₖ y los observables
    físicos: b, centralidad (índice), ⟨pT⟩, multiplicidad N.
    Devuelve una matriz (n_obs, z_dim).
    """
    observables = {
        "b":           data["b"],
        "centralidad": data["cent"].astype(float),
        r"$\langle p_T \rangle$": data["pt_mean"],
        "N partículas":   data["n_part"].astype(float),
    }
    obs_names = list(observables.keys())
    z_dim = Z.shape[1]
    corr  = np.zeros((len(obs_names), z_dim))
    for i, name in enumerate(obs_names):
        for j in range(z_dim):
            mask = np.isfinite(observables[name]) & np.isfinite(Z[:, j])
            if mask.sum() > 5:
                corr[i, j] = np.corrcoef(observables[name][mask],
                                         Z[mask, j])[0, 1]
    return corr, obs_names


def clustering_analysis(Z, n_clusters_range=(2, 8)):
    """
    K-means con k = 2..8 + DBSCAN. Selecciona el k óptimo por Silhouette.
    Retorna etiquetas kmeans (k óptimo), etiquetas dbscan, scores.
    """
    Zs = StandardScaler().fit_transform(Z)

    # K-means
    kmeans_scores = {}
    for k in range(*n_clusters_range):
        km   = KMeans(n_clusters=k, random_state=42, n_init=10)
        labs = km.fit_predict(Zs)
        if len(np.unique(labs)) > 1:
            kmeans_scores[k] = (silhouette_score(Zs, labs), labs)

    k_best  = max(kmeans_scores, key=lambda k: kmeans_scores[k][0])
    km_labs = kmeans_scores[k_best][1]
    km_sil  = kmeans_scores[k_best][0]

    # DBSCAN (eps adaptativo: percentil 5 de distancias entre vecinos)
    from sklearn.neighbors import NearestNeighbors
    nn  = NearestNeighbors(n_neighbors=5).fit(Zs)
    dists, _ = nn.kneighbors(Zs)
    eps = float(np.percentile(dists[:, -1], 5))
    eps = max(eps, 0.3)

    db      = DBSCAN(eps=eps, min_samples=5)
    db_labs = db.fit_predict(Zs)
    n_db    = len(set(db_labs) - {-1})
    db_sil  = silhouette_score(Zs, db_labs) if n_db > 1 else np.nan

    return km_labs, db_labs, k_best, km_sil, db_sil, kmeans_scores


# ═══════════════════════════════════════════════════════════════════════════════
#  6. GRÁFICAS
# ═══════════════════════════════════════════════════════════════════════════════

def _scatter_latent(ax, Z, color_vals, cmap, norm, label, title, colorbar_label):
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=color_vals, cmap=cmap, norm=norm,
                    s=6, alpha=0.5, rasterized=True)
    ax.set_xlabel(r"$z_1$", fontsize=9)
    ax.set_ylabel(r"$z_2$", fontsize=9)
    ax.set_title(title, fontsize=9)
    return sc


def plot_loss_curves(history, tag, outdir):
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    pairs = [
        ("train_loss", "val_loss",  "Pérdida total $\\mathcal{L}$"),
        ("train_rec",  "val_rec",   "Reconstrucción $\\mathcal{L}_{rec}$"),
        ("train_kl",   "val_kl",    "$\\beta \\cdot D_{KL}$"),
    ]
    for ax, (tr_key, val_key, title) in zip(axs, pairs):
        ax.plot(epochs, history[tr_key],  label="Train", linewidth=1.5)
        ax.plot(epochs, history[val_key], label="Val",   linewidth=1.5, linestyle="--")
        ax.set_xlabel("Época", fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(linestyle=":", alpha=0.6)

    fig.suptitle(f"Curvas de pérdida VAE — {tag}", fontsize=10)
    fig.tight_layout()
    path = outdir / f"loss_curves_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Guardado: {path}")


def plot_latent_scatter(Z, data, tag, outdir):
    """Tres scatter plots del espacio latente z₁-z₂."""
    if Z.shape[1] < 2:
        print("  [AVISO] z_dim < 2, omitiendo scatter latente.")
        return

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Espacio latente — {tag}", fontsize=11)

    # ── Panel 1: centralidad ─────────────────────────────────────────────────
    ax = axs[0]
    for ci, (label, color) in enumerate(zip(CENT_LABELS, CENT_COLORS)):
        mask = data["cent"] == ci
        if mask.sum() == 0:
            continue
        ax.scatter(Z[mask, 0], Z[mask, 1], c=color, s=6, alpha=0.5,
                   label=label, rasterized=True)
    ax.set_xlabel(r"$z_1$", fontsize=9)
    ax.set_ylabel(r"$z_2$", fontsize=9)
    ax.set_title("Coloreado por centralidad", fontsize=9)
    ax.legend(fontsize=5.5, ncol=2, markerscale=2, loc="best")

    # ── Panel 2: b continuo ──────────────────────────────────────────────────
    ax = axs[1]
    norm = Normalize(vmin=data["b"].min(), vmax=data["b"].max())
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=data["b"], cmap="plasma",
                    norm=norm, s=6, alpha=0.5, rasterized=True)
    plt.colorbar(sc, ax=ax, label="b [fm]", shrink=0.85)
    ax.set_xlabel(r"$z_1$", fontsize=9)
    ax.set_ylabel(r"$z_2$", fontsize=9)
    ax.set_title("Coloreado por $b$ [fm]", fontsize=9)

    # ── Panel 3: ⟨pT⟩ ───────────────────────────────────────────────────────
    ax = axs[2]
    norm2 = Normalize(vmin=data["pt_mean"].min(), vmax=data["pt_mean"].max())
    sc2 = ax.scatter(Z[:, 0], Z[:, 1], c=data["pt_mean"], cmap="viridis",
                     norm=norm2, s=6, alpha=0.5, rasterized=True)
    plt.colorbar(sc2, ax=ax, label=r"$\langle p_T \rangle$ [GeV/c]", shrink=0.85)
    ax.set_xlabel(r"$z_1$", fontsize=9)
    ax.set_ylabel(r"$z_2$", fontsize=9)
    ax.set_title(r"Coloreado por $\langle p_T \rangle$", fontsize=9)

    fig.tight_layout()
    path = outdir / f"latent_scatter_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Guardado: {path}")


def plot_correlation_heatmap(Z, data, tag, outdir):
    corr, obs_names = correlation_analysis(Z, data)
    z_dim = Z.shape[1]

    fig, ax = plt.subplots(figsize=(max(5, z_dim * 1.2 + 2), 4))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label=r"$\rho$ Pearson")

    ax.set_xticks(range(z_dim))
    ax.set_xticklabels([f"$z_{{{k+1}}}$" for k in range(z_dim)], fontsize=9)
    ax.set_yticks(range(len(obs_names)))
    ax.set_yticklabels(obs_names, fontsize=9)

    # Anotar valores
    for i in range(len(obs_names)):
        for j in range(z_dim):
            v = corr[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if abs(v) < 0.6 else "white")

    ax.set_title(f"Correlaciones espacio latente — {tag}", fontsize=10)
    fig.tight_layout()
    path = outdir / f"correlation_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Guardado: {path}")


def plot_clustering(Z, data, tag, outdir):
    if Z.shape[1] < 2:
        print("  [AVISO] z_dim < 2, omitiendo gráfica de clustering.")
        return

    km_labs, db_labs, k_best, km_sil, db_sil, kmeans_scores = \
        clustering_analysis(Z)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Clustering espacio latente — {tag}", fontsize=11)

    # ── K-means ──────────────────────────────────────────────────────────────
    ax = axs[0]
    cmap_km = matplotlib.colormaps.get_cmap("tab10").resampled(k_best)
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=km_labs, cmap=cmap_km,
                    vmin=-0.5, vmax=k_best - 0.5, s=6, alpha=0.6, rasterized=True)
    plt.colorbar(sc, ax=ax, ticks=range(k_best), label="Cluster")
    ax.set_title(f"K-means  k={k_best}  Silhouette={km_sil:.3f}", fontsize=9)
    ax.set_xlabel(r"$z_1$"); ax.set_ylabel(r"$z_2$")

    # ── DBSCAN ───────────────────────────────────────────────────────────────
    ax = axs[1]
    n_db = len(set(db_labs) - {-1})
    cmap_db = matplotlib.colormaps.get_cmap("tab10").resampled(max(n_db, 2))
    sc2 = ax.scatter(Z[:, 0], Z[:, 1], c=db_labs, cmap=cmap_db,
                     s=6, alpha=0.6, rasterized=True)
    plt.colorbar(sc2, ax=ax, label="Cluster (-1=ruido)")
    sil_str = f"{db_sil:.3f}" if np.isfinite(db_sil) else "N/A"
    ax.set_title(f"DBSCAN  n_clusters={n_db}  Silhouette={sil_str}", fontsize=9)
    ax.set_xlabel(r"$z_1$"); ax.set_ylabel(r"$z_2$")

    # ── Silhouette vs k ───────────────────────────────────────────────────────
    ax = axs[2]
    ks     = sorted(kmeans_scores.keys())
    scores = [kmeans_scores[k][0] for k in ks]
    ax.plot(ks, scores, marker="o", color="steelblue", linewidth=1.8)
    ax.axvline(k_best, color="tomato", linestyle="--", linewidth=1.2,
               label=f"k óptimo = {k_best}")
    ax.set_xlabel("Número de clusters $k$", fontsize=9)
    ax.set_ylabel("Silhouette score", fontsize=9)
    ax.set_title("Selección de $k$ óptimo", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(linestyle=":", alpha=0.6)

    fig.tight_layout()
    path = outdir / f"clustering_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Guardado: {path}")

    # Tabla por clase de centralidad
    print("\n  Distribución de clusters K-means por centralidad:")
    print(f"  {'Clase':10s}", end="")
    for k in range(k_best):
        print(f" | C{k:2d}", end="")
    print()
    print("  " + "-" * (13 + k_best * 6))
    for ci, label in enumerate(CENT_LABELS):
        mask = data["cent"] == ci
        if mask.sum() == 0:
            continue
        print(f"  {label:10s}", end="")
        for k in range(k_best):
            n = int((km_labs[mask] == k).sum())
            print(f" | {n:4d}", end="")
        print()
    print()


def plot_reconstruction(Z, recs, data, tag, outdir, n_examples=3):
    """
    Para cada clase de centralidad muestra n_examples espectros originales
    (gris) y sus reconstrucciones (color), más el promedio de ambos.
    """
    bin_centers = 0.5 * (PT_BINS[:-1] + PT_BINS[1:])
    n_cent_shown = sum(
        1 for ci in range(10) if (data["cent"] == ci).sum() >= 2
    )
    n_cols = 5
    n_rows = (n_cent_shown + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.2, n_rows * 2.8),
                             squeeze=False)
    fig.suptitle(f"Reconstrucción VAE por centralidad — {tag}", fontsize=10)

    plot_idx = 0
    for ci, label in enumerate(CENT_LABELS):
        mask  = np.where(data["cent"] == ci)[0]
        if len(mask) < 2:
            continue

        row, col = divmod(plot_idx, n_cols)
        ax = axs[row][col]
        plot_idx += 1

        rng = np.random.default_rng(42)
        idxs = rng.choice(mask, size=min(n_examples, len(mask)), replace=False)

        mean_orig = data["hist"][mask].mean(axis=0)
        mean_rec  = recs[mask].mean(axis=0)

        # Ejemplos individuales
        for idx in idxs:
            ax.step(bin_centers, data["hist"][idx], where="mid",
                    color="gray", alpha=0.4, linewidth=0.8)
            ax.step(bin_centers, recs[idx], where="mid",
                    color=CENT_COLORS[ci], alpha=0.4, linewidth=0.8)

        # Promedios
        ax.step(bin_centers, mean_orig, where="mid",
                color="black", linewidth=1.5, label="Original")
        ax.step(bin_centers, mean_rec,  where="mid",
                color=CENT_COLORS[ci], linewidth=1.5,
                linestyle="--", label="VAE")

        ax.set_title(f"{label}  (N={len(mask)})", fontsize=8)
        ax.set_xlabel(r"$p_T$ [GeV/c]", fontsize=7)
        ax.set_ylabel("Frec. norm.", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)

    # Ocultar paneles vacíos
    for i in range(plot_idx, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axs[row][col].set_visible(False)

    fig.tight_layout()
    path = outdir / f"reconstruction_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Guardado: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  7. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="VAE para análisis de fluctuaciones evento a evento de pT"
    )
    parser.add_argument("run_dirs", nargs="+",
                        help="Directorios con subdirectorios run_NNNN/")
    parser.add_argument("--charged",    action="store_true",
                        help="Partículas cargadas (default: neutras ncl>0)")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Límite total de eventos")
    parser.add_argument("--z-dim",      type=int, default=2,
                        help="Dimensión del espacio latente (default: 2)")
    parser.add_argument("--beta",       type=float, default=1.0,
                        help="Peso del término KL (default: 1.0)")
    parser.add_argument("--epochs",     type=int, default=80,
                        help="Épocas de entrenamiento (default: 80)")
    parser.add_argument("--batch",      type=int, default=256,
                        help="Tamaño de batch (default: 256)")
    parser.add_argument("--lr",         type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--bce",        action="store_true",
                        help="Usar BCE en lugar de MSE para L_rec")
    parser.add_argument("--outdir",     default=None,
                        help="Directorio de salida")
    parser.add_argument("--tag",        default=None,
                        help="Etiqueta para archivos de salida")
    parser.add_argument("--save-model", action="store_true",
                        help="Guardar pesos del modelo en .pt")
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
    tag += f"_z{args.z_dim}_b{args.beta}"

    outdir = Path(args.outdir) if args.outdir else run_dirs[0] / "vae_plots"
    outdir.mkdir(parents=True, exist_ok=True)

    particle_type = "cargadas" if args.charged else "neutras"
    print("=" * 60)
    print(f"VAE — pT histograma  |  partículas: {particle_type}")
    print(f"z_dim={args.z_dim}  β={args.beta}  épocas={args.epochs}  "
          f"batch={args.batch}  lr={args.lr}")
    print("=" * 60)

    # ── 1. Cargar datos ───────────────────────────────────────────────────────
    print("\n[1/4] Cargando datos...")
    if len(run_dirs) > 1:
        for d in run_dirs:
            print(f"  - {d}")
    data = load_dataset(run_dirs, charged=args.charged,
                        max_events=args.max_events)
    if len(data["hist"]) < 50:
        sys.exit("ERROR: menos de 50 eventos válidos. Revisa los datos.")

    print(f"  Eventos totales: {len(data['hist'])}")
    print(f"  Centralidades:   "
          + "  ".join(f"{l}:{(data['cent']==i).sum()}"
                      for i, l in enumerate(CENT_LABELS)
                      if (data['cent']==i).sum() > 0))

    # ── 2. Entrenar VAE ───────────────────────────────────────────────────────
    print("\n[2/4] Entrenando VAE...")
    model, history = train_vae(
        data,
        z_dim=args.z_dim,
        hidden_dims=(256, 128),
        beta=args.beta,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        use_bce=args.bce,
    )

    if args.save_model:
        model_path = outdir / f"vae_{tag}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"  Modelo guardado: {model_path}")

    # ── 3. Inferencia ─────────────────────────────────────────────────────────
    print("\n[3/4] Codificando todos los eventos...")
    device = next(model.parameters()).device
    Z, recs = encode_all(model, data, device=device)
    print(f"  Representaciones latentes: {Z.shape}")

    # ── 4. Análisis y gráficas ────────────────────────────────────────────────
    print("\n[4/4] Generando análisis y gráficas...")
    plot_loss_curves(history, tag, outdir)
    plot_latent_scatter(Z, data, tag, outdir)
    plot_correlation_heatmap(Z, data, tag, outdir)
    plot_clustering(Z, data, tag, outdir)
    plot_reconstruction(Z, recs, data, tag, outdir)

    print(f"\nListo. Resultados en: {outdir}")


if __name__ == "__main__":
    main()
