#!/usr/bin/env python3
"""
vae_moments.py — VAE sobre los tres primeros momentos de P(<pT>).

Input por evento: vector de 3 escalares normalizados
    x = (μ₁, μ₂, μ₃)
donde:
    μ₁ = <pT>_i                          media del evento
    μ₂ = (1/N) Σ (pT_j - μ₁)²           varianza intra-evento
    μ₃ = (1/N) Σ (pT_j - μ₁)³           tercer momento central intra-evento

Estos son los momentos de la distribución de pT de las partículas
DENTRO de cada evento, calculados evento por evento. El VAE aprende
una representación latente de la "estadística" del evento.

Diferencia clave respecto a vae_pt.py:
    vae_pt.py      → input: histograma de pT (60 bins) — forma del espectro
    vae_moments.py → input: (μ₁, μ₂, μ₃)  (3 escalares) — resumen estadístico

Arquitectura VAE (mucho más pequeña, input 3D):
    Encoder: 3 → 64 → 32 → (μ_z, log σ²_z)
    Decoder: z_dim → 32 → 64 → 3
    Salida sin activación (reconstruye escalares reales normalizados)
    Pérdida: MSE + β·D_KL

Análisis del espacio latente:
    · Correlación Pearson con b, centralidad, μ₁, μ₂, μ₃, N
    · K-means + DBSCAN con Silhouette
    · Scatter z₁-z₂ coloreado por centralidad / b / μ₁ / μ₂ / μ₃
    · Reconstrucción: μ₁, μ₂, μ₃ originales vs reconstruidos por centralidad

Uso:
    python3 analysis/vae_moments.py output/Bi_11GeV_MB* --charged
    python3 analysis/vae_moments.py output/Bi_11GeV_MB* --charged \\
        --z-dim 2 --beta 1.0 --epochs 150 --batch 512 \\
        --outdir output/vae_moments_results
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_f14 import iter_all


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
#  1. CARGA DE DATOS — calcula μ₁, μ₂, μ₃ por evento
# ═══════════════════════════════════════════════════════════════════════════════

def compute_moments(pt_vals):
    """
    Calcula los tres primeros momentos centrales del espectro de pT
    de las partículas DE UN EVENTO.

    μ₁ = (1/N) Σ pT_j                    media del evento
    μ₂ = (1/N) Σ (pT_j - μ₁)²           varianza intra-evento  [GeV²/c²]
    μ₃ = (1/N) Σ (pT_j - μ₁)³           tercer momento central [GeV³/c³]

    Requiere N >= 2 para que μ₂ esté definido.
    Devuelve (nan, nan, nan) si N < 2.
    """
    n = len(pt_vals)
    if n < 2:
        return np.nan, np.nan, np.nan
    mu1  = pt_vals.mean()
    diff = pt_vals - mu1
    mu2  = np.mean(diff**2)
    mu3  = np.mean(diff**3)
    return mu1, mu2, mu3


def load_dataset(output_dirs, charged=False, max_events=None):
    """
    Lee eventos y devuelve un dict con:
        moments  : array (N_ev, 3) — [μ₁, μ₂, μ₃] por evento  float32
        b        : array (N_ev,)   — parámetro de impacto [fm]  float32
        cent     : array (N_ev,)   — índice de centralidad 0-9  int32
        n_part   : array (N_ev,)   — multiplicidad filtrada      int32

    Solo se incluyen eventos con N >= 2 partículas filtradas.
    """
    if isinstance(output_dirs, (str, Path)):
        output_dirs = [output_dirs]

    rows, bs, cents, n_parts = [], [], [], []
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
            mu1, mu2, mu3 = compute_moments(pt_vals)

            if np.isfinite(mu1) and np.isfinite(mu2) and np.isfinite(mu3):
                rows.append([mu1, mu2, mu3])
                bs.append(float(ev["b"]))
                cents.append(assign_centrality(ev["b"]))
                n_parts.append(len(pt_vals))

            n_total += 1
            if n_total % 5000 == 0:
                print(f"    {n_total} eventos procesados...", flush=True)
            if max_events is not None and n_total >= max_events:
                break

    moments = np.array(rows,    dtype=np.float32)   # shape (N, 3)
    bs      = np.array(bs,      dtype=np.float32)
    cents   = np.array(cents,   dtype=np.int32)
    n_parts = np.array(n_parts, dtype=np.int32)

    print(f"  Total eventos cargados: {len(moments)}", flush=True)
    return {"moments": moments, "b": bs, "cent": cents, "n_part": n_parts}


# ── Normalización (StandardScaler por columna) ────────────────────────────────

def normalize(data):
    """
    Normaliza cada momento a media 0 y std 1 usando los parámetros del
    conjunto completo. Devuelve (X_norm, scaler).
    El scaler se guarda para poder invertir la normalización en las gráficas.
    """
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(data["moments"]).astype(np.float32)
    return X_norm, scaler


# ═══════════════════════════════════════════════════════════════════════════════
#  2. ARQUITECTURA VAE
# ═══════════════════════════════════════════════════════════════════════════════

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
        layers.append(nn.Linear(in_d, output_dim))   # salida lineal (escalares)
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class VAE(nn.Module):
    def __init__(self, input_dim=3, hidden_dims=(64, 32), z_dim=2):
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


# ═══════════════════════════════════════════════════════════════════════════════
#  3. ENTRENAMIENTO
# ═══════════════════════════════════════════════════════════════════════════════

def train_vae(X_norm, z_dim=2, hidden_dims=(64, 32), beta=1.0,
              epochs=150, batch_size=512, lr=1e-3,
              val_frac=0.15, device=None, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Dispositivo: {device}")

    X       = torch.tensor(X_norm, dtype=torch.float32)
    dataset = TensorDataset(X)
    n_val   = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model = VAE(input_dim=3, hidden_dims=hidden_dims, z_dim=z_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                        eta_min=lr / 20)

    history   = {k: [] for k in
                 ("train_loss","val_loss","train_rec","val_rec","train_kl","val_kl")}
    best_val   = np.inf
    best_state = None

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
            bs_ = xb.size(0)
            tl += loss.item() * bs_
            tr += lrec.item() * bs_
            tk += lkl.item()  * bs_
        sched.step()

        model.eval()
        vl = vr = vk = 0.
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                x_rec, mu, logvar, _ = model(xb)
                loss, lrec, lkl = model.loss(xb, x_rec, mu, logvar, beta)
                bs_ = xb.size(0)
                vl += loss.item() * bs_
                vr += lrec.item() * bs_
                vk += lkl.item()  * bs_

        for key, val in [
            ("train_loss", tl/n_train), ("val_loss", vl/n_val),
            ("train_rec",  tr/n_train), ("val_rec",  vr/n_val),
            ("train_kl",   tk/n_train), ("val_kl",   vk/n_val),
        ]:
            history[key].append(val)

        if vl / n_val < best_val:
            best_val   = vl / n_val
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Época {epoch:4d}/{epochs}  "
                  f"loss={tl/n_train:.6f}  val={vl/n_val:.6f}  "
                  f"rec={tr/n_train:.6f}  kl={tk/n_train:.6f}", flush=True)

    model.load_state_dict(best_state)
    print(f"  Mejor val_loss: {best_val:.6f}")
    return model, history


# ═══════════════════════════════════════════════════════════════════════════════
#  4. INFERENCIA
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def encode_all(model, X_norm, batch_size=2048, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    X      = torch.tensor(X_norm, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)
    zs, recs = [], []
    for (xb,) in loader:
        xb = xb.to(device)
        mu, _ = model.encoder(xb)
        x_rec, _, _, _ = model(xb)
        zs.append(mu.cpu().numpy())
        recs.append(x_rec.cpu().numpy())
    return np.vstack(zs), np.vstack(recs)


# ═══════════════════════════════════════════════════════════════════════════════
#  5. ANÁLISIS
# ═══════════════════════════════════════════════════════════════════════════════

MOM_LABELS = [r"$\mu_1$ [GeV/c]",
              r"$\mu_2$ [GeV$^2$/c$^2$]",
              r"$\mu_3$ [GeV$^3$/c$^3$]"]
MOM_KEYS   = ["mu1", "mu2", "mu3"]


def correlation_analysis(Z, data):
    obs = {
        r"$b$ [fm]":          data["b"],
        "Centralidad":        data["cent"].astype(float),
        r"$\mu_1$":           data["moments"][:, 0],
        r"$\mu_2$":           data["moments"][:, 1],
        r"$\mu_3$":           data["moments"][:, 2],
        r"$N$ partículas":    data["n_part"].astype(float),
    }
    names = list(obs.keys())
    z_dim = Z.shape[1]
    corr  = np.zeros((len(names), z_dim))
    for i, name in enumerate(names):
        for j in range(z_dim):
            mask = np.isfinite(obs[name]) & np.isfinite(Z[:, j])
            if mask.sum() > 5:
                corr[i, j] = np.corrcoef(obs[name][mask], Z[mask, j])[0, 1]
    return corr, names


def clustering_analysis(Z, k_range=(2, 9)):
    Zs = StandardScaler().fit_transform(Z)
    km_scores = {}
    for k in range(*k_range):
        km   = KMeans(n_clusters=k, random_state=42, n_init=10)
        labs = km.fit_predict(Zs)
        if len(np.unique(labs)) > 1:
            km_scores[k] = (silhouette_score(Zs, labs), labs)

    k_best  = max(km_scores, key=lambda k: km_scores[k][0])
    km_labs = km_scores[k_best][1]
    km_sil  = km_scores[k_best][0]

    from sklearn.neighbors import NearestNeighbors
    nn   = NearestNeighbors(n_neighbors=5).fit(Zs)
    d, _ = nn.kneighbors(Zs)
    eps  = max(float(np.percentile(d[:, -1], 5)), 0.3)
    db   = DBSCAN(eps=eps, min_samples=5)
    db_labs = db.fit_predict(Zs)
    n_db    = len(set(db_labs) - {-1})
    db_sil  = silhouette_score(Zs, db_labs) if n_db > 1 else np.nan

    return km_labs, db_labs, k_best, km_sil, db_sil, km_scores


# ═══════════════════════════════════════════════════════════════════════════════
#  6. GRÁFICAS
# ═══════════════════════════════════════════════════════════════════════════════

def _scatter(ax, x, y, c, cmap, norm, s=4, alpha=0.4):
    return ax.scatter(x, y, c=c, cmap=cmap, norm=norm,
                      s=s, alpha=alpha, rasterized=True)


def plot_loss_curves(history, tag, outdir):
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    ep = range(1, len(history["train_loss"]) + 1)
    pairs = [
        ("train_loss", "val_loss", r"Pérdida total $\mathcal{L}$"),
        ("train_rec",  "val_rec",  r"Reconstrucción $\mathcal{L}_{\rm rec}$"),
        ("train_kl",   "val_kl",   r"$\beta \cdot D_{KL}$"),
    ]
    for ax, (tr, vl, title) in zip(axs, pairs):
        ax.plot(ep, history[tr], label="Train", lw=1.5)
        ax.plot(ep, history[vl], label="Val",   lw=1.5, ls="--")
        ax.set_xlabel("Época"); ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8); ax.grid(ls=":", alpha=0.6)
    fig.suptitle(f"Curvas de pérdida VAE (momentos) — {tag}", fontsize=10)
    fig.tight_layout()
    path = outdir / f"loss_curves_{tag}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Guardado: {path}")


def plot_latent_scatter(Z, data, tag, outdir):
    if Z.shape[1] < 2:
        return
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"Espacio latente — {tag}\n"
        r"Input: $(\mu_1,\,\mu_2,\,\mu_3)$ por evento",
        fontsize=11
    )

    # Fila 1 — centralidad, b, μ₁
    ax = axs[0, 0]
    for ci, (label, color) in enumerate(zip(CENT_LABELS, CENT_COLORS)):
        mask = data["cent"] == ci
        if mask.sum():
            ax.scatter(Z[mask, 0], Z[mask, 1], c=color, s=5,
                       alpha=0.5, label=label, rasterized=True)
    ax.set_title("Por centralidad", fontsize=9)
    ax.legend(fontsize=5, ncol=2, markerscale=2)

    for ax, col, cmap, label in [
        (axs[0, 1], data["b"],              "plasma",  r"$b$ [fm]"),
        (axs[0, 2], data["moments"][:, 0],  "viridis", r"$\mu_1$ [GeV/c]"),
        (axs[1, 0], data["moments"][:, 1],  "inferno", r"$\mu_2$ [GeV²/c²]"),
        (axs[1, 1], data["moments"][:, 2],  "cividis", r"$\mu_3$ [GeV³/c³]"),
        (axs[1, 2], data["n_part"].astype(float), "cool", r"$N$ partículas"),
    ]:
        norm = Normalize(vmin=np.percentile(col, 1),
                         vmax=np.percentile(col, 99))
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=col, cmap=cmap, norm=norm,
                        s=5, alpha=0.5, rasterized=True)
        plt.colorbar(sc, ax=ax, label=label, shrink=0.85)

    titles = [r"Por $b$ [fm]",
              r"Por $\mu_1$",
              r"Por $\mu_2$",
              r"Por $\mu_3$",
              r"Por $N$ partículas"]
    for ax, t in zip(list(axs[0, 1:]) + list(axs[1, :]), titles):
        ax.set_title(t, fontsize=9)

    for ax in axs.flat:
        ax.set_xlabel(r"$z_1$", fontsize=8)
        ax.set_ylabel(r"$z_2$", fontsize=8)

    fig.tight_layout()
    path = outdir / f"latent_scatter_{tag}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Guardado: {path}")


def plot_correlation_heatmap(Z, data, tag, outdir):
    corr, names = correlation_analysis(Z, data)
    z_dim = Z.shape[1]
    fig, ax = plt.subplots(figsize=(max(5, z_dim * 1.5 + 2), 5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label=r"$\rho$ Pearson")
    ax.set_xticks(range(z_dim))
    ax.set_xticklabels([f"$z_{{{k+1}}}$" for k in range(z_dim)], fontsize=10)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    for i in range(len(names)):
        for j in range(z_dim):
            v = corr[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=9, color="black" if abs(v) < 0.6 else "white")
    ax.set_title(
        f"Correlaciones espacio latente — {tag}\n"
        r"Input VAE: $(\mu_1,\,\mu_2,\,\mu_3)$",
        fontsize=10
    )
    fig.tight_layout()
    path = outdir / f"correlation_{tag}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Guardado: {path}")


def plot_clustering(Z, data, tag, outdir):
    if Z.shape[1] < 2:
        return
    km_labs, db_labs, k_best, km_sil, db_sil, km_scores = \
        clustering_analysis(Z)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Clustering espacio latente — {tag}", fontsize=11)

    cmap_km = matplotlib.colormaps.get_cmap("tab10").resampled(k_best)
    sc = axs[0].scatter(Z[:, 0], Z[:, 1], c=km_labs, cmap=cmap_km,
                        vmin=-0.5, vmax=k_best - 0.5,
                        s=5, alpha=0.6, rasterized=True)
    plt.colorbar(sc, ax=axs[0], ticks=range(k_best), label="Cluster")
    axs[0].set_title(f"K-means  k={k_best}  Silhouette={km_sil:.3f}", fontsize=9)

    n_db = len(set(db_labs) - {-1})
    cmap_db = matplotlib.colormaps.get_cmap("tab10").resampled(max(n_db, 2))
    sc2 = axs[1].scatter(Z[:, 0], Z[:, 1], c=db_labs, cmap=cmap_db,
                          s=5, alpha=0.6, rasterized=True)
    plt.colorbar(sc2, ax=axs[1], label="Cluster (-1=ruido)")
    sil_str = f"{db_sil:.3f}" if np.isfinite(db_sil) else "N/A"
    axs[1].set_title(f"DBSCAN  n_clusters={n_db}  Silhouette={sil_str}", fontsize=9)

    ks     = sorted(km_scores.keys())
    scores = [km_scores[k][0] for k in ks]
    axs[2].plot(ks, scores, marker="o", color="steelblue", lw=1.8)
    axs[2].axvline(k_best, color="tomato", ls="--", lw=1.2,
                   label=f"k óptimo = {k_best}")
    axs[2].set_xlabel("$k$"); axs[2].set_ylabel("Silhouette")
    axs[2].set_title("Selección de $k$ óptimo", fontsize=9)
    axs[2].legend(fontsize=8); axs[2].grid(ls=":", alpha=0.6)

    for ax in axs[:2]:
        ax.set_xlabel(r"$z_1$", fontsize=9)
        ax.set_ylabel(r"$z_2$", fontsize=9)

    fig.tight_layout()
    path = outdir / f"clustering_{tag}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Guardado: {path}")

    # Tabla por centralidad
    print(f"\n  Distribución K-means por centralidad:")
    print(f"  {'Clase':10s}", end="")
    for k in range(k_best): print(f" | C{k:2d}", end="")
    print()
    print("  " + "-" * (13 + k_best * 6))
    for ci, label in enumerate(CENT_LABELS):
        mask = data["cent"] == ci
        if mask.sum() == 0: continue
        print(f"  {label:10s}", end="")
        for k in range(k_best):
            print(f" | {int((km_labs[mask] == k).sum()):4d}", end="")
        print()
    print()


def plot_reconstruction(recs_norm, data, scaler, tag, outdir):
    """
    Para cada clase de centralidad compara μ₁, μ₂, μ₃ originales vs
    reconstruidos. Muestra scatter original vs reconstruido y distribuciones.
    """
    orig  = data["moments"]                         # escala original
    recon = scaler.inverse_transform(recs_norm)     # volver a escala original

    mom_names  = [r"$\mu_1$ [GeV/c]",
                  r"$\mu_2$ [GeV$^2$/c$^2$]",
                  r"$\mu_3$ [GeV$^3$/c$^3$]"]
    mom_labels = ["mu1", "mu2", "mu3"]

    # ── Figura 1: scatter orig vs rec por momento (todas las clases) ──────────
    fig, axs = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        f"Reconstrucción VAE — {tag}\n"
        r"Original vs Reconstruido (cada punto = un evento)",
        fontsize=10
    )
    for j, (ax, name) in enumerate(zip(axs, mom_names)):
        for ci, (label, color) in enumerate(zip(CENT_LABELS, CENT_COLORS)):
            mask = data["cent"] == ci
            if mask.sum() == 0: continue
            ax.scatter(orig[mask, j], recon[mask, j],
                       c=color, s=3, alpha=0.3, label=label, rasterized=True)
        lims = [min(orig[:, j].min(), recon[:, j].min()),
                max(orig[:, j].max(), recon[:, j].max())]
        ax.plot(lims, lims, "k--", lw=1, label="Ideal")
        ax.set_xlabel(f"Original  {name}", fontsize=9)
        ax.set_ylabel(f"Reconstruido  {name}", fontsize=9)
        ax.set_title(name, fontsize=9)
        if j == 0:
            ax.legend(fontsize=5, ncol=2, markerscale=3)
    fig.tight_layout()
    path = outdir / f"reconstruction_scatter_{tag}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Guardado: {path}")

    # ── Figura 2: distribuciones de μ₁, μ₂, μ₃ por centralidad ──────────────
    fig, axs = plt.subplots(3, 1, figsize=(10, 11), sharex=False)
    fig.suptitle(
        f"Distribuciones de momentos: original (sólido) vs VAE (--) — {tag}",
        fontsize=10
    )
    for j, (ax, name) in enumerate(zip(axs, mom_names)):
        for ci, (label, color) in enumerate(zip(CENT_LABELS, CENT_COLORS)):
            mask = data["cent"] == ci
            if mask.sum() < 5: continue
            vals_o = orig[mask, j]
            vals_r = recon[mask, j]
            lo = np.percentile(vals_o, 0.5)
            hi = np.percentile(vals_o, 99.5)
            bins = np.linspace(lo, hi, 60)
            kw = dict(bins=bins, histtype="step", density=True)
            ax.hist(vals_o, color=color, lw=1.5,   label=label, **kw)
            ax.hist(vals_r, color=color, lw=1.2,
                    ls="--", alpha=0.7, **kw)
        ax.set_xlabel(name, fontsize=9)
        ax.set_ylabel("Densidad", fontsize=9)
        ax.grid(ls=":", alpha=0.5)
        if j == 0:
            ax.legend(fontsize=6, ncol=2, title="Centralidad")
    fig.tight_layout()
    path = outdir / f"reconstruction_dist_{tag}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Guardado: {path}")


def plot_moments_vs_centrality(data, tag, outdir):
    """
    Muestra μ₁, μ₂, μ₃ medios ± std por clase de centralidad,
    como referencia de lo que el VAE está aprendiendo.
    """
    x      = np.arange(len(CENT_LABELS))
    fig, axs = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    fig.suptitle(
        f"Momentos del input VAE — {tag}\n"
        r"$\mu_1$, $\mu_2$, $\mu_3$ promedio ± std por centralidad",
        fontsize=10
    )
    ylabels = [r"$\mu_1$ [GeV/c]",
               r"$\mu_2$ [GeV$^2$/c$^2$]",
               r"$\mu_3$ [GeV$^3$/c$^3$]"]
    for j, (ax, ylabel) in enumerate(zip(axs, ylabels)):
        means, stds = [], []
        for ci in range(len(CENT_LABELS)):
            mask = data["cent"] == ci
            vals = data["moments"][mask, j]
            means.append(vals.mean() if len(vals) > 0 else np.nan)
            stds.append(vals.std()   if len(vals) > 0 else np.nan)
        means, stds = np.array(means), np.array(stds)
        for i in x:
            if np.isfinite(means[i]):
                ax.errorbar(i, means[i], yerr=stds[i],
                            fmt="o", color=CENT_COLORS[i], markersize=6,
                            elinewidth=1.5, capsize=4, label=CENT_LABELS[i])
        ok = [i for i in x if np.isfinite(means[i])]
        ax.plot(ok, [means[i] for i in ok],
                color="gray", lw=0.9, alpha=0.6)
        if j == 2:
            ax.axhline(0, color="gray", ls="--", lw=0.8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y", ls=":", alpha=0.6)
        ax.legend(fontsize=6, ncol=2)
    axs[-1].set_xticks(x)
    axs[-1].set_xticklabels(CENT_LABELS, rotation=30, ha="right", fontsize=8)
    axs[-1].set_xlabel("Clase de centralidad", fontsize=9)
    fig.tight_layout()
    path = outdir / f"input_moments_{tag}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Guardado: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  7. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="VAE sobre (μ₁, μ₂, μ₃) por evento"
    )
    parser.add_argument("run_dirs", nargs="+",
                        help="Directorios con subdirectorios run_NNNN/")
    parser.add_argument("--charged",    action="store_true")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--z-dim",      type=int, default=2)
    parser.add_argument("--beta",       type=float, default=1.0)
    parser.add_argument("--epochs",     type=int, default=150)
    parser.add_argument("--batch",      type=int, default=512)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--outdir",     default=None)
    parser.add_argument("--tag",        default=None)
    parser.add_argument("--save-model", action="store_true")
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
    tag += f"_moments_z{args.z_dim}_b{args.beta}"

    outdir = Path(args.outdir) if args.outdir else run_dirs[0] / "vae_moments_plots"
    outdir.mkdir(parents=True, exist_ok=True)

    particle_type = "cargadas" if args.charged else "neutras"
    print("=" * 60)
    print(f"VAE — input: (μ₁, μ₂, μ₃)  |  partículas: {particle_type}")
    print(f"z_dim={args.z_dim}  β={args.beta}  épocas={args.epochs}  "
          f"batch={args.batch}  lr={args.lr}")
    print("=" * 60)

    # ── 1. Cargar datos ───────────────────────────────────────────────────────
    print("\n[1/5] Cargando datos y calculando momentos...")
    if len(run_dirs) > 1:
        for d in run_dirs: print(f"  - {d}")
    data = load_dataset(run_dirs, charged=args.charged,
                        max_events=args.max_events)
    if len(data["moments"]) < 50:
        sys.exit("ERROR: menos de 50 eventos válidos.")

    print(f"  Eventos: {len(data['moments'])}")
    print("  Centralidades: " + "  ".join(
        f"{l}:{(data['cent']==i).sum()}"
        for i, l in enumerate(CENT_LABELS)
        if (data['cent']==i).sum() > 0))
    print(f"  μ₁: [{data['moments'][:,0].min():.4f}, "
          f"{data['moments'][:,0].max():.4f}] GeV/c")
    print(f"  μ₂: [{data['moments'][:,1].min():.2e}, "
          f"{data['moments'][:,1].max():.2e}] GeV²/c²")
    print(f"  μ₃: [{data['moments'][:,2].min():.2e}, "
          f"{data['moments'][:,2].max():.2e}] GeV³/c³")

    # ── 2. Normalizar ─────────────────────────────────────────────────────────
    print("\n[2/5] Normalizando momentos (StandardScaler)...")
    X_norm, scaler = normalize(data)
    print(f"  Medias del scaler: {scaler.mean_}")
    print(f"  Stds  del scaler: {scaler.scale_}")

    # Gráfica de referencia de los momentos de entrada
    plot_moments_vs_centrality(data, tag, outdir)

    # ── 3. Entrenar VAE ───────────────────────────────────────────────────────
    print("\n[3/5] Entrenando VAE...")
    model, history = train_vae(
        X_norm,
        z_dim=args.z_dim,
        hidden_dims=(64, 32),
        beta=args.beta,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
    )
    if args.save_model:
        mp = outdir / f"vae_{tag}.pt"
        torch.save(model.state_dict(), mp)
        print(f"  Modelo guardado: {mp}")

    # ── 4. Inferencia ─────────────────────────────────────────────────────────
    print("\n[4/5] Codificando eventos...")
    device    = next(model.parameters()).device
    Z, recs_n = encode_all(model, X_norm, device=device)
    print(f"  Representaciones latentes: {Z.shape}")

    # ── 5. Análisis y gráficas ────────────────────────────────────────────────
    print("\n[5/5] Generando análisis y gráficas...")
    plot_loss_curves(history, tag, outdir)
    plot_latent_scatter(Z, data, tag, outdir)
    plot_correlation_heatmap(Z, data, tag, outdir)
    plot_clustering(Z, data, tag, outdir)
    plot_reconstruction(recs_n, data, scaler, tag, outdir)

    print(f"\nListo. Resultados en: {outdir}")


if __name__ == "__main__":
    main()