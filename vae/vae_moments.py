#!/usr/bin/env python3
"""
vae_moments.py — v4: VAE sobre (mu1, mu2, mu3) por evento, un VAE
                      INDEPENDIENTE por cada clase de centralidad.

Cambios respecto a v3 (retroalimentación de la directora):
    - Se elimina el estudio de bootstrap para n_min. Se vuelve al valor
      fijo n_min = 10 (configurable con --n-min).
    - El análisis completo (carga, escalamiento, entrenamiento del VAE,
      clustering, gráficas) se hace POR SEPARADO para cada una de las
      10 clases de centralidad, en vez de combinar todo en un solo
      dataset coloreado por clase.

Por qué esto resuelve, de paso, el problema del escalador global
(retroalimentación anterior, punto 3):
    Como cada clase entrena su propio StandardScaler sobre SUS PROPIOS
    eventos únicamente, el escalador es "por clase" por construcción —
    ya no hace falta mantener un diccionario de 10 escaladores ni
    preocuparse por heterocedasticidad entre clases, porque nunca se
    mezclan datos de dos clases distintas en el mismo ajuste.
    Por la misma razón, tampoco es necesario restar la media de la
    clase (delta_mu_k) antes de escalar: cada VAE ve directamente los
    momentos ABSOLUTOS de su propia clase, y el StandardScaler ya
    centra y normaliza esa clase específica.

Input por evento i (dimensión 3), calculado sobre las partículas del
PROPIO evento (N_i >= n_min):
    mu1_i = (1/N_i) * sum_j pT_j                     (media)
    mu2_i = (1/N_i) * sum_j (pT_j - mu1_i)^2          (varianza)
    mu3_i = (1/N_i) * sum_j (pT_j - mu1_i)^3          (asimetría/sesgo)

Nota metodológica (ya discutida): mu1_i coincide exactamente con
<pT>_i tal como lo define el protocolo. mu2_i y mu3_i, en cambio,
describen la forma del espectro DENTRO de ese evento (no la varianza
del ensemble P(<pT>) entre eventos) — son observables físicos válidos
por derecho propio, pero conceptualmente distintos de sigma^2_pT y del
sesgo de P(<pT>) del protocolo. Se documenta aquí para que quede
explícito en la tesis qué representa exactamente cada componente.

Uso:
    python3 analysis/vae_moments.py output/Bi_11GeV_MB* --charged \\
        --n-min 10 --z-dim 2 --beta 1.0 --epochs 200
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

def assign_centrality(b):
    for b_lo, b_hi, label, idx in B_CUTS:
        if b_lo <= b < b_hi:
            return idx
    return 9


# ═══════════════════════════════════════════════════════════════════════════
#  1. CARGA — momentos por evento, bucketizados por clase, con corte n_min
# ═══════════════════════════════════════════════════════════════════════════

def load_moments_by_class(output_dirs, charged=False, n_min=10,
                           max_events=None):
    """
    Lee eventos, calcula (mu1, mu2, mu3) por evento a partir de SUS
    PROPIAS partículas, aplica el corte N_i >= n_min, y agrupa por
    clase de centralidad.

    Retorna: data[ci] = dict con arrays mu1, mu2, mu3, b, n_part
    """
    if isinstance(output_dirs, (str, Path)):
        output_dirs = [output_dirs]

    buckets = {ci: {"mu1": [], "mu2": [], "mu3": [], "b": [], "n": []}
               for ci in range(len(CENT_LABELS))}
    n_total, n_kept = 0, 0

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

            pt = ev["pT"][mask]
            n_total += 1
            N = len(pt)

            if N >= n_min and np.all(np.isfinite(pt)):
                mu1 = pt.mean()
                d   = pt - mu1
                mu2 = np.mean(d**2)
                mu3 = np.mean(d**3)
                ci  = assign_centrality(float(ev["b"]))
                buckets[ci]["mu1"].append(mu1)
                buckets[ci]["mu2"].append(mu2)
                buckets[ci]["mu3"].append(mu3)
                buckets[ci]["b"].append(float(ev["b"]))
                buckets[ci]["n"].append(N)
                n_kept += 1

            if n_total % 10000 == 0:
                print(f"    {n_total} eventos procesados...", flush=True)
            if max_events is not None and n_total >= max_events:
                break

    data = {}
    for ci, label in enumerate(CENT_LABELS):
        b = buckets[ci]
        data[ci] = {k: np.array(v) for k, v in b.items()}
        print(f"  {label}: {len(b['mu1'])} eventos (N_i >= {n_min})")

    print(f"  Total leídos: {n_total}   Total conservados (N>={n_min}): {n_kept}")
    return data


# ═══════════════════════════════════════════════════════════════════════════
#  2. ARQUITECTURA VAE (salida lineal: se regresan escalares, no histograma)
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
        return self.net(z)   # lineal: regresión de escalares (mu1,mu2,mu3)


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
              epochs=200, batch_size=256, lr=1e-3, val_frac=0.15,
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
#  3. CLUSTERING
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

    return dict(km_labs=km_labs, k_best=k_best, km_sil=km_sil, k_scores=km_scores,
                db_labs=db_labs, n_db=n_db, db_sil=db_sil)


# ═══════════════════════════════════════════════════════════════════════════
#  4. GRÁFICAS (todas por clase de centralidad individual)
# ═══════════════════════════════════════════════════════════════════════════

def plot_input_distributions(mu1, mu2, mu3, tag, outdir):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, arr, name, unit in zip(
            axs, [mu1, mu2, mu3], [r"$\mu_1$", r"$\mu_2$", r"$\mu_3$"],
            ["GeV/c", r"GeV$^2$/c$^2$", r"GeV$^3$/c$^3$"]):
        ax.hist(arr, bins=40, color="#1f77b4", alpha=0.75)
        ax.axvline(arr.mean(), color="crimson", ls="--", lw=1.2,
                    label=f"media={arr.mean():.3f}")
        ax.set_xlabel(f"{name} [{unit}]"); ax.set_ylabel("Eventos")
        ax.legend(fontsize=7); ax.grid(ls=":", alpha=0.4)
    fig.suptitle(f"Distribución de momentos de entrada — {tag}", fontsize=10)
    fig.tight_layout()
    path = outdir / f"input_moments_{tag}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"    Guardado: {path}")


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
    path = outdir / f"loss_curves_{tag}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"    Guardado: {path}")


def plot_latent_scatter(Z, data_ci, tag, outdir):
    z1 = Z[:, 0]
    z2 = Z[:, 1] if Z.shape[1] > 1 else np.zeros_like(z1)
    fields = [("b",    r"$b$ [fm]",              "plasma"),
              ("mu1",  r"$\mu_1$ [GeV/c]",        "viridis"),
              ("mu2",  r"$\mu_2$ [GeV$^2$/c$^2$]","cividis"),
              ("mu3",  r"$\mu_3$ [GeV$^3$/c$^3$]","magma"),
              ("n",    r"$N$ partículas",          "cool")]
    fig, axs = plt.subplots(1, 5, figsize=(21, 4))
    for ax, (key, label, cmap) in zip(axs, fields):
        sc = ax.scatter(z1, z2, c=data_ci[key], cmap=cmap, s=8, alpha=0.6,
                        rasterized=True)
        plt.colorbar(sc, ax=ax, label=label, fraction=0.046)
        ax.set_title(f"Por {label}", fontsize=9)
        ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$")
        ax.grid(ls=":", alpha=0.4)
    fig.suptitle(f"Espacio latente — {tag}", fontsize=10)
    fig.tight_layout()
    path = outdir / f"latent_scatter_{tag}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"    Guardado: {path}")


def plot_correlation(Z, data_ci, tag, outdir):
    obs_keys  = ["b", "mu1", "mu2", "mu3", "n"]
    obs_names = [r"$b$ [fm]", r"$\mu_1$", r"$\mu_2$", r"$\mu_3$", r"$N$ partículas"]
    z_dim = Z.shape[1]
    corr = np.zeros((len(obs_keys), z_dim))
    for i, key in enumerate(obs_keys):
        for j in range(z_dim):
            corr[i, j] = np.corrcoef(Z[:, j], data_ci[key])[0, 1]

    fig, ax = plt.subplots(figsize=(3 + z_dim, 4.2))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(z_dim)); ax.set_xticklabels([f"$z_{j+1}$" for j in range(z_dim)])
    ax.set_yticks(range(len(obs_keys))); ax.set_yticklabels(obs_names)
    for i in range(len(obs_keys)):
        for j in range(z_dim):
            ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, label=r"$\rho$ Pearson")
    ax.set_title(f"Correlaciones — {tag}", fontsize=9)
    fig.tight_layout()
    path = outdir / f"correlation_{tag}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"    Guardado: {path}")


def plot_clustering(Z, cl, tag, outdir):
    if cl is None:
        return
    z1 = Z[:, 0]; z2 = Z[:, 1] if Z.shape[1] > 1 else np.zeros_like(z1)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))

    sc0 = axs[0].scatter(z1, z2, c=cl["km_labs"], cmap="tab10", s=8, alpha=0.6,
                          rasterized=True)
    axs[0].set_title(f"K-means k={cl['k_best']}  Silhouette={cl['km_sil']:.3f}",
                      fontsize=9)

    sc1 = axs[1].scatter(z1, z2, c=cl["db_labs"], cmap="tab10", s=8, alpha=0.6,
                          rasterized=True)
    axs[1].set_title(f"DBSCAN n={cl['n_db']}  Silhouette={cl['db_sil']:.3f}",
                      fontsize=9)

    ks = sorted(cl["k_scores"].keys())
    sils = [cl["k_scores"][k][0] for k in ks]
    axs[2].plot(ks, sils, "o-", color="#1f77b4")
    axs[2].axvline(cl["k_best"], color="crimson", ls="--")
    axs[2].set_xlabel("$k$"); axs[2].set_ylabel("Silhouette")
    axs[2].set_title("Selección de $k$ óptimo", fontsize=9)
    axs[2].grid(ls=":", alpha=0.5)

    for ax in axs[:2]:
        ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$"); ax.grid(ls=":", alpha=0.4)
    fig.suptitle(f"Clustering espacio latente — {tag}", fontsize=10)
    fig.tight_layout()
    path = outdir / f"clustering_{tag}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"    Guardado: {path}")


def plot_reconstruction(X_raw, recs_raw, tag, outdir):
    names = [r"$\mu_1$ [GeV/c]", r"$\mu_2$ [GeV$^2$/c$^2$]", r"$\mu_3$ [GeV$^3$/c$^3$]"]

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    for k, ax in enumerate(axs):
        ax.scatter(X_raw[:, k], recs_raw[:, k], s=6, alpha=0.35,
                    color="#1f77b4", rasterized=True)
        lo = min(X_raw[:, k].min(), recs_raw[:, k].min())
        hi = max(X_raw[:, k].max(), recs_raw[:, k].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="Ideal")
        ax.set_xlabel(f"Original {names[k]}"); ax.set_ylabel(f"Reconstruido {names[k]}")
        ax.legend(fontsize=7); ax.grid(ls=":", alpha=0.4)
    fig.suptitle(f"Reconstrucción VAE (dispersión) — {tag}", fontsize=10)
    fig.tight_layout()
    path = outdir / f"reconstruction_scatter_{tag}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"    Guardado: {path}")

    fig, axs = plt.subplots(3, 1, figsize=(9, 9))
    for k, ax in enumerate(axs):
        ax.hist(X_raw[:, k],   bins=40, histtype="step", color="black",
                lw=1.4, density=True, label="Original")
        ax.hist(recs_raw[:, k],bins=40, histtype="step", color="crimson",
                lw=1.4, ls="--", density=True, label="VAE")
        ax.set_xlabel(names[k]); ax.set_ylabel("Densidad")
        ax.legend(fontsize=8); ax.grid(ls=":", alpha=0.4)
    fig.suptitle(f"Reconstrucción VAE (distribuciones) — {tag}", fontsize=10)
    fig.tight_layout()
    path = outdir / f"reconstruction_dist_{tag}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"    Guardado: {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  5. PROCESAMIENTO POR CLASE
# ═══════════════════════════════════════════════════════════════════════════

def process_class(ci, data_ci, args, tag_base, outdir):
    label = CENT_LABELS[ci]
    tag   = f"{tag_base}_{label.replace('%','pct')}"
    n_ev  = len(data_ci["mu1"])
    print(f"\n{'='*65}\nClase de centralidad: {label}  ({n_ev} eventos, N_i>={args.n_min})\n{'='*65}")

    if n_ev < 50:
        print("  Muy pocos eventos en esta clase; se omite.")
        return None

    X_raw = np.column_stack([data_ci["mu1"], data_ci["mu2"], data_ci["mu3"]])
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw).astype(np.float32)
    print(f"  mu1: media={data_ci['mu1'].mean():.4f}  std={data_ci['mu1'].std():.4f} GeV/c")
    print(f"  mu2: media={data_ci['mu2'].mean():.4e}  std={data_ci['mu2'].std():.4e} GeV^2/c^2")
    print(f"  mu3: media={data_ci['mu3'].mean():.4e}  std={data_ci['mu3'].std():.4e} GeV^3/c^3")

    plot_input_distributions(data_ci["mu1"], data_ci["mu2"], data_ci["mu3"],
                              tag, outdir)

    print(f"  Entrenando VAE (input_dim=3, z_dim={args.z_dim}, beta={args.beta})...")
    model, history, best_val = train_vae(
        X, input_dim=3, z_dim=args.z_dim, hidden_dims=(16, 8),
        beta=args.beta, epochs=args.epochs, batch_size=args.batch,
        lr=args.lr, seed=args.seed)
    print(f"    Mejor val_loss: {best_val:.6f}")

    device = next(model.parameters()).device
    Z, recs_scaled = encode_all(model, X, device=device)
    recs_raw = scaler.inverse_transform(recs_scaled)

    cl = clustering_analysis(Z)
    if cl is not None:
        print(f"  K-means: k={cl['k_best']}  Silhouette={cl['km_sil']:.3f}  |  "
              f"DBSCAN: {cl['n_db']} clusters  Silhouette={cl['db_sil']:.3f}")

    plot_loss_curves(history, tag, outdir)
    plot_latent_scatter(Z, data_ci, tag, outdir)
    plot_correlation(Z, data_ci, tag, outdir)
    plot_clustering(Z, cl, tag, outdir)
    plot_reconstruction(X_raw, recs_raw, tag, outdir)

    return {
        "label": label, "n_events": n_ev, "best_val": best_val,
        "mu1_mean": data_ci["mu1"].mean(), "mu1_std": data_ci["mu1"].std(),
        "km_sil": cl["km_sil"] if cl else np.nan,
        "db_sil": cl["db_sil"] if cl else np.nan,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  6. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="VAE sobre (mu1,mu2,mu3) por evento — un modelo "
                     "independiente por clase de centralidad."
    )
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--charged",    action="store_true")
    parser.add_argument("--n-min",      type=int, default=10,
                        help="Multiplicidad mínima por evento (default 10).")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--z-dim",      type=int, default=2)
    parser.add_argument("--beta",       type=float, default=1.0)
    parser.add_argument("--epochs",     type=int, default=200)
    parser.add_argument("--batch",      type=int, default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--outdir",     default=None)
    parser.add_argument("--tag",        default=None)
    args = parser.parse_args()

    run_dirs = [Path(d) for d in args.run_dirs]
    for d in run_dirs:
        if not d.exists():
            sys.exit(f"ERROR: no existe {d}")

    tag_base = args.tag or (run_dirs[0].name if len(run_dirs) == 1
                             else f"{run_dirs[0].name}_combined")
    tag_base += f"_moments_z{args.z_dim}_b{args.beta}_nmin{args.n_min}"

    outdir = Path(args.outdir) if args.outdir else run_dirs[0] / "vae_moments_plots"
    outdir.mkdir(parents=True, exist_ok=True)

    particle_type = "cargadas" if args.charged else "neutras"
    print("=" * 65)
    print(f"VAE por clase — input: (mu1, mu2, mu3)  |  partículas: {particle_type}")
    print(f"z_dim={args.z_dim}  beta={args.beta}  epochs={args.epochs}  "
          f"n_min={args.n_min}")
    print("=" * 65)

    print("\n[1] Cargando datos y calculando momentos por evento...")
    data = load_moments_by_class(run_dirs, charged=args.charged,
                                  n_min=args.n_min, max_events=args.max_events)

    print("\n[2] Entrenando un VAE independiente por clase de centralidad...")
    summary = []
    for ci in range(len(CENT_LABELS)):
        res = process_class(ci, data[ci], args, tag_base, outdir)
        if res is not None:
            summary.append(res)

    print(f"\n{'='*65}\nResumen por clase de centralidad\n{'='*65}")
    hdr = f"{'Clase':<10}{'N_ev':>8}{'val_loss':>11}{'<mu1>':>9}{'std_mu1':>10}{'KM_sil':>8}{'DB_sil':>8}"
    print(hdr)
    for r in summary:
        print(f"{r['label']:<10}{r['n_events']:>8}{r['best_val']:>11.5f}"
              f"{r['mu1_mean']:>9.4f}{r['mu1_std']:>10.4f}"
              f"{r['km_sil']:>8.3f}{r['db_sil']:>8.3f}")

    print(f"\nListo. Resultados en: {outdir}")


if __name__ == "__main__":
    main()