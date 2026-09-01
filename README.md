# UrQMD → VAE: fluctuaciones de p_T en colisiones Bi+Bi a 11 GeV

Pipeline de análisis para estudiar fluctuaciones evento-por-evento de
`p_T`, `eta` y `phi` en colisiones Bi+Bi simuladas con **UrQMD** a
`sqrt(s_NN) = 11 GeV` (región de interés del punto crítico, CEP, en el
diagrama de fases de la QCD), y su representación en un espacio latente
mediante un **Autoencoder Variacional (VAE)**.

Este repositorio contiene únicamente el código de análisis en Python.
**No incluye UrQMD ni los datos simulados** (ver sección
[Datos de entrada](#datos-de-entrada)).

## Estructura del repositorio

```
.
├── analysis/                  # Lectura de datos y observables clásicos
│   ├── read_f14.py            # Parser del formato .f14 de UrQMD (base de todo lo demás)
│   ├── build_dataset.py       # Construye un dataset HDF5 (histogramas por evento) para ML
│   ├── p_o.py                 # Primera versión de histogramas pT/eta/phi (1 carpeta, 1 especie)
│   ├── plot_observables.py    # Versión final: multi-carpeta, multi-especie, participantes/espectadoras
│   ├── fluc_observables.py    # Fluctuaciones de multiplicidad y pT por centralidad
│   ├── pt_moments.py          # Momentos (mu1, mu2, mu3, skewness) de P(<pT>) por centralidad
│   ├── pt_correlator.py       # Correlador de 2 partículas <dp_t,i dp_t,j>, primarias vs secundarias
│   └── pt_correlator_norm.py  # Versión normalizada del correlador (Fig. 4.12)
├── vae/                       # Autoencoders Variacionales (distintas representaciones de entrada)
│   ├── vae_pt.py               # Input: histograma completo de pT (60 bins)
│   ├── vae_pt_mean.py          # Input: histograma bootstrap de <pT> por evento
│   ├── vae_moments.py          # Input: (mu1, mu2, mu3) intra-evento
│   ├── vae_3m.py               # Variante de vae_moments.py (ver nota de nombres más abajo)
│   └── vae_moments_V3.py       # vae_moments.py con correcciones metodológicas (v3)
├── docs/
│   └── apartado_urqmd_vae.tex  # Sección de tesis que documenta todo el pipeline
├── requirements.txt
└── .gitignore
```

> **Nota sobre nombres de archivo:** `vae_3m.py`, `vae_moments.py` y
> `vae_moments_V3.py` son iteraciones sucesivas del mismo experimento
> (VAE sobre momentos estadísticos), y sus docstrings internos no siempre
> coinciden con el nombre del archivo. Revisa el encabezado de cada script
> antes de usarlo — se documenta la cronología completa en
> `docs/apartado_urqmd_vae.tex`.

## Datos de entrada

Este pipeline consume archivos `.f14`, el formato de salida estándar de
**UrQMD** (Ultra-relativistic Quantum Molecular Dynamics). UrQMD **no**
se distribuye en este repositorio: se obtiene por separado desde
<https://urqmd.org> (requiere registro académico) y se compila en la
máquina de destino.

Todos los scripts esperan un directorio con subcarpetas `run_NNNN/`, cada
una con un archivo `urqmd.f14` dentro:

```
output/Bi_11GeV_MB/
├── run_0001/urqmd.f14
├── run_0002/urqmd.f14
└── ...
```

Para reproducir el análisis de la tesis:

1. Compilar UrQMD (energía `sqrt(s_NN) = 11 GeV`, sistema Bi+Bi, modo
   *minimum bias*).
2. Generar `N` eventos, **cada uno con una semilla distinta**, guardando
   cada corrida en su propia carpeta `run_NNNN/urqmd.f14` dentro de un
   directorio `output/<tag>/`. (En el trabajo original se generaron
   100 000 eventos, repartidos en varios lotes por límites de cómputo —
   ver `docs/apartado_urqmd_vae.tex`, todos los scripts aceptan varias
   carpetas a la vez.)
3. Continuar con la sección [Uso](#uso) de este README.

## Instalación

Requiere Python 3.10+.

```bash
git clone <URL-de-tu-repo>
cd urqmd-vae-pipeline
python3 -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`torch` se instala en su versión CPU por defecto vía `requirements.txt`.
Si se dispone de GPU, instalar antes la variante CUDA correspondiente
siguiendo <https://pytorch.org/get-started/locally/> y luego el resto de
`requirements.txt`.

## Uso

Todos los comandos se ejecutan desde la raíz del repositorio. Sustituye
`output/Bi_11GeV_MB` por la ruta real a tus datos `.f14`.

### 1. Observables clásicos

```bash
# Espectros pT/eta/phi por especie, participantes vs espectadoras
python3 analysis/plot_observables.py output/Bi_11GeV_MB --charged-only
python3 analysis/plot_observables.py output/Bi_11GeV_MB* --outdir output/plots_combinados

# Fluctuaciones de multiplicidad y pT por centralidad
python3 analysis/fluc_observables.py output/Bi_11GeV_MB --charged

# Momentos de P(<pT>) y comparación con la distribución Gamma
python3 analysis/pt_moments.py output/Bi_11GeV_MB* --charged

# Correlador de dos partículas (primarias vs secundarias)
python3 analysis/pt_correlator.py output/Bi_11GeV_MB* --charged --n-min 10
python3 analysis/pt_correlator_norm.py output/Bi_11GeV_MB* --charged --n-min 10
```

### 2. Dataset para el VAE

```bash
python3 analysis/build_dataset.py output/Bi_11GeV_MB features.h5 --select participants --charged-only
```

### 3. Entrenamiento de los VAE

```bash
# Histograma completo de pT
python3 vae/vae_pt.py output/Bi_11GeV_MB* --charged --z-dim 4 --beta 1.0 --epochs 100

# Histograma bootstrap de <pT>
python3 vae/vae_pt_mean.py output/Bi_11GeV_MB* --charged --n-boot 200 --d-bins 25 --z-dim 2

# Momentos intra-evento (mu1, mu2, mu3)
python3 vae/vae_moments_V3.py output/Bi_11GeV_MB* --charged --z-dim 2 --beta 1.0 --epochs 200
```

Cada script imprime en consola las rutas de las gráficas y, en el caso
de los VAE, las métricas de reconstrucción y clustering; añade `--help`
a cualquier script para ver todas sus opciones.

## Documentación

`docs/apartado_urqmd_vae.tex` contiene la descripción completa del
pipeline (generación de eventos, formato `.f14`, cada script y su rol
físico, y la evolución metodológica de los VAE), lista para insertarse
en el documento de tesis en LaTeX.

## Licencia

Añade aquí la licencia que prefieras (por ejemplo, MIT) antes de hacer
público el repositorio.
