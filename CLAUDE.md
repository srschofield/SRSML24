# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SRSML24 is a Python package (authored by Steven R. Schofield, UCL) for unsupervised machine learning analysis of Scanning Tunneling Microscopy (STM) images. It loads raw Scienta Omicron Matrix format (`.mtrx`) files, preprocesses them into windowed patches, trains a convolutional autoencoder, and clusters the bottleneck features with K-means to identify surface structure classes.

## Environment Setup

Tested configuration: Python 3.8, macOS 15.0.1 (Apple Silicon M3 Pro).

```bash
# Create and activate a conda environment
conda create -n srsml24 python=3.8
conda activate srsml24

# Install dependencies
pip install -r requirements-macos.txt
```

Key non-standard dependencies:
- `access2thematrix` — reads Scienta Omicron `.mtrx` binary files
- `spiepy` — STM image flattening algorithms
- `tensorflow-macos` + `tensorflow-metal` — GPU-accelerated TF on Apple Silicon

For HPC (Linux) use, replace `tensorflow-macos`/`tensorflow-metal` with standard `tensorflow`.

## Running the Code

The package is designed to be used interactively via Jupyter notebooks. Start a notebook server with:

```bash
bash scripts/jupyter_server/start_jn.sh
```

For HPC batch jobs:
```bash
# Process raw MTRX files into windowed numpy arrays
bash scripts/cluster_jobs/mtrx_process.sh

# Run K-means clustering on encoded features
bash scripts/cluster_jobs/cluster.sh
```

For large-scale model training:
```bash
bash scripts/jupyter_server/BIG_model.sh
```

## Architecture

### Data Pipeline (end-to-end)

```
.mtrx files  →  data_prep.process_mtrx_files()  →  .npy windows + .jpg previews
     ↓
model.create_tf_dataset() / create_tf_dataset_batched()
     ↓
model.build_autoencoder()  →  training  →  .keras model saved to trained_models/
     ↓
model.encode() / bottleneck features  →  KMeans clustering  →  cluster label maps
```

### Module Responsibilities

**`data_prep.py`** — All raw data I/O and preprocessing.
- `process_mtrx_files(mtrx_paths, save_data_path, **kwargs)` is the main entry point. It loads each `.mtrx` file, extracts all four scan directions (FU/FD/BU/BD), applies image flattening, resamples to a uniform pixel density, segments into overlapping windows, and saves them as `.npy` arrays (and optionally `.jpg` images with embedded metadata).
- Three flattening methods are supported via `flatten_method` kwarg: `'none'`, `'iterate_mask'` (default), `'poly_xy'`.
- Windows are saved into `<save_data_path>/windows/` as individual `.npy` files, or as batched `.npy` arrays. JPGs go to `<save_data_path>/jpg/`.
- A global `mtrx = access2thematrix.MtrxData()` instance is created at module import time and reused throughout.
- `data_analysis.py` currently contains `radial_profile()` for computing and plotting radially averaged intensity profiles from STM images.

**`model.py`** — ML model definition, training, and clustering.
- `build_autoencoder(window_size)` builds a UNET-style convolutional autoencoder with skip connections: encoder (Conv2D 32→64→128→256 with MaxPooling) and symmetric decoder (UpSampling + concatenation of skip connections). Output activation is sigmoid, suitable for normalised [0,1] input.
- `create_tf_dataset()` / `create_tf_dataset_batched()` wrap `.npy` files into `tf.data.Dataset` pipelines with shuffle/batch/prefetch.
- Models are saved/loaded as `.keras` format via `utils.save_model()` / `utils.load_model()`.
- K-means and MiniBatchKMeans from scikit-learn are used for clustering bottleneck feature vectors. Cluster models are saved with `joblib`.
- `print_system_info()` checks TF version and GPU availability (Metal on macOS, CUDA on Linux).

**`utils.py`** — General-purpose helpers.
- Image I/O: `load_jpg_as_norm_numpy()` loads a JPEG as a normalised float32 grayscale array.
- Colormap tools: `display_colormap_and_rgb()`, `plot_colormaps_image_grid()` for visual QA.
- `pad_cluster_image()` aligns a cluster label map back onto the original image coordinate grid (accounting for the half-window border lost during windowing).
- `summarize_parameters()` uses introspection to generate a CSV of named notebook variables and their values.

### Data Layout

```
data/sample_data/
    mtrx/           raw .mtrx input files (Scienta Omicron format)
    processed_images/
        jpg/        QA images with metadata overlay (bias, current, scan direction)
        windows/    windowed .npy arrays for ML training (shape: window_size × window_size)
        output/     results (cluster maps, etc.)
trained_models/
    autoencoder_models/   .keras model files
    cluster_models/       joblib-serialised KMeans models
```

### Key Parameters in `process_mtrx_files`

| Parameter | Default | Meaning |
|---|---|---|
| `flatten_method` | `'iterate_mask'` | Background flattening algorithm |
| `pixel_density` | `15.0` | Target pixels per nm after resampling |
| `window_size` | `32` | Patch size in pixels |
| `window_pitch` | `16` | Stride between patches (50% overlap) |
| `data_scaling` | `2e9` | Z-scale factor applied before normalisation |
| `pixel_limit` | `20000` | Max pixels per dimension before downsampling |

### Notebooks

Development work lives in `notebooks/steven/` (Steven) and `notebooks/adam/` (Adam). The canonical workflow is:
1. `data_process_to_jpg.ipynb` — test data loading and JPG generation
2. `load_mtrx_image.ipynb` — explore individual MTRX files
3. `autoencoder_model.ipynb` / `BIG_model.py` — train the autoencoder
4. `cluster.ipynb` / `cluster-SiC.ipynb` — cluster and visualise results
5. `radial_averaging.ipynb` — post-processing of cluster results

## Research Context

The STM data (`data/sample_data/mtrx/`) spans 2011–2024. Scan directions FU/FD/BU/BD correspond to Forward-Up, Forward-Down, Backward-Up, Backward-Down raster directions. Bias and tunnelling current metadata are extracted from the `.mtrx` files and embedded in output filenames and JPG overlays. The package is designed to be extensible to other SPM file formats and clustering approaches.
