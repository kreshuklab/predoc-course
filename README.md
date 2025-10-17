
# EIPP Theory@EMBL 2025 — Cell Segmentation

*This branch contains the 2025 materials. Previous years are in other branches.*

You will build a **practical cell instance segmentation pipeline** for immunofluorescence images from the COVID assay dataset (Microscopy-based assay for semi-quantitative detection of SARS-CoV-2 specific antibodies in human sera).

## Table of Contents

* [What you will build](#what-you-will-build)
* [Dataset](#dataset)
* [Software setup](#software-setup)
* [Step 1 — Nuclei probabilities with ilastik (Pixel Classification)](#step-1--nuclei-probabilities-with-ilastik-pixel-classification)
* [Step 2 — Foreground & Boundary maps (ilastik Neural Network)](#step-2--foreground--boundary-maps-ilastik-neural-network)
* [Step 3 — Seeded Watershed (instance segmentation)](#step-3--seeded-watershed-instance-segmentation)
* [(Optional) Compare to ground truth](#optional-compare-to-ground-truth)

---

## What you will build

A 3-stage pipeline:

1. **Nuclei** probabilities from DAPI (ilastik **Pixel Classification**).
2. **Cell foreground + boundary** probabilities from serum (ilastik **Neural Network** using a pre-trained model called [powerful-chipmunk](https://bioimage.io/#/artifacts/powerful-chipmunk) from bioimage.io).
3. **Seeded watershed** combining (1) and (2) to produce **cell instances**.

Output: a labeled instance image + quick PNG visualizations.

---

## Dataset

Each HDF5 sample contains:

* `/raw` — shape `(3, 1024, 1024)` with channels: **0: serum**, **1: infection (ignore)**, **2: nuclei (DAPI)**
* `/cells` — instance ground truth `(1024, 1024)`
* `/infected` — per-nucleus infection labels `(1024, 1024)` (not used here)

We also provide **separated channel** HDF5 files in `hdf5/nuclei/` and `hdf5/serum/` with `/raw` `(1, 1024, 1024)` for convenience.

---

## Software setup

* **ilastik**: download latest (GUI app) — no conda needed.
  [https://www.ilastik.org/download.html](https://www.ilastik.org/download.html)
* **Seeded watershed script** (Python, headless):

  * On the EMBL cluster, **activate the shared env**:

    ```bash
    source /g/kreshuk/almpanak/miniforge3/etc/profile.d/conda.sh
    conda activate predoc-challenge
    ```
  * Script you will run: `seeded_watershed_simple.py` (in this repo).
    Dependencies are already in the env (numpy, h5py, scikit-image, matplotlib, scipy).

> If you’re not on the cluster, create an env from `environment.yml` or install the packages listed in the script header.

---

## Step 1 — Nuclei probabilities with ilastik (Pixel Classification)

1. Open **Pixel Classification**.
2. Load **nuclei** images (DAPI). Make sure that you load them in the correct way (cyx - Edit properties in Raw Data tab)
3. Create 2 labels: **Nuclei**, **Background**. Scribble minimally, enable **Live Update**, iterate until the probability map is sensible.
4. **Export** → **HDF5** with:

   * Dataset: **Probabilities**
   * **Two channels** `[Nuclei, Background]` (keep both)

5. Name suggestion: `*_nucProb.h5` (dataset key must be `/exported_data` — ilastik default).

**What we need later:** the file with **2-channel probabilities** for nuclei (we will take **channel 0 = Nuclei**).

---

## Step 2 — Foreground & Boundary maps (ilastik Neural Network)

1. Open **Neural Network Classification (Local)** (requires a recent ilastik). It is already installed in Jupyterhub.
2. Load **serum** channel images as input.
3. In **NN Prediction**, load the pre-trained model from bioimage.io:
   model id “**powerful-chipmunk**” (CovidIF boundary/foreground model).
4. Click **Live Predict**. You should see **two output channels**:

   * channel 0 → **Foreground** (cell interior)
   * channel 1 → **Boundary** (cell borders)
5. **Export** → **HDF5** with both output channels (again under `/exported_data`).
   Name suggestion: `*_nnseg.h5`.

**What we need later:** a single H5 per image with **2 channels** `[Foreground, Boundary]`.

---

## Step 3 — Seeded Watershed (instance segmentation)

Use the two exported H5 files to produce labeled instances.

**Script assumptions (fixed):**

* Nuclei H5 `/exported_data` has **2 channels**: `[Nuclei, Background]` → we take **channel 0**.
* NN H5 `/exported_data` has **2 channels**: `[Foreground, Boundary]` → we take **0** and **1**, respectively.
* Thresholds are fixed in the script: `NUC_THR=0.60`, `FG_THR=0.50`.
  Seeds come from DAPI via distance-transform peaks; watershed elevation is the boundary map.

**Run:**

```bash
python seeded_watershed_simple.py \
  --nuc /path/to/WellXX_..._nucProb.h5 \
  --nn  /path/to/WellXX_..._nnseg.h5 \
  --out ./   # directory or a filename.tif
```

**Outputs:**

* `*_instances.tif` — labeled instances (`uint16`)
* `*_instances_color.png` — random color per cell
* `*_instances_overlay.png` — color overlay on grayscale foreground

You’ll also see concise debug prints: shapes, ranges, seed count, mask coverage, instance count.

---

## (Optional) Compare to ground truth

If you want a number, compare your `*_instances.tif` to `/cells` with **Adapted Rand Error** or **IoU** (scikit-image metrics).
This is optional; the main goal is to understand and execute the pipeline end-to-end.

---
