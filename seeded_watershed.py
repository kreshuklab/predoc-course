import argparse, os
import numpy as np
import h5py
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects
from skimage.measure import label as cc_label
from skimage.segmentation import watershed
from skimage.io import imsave
from skimage.color import label2rgb
from skimage import exposure
import matplotlib.pyplot as plt

# ---- Fixed parameters (tweak here if absolutely necessary) ----
DATASET       = "/exported_data"  # ilastik default dataset name
NUC_THR       = 0.60              # threshold for nuclei seed support
FG_THR        = 0.50              # threshold for foreground mask
MIN_SEED_AREA = 30                # remove tiny seed blobs (pixels)
MIN_DISTANCE  = 6                 # minimum distance between seed peaks (pixels)

def load_h5(path):
    """Load '/exported_data' from an ilastik HDF5 file and squeeze singleton axes."""
    with h5py.File(path, "r") as f:
        return np.squeeze(f[DATASET][()])

def normalize01(x):
    """Normalize an array to [0,1] (safe if constant)."""
    x = x.astype(np.float32)
    m, M = float(x.min()), float(x.max())
    return (x - m) / (M - m + 1e-8)

def pick_channel_2(a, idx):
    """
    Pick one channel from a 2-channel volume.
    Assumes 'a' contains exactly one axis of size 2 (e.g. (H,W,2) or (2,H,W)).
    """
    a = np.squeeze(a)
    ch_ax = [i for i, s in enumerate(a.shape) if s == 2][0]
    sl = [slice(None)] * a.ndim
    sl[ch_ax] = idx
    out = np.squeeze(a[tuple(sl)])
    return normalize01(out)

def seeds_from_nuclei(nuc_prob):
    """
    Build watershed seeds from nuclei probability:
      1) threshold nuclei → seed support
      2) distance transform inside support
      3) peak_local_max for separated seed points
      4) fall back to connected components if no peaks found
    Returns (markers, support_mask).
    """
    support = nuc_prob > NUC_THR
    support = remove_small_objects(support, min_size=MIN_SEED_AREA)
    dist = distance_transform_edt(support)
    coords = peak_local_max(dist, min_distance=MIN_DISTANCE, labels=support)

    markers = np.zeros_like(dist, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    if markers.max() == 0:  # if no peaks at all, use connected components
        markers = cc_label(support).astype(np.int32)

    return markers, support

def save_pngs(out_path, seg, fg_prob):
    """Save two PNGs: a colored instance map and an overlay on grayscale foreground."""
    color = label2rgb(seg, bg_label=0)
    png_color   = os.path.splitext(out_path)[0] + "_color.png"
    png_overlay = os.path.splitext(out_path)[0] + "_overlay.png"

    plt.figure(figsize=(8, 8))
    plt.imshow(color); plt.axis('off'); plt.tight_layout(pad=0)
    plt.savefig(png_color, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()

    bg = exposure.equalize_adapthist(fg_prob)  # contrast-enhanced grayscale background
    overlay = label2rgb(seg, image=bg, alpha=0.4, bg_label=0)
    plt.figure(figsize=(8, 8))
    plt.imshow(overlay); plt.axis('off'); plt.tight_layout(pad=0)
    plt.savefig(png_overlay, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"[OK] wrote {png_color} and {png_overlay}")

def main():
    
    ap = argparse.ArgumentParser(description="Seeded watershed from ilastik HDF5 (nuc + 2ch NN).")
    ap.add_argument("--nuc", required=True, help="HDF5 nuclei probs (/exported_data), 2ch [nuc, bg]")
    ap.add_argument("--nn",  required=True, help="HDF5 NN probs (/exported_data), 2ch [fg, bnd]")
    ap.add_argument("--out", required=True, help="output .tif OR output directory")
    args = ap.parse_args()

    # ---- Load inputs ----
    nuc_raw = load_h5(args.nuc)  # 2ch [nuclei, background]
    nn_raw  = load_h5(args.nn)   # 2ch [foreground, boundary]

    # ---- Select channels and normalize to [0,1] ----
    nuc_prob = pick_channel_2(nuc_raw, idx=0)  # nuclei channel
    fg_prob  = pick_channel_2(nn_raw,  idx=0)  # foreground channel
    bnd_prob = pick_channel_2(nn_raw,  idx=1)  # boundary channel

    # ---- Quick debug prints ----
    print(f"nuc_prob shape={nuc_prob.shape} range=[{nuc_prob.min():.3f},{nuc_prob.max():.3f}]")
    print(f"fg_prob  shape={fg_prob.shape}  range=[{fg_prob.min():.3f},{fg_prob.max():.3f}]")
    print(f"bnd_prob shape={bnd_prob.shape} range=[{bnd_prob.min():.3f},{bnd_prob.max():.3f}]")

    # ---- Build seeds and foreground mask ----
    seeds, support = seeds_from_nuclei(nuc_prob)
    print(f"seed_support={support.mean():.3f}  seed_count={int(seeds.max())}")

    fg_mask = fg_prob > FG_THR
    print(f"fg_mask={fg_mask.mean():.3f} (thr={FG_THR})")

    # ---- Seeded watershed: elevation = boundary prob; restrict to foreground mask ----
    seg = watershed(bnd_prob, markers=seeds, mask=fg_mask).astype(np.uint16)
    print(f"instances={int(seg.max())}")

    # ---- Resolve output filename (dir or file) and save ----
    out_path = args.out
    if os.path.isdir(out_path) or out_path.endswith(os.sep):
        os.makedirs(out_path, exist_ok=True)
        stem = os.path.basename(args.nuc).replace(".h5", "").replace(",", "_")
        out_path = os.path.join(out_path, f"{stem}_instances.tif")

    imsave(out_path, seg, check_contrast=False)
    print(f"[OK] wrote {out_path}")

    # ---- Save visualization PNGs ----
    save_pngs(out_path, seg, fg_prob)

if __name__ == "__main__":
    main()
