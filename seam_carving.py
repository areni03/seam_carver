"""
Seam Carving for Content-Aware Image Resizing (Avidan & Shamir, 2007)
Core algorithm implemented from scratch with NumPy.
OpenCV is used only for image I/O and for the ordinary resize baseline.

Usage:
    python seam_carving.py isro_satellite_image.jpg --reduce 0.2
"""
import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------- 1. Energy
def compute_energy(img):
    """Energy = |dI/dx| + |dI/dy| on the grayscale image (gradient magnitude)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gy, gx = np.gradient(gray)
    return np.abs(gx) + np.abs(gy)


# ---------------------------------------------------------------- 2. Seam (DP)
def find_vertical_seam(energy):
    """
    Dynamic programming:
        M(i,j) = E(i,j) + min(M(i-1,j-1), M(i-1,j), M(i-1,j+1))
    Returns an array `seam` of length H where seam[i] is the column removed in row i.
    """
    h, w = energy.shape
    M = energy.copy()
    backtrack = np.zeros((h, w), dtype=np.int64)

    for i in range(1, h):
        prev = M[i - 1]
        left = np.r_[np.inf, prev[:-1]]    # above-left
        up = prev                          # directly above
        right = np.r_[prev[1:], np.inf]    # above-right
        choices = np.stack([left, up, right])          # shape (3, w)
        idx = np.argmin(choices, axis=0)               # 0,1,2 -> offset -1,0,+1
        backtrack[i] = idx - 1
        M[i] += choices[idx, np.arange(w)]

    # Start from the cheapest pixel in the last row and trace back up
    seam = np.zeros(h, dtype=np.int64)
    seam[-1] = np.argmin(M[-1])
    for i in range(h - 1, 0, -1):
        seam[i - 1] = seam[i] + backtrack[i, seam[i]]
    return seam


# ---------------------------------------------------------------- 3. Removal
def remove_vertical_seam(img, seam):
    h, w = img.shape[:2]
    mask = np.ones((h, w), dtype=bool)
    mask[np.arange(h), seam] = False
    return img[mask].reshape(h, w - 1, img.shape[2])


# ---------------------------------------------------------------- 4. Loop
def carve_width(img, n_seams, return_seams=False):
    """Remove n_seams vertical seams, recomputing energy each time."""
    out = img.copy()
    # Track original column indices so seams can be drawn on the original image
    cols = np.tile(np.arange(img.shape[1]), (img.shape[0], 1))
    removed = np.zeros(img.shape[:2], dtype=bool)

    for k in range(n_seams):
        energy = compute_energy(out)
        seam = find_vertical_seam(energy)
        h = out.shape[0]
        removed[np.arange(h), cols[np.arange(h), seam]] = True
        mask = np.ones(cols.shape, dtype=bool)
        mask[np.arange(h), seam] = False
        cols = cols[mask].reshape(h, -1)
        out = remove_vertical_seam(out, seam)
        if (k + 1) % 50 == 0:
            print(f"  removed {k + 1}/{n_seams} seams")
    return (out, removed) if return_seams else out


# ---------------------------------------------------------------- 5. Baselines
def uniform_resize(img, new_w):
    return cv2.resize(img, (new_w, img.shape[0]), interpolation=cv2.INTER_AREA)


def center_crop(img, new_w):
    start = (img.shape[1] - new_w) // 2
    return img[:, start:start + new_w]


# ---------------------------------------------------------------- 6. Visuals
def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_overview(img, energy, seam_mask, carved, path):
    marked = img.copy()
    marked[seam_mask] = (0, 0, 255)  # red in BGR
    fig, ax = plt.subplots(1, 4, figsize=(18, 8))
    ax[0].imshow(to_rgb(img));        ax[0].set_title("Original")
    ax[1].imshow(energy, cmap="gray"); ax[1].set_title("Energy map")
    ax[2].imshow(to_rgb(marked));     ax[2].set_title("Removed seams (red)")
    ax[3].imshow(to_rgb(carved));     ax[3].set_title(f"Seam carved ({carved.shape[1]} px wide)")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def save_comparison(carved, resized, cropped, path, pct):
    fig, ax = plt.subplots(1, 3, figsize=(15, 8))
    for a, im, t in zip(ax, [carved, resized, cropped],
                        ["Seam carving", "Uniform resize", "Center crop"]):
        a.imshow(to_rgb(im)); a.set_title(f"{t}  (-{pct}% width)"); a.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--reduce", type=float, default=0.2,
                    help="fraction of width to remove (default 0.2)")
    ap.add_argument("--out", default="seam_output")
    ap.add_argument("--sweep", action="store_true",
                    help="also run 10/20/30/40%% reductions (Experiment 2)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    img = cv2.imread(args.image)
    h, w = img.shape[:2]
    energy0 = compute_energy(img)

    fractions = [0.1, 0.2, 0.3, 0.4] if args.sweep else [args.reduce]
    for f in fractions:
        n = int(w * f)
        pct = int(round(f * 100))
        print(f"Reducing width {w} -> {w - n} ({pct}%)")
        carved, seam_mask = carve_width(img, n, return_seams=True)
        resized = uniform_resize(img, w - n)
        cropped = center_crop(img, w - n)

        cv2.imwrite(f"{args.out}/carved_{pct}.jpg", carved)
        cv2.imwrite(f"{args.out}/resized_{pct}.jpg", resized)
        cv2.imwrite(f"{args.out}/cropped_{pct}.jpg", cropped)
        save_overview(img, energy0, seam_mask, carved, f"{args.out}/overview_{pct}.png")
        save_comparison(carved, resized, cropped, f"{args.out}/comparison_{pct}.png", pct)


if __name__ == "__main__":
    main()