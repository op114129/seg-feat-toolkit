# semantic_segmentation_and_features.py
"""Consolidated, PEP 8–compliant implementation of
1. A semantic‑segmentation inference + visualization pipeline based on MMSegmentation
2. A set of classical image‑content feature extractors (edge density, colour metrics, GLCM‑texture)

Both parts are intentionally kept in the same source file for simplicity. If you
prefer two separate modules, let me know and I will split them.

Usage
-----
# Semantic segmentation batch processing
python semantic_segmentation_and_features.py seg \
       --config path/to/seg_config.py \
       --checkpoint path/to/weights.pth \
       --images path/to/images_dir \
       --output path/to/save_dir

# Feature extraction batch processing
python semantic_segmentation_and_features.py feat \
       --images path/to/images_dir \
       --output results.xlsx
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import argparse
import logging
import os
import sys

import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import pandas as pd
import pywt  # noqa: F401  # (import kept; not directly used but kept for user later)
import torch
from matplotlib.font_manager import FontProperties  # noqa: F401  # retained for user needs
from mmseg.apis import inference_model, init_model
from skimage.feature import graycomatrix, graycoprops

# ---------------------------------------------------------------------------
# Logging & matplotlib defaults
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="[%(levelname)s] %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)

plt.rcParams["font.sans-serif"] = ["SimHei"]  # Chinese font for matplotlib
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# Semantic‑segmentation utilities
# ---------------------------------------------------------------------------


def _cityscapes_palette() -> List[tuple[int, int, int]]:
    """Return modified Cityscapes palette.

    The original *sky* colour is replaced with a darker blue and *car* with
    yellow to ensure better visual separation.
    """
    return [
        (153, 102, 153),  # road
        (153, 102, 153),  # sidewalk
        (102, 102, 102),  # building
        (51, 51, 51),  # wall
        (190, 153, 153),  # fence
        (153, 51, 255),  # pole
        (255, 255, 51),  # traffic light
        (204, 204, 102),  # traffic sign
        (102, 153, 51),  # vegetation
        (102, 102, 51),  # terrain
        (153, 204, 204),  # sky – darker blue
        (51, 51, 153),  # person
        (0, 0, 153),  # rider
        (255, 153, 51),  # car – yellow
        (102, 102, 153),  # truck
        (102, 102, 153),  # bus
        (102, 102, 153),  # train
        (102, 102, 153),  # motorcycle
        (102, 102, 153),  # bicycle
    ]


def _visualize_segmentation(
    *,
    image_path: Path,
    seg_map: np.ndarray,
    opacity: float = 0.8,
    out_file: Path | None = None,
) -> None:
    """Overlay ``seg_map`` on ``image_path`` and write the result to *out_file*.

    Parameters
    ----------
    image_path
        Path to the original RGB image.
    seg_map
        2‑D ndarray of integer labels.
    opacity
        Blending factor between original image (1 – *opacity*) and coloured
        segmentation (*opacity*).
    out_file
        If provided, the blended image is saved to this path. The parent
        folders are created automatically.
    """
    palette = _cityscapes_palette()

    # Build RGB mask from label map
    colour_mask = np.zeros((*seg_map.shape, 3), dtype=np.uint8)
    for label, colour in enumerate(palette):
        colour_mask[seg_map == label] = colour

    img = mmcv.imread(str(image_path))
    img = mmcv.imresize(img, (colour_mask.shape[1], colour_mask.shape[0]))

    blended = img * (1.0 - opacity) + colour_mask * opacity
    blended = blended.astype(np.uint8)

    plt.figure(figsize=(15, 10))
    plt.imshow(blended)
    plt.axis("off")

    if out_file is not None:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file, bbox_inches="tight", pad_inches=0)
    plt.close()


# ---------------------------------------------------------------------------
# Feature‑extraction utilities
# ---------------------------------------------------------------------------


def _edge_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150)
    return float(np.sum(edges > 0) / gray.size * 100.0)


def _color_entropy(bgr: np.ndarray) -> float:
    hist = cv2.calcHist([bgr], [0], None, [256], [0, 256])
    prob = hist / hist.sum()
    return float(-np.sum(prob * np.log2(prob + 1e-7)))


def _colorfulness(bgr: np.ndarray) -> float:
    b, g, r = cv2.split(bgr.astype("float32"))
    mu_r, mu_g, mu_b = map(float, (np.mean(r), np.mean(g), np.mean(b)))
    sigma_r, sigma_g, sigma_b = map(float, (np.std(r), np.std(g), np.std(b)))
    return (np.sqrt(sigma_r**2 + sigma_g**2 + sigma_b**2) + 0.3 * np.sqrt(mu_r**2 + mu_g**2 + mu_b**2))


def _color_moments(bgr: np.ndarray) -> List[float]:
    moments: List[float] = []
    for channel in cv2.split(bgr.astype("float32")):
        mu = float(np.mean(channel))
        sigma = float(np.std(channel))
        skew = float(np.mean(((channel - mu) / (sigma + 1e-7)) ** 3))
        moments.extend([mu, sigma, skew])
    return moments  # 9 values


def _glcm_features(gray: np.ndarray) -> Dict[str, float]:
    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )

    return {
        "GLCM_Contrast": float(np.mean(graycoprops(glcm, "contrast"))),
        "GLCM_Dissimilarity": float(np.mean(graycoprops(glcm, "dissimilarity"))),
        "GLCM_Homogeneity": float(np.mean(graycoprops(glcm, "homogeneity"))),
        "GLCM_Energy": float(np.mean(graycoprops(glcm, "energy"))),
        "GLCM_Correlation": float(np.mean(graycoprops(glcm, "correlation"))),
        "GLCM_ASM": float(np.mean(graycoprops(glcm, "ASM"))),
    }


# ---------------------------------------------------------------------------
# High‑level workflows
# ---------------------------------------------------------------------------


def run_segmentation(args: argparse.Namespace) -> None:  # noqa: D401
    """Batch inference + visualisation over a folder of images."""
    config = Path(args.config).expanduser()
    checkpoint = Path(args.checkpoint).expanduser()
    img_dir = Path(args.images).expanduser()
    out_dir = Path(args.output).expanduser()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logging.info("Initialising MMSeg model on %s", device)
    model = init_model(str(config), str(checkpoint), device=device)

    class_labels = [
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    ]

    interested_classes = {
        "绿植": ["vegetation", "terrain"],
        "建筑物": ["building", "wall"],
        "路面": ["road", "sidewalk"],
        "行人": ["person", "rider"],
        "非机动车": ["bicycle", "motorcycle"],
        "机动车": ["car", "truck", "bus", "train"],
        "天空": ["sky"],
        "标识占比": ["traffic light", "traffic sign", "pole"],
    }

    class_to_index = {label: idx for idx, label in enumerate(class_labels)}
    interested_idx = {k: [class_to_index[c] for c in v] for k, v in interested_classes.items()}

    totals = {k: 0 for k in interested_classes}
    total_pixels = 0
    per_image_stats: List[Dict[str, str]] = []

    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    with torch.no_grad():
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in valid_ext:
                continue
            logging.info("▶ Processing %s", img_path.name)
            try:
                # Inference
                result = inference_model(model, str(img_path))
                if not hasattr(result, "pred_sem_seg"):
                    raise AttributeError("Missing 'pred_sem_seg' in result")
                pred = result.pred_sem_seg.data.cpu().numpy()[0]

                # Visualisation
                _visualize_segmentation(
                    image_path=img_path,
                    seg_map=pred,
                    opacity=args.opacity,
                    out_file=out_dir / f"seg_{img_path.name}",
                )

                # Statistics
                img_pixels = pred.size
                img_stat: Dict[str, str] = {"image_name": img_path.name}
                for key, indices in interested_idx.items():
                    cnt = int(np.isin(pred, indices).sum())
                    totals[key] += cnt
                    img_stat[key] = f"{cnt / img_pixels * 100:.2f}%"
                per_image_stats.append(img_stat)
                total_pixels += img_pixels
            except Exception as exc:  # noqa: BLE001
                logging.error("Error processing %s: %s", img_path.name, exc)

    if total_pixels == 0:
        logging.warning("No images processed – please check the path and files.")
        return

    proportions = {k: v / total_pixels * 100 for k, v in totals.items()}
    (out_dir / "stats").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_image_stats).to_csv(out_dir / "stats/per_image_pixel_proportions.csv", index=False, encoding="utf-8-sig")
    pd.Series(proportions, name="percentage").to_csv(out_dir / "stats/overall_proportions.csv", header=True, encoding="utf-8-sig")

    logging.info("All statistics saved under %s", out_dir / "stats")


def run_feature_extraction(args: argparse.Namespace) -> None:  # noqa: D401
    """Batch feature extraction across a directory of images."""
    img_dir = Path(args.images).expanduser()
    output_xlsx = Path(args.output).expanduser()

    data: List[Dict[str, float]] = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        logging.info("▶ Extracting features from %s", img_path.name)
        img = cv2.imread(str(img_path))
        if img is None:
            logging.warning("Skipping unreadable file %s", img_path.name)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        feat: Dict[str, float | str] = {
            "Filename": img_path.name,
            "ED": _edge_density(gray),
            "Color_Entropy": _color_entropy(img),
            "Colorfulness": _colorfulness(img),
        }
        feat.update({f"Color_Moment_{i + 1}": v for i, v in enumerate(_color_moments(img))})
        feat.update(_glcm_features(gray))
        data.append(feat)

    if not data:
        logging.error("No features extracted – directory empty or unreadable.")
        return

    df = pd.DataFrame(data).set_index("Filename")
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_xlsx)
    logging.info("Feature table saved to %s", output_xlsx)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semantic segmentation or image‑feature extraction utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Segmentation sub‑command
    seg = subparsers.add_parser("seg", help="Run semantic segmentation batch inference")
    seg.add_argument("--config", required=True, help="Path to MMSegmentation config.py")
    seg.add_argument("--checkpoint", required=True, help="Path to model weights (.pth)")
    seg.add_argument("--images", required=True, help="Directory with input images")
    seg.add_argument("--output", required=True, help="Directory to store outputs")
    seg.add_argument("--opacity", type=float, default=0.8, help="Blending opacity for overlays [0‑1]")

    # Feature‑extraction sub‑command
    feat = subparsers.add_parser("feat", help="Run feature extraction over images")
    feat.add_argument("--images", required=True, help="Directory with input images")
    feat.add_argument("--output", required=True, help="Output Excel file path (.xlsx)")

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "seg":
        run_segmentation(args)
    elif args.command == "feat":
        run_feature_extraction(args)
    else:  # pragma: no cover – argparse guarantees this won't happen.
        parser.error("Unsupported command: %s" % args.command)


if __name__ == "__main__":
    main()
