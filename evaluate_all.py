"""Unified evaluation script: run all 5 categories and output markdown results."""

import argparse
import gc
import importlib
import os
from datetime import datetime

import numpy as np
import torch
import logging
from torch import nn

from anomalib.data import MVTecLoco
from anomalib.metrics.f1_max import F1Max
from anomalib.metrics.auroc import AUROC

CATEGORIES = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
FEW_SHOT_SAMPLES = [0, 1, 2, 3]

# Per-category image sizes (H, W) preserving original aspect ratios.
# CLIP always resized to 448x448; only DINOv3 (patch=16) needs divisibility by 16.
CATEGORY_IMAGE_SIZES = {
    "breakfast_box":        (576, 720),   # orig 1600x1280 (5:4)   ratio=1.250 exact
    "juice_bottle":         (896, 448),   # orig 800x1600  (1:2)   ratio=0.500 exact
    "pushpins":             (480, 816),   # orig 1700x1000 (17:10) ratio=1.700 exact
    "screw_bag":            (528, 768),   # orig 1600x1100 (16:11) ratio=1.455 exact
    "splicing_connectors":  (448, 896),   # orig 1700x850  (2:1)   ratio=2.000 exact
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module_path", type=str, required=True)
    parser.add_argument("--class_name", default="MyModel", type=str, required=False)
    parser.add_argument("--weights_path", type=str, required=False)
    parser.add_argument("--dataset_path", default="/root/autodl-tmp/mvtec_loco_anomaly_detection/", type=str, required=False)
    parser.add_argument("--output_dir", default="results", type=str, required=False)
    parser.add_argument("--viz", action="store_true", default=False)
    return parser.parse_args()


def load_model(module_path: str, class_name: str, weights_path: str) -> nn.Module:
    model_class = getattr(importlib.import_module(module_path), class_name)
    model = model_class()
    if weights_path:
        model.load_state_dict(torch.load(weights_path))
    return model


def evaluate_category(module_path, class_name, weights_path, dataset_path, category, viz, device):
    """Evaluate a single category and return metrics dict."""
    model = load_model(module_path, class_name, weights_path)
    model.to(device)

    image_size = CATEGORY_IMAGE_SIZES[category]
    datamodule = MVTecLoco(root=dataset_path, eval_batch_size=1, image_size=image_size, category=category)
    datamodule.setup()
    datamodule_clip = MVTecLoco(root=dataset_path, eval_batch_size=1, image_size=(448, 448), category=category)
    datamodule_clip.setup()

    model.set_viz(viz)

    image_metric = F1Max()
    image_metric_logical = F1Max()
    image_metric_structure = F1Max()
    image_metric_auroc = AUROC()
    image_metric_auroc_logical = AUROC()
    image_metric_auroc_structure = AUROC()

    setup_data = {
        "few_shot_samples": torch.stack([datamodule.train_data[idx]["image"] for idx in FEW_SHOT_SAMPLES]).to(device),
        "few_shot_samples_clip": torch.stack([datamodule_clip.train_data[idx]["image"] for idx in FEW_SHOT_SAMPLES]).to(device),
        "few_shot_samples_path": [datamodule.train_data[idx]["image_path"] for idx in FEW_SHOT_SAMPLES],
        "dataset_category": category,
        "image_size": image_size,
    }
    model.setup(setup_data)

    for data, data_clip in zip(datamodule.test_dataloader(), datamodule_clip.test_dataloader()):
        with torch.no_grad():
            image_path = data["image_path"]
            output = model(data["image"].to(device), data_clip["image"].to(device), data["image_path"])

        image_metric.update(output["pred_score"].cpu(), data["label"])
        image_metric_auroc.update(output["pred_score"].cpu(), data["label"])

        if "logical" not in image_path[0]:
            image_metric_structure.update(output["pred_score"].cpu(), data["label"])
            image_metric_auroc_structure.update(output["pred_score"].cpu(), data["label"])
        if "structural" not in image_path[0]:
            image_metric_logical.update(output["pred_score"].cpu(), data["label"])
            image_metric_auroc_logical.update(output["pred_score"].cpu(), data["label"])

    results = {
        "f1_image": np.round(image_metric.compute().item() * 100, decimals=2),
        "auroc_image": np.round(image_metric_auroc.compute().item() * 100, decimals=2),
        "f1_logical": np.round(image_metric_logical.compute().item() * 100, decimals=2),
        "auroc_logical": np.round(image_metric_auroc_logical.compute().item() * 100, decimals=2),
        "f1_structural": np.round(image_metric_structure.compute().item() * 100, decimals=2),
        "auroc_structural": np.round(image_metric_auroc_structure.compute().item() * 100, decimals=2),
    }

    # Cleanup
    del model, datamodule
    gc.collect()
    torch.cuda.empty_cache()

    return results


def generate_markdown(all_results, module_path):
    """Generate markdown string from results."""
    is_few_shot = "few_shot" in module_path
    protocol = f"Few-shot ({len(FEW_SHOT_SAMPLES)}-shot)" if is_few_shot else "Full-data"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("# LogSAD Evaluation Results\n")
    lines.append("## Configuration")
    lines.append(f"- Protocol: {protocol}")
    lines.append("- DINO Backbone: DINOv3 ViT-L/16")
    lines.append("- CLIP Backbone: ViT-L-14 (DataComp.XL)")
    lines.append("- SAM Backbone: ViT-H")
    lines.append("- Image Size: per-category rectangular (see table)")
    lines.append("  - " + ", ".join(f"{c}: {CATEGORY_IMAGE_SIZES[c][0]}x{CATEGORY_IMAGE_SIZES[c][1]}" for c in CATEGORIES))
    lines.append(f"- Date: {timestamp}")
    lines.append("")
    lines.append("## Results\n")
    lines.append("| Category | F1-Max (image) | AUROC (image) | F1-Max (logical) | AUROC (logical) | F1-Max (structural) | AUROC (structural) |")
    lines.append("|---|---|---|---|---|---|---|")

    metric_keys = ["f1_image", "auroc_image", "f1_logical", "auroc_logical", "f1_structural", "auroc_structural"]
    means = {k: [] for k in metric_keys}

    for cat in CATEGORIES:
        r = all_results[cat]
        for k in metric_keys:
            means[k].append(r[k])
        lines.append(f"| {cat} | {r['f1_image']:.2f} | {r['auroc_image']:.2f} | {r['f1_logical']:.2f} | {r['auroc_logical']:.2f} | {r['f1_structural']:.2f} | {r['auroc_structural']:.2f} |")

    mean_vals = {k: np.mean(v) for k, v in means.items()}
    lines.append(f"| **Mean** | **{mean_vals['f1_image']:.2f}** | **{mean_vals['auroc_image']:.2f}** | **{mean_vals['f1_logical']:.2f}** | **{mean_vals['auroc_logical']:.2f}** | **{mean_vals['f1_structural']:.2f}** | **{mean_vals['auroc_structural']:.2f}** |")
    lines.append("")

    return "\n".join(lines)


def main():
    args = parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    all_results = {}
    for category in CATEGORIES:
        print(f"\n{'='*60}")
        print(f"Evaluating: {category}")
        print(f"{'='*60}")
        results = evaluate_category(
            args.module_path, args.class_name, args.weights_path,
            args.dataset_path, category, args.viz, device,
        )
        all_results[category] = results
        print(f"  F1-Max(image): {results['f1_image']:.2f}  AUROC(image): {results['auroc_image']:.2f}")

    # Generate and save markdown
    md_content = generate_markdown(all_results, args.module_path)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    module_tag = args.module_path.replace("model_ensemble", "").strip("_") or "full"
    filename = f"results_{module_tag}_{timestamp}.md"
    filepath = os.path.join(args.output_dir, filename)

    with open(filepath, "w") as f:
        f.write(md_content)

    print(f"\n{'='*60}")
    print(f"Results saved to: {filepath}")
    print(f"{'='*60}\n")
    print(md_content)


if __name__ == "__main__":
    main()
