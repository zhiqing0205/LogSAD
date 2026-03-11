"""Compute normalization statistics on the validation set.

For each category, runs the model in validation mode on the validation set (normal images only),
collects hist_score / structural_score / instance_hungarian_match_score,
and saves their statistics (min, max, q_start, q_end, mean, std, unbiased_std) to a pkl file.

Usage:
  # Few-shot model
  python compute_stats.py --module_path model_ensemble_few_shot --dataset_path /path/to/mvtec_loco

  # Full-data model (must run compute_coreset.py first)
  python compute_stats.py --module_path model_ensemble --dataset_path /path/to/mvtec_loco
"""

import argparse
import importlib
import pickle

import numpy as np
import torch
from torch import nn
from anomalib.data import MVTecLoco

FEW_SHOT_SAMPLES = [0, 1, 2, 3]
CATEGORIES = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]

# Per-category image sizes (H, W) preserving original aspect ratios.
CATEGORY_IMAGE_SIZES = {
    "breakfast_box":        (448, 560),
    "juice_bottle":         (672, 336),
    "pushpins":             (320, 544),
    "screw_bag":            (352, 512),
    "splicing_connectors":  (336, 672),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module_path", type=str, required=True)
    parser.add_argument("--class_name", default="MyModel", type=str)
    parser.add_argument("--weights_path", type=str, required=False)
    parser.add_argument("--dataset_path", default="/root/autodl-tmp/mvtec_loco_anomaly_detection", type=str)
    return parser.parse_args()


def load_model(module_path, class_name, weights_path):
    model_class = getattr(importlib.import_module(module_path), class_name)
    model = model_class()
    if weights_path:
        model.load_state_dict(torch.load(weights_path))
    return model


def compute_statistics(scores_array):
    """Compute the same statistics as stored in the original pkl files."""
    scores = np.array(scores_array)
    return {
        "min": np.float64(scores.min()),
        "max": np.float64(scores.max()),
        "q_start": np.float64(np.percentile(scores, 10)),
        "q_end": np.float64(np.percentile(scores, 90)),
        "mean": np.float64(scores.mean()),
        "std": np.float64(scores.std()),          # ddof=0, population std
        "unbiased_std": np.float64(scores.std(ddof=1)),  # ddof=1, sample std
    }


def run(module_path, class_name, weights_path, dataset_path):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    all_stats = {}

    for category in CATEGORIES:
        print(f"\n{'='*50}")
        print(f"  Processing: {category}")
        print(f"{'='*50}")

        model = load_model(module_path, class_name, weights_path)
        model.to(device)

        image_size = CATEGORY_IMAGE_SIZES[category]
        datamodule = MVTecLoco(root=dataset_path, eval_batch_size=1, image_size=image_size, category=category)
        datamodule.setup()

        model.set_viz(False)
        model.set_val(True)  # validation mode: forward returns raw scores without standardization

        # Setup with few-shot samples
        setup_data = {
            "few_shot_samples": torch.stack([datamodule.train_data[idx]["image"] for idx in FEW_SHOT_SAMPLES]).to(device),
            "few_shot_samples_path": [datamodule.train_data[idx]["image_path"] for idx in FEW_SHOT_SAMPLES],
            "dataset_category": category,
            "image_size": image_size,
        }
        model.setup(setup_data)

        # Collect scores on the validation set
        hist_scores = []
        structural_scores = []
        instance_hungarian_match_scores = []

        for data in datamodule.val_dataloader():
            with torch.no_grad():
                output = model(data["image"].to(device), data["image_path"])

            hist_scores.append(output["hist_score"].item())
            structural_scores.append(output["structural_score"].item())
            instance_hungarian_match_scores.append(output["instance_hungarian_match_score"].item())

        print(f"  Validation samples: {len(hist_scores)}")
        print(f"  hist_scores:       mean={np.mean(hist_scores):.6f}, std={np.std(hist_scores):.6f}")
        print(f"  structural_scores: mean={np.mean(structural_scores):.6f}, std={np.std(structural_scores):.6f}")
        print(f"  hungarian_scores:  mean={np.mean(instance_hungarian_match_scores):.6f}, std={np.std(instance_hungarian_match_scores):.6f}")

        all_stats[category] = {
            "hist_scores": compute_statistics(hist_scores),
            "structural_scores": compute_statistics(structural_scores),
            "instance_hungarian_match_scores": compute_statistics(instance_hungarian_match_scores),
        }

        # Free GPU memory before next category
        del model
        torch.cuda.empty_cache()

    # Save
    output_path = f"memory_bank/statistic_scores_{module_path}_val.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(all_stats, f)

    print(f"\n{'='*50}")
    print(f"  Saved to: {output_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    args = parse_args()
    run(args.module_path, args.class_name, args.weights_path, args.dataset_path)
