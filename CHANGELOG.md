# Changelog

## [Unreleased]

### Added
- Per-category rectangular image sizes preserving original aspect ratios (multiples of 16 for DINOv3)
  - breakfast_box: 448x560, juice_bottle: 672x336, pushpins: 320x544, screw_bag: 352x512, splicing_connectors: 336x672
  - All five categories achieve exact original aspect ratio (0% error)
  - CLIP loads from disk at 448x448 independently (no double interpolation); DINOv3 receives rectangular input via RoPE

### Changed
- Migrate vision backbone from DINOv2 (ViT-L/14, patch_size=14) to DINOv3 (ViT-L/16, patch_size=16)
  - Feature extraction API: `forward_features` -> `get_intermediate_layers`
  - Feature map size: 32x32 -> 28x28 (448/16=28), interpolated to 64x64
  - Layer indices: 1-based `[6, 12, 18, 24]` -> 0-based `[5, 11, 17, 23]`
  - Coreset file naming: `dinov2` -> `dinov3`
- Replace deprecated NumPy type aliases (`np.bool`, `np.int`) with Python built-in types (`bool`, `int`)

### Reverted
- Removing z-score standardization + sigmoid caused significant performance degradation — kept original normalization using validation statistics

### Fixed
- Fix `RuntimeError: generator raised StopIteration` in `mvtec_loco.py` when loading training split (no masks available)

### Added
- `dinov3/` — DINOv3 model implementation
- `compute_stats.py` — Script to compute validation set normalization statistics
- `evaluate_all.py` — Unified evaluation script for all 5 categories with Markdown output
- `commands.sh` — Quick-reference commands for training and evaluation
- `.gitignore` — Ignore caches, checkpoints, memory bank data, and results
