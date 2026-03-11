#!/bin/bash
set -e

# ========== 配置（按需修改）==========
DATASET_PATH="/root/data/datasets/mvtec_loco_anomaly_detection"
FEW_SHOT_CATEGORIES=(breakfast_box juice_bottle pushpins screw_bag splicing_connectors)
FULL_CATEGORIES=(breakfast_box juice_bottle pushpins screw_bag splicing_connectors)

# ========== 用法 ==========
usage() {
    echo "Usage: ./run.sh [mode] [--dataset PATH]"
    echo ""
    echo "Modes:"
    echo "  few-shot    仅少样本 (4-shot) 评测"
    echo "  full        仅全量 (coreset) 评测"
    echo "  clean       清理 memory_bank 缓存"
    echo "  (无参数)     依次运行 few-shot + full"
    exit 1
}

# 解析参数
MODE="all"
while [[ $# -gt 0 ]]; do
    case "$1" in
        few-shot|full|clean|all) MODE="$1"; shift ;;
        --dataset) DATASET_PATH="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown: $1"; usage ;;
    esac
done

# ========== 少样本 (4-shot) ==========
run_few_shot() {
    echo "===== Few-shot (4-shot) 评测 ====="
    echo "Categories: ${FEW_SHOT_CATEGORIES[*]}"
    python compute_stats.py \
        --module_path model_ensemble_few_shot \
        --dataset_path "$DATASET_PATH"
    python evaluate_all.py \
        --module_path model_ensemble_few_shot \
        --dataset_path "$DATASET_PATH"
}

# ========== 全量 (coreset) ==========
run_full() {
    echo "===== Full (coreset) 评测 ====="
    echo "Categories: ${FULL_CATEGORIES[*]}"
    for cat in "${FULL_CATEGORIES[@]}"; do
        echo "--- coreset: $cat ---"
        python compute_coreset.py \
            --module_path model_ensemble \
            --category "$cat" \
            --dataset_path "$DATASET_PATH"
    done
    python compute_stats.py \
        --module_path model_ensemble \
        --dataset_path "$DATASET_PATH"
    python evaluate_all.py \
        --module_path model_ensemble \
        --dataset_path "$DATASET_PATH"
}

# ========== 清理缓存 ==========
run_clean() {
    echo "===== 清理 memory_bank ====="
    rm -fv memory_bank/mem_patch_feature_clip_*.pt \
           memory_bank/mem_patch_feature_dinov2_*.pt \
           memory_bank/mem_patch_feature_dinov3_*.pt \
           memory_bank/mem_instance_features_multi_stage_*.pt
}

case "$MODE" in
    few-shot) run_few_shot ;;
    full)     run_full ;;
    clean)    run_clean ;;
    all)      run_few_shot; run_full ;;
    *)        usage ;;
esac
