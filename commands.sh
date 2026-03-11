# 环境准备（每次开新终端执行一次）
source /etc/network_turbo
export HF_HUB_DISABLE_XET=1
cd /root/autodl-tmp/LogSAD

# 清理旧的 coreset 文件（DINOv2 + DINOv3）
rm -f memory_bank/mem_patch_feature_clip_*.pt memory_bank/mem_patch_feature_dinov2_*.pt memory_bank/mem_patch_feature_dinov3_*.pt memory_bank/mem_instance_features_multi_stage_*.pt

# ========== 计算验证集统计量 (生成 pkl) ==========
# 少样本模型的统计量（不依赖 coreset，可以直接跑）
python compute_stats.py --module_path model_ensemble_few_shot --dataset_path /root/autodl-tmp/mvtec_loco_anomaly_detection
# 全量模型的统计量（必须先跑完下面的 compute_coreset 之后再跑这条）
# python compute_stats.py --module_path model_ensemble --dataset_path /root/autodl-tmp/mvtec_loco_anomaly_detection

# ========== 少样本 (4-shot) 一键评测 ==========
python evaluate_all.py --module_path model_ensemble_few_shot --dataset_path /root/autodl-tmp/mvtec_loco_anomaly_detection

# ========== 全量数据 (coreset + eval) ==========
# 逐类别计算 coreset
python compute_coreset.py --module_path model_ensemble --category breakfast_box --dataset_path /root/autodl-tmp/mvtec_loco_anomaly_detection
python compute_coreset.py --module_path model_ensemble --category juice_bottle --dataset_path /root/autodl-tmp/mvtec_loco_anomaly_detection
python compute_coreset.py --module_path model_ensemble --category pushpins --dataset_path /root/autodl-tmp/mvtec_loco_anomaly_detection
python compute_coreset.py --module_path model_ensemble --category screw_bag --dataset_path /root/autodl-tmp/mvtec_loco_anomaly_detection
python compute_coreset.py --module_path model_ensemble --category splicing_connectors --dataset_path /root/autodl-tmp/mvtec_loco_anomaly_detection

# 全量模型统计量（在 coreset 全部算完后执行）
python compute_stats.py --module_path model_ensemble --dataset_path /root/autodl-tmp/mvtec_loco_anomaly_detection

# 全量一键评测
python evaluate_all.py --module_path model_ensemble --dataset_path /root/autodl-tmp/mvtec_loco_anomaly_detection
