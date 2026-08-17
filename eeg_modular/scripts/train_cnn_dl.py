from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from eeg_emotion.config.loader import load_config, require, get
from eeg_emotion.dl.common import set_seed
from eeg_emotion.dl.torch.data import MultiModalNPYConfig, load_multimodal_npy, MultiModalTensorDataset
from eeg_emotion.dl.torch.trainer import TorchTrainConfig, train_kfold
from eeg_emotion.dl.torch.losses import WeightedFocalLoss
from eeg_emotion.models.torch.multimodal_cvae_cnn import MultiModalCVAECNN, MultiModalCVAECNNConfig
from eeg_emotion.models.torch.cvae import CVAEConfig, load_cvae
from eeg_emotion.train.weights import ClassWeightConfig, compute_balanced_class_weights, apply_manual_weights, to_tensor
from eeg_emotion.utils.paths import make_run_paths
from eeg_emotion.utils.logging import setup_logging
from eeg_emotion.viz.umap_boundary import save_umap_svm_decision_boundary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(get(cfg, "seed", 42))
    set_seed(seed)

    out_cfg = get(cfg, "output", {})
    # 不使用配置文件中的run_name，所有模型都使用时间戳作为输出目录名
    run = make_run_paths(base_dir=str(out_cfg.get("base_dir", "outputs")), run_name=None)
    logger = setup_logging(os.path.join(run.logs_dir, "train.log"))
    # 记录最终使用的run_dir
    logger.info(f"📁 使用的输出目录: {run.run_dir}")

    cnn_cfg = require(cfg, "cnn", dict)
    data_dir = str(require(cnn_cfg, "data_dir", str))
    modalities = list(require(cnn_cfg, "modalities", list))

    input_cfg = cnn_cfg.get("input", {}) or {}
    time_steps = input_cfg.get("time_steps", 10)
    feat_dim = input_cfg.get("feat_dim", 4)

    # Load data (robust)
    X_train_dict, X_test_dict, y_train, y_test, class_names = load_multimodal_npy(
        MultiModalNPYConfig(
            data_dir=data_dir,
            modalities=modalities,
            time_steps=int(time_steps) if time_steps is not None else None,
            feat_dim=int(feat_dim) if feat_dim is not None else None,
        )
    )
    train_ds = MultiModalTensorDataset(X_train_dict, y_train)
    test_ds = MultiModalTensorDataset(X_test_dict, y_test)

    bs = int(cnn_cfg.get("train", {}).get("batch_size", 32))
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

    num_classes = len(class_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch version: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"cuda version: {torch.version.cuda}")
    print(f"cudnn enabled: {torch.backends.cudnn.enabled}, cudnn version: {torch.backends.cudnn.version()}")
    print(f"Using device: {device}")
    logger.info(f"Using device: {device}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}, cuda version: {torch.version.cuda}")
    logger.info(f"cuDNN enabled: {torch.backends.cudnn.enabled}, cuDNN version: {torch.backends.cudnn.version()}")

    # class weights + focal
    w_cfg = cnn_cfg.get("class_weight", {}) or {}
    cw = compute_balanced_class_weights(y_train)
    cw = apply_manual_weights(class_names, cw, ClassWeightConfig(sad_multiplier=float(w_cfg.get("sad_multiplier", 3.0))))
    class_weight_tensor = to_tensor(cw, device=device)
    gamma = float((cnn_cfg.get("loss", {}) or {}).get("focal_gamma", 2.0))
    criterion = WeightedFocalLoss(alpha=class_weight_tensor, gamma=gamma)

    # CVAE
    cvae_model = None
    cvae_cfg = cnn_cfg.get("cvae", {}) or {}
    use_cvae = bool(cvae_cfg.get("enabled", True))
    if use_cvae:
        # compute total_features based on modalities and their shapes (T*F per modality)
        total_features = 0
        for m in modalities:
            arr = X_train_dict[m]
            total_features += int(arr.shape[1] * arr.shape[2])
        cvae_model = load_cvae(
            input_dim=total_features,
            num_classes=num_classes,
            cfg=CVAEConfig(
                latent_dim=int(cvae_cfg.get("latent_dim", 64)),
                checkpoint=str(cvae_cfg.get("checkpoint", "")),
                py_path=cvae_cfg.get("py_path"),
                strict=bool(cvae_cfg.get("strict", False)),
            ),
            device=device,
        )
    else:
        logger.info("CVAE disabled (cnn.cvae.enabled=false)")

    mcfg = MultiModalCVAECNNConfig(
        modalities=modalities,
        dropout=float(cnn_cfg.get("dropout", 0.5)),
        cvae_latent_dim=int((cnn_cfg.get("cvae", {}) or {}).get("latent_dim", 64)),
        use_cvae=use_cvae,
    )

    train_cfg_raw = cnn_cfg.get("train", {}) or {}
    tcfg = TorchTrainConfig(
        epochs=int(train_cfg_raw.get("epochs", 200)),
        batch_size=bs,
        lr=float(train_cfg_raw.get("lr", 1e-3)),
        weight_decay=float(train_cfg_raw.get("weight_decay", 1e-3)),
        num_workers=int(train_cfg_raw.get("num_workers", 4)),
        n_splits=int(train_cfg_raw.get("n_splits", 5)),
        seed=seed,
        use_amp=bool(train_cfg_raw.get("use_amp", False)),
        log_every=int(train_cfg_raw.get("log_every", 10)),
    )

    def model_fn():
        return MultiModalCVAECNN(num_classes=num_classes, cfg=mcfg, cvae_model=cvae_model)

    def scheduler_fn(optimizer: torch.optim.Optimizer):
        s_cfg = cnn_cfg.get("scheduler", {}) or {}
        t0 = int(s_cfg.get("T_0", 10))
        tm = int(s_cfg.get("T_mult", 2))
        eta_min = float(s_cfg.get("eta_min", 1e-6))
        return CosineAnnealingWarmRestarts(optimizer, T_0=t0, T_mult=tm, eta_min=eta_min)

    logger.info("🚀 Training CNN (CVAE=%s, focal_gamma=%.2f, splits=%d)", use_cvae, gamma, tcfg.n_splits)

    # 获取可视化配置
    viz_config = get(cfg, 'viz', {})
    use_seaborn_cm = viz_config.get("seaborn_confusion_matrix", False)
    use_seaborn_cm = bool(use_seaborn_cm)
    logger.info(f"🔍 Seaborn混淆矩阵开关: {use_seaborn_cm}")
    
    out = train_kfold(
        model_fn=model_fn,
        train_dataset=train_ds,
        test_loader=test_loader,
        class_names=class_names,
        out_dir=run.run_dir,
        cfg=tcfg,
        scheduler_fn=scheduler_fn,
        criterion=criterion,
        use_labels_for_forward=use_cvae,  # CVAE needs labels in forward
        device=device,
        use_seaborn_confusion_matrix=use_seaborn_cm,
        run=run
    )

    # 绘制UMAP边界图（如果配置启用）
    generate_umap = viz_config.get("umap_boundary", False)
    generate_umap = bool(generate_umap)
    if generate_umap:
        try:
            # 准备UMAP数据
            # 我们需要将测试集数据转换为适合UMAP的格式
            # 对于多模态数据，我们可以将所有模态的数据展平并拼接
            X_test_flat = []
            for m in modalities:
                # 获取测试集数据
                modality_data = X_test_dict[m]
                # 展平数据：(samples, time_steps, feat_dim) -> (samples, time_steps * feat_dim)
                flattened = modality_data.reshape(modality_data.shape[0], -1)
                X_test_flat.append(flattened)
            # 拼接所有模态的数据
            X_test_combined = np.concatenate(X_test_flat, axis=1)
            
            save_umap_svm_decision_boundary(
                X=X_test_combined,  # 测试集特征
                y=y_test,    # 测试集标签
                class_names=class_names,  # 类别名称
                save_path=os.path.join(run.figures_dir, "umap_boundary.png"),  # 保存路径
                title="UMAP Projection with Decision Boundary (Test Set)",  # 标题
            )
            logger.info("✅ UMAP boundary plot saved")
        except ImportError:
            logger.warning("⚠️ umap-learn not installed, skipping UMAP boundary plot")
        except RuntimeError as e:
            if "torchvision" in str(e) or "nms" in str(e):
                logger.warning(f"⚠️ torchvision compatibility issue, skipping UMAP boundary plot: {e}")
                logger.warning("   建议：1) 安装兼容版本的torchvision 或 2) 降低umap-learn版本")
            else:
                logger.error(f"❌ Failed to generate UMAP boundary: {e}")
        except Exception as e:
            logger.error(f"❌ Failed to generate UMAP boundary: {e}")
    
    logger.info("✅ Done. test accuracy=%.4f run_dir=%s", out["accuracy"], run.run_dir)


if __name__ == "__main__":
    main()
