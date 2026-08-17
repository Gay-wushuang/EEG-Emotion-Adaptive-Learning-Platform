from __future__ import annotations
import argparse, json, os
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from eeg_emotion.config.loader import load_config, require, get
from eeg_emotion.features.sequence.extract import SequenceFeatureConfig, extract_all_features
from eeg_emotion.features.sequence.augment import (
    augment_class_samples, mixup_augment, compute_sample_stats, apply_gaussian_noise_batch
)
from eeg_emotion.features.npy_stats import load_multimodal_npy_for_sklearn, MultiModalNPYConfig
from eeg_emotion.preprocess.sequence import SequencePreprocessConfig, SequencePreprocessor
from eeg_emotion.utils.logging import setup_logging
from eeg_emotion.utils.paths import make_run_paths
from eeg_emotion.train.metrics import classification_metrics
from eeg_emotion.viz.confusion_matrix import save_confusion_matrix
from eeg_emotion.viz.seaborn_cm import save_confusion_matrix_seaborn
from eeg_emotion.viz.umap_boundary import save_umap_svm_decision_boundary

from eeg_emotion.models.torch.lstm_ae import LSTMAutoEncoder
from eeg_emotion.models.torch.lstm_clf import BiLSTMClassifier, MLPClassifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    return p.parse_args()



def plot_training_curves(history, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(history.get("loss", []), label="train_loss")
    if "val_loss" in history:
        ax1.plot(history["val_loss"], label="val_loss")
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    if "accuracy" in history:
        ax2.plot(history["accuracy"], label="train_acc")
    if "val_accuracy" in history:
        ax2.plot(history["val_accuracy"], label="val_acc")
    ax2.set_title("Accuracy Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def encode_dataset(ae: LSTMAutoEncoder, X: np.ndarray, device: torch.device, bs: int = 256):
    ae.eval()
    zs = []
    dl = DataLoader(TensorDataset(torch.from_numpy(X).float()), batch_size=bs, shuffle=False)
    for (x,) in dl:
        x = x.to(device, non_blocking=True)
        z = ae.encode(x)
        zs.append(z.detach().cpu().numpy())
    return np.concatenate(zs, axis=0)


def train_ae(ae, X_train, cfg, device, out_path, logger, run):
    ae_cfg = get(cfg, "autoencoder", {})
    epochs = int(ae_cfg.get("epochs", 100))
    bs = int(ae_cfg.get("batch_size", 32))
    val_split = float(ae_cfg.get("val_split", 0.1))
    lr = float(ae_cfg.get("lr", 1e-3))

    # split train/val
    n = X_train.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_val = max(1, int(n * val_split))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    Xtr = torch.from_numpy(X_train[tr_idx]).float()
    Xva = torch.from_numpy(X_train[val_idx]).float()

    tr_loader = DataLoader(TensorDataset(Xtr, Xtr), batch_size=bs, shuffle=True, pin_memory=True)
    va_loader = DataLoader(TensorDataset(Xva, Xva), batch_size=bs, shuffle=False, pin_memory=True)

    optim = torch.optim.Adam(ae.parameters(), lr=lr, weight_decay=float(ae_cfg.get("weight_decay", 0.0)))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=int(ae_cfg.get("lr_patience", 8)),
        min_lr=float(ae_cfg.get("min_lr", 1e-6)), verbose=True
    )
    crit = nn.MSELoss()

    best = 1e18
    patience = int(ae_cfg.get("early_stop_patience", 20))
    bad = 0

    use_amp = bool(get(cfg, "train", {}).get("use_amp", False))  # 也可单独给AE开关
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # 记录训练历史
    history = {
        "loss": [],
        "val_loss": []
    }

    for ep in range(1, epochs + 1):
        ae.train()
        tr_loss = 0.0
        for xb, yb in tr_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                recon, _ = ae(xb)
                loss = crit(recon, yb)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(tr_loader.dataset)

        ae.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                recon, _ = ae(xb)
                loss = crit(recon, yb)
                va_loss += loss.item() * xb.size(0)
        va_loss /= len(va_loader.dataset)

        # 记录历史
        history["loss"].append(tr_loss)
        history["val_loss"].append(va_loss)

        sched.step(va_loss)
        logger.info(f"[AE] epoch={ep} train_loss={tr_loss:.6f} val_loss={va_loss:.6f}")

        if va_loss < best - 1e-6:
            best = va_loss
            bad = 0
            torch.save(ae.state_dict(), out_path)
        else:
            bad += 1
            if bad >= patience:
                logger.info(f"[AE] Early stop at epoch={ep}")
                break

    # 绘制训练曲线
    plot_training_curves(history, os.path.join(run.figures_dir, "training_curves_ae.png"))
    logger.info("✅ AE training curves saved")


def train_clf(model, X, y, cfg, device, out_path, logger, run, class_weight=None):
    clf_cfg = get(cfg, "classifier", {})
    epochs = int(clf_cfg.get("epochs", 200))
    bs = int(clf_cfg.get("batch_size", 16))
    val_split = float(clf_cfg.get("val_split", 0.2))
    lr = float(clf_cfg.get("initial_lr", 1e-3))

    # split train/val
    n = X.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_val = max(1, int(n * val_split))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    Xtr = torch.from_numpy(X[tr_idx]).float()
    ytr = torch.from_numpy(y[tr_idx]).long()
    Xva = torch.from_numpy(X[val_idx]).float()
    yva = torch.from_numpy(y[val_idx]).long()

    tr_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=bs, shuffle=True, pin_memory=True)
    va_loader = DataLoader(TensorDataset(Xva, yva), batch_size=bs, shuffle=False, pin_memory=True)

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=float(clf_cfg.get("weight_decay", 0.0)))
    # cosine 可加，这里先对齐你 TF 的 ReduceLROnPlateau/early stop
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=int(clf_cfg.get("lr_patience", 10)),
        min_lr=float(clf_cfg.get("min_lr", 1e-6)), verbose=True
    )

    if class_weight is not None:
        w = torch.tensor(class_weight, dtype=torch.float32, device=device)
        crit = nn.CrossEntropyLoss(weight=w, label_smoothing=float(clf_cfg.get("label_smoothing", 0.0)))
    else:
        crit = nn.CrossEntropyLoss(label_smoothing=float(clf_cfg.get("label_smoothing", 0.0)))

    best = 1e18
    patience = int(clf_cfg.get("early_stop_patience", 30))
    bad = 0

    use_amp = bool(get(cfg, "train", {}).get("use_amp", False))
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # 记录训练历史
    history = {
        "loss": [],
        "val_loss": [],
        "accuracy": [],
        "val_accuracy": []
    }

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0
        for xb, yb in tr_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(xb)
                loss = crit(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            tr_loss += loss.item() * xb.size(0)
            # 计算准确率
            tr_pred = logits.argmax(dim=1)
            tr_correct += (tr_pred == yb).sum().item()
            tr_total += yb.size(0)
        tr_loss /= len(tr_loader.dataset)
        tr_acc = tr_correct / tr_total

        model.eval()
        va_loss = 0.0
        va_correct = 0
        va_total = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                loss = crit(logits, yb)
                va_loss += loss.item() * xb.size(0)
                # 计算准确率
                va_pred = logits.argmax(dim=1)
                va_correct += (va_pred == yb).sum().item()
                va_total += yb.size(0)
        va_loss /= len(va_loader.dataset)
        va_acc = va_correct / va_total

        # 记录历史
        history["loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["accuracy"].append(tr_acc)
        history["val_accuracy"].append(va_acc)

        sched.step(va_loss)
        logger.info(f"[CLF] epoch={ep} train_loss={tr_loss:.6f} val_loss={va_loss:.6f} train_acc={tr_acc:.4f} val_acc={va_acc:.4f}")

        if va_loss < best - 1e-6:
            best = va_loss
            bad = 0
            torch.save(model.state_dict(), out_path)
        else:
            bad += 1
            if bad >= patience:
                logger.info(f"[CLF] Early stop at epoch={ep}")
                break

    # 绘制训练曲线
    plot_training_curves(history, os.path.join(run.figures_dir, "training_curves_clf.png"))
    logger.info("✅ Classifier training curves saved")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # ---- device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # 输入shape固定时更快
    print("torch:", torch.__version__, "cuda:", torch.version.cuda, "device:", device)
    print("cudnn.enabled:", torch.backends.cudnn.enabled, "cudnn.version:", torch.backends.cudnn.version())

    out_cfg = get(cfg, "output", {})
    run = make_run_paths(base_dir=str(out_cfg.get("base_dir", "outputs")), run_name=None)
    logger = setup_logging(os.path.join(run.logs_dir, "train.log"))
    logger.info(f"📁 使用的输出目录: {run.run_dir}")
    logger.info(f"torch: {torch.__version__}, cuda: {torch.version.cuda}, device: {device}")
    logger.info(f"cudnn.enabled: {torch.backends.cudnn.enabled}, cudnn.version: {torch.backends.cudnn.version()}")

    data_dir = str(require(cfg, "data_dir", str))
    emotions = require(cfg, "emotions", list)
    time_steps = int(get(cfg, "time_steps", 128))

    # ✅ 检查是否使用 NPY 数据
    use_npy_data = bool(get(cfg, "use_npy_data", False))
    modalities = get(cfg, "modalities", ["filtered", "powerspec", "att", "med"])

    if use_npy_data:
        # ✅ 使用 NPY 数据加载器
        logger.info("🔎 Loading NPY data from %s", data_dir)
        X_train, X_test, y_train, y_test, class_names = load_multimodal_npy_for_sklearn(
            MultiModalNPYConfig(
                data_dir=data_dir,
                modalities=modalities,
            )
        )
        # ✅ 重塑为 3D 格式用于 LSTM: (N, T, F)
        # 读取第一个模态的原始形状以推断维度
        first_modality = modalities[0]
        train_path = os.path.join(data_dir, f"X_train_{first_modality}.npy")
        orig_shape = np.load(train_path).shape  # (N, T, F)
        logger.info(f"📊 Original NPY shape for {first_modality}: {orig_shape}")
        
        orig_n, orig_t, orig_f = orig_shape
        # 4个模态拼接后的总维度
        total_time = orig_t * len(modalities)  # 10 * 4 = 40
        total_features_per_step = orig_f       # 4
        
        # 验证维度是否匹配
        expected_flat_dim = total_time * total_features_per_step  # 40 * 4 = 160
        if X_train.shape[1] != expected_flat_dim:
            logger.warning(f"⚠️  Dimension mismatch: expected {expected_flat_dim}, got {X_train.shape[1]}")
            logger.warning(f"   Using actual flat_dim: {X_train.shape[1]}")
            # 使用实际维度重新计算
            total_features_per_step = X_train.shape[1] // total_time
        
        # 重塑为 3D
        num_samples = X_train.shape[0]
        X_train = X_train.reshape(num_samples, total_time, total_features_per_step)
        X_test = X_test.reshape(X_test.shape[0], total_time, total_features_per_step)
        
        # 更新 time_steps 变量以匹配实际数据
        time_steps = total_time
        logger.info("✅ Reshaped to 3D: train=%s test=%s (T=%d, F=%d)", 
                   X_train.shape, X_test.shape, total_time, total_features_per_step)
        logger.info("✅ NPY data loaded: train=%s test=%s", X_train.shape, X_test.shape)
        logger.info("📊 Train label distribution: %s", dict(Counter(y_train)))
        logger.info("📊 Test label distribution: %s", dict(Counter(y_test)))
    else:
        # ✅ 使用原始 CSV 文件加载
        csv_files = require(cfg, "csv_files", list)
        X, y = extract_all_features(SequenceFeatureConfig(
            data_dir=data_dir,
            emotions=list(emotions),
            csv_files=list(csv_files),
            time_steps=time_steps,
            min_cols_per_file=int(get(cfg, "min_cols_per_file", 10)),
        ))
        logger.info("Extracted X=%s y=%s labels=%s", X.shape, y.shape, dict(Counter(y)))

        # split
        split_cfg = get(cfg, "split", {})
        test_size = float(split_cfg.get("test_size", 0.30))
        seed = int(split_cfg.get("seed", 42))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed
        )
        logger.info("Split train=%s test=%s", X_train.shape, X_test.shape)

    # augment (train only)
    aug_cfg = get(cfg, "augment", {})
    aug_noise = get(aug_cfg, "noise", {}) or {}
    noise_mean = float(aug_noise.get("mean", 0.0))
    noise_std = float(aug_noise.get("std", 0.01))
    if bool(aug_cfg.get("enabled", True)):
        sad_times = int(aug_cfg.get("sad_times", 3))
        other_times = int(aug_cfg.get("other_times", 3))
        X_train, y_train = augment_class_samples(
            X_train, y_train, target_labels=[1],
            augment_times=sad_times, noise_mean=noise_mean, noise_std=noise_std,
        )
        X_train, y_train = augment_class_samples(
            X_train, y_train, target_labels=[0, 2],
            augment_times=other_times, noise_mean=noise_mean, noise_std=noise_std,
        )
        logger.info("After train-only augmentation labels=%s", dict(Counter(y_train)))

    # preprocess
    pp_cfg = get(cfg, "preprocess", {})
    pp = SequencePreprocessor(SequencePreprocessConfig(
        impute_strategy=str(pp_cfg.get("impute_strategy", "mean")),
        scale=bool(pp_cfg.get("scale", True)),
        pca_n_components=pp_cfg.get("pca_n_components", 64),
    ))
    X_train = pp.fit_transform(X_train)
    X_test = pp.transform(X_test)
    pp.save(run.artifacts_dir)
    logger.info("After preprocess train=%s test=%s", X_train.shape, X_test.shape)

    # gaussian noise (train only)
    noise_cfg = get(cfg, "gaussian_noise", {}) or {}
    noise_enabled = bool(noise_cfg.get("enabled", False))
    noise_apply_to = set(noise_cfg.get("apply_to", ["ae", "clf"]))
    noise_mean2 = float(noise_cfg.get("mean", 0.0))
    noise_std2 = float(noise_cfg.get("std", 0.01))
    X_train_for_ae = X_train
    X_train_for_clf_seq = X_train
    if noise_enabled and noise_std2 > 0:
        if "ae" in noise_apply_to:
            X_train_for_ae = apply_gaussian_noise_batch(X_train_for_ae, mean=noise_mean2, std=noise_std2)
        if "clf" in noise_apply_to:
            X_train_for_clf_seq = apply_gaussian_noise_batch(X_train_for_clf_seq, mean=noise_mean2, std=noise_std2)
        logger.info("Applied gaussian noise: mean=%.4f std=%.4f apply_to=%s",
                    noise_mean2, noise_std2, sorted(list(noise_apply_to)))

    # ---- AE train ----
    ae_cfg = get(cfg, "autoencoder", {})
    latent_dim = int(ae_cfg.get("latent_dim", 128))
    ae = LSTMAutoEncoder(
        input_dim=X_train_for_ae.shape[2],
        hidden_dim=int(ae_cfg.get("enc_units") or 128),
        latent_dim=latent_dim,
        num_layers=int(ae_cfg.get("enc_layers", 2)),
        dropout=float(ae_cfg.get("enc_dropout", 0.25)),
        bidir_decoder=bool(ae_cfg.get("use_bidirectional_decoder", True)),
    ).to(device)

    ae_ckpt = os.path.join(run.models_dir, "best_autoencoder.pt")
    logger.info("Training AE...")
    train_ae(ae, X_train_for_ae, cfg, device, ae_ckpt, logger, run)
    ae.load_state_dict(torch.load(ae_ckpt, map_location=device, weights_only=True))

    # encode
    X_train_enc = encode_dataset(ae, X_train, device=device, bs=256)
    X_test_enc = encode_dataset(ae, X_test, device=device, bs=256)

    # sample_stats
    stats_cfg = get(cfg, "sample_stats", {})
    if bool(stats_cfg.get("enabled", True)):
        X_train_enc = np.concatenate([X_train_enc, compute_sample_stats(X_train)], axis=1)
        X_test_enc = np.concatenate([X_test_enc, compute_sample_stats(X_test)], axis=1)

    # classifier mode
    clf_cfg = get(cfg, "classifier", {})
    clf_mode = str(clf_cfg.get("mode", "bilstm")).lower()
    num_classes = len(emotions)

    # mixup
    mix_cfg = get(cfg, "mixup", {}) or {}
    mix_ratio = float(mix_cfg.get("augment_ratio", 1.0))
    mix_alpha = float(mix_cfg.get("alpha", 0.3))
    use_mixup = bool(mix_cfg.get("enabled", True)) and mix_ratio > 0

    if clf_mode == "bilstm":
        Xtr = X_train_for_clf_seq
        ytr = y_train
        if use_mixup:
            # 你现有 mixup_augment 是 numpy 版，默认对 one-hot 更友好；
            # 这里为了最小改动：先不对 PyTorch CE 走 mixup（后续可改成 soft-label CE）。
            logger.info("Mixup for torch CE not enabled by default (keep parity first).")
        model = BiLSTMClassifier(
            input_dim=X_train_for_clf_seq.shape[2],
            num_classes=num_classes,
            hidden=int(clf_cfg.get("lstm_units", 128)),
            num_layers=int(clf_cfg.get("num_layers", 2)),
            dropout=float(clf_cfg.get("dropout", 0.35)),
            pooling=str(clf_cfg.get("pooling", "avgmax")),
        ).to(device)
        X_test_input = X_test
        y_eval = y_test
    else:
        Xtr = X_train_enc
        ytr = y_train
        model = MLPClassifier(in_dim=X_train_enc.shape[1], num_classes=num_classes,
                              dropout=float(clf_cfg.get("dropout", 0.35))).to(device)
        X_test_input = X_test_enc
        y_eval = y_test

    # class_weight（保持你 TF 逻辑：mixup 时可不使用）
    class_weight = None
    if not use_mixup:
        cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weight = cw  # numpy array

    clf_ckpt = os.path.join(run.models_dir, "best_classifier.pt")
    logger.info("Training classifier... mode=%s", clf_mode)
    train_clf(model, Xtr, ytr, cfg, device, clf_ckpt, logger, run, class_weight=class_weight)
    model.load_state_dict(torch.load(clf_ckpt, map_location=device, weights_only=True))
    model.eval()

    # eval
    with torch.no_grad():
        xb = torch.from_numpy(X_test_input).float().to(device)
        logits = model(xb)
        y_pred = logits.argmax(dim=1).cpu().numpy()

    m = classification_metrics(y_eval, y_pred, class_names=list(emotions))
    logger.info("Test accuracy: %.4f", m["accuracy"])

    # 绘制混淆矩阵（根据配置选择风格）
    viz_config = get(cfg, 'viz', {})
    use_seaborn_cm = viz_config.get("seaborn_confusion_matrix", False)
    use_seaborn_cm = bool(use_seaborn_cm)
    logger.info(f"🔍 Seaborn混淆矩阵开关: {use_seaborn_cm}")
    
    if use_seaborn_cm:
        try:
            save_confusion_matrix_seaborn(
                y_true=y_eval,
                y_pred=y_pred,
                class_names=list(emotions),
                save_path=os.path.join(run.figures_dir, "confusion_matrix.png"),
                normalize="true",
                title="Confusion Matrix (Normalized)",
            )
            logger.info("✅ Seaborn风格混淆矩阵已保存")
        except ImportError as e:
            logger.warning(f"⚠️ seaborn未安装，退回到matplotlib风格混淆矩阵: {e}")
            save_confusion_matrix(
                y_true=y_eval,
                y_pred=y_pred,
                class_names=list(emotions),
                save_path=os.path.join(run.figures_dir, "confusion_matrix.png"),
                normalize="true",
                title="Confusion Matrix (Normalized)",
            )
            logger.info("✅ Matplotlib风格混淆矩阵已保存")
    else:
        save_confusion_matrix(
            y_true=y_eval,
            y_pred=y_pred,
            class_names=list(emotions),
            save_path=os.path.join(run.figures_dir, "confusion_matrix.png"),
            normalize="true",
            title="Confusion Matrix (Normalized)",
        )
        logger.info("✅ Matplotlib风格混淆矩阵已保存")

    # 绘制UMAP边界图（如果配置启用）
    generate_umap = viz_config.get("umap_boundary", False)
    generate_umap = bool(generate_umap)
    if generate_umap:
        try:
            # 根据分类器模式选择不同的特征
            if clf_mode == "bilstm":
                # BiLSTM模式：使用测试集的原始序列特征（已经过预处理）
                # 但UMAP需要2D或3D数据，所以我们需要先降维
                # 这里使用编码后的特征，因为它们已经是低维的
                umap_X = X_test_enc
            else:
                # MLP模式：使用编码后的特征
                umap_X = X_test_enc
            
            save_umap_svm_decision_boundary(
                X=umap_X,  # 测试集特征
                y=y_eval,    # 测试集标签
                class_names=emotions,  # 类别名称
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

    out = {
        "accuracy": m["accuracy"],
        "report": m["report"],
        "best_params": {"classifier_mode": clf_mode, "time_steps": time_steps, "latent_dim": latent_dim},
        "config_path": os.path.abspath(args.config),
    }
    with open(os.path.join(run.run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    logger.info("✅ Saved to %s", run.run_dir)


if __name__ == "__main__":
    main()