from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold

from eeg_emotion.train.metrics import classification_metrics
from eeg_emotion.viz.confusion_matrix import save_confusion_matrix
from eeg_emotion.viz.seaborn_cm import save_confusion_matrix_seaborn
from eeg_emotion.viz.training_curves import plot_training_curves


@dataclass(frozen=True)
class TorchTrainConfig:
    epochs: int = 200
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-3
    num_workers: int = 0
    n_splits: int = 5
    seed: int = 42
    use_amp: bool = False
    log_every: int = 10


def _to_device_batch(x_dict: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in x_dict.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, use_labels_for_forward: bool = False):
    model.eval()
    all_preds, all_targets = [], []
    for x_dict, yb in loader:
        x_dict = _to_device_batch(x_dict, device)
        yb = yb.to(device)
        logits = model(x_dict, labels=yb if use_labels_for_forward else None)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(yb.cpu())
    return torch.cat(all_preds).numpy(), torch.cat(all_targets).numpy()


def train_kfold(
    model_fn: Callable[[], torch.nn.Module],
    train_dataset,
    test_loader: DataLoader,
    class_names: List[str],
    out_dir: str,
    cfg: TorchTrainConfig,
    optimizer_fn: Optional[Callable[[torch.nn.Module], torch.optim.Optimizer]] = None,
    scheduler_fn: Optional[Callable[[torch.optim.Optimizer], Any]] = None,
    criterion: Optional[torch.nn.Module] = None,
    use_labels_for_forward: bool = False,
    device: Optional[torch.device] = None,
    use_seaborn_confusion_matrix: bool = False,
    run: Optional[Any] = None,
) -> Dict[str, Any]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    models_dir = os.path.join(out_dir, "models")
    figs_dir = os.path.join(out_dir, "figures")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    def default_optim(m: torch.nn.Module):
        return torch.optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    optimizer_fn = optimizer_fn or default_optim

    kfold = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    n_samples = len(train_dataset)

    best_overall_val_acc = -1.0
    best_ckpt_path = None
    best_fold_history = None
    fold_best_accs: List[float] = []

    scaler = torch.amp.GradScaler('cuda', enabled=bool(cfg.use_amp))

    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(n_samples))):
        model = model_fn().to(device)
        optimizer = optimizer_fn(model)
        scheduler = scheduler_fn(optimizer) if scheduler_fn else None

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            sampler=SubsetRandomSampler(train_idx),
            num_workers=cfg.num_workers,
        )
        val_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            sampler=SubsetRandomSampler(val_idx),
            num_workers=cfg.num_workers,
        )

        best_fold_acc = -1.0
        fold_history = {
            'loss': [],
            'val_loss': [],
            'val_acc': []
        }
        ckpt_path = os.path.join(models_dir, f"best_fold{fold+1}.pt")

        for epoch in range(cfg.epochs):
            model.train()
            running_correct = 0
            total = 0
            running_loss = 0.0

            for i, (x_dict, yb) in enumerate(train_loader):
                x_dict = _to_device_batch(x_dict, device)
                yb = yb.to(device)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=bool(cfg.use_amp)):
                    logits = model(x_dict, labels=yb if use_labels_for_forward else None)
                    loss = criterion(logits, yb)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                preds = logits.argmax(dim=1)
                running_correct += (preds == yb).sum().item()
                total += yb.size(0)
                running_loss += float(loss.item()) * yb.size(0)

                # CosineWarmRestarts wants fractional epoch like original script
                if scheduler is not None:
                    try:
                        scheduler.step(epoch + (i / max(1, len(train_loader))))
                    except TypeError:
                        scheduler.step()

            train_loss = running_loss / total
            train_acc = running_correct / total

            # val
            val_preds, val_targets = evaluate(model, val_loader, device=device, use_labels_for_forward=use_labels_for_forward)
            val_acc = float((val_preds == val_targets).mean())
            
            # 计算验证损失
            val_loss = 0.0
            val_total = 0
            model.eval()
            with torch.no_grad():
                for i, (x_dict, yb) in enumerate(val_loader):
                    x_dict = _to_device_batch(x_dict, device)
                    yb = yb.to(device)
                    with torch.amp.autocast('cuda', enabled=bool(cfg.use_amp)):
                        logits = model(x_dict, labels=yb if use_labels_for_forward else None)
                        loss = criterion(logits, yb)
                    val_loss += float(loss.item()) * yb.size(0)
                    val_total += yb.size(0)
            val_loss /= val_total

            # 记录当前折的训练历史
            fold_history['loss'].append(train_loss)
            fold_history['val_loss'].append(val_loss)
            fold_history['val_acc'].append(val_acc)

            # 每log_every个epoch输出一次日志
            if epoch % cfg.log_every == 0:
                print(f"[FOLD {fold+1}/{cfg.n_splits}] epoch={epoch} train_loss={train_loss:.6f} train_acc={train_acc:.6f} val_loss={val_loss:.6f} val_acc={val_acc:.6f}")

            if val_acc > best_fold_acc:
                best_fold_acc = val_acc
                torch.save(model.state_dict(), ckpt_path)
                print(f"[FOLD {fold+1}/{cfg.n_splits}] New best val_acc={val_acc:.6f} at epoch={epoch}")

        fold_best_accs.append(best_fold_acc)
        if best_fold_acc > best_overall_val_acc:
            best_overall_val_acc = best_fold_acc
            best_ckpt_path = ckpt_path
            best_fold_history = fold_history.copy()
    
    # 删除非最佳折的模型文件
    for fold in range(cfg.n_splits):
        fold_ckpt = os.path.join(models_dir, f"best_fold{fold+1}.pt")
        if fold_ckpt != best_ckpt_path and os.path.exists(fold_ckpt):
            os.remove(fold_ckpt)
    
    # 使用最佳折的训练历史
    training_history = best_fold_history if best_fold_history is not None else {
        'loss': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # 绘制训练曲线
    if run is not None and hasattr(run, 'figures_dir'):
        plot_training_curves(training_history, os.path.join(run.figures_dir, "training_curves_cnn.png"))
    elif figs_dir:
        plot_training_curves(training_history, os.path.join(figs_dir, "training_curves_cnn.png"))

    # Evaluate best fold on test set
    final_model = model_fn().to(device)
    if best_ckpt_path and os.path.exists(best_ckpt_path):
        final_model.load_state_dict(torch.load(best_ckpt_path, map_location=device, weights_only=True))
    test_preds, test_targets = evaluate(final_model, test_loader, device=device, use_labels_for_forward=use_labels_for_forward)

    m = classification_metrics(test_targets, test_preds, class_names=class_names)

    if use_seaborn_confusion_matrix:
        save_confusion_matrix_seaborn(
            y_true=test_targets,
            y_pred=test_preds,
            class_names=class_names,
            save_path=os.path.join(figs_dir, "confusion_matrix.png"),
            normalize="true",
            title="Confusion Matrix (Normalized)",
        )
    else:
        save_confusion_matrix(
            y_true=test_targets,
            y_pred=test_preds,
            class_names=class_names,
            save_path=os.path.join(figs_dir, "confusion_matrix.png"),
            normalize="true",
            title="Confusion Matrix (Normalized)",
        )

    out = {
        "accuracy": m["accuracy"],
        "report": m["report"],
        "best_params": {
            "best_overall_val_acc": best_overall_val_acc,
            "fold_best_accs": fold_best_accs,
            "n_splits": cfg.n_splits,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "use_amp": cfg.use_amp,
        },
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    summary = {
        "val_best_acc": float(best_overall_val_acc),
        "test_acc": float(m["accuracy"]),
        "paths": {"run_dir": out_dir, "best_ckpt": best_ckpt_path},
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return out
