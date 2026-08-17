from __future__ import annotations

import json
import os
from collections import Counter

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

from eeg_emotion.features.csv_stats import DEFAULT_CSV_FILES, build_tabular_dataset
from eeg_emotion.preprocess.pipeline import PreprocessConfig, SklearnPreprocessPipeline, augment_tabular
from eeg_emotion.utils.logging import setup_logging
from eeg_emotion.utils.paths import make_run_paths
from eeg_emotion.utils.seed import set_seed
from eeg_emotion.viz.confusion_matrix import save_confusion_matrix
from eeg_emotion.viz.umap_boundary import save_umap_svm_decision_boundary


def main() -> None:
    # -------------------- 基本配置（先写死，下一步再上yaml config） -------------------- #
    set_seed(42)

    data_dir = "./data"  # 你的数据根目录：data/<emotion>/sample*/xxx.csv
    emotions = ["happy", "sad", "normal"]
    csv_files = list(DEFAULT_CSV_FILES)

    run = make_run_paths(base_dir="outputs")
    logger = setup_logging(os.path.join(run.logs_dir, "train.log"))

    logger.info("🔎 Building dataset from %s", data_dir)
    X_all, y_all, skipped = build_tabular_dataset(data_dir=data_dir, emotions=emotions, csv_files=csv_files)

    logger.info("✅ Samples: %d | Features: %d", X_all.shape[0], X_all.shape[1])
    logger.info("📊 Label distribution: %s", dict(Counter(y_all)))
    if skipped:
        logger.info("⚠️ Skipped samples: %d (first 5 shown)", len(skipped))
        for s in skipped[:5]:
            logger.info("   - %s", s)

    # -------------------- 划分：先 split，再 fit 预处理，避免信息泄漏 -------------------- #
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_all, y_all, test_size=0.30, random_state=42, stratify=y_all
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.10, random_state=42, stratify=y_temp
    )
    logger.info("Split: train=%s val=%s test=%s", X_train.shape, X_val.shape, X_test.shape)

    # -------------------- 预处理（fit on train only） -------------------- #
    pp_cfg = PreprocessConfig(impute_strategy="mean", scale=True, pca_n_components=None, select_k_best=None)
    pp = SklearnPreprocessPipeline(pp_cfg)

    # 注意：增强要发生在 impute 之后、scale 之前还是之后？
    # 你原脚本是：impute -> augment -> scale。这里保持一致：
    X_train_imputed = pp.imputer.fit_transform(X_train)  # imputer单独fit
    X_val_imputed = pp.imputer.transform(X_val)
    X_test_imputed = pp.imputer.transform(X_test)

    X_train_aug, y_train_aug = augment_tabular(X_train_imputed, y_train, noise_std=0.01, time_jitter=0.02)

    # scaler/select/pca 在增强后的训练集 fit
    if pp.scaler is not None:
        X_train_scaled = pp.scaler.fit_transform(X_train_aug)
        X_val_scaled = pp.scaler.transform(X_val_imputed)
        X_test_scaled = pp.scaler.transform(X_test_imputed)
    else:
        X_train_scaled, X_val_scaled, X_test_scaled = X_train_aug, X_val_imputed, X_test_imputed

    # 如果你后面要加 selectKBest / PCA：建议把“augment后再fit”统一放进 pipeline.fit()
    pp._is_fit = True  # 标记为fit，用于transform时不报错（本脚本目前只用到imputer/scaler）

    # -------------------- SVM 网格搜索（示例：先做最小集合） -------------------- #
    param_grid = {
        "C": [0.1, 1, 10],
        "gamma": ["scale", "auto"],
        "kernel": ["rbf", "poly", "sigmoid"],
        "class_weight": [None, "balanced"],
    }

    logger.info("⏳ GridSearchCV for SVM...")
    grid = GridSearchCV(SVC(probability=True), param_grid, refit=True, cv=5, n_jobs=-1)
    grid.fit(X_train_scaled, y_train_aug)

    model: SVC = grid.best_estimator_
    logger.info("✅ Best params: %s", grid.best_params_)

    # -------------------- 测试集评估 -------------------- #
    y_pred = model.predict(X_test_scaled)
    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, target_names=emotions, output_dict=True)

    logger.info("🎯 Test accuracy: %.4f", acc)
    logger.info("\n" + classification_report(y_test, y_pred, target_names=emotions))

    # -------------------- 产物保存 -------------------- #
    joblib.dump(model, os.path.join(run.models_dir, "svm.joblib"))

    # 保存“可复用的预处理部件”（先保存imputer+scaler；下一步我们把augment也抽象进pipeline）
    joblib.dump({"imputer": pp.imputer, "scaler": pp.scaler, "cfg": pp_cfg}, os.path.join(run.artifacts_dir, "preprocess_parts.joblib"))

    save_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        class_names=emotions,
        save_path=os.path.join(run.figures_dir, "confusion_matrix.png"),
        normalize="true",
        title="SVM Confusion Matrix (Normalized)",
    )

    # 绘制UMAP边界图
    try:
        save_umap_svm_decision_boundary(
            X=X_test_scaled,  # 测试集特征
            y=y_test,    # 测试集标签
            class_names=emotions,  # 类别名称
            save_path=os.path.join(run.figures_dir, "umap_boundary.png"),  # 保存路径
            title="UMAP Projection with Decision Boundary (Test Set)",  # 标题
        )
        logger.info("✅ UMAP boundary plot saved")
    except ImportError:
        logger.warning("⚠️ umap-learn not installed, skipping UMAP boundary plot")
    except Exception as e:
        logger.error(f"❌ Failed to generate UMAP boundary: {e}")

    metrics = {
        "accuracy": acc,
        "best_params": grid.best_params_,
        "report": report,
    }
    with open(os.path.join(run.run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info("✅ Saved to: %s", run.run_dir)


if __name__ == "__main__":
    main()

# NOTE: 推荐使用 scripts/train.py（第二步：更干净的 preprocess + model adapter）。

# NOTE: Step 3 起推荐使用：python -m scripts.train -c configs/svm.yaml
