from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any, Dict

import numpy as np
from sklearn.model_selection import train_test_split

from eeg_emotion.config.loader import ConfigError, load_config, get, require
from eeg_emotion.features.csv_stats import DEFAULT_CSV_FILES, build_tabular_dataset
from eeg_emotion.features.npy_stats import load_multimodal_npy_for_sklearn, MultiModalNPYConfig
from eeg_emotion.models.sklearn.mlp import MLPAdapter, MLPConfig
from eeg_emotion.models.sklearn.rf import RFAdapter, RFConfig
from eeg_emotion.models.sklearn.svm import SVMModel, from_dict, SVMConfig
from eeg_emotion.models.sklearn.xgb import XGBAdapter, XGBConfig
from eeg_emotion.models.sklearn.hybrid import HybridAdapter, HybridConfig

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib
from xgboost import XGBClassifier

from eeg_emotion.preprocess.tabular import TabularPreprocessConfig, TabularPreprocessor
from eeg_emotion.train.metrics import classification_metrics
from eeg_emotion.utils.logging import setup_logging
from eeg_emotion.utils.paths import make_run_paths
from eeg_emotion.utils.seed import set_seed
from eeg_emotion.viz.confusion_matrix import save_confusion_matrix
from eeg_emotion.viz.umap_boundary import save_umap_svm_decision_boundary

#通用 GridSearch 包装器
class SklearnSearchAdapter:
    def __init__(self, search):
        self.search = search
        self.best_params_ = None
        self.best_score_ = None

    def fit(self, X, y, **fit_params):
        self.search.fit(X, y, **fit_params)
        self.best_params_ = getattr(self.search, "best_params_", None)
        self.best_score_ = getattr(self.search, "best_score_", None)
        return self

    def predict(self, X):
        return self.search.predict(X)

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(self.search, os.path.join(out_dir, "model.joblib"))


def build_model(model_cfg: Dict[str, Any]):
    mtype = require(model_cfg, "type", str).lower()

    # Common grid keys are passed verbatim to GridSearchCV
    param_grid = model_cfg.get("param_grid")
    if mtype == "svm":
        # 兼容两种写法：
        # A) legacy: model: {type: svm, param_grid: {...}, probability: true, ...}
        # B) new:    model: {type: svm, svm: {...}, param_grid: {...}}
        svm_block = model_cfg.get("svm")
        svm_params = svm_block if isinstance(svm_block, dict) else model_cfg

        # 用你现在 svm.py 的 from_dict 解析（支持 kernel/C/gamma/max_iter/tol/solver 等）
        cfg = from_dict(svm_params)

        base_estimator = SVMModel(cfg).model  # 拿到真正的 sklearn estimator (SVC/LinearSVC)

        if param_grid:
            # 避免 GridSearchCV (外层并行) + XGBoost (内层并行) 造成线程嵌套
            try:
                base_estimator.set_params(n_jobs=1)
            except Exception:
                pass
            search = GridSearchCV(
                estimator=base_estimator,
                param_grid=param_grid,
                cv=int(model_cfg.get("cv", 5)),
                n_jobs=int(model_cfg.get("n_jobs", -1)),
            )
            return SklearnSearchAdapter(search)

        # 不做网格搜索就直接返回一个轻量适配器，提供 fit/predict/save
        class _SVMNoSearchAdapter:
            def __init__(self, est, cfg):
                self.est = est
                self.cfg = cfg

            def fit(self, X, y):
                self.est.fit(X, y)
                return self

            def predict(self, X):
                return self.est.predict(X)

            def save(self, out_dir: str):
                os.makedirs(out_dir, exist_ok=True)
                joblib.dump({"cfg": self.cfg, "model": self.est}, os.path.join(out_dir, "model.joblib"))

        return _SVMNoSearchAdapter(base_estimator, cfg)

    if mtype == "mlp":
        return MLPAdapter(
            MLPConfig(
                param_grid=param_grid,
                cv=int(model_cfg.get("cv", 5)),
                n_jobs=int(model_cfg.get("n_jobs", -1)),
                max_iter=int(model_cfg.get("max_iter", 500)),
                random_state=int(model_cfg.get("random_state", 42)),
            )
        )
    if mtype == "rf":
        return RFAdapter(
            RFConfig(
                param_grid=param_grid,
                cv=int(model_cfg.get("cv", 5)),
                n_jobs=int(model_cfg.get("n_jobs", -1)),
                random_state=int(model_cfg.get("random_state", 42)),
            )
        )
    if mtype in ("xgboost", "xgb"):
        # 兼容两种写法：
        # A) legacy: model: {type: xgboost, param_grid: {...}, ...}
        # B) new:    model: {type: xgboost, xgboost: {...}, param_grid: {...}}
        xgb_block = model_cfg.get("xgboost") or model_cfg.get("xgb")
        xgb_params = xgb_block if isinstance(xgb_block, dict) else model_cfg

        # 基础参数（不在 param_grid 里的部分）
        # 注意：多分类需要 objective + num_class
        base_params = {
            "n_estimators": int(xgb_params.get("n_estimators", 5000)),  # 配合 early stopping，不会真跑满
            "max_depth": int(xgb_params.get("max_depth", 6)),
            "learning_rate": float(xgb_params.get("learning_rate", 0.05)),
            "subsample": float(xgb_params.get("subsample", 0.9)),
            "colsample_bytree": float(xgb_params.get("colsample_bytree", 0.9)),
            "min_child_weight": float(xgb_params.get("min_child_weight", 1.0)),
            "gamma": float(xgb_params.get("gamma", 0.0)),
            "reg_lambda": float(xgb_params.get("reg_lambda", 1.0)),
            "reg_alpha": float(xgb_params.get("reg_alpha", 0.0)),
            "objective": str(xgb_params.get("objective", "multi:softprob")),
            "eval_metric": str(xgb_params.get("eval_metric", "mlogloss")),
            "tree_method": str(xgb_params.get("tree_method", "hist")),  # GPU 推荐 hist
            "device": str(xgb_params.get("device", "cuda")),  # 启用 GPU 加速
            "random_state": int(xgb_params.get("random_state", 42)),
            "n_jobs": int(xgb_params.get("n_jobs", 1)),  # ✅ 内层固定 1，防止并行套并行
            "verbosity": int(xgb_params.get("verbosity", 0)),  # 减少噪声
            "validate_parameters": bool(xgb_params.get("validate_parameters", False)),  # 减少噪声
        }

        # num_class: 由外部 main() 的 emotions 决定更准确，但 build_model 不拿 cfg。
        # 这里允许配置里显式给 num_class；否则留空，fit 时也能推断。
        if "num_class" in xgb_params:
            base_params["num_class"] = int(xgb_params["num_class"])

        base_estimator = XGBClassifier(**base_params)

        if param_grid:
            cv = int(model_cfg.get("cv", 5))
            outer_jobs = int(model_cfg.get("n_jobs", -1))
            use_random = bool(model_cfg.get("random_search", True))  # 默认开启随机搜索
            n_iter = int(model_cfg.get("n_iter", 40))

            if use_random:
                search = RandomizedSearchCV(
                    estimator=base_estimator,
                    param_distributions=param_grid,
                    n_iter=n_iter,
                    cv=cv,
                    n_jobs=outer_jobs,
                    random_state=int(model_cfg.get("random_state", 42)),
                    verbose=2,  # 改为2，显示更详细的训练进度
                )
            else:
                search = GridSearchCV(
                    estimator=base_estimator,
                    param_grid=param_grid,
                    cv=cv,
                    n_jobs=outer_jobs,
                    verbose=2,  # 改为2，显示更详细的训练进度
                )
            return SklearnSearchAdapter(search)

        class _XGBNoSearchAdapter:
            def __init__(self, est, base_params):
                self.est = est
                self.base_params = base_params

            def fit(self, X, y):
                # 若未显式设置 num_class，则在这里自动推断并设置
                if getattr(self.est, "objective", None) == "multi:softprob" and getattr(self.est, "num_class", None) in (None, 0):
                    try:
                        ncls = int(len(set(y)))
                        self.est.set_params(num_class=ncls)
                    except Exception:
                        pass
                self.est.fit(X, y)
                return self

            def predict(self, X):
                return self.est.predict(X)

            def save(self, out_dir: str):
                os.makedirs(out_dir, exist_ok=True)
                joblib.dump({"base_params": self.base_params, "model": self.est}, os.path.join(out_dir, "model.joblib"))

        return _XGBNoSearchAdapter(base_estimator, base_params)

    if mtype == "hybrid":
        # 构建混合模型配置
        mlp_config = MLPConfig(
            param_grid=model_cfg.get("mlp_config", {}).get("param_grid"),
            cv=int(model_cfg.get("mlp_config", {}).get("cv", 5)),
            n_jobs=int(model_cfg.get("mlp_config", {}).get("n_jobs", -1)),
            random_state=int(model_cfg.get("random_state", 42)),
        )
        
        rf_config = RFConfig(
            param_grid=model_cfg.get("rf_config", {}).get("param_grid"),
            cv=int(model_cfg.get("rf_config", {}).get("cv", 5)),
            n_jobs=int(model_cfg.get("rf_config", {}).get("n_jobs", -1)),
            random_state=int(model_cfg.get("random_state", 42)),
        )
        
        # 处理SVM配置，避免重复参数和不支持的参数
        svm_config_dict = model_cfg.get("svm_config", {}).copy()
        # 移除SVMConfig不支持的参数
        for key in ["cv", "n_jobs", "param_grid", "random_state"]:
            if key in svm_config_dict:
                del svm_config_dict[key]
        # 确保probability为True，因为soft voting需要概率输出
        svm_config_dict["probability"] = True
        svm_config_dict["solver"] = svm_config_dict.get("solver", "svc")
        
        svm_config = SVMConfig(**svm_config_dict)
        
        xgb_config = XGBConfig(
            param_grid=model_cfg.get("xgb_config", {}).get("param_grid"),
            cv=int(model_cfg.get("xgb_config", {}).get("cv", 5)),
            n_jobs=int(model_cfg.get("xgb_config", {}).get("n_jobs", -1)),
            random_state=int(model_cfg.get("random_state", 42)),
        )
        
        hybrid_config = HybridConfig(
            mlp_config=mlp_config,
            rf_config=rf_config,
            svm_config=svm_config,
            xgb_config=xgb_config,
            voting_method=str(model_cfg.get("voting_method", "soft")),
            use_stacking=bool(model_cfg.get("use_stacking", True)),
            random_state=int(model_cfg.get("random_state", 42)),
        )
        
        return HybridAdapter(hybrid_config)

    raise ConfigError(f"Unsupported model.type: {mtype} (supported: svm/mlp/rf/xgboost/hybrid)")

def build_preprocess(pp_cfg: Dict[str, Any]) -> TabularPreprocessor:
    cfg = TabularPreprocessConfig(
        impute_strategy=str(pp_cfg.get("impute_strategy", "mean")),
        scale=bool(pp_cfg.get("scale", True)),
        select_k_best=pp_cfg.get("select_k_best", None),
        pca_n_components=pp_cfg.get("pca_n_components", None),
        augment=bool(pp_cfg.get("augment", True)),
        noise_std=float(pp_cfg.get("noise_std", 0.01)),
        time_jitter=float(pp_cfg.get("time_jitter", 0.02)),
    )
    return TabularPreprocessor(cfg)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", required=True, help="Path to YAML/JSON config file.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(get(cfg, "seed", 42))
    set_seed(seed)

    data_dir = str(require(cfg, "data_dir", str))
    emotions = require(cfg, "emotions", list)
    if not all(isinstance(x, str) for x in emotions):
        raise ConfigError("emotions must be a list of strings.")
    csv_files = list(get(cfg, "csv_files", list(DEFAULT_CSV_FILES)))

    # Check if we should use NPY data (from time_data_preprocess)
    use_npy_data = bool(get(cfg, "use_npy_data", False))

    split_cfg = get(cfg, "split", {})
    test_size = float(split_cfg.get("test_size", 0.30))
    val_size = float(split_cfg.get("val_size", 0.10))
    random_state = int(split_cfg.get("random_state", seed))

    out_cfg = get(cfg, "output", {})
    base_dir = str(out_cfg.get("base_dir", "outputs"))

    # 使用配置文件中的run_name
    run_name = out_cfg.get("run_name", None)

    run = make_run_paths(base_dir=base_dir, run_name=run_name)
    logger = setup_logging(os.path.join(run.logs_dir, "train.log"))
    
    # 记录最终使用的run_dir
    logger.info(f"📁 使用的输出目录: {run.run_dir}")

    model = build_model(require(cfg, "model", dict))
    pp = build_preprocess(require(cfg, "preprocess", dict))

    # -------------------- dataset -------------------- #
    if use_npy_data:
        # Load preprocessed NPY data from time_data_preprocess
        logger.info("🔎 Loading NPY data from %s", data_dir)

        # Use all modalities by default
        modalities = list(get(cfg, "modalities", ["filtered", "powerspec", "att", "med"]))

        npy_cfg = MultiModalNPYConfig(
            data_dir=data_dir,
            modalities=modalities,
        )

        X_train, X_test, y_train, y_test, class_names_from_npy = load_multimodal_npy_for_sklearn(npy_cfg)

        logger.info("✅ NPY data loaded successfully")
        logger.info("   Train samples: %d | Test samples: %d", X_train.shape[0], X_test.shape[0])
        logger.info("   Feature dimension: %d", X_train.shape[1])
        logger.info("📊 Train label distribution: %s", dict(Counter(y_train)))
        logger.info("📊 Test label distribution: %s", dict(Counter(y_test)))
        logger.info("📊 Class names: %s", class_names_from_npy)

        # Use validation split from train set if needed
        if val_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train
            )
            logger.info("   Validation split: train=%s val=%s", X_train.shape, X_val.shape)
        else:
            X_val, y_val = None, None
            logger.info("   No validation split (val_size=0)")

        # Apply preprocessing (standardization, etc.)
        X_train_t, y_train_t = pp.fit_transform_train(X_train, y_train)
        X_test_t = pp.transform(X_test)
        X_val_t = pp.transform(X_val) if X_val is not None else None

        # NPY data is already standardized, but we still apply preprocessing pipeline
        # to ensure consistency (e.g., if additional preprocessing is needed)
    else:
        # Original CSV-based data loading
        logger.info("🔎 Building dataset from %s", data_dir)
        X_all, y_all, skipped = build_tabular_dataset(data_dir=data_dir, emotions=emotions, csv_files=csv_files)
        logger.info("✅ Samples: %d | Features: %d", X_all.shape[0], X_all.shape[1])
        logger.info("📊 Label distribution: %s", dict(Counter(y_all)))
        if skipped:
            logger.info("⚠️ Skipped samples: %d (first 5 shown)", len(skipped))
            for s in skipped[:5]:
                logger.info("   - %s", s)

        # -------------------- split -------------------- #
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_all, y_all, test_size=test_size, random_state=random_state, stratify=y_all
        )

        # val_size is proportion of temp
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )
        logger.info("Split: train=%s val=%s test=%s", X_train.shape, X_val.shape, X_test.shape)

        # -------------------- preprocess -------------------- #
        X_train_t, y_train_t = pp.fit_transform_train(X_train, y_train)
        X_val_t = pp.transform(X_val)
        X_test_t = pp.transform(X_test)

    # -------------------- fit -------------------- #
    logger.info("⏳ Training model...")
    # 简化训练逻辑，移除early stopping，确保所有模型都能正常训练
    model.fit(X_train_t, y_train_t)

    best_params = getattr(model, "best_params_", None)
    if best_params:
        logger.info("✅ Best params: %s", best_params)

    # -------------------- evaluate (test) -------------------- #
    y_pred = model.predict(X_test_t)
    m = classification_metrics(y_test, y_pred, class_names=emotions)
    logger.info("🎯 Test accuracy: %.4f", m["accuracy"])

    # -------------------- save -------------------- #
    model.save(run.models_dir)
    pp.save(run.artifacts_dir)

    # 绘制混淆矩阵（根据配置选择风格）
    viz_config = get(cfg, 'viz', {})
    logger.info(f"📋 viz配置: {viz_config}")
    use_seaborn_cm = viz_config.get("seaborn_confusion_matrix", False)
    use_seaborn_cm = bool(use_seaborn_cm)
    logger.info(f"🔍 Seaborn混淆矩阵开关: {use_seaborn_cm}")
    
    if use_seaborn_cm:
        try:
            from eeg_emotion.viz.seaborn_cm import save_confusion_matrix_seaborn
            save_confusion_matrix_seaborn(
                y_true=y_test,
                y_pred=y_pred,
                class_names=emotions,
                save_path=os.path.join(run.figures_dir, "confusion_matrix.png"),
                normalize="true",
                title="Confusion Matrix (Normalized)",
            )
            logger.info("✅ Seaborn风格混淆矩阵已保存")
        except ImportError as e:
            logger.warning(f"⚠️ seaborn未安装，退回到matplotlib风格混淆矩阵: {e}")
            from eeg_emotion.viz.confusion_matrix import save_confusion_matrix
            save_confusion_matrix(
                y_true=y_test,
                y_pred=y_pred,
                class_names=emotions,
                save_path=os.path.join(run.figures_dir, "confusion_matrix.png"),
                normalize="true",
                title="Confusion Matrix (Normalized)",
            )
            logger.info("✅ Matplotlib风格混淆矩阵已保存")
    else:
        from eeg_emotion.viz.confusion_matrix import save_confusion_matrix
        save_confusion_matrix(
            y_true=y_test,
            y_pred=y_pred,
            class_names=emotions,
            save_path=os.path.join(run.figures_dir, "confusion_matrix.png"),
            normalize="true",
            title="Confusion Matrix (Normalized)",
        )
        logger.info("✅ Matplotlib风格混淆矩阵已保存")

    # 绘制UMAP边界图（如果配置启用）
    # 直接从viz_config字典中获取值，而不是使用get函数
    generate_umap = viz_config.get("umap_boundary", False)
    # 确保generate_umap是布尔值
    generate_umap = bool(generate_umap)
    logger.info(f"🔍 UMAP生成开关: {generate_umap}")
    if generate_umap:
        logger.info("🎨 开始生成UMAP边界图...")
        try:
            # 打印UMAP绘制所需的信息
            logger.info(f"📊 UMAP输入特征形状: {X_test_t.shape}")
            logger.info(f"📊 UMAP输入标签形状: {y_test.shape}")
            logger.info(f"📊 类别名称: {emotions}")
            
            save_umap_svm_decision_boundary(
                X=X_test_t,  # 测试集特征
                y=y_test,    # 测试集标签
                class_names=emotions,  # 类别名称
                save_path=os.path.join(run.figures_dir, "umap_boundary.png"),  # 保存路径
                title="UMAP Projection with Decision Boundary (Test Set)",  # 标题
            )
            logger.info("✅ UMAP boundary plot saved")
        except ImportError as e:
            logger.warning(f"⚠️ umap-learn not installed, skipping UMAP boundary plot: {e}")
        except RuntimeError as e:
            if "torchvision" in str(e) or "nms" in str(e):
                logger.warning(f"⚠️ torchvision compatibility issue, skipping UMAP boundary plot: {e}")
                logger.warning("   建议：1) 安装兼容版本的torchvision 或 2) 降低umap-learn版本")
            else:
                logger.error(f"❌ Failed to generate UMAP boundary: {e}")
                import traceback
                logger.error(f"📋 错误堆栈: {traceback.format_exc()}")
        except Exception as e:
            logger.error(f"❌ Failed to generate UMAP boundary: {e}")
            import traceback
            logger.error(f"📋 错误堆栈: {traceback.format_exc()}")
    else:
        logger.info("🚫 UMAP边界图生成已关闭")

    out = {
        "accuracy": m["accuracy"],
        "report": m["report"],
        "best_params": best_params,
        "emotions": emotions,
        "seed": seed,
        "config_path": os.path.abspath(args.config),
    }
    with open(os.path.join(run.run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    logger.info("✅ Saved to: %s", run.run_dir)


if __name__ == "__main__":
    main()
