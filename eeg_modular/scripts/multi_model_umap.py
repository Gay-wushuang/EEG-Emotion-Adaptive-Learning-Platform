from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional

import numpy as np

from eeg_emotion.config.loader import ConfigError, load_config, get, require
from eeg_emotion.features.csv_stats import DEFAULT_CSV_FILES, build_tabular_dataset
from eeg_emotion.features.npy_stats import MultiModalNPYConfig, load_multimodal_npy_for_sklearn
from eeg_emotion.models.sklearn.mlp import MLPAdapter, MLPConfig
from eeg_emotion.models.sklearn.rf import RFAdapter, RFConfig
from eeg_emotion.models.sklearn.svm import SVMModel, from_dict, SVMConfig
from eeg_emotion.models.sklearn.xgb import XGBAdapter, XGBConfig
from eeg_emotion.models.sklearn.hybrid import HybridAdapter, HybridConfig
from eeg_emotion.preprocess.tabular import TabularPreprocessConfig, TabularPreprocessor
from eeg_emotion.utils.logging import setup_logging
from eeg_emotion.utils.paths import make_run_paths
from eeg_emotion.utils.seed import set_seed
from eeg_emotion.viz.confusion_matrix import save_confusion_matrix
from eeg_emotion.viz.umap_boundary import (
    UMAPBoundaryConfig,
    save_multi_model_umap_boundary,
    save_umap_svm_decision_boundary
)


def build_model(model_cfg: Dict[str, Any], model_type: str):
    """Build model based on model type and configuration."""
    if model_type == "svm":
        # 兼容两种写法：
        # A) legacy: model: {type: svm, param_grid: {...}, probability: true, ...}
        # B) new:    model: {type: svm, svm: {...}, param_grid: {...}}
        svm_block = model_cfg.get("svm")
        svm_params = svm_block if isinstance(svm_block, dict) else model_cfg

        # 用你现在 svm.py 的 from_dict 解析（支持 kernel/C/gamma/max_iter/tol/solver 等）
        cfg = from_dict(svm_params)

        base_estimator = SVMModel(cfg).model  # 拿到真正的 sklearn estimator (SVC/LinearSVC)
        return base_estimator
    elif model_type == "mlp":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(
            hidden_layer_sizes=(128,),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            learning_rate_init=0.001,
            max_iter=int(model_cfg.get("max_iter", 500)),
            random_state=int(model_cfg.get("random_state", 42)),
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
        )
    elif model_type == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=int(model_cfg.get("n_estimators", 200)),
            max_depth=model_cfg.get("max_depth", None),
            min_samples_split=int(model_cfg.get("min_samples_split", 2)),
            min_samples_leaf=int(model_cfg.get("min_samples_leaf", 1)),
            bootstrap=model_cfg.get("bootstrap", True),
            random_state=int(model_cfg.get("random_state", 42)),
            n_jobs=int(model_cfg.get("n_jobs", -1)),
        )
    elif model_type in ("xgboost", "xgb"):
        from xgboost import XGBClassifier
        
        # 兼容两种写法：
        # A) legacy: model: {type: xgboost, param_grid: {...}, ...}
        # B) new:    model: {type: xgboost, xgboost: {...}, param_grid: {...}}
        xgb_block = model_cfg.get("xgboost") or model_cfg.get("xgb")
        xgb_params = xgb_block if isinstance(xgb_block, dict) else model_cfg

        # 基础参数（不在 param_grid 里的部分）
        # 注意：多分类需要 objective + num_class
        base_params = {
            "n_estimators": int(xgb_params.get("n_estimators", 300)),
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
            "tree_method": str(xgb_params.get("tree_method", "auto")),
            "random_state": int(xgb_params.get("random_state", 42)),
            "n_jobs": int(xgb_params.get("n_jobs", model_cfg.get("n_jobs", -1))),
        }
        
        return XGBClassifier(**base_params)
    elif model_type == "hybrid":
        from sklearn.ensemble import StackingClassifier, VotingClassifier
        from sklearn.linear_model import LogisticRegression
        
        # 构建混合模型的基础模型
        # MLP
        mlp = MLPAdapter(MLPConfig(
            cv=int(model_cfg.get("mlp_config", {}).get("cv", 5)),
            n_jobs=int(model_cfg.get("mlp_config", {}).get("n_jobs", -1)),
            random_state=int(model_cfg.get("random_state", 42)),
        ))
        
        # RF
        rf = RFAdapter(RFConfig(
            cv=int(model_cfg.get("rf_config", {}).get("cv", 5)),
            n_jobs=int(model_cfg.get("rf_config", {}).get("n_jobs", -1)),
            random_state=int(model_cfg.get("random_state", 42)),
        ))
        
        # SVM
        svm_config_dict = model_cfg.get("svm_config", {}).copy()
        for key in ["cv", "n_jobs", "param_grid", "random_state"]:
            if key in svm_config_dict:
                del svm_config_dict[key]
        svm_config_dict["probability"] = True
        svm_config_dict["solver"] = svm_config_dict.get("solver", "svc")
        svm = SVMModel(SVMConfig(**svm_config_dict))
        
        # XGB
        xgb = XGBAdapter(XGBConfig(
            cv=int(model_cfg.get("xgb_config", {}).get("cv", 5)),
            n_jobs=int(model_cfg.get("xgb_config", {}).get("n_jobs", -1)),
            random_state=int(model_cfg.get("random_state", 42)),
        ))
        
        # 构建基础模型列表
        estimators = [
            ("mlp", mlp.model if hasattr(mlp, "model") and mlp.model else mlp._create_model() if hasattr(mlp, "_create_model") else None),
            ("rf", rf.model if hasattr(rf, "model") and rf.model else None),
            ("svm", svm.model),
            ("xgb", xgb.model if hasattr(xgb, "model") and xgb.model else None),
        ]
        
        # 移除None值
        estimators = [(name, est) for name, est in estimators if est is not None]
        
        voting_method = model_cfg.get("voting_method", "soft")
        use_stacking = model_cfg.get("use_stacking", True)
        
        if use_stacking:
            # 使用StackingClassifier
            meta_estimator = LogisticRegression(random_state=int(model_cfg.get("random_state", 42)))
            return StackingClassifier(
                estimators=estimators,
                final_estimator=meta_estimator,
                cv=5,
                stack_method="auto",
                n_jobs=-1,
            )
        else:
            # 使用VotingClassifier
            return VotingClassifier(
                estimators=estimators,
                voting=voting_method,
                n_jobs=-1,
            )
    else:
        # 目前只支持sklearn模型，CNN和LSTM是深度学习模型，需要不同的处理方式
        raise ValueError(f"Unsupported model type: {model_type}. Currently only sklearn models (SVM, MLP, RF, XGBoost, Hybrid) are supported.")


def build_preprocess(pp_cfg: Dict[str, Any]) -> TabularPreprocessor:
    """Build preprocessor based on configuration."""
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
    """Parse command line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--configs", "-c", required=True, nargs="+", help="Paths to YAML/JSON config files for different models.")
    p.add_argument("--output", "-o", required=True, help="Output directory for the multi-model UMAP figure.")
    return p.parse_args()


def main() -> None:
    """Main function to generate multi-model UMAP boundary plot."""
    args = parse_args()
    
    # 设置随机种子
    set_seed(42)
    
    # 创建输出目录
    run = make_run_paths(base_dir=args.output, run_name="multi_model_umap")
    logger = setup_logging(os.path.join(run.logs_dir, "multi_model_umap.log"))
    
    # 加载第一个配置文件，获取数据相关配置
    first_cfg = load_config(args.configs[0])
    data_dir = str(require(first_cfg, "data_dir", str))
    emotions = require(first_cfg, "emotions", list)
    
    # 检查是否使用NPY数据
    use_npy_data = bool(get(first_cfg, "use_npy_data", False))
    
    if use_npy_data:
        # 使用NPY数据
        modalities = require(first_cfg, "modalities", list)
        logger.info("🔎 Loading NPY data from %s", data_dir)
        
        npy_cfg = MultiModalNPYConfig(
            data_dir=data_dir,
            modalities=modalities
        )
        X_train, X_test, y_train, y_test, class_names = load_multimodal_npy_for_sklearn(npy_cfg)
        
        # 合并训练集和测试集用于UMAP可视化
        X_all = np.concatenate([X_train, X_test], axis=0)
        y_all = np.concatenate([y_train, y_test], axis=0)
        
        logger.info("✅ Samples: %d (train: %d, test: %d) | Features: %d", 
                    X_all.shape[0], X_train.shape[0], X_test.shape[0], X_all.shape[1])
        
        # 构建预处理管道
        pp = build_preprocess(require(first_cfg, "preprocess", dict))
        
        # 预处理数据
        X_train_t, y_train_t = pp.fit_transform_train(X_train, y_train)
        X_test_t = pp.transform(X_test)
    else:
        # 使用CSV数据
        csv_files = list(get(first_cfg, "csv_files", list(DEFAULT_CSV_FILES)))
        
        # 构建数据集
        logger.info("🔎 Building dataset from %s", data_dir)
        X_all, y_all, skipped = build_tabular_dataset(
            data_dir=data_dir, 
            emotions=emotions, 
            csv_files=csv_files
        )
        logger.info("✅ Samples: %d | Features: %d", X_all.shape[0], X_all.shape[1])
        
        # 构建预处理管道
        pp = build_preprocess(require(first_cfg, "preprocess", dict))
        
        # 划分数据集
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, 
            test_size=float(get(first_cfg, "split.test_size", 0.30)),
            random_state=int(get(first_cfg, "split.random_state", 42)),
            stratify=y_all
        )
        
        # 预处理数据
        X_train_t, y_train_t = pp.fit_transform_train(X_train, y_train)
        X_test_t = pp.transform(X_test)
    
    # 准备模型字典
    models = {}
    
    # 加载所有配置文件，构建模型
    for config_path in args.configs:
        cfg = load_config(config_path)
        
        try:
            model_dict = require(cfg, "model", dict)
            model_type = str(require(model_dict, "type", str)).lower()
            model_name = model_type.upper()
            
            # 构建模型
            model = build_model(model_dict, model_type)
            models[model_name] = model
            logger.info("✅ Added %s model from %s", model_name, config_path)
        except ConfigError as e:
            logger.warning("⚠️ Skipping %s: %s", config_path, e)
        except ValueError as e:
            logger.warning("⚠️ Skipping %s: %s", config_path, e)
    
    # 训练所有模型
    for model_name, model in models.items():
        logger.info("⏳ Training %s model...", model_name)
        model.fit(X_train_t, y_train_t)
        logger.info("✅ %s model trained successfully", model_name)
    
    # 生成多模型UMAP边界图
    logger.info("🎨 Generating multi-model UMAP boundary plot...")
    save_multi_model_umap_boundary(
        X=X_test_t,
        y=y_test,
        class_names=emotions,
        save_path=os.path.join(run.figures_dir, "multi_model_umap_boundary.png"),
        models=models,
        title="Multi-model UMAP Decision Boundaries",
    )
    logger.info("✅ Multi-model UMAP boundary plot saved to %s", run.figures_dir)
    
    logger.info("✅ All done! Output saved to %s", run.run_dir)


if __name__ == "__main__":
    main()
