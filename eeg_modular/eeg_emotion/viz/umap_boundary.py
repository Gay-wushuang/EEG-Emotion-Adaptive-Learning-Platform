from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class UMAPBoundaryConfig:
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "euclidean"
    random_state: Optional[int] = 42  # 允许为None，此时可以使用多线程
    n_jobs: int = -1  # 并行计算线程数，-1表示使用所有可用线程
    grid_res: int = 600
    margin: float = 0.5

    svm_kernel: str = "rbf"  # 'rbf' | 'linear'
    svm_C: float = 10.0
    svm_gamma: str = "scale"

    mode: Literal["filled", "lines", "both"] = "both"
    alpha: float = 0.25


def _fit_surrogate_and_predict_grid(X_2d: np.ndarray, y: np.ndarray, xx: np.ndarray, yy: np.ndarray, cfg: UMAPBoundaryConfig) -> np.ndarray:
    from sklearn.svm import SVC

    if cfg.svm_kernel == "rbf":
        clf = SVC(kernel="rbf", C=float(cfg.svm_C), gamma=cfg.svm_gamma)
    elif cfg.svm_kernel == "linear":
        clf = SVC(kernel="linear", C=float(cfg.svm_C))
    else:
        raise ValueError(f"Unknown svm_kernel: {cfg.svm_kernel}")

    clf.fit(X_2d, y)
    grid_pred = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    return grid_pred


def save_umap_svm_decision_boundary(
    X: np.ndarray,
    y: np.ndarray,
    class_names: List[str],
    save_path: str,
    cfg: Optional[UMAPBoundaryConfig] = None,
    title: str = "UMAP Projection with Decision Boundary (Test Set)",
    X_umap: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    plot_data: bool = True,
    return_grid: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Project X to 2D using UMAP (fit on X), then fit an SVM surrogate in 2D and plot decision boundary.

    Args:
        X: Input features (n_samples, n_features)
        y: Input labels (n_samples,)
        class_names: List of class names
        save_path: Path to save the figure
        cfg: UMAP and SVM configuration
        title: Plot title
        X_umap: Precomputed UMAP embeddings (optional). If provided, skips UMAP computation.
        ax: Existing matplotlib Axes to plot on (optional). If provided, adds to existing plot.
        plot_data: Whether to plot the data points (default: True)
        return_grid: Whether to return the grid predictions (default: True)

    Returns:
        Tuple of (X_umap, grid_pred). If return_grid is False, grid_pred is None.

    Requires `umap-learn`. Caller should catch ImportError if not installed.
    """
    cfg = cfg or UMAPBoundaryConfig()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    # 预计算UMAP结果
    if X_umap is None:
        # 改进的导入方式：只导入UMAP的UMAP类，而不是整个umap包
        # 这样可以避免导入parametric_umap和torchvision
        from umap import UMAP  # type: ignore

        reducer = UMAP(
            n_neighbors=int(cfg.n_neighbors),
            min_dist=float(cfg.min_dist),
            metric=str(cfg.metric),
            random_state=cfg.random_state,  # 不再强制转换为int，允许为None
            n_jobs=cfg.n_jobs,  # 添加多线程支持
        )
        X_umap = reducer.fit_transform(X)
    
    # 计算网格
    x_min, x_max = X_umap[:, 0].min() - cfg.margin, X_umap[:, 0].max() + cfg.margin
    y_min, y_max = X_umap[:, 1].min() - cfg.margin, X_umap[:, 1].max() + cfg.margin
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, int(cfg.grid_res)),
        np.linspace(y_min, y_max, int(cfg.grid_res)),
    )

    # 拟合SVM并预测网格
    grid_pred = _fit_surrogate_and_predict_grid(X_umap, y, xx, yy, cfg)
    
    # 处理绘图
    if ax is None:
        # 创建新图形
        fig, ax = plt.subplots(figsize=(8, 6))
        is_new_figure = True
    else:
        # 使用现有Axes
        is_new_figure = False
    
    cmap = plt.cm.Set1
    
    # 绘制决策边界
    if cfg.mode in ("filled", "both"):
        ax.contourf(xx, yy, grid_pred, alpha=float(cfg.alpha), cmap=cmap)

    if cfg.mode in ("lines", "both"):
        n_cls = len(np.unique(grid_pred))
        if n_cls >= 2:
            levels = np.arange(0.5, n_cls, 1.0)
            cs = ax.contour(xx, yy, grid_pred, levels=levels, colors="k", linewidths=1.0)
            ax.clabel(cs, inline=True, fontsize=8, fmt="")
    
    # 绘制数据点（仅在新图形或明确要求时）
    if plot_data:
        for i, name in enumerate(class_names):
            idxs = (y == i)
            if np.any(idxs):
                ax.scatter(
                    X_umap[idxs, 0],
                    X_umap[idxs, 1],
                    label=name,
                    alpha=0.75,
                    edgecolor="k",
                    s=36,
                )
        
        ax.legend()
        ax.set_title(title)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.grid(True)
    
    # 保存图形（仅在新图形时）
    if is_new_figure:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    return X_umap, (grid_pred if return_grid else None)


def save_multi_model_umap_boundary(
    X: np.ndarray,
    y: np.ndarray,
    class_names: List[str],
    save_path: str,
    models: dict,
    cfg: Optional[UMAPBoundaryConfig] = None,
    title: str = "Multi-model UMAP Decision Boundaries",
) -> np.ndarray:
    """Generate UMAP projection once, then plot decision boundaries for multiple models on the same plot.
    
    Args:
        X: Input features (n_samples, n_features)
        y: Input labels (n_samples,)
        class_names: List of class names
        save_path: Path to save the figure
        models: Dictionary of model names to their trained sklearn estimators
        cfg: Default UMAP configuration
        title: Plot title
        
    Returns:
        X_umap: UMAP embeddings (n_samples, 2)
    """
    cfg = cfg or UMAPBoundaryConfig()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    # 生成UMAP投影（仅一次）
    from umap import UMAP  # type: ignore
    reducer = UMAP(
        n_neighbors=int(cfg.n_neighbors),
        min_dist=float(cfg.min_dist),
        metric=str(cfg.metric),
        random_state=int(cfg.random_state),
    )
    X_umap = reducer.fit_transform(X)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制数据点
    for i, name in enumerate(class_names):
        idxs = (y == i)
        if np.any(idxs):
            ax.scatter(
                X_umap[idxs, 0],
                X_umap[idxs, 1],
                label=name,
                alpha=0.75,
                edgecolor="k",
                s=36,
            )
    
    # 计算网格
    x_min, x_max = X_umap[:, 0].min() - cfg.margin, X_umap[:, 0].max() + cfg.margin
    y_min, y_max = X_umap[:, 1].min() - cfg.margin, X_umap[:, 1].max() + cfg.margin
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, int(cfg.grid_res)),
        np.linspace(y_min, y_max, int(cfg.grid_res)),
    )
    
    # 为每个模型绘制决策边界
    # 创建一个自定义的线条列表用于图例
    custom_lines = []
    custom_labels = []
    
    for model_name, model in models.items():
        # 使用模型的predict方法在原始特征空间上进行预测
        # 首先需要将网格点从UMAP空间转换回原始特征空间（这是不可能的）
        # 因此，我们使用一个替代方案：在UMAP空间中训练一个代理模型，该代理模型学习原始模型的决策边界
        
        # 1. 使用原始模型对原始数据进行预测，得到预测标签
        y_pred = model.predict(X)
        
        # 2. 在UMAP空间中训练一个SVM代理模型，拟合原始模型的预测结果
        from sklearn.svm import SVC
        
        # 根据模型类型选择合适的SVM参数
        if model_name == "SVM":
            svm = SVC(kernel="rbf", C=1.0, gamma="scale")
        elif model_name == "MLP":
            svm = SVC(kernel="rbf", C=10.0, gamma="scale")
        elif model_name == "RF":
            svm = SVC(kernel="poly", C=0.1, gamma="auto")
        elif model_name == "LSTM" or model_name == "CNN":
            svm = SVC(kernel="rbf", C=5.0, gamma="scale")
        elif model_name == "XGB" or model_name == "XGBoost":
            svm = SVC(kernel="rbf", C=5.0, gamma="scale")
        else:
            svm = SVC(kernel="linear", C=1.0)
        
        # 训练代理SVM
        svm.fit(X_umap, y_pred)
        
        # 预测网格
        grid_pred = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        
        # 绘制决策边界（仅线条，不填充，便于区分）
        n_cls = len(np.unique(grid_pred))
        if n_cls >= 2:
            levels = np.arange(0.5, n_cls, 1.0)
            # 使用不同颜色区分不同模型的边界
            colors = plt.cm.tab10(list(range(len(models))))
            color_idx = list(models.keys()).index(model_name)
            color = colors[color_idx]
            linestyle = ['-', '--', '-.', ':'][color_idx % 4]
            
            # 绘制边界
            ax.contour(
                xx, yy, grid_pred, 
                levels=levels, 
                colors=[color], 
                linewidths=1.5,
                linestyles=[linestyle]
            )
            
            # 创建一个自定义的线条用于图例
            custom_lines.append(plt.Line2D([0], [0], color=color, linestyle=linestyle, linewidth=1.5))
            custom_labels.append(model_name)
    
    # 设置图例和标题
    # 首先获取数据点的图例
    handles, labels = ax.get_legend_handles_labels()
    
    # 添加模型边界的图例
    handles.extend(custom_lines)
    labels.extend(custom_labels)
    
    # 创建图例
    ax.legend(handles, labels, loc='best')
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True)
    
    # 保存图形
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    return X_umap
