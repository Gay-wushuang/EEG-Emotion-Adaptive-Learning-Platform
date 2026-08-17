import numpy as np


def stratified_split_by_class(y_int, test_ratio=0.2, random_seed=42):
    """按类别显式划分训练/测试集 - 优化版本"""
    np.random.seed(random_seed)
    train_idx, test_idx = [], []
    classes = np.unique(y_int)

    for c in classes:
        idx_c = np.where(y_int == c)[0]
        np.random.shuffle(idx_c)
        n_test = max(1, int(len(idx_c) * test_ratio))
        test_idx.extend(idx_c[:n_test])
        train_idx.extend(idx_c[n_test:])

    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    return np.array(train_idx), np.array(test_idx)
