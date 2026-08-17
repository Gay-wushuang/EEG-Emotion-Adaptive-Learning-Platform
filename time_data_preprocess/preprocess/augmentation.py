import numpy as np


def time_jitter(X, max_shift_ratio=0.05, modality_type='signal'):
    """时间抖动 - 仅信号数据使用"""
    if modality_type == 'scalar':
        return X  # 标量数据完全跳过时间抖动
    
    max_shift = max(1, int(X.shape[1] * max_shift_ratio))
    X_out = np.empty_like(X)
    for i, s in enumerate(X):
        shift = np.random.randint(-max_shift, max_shift + 1)
        X_out[i] = np.roll(s, shift, axis=0)
    return X_out.astype(np.float32)


def add_noise(X, noise_std=0.02, modality_type='signal'):
    """添加噪声 - 标量数据使用极小噪声"""
    if modality_type == 'scalar':
        noise_std = noise_std * 0.05  # 标量数据使用非常小的噪声（原噪声的5%）
    noise = np.random.normal(0, noise_std, X.shape).astype(np.float32)
    return (X + noise).astype(np.float32)


def mixup(X, y_onehot, alpha=0.4, modality_type='signal'):
    """Mixup增强 - 标量数据完全跳过，返回原始数据"""
    if modality_type == 'scalar':
        return X, y_onehot  # 标量数据完全跳过mixup，返回原始数据
    
    # 信号数据的原有mixup逻辑
    X, y = np.asarray(X, dtype=np.float32), np.asarray(y_onehot, dtype=np.float32)
    n = len(X)
    X_mix, y_mix = np.empty_like(X), np.empty_like(y)
    for i in range(n):
        j = np.random.randint(0, n)
        lam = np.random.beta(alpha, alpha)
        X_mix[i] = lam * X[i] + (1 - lam) * X[j]
        y_mixed = lam * y[i] + (1 - lam) * y[j]
        # 简化标签处理，保持概率和为1
        y_mixed = y_mixed / (y_mixed.sum() + 1e-8)
        y_mix[i] = y_mixed
    return X_mix, y_mix
