import numpy as np
import joblib
from pathlib import Path

# 加载训练时的特征数据
features_dir = Path("d:/proegg/eeg_modular/features")

print("=" * 80)
print("训练时的特征数据统计")
print("=" * 80)

modalities = ['filtered', 'powerspec', 'att', 'med']

for mod in modalities:
    X_train = np.load(features_dir / f"X_train_{mod}.npy")
    X_test = np.load(features_dir / f"X_test_{mod}.npy")
    
    print(f"\n{mod.upper()}:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  X_train - mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")
    print(f"  X_train - min: {X_train.min():.4f}, max: {X_train.max():.4f}")
    print(f"  X_test  - mean: {X_test.mean():.4f}, std: {X_test.std():.4f}")
    print(f"  X_test  - min: {X_test.min():.4f}, max: {X_test.max():.4f}")
    
    # 加载并检查 scaler
    scaler_path = features_dir / f"scaler_{mod}.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        print(f"  Scaler type: {type(scaler).__name__}")
        if hasattr(scaler, 'mean_'):
            print(f"  Scaler mean: {scaler.mean_}")
        if hasattr(scaler, 'scale_'):
            print(f"  Scaler scale: {scaler.scale_}")

# 查看一些样本数据
print("\n" + "=" * 80)
print("训练时的样本数据（前5个样本的前几个时间步）")
print("=" * 80)

for mod in modalities:
    X_train = np.load(features_dir / f"X_train_{mod}.npy")
    print(f"\n{mod.upper()} - 第一个样本:")
    print(X_train[0, :, :])  # 第一个样本，所有时间步，所有特征

# 查看标签分布
y_train = np.load(features_dir / "y_train_filtered.npy")
y_test = np.load(features_dir / "y_test_filtered.npy")

print("\n" + "=" * 80)
print("标签分布")
print("=" * 80)
print(f"y_train shape: {y_train.shape}")
print(f"y_train unique values: {np.unique(y_train)}")
print(f"y_train counts: {np.bincount(y_train.astype(int))}")
print(f"y_test shape: {y_test.shape}")
print(f"y_test unique values: {np.unique(y_test)}")
print(f"y_test counts: {np.bincount(y_test.astype(int))}")
