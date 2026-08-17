import os
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from configs.default_config import RANDOM_SEED, CSV_GROUPS, TIME_STEPS, TEST_RATIO, NOISE_STD, JITTER_RATIO, MIXUP_ALPHA, MODALITY_TYPES
from .feature_extraction import subsample_features
from .split import stratified_split_by_class
from .standardize import standardize_3d_features
from .augmentation import time_jitter, add_noise, mixup

# 计算数据目录路径
DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()), "..", "data")


def load_eeg_data_aligned(root_dir, csv_groups=CSV_GROUPS, time_steps=TIME_STEPS):
    samples = {k: [] for k in csv_groups}
    labels = []

    for label in sorted(os.listdir(root_dir)):
        label_path = os.path.join(root_dir, label)
        if not os.path.isdir(label_path):
            continue
        label_count = 0
        for sample in sorted(os.listdir(label_path)):
            sample_path = os.path.join(label_path, sample)
            if not os.path.isdir(sample_path):
                continue
            feats = {}
            for modality, files in csv_groups.items():
                feat = subsample_features(sample_path, files, time_steps, apply_filter=(modality=='filtered'))
                if feat is not None:
                    feats[modality] = feat
            if all(m in feats for m in csv_groups.keys()):
                for m in csv_groups.keys():
                    samples[m].append(feats[m])
                labels.append(label)
                label_count += 1
            else:
                print(f"[INFO] 样本被丢弃: {sample_path}, 缺失模态: {[m for m in csv_groups if m not in feats]}")
        print(f"[INFO] {label} 类有效样本数量: {label_count}")

    if not samples['filtered']:
        raise RuntimeError(f"[ERROR] No valid samples found in {root_dir}")

    X = {m: np.stack(samples[m], axis=0) for m in csv_groups.keys()}
    y = np.array(labels)
    print("[INFO] 加载完成:")
    for m in CSV_GROUPS.keys():
        print(f" - {m}: {X[m].shape}")
    return X, y


def preprocess_and_save(out_dir=None, test_ratio=TEST_RATIO, noise_std=NOISE_STD, 
                       jitter_ratio=JITTER_RATIO, mixup_alpha=MIXUP_ALPHA, time_steps=TIME_STEPS):
    # ✅ 修正：使用相对路径
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "features")
    os.makedirs(out_dir, exist_ok=True)
    
    print("[INFO] 开始加载数据...")
    X_raw, y_str = load_eeg_data_aligned(DATA_DIR, time_steps=time_steps)

    # 标签编码
    le = LabelEncoder()
    y_int = le.fit_transform(y_str)
    
    # ✅ 修正：兼容旧版 scikit-learn
    try:
        ohe = OneHotEncoder(sparse=False)  # 先尝试旧版参数
    except TypeError:
        ohe = OneHotEncoder(sparse_output=False)  # 新版参数
    
    y_onehot = ohe.fit_transform(y_int.reshape(-1, 1))

    # ✅ 修正：使用优化的分层划分
    train_idx, test_idx = stratified_split_by_class(y_int, test_ratio=test_ratio, random_seed=RANDOM_SEED)

    # 划分训练/测试集并标准化
    X_train, X_test, scalers = {}, {}, {}
    for m in X_raw.keys():
        X_train[m], X_test[m], scalers[m] = standardize_3d_features(
            X_raw[m][train_idx], X_raw[m][test_idx]
        )
        joblib.dump(scalers[m], os.path.join(out_dir, f'scaler_{m}.joblib'))

    y_train_oh, y_test_oh = y_onehot[train_idx], y_onehot[test_idx]
    y_train_int, y_test_int = y_int[train_idx], y_int[test_idx]

    # ==============================
    # 数据增强（按模态类型区分）- 优化版本
    # ==============================
    X_train_final, y_train_final = {}, {}

    for m in X_train.keys():
        modality_type = MODALITY_TYPES.get(m, 'signal')
        
        if modality_type == 'signal':
            # ✅ 修正：信号数据使用平衡的增强策略
            X_jitter = time_jitter(X_train[m], jitter_ratio, modality_type)
            X_noise = add_noise(X_train[m], noise_std, modality_type)
            X_mix, y_mix = mixup(X_train[m], y_train_oh, mixup_alpha, modality_type)
            
            # 平衡增强：原始 + 时间抖动 + 噪声 + mixup
            X_train_final[m] = np.concatenate([
                X_train[m],           # 原始
                X_jitter,             # 时间抖动增强
                X_noise,              # 噪声增强  
                X_mix                 # mixup增强
            ], axis=0)
            
            y_train_final[m] = np.concatenate([
                y_train_oh,           # 原始标签
                y_train_oh,           # 时间抖动标签
                y_train_oh,           # 噪声标签
                y_mix                 # mixup标签
            ], axis=0)
            
        else:  # scalar
            # ✅ 修正：标量数据只使用噪声增强，避免重复
            X_noise = add_noise(X_train[m], noise_std, modality_type)
            
            X_train_final[m] = np.concatenate([
                X_train[m],           # 原始
                X_noise               # 噪声增强
            ], axis=0)
            
            y_train_final[m] = np.concatenate([
                y_train_oh,           # 原始标签
                y_train_oh            # 噪声增强标签
            ], axis=0)
        
        # 数据质量检查
        if np.any(np.isnan(X_train_final[m])) or np.any(np.isinf(X_train_final[m])):
            print(f"[WARN] {m}模态包含NaN或无限值，进行清理...")
            X_train_final[m] = np.nan_to_num(X_train_final[m], nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 保存训练集
        np.save(os.path.join(out_dir, f'X_train_{m}.npy'), X_train_final[m])
        np.save(os.path.join(out_dir, f'y_train_{m}.npy'), y_train_final[m])
        print(f"[INFO] {m}模态({modality_type})增强: {X_train[m].shape} -> {X_train_final[m].shape}")

    # ==============================
    # 保存测试集
    # ==============================
    for m in X_test.keys():
        # 测试集数据质量检查
        if np.any(np.isnan(X_test[m])) or np.any(np.isinf(X_test[m])):
            print(f"[WARN] {m}模态测试集包含NaN或无限值，进行清理...")
            X_test[m] = np.nan_to_num(X_test[m], nan=0.0, posinf=1.0, neginf=-1.0)
        
        np.save(os.path.join(out_dir, f'X_test_{m}.npy'), X_test[m])
        np.save(os.path.join(out_dir, f'y_test_{m}.npy'), y_test_int)
        print(f"[INFO] {m}模态测试集已保存: {X_test[m].shape}")

    # ==============================
    # 保存标签编码器
    # ==============================
    joblib.dump(le, os.path.join(out_dir, 'label_encoder.joblib'))
    joblib.dump(ohe, os.path.join(out_dir, 'onehot_encoder.joblib'))

    # ==============================
    # 最终统计信息
    # ==============================
    print("\n[INFO] 处理完成 ✅")
    print("最终数据统计:")
    for m in X_train_final.keys():
        print(f"{m} - 训练集: {X_train_final[m].shape}, 测试集: {X_test[m].shape}")
    
    print(f"标签分布 - 训练集: {np.bincount(y_train_int)}, 测试集: {np.bincount(y_test_int)}")
