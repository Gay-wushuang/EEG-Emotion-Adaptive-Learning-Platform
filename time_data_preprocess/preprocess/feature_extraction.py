import os
import numpy as np
import pandas as pd
from .filters import bandpass_filter
from .alignment import merge_csvs


def extract_time_features(df, time_steps=10):
    if df is None or len(df) == 0:
        return None
    n_points = len(df)
    seg_len = max(1, n_points // time_steps)
    feat_segments = []

    for t in range(time_steps):
        start, end = t * seg_len, min((t + 1) * seg_len, n_points)
        if start >= n_points:
            feat_segments.append(np.zeros(df.shape[1] * 4, dtype=np.float32))
            continue

        seg = df.iloc[start:end].to_numpy()
        feat_list = []
        for c in range(seg.shape[1]):
            col = seg[:, c]
            if len(col) > 0:
                feat_list.extend([np.mean(col), np.std(col), np.max(col), np.min(col)])
            else:
                feat_list.extend([0, 0, 0, 0])
        feat_segments.append(np.array(feat_list, dtype=np.float32))
    return np.stack(feat_segments, axis=0)


def subsample_features(sample_path, group_files, time_steps=10, apply_filter=True):
    csv_paths = [os.path.join(sample_path, f) for f in group_files if os.path.exists(os.path.join(sample_path, f))]
    if not csv_paths:
        print(f"[WARN] 样本缺失CSV文件: {sample_path} -> {group_files}")
        return None
    merged_df = merge_csvs(csv_paths, freq='100ms')
    if merged_df is None or merged_df.empty:
        print(f"[WARN] 样本合并失败或为空: {sample_path}")
        return None
    numeric_data = merged_df.select_dtypes(include=[np.number])
    if numeric_data.empty:
        print(f"[WARN] 样本无数值列: {sample_path}")
        return None
    
    # ✅ 修正：移除冗余的 'filtered' 判断
    if apply_filter:
        for col in numeric_data.columns:
            filtered_data = bandpass_filter(numeric_data[col].values)
            numeric_data[col] = filtered_data.astype(np.float32)
        
    return extract_time_features(numeric_data, time_steps=time_steps)
