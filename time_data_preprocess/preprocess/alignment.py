import os
import pandas as pd


def load_and_align_csv(csv_path, freq='200ms'):
    """
    统一重采样到 freq（默认200ms），保证不同 CSV 时间步一致
    """
    try:
        df = pd.read_csv(csv_path)
        time_col = [c for c in df.columns if 'Time' in c][0]
        value_col = [c for c in df.columns if c != time_col][0]

        # 时间解析
        try:
            df[time_col] = pd.to_datetime(df[time_col], format='%H:%M:%S', errors='coerce')
        except:
            df[time_col] = pd.to_timedelta(df[time_col], errors='coerce')
        
        df = df.dropna(subset=[time_col]).set_index(time_col)
        # 统一重采样
        df_resampled = df.resample(freq).mean().interpolate(method='linear').ffill().bfill()
        return df_resampled[value_col]
    except Exception as e:
        print(f"[WARN] 处理失败 {csv_path}: {e}")
        return None


def merge_csvs(csv_list, freq='200ms', min_length=10):
    """
    合并多 CSV 并统一长度，返回长度一致的 DataFrame
    """
    aligned = []
    for p in csv_list:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            series = load_and_align_csv(p, freq)
            if series is not None and len(series) >= min_length:
                aligned.append(series)
    if not aligned:
        return None

    # 找到最小长度，截断所有序列
    min_len = min(len(s) for s in aligned)
    truncated = [s.iloc[:min_len] for s in aligned]

    merged = pd.concat(truncated, axis=1)
    merged = merged.interpolate(method='linear').ffill().bfill()
    return merged
