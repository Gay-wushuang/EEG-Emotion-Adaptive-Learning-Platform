import os
import numpy as np
import joblib


def save_numpy(data, file_path):
    """
    保存 numpy 数组到文件
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.save(file_path, data)


def save_joblib(obj, file_path):
    """
    保存对象到 joblib 文件
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(obj, file_path)
