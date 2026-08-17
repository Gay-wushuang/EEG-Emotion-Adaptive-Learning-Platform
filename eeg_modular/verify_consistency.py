"""Verify that all models use the same test set after the fix."""

import os
import json
import numpy as np


def check_model_metrics(output_dir: str, model_name: str):
    """Check metrics for a specific model."""
    metrics_path = os.path.join(output_dir, model_name, "metrics.json")

    if not os.path.exists(metrics_path):
        print(f"⚠️  {model_name}: metrics.json not found")
        return None

    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)

    accuracy = metrics.get("accuracy", None)
    report = metrics.get("report", {})

    # Extract test set size from report
    test_size = 0
    if "macro avg" in report:
        test_size = int(report["macro avg"].get("support", 0))
    elif "weighted avg" in report:
        test_size = int(report["weighted avg"].get("support", 0))

    return {
        "model": model_name,
        "accuracy": accuracy,
        "test_size": test_size,
        "metrics_path": metrics_path,
    }


def verify_npy_data_consistency():
    """Verify that NPY data has the correct test set size."""
    print("\n🔍 检查 NPY 数据一致性...")

    features_dir = "./features"

    if not os.path.exists(features_dir):
        print(f"❌ Features directory not found: {features_dir}")
        return False

    # Check test set size from NPY files
    y_test_files = [
        "y_test_filtered.npy",
        "y_test_powerspec.npy",
        "y_test_att.npy",
        "y_test_med.npy",
    ]

    test_sizes = []
    for filename in y_test_files:
        filepath = os.path.join(features_dir, filename)
        if os.path.exists(filepath):
            y_test = np.load(filepath)
            test_sizes.append(len(y_test))
            print(f"   ✅ {filename}: {len(y_test)} samples")
        else:
            print(f"   ⚠️  {filename}: not found")

    if len(set(test_sizes)) == 1:
        print(f"   ✅ 所有 NPY 测试集大小一致: {test_sizes[0]}")
        return True
    else:
        print(f"   ❌ NPY 测试集大小不一致: {test_sizes}")
        return False


def main():
    """Main verification function."""
    print("=" * 60)
    print("📊 验证所有模型使用相同的测试集")
    print("=" * 60)

    # First verify NPY data consistency
    npy_ok = verify_npy_data_consistency()

    print("\n🔍 检查各模型的测试集大小...")

    models = ["SVM", "MLP", "RF", "XGB", "LSTM", "CNN", "HYBRID"]
    results = []

    for model_name in models:
        result = check_model_metrics("outputs", model_name)
        if result:
            results.append(result)

    # Print summary
    print("\n" + "=" * 60)
    print("📋 模型测试集大小汇总")
    print("=" * 60)

    for result in results:
        test_size = result["test_size"]
        accuracy = result["accuracy"]

        if test_size == 81:
            status = "✅ 正确 (与 NPY 一致)"
        elif test_size == 117:
            status = "❌ 错误 (使用原始数据)"
        else:
            status = f"⚠️  未知 ({test_size})"

        print(f"{result['model']:>10}: 测试集={test_size:3d}, 准确率={accuracy:.4f} | {status}")

    print("=" * 60)

    # Check consistency
    if npy_ok:
        test_sizes = [r["test_size"] for r in results if r["test_size"] > 0]

        if len(set(test_sizes)) == 1:
            print("\n✅ 成功！所有模型使用相同的测试集大小:", test_sizes[0])
            return True
        else:
            print("\n❌ 失败！不同模型使用不同的测试集大小:", set(test_sizes))
            return False
    else:
        print("\n❌ NPY 数据不一致，请检查 time_data_preprocess 输出")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)