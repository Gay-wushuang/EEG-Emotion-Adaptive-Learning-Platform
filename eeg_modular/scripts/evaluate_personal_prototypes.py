"""Strict validation-tuned personal prototype calibration for three EEG classes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from evaluate_cnn_head_fewshot import split_subject_samples


def centroids(x, y):
    return np.stack([x[y == label].mean(axis=0) for label in range(3)])


def classify(x, centers, metric):
    if metric == "cosine":
        xn = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
        cn = centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8)
        return (xn @ cn.T).argmax(axis=1)
    distances = ((x[:, None, :] - centers[None, :, :]) ** 2).mean(axis=2)
    return distances.argmin(axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-file", type=Path, default=Path("outputs_v3/nonlinear_tree_v1/X_nonlinear.npy"))
    parser.add_argument("--data-dir", type=Path, default=Path("features_v3_12000"))
    parser.add_argument("--splits-dir", type=Path, default=Path("splits_v3_12000"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_v3/personal_prototypes_v1"))
    parser.add_argument("--shots", type=int, default=3)
    args = parser.parse_args()
    x = np.load(args.feature_file); y = np.load(args.data_dir / "y.npy"); groups = np.load(args.data_dir / "groups.npy")
    manifest = json.loads((args.splits_dir / "manifest.json").read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, choices = [], []
    for split_name in manifest["files"]:
        split = json.loads((args.splits_dir / split_name).read_text(encoding="utf-8"))
        train=np.asarray(split["train_indices"]); val=np.asarray(split["val_indices"]); test=np.asarray(split["test_indices"])
        run_seed=int(split["seed"])*100+int(split["fold"])
        options=[]
        for k in (20, 40, 80, "all"):
            scaler=StandardScaler().fit(x[train]); z=scaler.transform(x)
            selector=SelectKBest(f_classif,k=k).fit(z[train],y[train]); q=selector.transform(z)
            source=centroids(q[train],y[train])
            for alpha in (0.25,0.5,0.75,1.0):
                for metric in ("euclidean","cosine"):
                    truth=[]; predicted=[]
                    for subject in np.unique(groups[val]):
                        selected=split_subject_samples(val,y,groups,subject,args.shots,run_seed)
                        if selected is None: continue
                        calibration,evaluation=selected
                        personal=centroids(q[calibration],y[calibration]); centers=(1-alpha)*source+alpha*personal
                        truth.extend(y[evaluation]); predicted.extend(classify(q[evaluation],centers,metric))
                    score=f1_score(truth,predicted,average="macro")
                    options.append((score,k,alpha,metric))
        _,best_k,best_alpha,best_metric=max(options,key=lambda item:item[0])
        development=np.concatenate([train,val]); scaler=StandardScaler().fit(x[development]); z=scaler.transform(x)
        selector=SelectKBest(f_classif,k=best_k).fit(z[development],y[development]); q=selector.transform(z)
        source=centroids(q[development],y[development])
        choices.append({"seed":split["seed"],"fold":split["fold"],"k":best_k,"alpha":best_alpha,"metric":best_metric})
        for subject in np.unique(groups[test]):
            calibration,evaluation=split_subject_samples(test,y,groups,subject,args.shots,run_seed)
            personal=centroids(q[calibration],y[calibration]); centers=(1-best_alpha)*source+best_alpha*personal
            zero=classify(q[evaluation],source,best_metric); adapted=classify(q[evaluation],centers,best_metric)
            rows.append({"seed":split["seed"],"fold":split["fold"],"subject_id":int(subject),"shots_per_class":args.shots,
                "evaluation_samples":len(evaluation),"zero_shot_accuracy":accuracy_score(y[evaluation],zero),
                "calibrated_accuracy":accuracy_score(y[evaluation],adapted),"zero_shot_macro_f1":f1_score(y[evaluation],zero,average="macro"),
                "calibrated_macro_f1":f1_score(y[evaluation],adapted,average="macro")})
        print(f"seed={split['seed']} fold={split['fold']} k={best_k} alpha={best_alpha} metric={best_metric}",flush=True)
    for filename,data in (("subject_metrics.csv",rows),("selected_hyperparameters.csv",choices)):
        with (args.output_dir/filename).open("w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)

if __name__ == "__main__": main()
