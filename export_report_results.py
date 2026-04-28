"""
export_report_results.py
========================

Exports report-ready text tables from generated outputs.

Inputs:
  - test_outputs/<DATASET>_test_outputs.npz
  - training_logs/<DATASET>_training_log.csv

Outputs:
  - report_results_text/results_summary_new_domains.csv
  - report_results_text/results_summary_new_domains.md
  - report_results_text/confusion_matrices_new_domains.csv
  - report_results_text/training_summary_new_domains.csv
  - report_results_text/original_benchmark_comparison.csv
  - report_results_text/original_benchmark_comparison.md
  - report_results_text/report_results.json

Run:
  python export_report_results.py
"""

import csv
import json
import os

import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "report_results_text")
os.makedirs(OUT_DIR, exist_ok=True)


NEW_DATASETS = {
    "SKAB": {
        "domain": "Industrial pump/valve",
        "dims": 8,
        "train_size": 9405,
        "test_output_key": "SKAB",
    },
    "TEP": {
        "domain": "Chemical plant process monitoring",
        "dims": 52,
        "train_size": 72573,
        "test_output_key": "TEP",
    },
    "GECCO": {
        "domain": "Drinking water quality IoT",
        "dims": 9,
        "train_size": 96488,
        "test_output_key": "GECCO",
    },
    "MIT-BIH": {
        "domain": "ECG arrhythmia detection",
        "dims": 2,
        "train_size": 3856518,
        "test_output_key": "MITBIH",
    },
}


ORIGINAL_PAPER_RESULTS = {
    "SMD": {"P": 89.40, "R": 95.45, "F1": 92.33},
    "MSL": {"P": 92.09, "R": 95.15, "F1": 93.59},
    "SMAP": {"P": 94.13, "R": 99.40, "F1": 96.69},
    "PSM": {"P": 96.91, "R": 98.90, "F1": 97.89},
}


def output_path(*parts):
    return os.path.join(OUT_DIR, *parts)


def load_test_output(key):
    path = os.path.join(BASE_DIR, "test_outputs", f"{key}_test_outputs.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing test output: {path}\n"
            "Run the corresponding train/test command first."
        )
    return np.load(path)


def load_training_log(key):
    path = os.path.join(BASE_DIR, "training_logs", f"{key}_training_log.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def confusion_counts(gt, pred):
    gt = gt.astype(int)
    pred = pred.astype(int)
    return {
        "TN": int(((gt == 0) & (pred == 0)).sum()),
        "FP": int(((gt == 0) & (pred == 1)).sum()),
        "FN": int(((gt == 1) & (pred == 0)).sum()),
        "TP": int(((gt == 1) & (pred == 1)).sum()),
    }


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def write_csv(path, headers, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def export_new_domain_results(report):
    headers = [
        "Dataset",
        "Domain",
        "Dims",
        "Train Size",
        "Test Size",
        "Anomaly Rate (%)",
        "Threshold",
        "Accuracy (%)",
        "Precision (%)",
        "Recall (%)",
        "F1 (%)",
    ]
    rows = []
    cm_rows = []

    for dataset, meta in NEW_DATASETS.items():
        key = meta["test_output_key"]
        output = load_test_output(key)
        gt = output["gt"].astype(int)
        pred = output["pred"].astype(int)
        cm = confusion_counts(gt, pred)
        test_size = len(gt)
        anomaly_rate = float(gt.mean()) * 100
        row = [
            dataset,
            meta["domain"],
            meta["dims"],
            meta["train_size"],
            test_size,
            f"{anomaly_rate:.3f}",
            f"{float(output['threshold']):.8g}",
            f"{float(output['accuracy']) * 100:.2f}",
            f"{float(output['precision']) * 100:.2f}",
            f"{float(output['recall']) * 100:.2f}",
            f"{float(output['f_score']) * 100:.2f}",
        ]
        rows.append(row)
        cm_rows.append([
            dataset,
            cm["TN"],
            cm["FP"],
            cm["FN"],
            cm["TP"],
            test_size,
            f"{cm['TN'] / test_size * 100:.2f}",
            f"{cm['FP'] / test_size * 100:.2f}",
            f"{cm['FN'] / test_size * 100:.2f}",
            f"{cm['TP'] / test_size * 100:.2f}",
        ])

        report["new_domains"][dataset] = {
            "domain": meta["domain"],
            "dims": meta["dims"],
            "train_size": meta["train_size"],
            "test_size": test_size,
            "anomaly_rate_percent": anomaly_rate,
            "threshold": float(output["threshold"]),
            "accuracy_percent": float(output["accuracy"]) * 100,
            "precision_percent": float(output["precision"]) * 100,
            "recall_percent": float(output["recall"]) * 100,
            "f1_percent": float(output["f_score"]) * 100,
            "confusion_matrix": cm,
        }

    write_csv(output_path("results_summary_new_domains.csv"), headers, rows)
    with open(output_path("results_summary_new_domains.md"), "w") as f:
        f.write(markdown_table(headers, rows))

    cm_headers = [
        "Dataset",
        "TN",
        "FP",
        "FN",
        "TP",
        "Total",
        "TN (%)",
        "FP (%)",
        "FN (%)",
        "TP (%)",
    ]
    write_csv(output_path("confusion_matrices_new_domains.csv"), cm_headers, cm_rows)


def export_training_summary(report):
    headers = [
        "Dataset",
        "Epochs Logged",
        "Final Train Loss",
        "Final Validation Loss",
        "Best Validation Loss",
        "Early Stopped",
    ]
    rows = []
    for dataset, meta in NEW_DATASETS.items():
        key = meta["test_output_key"]
        log_rows = load_training_log(key)
        if not log_rows:
            rows.append([dataset, "missing", "", "", "", ""])
            report["training"][dataset] = {"status": "missing"}
            continue

        train_losses = [float(row["train_loss"]) for row in log_rows]
        vali_losses = [float(row["vali_loss"]) for row in log_rows]
        early_stopped = any(int(row.get("early_stop", 0)) for row in log_rows)
        rows.append([
            dataset,
            len(log_rows),
            f"{train_losses[-1]:.6f}",
            f"{vali_losses[-1]:.6f}",
            f"{min(vali_losses):.6f}",
            "yes" if early_stopped else "no",
        ])
        report["training"][dataset] = {
            "epochs_logged": len(log_rows),
            "final_train_loss": train_losses[-1],
            "final_validation_loss": vali_losses[-1],
            "best_validation_loss": min(vali_losses),
            "early_stopped": early_stopped,
        }

    write_csv(output_path("training_summary_new_domains.csv"), headers, rows)


def export_original_benchmark_comparison(report):
    headers = [
        "Dataset",
        "Paper Precision (%)",
        "Our Precision (%)",
        "Delta Precision",
        "Paper Recall (%)",
        "Our Recall (%)",
        "Delta Recall",
        "Paper F1 (%)",
        "Our F1 (%)",
        "Delta F1",
    ]
    rows = []

    for dataset, paper in ORIGINAL_PAPER_RESULTS.items():
        try:
            output = load_test_output(dataset)
        except FileNotFoundError as exc:
            print(f"Skipping original benchmark comparison for {dataset}: {exc}")
            report["original_benchmarks"][dataset] = {"status": "missing"}
            continue
        ours = {
            "P": float(output["precision"]) * 100,
            "R": float(output["recall"]) * 100,
            "F1": float(output["f_score"]) * 100,
        }
        rows.append([
            dataset,
            f"{paper['P']:.2f}",
            f"{ours['P']:.2f}",
            f"{ours['P'] - paper['P']:+.2f}",
            f"{paper['R']:.2f}",
            f"{ours['R']:.2f}",
            f"{ours['R'] - paper['R']:+.2f}",
            f"{paper['F1']:.2f}",
            f"{ours['F1']:.2f}",
            f"{ours['F1'] - paper['F1']:+.2f}",
        ])
        report["original_benchmarks"][dataset] = {
            "paper": paper,
            "reproduced": ours,
            "delta": {metric: ours[metric] - paper[metric] for metric in ["P", "R", "F1"]},
        }

    if rows:
        write_csv(output_path("original_benchmark_comparison.csv"), headers, rows)
        with open(output_path("original_benchmark_comparison.md"), "w") as f:
            f.write(markdown_table(headers, rows))


def main():
    report = {
        "new_domains": {},
        "training": {},
        "original_benchmarks": {},
        "notes": [
            "New-domain metrics are loaded from test_outputs/*.npz.",
            "Training summaries are loaded from training_logs/*.csv when available.",
            "Original paper benchmark values are from Anomaly Transformer ICLR 2022 Table 1.",
        ],
    }

    export_new_domain_results(report)
    export_training_summary(report)
    export_original_benchmark_comparison(report)

    with open(output_path("report_results.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved report text outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
