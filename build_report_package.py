"""
build_report_package.py
=======================

Builds a timestamped report package from the generated model outputs.

This script does not train or test models. Instead, it expects that the notebook
or shell scripts have just generated:

  test_outputs/*.npz
  training_logs/*.csv

It then:
  1. Verifies the required generated result files exist.
  2. Runs figure generation from those outputs.
  3. Runs text/table export from those outputs.
  4. Copies figures, CSV/Markdown/JSON tables, and raw generated outputs into a
     timestamped package folder.

Output:
  REPORT_RESULTS/report_package_YYYYMMDD_HHMMSS/

Run:
  python build_report_package.py
"""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.join(BASE_DIR, "REPORT_RESULTS")

NEW_DOMAIN_KEYS = ["SKAB", "TEP", "GECCO", "MITBIH"]
ORIGINAL_KEYS = ["SMD", "MSL", "SMAP", "PSM"]


def path(*parts):
    return os.path.join(BASE_DIR, *parts)


def require_files(keys, folder, suffix, required=True):
    files = {}
    missing = []
    for key in keys:
        file_path = path(folder, f"{key}_{suffix}")
        if os.path.exists(file_path):
            files[key] = file_path
        else:
            missing.append(file_path)
    if missing and required:
        message = "\n".join(missing)
        raise FileNotFoundError(f"Missing required generated files:\n{message}")
    return files, missing


def run_script(script_name):
    print(f"\nRunning {script_name}...")
    subprocess.run([sys.executable, script_name], cwd=BASE_DIR, check=True)


def copy_tree_if_exists(src, dst):
    if os.path.exists(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)


def copy_file_if_exists(src, dst):
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


def file_info(file_path):
    stat = os.stat(file_path)
    return {
        "path": os.path.relpath(file_path, BASE_DIR),
        "size_bytes": stat.st_size,
        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
    }


def main():
    os.makedirs(PACKAGE_ROOT, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_dir = os.path.join(PACKAGE_ROOT, f"report_package_{run_id}")
    os.makedirs(package_dir, exist_ok=True)

    required_test_outputs, _ = require_files(
        NEW_DOMAIN_KEYS,
        "test_outputs",
        "test_outputs.npz",
        required=True,
    )
    required_training_logs, _ = require_files(
        NEW_DOMAIN_KEYS,
        "training_logs",
        "training_log.csv",
        required=True,
    )
    original_outputs, missing_original = require_files(
        ORIGINAL_KEYS,
        "test_outputs",
        "test_outputs.npz",
        required=False,
    )

    run_script("generate_report_figures_real.py")
    run_script("export_report_results.py")

    # Original comparison figures are only possible if original benchmark outputs exist.
    if not missing_original:
        run_script("generate_original_benchmark_comparison.py")
    else:
        print("\nSkipping original benchmark figure comparison.")
        print("Missing original benchmark outputs:")
        for missing in missing_original:
            print(f"  {os.path.relpath(missing, BASE_DIR)}")

    copy_tree_if_exists(path("report_figures_real"), os.path.join(package_dir, "report_figures_real"))
    copy_tree_if_exists(path("report_results_text"), os.path.join(package_dir, "report_results_text"))
    copy_tree_if_exists(path("test_outputs"), os.path.join(package_dir, "test_outputs"))
    copy_tree_if_exists(path("training_logs"), os.path.join(package_dir, "training_logs"))
    if os.path.exists(path("original_benchmark_comparison")):
        copy_tree_if_exists(
            path("original_benchmark_comparison"),
            os.path.join(package_dir, "original_benchmark_comparison"),
        )

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "required_new_domain_test_outputs": {
            key: file_info(file_path) for key, file_path in required_test_outputs.items()
        },
        "required_new_domain_training_logs": {
            key: file_info(file_path) for key, file_path in required_training_logs.items()
        },
        "optional_original_benchmark_test_outputs": {
            key: file_info(file_path) for key, file_path in original_outputs.items()
        },
        "missing_original_benchmark_test_outputs": [
            os.path.relpath(file_path, BASE_DIR) for file_path in missing_original
        ],
    }

    with open(os.path.join(package_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nReport package created:")
    print(package_dir)


if __name__ == "__main__":
    main()
