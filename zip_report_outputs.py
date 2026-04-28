"""
zip_report_outputs.py
=====================

Creates one downloadable ZIP archive containing the report outputs.

It prefers the latest timestamped package under:
  REPORT_RESULTS/report_package_*/

If no package exists yet, it zips the current output folders directly.

Output:
  REPORT_RESULTS/report_outputs_latest.zip

Run:
  python zip_report_outputs.py
"""

import os
import shutil
from glob import glob


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_ROOT = os.path.join(BASE_DIR, "REPORT_RESULTS")
ZIP_BASE = os.path.join(REPORT_ROOT, "report_outputs_latest")
ZIP_PATH = f"{ZIP_BASE}.zip"


def latest_report_package():
    packages = sorted(
        glob(os.path.join(REPORT_ROOT, "report_package_*")),
        key=os.path.getmtime,
        reverse=True,
    )
    return packages[0] if packages else None


def zip_directory(source_dir):
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    archive_path = shutil.make_archive(ZIP_BASE, "zip", source_dir)
    return archive_path


def build_fallback_folder():
    fallback_dir = os.path.join(REPORT_ROOT, "report_outputs_fallback")
    if os.path.exists(fallback_dir):
        shutil.rmtree(fallback_dir)
    os.makedirs(fallback_dir, exist_ok=True)

    for folder in [
        "report_figures_real",
        "report_results_text",
        "original_benchmark_comparison",
        "test_outputs",
        "training_logs",
    ]:
        src = os.path.join(BASE_DIR, folder)
        if os.path.exists(src):
            shutil.copytree(src, os.path.join(fallback_dir, folder), dirs_exist_ok=True)

    return fallback_dir


def main():
    os.makedirs(REPORT_ROOT, exist_ok=True)

    package_dir = latest_report_package()
    if package_dir:
        source_dir = package_dir
        print(f"Zipping latest report package: {os.path.relpath(package_dir, BASE_DIR)}")
    else:
        source_dir = build_fallback_folder()
        print("No timestamped package found; zipping current output folders instead.")

    archive_path = zip_directory(source_dir)
    print(f"Created ZIP archive: {archive_path}")

    # Helpful for Colab users.
    print("\nIn Colab, download it with:")
    print("from google.colab import files")
    print(f"files.download('{os.path.relpath(archive_path, BASE_DIR)}')")


if __name__ == "__main__":
    main()
