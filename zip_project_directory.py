"""
zip_project_directory.py
========================

Creates a ZIP archive of the entire project directory for backup/download.

Output:
  PROJECT_ARCHIVE/Anomaly-Transformer-robtest_full_project.zip

The archive excludes common local/cache folders that should not be downloaded:
  .git, venv, __pycache__, .ipynb_checkpoints, .mplconfig

Run:
  python zip_project_directory.py
"""

import os
import zipfile


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_NAME = os.path.basename(BASE_DIR.rstrip(os.sep))
OUT_DIR = os.path.join(BASE_DIR, "PROJECT_ARCHIVE")
ZIP_PATH = os.path.join(OUT_DIR, f"{PROJECT_NAME}_full_project.zip")

EXCLUDE_DIRS = {
    ".git",
    "venv",
    "__pycache__",
    ".ipynb_checkpoints",
    ".mplconfig",
    "PROJECT_ARCHIVE",
}

EXCLUDE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".DS_Store",
}


def should_exclude(path):
    rel_parts = os.path.relpath(path, BASE_DIR).split(os.sep)
    if any(part in EXCLUDE_DIRS for part in rel_parts):
        return True
    return any(path.endswith(suffix) for suffix in EXCLUDE_SUFFIXES)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

    file_count = 0
    total_bytes = 0
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for root, dirs, files in os.walk(BASE_DIR):
            dirs[:] = [d for d in dirs if not should_exclude(os.path.join(root, d))]
            for filename in files:
                file_path = os.path.join(root, filename)
                if should_exclude(file_path):
                    continue
                arcname = os.path.relpath(file_path, BASE_DIR)
                archive.write(file_path, arcname)
                file_count += 1
                total_bytes += os.path.getsize(file_path)

    print(f"Created full project archive: {ZIP_PATH}")
    print(f"Files included: {file_count:,}")
    print(f"Uncompressed size included: {total_bytes / (1024 ** 2):.2f} MB")
    print("\nIn Colab, download it with:")
    print("from google.colab import files")
    print(f"files.download('{os.path.relpath(ZIP_PATH, BASE_DIR)}')")


if __name__ == "__main__":
    main()
