import os
import shutil
import glob

import gdown

# Shared Google Drive folder ID containing the OULAD datasets in CSV format.
GDRIVE_FOLDER_ID = "1yLdE4Za-5mb5lWxZgjaa4CV85YKVc-Rx"

DATASET_FILES = [
    "studentInfo.csv",
    "studentAssessment.csv",
    "studentVle.csv",
    "studentRegistration.csv",
    "assessments.csv",
    "courses.csv",
    "vle.csv",
]

# Anything smaller than this is treated as a failed/corrupt download.
MIN_VALID_FILE_SIZE_BYTES = 100


def _get_data_dir() -> str:
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.abspath(data_dir)


def _st_or_print(msg: str, kind: str = "info"):
    """Show message via Streamlit if available, otherwise print."""
    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            if kind == "info":
                st.info(msg)
            elif kind == "success":
                st.success(msg)
            elif kind == "error":
                st.error(msg)
            return
    except Exception:
        pass
    print(msg)


def _is_valid_file(path: str) -> bool:
    """A file 'counts' only if it exists and isn't empty/truncated."""
    return os.path.exists(path) and os.path.getsize(path) >= MIN_VALID_FILE_SIZE_BYTES


def _find_missing(data_dir: str) -> list:
    return [f for f in DATASET_FILES if not _is_valid_file(os.path.join(data_dir, f))]


def _flatten_subfolders(data_dir: str) -> None:
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if not os.path.isdir(item_path):
            continue

        print(f"[PRISM] Found subfolder '{item}', moving files up…")
        for fname in os.listdir(item_path):
            src = os.path.join(item_path, fname)
            dst = os.path.join(data_dir, fname)

            if os.path.isdir(src):
                continue

            # Overwrite existing file at destination if it's missing/invalid,
            # so a partial earlier download doesn't block the fresh copy.
            if os.path.exists(dst):
                if _is_valid_file(dst):
                    os.remove(src)  # keep the good copy already in place
                    continue
                os.remove(dst)

            shutil.move(src, dst)
            print(f"[PRISM] Moved {fname} -> data/")

        # Clean up leftover subfolder (and any stray "(1)" duplicates inside it)
        try:
            shutil.rmtree(item_path)
        except OSError as e:
            print(f"[PRISM] Could not remove subfolder '{item}': {e}")

    # gdown sometimes saves collisions as "name (1).csv" in data_dir itself
    for dup in glob.glob(os.path.join(data_dir, "* (*).csv")):
        os.remove(dup)
        print(f"[PRISM] Removed duplicate file: {os.path.basename(dup)}")


def ensure_datasets_available(data_dir: str | None = None) -> bool:
    if data_dir is None:
        data_dir = _get_data_dir()

    os.makedirs(data_dir, exist_ok=True)

    missing = _find_missing(data_dir)

    if not missing:
        print(f"[PRISM] All datasets found in {data_dir}")
        return True

    print(f"[PRISM] {len(missing)} file(s) missing or invalid: {missing}")
    print(f"[PRISM] Downloading from Google Drive folder: {GDRIVE_FOLDER_ID}")
    _st_or_print(
        f"📥 Downloading {len(missing)} dataset files"
    )

    url = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}"

    try:
        gdown.download_folder(
            url,
            output=data_dir,
            quiet=False,
            use_cookies=False,
        )
    except Exception as e:
        print(f"[PRISM] gdown error: {e}")
        _st_or_print(f"❌ Download failed: {e}", kind="error")
        return False

    _flatten_subfolders(data_dir)

    # Verify
    still_missing = _find_missing(data_dir)

    if not still_missing:
        print("[PRISM] All datasets ready.")
        _st_or_print("✅ All datasets downloaded successfully.", kind="success")
        return True
    else:
        print(f"[PRISM] Still missing after download: {still_missing}")
        _st_or_print(
            f"❌ Could not find or validate: {still_missing}. "
            "Make sure the Google Drive folder is shared as 'Anyone with the link'.",
            kind="error"
        )
        return False