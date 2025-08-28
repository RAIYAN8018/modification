"""
app.py
Streamlit UI for validating/correcting Q/A JSON files (multi-file + ZIP).
- No translation: only original-language outputs are produced.
- No expanders/wrappers around processed outputs.
"""

import json
import io
import os
import zipfile
import streamlit as st
from typing import Any, Dict, List, Tuple
from chatbot_setup import process_records_parallel, QuotaExceededException

st.set_page_config(page_title="Q/A Validator", layout="wide")
st.title("JSON Question-Answer Validator & Fixer (Multi-file + ZIP)")

# ---------------------------
# Config (optional)
# ---------------------------
SAVE_TO_LOCAL_DOWNLOAD_DIR = False
LOCAL_DOWNLOAD_DIR = "download"

# ---------------------------
# Helpers
# ---------------------------
def _is_record_like(x: Any) -> bool:
    return isinstance(x, dict)

def _flatten_uploaded_json(obj: Any) -> List[Dict[str, Any]]:
    """
    Accepts:
      - list of dicts -> returns as-is
      - dict of records -> returns list(dict.values())
      - single dict (single record) -> [dict]
    """
    if isinstance(obj, list):
        return [r for r in obj if _is_record_like(r)]
    if isinstance(obj, dict):
        vals = list(obj.values())
        if vals and all(isinstance(v, dict) for v in vals):
            return vals
        return [obj]
    return []

def _is_new_upload(prev_names: Tuple[str, ...], new_files) -> bool:
    new_names = tuple(sorted([f.name for f in (new_files or [])]))
    return new_names != prev_names

def _derive_output_name(uploaded_name: str) -> str:
    return uploaded_name if uploaded_name.lower().endswith(".json") else f"{uploaded_name}.json"

def _ensure_local_download_dir():
    if SAVE_TO_LOCAL_DOWNLOAD_DIR:
        os.makedirs(LOCAL_DOWNLOAD_DIR, exist_ok=True)

def _save_json_local(filename: str, data_obj: Any):
    if not SAVE_TO_LOCAL_DOWNLOAD_DIR:
        return
    _ensure_local_download_dir()
    path = os.path.join(LOCAL_DOWNLOAD_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data_obj, f, indent=2, ensure_ascii=False)

def _unique_name(desired: str, existing: set) -> str:
    if desired not in existing:
        return desired
    stem, ext = (desired[:-5], ".json") if desired.lower().endswith(".json") else (desired, "")
    i = 1
    while True:
        candidate = f"{stem} ({i}){ext}"
        if candidate not in existing:
            return candidate
        i += 1

def _load_json_from_bytes(b: bytes, source_name: str) -> List[Dict[str, Any]]:
    try:
        text = b.decode("utf-8-sig", errors="replace")  # strip BOM if present
        data = json.loads(text)
        flattened = _flatten_uploaded_json(data)
        if not flattened:
            st.warning(f"⚠️ {source_name}: no valid records found (skipped).")
        return flattened
    except json.JSONDecodeError as e:
        st.error(f"❌ {source_name}: invalid JSON — {str(e)}")
        return []
    except Exception as e:
        st.error(f"❌ {source_name}: error loading JSON — {str(e)}")
        return []

# ---------------------------
# Session state
# ---------------------------
for key, default in [
    ("original_data_by_file", {}),   # { output_filename: [records] }
    ("processed_data_by_file", {}),  # { output_filename: [processed] }
    ("last_uploaded_names", tuple()),
    ("quota_exceeded", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

def display_quota_exceeded_message():
    st.error("🚨 **QUOTA EXCEEDED**")
    st.write(
        "OpenAI API usage limit exceeded. "
        "Please add credits / verify billing, then click **Reset Session** and try again."
    )
    if st.button("🔄 Reset Session"):
        st.session_state["processed_data_by_file"] = {}
        st.session_state["quota_exceeded"] = None
        st.rerun()

# ---------------------------
# Upload (multi-file JSON + ZIP)
# ---------------------------
uploaded_files = st.file_uploader(
    "Upload one or more JSON or ZIP files",
    type=["json", "zip"],
    accept_multiple_files=True
)

if _is_new_upload(st.session_state["last_uploaded_names"], uploaded_files):
    try:
        original_map = {}
        names = []
        existing_names = set()

        for uf in (uploaded_files or []):
            names.append(uf.name)

            if uf.name.lower().endswith(".json"):
                data_bytes = uf.getvalue()
                flattened = _load_json_from_bytes(data_bytes, uf.name)
                if flattened:
                    out_name = _unique_name(_derive_output_name(uf.name), existing_names)
                    existing_names.add(out_name)
                    original_map[out_name] = flattened

            elif uf.name.lower().endswith(".zip"):
                try:
                    zbytes = uf.getvalue()
                    with zipfile.ZipFile(io.BytesIO(zbytes), "r") as zf:
                        members = [m for m in zf.infolist()
                                   if not m.is_dir() and m.filename.lower().endswith(".json")]
                        if not members:
                            st.warning(f"⚠️ {uf.name}: no JSON files found inside (skipped).")
                        for zi in members:
                            try:
                                with zf.open(zi, "r") as f:
                                    content = f.read()
                                entry_name_base = os.path.basename(zi.filename) or zi.filename
                                candidate_name = _derive_output_name(entry_name_base)
                                out_name = _unique_name(candidate_name, existing_names)
                                flattened = _load_json_from_bytes(content, f"{uf.name}::{zi.filename}")
                                if flattened:
                                    existing_names.add(out_name)
                                    original_map[out_name] = flattened
                            except Exception as e:
                                st.error(f"❌ {uf.name}::{zi.filename}: error reading entry — {str(e)}")
                except zipfile.BadZipFile:
                    st.error(f"❌ {uf.name}: not a valid ZIP archive.")
                except Exception as e:
                    st.error(f"❌ {uf.name}: error opening ZIP — {str(e)}")
            else:
                st.warning(f"⚠️ Unsupported file type: {uf.name} (skipped).")

        st.session_state["original_data_by_file"] = original_map
        st.session_state["processed_data_by_file"] = {}
        st.session_state["quota_exceeded"] = None
        st.session_state["last_uploaded_names"] = tuple(sorted(names))

        if original_map:
            st.success(f"✅ Loaded {len(original_map)} JSON file(s) (including ZIP contents).")
        else:
            st.info("Please upload at least one valid JSON (direct or inside a ZIP).")

    except Exception as e:
        st.error(f"❌ Error during upload: {str(e)}")

# Quota handling
if st.session_state.get("quota_exceeded"):
    display_quota_exceeded_message()
    st.stop()

# ---------------------------
# Optional Originals (inline, no expanders)
# ---------------------------
if st.session_state["original_data_by_file"]:
    show_originals = st.checkbox("Show uploaded JSON contents (originals)", value=False)
    if show_originals:
        st.subheader("Original Data (by file)")
        for fname, records in st.session_state["original_data_by_file"].items():
            st.markdown(f"**{fname} — {len(records)} record(s)**")
            st.json(records)

# ---------------------------
# Processing (all files)
# ---------------------------
if st.button("🚀 Start Processing All", disabled=not st.session_state["original_data_by_file"]):
    if not st.session_state["processed_data_by_file"]:
        st.info("Starting validation & correction...")
        file_names = list(st.session_state["original_data_by_file"].keys())
        total_files = len(file_names)

        file_progress = st.progress(0.0)
        status_text = st.empty()

        try:
            for idx, fname in enumerate(file_names, start=1):
                records = st.session_state["original_data_by_file"][fname]
                status_text.write(f"Processing file {idx}/{total_files}: **{fname}** ({len(records)} records)")

                try:
                    processed = process_records_parallel(records, max_workers=5)
                    st.session_state["processed_data_by_file"][fname] = processed
                except QuotaExceededException as e:
                    st.session_state["quota_exceeded"] = str(e)
                    st.rerun()

                _save_json_local(fname, processed)

                file_progress.progress(idx / max(total_files, 1))

            status_text.write("✅ All files processed successfully!")
            st.success("Done!")

        except QuotaExceededException as e:
            st.session_state["quota_exceeded"] = str(e)
            st.rerun()
    else:
        st.info("Already processed. See results below.")

# ---------------------------
# Results & Downloads (inline, no expanders)
# ---------------------------
if st.session_state["processed_data_by_file"]:
    st.subheader("Processed Data (by file)")

    for fname, processed in st.session_state["processed_data_by_file"].items():
        st.markdown(f"### {fname} — Processed")
        st.json(processed)  # inline, no wrapper
        st.download_button(
            f"📄 Download {fname}",
            data=json.dumps(processed, indent=2, ensure_ascii=False),
            file_name=fname,
            mime="application/json",
        )
        st.markdown("---")

    # Build a ZIP containing all outputs
    st.subheader("Download All Results as ZIP")
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, processed in st.session_state["processed_data_by_file"].items():
            zf.writestr(fname, json.dumps(processed, indent=2, ensure_ascii=False))

    zip_buf.seek(0)
    st.download_button(
        "⬇️ Download All (ZIP)",
        data=zip_buf.getvalue(),
        file_name="corrected_batch.zip",
        mime="application/zip",
        help="A zip containing all corrected JSONs (original languages preserved)"
    )

# ---------------------------
# Footer
# ---------------------------
if not st.session_state["original_data_by_file"]:
    st.info("Upload `.json` files or a `.zip` of `.json` files, then click **Start Processing All**.")
