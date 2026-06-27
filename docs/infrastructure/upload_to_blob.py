"""
Upload script for Blob Storage.
Usage: python upload_to_blob.py
"""
import os

# ── Configuration ───────────────────────────────────────────────────────────
CONNECTION_STRING = "YOUR_CONNECTION_STRING_HERE"
CONTAINER_NAME = "data"
# ───────────────────────────────────────────────────────────────────────────────

FILES = {
    "store/legal_cases.index": r"YOUR_LOCAL_PATH\LexFind\api\data\store\legal_cases.index",
    "store/id2name.json":      r"YOUR_LOCAL_PATH\LexFind\api\data\store\id2name.json",
}

def upload():
    # Implementation depends on the blob storage provider (e.g. AWS, GCP)
    pass

if __name__ == "__main__":
    if CONNECTION_STRING == "YOUR_CONNECTION_STRING_HERE":
        print("ERROR: Set your CONNECTION_STRING in this script first.")
    else:
        upload()