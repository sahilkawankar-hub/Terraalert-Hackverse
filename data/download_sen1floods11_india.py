#!/usr/bin/env python3
"""
Download India hand-labelled tiles from the public Sen1Floods11 GCS bucket.
Does not require gsutil or GCP credentials.
"""

import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BUCKET_NAME = "sen1floods11"
API_URL = f"https://storage.googleapis.com/storage/v1/b/{BUCKET_NAME}/o"
BASE_DOWNLOAD_URL = f"https://storage.googleapis.com/{BUCKET_NAME}"

CATEGORIES = ["S2Hand", "LabelHand", "JRCWaterHand", "S1Hand"]

def get_file_list(category: str):
    prefix = f"v1.1/data/flood_events/HandLabeled/{category}/India_"
    url = f"{API_URL}?prefix={prefix}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("items", [])

def download_file(item, dest_dir: Path):
    object_name = item["name"]
    file_name = Path(object_name).name
    expected_size = int(item.get("size", 0))
    dest_path = dest_dir / file_name

    # Check if already downloaded and complete
    if dest_path.exists() and dest_path.stat().st_size == expected_size:
        return file_name, expected_size, "skipped"

    media_url = f"{BASE_DOWNLOAD_URL}/{object_name}"
    req = urllib.request.Request(media_url, headers={"User-Agent": "Mozilla/5.0"})

    temp_path = dest_dir / f"{file_name}.tmp"
    with urllib.request.urlopen(req, timeout=60) as response, open(temp_path, "wb") as out_file:
        chunk = response.read(64 * 1024)
        while chunk:
            out_file.write(chunk)
            chunk = response.read(64 * 1024)

    # Verify size
    actual_size = temp_path.stat().st_size
    if expected_size and actual_size != expected_size:
        temp_path.unlink(missing_ok=True)
        raise IOError(f"Size mismatch for {file_name}: got {actual_size}, expected {expected_size}")

    temp_path.replace(dest_path)
    return file_name, actual_size, "downloaded"

def main():
    script_dir = Path(__file__).resolve().parent
    raw_dir = script_dir / "raw"

    print(f"Target raw directory: {raw_dir}")
    total_files = 0
    total_bytes = 0

    all_tasks = []
    print("Fetching file lists from GCS...")
    for cat in CATEGORIES:
        dest_dir = raw_dir / cat
        dest_dir.mkdir(parents=True, exist_ok=True)
        items = get_file_list(cat)
        cat_bytes = sum(int(item.get("size", 0)) for item in items)
        print(f"  [{cat}] Found {len(items)} files ({cat_bytes / (1024 * 1024):.2f} MB)")
        total_files += len(items)
        total_bytes += cat_bytes
        for item in items:
            all_tasks.append((item, dest_dir))

    print(f"Total files to process: {total_files} ({total_bytes / (1024 * 1024):.2f} MB)")

    start_time = time.time()
    completed_count = 0
    downloaded_count = 0
    skipped_count = 0
    downloaded_bytes = 0

    # Concurrently download using 8 threads
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(download_file, item, dest): item["name"] for item, dest in all_tasks}
        for future in as_completed(futures):
            name = futures[future]
            try:
                fname, fsize, status = future.result()
                completed_count += 1
                if status == "downloaded":
                    downloaded_count += 1
                    downloaded_bytes += fsize
                else:
                    skipped_count += 1

                if completed_count % 20 == 0 or completed_count == total_files:
                    elapsed = time.time() - start_time
                    print(f"Progress: [{completed_count}/{total_files}] files | "
                          f"Downloaded: {downloaded_count} ({downloaded_bytes / (1024 * 1024):.1f} MB) | "
                          f"Skipped: {skipped_count} | Elapsed: {elapsed:.1f}s")
            except Exception as e:
                print(f"Error downloading {name}: {e}", file=sys.stderr)

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s!")
    print(f"Downloaded: {downloaded_count} files, Skipped: {skipped_count} files.")

if __name__ == "__main__":
    main()
