"""Download GLAMI-1M 800px archives and retain only dresses training images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import subprocess
import zipfile
from pathlib import Path, PurePosixPath


RECORD_URL = "https://zenodo.org/records/7338792/files"
ARCHIVE_MD5 = {
    0: "9d7aa43dc315a5d567389e68e4e50ccb", 1: "36d1ffbef560b707f815faad98b039f1",
    2: "47598ccfddcfb3ed9a00e851aa1165b9", 3: "d92c0a41e093bd07fd59a6a80c65cf25",
    4: "0134506e9d47ddf2e4c7b286b68e3c09", 5: "19dcb70eb67c4ecb560d1ba6580d0b3c",
    6: "33cddce2f4c3df3be3ae8c0e7fb841ed", 7: "2ed26dcc539e1e57af4305e9b1f22686",
    8: "6895f21aa72a790fb8ce9556637f42d7", 9: "235f5003edaf7b45a7d99b38d814311e",
    10: "cec58af85ff2cc9eefd9ba20ebdd334e",
}
EXPECTED_DRESS_ROWS = 29_350
EXPECTED_DRESS_IMAGES = 26_326


def archive_name(part: int) -> str:
    return f"GLAMI-1M-dataset-800px--{part}.zip"


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(part: int, archive_dir: Path) -> Path:
    destination = archive_dir / archive_name(part)
    expected_md5 = ARCHIVE_MD5[part]
    if destination.exists():
        if md5(destination) == expected_md5:
            return destination
        raise RuntimeError(f"Checksum mismatch for existing archive: {destination}")

    temporary = destination.with_suffix(".zip.part")
    downloader = os.environ.get("ARIA2C") or shutil.which("aria2c")
    if downloader is None:
        raise RuntimeError("aria2c is required for resumable parallel archive downloads")
    action = "Resuming" if temporary.exists() else "Downloading"
    print(f"{action} {destination.name} with 16 connections", flush=True)
    subprocess.run(
        [downloader, "--continue=true", "--max-connection-per-server=16", "--split=16",
         "--min-split-size=16M", "--file-allocation=none", "--summary-interval=10",
         "--max-tries=5", "--retry-wait=10", "--dir", str(archive_dir), "--out",
         temporary.name, f"{RECORD_URL}/{destination.name}?download=1"],
        check=True,
    )
    actual_md5 = md5(temporary)
    if actual_md5 != expected_md5:
        raise RuntimeError(f"Checksum mismatch for {destination.name}: {actual_md5} != {expected_md5}")
    temporary.replace(destination)
    return destination


def find_train_csv(archive_dir: Path, metadata_dir: Path) -> Path:
    destination = metadata_dir / "GLAMI-1M-train.csv"
    if destination.exists():
        return destination
    for part in sorted(ARCHIVE_MD5):
        archive_path = download_archive(part, archive_dir)
        with zipfile.ZipFile(archive_path) as archive:
            matches = [entry for entry in archive.namelist() if PurePosixPath(entry).name == destination.name]
            if not matches:
                archive_path.unlink()
                continue
            if len(matches) != 1:
                raise RuntimeError(f"Expected one train CSV in {archive_path}, found {matches}")
            metadata_dir.mkdir(parents=True, exist_ok=True)
            with archive.open(matches[0]) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)
            return destination
    raise RuntimeError("GLAMI-1M-train.csv was not found in the 800px archives")


def build_dresses_csv(train_csv: Path, output_csv: Path) -> set[str]:
    if output_csv.exists():
        with output_csv.open("r", encoding="utf-8", newline="") as source:
            dresses = list(csv.DictReader(source))
    else:
        with train_csv.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            required = {"image_id", "item_id", "category_name"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise RuntimeError(f"Unexpected train CSV columns: {reader.fieldnames}")
            dresses = [row for row in reader if row["category_name"] == "dresses"]
            fieldnames = reader.fieldnames
        with output_csv.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(dresses)
    image_ids = {row["image_id"] for row in dresses}
    if len(dresses) != EXPECTED_DRESS_ROWS or len(image_ids) != EXPECTED_DRESS_IMAGES:
        raise RuntimeError(f"Unexpected dresses subset: rows={len(dresses)}, unique_image_ids={len(image_ids)}")
    return image_ids


def extract_dress_images(archive_dir: Path, image_dir: Path, image_ids: set[str]) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    wanted = {f"{image_id}.jpg" for image_id in image_ids}
    extracted = {path.name for path in image_dir.glob("*.jpg") if path.name in wanted}
    for part in sorted(ARCHIVE_MD5):
        archive_path = download_archive(part, archive_dir)
        print(f"Extracting dresses from {archive_path.name}", flush=True)
        with zipfile.ZipFile(archive_path) as archive:
            for entry in archive.infolist():
                filename = PurePosixPath(entry.filename).name
                if entry.is_dir() or filename not in wanted or filename in extracted:
                    continue
                target = image_dir / filename
                temporary = target.with_suffix(".jpg.part")
                with archive.open(entry) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output)
                temporary.replace(target)
                extracted.add(filename)
        archive_path.unlink()
    missing = wanted - extracted
    if missing:
        raise RuntimeError(f"Missing {len(missing)} dress images after extraction: {', '.join(sorted(missing)[:10])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    data_root = parser.parse_args().data_root.resolve()
    archive_dir = data_root.parent.parent / "cache" / "glami_archives"
    metadata_dir = data_root / "GLAMI-1M-dataset"
    archive_dir.mkdir(parents=True, exist_ok=True)
    image_ids = build_dresses_csv(
        find_train_csv(archive_dir, metadata_dir), data_root / "GLAMI-1M-dresses-train.csv"
    )
    extract_dress_images(archive_dir, data_root / "images-800px", image_ids)
    archive_dir.rmdir()
    print(f"Complete: {len(image_ids)} unique dresses images", flush=True)


if __name__ == "__main__":
    main()
