"""Download GLAMI-1M 800px archives and retain only dresses training images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


RECORD_URL = "https://zenodo.org/records/7338792/files"
ARCHIVE_MD5 = {
    0: "9d7aa43dc315a5d567389e68e4e50ccb",
    1: "36d1ffbef560b707f815faad98b039f1",
    2: "47598ccfddcfb3ed9a00e851aa1165b9",
    3: "d92c0a41e093bd07fd59a6a80c65cf25",
    4: "0134506e9d47ddf2e4c7b286b68e3c09",
    5: "19dcb70eb67c4ecb560d1ba6580d0b3c",
    6: "33cddce2f4c3df3be3ae8c0e7fb841ed",
    7: "2ed26dcc539e1e57af4305e9b1f22686",
    8: "6895f21aa72a790fb8ce9556637f42d7",
    9: "235f5003edaf7b45a7d99b38d814311e",
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
    if temporary.exists():
        raise RuntimeError(f"Remove incomplete archive before retrying: {temporary}")

    url = f"{RECORD_URL}/{destination.name}?download=1"
    print(f"Downloading {destination.name}")
    try:
        with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    actual_md5 = md5(temporary)
    if actual_md5 != expected_md5:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {destination.name}: {actual_md5} != {expected_md5}"
        )
    temporary.replace(destination)
    return destination


def find_train_csv(archive_dir: Path, metadata_dir: Path) -> Path:
    destination = metadata_dir / "GLAMI-1M-train.csv"
    if destination.exists():
        return destination

    for part in sorted(ARCHIVE_MD5):
        archive_path = download_archive(part, archive_dir)
        with zipfile.ZipFile(archive_path) as archive:
            matches = [
                entry
                for entry in archive.namelist()
                if PurePosixPath(entry).name == "GLAMI-1M-train.csv"
            ]
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
    with train_csv.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        required_columns = {"image_id", "item_id", "category_name"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise RuntimeError(f"Unexpected train CSV columns: {reader.fieldnames}")
        dresses = [row for row in reader if row["category_name"] == "dresses"]

    image_ids = {row["image_id"] for row in dresses}
    if len(dresses) != EXPECTED_DRESS_ROWS or len(image_ids) != EXPECTED_DRESS_IMAGES:
        raise RuntimeError(
            "Unexpected dresses subset size: "
            f"rows={len(dresses)}, unique_image_ids={len(image_ids)}"
        )

    if output_csv.exists():
        raise RuntimeError(f"Refusing to overwrite existing filtered CSV: {output_csv}")
    with output_csv.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(dresses)
    return image_ids


def extract_dress_images(archive_dir: Path, image_dir: Path, image_ids: set[str]) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    wanted_filenames = {f"{image_id}.jpg" for image_id in image_ids}
    extracted = {path.name for path in image_dir.glob("*.jpg") if path.name in wanted_filenames}

    for part in sorted(ARCHIVE_MD5):
        archive_path = download_archive(part, archive_dir)
        print(f"Extracting dresses from {archive_path.name}")
        with zipfile.ZipFile(archive_path) as archive:
            for entry in archive.infolist():
                filename = PurePosixPath(entry.filename).name
                if entry.is_dir() or filename not in wanted_filenames or filename in extracted:
                    continue
                target = image_dir / filename
                temporary = target.with_suffix(".jpg.part")
                with archive.open(entry) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output)
                temporary.replace(target)
                extracted.add(filename)
        archive_path.unlink()

    missing = wanted_filenames - extracted
    if missing:
        preview = ", ".join(sorted(missing)[:10])
        raise RuntimeError(f"Missing {len(missing)} dress images after extraction: {preview}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    archive_dir = data_root.parent.parent / "cache" / "glami_archives"
    metadata_dir = data_root / "GLAMI-1M-dataset"
    filtered_csv = data_root / "GLAMI-1M-dresses-train.csv"
    image_dir = data_root / "images-800px"

    archive_dir.mkdir(parents=True, exist_ok=True)
    train_csv = find_train_csv(archive_dir, metadata_dir)
    image_ids = build_dresses_csv(train_csv, filtered_csv)
    extract_dress_images(archive_dir, image_dir, image_ids)
    archive_dir.rmdir()
    print(f"Complete: {len(image_ids)} unique dresses images in {image_dir}")


if __name__ == "__main__":
    main()
