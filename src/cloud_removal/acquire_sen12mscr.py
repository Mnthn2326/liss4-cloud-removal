"""Acquisition utility for downloading a reproducible subset of the SEN12MS-CR dataset.

Streams targeted regions of interest (ROIs) from the official MediaTUM FTP repository
without downloading entire multi-gigabyte archives.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import ftplib
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import zlib


FTP_HOST = "dataserv.ub.tum.de"
FTP_USER = "m1554803"
FTP_PASS = "m1554803"

DEFAULT_SEASON = "winter"
DEFAULT_ROI = "102"
ARCHIVE_MAP = {
    "winter": {
        "cloudy": "ROIs2017_winter_s2_cloudy.tar.gz",
        "clear": "ROIs2017_winter_s2.tar.gz",
        "cloudy_prefix": "ROIs2017_winter_s2_cloudy/s2_cloudy_",
        "clear_prefix": "ROIs2017_winter_s2/s2_",
    },
    "spring": {
        "cloudy": "ROIs1158_spring_s2_cloudy.tar.gz",
        "clear": "ROIs1158_spring_s2.tar.gz",
        "cloudy_prefix": "ROIs1158_spring_s2_cloudy/s2_cloudy_",
        "clear_prefix": "ROIs1158_spring_s2/s2_",
    },
}


def stream_extract_roi_tar(
    archive_name: str,
    target_prefix: str,
    output_dir: Path,
    max_patches: Optional[int] = None,
) -> List[Path]:
    """Stream tar.gz from FTP and extract only members matching target_prefix."""
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_paths: List[Path] = []

    print(f"Connecting to FTP ({FTP_HOST}) for {archive_name}...")
    ftp = ftplib.FTP(FTP_HOST, timeout=60)
    ftp.login(FTP_USER, FTP_PASS)

    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    raw_tar = bytearray()
    pos = 0
    done = False
    started = False

    def process_raw():
        nonlocal pos, done, started
        while True:
            if len(raw_tar) - pos < 512:
                break

            header = raw_tar[pos : pos + 512]
            if header == b"\x00" * 512:
                pos += 512
                continue

            name_raw = header[0:100].split(b"\x00")[0].decode("utf-8", errors="ignore")
            size_raw = header[124:136].split(b"\x00")[0].strip()

            try:
                size = int(size_raw, 8) if size_raw else 0
            except ValueError:
                size = 0

            typeflag = chr(header[156]) if len(header) > 156 else "0"

            if name_raw.startswith(target_prefix):
                started = True
            elif started:
                print(f"Reached boundary of target ROI at {name_raw}. Halting stream.")
                done = True
                return

            blocks = (size + 511) // 512
            total_member_bytes = 512 + blocks * 512

            if len(raw_tar) - pos < total_member_bytes:
                break

            if (
                typeflag in ("0", "\x00")
                and size > 0
                and name_raw.startswith(target_prefix)
                and name_raw.endswith(".tif")
            ):
                dest = output_dir / Path(name_raw).name
                file_bytes = raw_tar[pos + 512 : pos + 512 + size]
                with open(dest, "wb") as f:
                    f.write(file_bytes)
                extracted_paths.append(dest)

                if len(extracted_paths) % 100 == 0:
                    print(f"  Extracted {len(extracted_paths)} patches...")

                if max_patches and len(extracted_paths) >= max_patches:
                    print(f"  Reached max_patches limit ({max_patches}). Done!")
                    done = True
                    return

            pos += total_member_bytes

        if pos > 20 * 1024 * 1024:
            del raw_tar[:pos]
            pos = 0

    def ftp_callback(chunk):
        if done:
            raise StopIteration
        decompressed = d.decompress(chunk)
        if decompressed:
            raw_tar.extend(decompressed)
            process_raw()
            if done:
                raise StopIteration

    try:
        ftp.retrbinary(f"RETR {archive_name}", ftp_callback, blocksize=64 * 1024)
    except StopIteration:
        pass
    except Exception as e:
        print(f"FTP transfer finished/aborted: {e}")
    finally:
        try:
            ftp.close()
        except Exception:
            pass

    print(f"Extracted {len(extracted_paths)} patches to {output_dir}")
    return extracted_paths


def acquire_sen12mscr_subset(
    output_dir: Path,
    season: str = DEFAULT_SEASON,
    roi_id: str = DEFAULT_ROI,
    max_patches: Optional[int] = None,
) -> Dict[str, Any]:
    """Acquire a small, reproducible paired optical subset from SEN12MS-CR."""
    if season not in ARCHIVE_MAP:
        raise ValueError(f"Unsupported season: '{season}'. Choose from: {list(ARCHIVE_MAP.keys())}")

    cloudy_archive = ARCHIVE_MAP[season]["cloudy"]
    clear_archive = ARCHIVE_MAP[season]["clear"]
    cloudy_prefix = f"{ARCHIVE_MAP[season]['cloudy_prefix']}{roi_id}/"
    clear_prefix = f"{ARCHIVE_MAP[season]['clear_prefix']}{roi_id}/"

    cloudy_out = output_dir / f"ROIs2017_{season}_s2_cloudy" / f"s2_cloudy_{roi_id}"
    clear_out = output_dir / f"ROIs2017_{season}_s2" / f"s2_{roi_id}"

    print(f"=== Acquiring SEN12MS-CR Optical Subset (Season: {season}, ROI: {roi_id}) ===")
    print(f"Target cloudy directory: {cloudy_out}")
    print(f"Target clear directory:  {clear_out}")

    cloudy_files = stream_extract_roi_tar(
        archive_name=cloudy_archive,
        target_prefix=cloudy_prefix,
        output_dir=cloudy_out,
        max_patches=max_patches,
    )

    clear_files = stream_extract_roi_tar(
        archive_name=clear_archive,
        target_prefix=clear_prefix,
        output_dir=clear_out,
        max_patches=max_patches,
    )

    # Match pairs based on patch number (e.g. p100)
    cloudy_map = {f.name.split("_")[-1]: f for f in cloudy_files}
    clear_map = {f.name.split("_")[-1]: f for f in clear_files}

    common_patch_keys = sorted(list(set(cloudy_map.keys()) & set(clear_map.keys())))
    print(f"\nPaired patches found: {len(common_patch_keys)}")

    # Generate acquisition manifest
    manifest_path = output_dir / "manifest.json"
    manifest_data = {
        "dataset_name": "SEN12MS-CR",
        "source": "MediaTUM (dataserv.ub.tum.de)",
        "citation": "P. Ebel, A. Meraner, M. Schmitt, X. X. Zhu (IEEE TGRS, 2020)",
        "license": "CC-BY",
        "download_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "season": season,
        "roi_id": roi_id,
        "official_split": "train",
        "official_split_id": 1,
        "total_paired_patches": len(common_patch_keys),
        "cloudy_directory": str(cloudy_out.relative_to(output_dir)).replace("\\", "/"),
        "clear_directory": str(clear_out.relative_to(output_dir)).replace("\\", "/"),
        "paired_samples": [
            {
                "patch_id": p_key.replace(".tif", ""),
                "cloudy_file": cloudy_map[p_key].name,
                "clear_file": clear_map[p_key].name,
                "size_bytes_cloudy": cloudy_map[p_key].stat().st_size,
                "size_bytes_clear": clear_map[p_key].stat().st_size,
            }
            for p_key in common_patch_keys
        ],
    }

    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest_data, fp, indent=2)

    print(f"Saved dataset manifest to {manifest_path}")
    return manifest_data


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a reproducible subset of SEN12MS-CR")
    parser.add_argument("--output-dir", type=str, default="data/raw/sen12mscr")
    parser.add_argument("--season", type=str, default="winter")
    parser.add_argument("--roi", type=str, default="102")
    parser.add_argument("--max-patches", type=int, default=None)

    args = parser.parse_args()
    out = Path(args.output_dir)
    acquire_sen12mscr_subset(output_dir=out, season=args.season, roi_id=args.roi, max_patches=args.max_patches)
    return 0


if __name__ == "__main__":
    sys.exit(main())
