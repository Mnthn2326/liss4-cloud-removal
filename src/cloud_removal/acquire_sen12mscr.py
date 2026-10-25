"""Acquisition utility for downloading a reproducible subset of the SEN12MS-CR dataset.

Streams targeted regions of interest (ROIs) from the official MediaTUM FTP repository
without downloading entire multi-gigabyte archives.
"""

from __future__ import annotations

import argparse
import ftplib
import json
import os
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

FTP_HOST = os.environ.get("SEN12MSCR_FTP_HOST", "dataserv.ub.tum.de")
FTP_USER = os.environ.get("SEN12MSCR_FTP_USER", "")
FTP_PASS = os.environ.get("SEN12MSCR_FTP_PASS", "")

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
    if FTP_USER and FTP_PASS:
        ftp.login(FTP_USER, FTP_PASS)
    else:
        try:
            ftp.login()
        except ftplib.error_perm as perm_err:
            raise PermissionError(
                f"FTP access requires credentials. Please set SEN12MSCR_FTP_USER and "
                f"SEN12MSCR_FTP_PASS environment variables (see .env.example): {perm_err}"
            ) from perm_err

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


def verify_existing_subset(
    output_dir: Path,
    output_manifest: Optional[Path] = None,
    max_patches: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Verify an existing SEN12MS-CR dataset subset and generate a versioned manifest.

    Args:
        output_dir: Path to raw SEN12MS-CR directory (containing manifest.json and folders).
        output_manifest: Destination path for versioned output manifest.
        max_patches: Optional cap on verified patch count.

    Returns:
        Manifest dictionary if valid, None otherwise.
    """
    raw_manifest = output_dir / "manifest.json"
    if not raw_manifest.is_file():
        return None

    try:
        with open(raw_manifest, "r", encoding="utf-8") as fp:
            manifest_data = json.load(fp)
    except Exception:
        return None

    cloudy_dir = output_dir / manifest_data.get("cloudy_directory", "")
    clear_dir = output_dir / manifest_data.get("clear_directory", "")

    verified_samples = []
    for item in manifest_data.get("paired_samples", []):
        c_path = cloudy_dir / item["cloudy_file"]
        clr_path = clear_dir / item["clear_file"]
        if c_path.is_file() and clr_path.is_file():
            verified_samples.append(item)
            if max_patches and len(verified_samples) >= max_patches:
                break

    if not verified_samples:
        return None

    print(f"Verified {len(verified_samples)} existing paired optical patches in {output_dir}.")
    manifest_data["total_paired_patches"] = len(verified_samples)
    manifest_data["paired_samples"] = verified_samples
    manifest_data["verified_timestamp_utc"] = datetime.now(timezone.utc).isoformat()

    out_m = output_manifest if output_manifest else output_dir / "manifest.json"
    out_m.parent.mkdir(parents=True, exist_ok=True)
    with open(out_m, "w", encoding="utf-8") as fp:
        json.dump(manifest_data, fp, indent=2)

    print(f"Versioned manifest written to: {out_m}")
    return manifest_data


def acquire_sen12mscr_subset(
    output_dir: Path,
    season: str = DEFAULT_SEASON,
    roi_id: str = DEFAULT_ROI,
    max_patches: Optional[int] = None,
    output_manifest: Optional[Path] = None,
    verify_existing: bool = True,
) -> Dict[str, Any]:
    """Acquire or verify a small, reproducible paired optical subset from SEN12MS-CR."""
    if verify_existing:
        verified = verify_existing_subset(
            output_dir=output_dir,
            output_manifest=output_manifest,
            max_patches=max_patches,
        )
        if verified is not None:
            return verified

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
    parser = argparse.ArgumentParser(
        description="Download or verify a reproducible subset of SEN12MS-CR"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/raw/sen12mscr", help="Path to raw dataset directory"
    )
    parser.add_argument(
        "--output-manifest",
        type=str,
        default="data/manifests/sen12mscr_manifest.json",
        help="Path for versioned manifest",
    )
    parser.add_argument(
        "--season", type=str, default="winter", help="Season identifier ('winter' or 'spring')"
    )
    parser.add_argument("--roi", type=str, default="102", help="Region of Interest number")
    parser.add_argument(
        "--max-patches", type=int, default=None, help="Cap on number of patches to acquire/verify"
    )
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        default=True,
        help="Verify existing files if present before network download",
    )
    parser.add_argument(
        "--force-download", action="store_true", help="Force network re-download from MediaTUM FTP"
    )

    args = parser.parse_args()
    out = Path(args.output_dir)
    out_m = Path(args.output_manifest) if args.output_manifest else None
    verify = args.verify_existing and not args.force_download

    try:
        acquire_sen12mscr_subset(
            output_dir=out,
            season=args.season,
            roi_id=args.roi,
            max_patches=args.max_patches,
            output_manifest=out_m,
            verify_existing=verify,
        )
        return 0
    except Exception as err:
        print(f"Acquisition error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
