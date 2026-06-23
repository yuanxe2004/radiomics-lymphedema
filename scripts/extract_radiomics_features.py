# -*- coding: utf-8 -*-
"""Example PyRadiomics feature extraction for subcutaneous-tissue VOIs.

This script is adapted from the manuscript extraction workflow. It expects
pre-resampled NIfTI images and binary masks. It does not perform DICOM
conversion, image registration, VOI segmentation, or manual mask review.
"""

import argparse
import logging
import os
import time
import traceback
from collections import defaultdict
from multiprocessing import Pool

import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor


def strip_nii_ext(filename):
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return filename


def is_nii_file(filename):
    return filename.endswith(".nii") or filename.endswith(".nii.gz")


def parse_mask_filename(mask_name):
    base = strip_nii_ext(os.path.basename(mask_name))
    parts = base.split("_")
    if len(parts) < 3:
        return None
    return parts[0], parts[1], parts[2]


def parse_img_filename(img_name):
    base = strip_nii_ext(os.path.basename(img_name))
    parts = base.split("_")
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def build_image_index(img_dir):
    index = defaultdict(list)
    for fn in os.listdir(img_dir):
        if not is_nii_file(fn):
            continue
        parsed = parse_img_filename(fn)
        if parsed is None:
            continue
        case_id, side = parsed
        index[f"{case_id}_{side}"].append(os.path.join(img_dir, fn))
    return index


def choose_best_image(candidates):
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: (len(os.path.basename(x)), os.path.basename(x)))[0]


def safe_int_or_str(x):
    try:
        xf = float(x)
        return int(xf) if xf.is_integer() else xf
    except Exception:
        return x


def check_image_mask_compatibility(img, msk):
    if img.GetSize() != msk.GetSize():
        raise ValueError(f"image/mask size mismatch: image={img.GetSize()}, mask={msk.GetSize()}")
    if tuple(img.GetSpacing()) != tuple(msk.GetSpacing()):
        raise ValueError(f"image/mask spacing mismatch: image={img.GetSpacing()}, mask={msk.GetSpacing()}")
    if tuple(img.GetOrigin()) != tuple(msk.GetOrigin()):
        raise ValueError(f"image/mask origin mismatch: image={img.GetOrigin()}, mask={msk.GetOrigin()}")
    if tuple(img.GetDirection()) != tuple(msk.GetDirection()):
        raise ValueError("image/mask direction mismatch")


def count_mask_voxels(msk, label=1):
    arr = sitk.GetArrayFromImage(msk)
    return int((arr == label).sum())


def filter_feature_dict(feature_dict, keep_spacing_info=True):
    filtered = {}
    for key, value in feature_dict.items():
        key_text = str(key)
        if keep_spacing_info and "Spacing" in key_text:
            filtered[key] = value
            continue
        if key_text.startswith("diagnostics_"):
            continue
        filtered[key] = value
    return filtered


def build_extractor(config_path):
    extractor = featureextractor.RadiomicsFeatureExtractor(config_path)
    extractor.disableFeatureClassByName("shape")
    extractor.disableFeatureClassByName("shape2D")
    return extractor


def collect_matched_cases(image_dir, mask_dir, max_cases=None):
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"image directory does not exist: {image_dir}")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"mask directory does not exist: {mask_dir}")

    image_index = build_image_index(image_dir)
    matched_cases = []
    unmatched_cases = []

    for mask_fn in sorted(f for f in os.listdir(mask_dir) if is_nii_file(f)):
        parsed = parse_mask_filename(mask_fn)
        if parsed is None:
            unmatched_cases.append({"mask_file": mask_fn, "reason": "expected mask filename: case_side_label.nii.gz"})
            continue
        case_id, side, label = parsed
        candidates = image_index.get(f"{case_id}_{side}", [])
        if not candidates:
            unmatched_cases.append({"mask_file": mask_fn, "case_id": case_id, "side": side, "label": label, "reason": "no matching image"})
            continue
        image_path = choose_best_image(candidates)
        matched_cases.append({
            "case_id": case_id,
            "side": side,
            "label": safe_int_or_str(label),
            "img_path": image_path,
            "mask_path": os.path.join(mask_dir, mask_fn),
            "img_file": os.path.basename(image_path),
            "mask_file": mask_fn,
        })

    if max_cases is not None:
        matched_cases = matched_cases[:max_cases]
    return matched_cases, unmatched_cases


def process_one_case(args):
    item, config_path, mask_label, keep_spacing_info = args
    started = time.time()
    row = {
        "序号": item["case_id"],
        "肢体": item["side"],
        "标签": item["label"],
        "image_file": item["img_file"],
        "mask_file": item["mask_file"],
        "image_path": item["img_path"],
        "mask_path": item["mask_path"],
    }
    try:
        extractor = build_extractor(config_path)
        img = sitk.ReadImage(item["img_path"])
        msk = sitk.ReadImage(item["mask_path"])
        check_image_mask_compatibility(img, msk)
        row["image_spacing_x"], row["image_spacing_y"], row["image_spacing_z"] = img.GetSpacing()
        row["image_size_x"], row["image_size_y"], row["image_size_z"] = img.GetSize()
        voxel_count = count_mask_voxels(msk, label=mask_label)
        row["mask_voxel_count"] = voxel_count
        if voxel_count == 0:
            raise ValueError(f"mask contains no voxels with label={mask_label}")
        feature_dict = filter_feature_dict(extractor.execute(item["img_path"], item["mask_path"]), keep_spacing_info)
        row.update(feature_dict)
        row["status"] = "success"
    except Exception as exc:
        row["status"] = "failed"
        row["error_message"] = str(exc)
        row["traceback"] = traceback.format_exc()
    row["elapsed_sec"] = round(time.time() - started, 3)
    return row


def parse_args():
    parser = argparse.ArgumentParser(description="Extract PyRadiomics features from matched NIfTI image/mask files.")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--output-xlsx", required=True)
    parser.add_argument("--config", default=os.path.join("configs", "pyradiomics.yaml"))
    parser.add_argument("--mask-label", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--drop-spacing-info", action="store_true")
    return parser.parse_args()


def main():
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_xlsx)), exist_ok=True)

    matched_cases, unmatched_cases = collect_matched_cases(args.image_dir, args.mask_dir, args.max_cases)
    if not matched_cases:
        raise RuntimeError("No matched image/mask cases found.")

    worker_args = [(item, args.config, args.mask_label, not args.drop_spacing_info) for item in matched_cases]
    results = []
    with Pool(processes=args.workers) as pool:
        for row in pool.imap_unordered(process_one_case, worker_args):
            results.append(row)

    df_results = pd.DataFrame([row for row in results if row.get("status") == "success"])
    df_failed = pd.DataFrame([row for row in results if row.get("status") != "success"])
    df_unmatched = pd.DataFrame(unmatched_cases)

    with pd.ExcelWriter(args.output_xlsx, engine="openpyxl") as writer:
        df_results.to_excel(writer, sheet_name="radiomics_all", index=False)
        if not df_failed.empty:
            df_failed.to_excel(writer, sheet_name="failed_cases", index=False)
        if not df_unmatched.empty:
            df_unmatched.to_excel(writer, sheet_name="unmatched_cases", index=False)


if __name__ == "__main__":
    main()
