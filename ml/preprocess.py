import argparse
import os
import shutil
from collections import Counter

import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_BLUR_THRESHOLD = 120.0
DEFAULT_RAW_DIR = "data/raw"
DEFAULT_CLEAN_DIR = "data/processed/cleaned"
DEFAULT_REJECT_DIR = "data/processed/rejected"


def is_image_file(filename):
    return os.path.splitext(filename.lower())[1] in IMAGE_EXTENSIONS


def blur_score(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def prepare_output_dirs(class_names, clean_dir, reject_dir):
    for class_name in class_names:
        os.makedirs(os.path.join(clean_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(reject_dir, class_name), exist_ok=True)


def collect_class_names(raw_dir):
    class_names = []
    for entry in sorted(os.listdir(raw_dir)):
        class_path = os.path.join(raw_dir, entry)
        if os.path.isdir(class_path):
            class_names.append(entry)
    return class_names


def process_dataset(raw_dir, clean_dir, reject_dir, threshold, delete_rejected=False):
    class_names = collect_class_names(raw_dir)
    if not class_names:
        raise ValueError(f"No class folders found in: {raw_dir}")

    prepare_output_dirs(class_names, clean_dir, reject_dir)
    stats = Counter()

    for class_name in class_names:
        class_input_dir = os.path.join(raw_dir, class_name)
        class_clean_dir = os.path.join(clean_dir, class_name)
        class_reject_dir = os.path.join(reject_dir, class_name)

        for filename in os.listdir(class_input_dir):
            if not is_image_file(filename):
                stats["skipped_non_image"] += 1
                continue

            source_path = os.path.join(class_input_dir, filename)
            score = blur_score(source_path)

            if score is None:
                stats["unreadable"] += 1
                continue

            if score < threshold:
                stats["blurry"] += 1
                if not delete_rejected:
                    shutil.copy2(source_path, os.path.join(class_reject_dir, filename))
                continue

            shutil.copy2(source_path, os.path.join(class_clean_dir, filename))
            stats["kept"] += 1

    return stats


def print_class_balance(folder_path, title):
    print(f"\n{title}")
    for class_name in sorted(os.listdir(folder_path)):
        class_path = os.path.join(folder_path, class_name)
        if os.path.isdir(class_path):
            count = sum(1 for file_name in os.listdir(class_path) if is_image_file(file_name))
            print(f"  {class_name}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Clean cattle breed dataset using blur detection")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, help="Input folder containing class subfolders")
    parser.add_argument("--clean-dir", default=DEFAULT_CLEAN_DIR, help="Output folder for accepted images")
    parser.add_argument("--reject-dir", default=DEFAULT_REJECT_DIR, help="Output folder for rejected images")
    parser.add_argument("--threshold", type=float, default=DEFAULT_BLUR_THRESHOLD, help="Laplacian variance threshold")
    parser.add_argument("--delete-rejected", action="store_true", help="Delete blurry images instead of copying them")
    args = parser.parse_args()

    stats = process_dataset(
        raw_dir=args.raw_dir,
        clean_dir=args.clean_dir,
        reject_dir=args.reject_dir,
        threshold=args.threshold,
        delete_rejected=args.delete_rejected,
    )

    print("Cleaning complete.")
    print(f"Kept images: {stats['kept']}")
    print(f"Blurry images: {stats['blurry']}")
    print(f"Unreadable images: {stats['unreadable']}")
    print(f"Skipped non-image files: {stats['skipped_non_image']}")

    print_class_balance(args.clean_dir, "Class balance in cleaned dataset")
