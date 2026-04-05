import os
import shutil
import random
from collections import Counter

CLEANED_DIR = "data/processed/cleaned"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42


def collect_images_by_class(source_dir):
    class_images = {}
    for class_name in sorted(os.listdir(source_dir)):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        if image_files:
            class_images[class_name] = image_files
    return class_images


def create_split_folders(class_names, train_dir, val_dir, test_dir):
    for class_name in class_names:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)


def split_dataset(source_dir=CLEANED_DIR, train_dir=TRAIN_DIR, val_dir=VAL_DIR, test_dir=TEST_DIR,
                  train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO, seed=RANDOM_SEED):
    random.seed(seed)
    class_images = collect_images_by_class(source_dir)
    if not class_images:
        raise ValueError(f"No images found in: {source_dir}")
    
    create_split_folders(class_images.keys(), train_dir, val_dir, test_dir)
    stats = Counter()
    
    for class_name, image_files in class_images.items():
        shuffled_files = image_files.copy()
        random.shuffle(shuffled_files)
        
        n_total = len(shuffled_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_files = shuffled_files[:n_train]
        val_files = shuffled_files[n_train:n_train + n_val]
        test_files = shuffled_files[n_train + n_val:]
        
        source_class_path = os.path.join(source_dir, class_name)
        
        for filename in train_files:
            src = os.path.join(source_class_path, filename)
            dst = os.path.join(train_dir, class_name, filename)
            shutil.copy2(src, dst)
            stats["train"] += 1
        
        for filename in val_files:
            src = os.path.join(source_class_path, filename)
            dst = os.path.join(val_dir, class_name, filename)
            shutil.copy2(src, dst)
            stats["val"] += 1
        
        for filename in test_files:
            src = os.path.join(source_class_path, filename)
            dst = os.path.join(test_dir, class_name, filename)
            shutil.copy2(src, dst)
            stats["test"] += 1
    
    return stats, class_images


def count_images_per_class(split_dir):
    counts = {}
    for class_name in sorted(os.listdir(split_dir)):
        class_path = os.path.join(split_dir, class_name)
        if os.path.isdir(class_path):
            counts[class_name] = len([f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
    return counts


def validate_split(train_dir, val_dir, test_dir):
    train_counts = count_images_per_class(train_dir)
    val_counts = count_images_per_class(val_dir)
    test_counts = count_images_per_class(test_dir)
    
    train_total = sum(train_counts.values())
    val_total = sum(val_counts.values())
    test_total = sum(test_counts.values())
    grand_total = train_total + val_total + test_total
    
    train_pct = (train_total / grand_total * 100) if grand_total > 0 else 0
    val_pct = (val_total / grand_total * 100) if grand_total > 0 else 0
    test_pct = (test_total / grand_total * 100) if grand_total > 0 else 0
    
    print("\n--- Dataset Split Summary ---")
    print(f"Train: {train_total} images ({train_pct:.1f}%)")
    print(f"Validation: {val_total} images ({val_pct:.1f}%)")
    print(f"Test: {test_total} images ({test_pct:.1f}%)")
    print(f"Total: {grand_total} images\n")
    
    print("Train set per class:")
    for class_name, count in sorted(train_counts.items()):
        print(f"  {class_name}: {count}")
    
    print("\nValidation set per class:")
    for class_name, count in sorted(val_counts.items()):
        print(f"  {class_name}: {count}")
    
    print("\nTest set per class:")
    for class_name, count in sorted(test_counts.items()):
        print(f"  {class_name}: {count}")


def main():
    stats, class_images = split_dataset()
    print("Dataset splitting complete.")
    print(f"Total images processed: {sum(stats.values())}")
    print(f"Train: {stats['train']}, Validation: {stats['val']}, Test: {stats['test']}")
    validate_split(TRAIN_DIR, VAL_DIR, TEST_DIR)


if __name__ == "__main__":
    main()
