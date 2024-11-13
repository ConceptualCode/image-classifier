import os

def count_images_in_directory(data_dir):
    class_counts = {}
    total_count = 0

    for root, dirs, files in os.walk(data_dir):
        if dirs:  # Skip the root directory itself
            continue
        class_name = os.path.basename(root)
        count = len(files)
        class_counts[class_name] = count
        total_count += count

    print("\nOriginal Dataset Counts by Class:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} images")
    print(f"Total images in dataset: {total_count}")
    
    return class_counts, total_count


def estimate_effective_dataset_size(class_counts, epochs):
    effective_class_sizes = {class_name: count * epochs for class_name, count in class_counts.items()}
    total_effective_size = sum(effective_class_sizes.values())
    
    print("\nEstimated Effective Dataset Size with Augmentation per Class:")
    for class_name, size in effective_class_sizes.items():
        print(f"{class_name}: {size} images (over {epochs} epochs)")
    
    print(f"\nTotal Estimated Effective Dataset Size with Augmentation: {total_effective_size} images")
    return effective_class_sizes, total_effective_size