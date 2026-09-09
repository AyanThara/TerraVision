"""
TerraVision - Phase 1.3: EuroSAT RGB Dataset Inspection Script

This script loads the EuroSAT RGB satellite image dataset from Hugging Face datasets,
inspects its structure (dataset splits, dimensions, and class names), and displays
10 representative training images (one per class) in a 2x5 grid layout using matplotlib.
"""

from datasets import load_dataset
import matplotlib.pyplot as plt


def main():
    print("=" * 60)
    print("TerraVision: EuroSAT RGB Dataset Inspection")
    print("=" * 60)

    # 1. Load the EuroSAT RGB dataset from Hugging Face
    dataset_name = "giswqs/EuroSAT_RGB"
    print(f"\nLoading dataset '{dataset_name}' from Hugging Face...")
    dataset = load_dataset(dataset_name)

    # 2. Extract split sizes
    train_size = len(dataset["train"])
    val_size = len(dataset["validation"])
    test_size = len(dataset["test"])

    print("\n--- Dataset Split Sizes ---")
    print(f"Train set:       {train_size:,} images")
    print(f"Validation set:  {val_size:,} images")
    print(f"Test set:        {test_size:,} images")
    print(f"Total dataset:   {train_size + val_size + test_size:,} images")

    # 3. Extract class information
    class_names = dataset["train"].features["label"].names
    num_classes = len(class_names)

    print("\n--- Class Information ---")
    print(f"Number of classes: {num_classes}")
    print("Class names:")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")

    # 4. Extract image dimensions and properties
    sample_image = dataset["train"][0]["image"]
    width, height = sample_image.size
    mode = sample_image.mode
    num_channels = len(sample_image.getbands())

    print("\n--- Image Dimensions & Properties ---")
    print(f"Resolution:  {width} x {height} pixels")
    print(f"Color Mode:  {mode} ({num_channels} channels)")

    # 5. Collect 10 representative training images (one for each class)
    print("\nCollecting representative images for visualization...")
    samples_per_class = {}
    for sample in dataset["train"]:
        label = sample["label"]
        if label not in samples_per_class:
            samples_per_class[label] = sample["image"]
        if len(samples_per_class) == num_classes:
            break

    # 6. Display the 10 representative images in a 2x5 grid
    print("Displaying 10 representative training images in a 2x5 grid...")
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle("EuroSAT RGB - Representative Training Images (1 per Class)", fontsize=14, fontweight="bold")

    for label_id in range(num_classes):
        row = label_id // 5
        col = label_id % 5
        ax = axes[row, col]

        img = samples_per_class[label_id]
        class_name = class_names[label_id]

        ax.imshow(img)
        ax.set_title(class_name, fontsize=10, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    # Save a copy to results/ for verification artifact
    plt.savefig("results/eurosat_sample_grid.png", dpi=150, bbox_inches="tight")
    print("Saved visual grid to 'results/eurosat_sample_grid.png'")
    plt.show()

    print("\nDataset inspection complete!")


if __name__ == "__main__":
    main()
