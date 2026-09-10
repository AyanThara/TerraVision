"""
TerraVision - Phase 2: V1 Baseline Evaluation Script

This script loads the best V1 model checkpoint, evaluates performance strictly on the TEST split (2,700 images),
computes overall accuracy, macro and per-class precision, recall, and F1-score, generates a confusion matrix plot,
and saves metrics to results/.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from src.dataset import get_dataloaders, CLASS_NAMES
from src.model import SimpleCNN


@torch.no_grad()
def evaluate_test_set(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluates the model on the test dataset and returns ground-truth labels and predictions.
    """
    model.eval()
    all_preds = []
    all_targets = []

    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.numpy())

    return np.array(all_targets), np.array(all_preds)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path,
):
    """Generates and saves a confusion matrix visualization using matplotlib."""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("EuroSAT RGB Baseline CNN - Test Set Confusion Matrix", fontsize=14, pad=15)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right", fontsize=9)
    plt.yticks(tick_marks, class_names, fontsize=9)

    # Normalize matrix for display percentage text over raw counts
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            plt.text(
                j, i, f"{val:d}",
                horizontalalignment="center",
                verticalalignment="center",
                color=color,
                fontsize=8,
            )

    plt.ylabel("True Label", fontsize=11, labelpad=10)
    plt.xlabel("Predicted Label", fontsize=11, labelpad=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix plot to '{output_path}'")


def main():
    # Device selection: MPS (Apple Silicon), CUDA, or CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Evaluating V1 Baseline on device: {device}")

    checkpoint_path = PROJECT_ROOT / "models" / "v1_baseline.pth"
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    cm_output_path = results_dir / "v1_confusion_matrix.png"
    metrics_output_path = results_dir / "v1_evaluation_metrics.json"

    assert checkpoint_path.exists(), f"Checkpoint not found at '{checkpoint_path}'! Train the model first."

    # 1. Load DataLoaders (Test split only)
    print("\nLoading EuroSAT RGB Test DataLoader...")
    _, _, test_loader = get_dataloaders(batch_size=64, num_workers=0)

    # 2. Load Model Checkpoint
    print(f"Loading best checkpoint from '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SimpleCNN(num_classes=10).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_epoch = checkpoint.get("epoch", "N/A")
    best_val_acc = checkpoint.get("val_acc", 0.0)

    print(f"Checkpoint loaded (Best Validation Accuracy: {best_val_acc * 100:.2f}% at Epoch {best_epoch})")

    # 3. Predict on Test Set
    print("\nRunning inference on TEST set (2,700 images)...")
    y_true, y_pred = evaluate_test_set(model, test_loader, device)

    # 4. Compute Metrics
    acc = accuracy_score(y_true, y_pred)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    per_class_p, per_class_r, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(len(CLASS_NAMES)))
    )
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 65)
    print("V1 Baseline Model - Final Test Set Evaluation Results")
    print("=" * 65)
    print(f"Test Accuracy:    {acc * 100:.2f}%")
    print(f"Macro Precision:  {macro_precision * 100:.2f}%")
    print(f"Macro Recall:     {macro_recall * 100:.2f}%")
    print(f"Macro F1-Score:   {macro_f1 * 100:.2f}%")
    print("=" * 65)

    print("\n--- Per-Class Metrics ---")
    print(f"{'Class Name':<22} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 70)
    per_class_results = {}
    for idx, class_name in enumerate(CLASS_NAMES):
        p, r, f1, supp = per_class_p[idx], per_class_r[idx], per_class_f1[idx], per_class_support[idx]
        per_class_results[class_name] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1_score": round(float(f1), 4),
            "support": int(supp),
        }
        print(f"{class_name:<22} | {p * 100:<9.2f}% | {r * 100:<9.2f}% | {f1 * 100:<9.2f}% | {supp:<8d}")

    # 5. Plot & Save Confusion Matrix
    plot_confusion_matrix(cm, CLASS_NAMES, cm_output_path)

    # 6. Save JSON Metrics Summary
    eval_summary: Dict[str, Any] = {
        "model_name": "SimpleCNN_V1_Baseline",
        "best_epoch": best_epoch,
        "best_val_accuracy": round(float(best_val_acc), 4),
        "test_accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_precision), 4),
        "macro_recall": round(float(macro_recall), 4),
        "macro_f1_score": round(float(macro_f1), 4),
        "per_class": per_class_results,
        "confusion_matrix": cm.tolist(),
    }

    with open(metrics_output_path, "w") as f:
        json.dump(eval_summary, f, indent=4)

    print(f"Saved evaluation summary JSON to '{metrics_output_path}'")
    print("=" * 65)


if __name__ == "__main__":
    main()
