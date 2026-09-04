"""
Rigorous Held-Out Evaluation and Validation Suite for 1D-CNN AMC Classifier.

Evaluates trained model weights (models/amc_1dcnn_weights.pt) exclusively
against an independently generated held-out test dataset with zero data leakage.

Calculates and reports:
- Overall Accuracy
- Per-Class Precision, Recall, F1-Score
- Full Confusion Matrix
- Stratified Accuracy breakdown across SNR levels (30 dB, 25 dB, 20 dB, 15 dB, 10 dB, 5 dB, 0 dB)
- Mean Inference Latency per sample window
"""

import argparse
from pathlib import Path
import time
from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F

from src.amc.dataset_generator import (
    generate_synthetic_dataset,
    CNN_CLASS_LABELS,
    DEFAULT_WINDOW_SIZE,
)
from src.amc.models.cnn1d_classifier import Modulation1DCNN, ModulationClassifier


def run_cnn_validation(
    weights_path: str = "models/amc_1dcnn_weights.pt",
    test_samples_per_class: int = 150,
    window_length: int = DEFAULT_WINDOW_SIZE,
    seed: int = 999999,
) -> Dict:
    print("=" * 65)
    print("1D-CNN AUTOMATIC MODULATION CLASSIFIER — HELD-OUT VALIDATION")
    print("=" * 65)

    weights_file = Path(weights_path)
    if not weights_file.exists():
        raise FileNotFoundError(f"Model weights file not found: {weights_path}")

    # Load weights and metadata
    checkpoint = torch.load(weights_file, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    class_labels = checkpoint.get("class_labels", CNN_CLASS_LABELS) if isinstance(checkpoint, dict) else CNN_CLASS_LABELS

    model = Modulation1DCNN(sequence_length=window_length, num_classes=len(class_labels), class_labels=class_labels)
    model.load_state_dict(state_dict)
    model.eval()

    num_classes = len(class_labels)
    total_test_samples = test_samples_per_class * num_classes

    print(f"Loading weights from: {weights_file}")
    print(f"Generating independent held-out test set: {total_test_samples} samples ({test_samples_per_class}/class)...")
    snr_levels = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]

    X_test, y_test, snrs_test = generate_synthetic_dataset(
        samples_per_class=test_samples_per_class,
        window_length=window_length,
        snr_levels=snr_levels,
        seed_offset=seed,
    )

    # Run inference and measure latency
    predictions = []
    probabilities = []
    latencies = []

    with torch.no_grad():
        for i in range(len(X_test)):
            sample_tensor = torch.from_numpy(X_test[i]).unsqueeze(0).float()
            
            t0 = time.perf_counter()
            logits = model(sample_tensor)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000.0) # ms
            pred_idx = int(np.argmax(probs))
            predictions.append(pred_idx)
            probabilities.append(probs)

    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    latencies = np.array(latencies)

    # 1. Overall Accuracy
    correct_mask = (predictions == y_test)
    overall_acc = float(np.mean(correct_mask))

    # 2. Confusion Matrix
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_lbl, pred_lbl in zip(y_test, predictions):
        conf_matrix[true_lbl, pred_lbl] += 1

    # 3. Per-Class Precision, Recall, F1
    per_class_metrics = {}
    for c_idx, c_name in enumerate(class_labels):
        tp = conf_matrix[c_idx, c_idx]
        fp = np.sum(conf_matrix[:, c_idx]) - tp
        fn = np.sum(conf_matrix[c_idx, :]) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class_metrics[c_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(np.sum(y_test == c_idx)),
        }

    # 4. Stratified Accuracy by SNR
    snr_breakdown = {}
    for snr in sorted(snr_levels):
        mask = (snrs_test == snr)
        if np.sum(mask) > 0:
            acc_snr = float(np.mean(predictions[mask] == y_test[mask]))
            snr_breakdown[snr] = {"accuracy": acc_snr, "count": int(np.sum(mask))}

    # Print Summary Tables
    print("\n" + "-" * 65)
    print(f"HELD-OUT TEST EVALUATION METRICS (Total: {total_test_samples} Samples)")
    print("-" * 65)
    print(f"Overall Accuracy:  {overall_acc * 100:.2f}%")
    print(f"Mean Latency:      {np.mean(latencies):.3f} ms / window (P95: {np.percentile(latencies, 95):.3f} ms)")

    print("\nPER-CLASS CLASSIFICATION REPORT:")
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
    print("-" * 56)
    for c_name, m in per_class_metrics.items():
        print(f"{c_name:<12} {m['precision']*100:>8.2f}%    {m['recall']*100:>8.2f}%    {m['f1']*100:>8.2f}%    {m['support']:<8}")

    print("\nCONFUSION MATRIX (Rows: True, Cols: Predicted):")
    header = f"{'':<12}" + "".join([f"{c:>10}" for c in class_labels])
    print(header)
    for c_idx, c_name in enumerate(class_labels):
        row_str = f"{c_name:<12}" + "".join([f"{conf_matrix[c_idx, j]:>10d}" for j in range(num_classes)])
        print(row_str)

    print("\nACCURACY BY SNR LEVEL:")
    print(f"{'SNR (dB)':<12} {'Accuracy':<12} {'Test Count':<10}")
    print("-" * 34)
    for snr, info in sorted(snr_breakdown.items(), reverse=True):
        print(f"{snr:>6.1f} dB     {info['accuracy']*100:>8.2f}%    {info['count']:<10}")
    print("=" * 65)

    return {
        "overall_accuracy": overall_acc,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": conf_matrix.tolist(),
        "snr_breakdown": snr_breakdown,
        "mean_latency_ms": float(np.mean(latencies)),
        "total_test_samples": total_test_samples,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 1D-CNN AMC Model on Held-out Dataset")
    parser.add_argument("--weights", type=str, default="models/amc_1dcnn_weights.pt")
    parser.add_argument("--samples-per-class", type=int, default=150)
    args = parser.parse_args()

    run_cnn_validation(
        weights_path=args.weights,
        test_samples_per_class=args.samples_per_class,
    )
