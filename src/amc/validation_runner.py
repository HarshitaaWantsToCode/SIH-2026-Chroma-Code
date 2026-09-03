"""
Automated AMC Validation Runner & Benchmark Tool.

Executes end-to-end evaluation:
1. Synthesizes / loads known ground truth signals across SNRs (0dB to 30dB) and CFOs
2. Feeds signals through identical ingestion & normalizer pipelines
3. Runs the AMC service
4. Records predicted classes, confidence, evidence, and status
5. Computes empirical accuracy, per-class metrics, and confusion matrix.
"""

from typing import Dict, List, Tuple
import numpy as np
from src.amc.ground_truth_generator import GroundTruthSignal, GroundTruthSignalGenerator
from src.amc.heuristic_classifier import HeuristicModulationClassifier
from src.ingestion.normalizer import SignalNormalizer


class AMCValidationRunner:
    """
    Automated Validation Harness for Modulation Recognition.
    """

    @classmethod
    def run_benchmark(cls, dataset: List[GroundTruthSignal] = None) -> Dict[str, any]:
        """
        Executes validation on dataset and computes empirical confusion matrix & accuracy.
        """
        if dataset is None:
            dataset = GroundTruthSignalGenerator.generate_validation_suite()

        classes = ["BPSK", "QPSK", "16-QAM", "2-FSK", "UNKNOWN"]
        class_to_idx = {c: i for i, c in enumerate(classes)}
        
        confusion_matrix = np.zeros((len(classes), len(classes)), dtype=int)
        
        records = []
        correct_count = 0
        total_evals = 0

        by_snr: Dict[float, List[bool]] = {}

        for sample in dataset:
            # 1. Pipeline Normalization
            norm_sig, _ = SignalNormalizer.normalize_unit_power(sample.signal_iq)
            
            # 2. AMC Evaluation
            res = HeuristicModulationClassifier.classify(norm_sig)

            # Ground truth index
            gt_mod = sample.modulation
            pred_mod = res.predicted_modulation

            gt_idx = class_to_idx.get(gt_mod, 4)
            pred_idx = class_to_idx.get(pred_mod, 4)

            confusion_matrix[gt_idx, pred_idx] += 1
            
            is_correct = (gt_mod == pred_mod)
            if is_correct:
                correct_count += 1
            total_evals += 1

            snr_val = sample.snr_db
            if snr_val not in by_snr:
                by_snr[snr_val] = []
            by_snr[snr_val].append(is_correct)

            records.append({
                "ground_truth": gt_mod,
                "predicted": pred_mod,
                "confidence": res.confidence,
                "snr_db": sample.snr_db,
                "cfo_hz": sample.cfo_hz,
                "status": res.status,
                "is_correct": is_correct
            })

        overall_accuracy = (correct_count / max(total_evals, 1)) * 100.0

        # Per-class accuracy
        per_class = {}
        for c in ["BPSK", "QPSK", "16-QAM", "2-FSK"]:
            c_idx = class_to_idx[c]
            row_total = np.sum(confusion_matrix[c_idx, :])
            row_correct = confusion_matrix[c_idx, c_idx]
            acc = (row_correct / max(row_total, 1)) * 100.0 if row_total > 0 else 0.0
            per_class[c] = {"total": int(row_total), "correct": int(row_correct), "accuracy_pct": acc}

        # SNR Breakdown
        snr_acc = {float(s): float((sum(v)/len(v))*100.0) for s, v in sorted(by_snr.items(), reverse=True)}

        return {
            "total_samples": total_evals,
            "overall_accuracy_pct": overall_accuracy,
            "per_class_accuracy": per_class,
            "snr_breakdown_pct": snr_acc,
            "confusion_matrix": confusion_matrix.tolist(),
            "classes": classes,
            "records": records
        }


if __name__ == "__main__":
    print("Executing Empirical AMC Validation Run...")
    results = AMCValidationRunner.run_benchmark()
    print(f"\n==========================================")
    print(f"AMC VALIDATION BENCHMARK RESULTS")
    print(f"Total Evaluated Signals: {results['total_samples']}")
    print(f"Overall Empirical Accuracy: {results['overall_accuracy_pct']:.2f}%")
    print(f"==========================================")
    print("Per-Class Accuracy:")
    for c, stats in results["per_class_accuracy"].items():
        print(f"  - {c:8s}: {stats['accuracy_pct']:.1f}% ({stats['correct']}/{stats['total']})")
    print("\nAccuracy across SNR Levels:")
    for s, acc in results["snr_breakdown_pct"].items():
        print(f"  - SNR {s:4.1f} dB: {acc:.1f}%")
    print("==========================================")
