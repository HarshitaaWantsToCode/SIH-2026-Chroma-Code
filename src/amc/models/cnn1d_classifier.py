"""
Automatic Modulation Classification (AMC) Neural Network & Classifier Service.

Provides:
- Modulation1DCNN: PyTorch 1D-CNN Architecture.
- ClassificationResult: Standardized prediction dataclass.
- ModulationClassifier: Resilient Service abstraction supporting trained PyTorch weights
  with deterministic demo inference fallback when weights are not present.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = object
    F = None


@dataclass
class ClassificationResult:
    """Standardized result returned by ModulationClassifier."""
    modulation: str
    confidence: float                          # Float between 0.0 and 1.0
    probabilities: Dict[str, float]            # Mapping like {"QPSK": 0.92, "BPSK": 0.04, ...}
    model_status: str                          # "REAL_TRAINED_MODEL", "HEURISTIC_EVALUATION", "INSUFFICIENT_EVIDENCE"
    classifier_type: str                       # "DEEP_1D_CNN" or "HEURISTIC_FEATURE_EXTRACTION"
    evidence: List[str] = field(default_factory=list)
    explanation: str = ""
    is_comm_like: bool = True
    model_name: str = "1D-CNN Feature Extractor"
    architecture_summary: Dict[str, str] = field(default_factory=lambda: {
        "Layer 1": "Conv1D(in=2, out=64, k=7) + BatchNorm1D + ReLU + MaxPool1D(2)",
        "Layer 2": "Conv1D(in=64, out=128, k=5) + BatchNorm1D + ReLU + MaxPool1D(2)",
        "Layer 3": "Conv1D(in=128, out=256, k=3) + BatchNorm1D + ReLU + MaxPool1D(2)",
        "Classifier Head": "Dropout(0.4) -> Linear(flattened, 256) -> ReLU -> Linear(256, 4) -> Softmax"
    })


if TORCH_AVAILABLE:
    class Modulation1DCNN(nn.Module):
        """
        1D-CNN Feature Extractor and Softmax Classifier for raw I/Q arrays.
        Input Tensor Shape: [Batch_Size, 2, Sequence_Length] (Ch0: I, Ch1: Q)
        """

        def __init__(
            self,
            sequence_length: int = 1024,
            num_classes: int = 4,
            class_labels: Optional[List[str]] = None
        ) -> None:
            super().__init__()
            self.sequence_length = sequence_length
            self.num_classes = num_classes
            self.class_labels = class_labels or ["BPSK", "QPSK", "16QAM", "FSK"]

            # Feature Extraction Layers
            self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=7, padding=3)
            self.bn1 = nn.BatchNorm1d(64)

            self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
            self.bn2 = nn.BatchNorm1d(128)

            self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm1d(256)

            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
            self.dropout = nn.Dropout(p=0.4)

            # Compute flattened feature size
            flattened_dim = (sequence_length // 8) * 256

            # Classification Head
            self.fc1 = nn.Linear(flattened_dim, 256)
            self.fc2 = nn.Linear(256, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = self.dropout(x)
            x = torch.flatten(x, start_dim=1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            logits = self.fc2(x)
            return logits
else:
    class Modulation1DCNN:
        def __init__(self, *args, **kwargs):
            pass


class ModulationClassifier:
    """
    Classifier Service Abstraction.
    Checks for trained PyTorch model weights on disk.
    If weights exist -> Executes real neural inference.
    Otherwise -> Executes a deterministic mock/heuristic classifier based on signal characteristics.
    """

    def __init__(self, weights_path: Optional[Union[str, Path]] = None) -> None:
        self.class_labels = ["QPSK", "BPSK", "16-QAM", "2-FSK"]
        self.weights_path = Path(weights_path) if weights_path else Path("models/amc_1dcnn_weights.pt")
        self.has_trained_weights = False
        self.model = None

        if TORCH_AVAILABLE and self.weights_path.exists():
            try:
                self.model = Modulation1DCNN(sequence_length=1024, num_classes=4, class_labels=self.class_labels)
                state_dict = torch.load(self.weights_path, map_location="cpu")
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.has_trained_weights = True
            except Exception:
                self.has_trained_weights = False

    def predict(
        self,
        signal: np.ndarray,
        demo_modulation_hint: Optional[str] = None
    ) -> ClassificationResult:
        """
        Predicts modulation scheme for the input complex signal.

        Args:
            signal: 1D complex NumPy array.
            demo_modulation_hint: Optional known ground truth (e.g. from selected demo preset).

        Returns:
            ClassificationResult containing dominant modulation, confidence, probability distribution, and status.
        """
        # If real trained weights are loaded, run full PyTorch model forward pass
        if self.has_trained_weights:
            seq_len = min(1024, len(signal))
            i_comp = np.real(signal[:seq_len]).astype(np.float32)
            q_comp = np.imag(signal[:seq_len]).astype(np.float32)
            tensor = torch.tensor(np.stack([i_comp, q_comp]), dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            prob_dict = {label: float(probs[i]) for i, label in enumerate(self.class_labels)}
            best_mod = max(prob_dict, key=prob_dict.get)
            return ClassificationResult(
                modulation=best_mod,
                confidence=prob_dict[best_mod],
                probabilities=prob_dict,
                model_status="REAL_TRAINED_MODEL",
                classifier_type="DEEP_1D_CNN",
                evidence=[f"PyTorch 1D-CNN softmax activation peak on {best_mod} ({prob_dict[best_mod]*100:.1f}%)"],
                explanation=f"Deep 1D-CNN forward pass completed with trained weights. Highest softmax activation on {best_mod}."
            )

        # ----------------- TRANSPARENT HEURISTIC FEATURE EVALUATION -----------------
        from src.amc.heuristic_classifier import HeuristicModulationClassifier
        h_res = HeuristicModulationClassifier.classify(signal)

        return ClassificationResult(
            modulation=h_res.predicted_modulation,
            confidence=h_res.confidence,
            probabilities=h_res.candidate_scores,
            model_status=h_res.status,
            classifier_type=h_res.classifier_type,
            evidence=h_res.evidence,
            explanation=h_res.explanation,
            is_comm_like=h_res.is_comm_like,
            model_name="Interpretable Statistical Feature Extractor (CNN Weights Unloaded)"
        )
