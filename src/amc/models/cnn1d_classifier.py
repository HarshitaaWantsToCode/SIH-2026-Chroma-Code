"""
Automatic Modulation Classification (AMC) Neural Network & Classifier Service.

Provides:
- Modulation1DCNN: PyTorch 1D-CNN Architecture.
- ClassificationResult: Standardized prediction dataclass.
- ModulationClassifier: Resilient Service abstraction executing real neural inference
  when trained PyTorch weights exist, or falling back cleanly to interpretable heuristic evaluation.
"""

from dataclasses import dataclass, field
import math
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


CNN_SUPPORTED_CLASSES = ["BPSK", "QPSK", "16-QAM", "2-FSK"]
DEFAULT_WINDOW_LENGTH = 4096


@dataclass
class ClassificationResult:
    """Standardized result returned by ModulationClassifier."""
    modulation: str
    confidence: float                          # Float between 0.0 and 1.0
    probabilities: Dict[str, float]            # Mapping like {"QPSK": 0.92, "BPSK": 0.04, ...}
    model_status: str                          # "REAL_TRAINED_MODEL", "HEURISTIC_EVALUATION", "INSUFFICIENT_EVIDENCE", "OUT_OF_DISTRIBUTION"
    classifier_type: str                       # "DEEP_1D_CNN" or "HEURISTIC_FEATURE_EXTRACTION"
    evidence: List[str] = field(default_factory=list)
    explanation: str = ""
    is_comm_like: bool = True
    model_name: str = "1D-CNN Modulation Classifier"
    architecture_summary: Dict[str, str] = field(default_factory=lambda: {
        "Layer 1": "Conv1D(in=2, out=64, k=7, p=3) + BatchNorm1D + ReLU + MaxPool1D(2)",
        "Layer 2": "Conv1D(in=64, out=128, k=5, p=2) + BatchNorm1D + ReLU + MaxPool1D(2)",
        "Layer 3": "Conv1D(in=128, out=128, k=3, p=1) + BatchNorm1D + ReLU + MaxPool1D(2)",
        "Layer 4": "Conv1D(in=128, out=256, k=3, p=1) + BatchNorm1D + ReLU + AdaptiveAvgPool1D(16)",
        "Classifier Head": "Flatten(4096) -> Dropout(0.4) -> Linear(4096, 128) -> ReLU -> Dropout(0.2) -> Linear(128, 4)"
    })


if TORCH_AVAILABLE:
    class Modulation1DCNN(nn.Module):
        """
        1D-CNN Feature Extractor and Softmax Classifier for raw I/Q arrays.
        Input Tensor Shape: [Batch_Size, 2, Sequence_Length] (Ch0: In-Phase I, Ch1: Quadrature Q)
        """

        def __init__(
            self,
            sequence_length: int = DEFAULT_WINDOW_LENGTH,
            num_classes: int = 4,
            class_labels: Optional[List[str]] = None
        ) -> None:
            super().__init__()
            self.sequence_length = sequence_length
            self.num_classes = num_classes
            self.class_labels = class_labels or CNN_SUPPORTED_CLASSES

            # Feature Extraction Layers
            self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=7, padding=3)
            self.bn1 = nn.BatchNorm1d(64)

            self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
            self.bn2 = nn.BatchNorm1d(128)

            self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm1d(128)

            self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
            self.bn4 = nn.BatchNorm1d(256)

            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
            self.adapt_pool = nn.AdaptiveAvgPool1d(16)
            self.dropout1 = nn.Dropout(p=0.4)
            self.dropout2 = nn.Dropout(p=0.2)

            # Classification Head: 256 * 16 = 4096
            self.fc1 = nn.Linear(256 * 16, 128)
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = F.relu(self.bn4(self.conv4(x)))
            x = self.adapt_pool(x)
            x = torch.flatten(x, start_dim=1)
            x = self.dropout1(x)
            x = F.relu(self.fc1(x))
            x = self.dropout2(x)
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
    If weights exist -> Executes real neural inference with OOD/confidence gating.
    Otherwise -> Executes interpretable statistical feature classifier.
    """

    def __init__(self, weights_path: Optional[Union[str, Path]] = None) -> None:
        self.class_labels = list(CNN_SUPPORTED_CLASSES)
        self.weights_path = Path(weights_path) if weights_path else Path("models/amc_1dcnn_weights.pt")
        self.has_trained_weights = False
        self.model = None
        self.window_length = DEFAULT_WINDOW_LENGTH
        self.model_metadata = {}

        if TORCH_AVAILABLE and self.weights_path.exists():
            try:
                checkpoint = torch.load(self.weights_path, map_location="cpu", weights_only=False)
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                    self.class_labels = checkpoint.get("class_labels", self.class_labels)
                    self.window_length = checkpoint.get("window_length", DEFAULT_WINDOW_LENGTH)
                    self.model_metadata = checkpoint
                else:
                    state_dict = checkpoint

                self.model = Modulation1DCNN(
                    sequence_length=self.window_length,
                    num_classes=len(self.class_labels),
                    class_labels=self.class_labels
                )
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
            signal: 1D complex NumPy array representing baseband/analytic s(t) = I(t) + j*Q(t).
            demo_modulation_hint: Unused (maintained for API backward compatibility).

        Returns:
            ClassificationResult containing dominant modulation, confidence, probability distribution, and status.
        """
        from src.amc.heuristic_classifier import HeuristicModulationClassifier

        # ----------------- 1. INPUT VALIDATION & DISCRIMINATOR / AUDIO CHECK -----------------
        if len(signal) < 64:
            return ClassificationResult(
                modulation="UNKNOWN",
                confidence=0.0,
                probabilities={c: 0.25 for c in self.class_labels},
                model_status="INSUFFICIENT_EVIDENCE",
                classifier_type="HEURISTIC_FEATURE_EXTRACTION",
                evidence=["Signal duration too short for classification (< 64 samples)."],
                explanation="Input buffer has fewer than 64 samples.",
                is_comm_like=False,
                model_name="Signal Length Check"
            )

        # Check for mono discriminator audio / telemetry capture (e.g. AIST-2D.wav) or non-comm audio
        h_res = HeuristicModulationClassifier.classify(signal)
        
        # If signal is rejected by comms gate or is specifically a mono discriminator telemetry capture
        is_telemetry_like = "TELEMETRY" in h_res.predicted_modulation or "PM/PCM" in h_res.explanation
        if (not h_res.is_comm_like) or is_telemetry_like:
            return ClassificationResult(
                modulation=h_res.predicted_modulation,
                confidence=h_res.confidence,
                probabilities=h_res.candidate_scores,
                model_status=h_res.status,
                classifier_type=h_res.classifier_type,
                evidence=h_res.evidence,
                explanation=h_res.explanation,
                is_comm_like=h_res.is_comm_like,
                model_name="Signal-Derived Heuristic Telemetry/Audio Analysis"
            )

        # ----------------- 2. REAL TRAINED 1D-CNN INFERENCE -----------------
        if self.has_trained_weights and self.model is not None:
            # Deterministic 2-channel tensor conditioning [2, window_length]
            tensor_np = self._preprocess_signal(signal, target_length=self.window_length)
            tensor = torch.from_numpy(tensor_np).unsqueeze(0).float()

            with torch.no_grad():
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            prob_dict = {label: float(probs[i]) for i, label in enumerate(self.class_labels)}
            best_mod = max(prob_dict, key=prob_dict.get)
            max_prob = float(prob_dict[best_mod])

            # Calculate predictive entropy (in bits)
            entropy = -sum(p * math.log2(p + 1e-12) for p in probs)
            max_entropy = math.log2(len(self.class_labels)) # 2.0 bits for 4 classes

            # Conservative Confidence Gate for Out-of-Distribution / Unknown
            if max_prob < 0.45 or entropy > 1.85:
                return ClassificationResult(
                    modulation="UNKNOWN",
                    confidence=max_prob,
                    probabilities=prob_dict,
                    model_status="OUT_OF_DISTRIBUTION",
                    classifier_type="DEEP_1D_CNN",
                    evidence=[
                        f"High predictive entropy ({entropy:.2f} / {max_entropy:.2f} bits) exceeds confidence threshold.",
                        f"Max softmax probability on {best_mod} ({max_prob*100:.1f}%) is below decision margin."
                    ],
                    explanation="1D-CNN produced ambiguous class activations consistent with Out-of-Distribution or low-SNR signal.",
                    is_comm_like=True,
                    model_name="Trained 1D-CNN (Conservative Gate)"
                )

            return ClassificationResult(
                modulation=best_mod,
                confidence=max_prob,
                probabilities=prob_dict,
                model_status="REAL_TRAINED_MODEL",
                classifier_type="DEEP_1D_CNN",
                evidence=[
                    f"PyTorch 1D-CNN softmax activation peak on {best_mod} ({max_prob*100:.1f}%)",
                    f"Softmax entropy = {entropy:.2f} bits (Max = {max_entropy:.2f} bits)",
                    f"Processed input window length N = {self.window_length} complex I/Q samples"
                ],
                explanation=f"1D-CNN inference completed with trained weights. Peak softmax probability on {best_mod} ({max_prob*100:.1f}%).",
                is_comm_like=True,
                model_name="Trained 1D-CNN Feature Extractor"
            )

        # ----------------- 3. FALLBACK HEURISTIC EVALUATION -----------------
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

    @staticmethod
    def _preprocess_signal(signal: np.ndarray, target_length: int = DEFAULT_WINDOW_LENGTH) -> np.ndarray:
        """
        Conditions raw complex signal to [2, target_length] float32 array.
        """
        # Unit energy normalization
        pwr = np.mean(np.abs(signal) ** 2)
        if pwr > 1e-12:
            s = signal / np.sqrt(pwr)
        else:
            s = signal

        sig_len = len(s)
        if sig_len < target_length:
            padded = np.zeros(target_length, dtype=np.complex64)
            padded[:sig_len] = s
            s = padded
        elif sig_len > target_length:
            s = s[:target_length]

        i_ch = np.real(s).astype(np.float32)
        q_ch = np.imag(s).astype(np.float32)

        win_pwr = np.mean(i_ch**2 + q_ch**2)
        if win_pwr > 1e-12:
            scale = 1.0 / np.sqrt(win_pwr)
            i_ch = i_ch * scale
            q_ch = q_ch * scale

        return np.stack([i_ch, q_ch], axis=0)
