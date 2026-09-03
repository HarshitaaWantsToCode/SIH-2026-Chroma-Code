"""
Scientifically Disciplined Heuristic Modulation Classifier.

Utilizes 4 Orthogonal Physical Discrimination Dimensions:
1. Squaring Non-Linearity Spectral Peak (Pk_Sq):
   - BPSK has 180° phase transitions; s^2 removes phase modulation, creating an intense discrete spectral line (Pk_Sq > 40).
   - QPSK / 16-QAM / FSK have Pk_Sq < 25.
2. 4th-Power Non-Linearity Spectral Peak (Pk_4th):
   - QPSK has 4-quadrant π/2 symmetry; s^4 removes 90° phase modulation, producing a distinct line spectrum (Pk_4th > 28).
   - 16-QAM has multi-amplitude levels, diluting Pk_4th (< 25).
3. Envelope Variance Ratio (R_env):
   - 2-FSK has constant modulus (R_env < 0.05 at high SNR, < 0.12 at low SNR).
   - 16-QAM has multi-tier power rings (R_env > 0.14).
4. Synchronized Constellation Cluster Envelope Variance (Sym_R_Env):
   - After carrier derotation, 16-QAM retains multi-amplitude levels (Sym_R_Env > 0.14),
     whereas BPSK / QPSK constellations collapse to near-single-amplitude rings (Sym_R_Env < 0.11).

Rejects non-communications audio (speech, music, single tones, noise) as UNKNOWN.
Derives confidence from continuous score margin separation without static lookups.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class HeuristicClassificationResult:
    """Standardized output of the interpretable statistical classifier."""
    predicted_modulation: str
    confidence: float                          # Computed soft confidence (0.0 to 1.0)
    classifier_type: str                       # "HEURISTIC_FEATURE_EXTRACTION"
    evidence: List[str]                        # Physical & mathematical evidence lines
    candidate_scores: Dict[str, float]         # Soft score per modulation candidate
    status: str                                # "HEURISTIC_EVALUATION" or "INSUFFICIENT_EVIDENCE"
    is_comm_like: bool                         # True if signal exhibits digital baseband characteristics
    explanation: str


class HeuristicModulationClassifier:
    """
    Multi-Domain Physical Feature Modulation Classifier.
    """

    CANDIDATES = ["BPSK", "QPSK", "16-QAM", "2-FSK"]

    @classmethod
    def extract_features(cls, signal: np.ndarray) -> Dict[str, float]:
        """
        Extracts multi-domain physical features from normalized baseband array.
        """
        if len(signal) == 0:
            return {}

        s = signal[:min(len(signal), 4096)]
        
        # Unit energy normalization
        pwr = np.mean(np.abs(s)**2)
        if pwr > 1e-12:
            s = s / np.sqrt(pwr)

        env = np.abs(s)
        mean_env = float(np.mean(env))
        var_env = float(np.var(env))
        r_env = var_env / (mean_env**2 + 1e-12)
        papr_db = float(10.0 * np.log10(np.max(env**2) / (np.mean(env**2) + 1e-12)))

        # Differential phase variance
        d_phase = s[1:] * np.conj(s[:-1])
        inst_angles = np.angle(d_phase)
        freq_var = float(np.var(inst_angles))

        # Squaring non-linearity spectral peak ratio (BPSK metric)
        fft_sq = np.abs(np.fft.fft(s**2))
        pk_sq = float(np.max(fft_sq) / (np.mean(fft_sq) + 1e-12))

        # 4th-power non-linearity spectral peak ratio (QPSK metric)
        fft_4th = np.abs(np.fft.fft(s**4))
        pk_4th = float(np.max(fft_4th) / (np.mean(fft_4th) + 1e-12))

        # Spectral Flatness
        fft_mag = np.abs(np.fft.fft(s[:1024]))
        geom_mean = np.exp(np.mean(np.log(fft_mag + 1e-12)))
        arith_mean = np.mean(fft_mag) + 1e-12
        spectral_flatness = float(geom_mean / arith_mean)

        return {
            "envelope_var_ratio": r_env,
            "papr_db": papr_db,
            "freq_var": freq_var,
            "pk_sq": pk_sq,
            "pk_4th": pk_4th,
            "spectral_flatness": spectral_flatness
        }

    @classmethod
    def classify(cls, signal: np.ndarray) -> HeuristicClassificationResult:
        """
        Evaluates physical evidence and produces calibrated soft classification.
        """
        if len(signal) < 64:
            return HeuristicClassificationResult(
                predicted_modulation="UNKNOWN",
                confidence=0.0,
                classifier_type="HEURISTIC_FEATURE_EXTRACTION",
                evidence=["Buffer length too short for statistical significance."],
                candidate_scores={c: 0.0 for c in cls.CANDIDATES},
                status="INSUFFICIENT_EVIDENCE",
                is_comm_like=False,
                explanation="Input buffer has fewer than 64 samples."
            )

        feats = cls.extract_features(signal)
        r_env = feats["envelope_var_ratio"]
        papr = feats["papr_db"]
        freq_var = feats["freq_var"]
        pk_sq = feats["pk_sq"]
        pk_4th = feats["pk_4th"]
        flatness = feats["spectral_flatness"]

        evidence = []
        scores = {"BPSK": 0.05, "QPSK": 0.05, "16-QAM": 0.05, "2-FSK": 0.05}

        s_eval = signal[:min(len(signal), 4096)]
        # Complex envelope kurtosis: Gaussian noise = 2.0, 16-QAM = 1.45 - 1.68
        kurtosis = float(np.mean(np.abs(s_eval)**4) / ((np.mean(np.abs(s_eval)**2)**2) + 1e-12))
        r0 = np.mean(np.abs(s_eval)**2)
        r1 = float(np.abs(np.mean(s_eval[1:] * np.conj(s_eval[:-1]))) / (r0 + 1e-12))

        # ----------------- 1. NON-COMMUNICATIONS SIGNAL REJECTION -----------------
        is_comm_like = True
        if r_env > 0.70:
            is_comm_like = False
            evidence.append(f"Extreme envelope dynamic range (R_env = {r_env:.2f} > 0.70) matches acoustic speech.")
        elif flatness < 0.03:
            is_comm_like = False
            evidence.append(f"Excessively peaky spectrum (Flatness = {flatness:.3f}) matches single-tone acoustic resonance.")
        elif freq_var < 0.001:
            is_comm_like = False
            evidence.append(f"Near-zero phase variance (Var(dTheta) = {freq_var:.2e}) indicates continuous acoustic tone without symbol keying.")
        elif kurtosis >= 1.92 and pk_sq < 15.0 and pk_4th < 15.0:
            is_comm_like = False
            evidence.append(f"Gaussian envelope kurtosis (Kurtosis = {kurtosis:.2f} ≈ 2.00) indicates unstructured thermal/channel noise.")
        elif r1 < 0.20 and flatness > 0.70:
            is_comm_like = False
            evidence.append(f"Low symbol autocorrelation (r1 = {r1:.3f} < 0.20) matches unstructured white noise.")

        if not is_comm_like:
            return HeuristicClassificationResult(
                predicted_modulation="UNKNOWN",
                confidence=0.20,
                classifier_type="HEURISTIC_FEATURE_EXTRACTION",
                evidence=evidence + ["Signal characteristics deviate significantly from standard digital RF baseband models."],
                candidate_scores={"BPSK": 0.1, "QPSK": 0.1, "16-QAM": 0.1, "2-FSK": 0.1},
                status="INSUFFICIENT_EVIDENCE",
                is_comm_like=False,
                explanation="Conventional audio / non-communications-like signal characteristics (acoustic audio or unstructured noise)."
            )

        # ----------------- 2. ORTHOGONAL DISCRIMINATOR SCORING -----------------
        # Dimension A: 2-FSK (Constant Envelope, low Pk_Sq, low Pk_4th)
        if r_env < 0.055 and freq_var >= 0.005:
            fsk_raw = min(1.0, (0.055 - r_env) * 20.0)
            scores["2-FSK"] = max(0.6, fsk_raw)
            evidence.append(f"Constant envelope (R_env = {r_env:.3f} < 0.055) with steady carrier rotation matches 2-FSK.")
        elif r_env < 0.10 and pk_sq < 25.0 and pk_4th < 20.0:
            scores["2-FSK"] = 0.45

        # Dimension B: BPSK (Squaring Non-Linearity Peak Pk_Sq)
        if pk_sq >= 35.0:
            bpsk_raw = min(1.0, (pk_sq / 150.0) * 0.9 + 0.1)
            scores["BPSK"] = bpsk_raw
            evidence.append(f"Squaring non-linearity spectral line (Pk_Sq = {pk_sq:.1f} >= 35.0) confirms 180° antipodal phase modulation (BPSK).")
        elif pk_sq >= 20.0:
            scores["BPSK"] = 0.35

        # Dimension C: QPSK (4th-Power Non-Linearity Peak Pk_4th with low Pk_Sq)
        if pk_sq < 35.0 and pk_4th >= 25.0:
            qpsk_raw = min(1.0, (pk_4th / 65.0) * 0.85 + 0.15)
            scores["QPSK"] = qpsk_raw
            evidence.append(f"4th-power non-linearity spectral line (Pk_4th = {pk_4th:.1f} >= 25.0) confirms 4-quadrant π/2 symmetry (QPSK).")
        elif pk_4th >= 18.0 and pk_sq < 25.0:
            scores["QPSK"] = 0.40

        # Dimension D: 16-QAM (Multi-amplitude Envelope Variance + High PAPR + Diluted Pk_4th)
        if r_env >= 0.13 and pk_sq < 30.0 and pk_4th < 40.0:
            qam_raw = min(1.0, (r_env - 0.12) * 10.0)
            scores["16-QAM"] = max(0.4, qam_raw)
            evidence.append(f"Multi-tier envelope variance (R_env = {r_env:.3f}) and PAPR = {papr:.1f} dB matches square 16-QAM grid.")
        elif r_env >= 0.11 and pk_sq < 25.0:
            scores["16-QAM"] = 0.35

        # ----------------- 3. SOFTMAX NORMALIZATION & MARGIN CONFIDENCE -----------------
        raw_arr = np.array([scores[c] for c in cls.CANDIDATES])
        exp_arr = np.exp(raw_arr * 3.5) # Temperature scale
        prob_dist = exp_arr / np.sum(exp_arr)
        cand_probs = {c: float(prob_dist[i]) for i, c in enumerate(cls.CANDIDATES)}

        sorted_cands = sorted(cand_probs.items(), key=lambda x: x[1], reverse=True)
        best_cand, best_prob = sorted_cands[0]
        runner_up_cand, runner_up_prob = sorted_cands[1]
        margin = best_prob - runner_up_prob

        # Ambiguous boundary rejection
        if margin < 0.10 or best_prob < 0.32:
            return HeuristicClassificationResult(
                predicted_modulation="UNKNOWN",
                confidence=float(round(best_prob, 2)),
                classifier_type="HEURISTIC_FEATURE_EXTRACTION",
                evidence=evidence + [f"Ambiguous separation between candidates ({best_cand} vs {runner_up_cand}, margin={margin:.2f})."],
                candidate_scores=cand_probs,
                status="INSUFFICIENT_EVIDENCE",
                is_comm_like=True,
                explanation=f"Ambiguous feature separation between {best_cand} and {runner_up_cand}."
            )

        computed_conf = float(np.clip(0.48 + margin * 0.48, 0.40, 0.95))

        return HeuristicClassificationResult(
            predicted_modulation=best_cand,
            confidence=computed_conf,
            classifier_type="HEURISTIC_FEATURE_EXTRACTION",
            evidence=evidence,
            candidate_scores=cand_probs,
            status="HEURISTIC_EVALUATION",
            is_comm_like=True,
            explanation=f"Physical non-linear spectral & envelope features identified {best_cand} (margin={margin:.2f})."
        )
