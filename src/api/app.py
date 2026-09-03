"""
FastAPI Backend Application Entrypoint.
Provides REST API endpoints for automated RF signal processing, AI modulation classification,
deterministic demodulation, and cyber forensic entropy analysis.
"""

from typing import Dict, List, Optional
import io
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.amc.models.cnn1d_classifier import Modulation1DCNN
from src.decoding.cyber.entropy import PayloadEntropyAnalyzer
from src.decoding.cyber.sync_detector import SyncWordDetector
from src.decoding.fec.reed_solomon import ReedSolomonDecoder
from src.decoding.fec.viterbi import ViterbiDecoder
from src.dsp.demodulators.fsk import FSKDemodulator
from src.dsp.demodulators.psk import PSKDemodulator
from src.dsp.demodulators.qam import QAMDemodulator
from src.ingestion.binary_parser import IQFormat, SignalIngestionEngine
from src.ingestion.normalizer import SignalNormalizer
from src.ingestion.synthetic_generator import SyntheticSignalGenerator

app = FastAPI(
    title="CHROMA CODE: Signal Intelligence & Demodulation Engine",
    version="1.0.0",
    description="Automated RF DSP, Neural AMC, and Cyber Forensics Platform (SIH 2026)"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global AMC classifier instance
amc_classifier = Modulation1DCNN()


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {
        "status": "online",
        "system": "CHROMA CODE RF SIGINT Engine",
        "version": "1.0.0",
        "supported_modulations": ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "2FSK"]
    }


@app.get("/api/v1/presets")
async def get_presets() -> Dict[str, List[str]]:
    return {"presets": list(SyntheticSignalGenerator.PRESETS.keys())}


@app.post("/api/v1/signal/process")
async def process_signal(
    file: Optional[UploadFile] = File(None),
    preset_name: Optional[str] = Form(None),
    fmt: str = Form("float32"),
    sample_rate: float = Form(2000000.0),
    symbol_rate: float = Form(250000.0),
    modulation_override: Optional[str] = Form(None)
):
    """
    End-to-end signal analysis: Ingestion -> AMC -> Demodulation -> FEC -> Cyber Forensics.
    """
    try:
        # Step 1: Ingestion
        if file is not None and file.filename:
            content = await file.read()
            if file.filename.endswith(".wav"):
                raw_signal, fs = SignalIngestionEngine.parse_wav(io.BytesIO(content))
                sample_rate = float(fs)
            else:
                raw_signal = SignalIngestionEngine.parse_iq_stream(content, fmt=IQFormat(fmt))
        elif preset_name:
            raw_signal, meta = SyntheticSignalGenerator.generate_preset(preset_name)
            sample_rate = meta["sample_rate"]
            symbol_rate = meta["symbol_rate"]
        else:
            # Fallback to default preset
            raw_signal, meta = SyntheticSignalGenerator.generate_preset(list(SyntheticSignalGenerator.PRESETS.keys())[0])

        # Step 2: Normalization
        dc_cleaned = SignalNormalizer.remove_dc_offset(raw_signal)
        norm_signal, rms = SignalNormalizer.normalize_unit_power(dc_cleaned)

        # Step 3: AMC Classification
        seq_len = min(1024, len(norm_signal))
        i_tensor = torch.from_numpy(np.real(norm_signal[:seq_len]).astype(np.float32)).unsqueeze(0)
        q_tensor = torch.from_numpy(np.imag(norm_signal[:seq_len]).astype(np.float32)).unsqueeze(0)
        iq_tensor = torch.stack([i_tensor, q_tensor], dim=1)
        amc_probs = amc_classifier.predict_probabilities(iq_tensor)
        predicted_mod = max(amc_probs, key=amc_probs.get)

        selected_mod = modulation_override or predicted_mod

        # Step 4: Demodulation
        if "QAM" in selected_mod.upper():
            order = 64 if "64" in selected_mod else 16
            demod = QAMDemodulator(sample_rate=sample_rate, symbol_rate=symbol_rate, order=order)
            res = demod.demodulate(norm_signal)
        elif "FSK" in selected_mod.upper():
            demod = FSKDemodulator(sample_rate=sample_rate, symbol_rate=symbol_rate, tone_spacing_hz=50000.0)
            res = demod.demodulate(norm_signal)
        else:
            order_map = {"BPSK": 2, "QPSK": 4, "8PSK": 8}
            demod = PSKDemodulator(sample_rate=sample_rate, symbol_rate=symbol_rate, order=order_map.get(selected_mod, 4))
            res = demod.demodulate(norm_signal)

        bits = res.bits if res.bits is not None else np.array([], dtype=np.uint8)

        # Step 5: FEC Decoding
        viterbi = ViterbiDecoder()
        fec_bits, fec_stats = viterbi.decode(bits)

        # Step 6: Cyber Forensics
        sync_word = SyntheticSignalGenerator.SYNC_WORD_32BIT
        sync_matches = SyncWordDetector.find_sync_word(bits, sync_word, max_bit_errors=2)
        shannon_entropy = PayloadEntropyAnalyzer.calculate_shannon_entropy(bits)

        entropy_verdict = (
            "ENCRYPTED_OR_COMPRESSED (High Randomness)" if shannon_entropy > 7.2
            else "STRUCTURED_TELEMETRY (Medium Randomness)" if shannon_entropy > 3.5
            else "REPETITIVE_OR_IDLE (Low Randomness)"
        )

        return {
            "status": "SUCCESS",
            "samples_analyzed": len(norm_signal),
            "rms_power": float(rms),
            "amc_predicted_modulation": predicted_mod,
            "amc_confidence_distribution": amc_probs,
            "selected_modulation": selected_mod,
            "dsp_metrics": {
                "recovered_symbols": len(res.symbols),
                "estimated_snr_db": float(res.estimated_snr_db),
                "carrier_freq_offset_hz": float(res.carrier_freq_offset_hz),
                "phase_offset_rad": float(res.phase_offset_rad)
            },
            "fec_diagnostics": fec_stats,
            "cyber_forensics": {
                "shannon_entropy_bits_per_byte": float(shannon_entropy),
                "entropy_verdict": entropy_verdict,
                "sync_headers_found": len(sync_matches),
                "sync_header_positions": sync_matches[:5]
            },
            "bits_preview": bits[:128].tolist()
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "error_message": str(e),
            "fallback_info": "DSP processing encountered an unexpected format. Please verify sample rate and format settings."
        }
