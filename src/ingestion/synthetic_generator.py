"""
Synthetic RF Signal Generator for Demonstration and Pipeline Testing.

Generates realistic test vectors for various digital modulations (BPSK, QPSK, 8PSK, 16QAM, 64QAM, 2FSK)
with configurable SNR, Carrier Frequency Offset (CFO), phase jitter, pulse shaping, and pre-embedded
sync headers & forensic payload patterns.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from src.dsp.synchronization.rrc_filter import RootRaisedCosineFilter


class SyntheticSignalGenerator:
    """
    Synthesizes standard RF baseband/passband signals for demo and testing.
    """

    # CCSDS Standard 32-bit Frame Synchronization Preamble (0x1ACFFC1D)
    SYNC_WORD_32BIT = np.array([
        0, 0, 0, 1, 1, 0, 1, 0,  # 0x1A
        1, 1, 0, 0, 1, 1, 1, 1,  # 0xCF
        1, 1, 1, 1, 1, 1, 0, 0,  # 0xFC
        0, 0, 0, 1, 1, 1, 0, 1   # 0x1D
    ], dtype=np.uint8)

    PRESETS = {
        "Preset 1: Clean QPSK Telemetry (Satellite Link)": {
            "modulation": "QPSK",
            "snr_db": 22.0,
            "cfo_hz": 1200.0,
            "sample_rate": 2000000.0,
            "symbol_rate": 250000.0,
            "payload_type": "plaintext",
            "text": "MISSION_CONTROL: SAT-9 TELEMETRY OK. ALTITUDE: 420KM. STATUS: NOMINAL."
        },
        "Preset 2: Noisy BPSK Tactical Beacon (Low SNR = 5 dB)": {
            "modulation": "BPSK",
            "snr_db": 5.0,
            "cfo_hz": 3400.0,
            "sample_rate": 2000000.0,
            "symbol_rate": 250000.0,
            "payload_type": "telemetry",
            "text": "BEACON_ID_9941_LOCATION_UNKNOWN_EMERGENCY_PING"
        },
        "Preset 3: High-Order 16-QAM Encrypted Payload (H ~ 7.95)": {
            "modulation": "16-QAM",
            "snr_db": 26.0,
            "cfo_hz": 800.0,
            "sample_rate": 2000000.0,
            "symbol_rate": 250000.0,
            "payload_type": "encrypted",
            "text": "TOP_SECRET_CIPHERTEXT_SIMULATION"
        },
        "Preset 4: 2-FSK Emergency Dispatch Channel": {
            "modulation": "2-FSK",
            "snr_db": 18.0,
            "cfo_hz": 0.0,
            "sample_rate": 2000000.0,
            "symbol_rate": 250000.0,
            "payload_type": "plaintext",
            "text": "DISPATCH_CHANNEL_7: ALL UNITS STANDBY FOR COORDINATES."
        },
        "Preset 5: 8-PSK Aeronautical High-Throughput Stream": {
            "modulation": "8PSK",
            "snr_db": 24.0,
            "cfo_hz": 1850.0,
            "sample_rate": 2000000.0,
            "symbol_rate": 250000.0,
            "payload_type": "telemetry",
            "text": "AIRCRAFT_ADSB_HEX_TRACKING_SQUAWK_7700"
        },
        "Preset 6: AIST-2D Russian Microsatellite (435.315 MHz PM/PCM Telemetry)": {
            "modulation": "BPSK",
            "snr_db": 16.0,
            "cfo_hz": 650.0,
            "sample_rate": 2000000.0,
            "symbol_rate": 250000.0,
            "payload_type": "telemetry",
            "text": "AIST-2D_TELEMETRY: SAMARA_UNIV_BEACON_435.315MHZ_SYS_OK"
        }
    }

    @classmethod
    def generate_preset(cls, preset_key: str, num_symbols: int = 2048) -> Tuple[np.ndarray, Dict]:
        """Generates signal and metadata for a given preset string."""
        if preset_key not in cls.PRESETS:
            preset_key = list(cls.PRESETS.keys())[0]

        cfg = cls.PRESETS[preset_key]
        signal, bits = cls.generate_signal(
            modulation=cfg["modulation"],
            sample_rate=cfg["sample_rate"],
            symbol_rate=cfg["symbol_rate"],
            snr_db=cfg["snr_db"],
            cfo_hz=cfg["cfo_hz"],
            num_symbols=num_symbols,
            payload_type=cfg["payload_type"],
            custom_text=cfg["text"]
        )

        meta = dict(cfg)
        meta["num_symbols"] = num_symbols
        meta["total_samples"] = len(signal)
        meta["ground_truth_bits"] = bits
        return signal, meta

    @classmethod
    def generate_signal(
        cls,
        modulation: str = "QPSK",
        sample_rate: float = 2000000.0,
        symbol_rate: float = 250000.0,
        snr_db: float = 20.0,
        cfo_hz: float = 0.0,
        phase_offset_deg: float = 15.0,
        num_symbols: int = 2048,
        payload_type: str = "plaintext",
        custom_text: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates synthetic complex I/Q samples with known modulation and realistic channel impairments.
        """
        sps = int(round(sample_rate / symbol_rate))
        mod = modulation.upper()

        # Generate binary bits
        if custom_text:
            text_bytes = custom_text.encode("utf-8")
            raw_bits = np.unpackbits(np.frombuffer(text_bytes, dtype=np.uint8))
        else:
            raw_bits = np.random.randint(0, 2, size=num_symbols * 4, dtype=np.uint8)

        # Prepend sync word
        sync = cls.SYNC_WORD_32BIT
        if payload_type == "encrypted":
            # High entropy random payload after sync
            payload_bits = np.random.randint(0, 2, size=num_symbols * 4, dtype=np.uint8)
            full_bits = np.concatenate([sync, payload_bits])
        else:
            # Repeat text bits to reach target length
            repeats = int(np.ceil((num_symbols * 6) / len(raw_bits)))
            tiled = np.tile(raw_bits, repeats)
            full_bits = np.concatenate([sync, tiled])

        # Symbol Mapping
        if mod == "BPSK":
            bits = full_bits[:num_symbols]
            symbols = 2.0 * bits.astype(np.float32) - 1.0 + 0.0j

        elif mod == "QPSK":
            bits = full_bits[: num_symbols * 2]
            b0 = bits[0::2]
            b1 = bits[1::2]
            i_val = 2.0 * b0.astype(np.float32) - 1.0
            q_val = 2.0 * b1.astype(np.float32) - 1.0
            symbols = (i_val + 1j * q_val) / np.sqrt(2.0)

        elif mod == "8PSK":
            bits = full_bits[: num_symbols * 3]
            # Reshape into groups of 3
            groups = bits.reshape(-1, 3)[:num_symbols]
            # Map 3 bits to angle
            int_vals = groups[:, 0] * 4 + groups[:, 1] * 2 + groups[:, 2]
            angles = int_vals * (2 * np.pi / 8.0)
            symbols = np.exp(1j * angles)

        elif mod in ("16QAM", "16-QAM"):
            bits = full_bits[: num_symbols * 4]
            groups = bits.reshape(-1, 4)[:num_symbols]
            
            # Gray map for 4 levels: 00 -> -3, 01 -> -1, 11 -> +1, 10 -> +3
            lvl_map = { (0,0): -3.0, (0,1): -1.0, (1,1): 1.0, (1,0): 3.0 }
            i_vals = np.array([lvl_map[(g[0], g[1])] for g in groups], dtype=np.float32)
            q_vals = np.array([lvl_map[(g[2], g[3])] for g in groups], dtype=np.float32)
            symbols = (i_vals + 1j * q_vals) / np.sqrt(10.0)

        elif mod in ("64QAM", "64-QAM"):
            bits = full_bits[: num_symbols * 6]
            groups = bits.reshape(-1, 6)[:num_symbols]
            lvl8_map = {
                (0,0,0): -7.0, (0,0,1): -5.0, (0,1,1): -3.0, (0,1,0): -1.0,
                (1,1,0): 1.0, (1,1,1): 3.0, (1,0,1): 5.0, (1,0,0): 7.0
            }
            i_vals = np.array([lvl8_map[(g[0], g[1], g[2])] for g in groups], dtype=np.float32)
            q_vals = np.array([lvl8_map[(g[3], g[4], g[5])] for g in groups], dtype=np.float32)
            symbols = (i_vals + 1j * q_vals) / np.sqrt(42.0)

        elif mod in ("2FSK", "2-FSK", "FSK"):
            bits = full_bits[:num_symbols]
            dev_hz = 50000.0  # 50 kHz deviation
            freq_seq = np.where(bits == 1, dev_hz, -dev_hz)
            # Repeat frequency over SPS
            freq_samples = np.repeat(freq_seq, sps)
            phase = np.cumsum(2 * np.pi * freq_samples / sample_rate)
            signal_tx = np.exp(1j * phase)
            symbols = signal_tx[sps // 2 :: sps]
        else:
            # Default to QPSK
            return cls.generate_signal("QPSK", sample_rate, symbol_rate, snr_db, cfo_hz, phase_offset_deg, num_symbols)

        if mod not in ("2FSK", "2-FSK", "FSK"):
            # Upsample by SPS
            upsampled = np.zeros(len(symbols) * sps, dtype=np.complex64)
            upsampled[::sps] = symbols

            # Pulse Shaping (RRC Filter)
            rrc = RootRaisedCosineFilter()
            signal_tx = rrc.apply(upsampled, sps, alpha=0.35)

        # Apply Carrier Frequency Offset (CFO) and Initial Phase Offset
        t = np.arange(len(signal_tx)) / sample_rate
        phase_carrier = 2 * np.pi * cfo_hz * t + np.deg2rad(phase_offset_deg)
        signal_cfo = signal_tx * np.exp(1j * phase_carrier)

        # Add Additive White Gaussian Noise (AWGN)
        sig_pwr = np.mean(np.abs(signal_cfo) ** 2)
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_pwr = sig_pwr / (snr_linear + 1e-12)
        noise = np.sqrt(noise_pwr / 2.0) * (
            np.random.randn(len(signal_cfo)) + 1j * np.random.randn(len(signal_cfo))
        )

        rx_signal = signal_cfo + noise
        return rx_signal.astype(np.complex64), full_bits
