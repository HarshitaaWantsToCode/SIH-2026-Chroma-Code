"""
Data Ingestion and Binary Parsing Module.

Handles high-throughput ingestion of raw In-phase/Quadrature (.IQ) and .wav RF recordings,
converting byte streams into standardized double-precision/single-precision complex signals:
    s(t) = I(t) + j * Q(t)
"""

from enum import Enum
from pathlib import Path
from typing import BinaryIO, Optional, Tuple, Union
import io


import numpy as np


class IQFormat(str, Enum):
    """Supported binary raw I/Q interleaved formats."""
    FLOAT32 = "float32"       # 32-bit IEEE 754 floating-point per channel (8 bytes/sample)
    FLOAT64 = "float64"       # 64-bit IEEE 754 floating-point per channel (16 bytes/sample)
    INT16 = "int16"           # 16-bit signed integer (e.g., HackRF/BladeRF, 4 bytes/sample)
    INT8 = "int8"             # 8-bit signed integer (e.g., RTL-SDR signed, 2 bytes/sample)
    UINT8 = "uint8"           # 8-bit unsigned integer (e.g., RTL-SDR raw, offset by 127.5)


class SignalIngestionEngine:
    """
    High-performance parser for raw RF signal files.
    """

    @staticmethod
    def parse_iq_stream(
        source: Union[str, Path, BinaryIO, bytes],
        fmt: IQFormat = IQFormat.FLOAT32,
        max_samples: Optional[int] = None,
        offset_samples: int = 0
    ) -> np.ndarray:
        """
        Parses raw interleaved I/Q binary streams into complex NumPy arrays.

        Mathematical Representation:
            Given interleaved samples: [I_0, Q_0, I_1, Q_1, ..., I_N, Q_N]
            Construct: s[n] = I[n] + 1j * Q[n]

        Args:
            source: File path, open binary file-like object, or raw byte buffer.
            fmt: Format specifying byte layout and numeric type.
            max_samples: Maximum number of complex samples to read (None for all).
            offset_samples: Number of complex samples to skip from beginning.

        Returns:
            np.ndarray: 1D complex64/complex128 array representing s(t).

        Raises:
            ValueError: If buffer size is inconsistent with format alignment.
        """
        dtype_map = {
            IQFormat.FLOAT32: np.float32,
            IQFormat.FLOAT64: np.float64,
            IQFormat.INT16: np.int16,
            IQFormat.INT8: np.int8,
            IQFormat.UINT8: np.uint8,
        }
        target_dtype = dtype_map[fmt]
        element_size = np.dtype(target_dtype).itemsize
        bytes_per_complex = element_size * 2

        count = max_samples * 2 if max_samples is not None else -1
        offset_bytes = offset_samples * bytes_per_complex

        if isinstance(source, (str, Path)):
            with open(source, "rb") as f:
                f.seek(offset_bytes)
                raw_data = np.fromfile(f, dtype=target_dtype, count=count)
        elif isinstance(source, bytes):
            avail_bytes = max(0, len(source) - offset_bytes)
            avail_elements = avail_bytes // element_size
            actual_count = avail_elements if count == -1 else min(count, avail_elements)
            raw_data = np.frombuffer(source, dtype=target_dtype, count=actual_count, offset=offset_bytes)
        elif hasattr(source, "read"):
            if offset_bytes > 0:
                source.seek(offset_bytes, io.SEEK_SET)
            buffer = source.read() if count == -1 else source.read(count * element_size)
            avail_elements = len(buffer) // element_size
            actual_count = avail_elements if count == -1 else min(count, avail_elements)
            raw_data = np.frombuffer(buffer, dtype=target_dtype, count=actual_count)
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        # Ensure even count for interleaved I and Q channels
        if len(raw_data) % 2 != 0:
            raw_data = raw_data[:-1]

        # De-interleave I and Q
        i_channel = raw_data[0::2]
        q_channel = raw_data[1::2]

        # Normalization and offset correction
        if fmt == IQFormat.UINT8:
            i_float = (i_channel.astype(np.float32) - 127.5) / 127.5
            q_float = (q_channel.astype(np.float32) - 127.5) / 127.5
        elif fmt == IQFormat.INT16:
            i_float = i_channel.astype(np.float32) / 32768.0
            q_float = q_channel.astype(np.float32) / 32768.0
        elif fmt == IQFormat.INT8:
            i_float = i_channel.astype(np.float32) / 128.0
            q_float = q_channel.astype(np.float32) / 128.0
        else:
            i_float = i_channel.astype(np.float32)
            q_float = q_channel.astype(np.float32)

        # Sanitize NaNs or Infs that can happen if raw bytes are misinterpreted under wrong format
        i_clean = np.nan_to_num(i_float, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
        q_clean = np.nan_to_num(q_float, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

        return i_clean + 1j * q_clean

    @staticmethod
    def parse_wav(
        source: Union[str, Path, BinaryIO],
        max_frames: Optional[int] = None,
        offset_frames: int = 0
    ) -> Tuple[np.ndarray, int]:
        """
        Parses stereo/mono .wav files into complex analytic signal representation.

        If Stereo: Channel 0 is mapped to In-Phase (I), Channel 1 to Quadrature (Q).
        If Mono: Real signal r(t) is transformed to analytic signal via Hilbert Transform:
            s(t) = r(t) + j * H{r(t)}

        Args:
            source: File path or file-like buffer containing WAV audio data.
            max_frames: Number of frames to read.
            offset_frames: Number of frames to skip.

        Returns:
            Tuple[np.ndarray, int]: (Complex analytic signal s(t), Sample rate Fs).
        """
        from scipy.signal import hilbert

        try:
            import soundfile as sf
            with sf.SoundFile(source, mode="r") as sound_file:
                if offset_frames > 0:
                    sound_file.seek(offset_frames)
                data = sound_file.read(frames=max_frames if max_frames is not None else -1, dtype="float32")
                samplerate = sound_file.samplerate
                channels = sound_file.channels
        except (ImportError, Exception):
            from scipy.io import wavfile
            samplerate, raw_wav = wavfile.read(source)
            if raw_wav.dtype == np.int16:
                data = raw_wav.astype(np.float32) / 32768.0
            elif raw_wav.dtype == np.int32:
                data = raw_wav.astype(np.float32) / 2147483648.0
            elif raw_wav.dtype == np.uint8:
                data = (raw_wav.astype(np.float32) - 128.0) / 128.0
            else:
                data = raw_wav.astype(np.float32)

            if offset_frames > 0:
                data = data[offset_frames:]
            if max_frames is not None:
                data = data[:max_frames]
            channels = 1 if data.ndim == 1 else data.shape[1]

        if channels >= 2:
            # Multi-channel: Map Ch0 -> I, Ch1 -> Q
            signal = data[:, 0] + 1j * data[:, 1]
        else:
            # Single-channel: Hilbert transform for single-sideband analytic continuation
            analytic = hilbert(data if data.ndim == 1 else data[:, 0])
            signal = analytic.astype(np.complex64)

        return signal, int(samplerate)
