"""
Viterbi Maximum Likelihood Decoder for Convolutional Codes.

Implements mathematically rigorous Trellis construction, Branch Metric Calculation,
Add-Compare-Select (ACS), Survivor Path Storage, and Traceback.

Default Standard Configuration:
- Rate R = 1/2
- Constraint Length K = 7
- Generator Polynomials: G1 = 171 (octal) = 0b1111001, G2 = 133 (octal) = 0b1011011
- Trellis States: 2^(K-1) = 64 states
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class ViterbiResult:
    """Structured result returned by ViterbiDecoder."""
    decoded_bits: np.ndarray
    input_bit_count: int
    output_bit_count: int
    path_metric: float
    traceback_depth: int
    terminated: bool
    mode: str
    success: bool
    estimated_bit_errors: int
    diagnostics: Dict[str, Union[int, float, str]]


class ConvolutionalEncoder:
    """
    Reference Rate 1/2 Convolutional Encoder with constraint length K=7.
    Standard polynomials: G1 = 171 (octal), G2 = 133 (octal).
    
    Trellis State Encoding:
        State is represented by the 6 shift-register delay stages: (s5, s4, s3, s2, s1, s0).
        Input bit 'b' enters at the most significant or least significant position.
        Here we adopt the standard convention:
            Shift register holds: [b, s5, s4, s3, s2, s1, s0] (total K=7 bits)
            g1_bits = (1, 1, 1, 1, 0, 0, 1) -> octal 171
            g2_bits = (1, 0, 1, 1, 0, 1, 1) -> octal 133
            out1 = sum(g1_i * stage_i) mod 2
            out2 = sum(g2_i * stage_i) mod 2
            Next state = (b << 5) | (current_state >> 1)  OR ( (current_state << 1) | b ) & 0x3F.
    """

    def __init__(self, constraint_length: int = 7, polynomials: Tuple[int, int] = (0o171, 0o133)) -> None:
        self.k = constraint_length
        self.num_states = 1 << (self.k - 1)  # 64 states for K=7
        self.g1 = polynomials[0]
        self.g2 = polynomials[1]

    def encode(self, bits: Union[np.ndarray, List[int]], terminate: bool = True) -> np.ndarray:
        """
        Encodes a sequence of input message bits.
        
        Args:
            bits: 1D array of binary bits (0 or 1).
            terminate: If True, appends (K-1) zero bits to flush encoder back to state 0.

        Returns:
            np.ndarray: 1D array of encoded bits with length 2 * (len(bits) + (K-1 if terminate else 0)).
        """
        bits = np.asarray(bits, dtype=np.uint8)
        if np.any((bits != 0) & (bits != 1)):
            raise ValueError("Input bits must be binary {0, 1}.")

        seq = list(bits)
        if terminate:
            seq.extend([0] * (self.k - 1))

        coded = []
        state = 0  # 6-bit state: [s5 s4 s3 s2 s1 s0]

        for bit in seq:
            # Shift register content with new input bit at MSB (bit 6)
            sr = (int(bit) << (self.k - 1)) | state
            # Compute parity with G1 and G2 masks
            out1 = bin(sr & self.g1).count("1") % 2
            out2 = bin(sr & self.g2).count("1") % 2
            coded.extend([out1, out2])
            # State transitions: shift right, insert input bit at MSB
            state = sr >> 1

        return np.array(coded, dtype=np.uint8)


class ViterbiDecoder:
    """
    Viterbi Maximum Likelihood Sequence Estimator (MLSE) Decoder.
    Supports:
    - Hard-decision decoding (Hamming branch metric).
    - Soft-decision decoding (Signed LLR / Euclidean distance).
    - Configurable traceback depth.
    - Terminated trellis or continuous best-path traceback.
    """

    def __init__(
        self,
        constraint_length: int = 7,
        polynomials: Tuple[int, int] = (0o171, 0o133),
        traceback_depth: int = 35
    ) -> None:
        if constraint_length < 3 or constraint_length > 9:
            raise ValueError(f"Constraint length {constraint_length} out of supported range [3..9].")
        
        self.k = constraint_length
        self.num_states = 1 << (self.k - 1)  # 64 states for K=7
        self.g1 = polynomials[0]
        self.g2 = polynomials[1]
        self.traceback_depth = traceback_depth

        # Build Trellis Tables:
        # For each state (0..63) and input bit (0 or 1):
        # next_state[state][bit]
        # output_symbols[state][bit] = (c0, c1)
        # prev_states[next_state] = [(state, bit, (c0, c1)), ...]
        self._build_trellis()

    def _build_trellis(self) -> None:
        """Constructs forward and backward state transition lookup matrices."""
        self.next_state = np.zeros((self.num_states, 2), dtype=np.int32)
        self.outputs = np.zeros((self.num_states, 2, 2), dtype=np.uint8)
        # predecessors[next_state] = list of (prev_state, input_bit, (out0, out1))
        self.predecessors: List[List[Tuple[int, int, Tuple[int, int]]]] = [[] for _ in range(self.num_states)]

        for s in range(self.num_states):
            for bit in (0, 1):
                sr = (bit << (self.k - 1)) | s
                out1 = bin(sr & self.g1).count("1") % 2
                out2 = bin(sr & self.g2).count("1") % 2
                ns = sr >> 1
                self.next_state[s, bit] = ns
                self.outputs[s, bit] = (out1, out2)
                self.predecessors[ns].append((s, bit, (out1, out2)))

    def decode(
        self,
        coded_bits: Union[np.ndarray, List[int]],
        terminated: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Union[int, float, str]]]:
        """
        Hard-decision Viterbi decoding matching standard project signature.

        Args:
            coded_bits: 1D array of received encoded bits (0 or 1). Length must be even.
            terminated: True if encoder was flushed with (K-1) zeros at the end.

        Returns:
            Tuple[np.ndarray, dict]: (decoded_bits, diagnostic_stats)
        """
        res = self.decode_hard(coded_bits, terminated=terminated)
        return res.decoded_bits, res.diagnostics

    def decode_hard(
        self,
        coded_bits: Union[np.ndarray, List[int]],
        terminated: bool = False
    ) -> ViterbiResult:
        """
        Executes mathematically rigorous Hard-Decision Viterbi Search with Hamming branch metrics.
        """
        bits = np.asarray(coded_bits, dtype=np.uint8)
        if bits.ndim != 1:
            raise ValueError(f"Coded bits must be 1D, got {bits.shape}.")
        if len(bits) % 2 != 0:
            raise ValueError(f"Rate 1/2 Viterbi requires an even number of coded bits, got {len(bits)}.")
        if np.any((bits != 0) & (bits != 1)):
            raise ValueError("Input bits must be strictly binary {0, 1}.")

        num_steps = len(bits) // 2
        if num_steps == 0:
            return ViterbiResult(
                decoded_bits=np.empty(0, dtype=np.uint8),
                input_bit_count=0,
                output_bit_count=0,
                path_metric=0.0,
                traceback_depth=self.traceback_depth,
                terminated=terminated,
                mode="hard_decision",
                success=True,
                estimated_bit_errors=0,
                diagnostics={"status": "EMPTY_INPUT"}
            )

        # Path metrics initialization:
        # State 0 starts with metric 0; all other states start with infinity (large integer)
        INF = 1000000000
        path_metrics = np.full(self.num_states, INF, dtype=np.int32)
        path_metrics[0] = 0

        # Traceback matrix: stores predecessor state and input bit for each step and state
        # survivor_prev_state[t, s] = previous state that gave minimal metric
        # survivor_input_bit[t, s] = input bit that caused transition
        survivor_prev_state = np.zeros((num_steps, self.num_states), dtype=np.int32)
        survivor_input_bit = np.zeros((num_steps, self.num_states), dtype=np.uint8)

        # Trellis Add-Compare-Select (ACS) loop
        for t in range(num_steps):
            r0 = int(bits[2 * t])
            r1 = int(bits[2 * t + 1])
            new_path_metrics = np.full(self.num_states, INF, dtype=np.int32)

            for ns in range(self.num_states):
                # Check all incoming candidate branches to state 'ns'
                best_metric = INF
                best_prev = 0
                best_bit = 0

                for ps, bit, (e0, e1) in self.predecessors[ns]:
                    prev_m = path_metrics[ps]
                    if prev_m >= INF:
                        continue
                    # Branch metric: Hamming distance
                    bm = (r0 ^ e0) + (r1 ^ e1)
                    cand_m = prev_m + bm
                    if cand_m < best_metric:
                        best_metric = cand_m
                        best_prev = ps
                        best_bit = bit

                new_path_metrics[ns] = best_metric
                survivor_prev_state[t, ns] = best_prev
                survivor_input_bit[t, ns] = best_bit

            path_metrics = new_path_metrics

        # Traceback selection:
        # If terminated, final state is guaranteed to be state 0.
        # Otherwise, choose state with minimum accumulated path metric.
        if terminated:
            curr_state = 0
            final_metric = path_metrics[0]
            if final_metric >= INF:
                # Fallback to min state if state 0 not reached
                curr_state = int(np.argmin(path_metrics))
                final_metric = path_metrics[curr_state]
        else:
            curr_state = int(np.argmin(path_metrics))
            final_metric = path_metrics[curr_state]

        # Traceback backwards through time
        decoded = np.empty(num_steps, dtype=np.uint8)
        for t in range(num_steps - 1, -1, -1):
            bit = survivor_input_bit[t, curr_state]
            prev_s = survivor_prev_state[t, curr_state]
            decoded[t] = bit
            curr_state = prev_s

        # If terminated, the last (K-1) bits were known flush zeros
        if terminated:
            flush_len = self.k - 1
            if len(decoded) >= flush_len:
                output_bits = decoded[:-flush_len]
            else:
                output_bits = decoded
        else:
            output_bits = decoded

        stats: Dict[str, Union[int, float, str]] = {
            "rate": "1/2",
            "constraint_length": self.k,
            "polynomials": f"G1={oct(self.g1)}, G2={oct(self.g2)}",
            "input_coded_bits": len(bits),
            "output_decoded_bits": len(output_bits),
            "accumulated_path_metric": int(final_metric),
            "traceback_steps": num_steps,
            "terminated_mode": str(terminated),
            "status": "CONVERGED_OPTIMAL" if final_metric < INF else "NO_VALID_PATH"
        }

        return ViterbiResult(
            decoded_bits=output_bits,
            input_bit_count=len(bits),
            output_bit_count=len(output_bits),
            path_metric=float(final_metric),
            traceback_depth=self.traceback_depth,
            terminated=terminated,
            mode="hard_decision",
            success=(final_metric < INF),
            estimated_bit_errors=int(final_metric),
            diagnostics=stats
        )

    def decode_soft(
        self,
        soft_symbols: Union[np.ndarray, List[float]],
        terminated: bool = False
    ) -> ViterbiResult:
        """
        Soft-decision Viterbi decoding.
        
        Input Convention:
            soft_symbols contains signed Log-Likelihood Ratios (LLR) or normalized soft metrics:
            - Positive value (> 0): represents belief towards bit 0.
            - Negative value (< 0): represents belief towards bit 1.
            - Magnitude (|x|): confidence of the observation.
            
            Soft branch metric for symbol r and expected bit e in {0, 1}:
            expected symbol s = +1 if e == 0 else -1
            Distance = (r - s)^2 OR Correlation Metric = - r * s.
        """
        symbols = np.asarray(soft_symbols, dtype=np.float64)
        if len(symbols) % 2 != 0:
            raise ValueError(f"Soft Viterbi requires an even number of symbols, got {len(symbols)}.")

        num_steps = len(symbols) // 2
        if num_steps == 0:
            return ViterbiResult(
                decoded_bits=np.empty(0, dtype=np.uint8),
                input_bit_count=0,
                output_bit_count=0,
                path_metric=0.0,
                traceback_depth=self.traceback_depth,
                terminated=terminated,
                mode="soft_decision",
                success=True,
                estimated_bit_errors=0,
                diagnostics={"status": "EMPTY_INPUT"}
            )

        INF = 1e9
        path_metrics = np.full(self.num_states, INF, dtype=np.float64)
        path_metrics[0] = 0.0

        survivor_prev_state = np.zeros((num_steps, self.num_states), dtype=np.int32)
        survivor_input_bit = np.zeros((num_steps, self.num_states), dtype=np.uint8)

        for t in range(num_steps):
            r0 = symbols[2 * t]
            r1 = symbols[2 * t + 1]
            new_path_metrics = np.full(self.num_states, INF, dtype=np.float64)

            for ns in range(self.num_states):
                best_metric = INF
                best_prev = 0
                best_bit = 0

                for ps, bit, (e0, e1) in self.predecessors[ns]:
                    prev_m = path_metrics[ps]
                    if prev_m >= INF:
                        continue
                    # Map expected bit {0, 1} to ideal BPSK level {+1.0, -1.0}
                    s0 = 1.0 if e0 == 0 else -1.0
                    s1 = 1.0 if e1 == 0 else -1.0
                    
                    # Squared Euclidean distance branch metric
                    bm = (r0 - s0) ** 2 + (r1 - s1) ** 2
                    cand_m = prev_m + bm
                    if cand_m < best_metric:
                        best_metric = cand_m
                        best_prev = ps
                        best_bit = bit

                new_path_metrics[ns] = best_metric
                survivor_prev_state[t, ns] = best_prev
                survivor_input_bit[t, ns] = best_bit

            path_metrics = new_path_metrics

        if terminated:
            curr_state = 0
            final_metric = path_metrics[0]
            if final_metric >= INF:
                curr_state = int(np.argmin(path_metrics))
                final_metric = path_metrics[curr_state]
        else:
            curr_state = int(np.argmin(path_metrics))
            final_metric = path_metrics[curr_state]

        decoded = np.empty(num_steps, dtype=np.uint8)
        for t in range(num_steps - 1, -1, -1):
            bit = survivor_input_bit[t, curr_state]
            prev_s = survivor_prev_state[t, curr_state]
            decoded[t] = bit
            curr_state = prev_s

        if terminated:
            flush_len = self.k - 1
            output_bits = decoded[:-flush_len] if len(decoded) >= flush_len else decoded
        else:
            output_bits = decoded

        stats: Dict[str, Union[int, float, str]] = {
            "rate": "1/2",
            "constraint_length": self.k,
            "polynomials": f"G1={oct(self.g1)}, G2={oct(self.g2)}",
            "input_soft_symbols": len(symbols),
            "output_decoded_bits": len(output_bits),
            "accumulated_euclidean_metric": float(final_metric),
            "mode": "SOFT_DECISION_LLR",
            "status": "CONVERGED_OPTIMAL" if final_metric < INF else "NO_VALID_PATH"
        }

        return ViterbiResult(
            decoded_bits=output_bits,
            input_bit_count=len(symbols),
            output_bit_count=len(output_bits),
            path_metric=float(final_metric),
            traceback_depth=self.traceback_depth,
            terminated=terminated,
            mode="soft_decision",
            success=(final_metric < INF),
            estimated_bit_errors=0,
            diagnostics=stats
        )
