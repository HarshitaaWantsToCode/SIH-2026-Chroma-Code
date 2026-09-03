"""
De-interleaving Module for Dispersing Burst Errors.

Implements:
1. Matrix Block De-interleaver (inverting row-write / column-read).
2. Convolutional / Shift-Register De-interleaver with delay branches.
3. Pseudo-Random / Permutation-based De-interleaver.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np


@dataclass
class DeinterleaverResult:
    """Metadata and output from de-interleaving operations."""
    data: np.ndarray
    original_length: int
    output_length: int
    mode: str
    padding_added: int = 0
    truncated_elements: int = 0
    info: str = ""


class BlockInterleaver:
    """
    Reference Block Interleaver for testing and transmission simulation.
    Transmitter Operation:
        - Writes data into matrix row-by-row (M rows x N cols).
        - Reads data out column-by-column.
    """

    @staticmethod
    def interleave(bits: np.ndarray, rows: int, cols: int) -> np.ndarray:
        """
        Interleaves bits: writes row-major (shape (rows, cols)), reads col-major (Fortran order).
        """
        if rows <= 0 or cols <= 0:
            raise ValueError(f"Rows ({rows}) and cols ({cols}) must be positive integers.")
        
        bits = np.asarray(bits, dtype=np.uint8)
        block_size = rows * cols
        if len(bits) % block_size != 0:
            raise ValueError(f"Input length ({len(bits)}) must be a multiple of block size ({block_size}).")
        
        num_blocks = len(bits) // block_size
        interleaved = np.empty_like(bits)
        
        for b in range(num_blocks):
            chunk = bits[b * block_size : (b + 1) * block_size]
            # Row-write (C order) -> Column-read (F order flatten or transpose flatten)
            matrix = chunk.reshape((rows, cols))
            interleaved[b * block_size : (b + 1) * block_size] = matrix.T.flatten()
            
        return interleaved


class ConvolutionalInterleaver:
    """
    Reference Parameterized Convolutional Interleaver with B branches.
    Branch i delay = i * M symbols/bits.
    """

    def __init__(self, branches: int, delay_increment: int) -> None:
        if branches <= 0 or delay_increment < 0:
            raise ValueError(f"Branches ({branches}) > 0 and delay_increment ({delay_increment}) >= 0 required.")
        self.b = branches
        self.m = delay_increment
        self.delays = [i * self.m for i in range(self.b)]
        self.shift_registers: List[List[int]] = [[0] * d for d in self.delays]
        self.branch_idx = 0

    def reset(self) -> None:
        """Flushes shift registers to zero state."""
        self.shift_registers = [[0] * d for d in self.delays]
        self.branch_idx = 0

    def process(self, symbols: np.ndarray) -> np.ndarray:
        """Streams symbols through the delay branches commutating cyclically."""
        symbols = np.asarray(symbols, dtype=np.uint8)
        output = np.empty_like(symbols)
        for idx, sym in enumerate(symbols):
            b = self.branch_idx
            sr = self.shift_registers[b]
            if len(sr) == 0:
                out_val = sym
            else:
                out_val = sr.pop(0)
                sr.append(int(sym))
            output[idx] = out_val
            self.branch_idx = (self.branch_idx + 1) % self.b
        return output


class ConvolutionalDeinterleaver:
    """
    Convolutional De-interleaver with B branches.
    Inverts Convolutional Interleaver by applying complementary delays:
    Branch i delay = (B - 1 - i) * M.
    Total round-trip delay across interleaver + deinterleaver is B * (B - 1) * M.
    """

    def __init__(self, branches: int, delay_increment: int) -> None:
        if branches <= 0 or delay_increment < 0:
            raise ValueError(f"Branches ({branches}) > 0 and delay_increment ({delay_increment}) >= 0 required.")
        self.b = branches
        self.m = delay_increment
        # Complementary delays
        self.delays = [(self.b - 1 - i) * self.m for i in range(self.b)]
        self.shift_registers: List[List[int]] = [[0] * d for d in self.delays]
        self.branch_idx = 0

    @property
    def total_latency(self) -> int:
        """Total latency in symbols introduced by interleaver + deinterleaver pair."""
        return self.b * (self.b - 1) * self.m

    def reset(self) -> None:
        """Resets internal delay states to zero."""
        self.shift_registers = [[0] * d for d in self.delays]
        self.branch_idx = 0

    def process(self, symbols: np.ndarray) -> np.ndarray:
        """Processes received interleaved symbols through complementary delay branches."""
        symbols = np.asarray(symbols, dtype=np.uint8)
        output = np.empty_like(symbols)
        for idx, sym in enumerate(symbols):
            b = self.branch_idx
            sr = self.shift_registers[b]
            if len(sr) == 0:
                out_val = sym
            else:
                out_val = sr.pop(0)
                sr.append(int(sym))
            output[idx] = out_val
            self.branch_idx = (self.branch_idx + 1) % self.b
        return output


class Deinterleaver:
    """
    Production De-interleaving Engine providing Block, Convolutional, and Pseudo-Random deinterleaving.
    """

    @staticmethod
    def block_deinterleave(
        bits: Union[np.ndarray, List[int]],
        rows: int,
        cols: int,
        incomplete_mode: str = "pad",
        pad_value: int = 0
    ) -> np.ndarray:
        """
        Reconstructs sequential bitstream from column-read interleaved stream.
        
        Mathematical Inversion:
            Transmitter: Row-write (rows x cols) -> Column-read.
            Receiver Deinterleaver: Column-write (rows x cols) -> Row-read.
            Given an input block of length (rows * cols), we reshape into (cols, rows)
            and transpose to (rows, cols), then flatten row-major.

        Args:
            bits: 1D array-like of binary bits or symbols.
            rows: Matrix row count (M > 0).
            cols: Matrix column count (N > 0).
            incomplete_mode: "pad" (pads trailing block) or "truncate" (drops trailing bits).
            pad_value: Fill value for padding (default 0).

        Returns:
            np.ndarray: De-interleaved bit array (np.uint8).
        """
        if rows <= 0 or cols <= 0:
            raise ValueError(f"Rows ({rows}) and columns ({cols}) must be positive integers.")
        
        bits_arr = np.asarray(bits)
        if bits_arr.ndim != 1:
            raise ValueError(f"Input must be 1-dimensional array, got shape {bits_arr.shape}")
        
        # Check binary bit values if integer
        if np.issubdtype(bits_arr.dtype, np.integer) or np.issubdtype(bits_arr.dtype, np.bool_):
            if np.any((bits_arr != 0) & (bits_arr != 1)):
                raise ValueError("Input bits array contains non-binary values outside {0, 1}.")
        
        orig_len = len(bits_arr)
        block_size = rows * cols
        rem = orig_len % block_size

        padded_count = 0
        truncated_count = 0

        if rem != 0:
            if incomplete_mode == "pad":
                padded_count = block_size - rem
                pad_arr = np.full(padded_count, pad_value, dtype=bits_arr.dtype)
                work_arr = np.concatenate([bits_arr, pad_arr])
            elif incomplete_mode == "truncate":
                truncated_count = rem
                work_arr = bits_arr[: orig_len - rem]
            else:
                raise ValueError(f"Unsupported incomplete_mode: {incomplete_mode}. Must be 'pad' or 'truncate'.")
        else:
            work_arr = bits_arr

        if len(work_arr) == 0:
            return np.empty(0, dtype=np.uint8)

        num_blocks = len(work_arr) // block_size
        deinterleaved = np.empty(len(work_arr), dtype=np.uint8)

        for b in range(num_blocks):
            chunk = work_arr[b * block_size : (b + 1) * block_size]
            matrix = chunk.reshape((cols, rows)).T
            deinterleaved[b * block_size : (b + 1) * block_size] = matrix.flatten()

        return deinterleaved

    @staticmethod
    def block_deinterleave_with_meta(
        bits: Union[np.ndarray, List[int]],
        rows: int,
        cols: int,
        incomplete_mode: str = "pad",
        pad_value: int = 0
    ) -> DeinterleaverResult:
        """Block de-interleave returning full metadata."""
        orig_len = len(bits)
        out = Deinterleaver.block_deinterleave(bits, rows, cols, incomplete_mode, pad_value)
        block_size = rows * cols
        rem = orig_len % block_size
        pad_added = (block_size - rem) if (rem != 0 and incomplete_mode == "pad") else 0
        trunc = rem if (rem != 0 and incomplete_mode == "truncate") else 0
        
        return DeinterleaverResult(
            data=out,
            original_length=orig_len,
            output_length=len(out),
            mode=f"block_{incomplete_mode}",
            padding_added=pad_added,
            truncated_elements=trunc,
            info=f"Matrix {rows}x{cols} block deinterleaver"
        )

    @staticmethod
    def convolutional_deinterleave(
        bits: np.ndarray,
        branches: int,
        delay_increment: int
    ) -> np.ndarray:
        """
        Performs convolutional de-interleaving on a stream.
        """
        cdeint = ConvolutionalDeinterleaver(branches, delay_increment)
        return cdeint.process(bits)

    @staticmethod
    def inverse_permutation(permutation: Union[np.ndarray, List[int]]) -> np.ndarray:
        """
        Computes the inverse permutation pi_inv such that pi_inv[pi[i]] == i.

        Args:
            permutation: 1D array of length N containing a bijection of [0 .. N-1].
        """
        p = np.asarray(permutation, dtype=np.int64)
        if p.ndim != 1:
            raise ValueError("Permutation must be 1D.")
        n = len(p)
        if n == 0:
            return np.empty(0, dtype=np.int64)
        
        # Validation: check range and unique elements
        if np.min(p) < 0 or np.max(p) >= n or len(np.unique(p)) != n:
            raise ValueError(f"Invalid permutation: must contain all indices in range [0, {n-1}] exactly once.")
        
        inv = np.empty(n, dtype=np.int64)
        inv[p] = np.arange(n, dtype=np.int64)
        return inv

    @staticmethod
    def pseudo_random_deinterleave(
        data: np.ndarray,
        permutation: Union[np.ndarray, List[int]]
    ) -> np.ndarray:
        """
        Inverts a pseudo-random permutation interleaving:
            If interleaved[i] = data[permutation[i]],
            then original[permutation[i]] = interleaved[i].

        Args:
            data: Interleaved 1D array of length N.
            permutation: 1D permutation indices of length N.

        Returns:
            np.ndarray: Reordered original data array.
        """
        data_arr = np.asarray(data)
        if data_arr.ndim != 1:
            raise ValueError("Data must be 1D.")
        if len(data_arr) != len(permutation):
            raise ValueError(f"Data length ({len(data_arr)}) must match permutation length ({len(permutation)}).")
        
        inv_perm = Deinterleaver.inverse_permutation(permutation)
        return data_arr[inv_perm]
