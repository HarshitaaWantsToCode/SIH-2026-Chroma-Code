"""
Reed-Solomon Algebraic Block Decoder and Encoder over GF(2^8).

Mathematical Machinery:
- GF(256) arithmetic with primitive polynomial p(x) = x^8 + x^4 + x^3 + x^2 + 1 = 0x11D (285).
- Log and Exponential lookup tables.
- Systematic Polynomial LFSR Encoder.
- Syndrome computation.
- Berlekamp-Massey Key Equation Solver for Error Locator Polynomial Lambda(x).
- Chien root search over GF(256).
- Forney algorithm for Error Evaluator Omega(x) and Error Magnitudes.
- Post-correction syndrome verification for detecting uncorrectable errors.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class GF256:
    """
    Galois Field GF(2^8) Arithmetic Engine.
    Default primitive polynomial: p(x) = x^8 + x^4 + x^3 + x^2 + 1 = 0x11D (285).
    Generator element alpha = 2 (0x02).
    """

    def __init__(self, prim_poly: int = 0x11D, generator: int = 2) -> None:
        self.prim = prim_poly
        self.generator = generator
        self.exp = [0] * 512
        self.log = [0] * 256
        
        # Build exp and log tables
        x = 1
        for i in range(255):
            self.exp[i] = x
            self.log[x] = i
            x <<= 1
            if x & 0x100:
                x ^= self.prim
        
        # Repeat table for easy modulo-free index lookup
        for i in range(255, 512):
            self.exp[i] = self.exp[i - 255]

    def add(self, a: int, b: int) -> int:
        """Addition in GF(2^8) is bitwise XOR."""
        return a ^ b

    def sub(self, a: int, b: int) -> int:
        """Subtraction in GF(2^8) is identical to addition (XOR)."""
        return a ^ b

    def mul(self, a: int, b: int) -> int:
        """Multiplication using log/exp tables."""
        if a == 0 or b == 0:
            return 0
        return self.exp[self.log[a] + self.log[b]]

    def div(self, a: int, b: int) -> int:
        """Division in GF(2^8)."""
        if b == 0:
            raise ZeroDivisionError("Division by zero in GF(2^8).")
        if a == 0:
            return 0
        return self.exp[(self.log[a] - self.log[b] + 255) % 255]

    def inv(self, a: int) -> int:
        """Multiplicative inverse in GF(2^8)."""
        if a == 0:
            raise ZeroDivisionError("Multiplicative inverse of 0 does not exist in GF(2^8).")
        return self.exp[255 - self.log[a]]

    def pow(self, a: int, power: int) -> int:
        """Exponentiation in GF(2^8)."""
        if a == 0:
            return 0
        return self.exp[(self.log[a] * power) % 255]

    # ---------------- POLYNOMIAL OPERATIONS ----------------
    # High-to-low representation: poly[0]*x^(n-1) + ... + poly[n-1]
    def poly_scale(self, p: List[int], c: int) -> List[int]:
        """Multiplies polynomial p(x) by scalar c in GF(256)."""
        return [self.mul(coef, c) for coef in p]

    def poly_add(self, p1: List[int], p2: List[int]) -> List[int]:
        """Adds two polynomials in GF(256)[x]."""
        l1, l2 = len(p1), len(p2)
        out = [0] * max(l1, l2)
        for i in range(l1):
            out[i + len(out) - l1] ^= p1[i]
        for i in range(l2):
            out[i + len(out) - l2] ^= p2[i]
        return out

    def poly_mul(self, p1: List[int], p2: List[int]) -> List[int]:
        """Multiplies two polynomials in GF(256)[x]."""
        out = [0] * (len(p1) + len(p2) - 1)
        for j in range(len(p2)):
            for i in range(len(p1)):
                out[i + j] ^= self.mul(p1[i], p2[j])
        return out

    def poly_eval(self, p: List[int], x: int) -> int:
        """Evaluates polynomial p(x) at point x using Horner's rule (high to low degree)."""
        if not p:
            return 0
        y = p[0]
        for i in range(1, len(p)):
            y = self.add(self.mul(y, x), p[i])
        return y


# Singleton default field
GF = GF256()


@dataclass
class ReedSolomonResult:
    """Structured result returned by ReedSolomonDecoder."""
    decoded_data: bytes
    corrected_symbol_count: int
    detected_error_positions: List[int]
    success: bool
    uncorrectable: bool
    n: int
    k: int
    t: int
    syndromes: List[int]
    diagnostics: Dict[str, Union[int, float, str]]


class ReedSolomonEncoder:
    """
    Reference Reed-Solomon Encoder over GF(256).
    Generates systematic codewords: [Message (K symbols) | Parity (2t symbols)].
    """

    def __init__(self, n: int = 255, k: int = 223, fcr: int = 0, prim_poly: int = 0x11D) -> None:
        if k >= n or n > 255 or k <= 0:
            raise ValueError(f"Invalid RS parameters: N={n}, K={k}. Must have 0 < K < N <= 255.")
        self.n = n
        self.k = k
        self.nsym = n - k
        self.t = self.nsym // 2
        self.fcr = fcr
        self.gf = GF256(prim_poly=prim_poly)
        self.gen = self._build_generator_poly()

    def _build_generator_poly(self) -> List[int]:
        """Constructs generator polynomial: g(x) = prod_{i=0}^{2t-1} (x - alpha^(fcr + i))."""
        g = [1]
        for i in range(self.nsym):
            root = self.gf.pow(self.gf.generator, self.fcr + i)
            g = self.gf.poly_mul(g, [1, root])
        return g

    def encode(self, msg: Union[bytes, bytearray, List[int], np.ndarray]) -> bytes:
        """
        Encodes K bytes of message data into an N-byte systematic codeword.
        """
        if isinstance(msg, np.ndarray):
            msg_bytes = msg.tobytes()
        elif isinstance(msg, list):
            msg_bytes = bytes(msg)
        else:
            msg_bytes = bytes(msg)

        if len(msg_bytes) != self.k:
            raise ValueError(f"Message length ({len(msg_bytes)}) must equal K ({self.k}).")

        msg_list = list(msg_bytes)
        # Shift message by nsym positions (multiply by x^nsym)
        shifted_msg = msg_list + [0] * self.nsym
        
        # Synthetic polynomial division (LFSR simulation)
        rem = list(shifted_msg)
        for i in range(len(msg_list)):
            coef = rem[i]
            if coef != 0:
                for j in range(1, len(self.gen)):
                    rem[i + j] ^= self.gf.mul(self.gen[j], coef)

        parity = rem[len(msg_list):]
        codeword = msg_list + parity
        return bytes(codeword)


class ReedSolomonDecoder:
    """
    Algebraic Reed-Solomon Decoder over GF(2^8).
    Default configuration: RS(255, 223), t = 16.
    """

    def __init__(self, n: int = 255, k: int = 223, fcr: int = 0, prim_poly: int = 0x11D) -> None:
        if k >= n or n > 255 or k <= 0:
            raise ValueError(f"Invalid RS parameters: N={n}, K={k}. Must have 0 < K < N <= 255.")
        self.n = n
        self.k = k
        self.nsym = n - k
        self.t = self.nsym // 2
        self.fcr = fcr
        self.gf = GF256(prim_poly=prim_poly)

    def decode(self, block: Union[bytes, bytearray, List[int], np.ndarray]) -> Tuple[bytes, Dict[str, Union[int, float, str]]]:
        """
        Standard project signature: decodes block and returns (corrected_bytes, diagnostics_dict).
        """
        res = self.decode_block(block)
        return res.decoded_data, res.diagnostics

    def decode_block(self, block: Union[bytes, bytearray, List[int], np.ndarray]) -> ReedSolomonResult:
        """
        Full algebraic decode pipeline:
        1. Syndrome computation: S_i = R(alpha^(fcr + i)) for i in 0..nsym-1.
        2. Berlekamp-Massey algorithm for Lambda(x) and Error Evaluator Omega(x).
        3. Chien root search.
        4. Forney error magnitude computation.
        5. Codeword correction & Syndrome verification.
        """
        if isinstance(block, np.ndarray):
            raw_bytes = block.tobytes()
        elif isinstance(block, list):
            raw_bytes = bytes(block)
        else:
            raw_bytes = bytes(block)

        if len(raw_bytes) != self.n:
            raise ValueError(f"Received block length ({len(raw_bytes)}) must equal N ({self.n}).")

        r = list(raw_bytes)

        # 1. Compute Syndromes: S_i = R(alpha^(fcr + i))
        # Codeword polynomial R(x) = r[0]*x^(n-1) + r[1]*x^(n-2) + ... + r[n-1]
        syndromes = [0] * self.nsym
        has_error = False
        for i in range(self.nsym):
            root = self.gf.pow(self.gf.generator, self.fcr + i)
            s_val = self.gf.poly_eval(r, root)
            syndromes[i] = s_val
            if s_val != 0:
                has_error = True

        # Case 0: No errors detected
        if not has_error:
            msg = bytes(r[: self.k])
            stats = {
                "code_parameters": f"RS({self.n}, {self.k})",
                "parity_symbols": self.nsym,
                "max_correctable_errors_t": self.t,
                "syndromes_computed": self.nsym,
                "detected_symbol_errors": 0,
                "status": "VALID_NO_UNCORRECTABLE_ERRORS"
            }
            return ReedSolomonResult(
                decoded_data=msg,
                corrected_symbol_count=0,
                detected_error_positions=[],
                success=True,
                uncorrectable=False,
                n=self.n,
                k=self.k,
                t=self.t,
                syndromes=syndromes,
                diagnostics=stats
            )

        # 2. Berlekamp-Massey Algorithm
        # Let Lambda(x) = Lambda_0 + Lambda_1 x + ... + Lambda_v x^v where Lambda_0 = 1.
        # Represent Lambda as low-to-high coefficients: [1, Lambda_1, Lambda_2, ...]
        lambda_low = [1]
        b_low = [1]
        l_deg = 0

        for m in range(self.nsym):
            # Compute discrepancy delta
            delta = syndromes[m]
            for i in range(1, len(lambda_low)):
                delta ^= self.gf.mul(lambda_low[i], syndromes[m - i])

            # Shift B(x) by multiplying by x (insert 0 at lowest degree)
            b_low.insert(0, 0)

            if delta != 0:
                # T(x) = Lambda(x) + delta * B(x)
                # Align lengths
                max_len = max(len(lambda_low), len(b_low))
                t_low = [0] * max_len
                for i in range(len(lambda_low)):
                    t_low[i] ^= lambda_low[i]
                for i in range(len(b_low)):
                    t_low[i] ^= self.gf.mul(delta, b_low[i])

                if 2 * l_deg <= m:
                    # B(x) = Lambda(x) * delta^-1
                    delta_inv = self.gf.inv(delta)
                    b_low = [self.gf.mul(coef, delta_inv) for coef in lambda_low]
                    l_deg = m + 1 - l_deg

                lambda_low = t_low

        num_errors = len(lambda_low) - 1
        if num_errors > self.t:
            return self._make_failure_result(r, syndromes, "ERRORS_EXCEED_T_CAPABILITY")

        # 3. Chien Search: Find roots of Lambda(x)
        # Root equation: Lambda(alpha^-i) == 0 where i is the power of x in R(x): term r[n-1-i] * x^i.
        # Therefore index in codeword is idx = n - 1 - i.
        err_pos = []
        for i in range(self.n):
            # Evaluate Lambda at alpha^-i = alpha^(255 - i)
            x_val = self.gf.pow(self.gf.generator, (255 - i) % 255)
            # Evaluate Lambda_low: sum_{j} lambda_low[j] * (x_val)^j
            val = 0
            cur_pow = 1
            for coef in lambda_low:
                if coef != 0:
                    val ^= self.gf.mul(coef, cur_pow)
                cur_pow = self.gf.mul(cur_pow, x_val)

            if val == 0:
                idx = self.n - 1 - i
                err_pos.append(idx)

        if len(err_pos) != num_errors:
            return self._make_failure_result(r, syndromes, "CHIEN_SEARCH_ROOT_COUNT_MISMATCH")

        # 4. Error Evaluator Polynomial Omega(x) = [ Lambda(x) * S(x) ] mod x^nsym (low-to-high)
        s_low = list(syndromes)
        omega_low = [0] * (len(lambda_low) + len(s_low) - 1)
        for i in range(len(lambda_low)):
            for j in range(len(s_low)):
                omega_low[i + j] ^= self.gf.mul(lambda_low[i], s_low[j])
        omega_low = omega_low[: self.nsym]

        # Lambda'(x): formal derivative (odd power terms survive)
        # Lambda(x) = sum_j lambda_low[j] x^j -> Lambda'(x) = sum_{j odd} lambda_low[j] x^(j-1)
        lambda_prime_low = [0] * max(1, len(lambda_low) - 1)
        for j in range(1, len(lambda_low), 2):
            lambda_prime_low[j - 1] = lambda_low[j]

        # 5. Forney Algorithm:
        # For error location X_k = alpha^i (where i = n - 1 - idx):
        # Y_k = [ X_k^(1 - fcr) * Omega(X_k^-1) ] / Lambda'(X_k^-1)
        corrected = list(r)
        for idx in err_pos:
            i = self.n - 1 - idx
            x_k = self.gf.pow(self.gf.generator, i)                 # X_k = alpha^i
            x_inv = self.gf.pow(self.gf.generator, (255 - i) % 255)  # X_k^-1 = alpha^-i

            # Evaluate Omega(X_k^-1)
            omega_val = 0
            cur_pow = 1
            for coef in omega_low:
                if coef != 0:
                    omega_val ^= self.gf.mul(coef, cur_pow)
                cur_pow = self.gf.mul(cur_pow, x_inv)

            # Evaluate Lambda'(X_k^-1)
            lp_val = 0
            cur_pow = 1
            for coef in lambda_prime_low:
                if coef != 0:
                    lp_val ^= self.gf.mul(coef, cur_pow)
                cur_pow = self.gf.mul(cur_pow, x_inv)

            if lp_val == 0:
                return self._make_failure_result(r, syndromes, "FORNEY_DIVISION_BY_ZERO")

            num = self.gf.mul(self.gf.pow(x_k, 1 - self.fcr), omega_val)
            y_k = self.gf.div(num, lp_val)
            corrected[idx] ^= y_k

        # 6. Post-Correction Syndrome Verification
        post_syndromes = [self.gf.poly_eval(corrected, self.gf.pow(self.gf.generator, self.fcr + s)) for s in range(self.nsym)]
        if any(s != 0 for s in post_syndromes):
            return self._make_failure_result(r, syndromes, "POST_CORRECTION_SYNDROME_VERIFICATION_FAILED")

        msg = bytes(corrected[: self.k])
        stats = {
            "code_parameters": f"RS({self.n}, {self.k})",
            "parity_symbols": self.nsym,
            "max_correctable_errors_t": self.t,
            "syndromes_computed": self.nsym,
            "detected_symbol_errors": len(err_pos),
            "status": "VALID_CORRECTED"
        }

        return ReedSolomonResult(
            decoded_data=msg,
            corrected_symbol_count=len(err_pos),
            detected_error_positions=sorted(err_pos),
            success=True,
            uncorrectable=False,
            n=self.n,
            k=self.k,
            t=self.t,
            syndromes=syndromes,
            diagnostics=stats
        )

    def _make_failure_result(self, r: List[int], syndromes: List[int], reason: str) -> ReedSolomonResult:
        """Helper to safely construct an uncorrectable failure result without corrupting data."""
        stats = {
            "code_parameters": f"RS({self.n}, {self.k})",
            "parity_symbols": self.nsym,
            "max_correctable_errors_t": self.t,
            "syndromes_computed": self.nsym,
            "detected_symbol_errors": -1,
            "status": f"UNCORRECTABLE_ERROR_{reason}"
        }
        return ReedSolomonResult(
            decoded_data=bytes(r[: self.k]),
            corrected_symbol_count=0,
            detected_error_positions=[],
            success=False,
            uncorrectable=True,
            n=self.n,
            k=self.k,
            t=self.t,
            syndromes=syndromes,
            diagnostics=stats
        )
