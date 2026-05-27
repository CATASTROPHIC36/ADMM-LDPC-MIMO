"""
MIMO Channel and Basic Detection Utilities

This module contains functions for simulating a flat-fading MIMO channel,
applying basic BPSK modulation/demodulation, adding AWGN, and performing
linear MMSE detection.
"""

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
from typing import Tuple

# =============================================================================
# CHANNEL AND MODULATION
# =============================================================================

def generate_channel(Nr: int, Nt: int, seed: int = None) -> np.ndarray:
    """
    Generates a Rayleigh flat-fading MIMO channel matrix.
    """
    rng = np.random.default_rng(seed)

    scale = 1.0 / np.sqrt(2 * Nt)
    H_real = rng.standard_normal((Nr, Nt)) * scale
    H_imag = rng.standard_normal((Nr, Nt)) * scale
    return H_real + 1j * H_imag

def bpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """
    Maps binary bits {0, 1} to BPSK symbols {-1, +1}.
    """
    return (2.0 * bits.astype(float)) - 1.0

def bpsk_demodulate_hard(symbols: np.ndarray) -> np.ndarray:
    """
    Slices BPSK symbols to hard bits {0, 1}.
    """
    return (np.real(symbols) >= 0).astype(int)

def awgn_channel(
    H: np.ndarray,
    x: np.ndarray,
    snr_db: float,
    rng: np.random.Generator = None
) -> Tuple[np.ndarray, float]:
    """
    Passes the transmitted symbols through the MIMO channel and adds AWGN.
    Returns the received signal vector 'y' and the noise variance 'sigma2'.
    """
    if rng is None:
        rng = np.random.default_rng()

    Nr, Nt = H.shape
    snr_linear = 10.0 ** (snr_db / 10.0)

    sigma2 = Nt / (Nr * snr_linear)
    noise_std = np.sqrt(sigma2 / 2)

    noise = noise_std * (rng.standard_normal(Nr) + 1j * rng.standard_normal(Nr))

    y = H @ x + noise
    return y, sigma2

# =============================================================================
# DETECTION AND ERROR METRICS
# =============================================================================

def mmse_detect(
    y: np.ndarray,
    H: np.ndarray,
    sigma2: float
) -> np.ndarray:
    """
    Performs linear Minimum Mean Square Error (MMSE) detection.
    """
    Nr, Nt = H.shape

    gram = H.conj().T @ H + sigma2 * np.eye(Nt)

    matched = H.conj().T @ y

    x_hat = np.linalg.solve(gram, matched)
    return x_hat

def compute_ber(bits_tx: np.ndarray, bits_rx: np.ndarray) -> float:
    """
    Computes the Bit Error Rate (BER) between transmitted and received bits.
    """
    return np.mean(bits_tx != bits_rx)

# =============================================================================
# SIMULATION
# =============================================================================

def simulate_ber_curve(
    Nr: int,
    Nt: int,
    snr_db_range: np.ndarray,
    n_trials: int = 5000,
    seed: int = 42
) -> np.ndarray:
    """
    Simulates a BER curve over a range of SNRs for the MMSE detector.
    """
    rng = np.random.default_rng(seed)
    ber_curve = np.zeros(len(snr_db_range))

    for i, snr_db in enumerate(snr_db_range):
        snr_linear = 10.0 ** (snr_db / 10.0)
        sigma2 = Nt / (Nr * snr_linear)
        noise_std = np.sqrt(sigma2 / 2.0)

        scale = 1.0 / np.sqrt(2 * Nt)
        H_r = rng.standard_normal((n_trials, Nr, Nt)) * scale
        H_i = rng.standard_normal((n_trials, Nr, Nt)) * scale
        H_batch = H_r + 1j * H_i

        bits_tx = rng.integers(0, 2, size=(n_trials, Nt))
        x_batch = (2.0 * bits_tx - 1.0).astype(complex)

        Hx = np.einsum('bij,bj->bi', H_batch, x_batch)
        noise = noise_std * (
            rng.standard_normal((n_trials, Nr)) +
            1j * rng.standard_normal((n_trials, Nr))
        )
        y_batch = Hx + noise

        H_H = H_batch.conj().transpose(0, 2, 1)
        gram = H_H @ H_batch + sigma2 * np.eye(Nt)
        matched = np.einsum('bij,bj->bi', H_H, y_batch)

        x_hat = np.linalg.solve(gram, matched[..., np.newaxis]).squeeze(-1)

        bits_rx = (np.real(x_hat) >= 0).astype(int)

        errors = np.sum(bits_tx != bits_rx)
        ber_curve[i] = errors / (n_trials * Nt)
        print(f"  SNR = {snr_db:5.1f} dB  |  BER = {ber_curve[i]:.5f}"
              f"  ({errors} errors / {n_trials * Nt} bits)")

    return ber_curve

