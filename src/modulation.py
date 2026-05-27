

import numpy as np
from typing import Tuple

def _gray_to_natural(g: int) -> int:

    n = g
    mask = g >> 1
    while mask:
        n ^= mask
        mask >>= 1
    return n

def _pam_levels(m_pam: int) -> np.ndarray:

    return np.arange(-(m_pam - 1), m_pam, 2, dtype=float)

def _pam_gray_bit_table(m_pam: int) -> np.ndarray:

    bits_per_level = int(np.log2(m_pam))
    table = np.zeros((m_pam, bits_per_level), dtype=int)
    for i in range(m_pam):
        gray = i ^ (i >> 1)
        for b in range(bits_per_level):
            table[i, bits_per_level - 1 - b] = (gray >> b) & 1
    return table

def qam_modulate(bits: np.ndarray, M: int = 16) -> np.ndarray:

    assert M in (4, 16, 64), f"Unsupported M={M}, use 4, 16, or 64"
    bps = int(np.log2(M))
    assert len(bits) % bps == 0, f"len(bits)={len(bits)} must be divisible by bps={bps}"

    m_pam = int(np.sqrt(M))
    bpp = bps // 2

    levels = _pam_levels(m_pam)
    table  = _pam_gray_bit_table(m_pam)

    inv_gray = {}
    for idx in range(m_pam):
        gray_int = int("".join(str(b) for b in table[idx]), 2)
        inv_gray[gray_int] = idx

    bits_matrix = bits.reshape(-1, bps)
    n_sym = bits_matrix.shape[0]

    I_bits = bits_matrix[:, :bpp]
    Q_bits = bits_matrix[:, bpp:]

    I_amps = np.array([
        levels[inv_gray[int("".join(str(b) for b in row), 2)]]
        for row in I_bits
    ])
    Q_amps = np.array([
        levels[inv_gray[int("".join(str(b) for b in row), 2)]]
        for row in Q_bits
    ])

    symbols = I_amps + 1j * Q_amps

    E_s_unnorm = (2.0 / 3.0) * (M - 1)
    symbols /= np.sqrt(E_s_unnorm)

    return symbols

def qam_demodulate_hard(symbols: np.ndarray, M: int = 16) -> np.ndarray:

    bps   = int(np.log2(M))
    m_pam = int(np.sqrt(M))
    bpp   = bps // 2

    levels = _pam_levels(m_pam)
    table  = _pam_gray_bit_table(m_pam)

    E_s_unnorm = (2.0 / 3.0) * (M - 1)
    symbols_raw = symbols * np.sqrt(E_s_unnorm)

    I_raw = np.real(symbols_raw)
    Q_raw = np.imag(symbols_raw)

    def nearest_idx(vals):

        dists = np.abs(vals[:, None] - levels[None, :])
        return np.argmin(dists, axis=1)

    I_idx = nearest_idx(I_raw)
    Q_idx = nearest_idx(Q_raw)

    I_bits = table[I_idx]
    Q_bits = table[Q_idx]

    bits_matrix = np.concatenate([I_bits, Q_bits], axis=1)
    return bits_matrix.ravel()

def compute_llr_awgn(
    y_scalar: np.ndarray,
    sigma2: float,
    M: int = 16
) -> np.ndarray:

    m_pam   = M
    bpp     = int(np.log2(m_pam))
    levels  = _pam_levels(m_pam)
    table   = _pam_gray_bit_table(m_pam)

    n = len(y_scalar)
    llrs = np.zeros((n, bpp))

    dist2 = (y_scalar[:, None] - levels[None, :]) ** 2
    log_like = -dist2 / sigma2

    for k in range(bpp):

        idx0 = np.where(table[:, k] == 0)[0]
        idx1 = np.where(table[:, k] == 1)[0]

        ll0 = log_like[:, idx0]
        ll1 = log_like[:, idx1]

        def log_sum_exp(x):

            x_max = np.max(x, axis=1, keepdims=True)
            return x_max[:, 0] + np.log(np.sum(np.exp(x - x_max), axis=1))

        llrs[:, k] = log_sum_exp(ll0) - log_sum_exp(ll1)

    return llrs

def compute_llrs_qam(
    y_eq: np.ndarray,
    sigma2_eq: float,
    M: int = 16
) -> np.ndarray:

    bps   = int(np.log2(M))
    m_pam = int(np.sqrt(M))
    bpp   = bps // 2

    E_s_unnorm = (2.0 / 3.0) * (M - 1)
    scale = np.sqrt(E_s_unnorm)

    y_I = np.real(y_eq) * scale
    y_Q = np.imag(y_eq) * scale
    sigma2_pam = sigma2_eq * E_s_unnorm

    llr_I = compute_llr_awgn(y_I, sigma2_pam, m_pam)

    llr_Q = compute_llr_awgn(y_Q, sigma2_pam, m_pam)

    n_sym = len(y_eq)
    llrs_matrix = np.concatenate([llr_I, llr_Q], axis=1)
    return llrs_matrix.ravel()

def bpsk_llr(
    y_eq: np.ndarray,
    sigma2_eq: float
) -> np.ndarray:

    return -4.0 * np.real(y_eq) / sigma2_eq

def mmse_equalize_per_stream(
    y: np.ndarray,
    H: np.ndarray,
    sigma2: float
) -> Tuple[np.ndarray, np.ndarray]:

    Nr, Nt = H.shape
    G    = H.conj().T @ H + sigma2 * np.eye(Nt)
    W    = np.linalg.solve(G, H.conj().T)
    WH   = W @ H

    mu   = np.real(np.diag(WH))
    y_eq = W @ y

    return y_eq, mu

def compute_llrs_mmse(
    y: np.ndarray,
    H: np.ndarray,
    sigma2: float,
    M: int = 16
) -> np.ndarray:

    y_eq, mu = mmse_equalize_per_stream(y, H, sigma2)

    Nt = H.shape[1]
    bps = int(np.log2(M))
    llrs = np.zeros(Nt * bps)

    for k in range(Nt):
        mu_k = mu[k]
        if mu_k <= 0:
            mu_k = 1e-9

        y_eq_k = y_eq[k] / mu_k

        sigma2_eq_k = (1.0 - mu_k) / mu_k
        if sigma2_eq_k <= 0:
            sigma2_eq_k = 1e-9

        if M == 2:
            llrs[k] = bpsk_llr(np.array([y_eq_k]), sigma2_eq_k)[0]
        else:

            stream_llrs = compute_llrs_qam(
                np.array([y_eq_k]),
                sigma2_eq_k,
                M=M
            )
            llrs[k * bps: (k + 1) * bps] = stream_llrs

    return llrs

def simulate_ber_qam(
    Nr: int,
    Nt: int,
    M: int,
    snr_db_range: np.ndarray,
    n_trials: int = 3000,
    seed: int = 42
) -> np.ndarray:

    from src.channel import generate_channel, awgn_channel

    rng   = np.random.default_rng(seed)
    bps   = int(np.log2(M))
    ber_curve = np.zeros(len(snr_db_range))

    for i, snr_db in enumerate(snr_db_range):
        snr_linear = 10.0 ** (snr_db / 10.0)
        sigma2 = Nt / (Nr * snr_linear)
        errors = 0
        total_bits = 0

        for _ in range(n_trials):

            bits_tx = rng.integers(0, 2, size=Nt * bps)
            x = qam_modulate(bits_tx, M=M)

            H = generate_channel(Nr, Nt, seed=None)
            y, sigma2_actual = awgn_channel(H, x, snr_db, rng)

            llrs = compute_llrs_mmse(y, H, sigma2_actual, M=M)
            bits_rx = (llrs <= 0).astype(int)

            errors     += np.sum(bits_tx != bits_rx)
            total_bits += len(bits_tx)

        ber_curve[i] = errors / total_bits
        print(f"  SNR = {snr_db:5.1f} dB  |  BER = {ber_curve[i]:.5f}")

    return ber_curve


