# ==============================================================================
# IMPORTS
# ==============================================================================
import numpy as np
from typing import Tuple, Optional

# ==============================================================================
# CONSTELLATION PROJECTION UTILITIES
# ==============================================================================

def _pam_levels_admm(m_pam: int) -> np.ndarray:
    """Generates 1D PAM constellation levels centered at zero."""

    return np.arange(-(m_pam - 1), m_pam, 2, dtype=float)

def project_bpsk(v: np.ndarray) -> np.ndarray:
    """Projects complex or real values onto the BPSK constellation."""

    return np.where(v >= 0, 1.0, -1.0)

def project_qam(v: np.ndarray, M: int = 16) -> np.ndarray:
    """Projects complex values onto a square QAM constellation."""

    m_pam = int(np.sqrt(M))
    levels = _pam_levels_admm(m_pam)
    E_s    = (2.0 / 3.0) * (M - 1)
    scale  = np.sqrt(E_s)

    v_raw = v * scale

    def nearest_pam(x_real: np.ndarray) -> np.ndarray:

        dists = np.abs(x_real[:, None] - levels[None, :])
        return levels[np.argmin(dists, axis=1)]

    I_proj = nearest_pam(np.real(v_raw))
    Q_proj = nearest_pam(np.imag(v_raw))

    return (I_proj + 1j * Q_proj) / scale

def project_constellation(v: np.ndarray, M: int = 2) -> np.ndarray:
    """Projects values onto the appropriate constellation (BPSK or QAM)."""

    if M == 2:
        return project_bpsk(np.real(v))
    else:
        return project_qam(v, M=M)

def soft_project_bpsk(
    v: np.ndarray,
    llr_prior: np.ndarray,
    rho: float
) -> np.ndarray:
    """Performs soft projection for BPSK using prior LLR information."""

    v_eff = rho * v + llr_prior / 2.0
    return np.where(v_eff >= 0, 1.0, -1.0)

# ==============================================================================
# ADMM CORE DETECTION
# ==============================================================================

def get_adaptive_rho(sigma2: float, scale: float = 1.0) -> float:
    """Computes an adaptive penalty parameter rho based on noise variance."""

    return float(scale / max(sigma2, 1e-9))

def admm_detect(
    y: np.ndarray,
    H: np.ndarray,
    sigma2: float,
    M: int = 2,
    rho: Optional[float] = None,
    max_iter: int = 20,
    llr_prior: Optional[np.ndarray] = None,
    return_soft: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Performs Alternating Direction Method of Multipliers (ADMM) detection.
    Iteratively estimates the transmitted symbols in a MIMO system.
    """

    Nr, Nt = H.shape

    if rho is None:
        rho = get_adaptive_rho(sigma2, scale=1.0)

    HH = H.conj().T @ H
    A  = HH / sigma2 + rho * np.eye(Nt)

    Hhy = H.conj().T @ y / sigma2

    z = np.zeros(Nt, dtype=complex)
    u = np.zeros(Nt, dtype=complex)

    if M == 2:
        z = np.zeros(Nt, dtype=float)
        u = np.zeros(Nt, dtype=float)

    for k in range(max_iter):

        rhs = Hhy + rho * (z - u)
        if M == 2:

            x = np.linalg.solve(A, rhs)
            x = np.real(x)
        else:
            x = np.linalg.solve(A, rhs)

        v = x + u

        if M == 2:
            if llr_prior is not None:
                z = soft_project_bpsk(np.real(v), llr_prior, rho)
            else:
                z = project_bpsk(np.real(v))
        else:

            z = project_qam(v, M=M)

        u = u + x - z

    z_soft = x if return_soft else None
    return z, z_soft

# ==============================================================================
# SOFT LLR COMPUTATION
# ==============================================================================

def admm_llrs(
    z_soft: np.ndarray,
    sigma2_eff: float,
    M: int = 2
) -> np.ndarray:
    """Computes approximate Log-Likelihood Ratios (LLRs) from ADMM soft outputs."""

    from src.modulation import compute_llrs_qam

    Nt = z_soft.shape[0]

    if M == 2:

        return 2.0 * np.real(z_soft) / sigma2_eff
    else:
        return compute_llrs_qam(z_soft, sigma2_eff, M=M)

def admm_effective_sigma2(
    H: np.ndarray,
    sigma2: float,
    rho: float
) -> float:
    """Estimates the effective noise variance after ADMM iterations."""

    Nr, Nt = H.shape
    HH     = H.conj().T @ H
    A_inv  = np.linalg.inv(HH / sigma2 + rho * np.eye(Nt))

    mid    = A_inv @ (HH / sigma2)
    sigma2_eff = float(np.real(sigma2 * np.trace(mid @ mid.T.conj()) / Nt))
    return max(sigma2_eff, 1e-9)

# ==============================================================================
# SIMULATION AND EVALUATION
# ==============================================================================

def simulate_admm_ber(
    Nr: int,
    Nt: int,
    M: int,
    snr_db_range: np.ndarray,
    rho: Optional[float] = None,
    max_iter: int = 20,
    n_trials: int = 500,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates Bit Error Rate (BER) performance for ADMM and MMSE detectors.
    Runs Monte Carlo trials over a range of SNR values.
    """

    from src.channel    import generate_channel, awgn_channel
    from src.modulation import qam_modulate, qam_demodulate_hard, compute_llrs_mmse

    rng      = np.random.default_rng(seed)
    bps      = int(np.log2(M))
    ber_admm = np.zeros(len(snr_db_range))
    ber_mmse = np.zeros(len(snr_db_range))

    for i, snr_db in enumerate(snr_db_range):
        err_admm = 0
        err_mmse = 0
        total    = 0

        for _ in range(n_trials):
            bits_tx = rng.integers(0, 2, size=Nt * bps)

            if M == 2:

                x = (2.0 * bits_tx - 1.0).astype(float)
            else:
                x = qam_modulate(bits_tx, M=M)

            H      = generate_channel(Nr, Nt, seed=None)
            y, s2  = awgn_channel(H, x, snr_db, rng)

            z_hard, _ = admm_detect(y, H, s2, M=M, rho=rho, max_iter=max_iter)

            if M == 2:
                bits_admm = (np.real(z_hard) >= 0).astype(int)
            else:
                bits_admm = qam_demodulate_hard(z_hard, M=M)

            err_admm += int(np.sum(bits_tx != bits_admm))

            from src.channel import mmse_detect
            from src.modulation import bpsk_llr, mmse_equalize_per_stream

            if M == 2:
                from src.channel import mmse_detect
                x_mmse = mmse_detect(y, H, s2)
                bits_mmse = (np.real(x_mmse) >= 0).astype(int)
            else:
                llrs_mmse  = compute_llrs_mmse(y, H, s2, M=M)
                bits_mmse  = (llrs_mmse <= 0).astype(int)

            err_mmse += int(np.sum(bits_tx != bits_mmse))
            total    += Nt * bps

        ber_admm[i] = err_admm / total
        ber_mmse[i] = err_mmse / total
        print(
            f"  SNR={snr_db:4.1f} dB | "
            f"ADMM BER={ber_admm[i]:.5f} | "
            f"MMSE BER={ber_mmse[i]:.5f}"
        )

    return ber_admm, ber_mmse

