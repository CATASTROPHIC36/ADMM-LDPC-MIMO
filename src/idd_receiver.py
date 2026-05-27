"""
Iterative Detection and Decoding (IDD) Receiver

This module implements an iterative receiver that jointly performs MIMO
detection using ADMM and LDPC decoding using Sum-Product Algorithm (SPA).
Information is exchanged iteratively between the detector and decoder
in the form of Log-Likelihood Ratios (LLRs).
"""

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
from typing import Tuple, Optional, List

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clip_llrs(llrs: np.ndarray, clip: float = 20.0) -> np.ndarray:
    """
    Clips LLR values to prevent numerical overflow in the SPA decoder.
    """
    return np.clip(llrs, -clip, clip)

def extract_message_bits(b_hat: np.ndarray, H_ldpc: np.ndarray) -> np.ndarray:
    """
    Extracts the message bits from the decoded codeword.
    """
    m, n = H_ldpc.shape
    k    = n - m

    return b_hat[:k]

# =============================================================================
# IDD RECEIVER CORE
# =============================================================================

def idd_receiver_bpsk(
    y: np.ndarray,
    H: np.ndarray,
    sigma2: float,
    H_ldpc: np.ndarray,
    rho: float = 1.0,
    admm_iter: int = 20,
    bp_iter: int = 10,
    idd_iter: int = 4,
    sys_cols: Optional[List[int]] = None,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Runs the IDD algorithm with an ADMM detector and an SPA LDPC decoder.
    Passes soft information (extrinsic LLRs) back and forth.
    """
    from src.admm_detector import admm_detect, admm_llrs, admm_effective_sigma2
    from src.ldpc           import spa_decode_soft, build_tanner_graph

    Nt     = H.shape[1]
    m, n   = H_ldpc.shape
    k      = n - m
    if sys_cols is None:
        sys_cols = list(range(k))

    vn_to_cn, cn_to_vn = build_tanner_graph(H_ldpc)

    sigma2_eff = admm_effective_sigma2(H, sigma2, rho)

    llr_prior    = np.zeros(Nt)
    llr_dec_soft = np.zeros(Nt)
    b_hat        = np.zeros(Nt, dtype=int)
    ber_hist     = []

    for t in range(idd_iter):

        prior_input = llr_prior if t > 0 else None
        z_hard, z_soft = admm_detect(
            y, H, sigma2, M=2, rho=rho,
            max_iter=admm_iter,
            llr_prior=prior_input,
            return_soft=True
        )

        llr_det = admm_llrs(z_soft, sigma2_eff, M=2)
        llr_det = clip_llrs(llr_det)

        llr_det_extrinsic = llr_det - llr_prior
        llr_det_extrinsic = clip_llrs(llr_det_extrinsic)

        b_hat, llr_dec_soft, converged, n_iter = spa_decode_soft(
            llr_det_extrinsic, H_ldpc,
            max_iter=bp_iter,
            vn_to_cn=vn_to_cn,
            cn_to_vn=cn_to_vn
        )

        llr_prior = clip_llrs(llr_dec_soft - llr_det_extrinsic)

        ber_iter = float(np.mean(b_hat != np.zeros(Nt, dtype=int)))
        ber_hist.append(ber_iter)

        if verbose:
            print(f"    IDD iter {t+1}/{idd_iter}: "
                  f"BP {'converged' if converged else 'not converged'} "
                  f"in {n_iter} steps")

    return b_hat, llr_dec_soft, ber_hist

# =============================================================================
# SIMULATION ROUTINES
# =============================================================================

def simulate_idd_ber(
    Nr: int,
    Nt: int,
    n_ldpc: int,
    rate: float,
    snr_db_range: np.ndarray,
    rho: float = 1.0,
    admm_iter: int = 15,
    bp_iter: int = 10,
    idd_iter: int = 3,
    n_trials: int = 200,
    seed: int = 42
) -> dict:
    """
    Simulates Bit Error Rate performance across different SNRs for:
    1) MMSE + SPA
    2) ADMM + SPA (no iterations)
    3) ADMM IDD (iterative feedback)
    """
    from src.channel    import generate_channel, awgn_channel, mmse_detect
    from src.ldpc       import (make_ldpc_matrix, ldpc_encode, verify_codeword,
                                spa_decode_soft, build_tanner_graph)
    from src.admm_detector import (admm_detect, admm_llrs,
                                   admm_effective_sigma2)

    rng    = np.random.default_rng(seed)
    m      = int(n_ldpc * (1 - rate))
    k      = n_ldpc - m

    assert n_ldpc == Nt, (
        f"For this single-channel-use IDD demo, n_ldpc ({n_ldpc}) must equal "
        f"Nt ({Nt}).  For larger codes, use BICM (Day 8)."
    )

    H_ldpc = make_ldpc_matrix(n=n_ldpc, rate=rate, seed=0)
    vn_to_cn, cn_to_vn = build_tanner_graph(H_ldpc)

    sys_cols = list(range(k))

    results = {
        'snr':          list(snr_db_range),
        'ber_mmse_spa': np.zeros(len(snr_db_range)),
        'ber_admm_spa': np.zeros(len(snr_db_range)),
        'ber_idd':      np.zeros(len(snr_db_range)),
    }

    for i, snr_db in enumerate(snr_db_range):
        err_mmse = 0
        err_admm = 0
        err_idd  = 0
        total    = 0

        for _ in range(n_trials):

            u    = rng.integers(0, 2, size=k)
            c    = ldpc_encode(u, H_ldpc)

            x    = 1.0 - 2.0 * c.astype(float)

            H    = generate_channel(Nr, Nt, seed=None)
            y, s2 = awgn_channel(H, x, snr_db, rng)
            s2_eff = admm_effective_sigma2(H, s2, rho)

            x_mmse   = mmse_detect(y, H, s2)

            llr_mmse = 2.0 * np.real(x_mmse) / s2_eff
            llr_mmse = clip_llrs(llr_mmse)
            b_mmse, _, _, _ = spa_decode_soft(
                llr_mmse, H_ldpc, max_iter=bp_iter,
                vn_to_cn=vn_to_cn, cn_to_vn=cn_to_vn
            )
            err_mmse += int(np.sum(b_mmse[sys_cols] != u))

            z_hard, z_soft = admm_detect(
                y, H, s2, M=2, rho=rho,
                max_iter=admm_iter, return_soft=True
            )
            llr_admm = clip_llrs(admm_llrs(z_soft, s2_eff, M=2))
            b_admm, _, _, _ = spa_decode_soft(
                llr_admm, H_ldpc, max_iter=bp_iter,
                vn_to_cn=vn_to_cn, cn_to_vn=cn_to_vn
            )
            err_admm += int(np.sum(b_admm[sys_cols] != u))

            llr_prior = np.zeros(Nt)

            for _t in range(idd_iter):
                prior_in = llr_prior if _t > 0 else None
                _, z_soft = admm_detect(
                    y, H, s2, M=2, rho=rho,
                    max_iter=admm_iter,
                    llr_prior=prior_in,
                    return_soft=True
                )
                llr_det = clip_llrs(admm_llrs(z_soft, s2_eff, M=2))
                llr_ext_det = clip_llrs(llr_det - llr_prior)

                b_idd, llr_dec, _, _ = spa_decode_soft(
                    llr_ext_det, H_ldpc, max_iter=bp_iter,
                    vn_to_cn=vn_to_cn, cn_to_vn=cn_to_vn
                )
                llr_prior = clip_llrs(llr_dec - llr_ext_det)

            err_idd += int(np.sum(b_idd[sys_cols] != u))
            total   += k

        results['ber_mmse_spa'][i] = err_mmse / total
        results['ber_admm_spa'][i] = err_admm / total
        results['ber_idd'][i]      = err_idd  / total

        print(
            f"  SNR={snr_db:4.1f} dB | "
            f"MMSE+SPA={results['ber_mmse_spa'][i]:.5f} | "
            f"ADMM+SPA={results['ber_admm_spa'][i]:.5f} | "
            f"ADMM-IDD={results['ber_idd'][i]:.5f}"
        )

    return results

