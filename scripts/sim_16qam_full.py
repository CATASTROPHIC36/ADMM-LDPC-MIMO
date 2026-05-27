# ==============================================================================
# IMPORTS AND CONFIGURATION
# ==============================================================================
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

from src.channel import generate_channel, awgn_channel
from src.modulation import qam_modulate, qam_demodulate_hard, compute_llrs_mmse
from src.admm_detector import admm_detect, project_qam

Nr, Nt   = 4, 4
M_QAM    = 16
BPS      = int(np.log2(M_QAM))
N_TRIALS = 500
SNR_RANGE = np.arange(0, 26, 2)

ADMM_ITER = 20
EP_ITER   = 15
RHO_MAX   = 50.0
LLR_CLIP  = 30.0

rng = np.random.default_rng(42)
SEP = "=" * 70

# ==============================================================================
# DETECTOR FUNCTIONS
# ==============================================================================

def mmse_detect_16qam(y, H, sigma2):
    """
    Performs MMSE detection for 16-QAM signals.
    Computes LLRs using the MMSE estimator and returns hard bit decisions.
    """

    llrs = compute_llrs_mmse(y, H, sigma2, M=M_QAM)
    bits = (llrs < 0).astype(int)
    return bits

def admm_detect_16qam(y, H, sigma2, rho, max_iter=20):
    """
    Performs ADMM-based detection for 16-QAM signals.
    Iteratively solves the regularized least squares problem and projects
    results onto the 16-QAM constellation.
    """

    Nr, Nt = H.shape
    HH  = H.conj().T @ H
    Hhy = H.conj().T @ y / sigma2

    z = np.zeros(Nt, dtype=complex)
    u = np.zeros(Nt, dtype=complex)

    A = HH / sigma2 + rho * np.eye(Nt)
    A_inv = np.linalg.inv(A)

    for k in range(max_iter):
        rhs = Hhy + rho * (z - u)
        x = A_inv @ rhs
        v = x + u
        z = project_qam(v, M=M_QAM)
        u = u + x - z

    bits = qam_demodulate_hard(z, M=M_QAM)
    return bits

def ep_detect_16qam(y, H, sigma2, n_iter=15):
    """
    Performs Expectation Propagation (EP) detection for 16-QAM signals.
    Iteratively refines posterior marginals of the transmitted symbols
    using Gaussian approximations.
    """

    Nr, Nt = H.shape
    from src.modulation import _pam_levels, _pam_gray_bit_table

    m_pam = int(np.sqrt(M_QAM))
    levels = _pam_levels(m_pam)
    E_s = (2.0 / 3.0) * (M_QAM - 1)
    scale = np.sqrt(E_s)

    constellation_1d = levels / scale

    HH = H.conj().T @ H
    Hhy = H.conj().T @ y

    tau_site = np.full(Nt, 1e-6, dtype=complex)
    eta_site = np.zeros(Nt, dtype=complex)

    for it in range(n_iter):

        Lambda = HH / sigma2 + np.diag(np.real(tau_site))
        try:
            Sigma = np.linalg.inv(Lambda)
        except np.linalg.LinAlgError:
            Sigma = np.linalg.pinv(Lambda)

        mu = Sigma @ (Hhy / sigma2 + np.real(eta_site))

        for k in range(Nt):
            s_kk = max(np.real(Sigma[k, k]), 1e-12)
            tau_s_k = max(np.real(tau_site[k]), 1e-12)

            inv_v_cav = 1.0 / s_kk - tau_s_k
            if inv_v_cav <= 1e-12:
                continue
            v_cav = 1.0 / inv_v_cav
            mu_cav_real = v_cav * (np.real(mu[k]) / s_kk
                                    - np.real(eta_site[k]))

            log_probs = -0.5 * (constellation_1d - mu_cav_real)**2 / v_cav
            log_probs -= np.max(log_probs)
            probs = np.exp(log_probs)
            probs /= np.sum(probs) + 1e-30

            mu_new_real = np.sum(probs * constellation_1d)
            var_new_real = max(np.sum(probs * constellation_1d**2)
                               - mu_new_real**2, 1e-6)

            tau_new = 1.0 / var_new_real - inv_v_cav
            if tau_new <= 1e-12:
                continue
            tau_site[k] = tau_new
            eta_site[k] = tau_new * mu_new_real + inv_v_cav * mu_cav_real

    Lambda = HH / sigma2 + np.diag(np.real(tau_site))
    try:
        Sigma = np.linalg.inv(Lambda)
    except np.linalg.LinAlgError:
        Sigma = np.linalg.pinv(Lambda)
    mu = Sigma @ (Hhy / sigma2 + np.real(eta_site))

    z_hat = np.zeros(Nt, dtype=complex)
    for k in range(Nt):
        i_val = constellation_1d[np.argmin(np.abs(np.real(mu[k])
                                                    - constellation_1d))]
        q_val = constellation_1d[np.argmin(np.abs(np.imag(mu[k])
                                                    - constellation_1d))]
        z_hat[k] = i_val + 1j * q_val

    bits = qam_demodulate_hard(z_hat, M=M_QAM)
    return bits

print(SEP)
print("16-QAM End-to-End Comparison")
print(f"  {Nr}x{Nt} MIMO | 16-QAM ({BPS} bits/symbol)")
print(f"  {N_TRIALS} trials/SNR | SNR range: {SNR_RANGE[0]}-{SNR_RANGE[-1]} dB")
print(SEP)

# ==============================================================================
# PHASE 0: QUICK SANITY CHECKS
# ==============================================================================
print("\nPHASE 0: Quick sanity checks")
for snr_test in [10, 16, 22]:
    errs_mmse = errs_admm = errs_ep = 0
    total = 0
    for _ in range(200):
        bits_tx = rng.integers(0, 2, size=Nt * BPS)
        x = qam_modulate(bits_tx, M=M_QAM)
        H_t = generate_channel(Nr, Nt)
        y_t, s2_t = awgn_channel(H_t, x, snr_test, rng)
        rho_t = min(1.0 / max(s2_t, 1e-6), RHO_MAX)

        b_mmse = mmse_detect_16qam(y_t, H_t, s2_t)
        b_admm = admm_detect_16qam(y_t, H_t, s2_t, rho_t, ADMM_ITER)
        b_ep   = ep_detect_16qam(y_t, H_t, s2_t, EP_ITER)

        errs_mmse += int(np.sum(bits_tx != b_mmse))
        errs_admm += int(np.sum(bits_tx != b_admm))
        errs_ep   += int(np.sum(bits_tx != b_ep))
        total += Nt * BPS

    print(f"  SNR={snr_test:2d} dB: MMSE={errs_mmse/total:.4f}  "
          f"ADMM={errs_admm/total:.4f}  EP={errs_ep/total:.4f}")

# ==============================================================================
# PHASE 1: MAIN BER SWEEP
# ==============================================================================
print(f"\nPHASE 1: Main BER sweep ({N_TRIALS} trials/SNR)")
print(SEP)

n_snr = len(SNR_RANGE)
ber_mmse = np.zeros(n_snr)
ber_admm = np.zeros(n_snr)
ber_ep   = np.zeros(n_snr)

t_start = time.time()

for j, snr_db in enumerate(SNR_RANGE):
    e_mmse = e_admm = e_ep = 0
    total = 0

    for trial in range(N_TRIALS):
        bits_tx = rng.integers(0, 2, size=Nt * BPS)
        x = qam_modulate(bits_tx, M=M_QAM)
        H_t = generate_channel(Nr, Nt)
        y_t, s2_t = awgn_channel(H_t, x, snr_db, rng)
        rho_t = min(1.0 / max(s2_t, 1e-6), RHO_MAX)

        b_mmse = mmse_detect_16qam(y_t, H_t, s2_t)
        e_mmse += int(np.sum(bits_tx != b_mmse))

        b_admm = admm_detect_16qam(y_t, H_t, s2_t, rho_t, ADMM_ITER)
        e_admm += int(np.sum(bits_tx != b_admm))

        b_ep = ep_detect_16qam(y_t, H_t, s2_t, EP_ITER)
        e_ep += int(np.sum(bits_tx != b_ep))

        total += Nt * BPS

    ber_mmse[j] = e_mmse / total
    ber_admm[j] = e_admm / total
    ber_ep[j]   = e_ep / total

    elapsed = time.time() - t_start
    print(f"  SNR={snr_db:4.0f} dB | "
          f"MMSE={ber_mmse[j]:.5f} | "
          f"ADMM={ber_admm[j]:.5f} | "
          f"EP={ber_ep[j]:.5f} | "
          f"{elapsed:.0f}s")

# ==============================================================================
# PHASE 2: PLOTTING AND SUMMARY
# ==============================================================================
print(f"\nPHASE 2: Generating plot")

fig, ax = plt.subplots(1, 1, figsize=(10, 7))

ax.semilogy(SNR_RANGE, np.maximum(ber_mmse, 1e-6), 'b-o',
            linewidth=2, markersize=6, label='MMSE')
ax.semilogy(SNR_RANGE, np.maximum(ber_admm, 1e-6), 'r-d',
            linewidth=2.5, markersize=7, label=r'ADMM (adaptive $\rho$)')
ax.semilogy(SNR_RANGE, np.maximum(ber_ep, 1e-6), 'g-^',
            linewidth=2, markersize=6, label='EP')

ax.set_xlabel('SNR (dB)', fontsize=13)
ax.set_ylabel('Bit Error Rate (BER)', fontsize=13)
ax.set_title(f'{Nr}x{Nt} 16-QAM MIMO — Uncoded BER Comparison\n'
             f'({N_TRIALS} trials/SNR, ADMM_ITER={ADMM_ITER}, EP_ITER={EP_ITER})',
             fontsize=14)
ax.legend(fontsize=12, loc='lower left')
ax.grid(True, which='both', alpha=0.3)
ax.set_ylim([1e-5, 0.6])
ax.set_xlim([SNR_RANGE[0] - 1, SNR_RANGE[-1] + 1])

plt.tight_layout()
plt.savefig('data/16qam_comparison.png', dpi=200, bbox_inches='tight')
print("  Saved: data/16qam_comparison.png")

print(f"\n{SEP}")
print("SUMMARY: 16-QAM BER vs SNR")
print(SEP)
print(f"    SNR |       MMSE |       ADMM |         EP")
print(f"  {'-'*50}")
for j, snr_db in enumerate(SNR_RANGE):
    print(f"  {snr_db:3.0f}dB | {ber_mmse[j]:>10.5f} | "
          f"{ber_admm[j]:>10.5f} | {ber_ep[j]:>10.5f}")

print(f"\n  Monotonicity check:")
for name, arr in [("MMSE", ber_mmse), ("ADMM", ber_admm), ("EP", ber_ep)]:
    violations = sum(1 for i in range(1, len(arr))
                     if arr[i] > arr[i-1] + 1e-4)
    status = ('PASS (monotonic)' if violations == 0
              else f'FAIL ({violations} violations)')
    print(f"    {name:>8}: {status}")

total_time = time.time() - t_start
print(f"\nTotal runtime: {total_time:.0f}s ({total_time/60:.1f} min)")
print(SEP)

