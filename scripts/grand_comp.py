

# =============================================================================
# IMPORTS AND CONFIGURATION
# =============================================================================
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

from src.channel       import generate_channel, awgn_channel
from src.admm_detector import admm_detect
from src.ldpc          import (make_ldpc_matrix, ldpc_encode, verify_codeword,
                                spa_decode_soft, build_tanner_graph)

Nr, Nt    = 4, 4
N_LDPC    = 64
RATE      = 0.5
N_TRIALS  = 1500
SNR_RANGE = np.arange(0, 16, 2)

ADMM_ITER = 20
AMP_ITER  = 30
EP_ITER   = 15
IDD_ITER_LIST = [1, 2, 4]
BP_ITER   = 30
LLR_CLIP  = 30.0
AMP_DAMP  = 0.8

IDD_DAMP_SCHEDULE = [0.3, 0.5, 0.65, 0.8]

def get_idd_damp(idd_iter_idx, snr_db):

    base = IDD_DAMP_SCHEDULE[min(idd_iter_idx, len(IDD_DAMP_SCHEDULE) - 1)]

    snr_scale = np.clip((snr_db - 2.0) / 10.0, 0.3, 1.0)
    return base * snr_scale

USES_PER_BLOCK = N_LDPC // Nt

rng = np.random.default_rng(42)
SEP = "=" * 70

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_sys_cols(H):
    """
    Finds the systematic columns of a parity-check matrix H using Gaussian elimination.
    
    Args:
        H (np.ndarray): The parity check matrix.
        
    Returns:
        list: Indices of the systematic columns.
    """

    m, n = H.shape
    H_w  = H.astype(np.int8).copy()
    pivots, pr = [], 0
    for col in range(n):
        if pr >= m: break
        for row in range(pr, m):
            if H_w[row, col] == 1:
                H_w[[pr, row]] = H_w[[row, pr]]
                for r2 in range(m):
                    if r2 != pr and H_w[r2, col]:
                        H_w[r2] = (H_w[r2] + H_w[pr]) % 2
                pivots.append(col)
                pr += 1
                break
    return [c for c in range(n) if c not in set(pivots)]

# =============================================================================
# LDPC SETUP & VALIDATION
# =============================================================================
H_ldpc = make_ldpc_matrix(n=N_LDPC, rate=RATE, seed=0)
vn_to_cn, cn_to_vn = build_tanner_graph(H_ldpc)
m_ldpc  = H_ldpc.shape[0]
k_ldpc  = N_LDPC - m_ldpc
sys_cols = get_sys_cols(H_ldpc)
assert len(sys_cols) == k_ldpc, f"sys_cols count {len(sys_cols)} != k={k_ldpc}"

u_test = rng.integers(0, 2, size=k_ldpc)
c_test = ldpc_encode(u_test, H_ldpc)
assert verify_codeword(c_test, H_ldpc), "LDPC encoder broken!"

print(SEP)
print("DAY 14 v2 -- Grand Comparison -- Publication-Quality Summary")
print(f"  {Nr}x{Nt} BPSK | LDPC(n={N_LDPC}, R={RATE}) | "
      f"{USES_PER_BLOCK} uses/block | {N_TRIALS} trials/SNR")
print(f"  ADMM_ITER={ADMM_ITER} | EP_ITER={EP_ITER} | AMP_ITER={AMP_ITER}")
print(f"  IDD iterations: {IDD_ITER_LIST} | BP_ITER={BP_ITER}")
print(f"  LLR_CLIP={LLR_CLIP} | IDD_DAMP_SCHED={IDD_DAMP_SCHEDULE} | AMP_DAMP={AMP_DAMP}")
print(SEP)

# =============================================================================
# DETECTOR IMPLEMENTATIONS
# =============================================================================
def compute_mmse_llr(y, H, sigma2):
    """
    Computes LLRs using the Minimum Mean Square Error (MMSE) detector.
    
    Args:
        y (np.ndarray): Received signal vector.
        H (np.ndarray): Channel matrix.
        sigma2 (float): Noise variance.
        
    Returns:
        np.ndarray: Log-likelihood ratios.
    """

    Nr, Nt = H.shape
    HH  = H.conj().T @ H
    A   = HH / sigma2 + np.eye(Nt)
    Hhy = H.conj().T @ y / sigma2
    x_hat = np.real(np.linalg.solve(A, Hhy))

    A_inv = np.linalg.inv(A)
    s2_eff = np.maximum(np.real(np.diag(A_inv)), 1e-9)
    llr = 2.0 * x_hat / s2_eff
    return np.clip(llr, -LLR_CLIP, LLR_CLIP)

RHO_MAX = 50.0

def get_rho(sigma2):
    return min(1.0 / max(sigma2, 1e-6), RHO_MAX)

def compute_admm_llr(y, H, sigma2, rho, llr_prior=None):
    """
    Computes LLRs using the ADMM-based detector.
    
    Args:
        y (np.ndarray): Received signal vector.
        H (np.ndarray): Channel matrix.
        sigma2 (float): Noise variance.
        rho (float): ADMM penalty parameter.
        llr_prior (np.ndarray, optional): Prior LLRs from decoder.
        
    Returns:
        np.ndarray: Extrinsic Log-likelihood ratios.
    """

    Nr, Nt = H.shape
    _, x_soft = admm_detect(y, H, sigma2, M=2, rho=rho,
                             max_iter=ADMM_ITER,
                             llr_prior=llr_prior,
                             return_soft=True)

    HH    = H.conj().T @ H
    A     = HH / sigma2 + rho * np.eye(Nt)
    A_inv = np.linalg.inv(A)
    s2_eff = np.maximum(np.real(np.diag(A_inv)), 1e-9)
    llr    = 2.0 * np.real(x_soft) / s2_eff
    return np.clip(llr, -LLR_CLIP, LLR_CLIP)

def ep_detect_bpsk(y, H, sigma2, n_iter=15, llr_prior=None):
    """
    Computes LLRs using Expectation Propagation (EP) detector.
    
    Args:
        y (np.ndarray): Received signal vector.
        H (np.ndarray): Channel matrix.
        sigma2 (float): Noise variance.
        n_iter (int): Number of EP iterations.
        llr_prior (np.ndarray, optional): Prior LLRs from decoder.
        
    Returns:
        np.ndarray: Log-likelihood ratios.
    """

    Nr, Nt = H.shape
    HH  = np.real(H.conj().T @ H)
    Hhy = np.real(H.conj().T @ y)

    v_site = np.full(Nt, 1e6)
    m_site = np.zeros(Nt)

    if llr_prior is not None:
        m_site = np.tanh(np.clip(llr_prior / 2.0, -10, 10))
        v_site = np.maximum(1.0 - m_site**2, 1e-6)

    for _ in range(n_iter):

        Lambda = HH / sigma2 + np.diag(1.0 / np.maximum(v_site, 1e-9))
        try:    Sigma = np.linalg.inv(Lambda)
        except: Sigma = np.linalg.pinv(Lambda)
        mu = Sigma @ (Hhy / sigma2 + m_site / np.maximum(v_site, 1e-9))

        for k in range(Nt):
            s_kk = max(Sigma[k, k], 1e-12)
            v_sk  = max(v_site[k], 1e-12)

            inv_v_cav = 1.0 / s_kk - 1.0 / v_sk
            if inv_v_cav <= 1e-12:
                continue
            v_cav  = 1.0 / inv_v_cav
            mu_cav = v_cav * (mu[k] / s_kk - m_site[k] / v_sk)

            arg    = np.clip(mu_cav / max(v_cav, 1e-9), -15, 15)
            mu_new = np.tanh(arg)
            v_new  = max(1.0 - mu_new**2, 1e-6)

            inv_vs_new = 1.0 / v_new - inv_v_cav
            if inv_vs_new <= 1e-12:
                continue
            v_site[k] = 1.0 / inv_vs_new
            m_site[k] = v_site[k] * (mu_new / v_new - mu_cav / v_cav)

    Lambda = HH / sigma2 + np.diag(1.0 / np.maximum(v_site, 1e-9))
    try:    Sigma = np.linalg.inv(Lambda)
    except: Sigma = np.linalg.pinv(Lambda)
    mu = Sigma @ (Hhy / sigma2 + m_site / np.maximum(v_site, 1e-9))

    s2_diag = np.maximum(np.real(np.diag(Sigma)), 1e-9)
    llr_ep  = np.clip(2.0 * mu / s2_diag, -LLR_CLIP, LLR_CLIP)
    return llr_ep

def amp_detect_bpsk(y, H, sigma2, n_iter=30, damp=0.8, llr_prior=None):
    """
    Computes LLRs using Approximate Message Passing (AMP) detector.
    
    Args:
        y (np.ndarray): Received signal vector.
        H (np.ndarray): Channel matrix.
        sigma2 (float): Noise variance.
        n_iter (int): Number of AMP iterations.
        damp (float): Damping factor to assist convergence.
        llr_prior (np.ndarray, optional): Prior LLRs from decoder.
        
    Returns:
        np.ndarray: Log-likelihood ratios.
    """

    Nr, Nt = H.shape
    beta = Nt / Nr

    H_real = np.real(H)
    y_real = np.real(y)

    x_hat = np.zeros(Nt)
    r = y_real.copy()
    tau2 = sigma2 + 1.0

    if llr_prior is not None:
        x_hat = np.tanh(np.clip(llr_prior / 2.0, -10, 10))

    for it in range(n_iter):

        p = H_real.T @ r + x_hat

        v_site = np.maximum(1.0 - x_hat**2, 1e-6)
        tau2_new = sigma2 + beta * np.mean(v_site)
        tau2 = max(tau2_new, 1e-6)

        arg = np.clip(p / max(tau2, 1e-9), -15, 15)
        x_hat_new = np.tanh(arg)

        x_hat = damp * x_hat_new + (1 - damp) * x_hat

        v_site = np.maximum(1.0 - x_hat**2, 1e-6)
        onsager = (Nt / Nr) * np.mean(v_site) * r / max(tau2, 1e-9)
        r = y_real - H_real @ x_hat + onsager

    p_final = H_real.T @ r + x_hat
    llr = 2.0 * p_final / max(tau2, 1e-9)
    return np.clip(llr, -LLR_CLIP, LLR_CLIP)

# =============================================================================
# PHASE 0: ISOLATION TESTS
# =============================================================================
print("\n" + SEP)
print("PHASE 0: Isolation tests")
print(SEP)

print("\n  Test 1: SISO BPSK AWGN (no MIMO, no LDPC)")
test_ok = True
for snr_db in [0, 4, 8]:
    snr_lin = 10**(snr_db/10)
    s2 = 1.0 / snr_lin
    errs = 0
    for _ in range(5000):
        b = rng.integers(0, 2)
        x = 1.0 - 2.0 * float(b)
        n_r = rng.standard_normal() * np.sqrt(s2 / 2)
        y_r = x + n_r
        llr = 2.0 * y_r / s2
        b_hat = int(llr < 0)
        errs += int(b != b_hat)
    ber = errs / 5000
    from scipy.special import erfc
    ber_theory = 0.5 * erfc(np.sqrt(snr_lin))
    ok = abs(ber - ber_theory) < 0.02
    test_ok &= ok
    print(f"    SNR={snr_db:2d} dB: BER={ber:.4f} theory={ber_theory:.4f} {'OK' if ok else 'FAIL'}")
print(f"  Test 1: {'PASSED' if test_ok else 'FAILED'}")

for det_name, det_fn in [("MMSE", lambda y,H,s2: compute_mmse_llr(y,H,s2)),
                          ("ADMM", lambda y,H,s2: compute_admm_llr(y,H,s2,get_rho(s2))),
                          ("EP",   lambda y,H,s2: ep_detect_bpsk(y,H,s2,n_iter=EP_ITER)),
                          ("AMP",  lambda y,H,s2: amp_detect_bpsk(y,H,s2,n_iter=AMP_ITER,damp=AMP_DAMP))]:
    print(f"\n  Test 2-{det_name}: {Nr}x{Nt} MIMO {det_name} uncoded (BER must decrease)")
    ber_prev = 1.0
    test_ok = True
    for snr_db in [0, 4, 8, 12]:
        errs, total = 0, 0
        for _ in range(1000):
            bits = rng.integers(0, 2, size=Nt)
            x = 1.0 - 2.0 * bits.astype(float)
            H_t = generate_channel(Nr, Nt)
            y_t, s2_t = awgn_channel(H_t, x, snr_db, rng)
            llr = det_fn(y_t, H_t, s2_t)
            b_hat = (llr < 0).astype(int)
            errs += int(np.sum(bits != b_hat))
            total += Nt
        ber = errs / total
        mono = ber <= ber_prev + 0.01
        test_ok &= mono
        print(f"    SNR={snr_db:2d} dB: BER={ber:.4f} {'OK' if mono else 'NOT MONOTONIC'}")
        ber_prev = ber
    print(f"  Test 2-{det_name}: {'PASSED' if test_ok else 'FAILED'}")

# =============================================================================
# PHASE 1: MAIN BER SIMULATION
# =============================================================================
print("\n" + SEP)
print("PHASE 1: Main BER simulation")
print(SEP)

n_snr = len(SNR_RANGE)
IDD_MAX = max(IDD_ITER_LIST)

ber_uncoded_mmse = np.zeros(n_snr)
ber_mmse_spa     = np.zeros(n_snr)
ber_admm_spa     = np.zeros(n_snr)
ber_ep_spa       = np.zeros(n_snr)
ber_amp_spa      = np.zeros(n_snr)
ber_admm_idd     = {k: np.zeros(n_snr) for k in IDD_ITER_LIST}
ber_ep_idd       = np.zeros(n_snr)
ber_amp_idd      = np.zeros(n_snr)

CONV_SNRS = [6, 8, 10, 12]
conv_admm_ber = {snr: [[] for _ in range(IDD_MAX)] for snr in CONV_SNRS}
conv_ep_ber   = {snr: [[] for _ in range(IDD_MAX)] for snr in CONV_SNRS}

t_start = time.time()

for j, snr_db in enumerate(SNR_RANGE):
    e_uncoded = 0
    e_mmse = e_admm = e_ep = e_amp = 0
    e_admm_idd = {k: 0 for k in IDD_ITER_LIST}
    e_ep_idd = e_amp_idd = 0
    total_info = 0
    total_uncoded = 0

    is_conv = (snr_db in CONV_SNRS)

    for trial in range(N_TRIALS):
        u = rng.integers(0, 2, size=k_ldpc)
        c = ldpc_encode(u, H_ldpc)
        c_2d = c.reshape(USES_PER_BLOCK, Nt)
        x_2d = 1.0 - 2.0 * c_2d.astype(float)

        channels = []
        for t in range(USES_PER_BLOCK):
            H_t = generate_channel(Nr, Nt)
            y_t, s2_t = awgn_channel(H_t, x_2d[t], snr_db, rng)
            rho_t = get_rho(s2_t)
            channels.append((H_t, y_t, s2_t, rho_t))

        for t in range(USES_PER_BLOCK):
            H_t, y_t, s2_t, _ = channels[t]
            llr_t = compute_mmse_llr(y_t, H_t, s2_t)
            b_hat = (llr_t < 0).astype(int)
            e_uncoded += int(np.sum(b_hat != c_2d[t]))
        total_uncoded += N_LDPC

        llr_block = np.zeros(N_LDPC)
        for t in range(USES_PER_BLOCK):
            H_t, y_t, s2_t, _ = channels[t]
            llr_block[t*Nt:(t+1)*Nt] = compute_mmse_llr(y_t, H_t, s2_t)
        b_dec, _, _, _ = spa_decode_soft(llr_block, H_ldpc, max_iter=BP_ITER,
                                          vn_to_cn=vn_to_cn, cn_to_vn=cn_to_vn)
        e_mmse += int(np.sum(b_dec[sys_cols] != u))

        llr_block = np.zeros(N_LDPC)
        for t in range(USES_PER_BLOCK):
            H_t, y_t, s2_t, rho_t = channels[t]
            llr_block[t*Nt:(t+1)*Nt] = compute_admm_llr(y_t, H_t, s2_t, rho_t)
        b_dec, _, _, _ = spa_decode_soft(llr_block, H_ldpc, max_iter=BP_ITER,
                                          vn_to_cn=vn_to_cn, cn_to_vn=cn_to_vn)
        e_admm += int(np.sum(b_dec[sys_cols] != u))

        llr_block = np.zeros(N_LDPC)
        for t in range(USES_PER_BLOCK):
            H_t, y_t, s2_t, _ = channels[t]
            llr_block[t*Nt:(t+1)*Nt] = ep_detect_bpsk(y_t, H_t, s2_t,
                                                        n_iter=EP_ITER)
        b_dec, _, _, _ = spa_decode_soft(llr_block, H_ldpc, max_iter=BP_ITER,
                                          vn_to_cn=vn_to_cn, cn_to_vn=cn_to_vn)
        e_ep += int(np.sum(b_dec[sys_cols] != u))

        llr_block = np.zeros(N_LDPC)
        for t in range(USES_PER_BLOCK):
            H_t, y_t, s2_t, _ = channels[t]
            llr_block[t*Nt:(t+1)*Nt] = amp_detect_bpsk(y_t, H_t, s2_t,
                                                         n_iter=AMP_ITER,
                                                         damp=AMP_DAMP)
        b_dec, _, _, _ = spa_decode_soft(llr_block, H_ldpc, max_iter=BP_ITER,
                                          vn_to_cn=vn_to_cn, cn_to_vn=cn_to_vn)
        e_amp += int(np.sum(b_dec[sys_cols] != u))

        llr_prior = np.zeros(N_LDPC)
        b_ai = None
        for idd_t in range(IDD_MAX):
            llr_det = np.zeros(N_LDPC)
            for t in range(USES_PER_BLOCK):
                H_t, y_t, s2_t, rho_t = channels[t]
                prior_t = llr_prior[t*Nt:(t+1)*Nt] if idd_t > 0 else None
                llr_det[t*Nt:(t+1)*Nt] = compute_admm_llr(
                    y_t, H_t, s2_t, rho_t, llr_prior=prior_t)

            if idd_t == 0:
                llr_ext = llr_det
            else:
                llr_ext = np.clip(llr_det - llr_prior, -LLR_CLIP, LLR_CLIP)

            b_ai, llr_dec, _, _ = spa_decode_soft(
                llr_ext, H_ldpc, max_iter=BP_ITER,
                vn_to_cn=vn_to_cn, cn_to_vn=cn_to_vn)

            curr_iter = idd_t + 1
            if curr_iter in IDD_ITER_LIST:
                e_admm_idd[curr_iter] += int(np.sum(b_ai[sys_cols] != u))

            if is_conv:
                ber_this = float(np.mean(b_ai[sys_cols] != u))
                conv_admm_ber[snr_db][idd_t].append(ber_this)

            if idd_t < IDD_MAX - 1:
                damp = get_idd_damp(idd_t, snr_db)
                llr_prior = damp * np.clip(llr_dec - llr_ext, -LLR_CLIP, LLR_CLIP)

        llr_prior = np.zeros(N_LDPC)
        b_ei = None
        for idd_t in range(IDD_MAX):
            llr_det = np.zeros(N_LDPC)
            for t in range(USES_PER_BLOCK):
                H_t, y_t, s2_t, _ = channels[t]
                prior_t = llr_prior[t*Nt:(t+1)*Nt] if idd_t > 0 else None
                llr_det[t*Nt:(t+1)*Nt] = ep_detect_bpsk(
                    y_t, H_t, s2_t, n_iter=EP_ITER, llr_prior=prior_t)

            if idd_t == 0:
                llr_ext = llr_det
            else:
                llr_ext = np.clip(llr_det - llr_prior, -LLR_CLIP, LLR_CLIP)

            b_ei, llr_dec, _, _ = spa_decode_soft(
                llr_ext, H_ldpc, max_iter=BP_ITER,
                vn_to_cn=vn_to_cn, cn_to_vn=cn_to_vn)

            if is_conv:
                ber_this = float(np.mean(b_ei[sys_cols] != u))
                conv_ep_ber[snr_db][idd_t].append(ber_this)

            if idd_t < IDD_MAX - 1:
                damp = get_idd_damp(idd_t, snr_db)
                llr_prior = damp * np.clip(llr_dec - llr_ext, -LLR_CLIP, LLR_CLIP)

        e_ep_idd += int(np.sum(b_ei[sys_cols] != u))

        llr_prior = np.zeros(N_LDPC)
        b_ampi = None
        for idd_t in range(IDD_MAX):
            llr_det = np.zeros(N_LDPC)
            for t in range(USES_PER_BLOCK):
                H_t, y_t, s2_t, _ = channels[t]
                prior_t = llr_prior[t*Nt:(t+1)*Nt] if idd_t > 0 else None
                llr_det[t*Nt:(t+1)*Nt] = amp_detect_bpsk(
                    y_t, H_t, s2_t, n_iter=AMP_ITER, damp=AMP_DAMP,
                    llr_prior=prior_t)

            if idd_t == 0:
                llr_ext = llr_det
            else:
                llr_ext = np.clip(llr_det - llr_prior, -LLR_CLIP, LLR_CLIP)

            b_ampi, llr_dec, _, _ = spa_decode_soft(
                llr_ext, H_ldpc, max_iter=BP_ITER,
                vn_to_cn=vn_to_cn, cn_to_vn=cn_to_vn)

            if idd_t < IDD_MAX - 1:
                damp = get_idd_damp(idd_t, snr_db)
                llr_prior = damp * np.clip(llr_dec - llr_ext, -LLR_CLIP, LLR_CLIP)

        e_amp_idd += int(np.sum(b_ampi[sys_cols] != u))

        total_info += k_ldpc

    ber_uncoded_mmse[j] = e_uncoded / total_uncoded
    ber_mmse_spa[j]     = e_mmse    / total_info
    ber_admm_spa[j]     = e_admm    / total_info
    ber_ep_spa[j]       = e_ep      / total_info
    ber_amp_spa[j]      = e_amp     / total_info
    for k in IDD_ITER_LIST:
        ber_admm_idd[k][j] = e_admm_idd[k] / total_info
    ber_ep_idd[j]       = e_ep_idd  / total_info
    ber_amp_idd[j]      = e_amp_idd / total_info

    elapsed = time.time() - t_start
    print(
        f"  SNR={snr_db:4.0f} dB | "
        f"Uncoded={ber_uncoded_mmse[j]:.5f} | "
        f"MMSE={ber_mmse_spa[j]:.5f} | "
        f"ADMM={ber_admm_spa[j]:.5f} | "
        f"EP={ber_ep_spa[j]:.5f} | "
        f"AMP={ber_amp_spa[j]:.5f} | "
        f"ADMM-IDD4={ber_admm_idd[4][j]:.5f} | "
        f"EP-IDD4={ber_ep_idd[j]:.5f} | "
        f"AMP-IDD4={ber_amp_idd[j]:.5f} | "
        f"{elapsed:.0f}s"
    )

# =============================================================================
# PHASE 2: PLOTTING
# =============================================================================
print("\n" + SEP)
print("PHASE 2: Generating publication-quality plots")
print(SEP)

os.makedirs("data", exist_ok=True)
eps = 1e-6

colors = {
    'uncoded':     '#888888',
    'mmse_spa':    '#1f77b4',
    'admm_spa':    '#ff7f0e',
    'ep_spa':      '#2ca02c',
    'amp_spa':     '#d62728',
    'admm_idd1':   '#9467bd',
    'admm_idd2':   '#8c564b',
    'admm_idd4':   '#e377c2',
    'ep_idd4':     '#17becf',
    'amp_idd4':    '#bcbd22',
}

fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))

ax1.semilogy(SNR_RANGE, np.maximum(ber_uncoded_mmse, eps),
             color=colors['uncoded'], ls=':', lw=1.5, marker='x', ms=7,
             label='Uncoded MMSE (no LDPC)')

ax1.semilogy(SNR_RANGE, np.maximum(ber_mmse_spa, eps),
             color=colors['mmse_spa'], ls='--', lw=1.5, marker='o', ms=6,
             label='MMSE + SPA')

ax1.semilogy(SNR_RANGE, np.maximum(ber_admm_spa, eps),
             color=colors['admm_spa'], ls='--', lw=1.5, marker='s', ms=6,
             label='ADMM + SPA')

ax1.semilogy(SNR_RANGE, np.maximum(ber_ep_spa, eps),
             color=colors['ep_spa'], ls='--', lw=1.5, marker='^', ms=6,
             label='EP + SPA')

ax1.semilogy(SNR_RANGE, np.maximum(ber_amp_spa, eps),
             color=colors['amp_spa'], ls='--', lw=1.5, marker='v', ms=6,
             label='AMP + SPA')

ax1.semilogy(SNR_RANGE, np.maximum(ber_admm_idd[1], eps),
             color=colors['admm_idd1'], ls='-', lw=2.5, marker='D', ms=7,
             label='ADMM-IDD (1 iter)')

ax1.semilogy(SNR_RANGE, np.maximum(ber_admm_idd[2], eps),
             color=colors['admm_idd2'], ls='-', lw=2.5, marker='p', ms=7,
             label='ADMM-IDD (2 iter)')

ax1.semilogy(SNR_RANGE, np.maximum(ber_admm_idd[4], eps),
             color=colors['admm_idd4'], ls='-', lw=2.5, marker='*', ms=9,
             label='ADMM-IDD (4 iter)')

ax1.semilogy(SNR_RANGE, np.maximum(ber_ep_idd, eps),
             color=colors['ep_idd4'], ls='-', lw=2.5, marker='h', ms=8,
             label='EP-IDD (4 iter)')

ax1.semilogy(SNR_RANGE, np.maximum(ber_amp_idd, eps),
             color=colors['amp_idd4'], ls='-', lw=2.5, marker='H', ms=8,
             label='AMP-IDD (4 iter)')

ax1.set_xlabel("SNR (dB)", fontsize=14)
ax1.set_ylabel("BER", fontsize=14)
ax1.set_title(f"$4 \\times 4$ BPSK MIMO, LDPC(64, 0.5), N={N_TRIALS} trials",
              fontsize=14)
ax1.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1.05, 1.0),
           borderaxespad=0, framealpha=0.9)
ax1.grid(True, which='both', alpha=0.3)
ax1.set_ylim([eps, 0.5])
ax1.set_xlim([SNR_RANGE[0] - 0.5, SNR_RANGE[-1] + 0.5])
ax1.tick_params(labelsize=12)

fig1.tight_layout()
fig1.savefig("data/day14_main_comparison.png", dpi=150, bbox_inches='tight')
print("  Saved: data/day14_main_comparison.png")

fig2, axes2 = plt.subplots(2, 2, figsize=(16, 13))

idd_iters_axis = list(range(1, IDD_MAX + 1))
conv_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

ax_p1 = axes2[0, 0]
for i_s, snr_val in enumerate(CONV_SNRS):
    if snr_val in conv_admm_ber and len(conv_admm_ber[snr_val][0]) > 0:
        ber_per_iter = [np.mean(conv_admm_ber[snr_val][t])
                        for t in range(IDD_MAX)]
        ax_p1.plot(idd_iters_axis, ber_per_iter,
                   '-o', lw=2, ms=8, color=conv_colors[i_s],
                   label=f'SNR={snr_val} dB')
        for it_idx, v in enumerate(ber_per_iter):
            ax_p1.annotate(f"{v:.4f}", (it_idx+1, v),
                           textcoords="offset points", xytext=(5, 5),
                           fontsize=7, color=conv_colors[i_s])
ax_p1.set_xlabel("IDD Outer Iteration", fontsize=12)
ax_p1.set_ylabel("Mean BER", fontsize=12)
ax_p1.set_title("ADMM-IDD: BER per IDD Iteration", fontsize=13)
ax_p1.legend(fontsize=10)
ax_p1.set_xticks(idd_iters_axis)
ax_p1.grid(True, alpha=0.4)

ax_p2 = axes2[0, 1]
for i_s, snr_val in enumerate(CONV_SNRS):
    if snr_val in conv_ep_ber and len(conv_ep_ber[snr_val][0]) > 0:
        ber_per_iter = [np.mean(conv_ep_ber[snr_val][t])
                        for t in range(IDD_MAX)]
        ax_p2.plot(idd_iters_axis, ber_per_iter,
                   '-s', lw=2, ms=8, color=conv_colors[i_s],
                   label=f'SNR={snr_val} dB')
        for it_idx, v in enumerate(ber_per_iter):
            ax_p2.annotate(f"{v:.4f}", (it_idx+1, v),
                           textcoords="offset points", xytext=(5, 5),
                           fontsize=7, color=conv_colors[i_s])
ax_p2.set_xlabel("IDD Outer Iteration", fontsize=12)
ax_p2.set_ylabel("Mean BER", fontsize=12)
ax_p2.set_title("EP-IDD: BER per IDD Iteration", fontsize=13)
ax_p2.legend(fontsize=10)
ax_p2.set_xticks(idd_iters_axis)
ax_p2.grid(True, alpha=0.4)

ax_p3 = axes2[1, 0]
target_ber = 0.01

def interp_snr_at_ber(snr_arr, ber_arr, target):

    ber_log = np.log10(np.maximum(ber_arr, 1e-10))
    target_log = np.log10(target)
    for i in range(len(ber_log) - 1):
        if (ber_log[i] >= target_log >= ber_log[i+1]) or           (ber_log[i] <= target_log <= ber_log[i+1]):

            frac = (target_log - ber_log[i]) / (ber_log[i+1] - ber_log[i] + 1e-15)
            return snr_arr[i] + frac * (snr_arr[i+1] - snr_arr[i])
    return np.nan

snr_uncoded_ref = interp_snr_at_ber(SNR_RANGE, ber_uncoded_mmse, target_ber)

methods_for_gain = [
    ("MMSE+SPA",     ber_mmse_spa),
    ("ADMM+SPA",     ber_admm_spa),
    ("EP+SPA",       ber_ep_spa),
    ("AMP+SPA",      ber_amp_spa),
    ("ADMM-IDD(1)",  ber_admm_idd[1]),
    ("ADMM-IDD(2)",  ber_admm_idd[2]),
    ("ADMM-IDD(4)",  ber_admm_idd[4]),
    ("EP-IDD(4)",    ber_ep_idd),
    ("AMP-IDD(4)",   ber_amp_idd),
]

gain_names = []
gain_values = []
gain_bar_colors = [colors['mmse_spa'], colors['admm_spa'], colors['ep_spa'],
                   colors['amp_spa'], colors['admm_idd1'], colors['admm_idd2'],
                   colors['admm_idd4'], colors['ep_idd4'], colors['amp_idd4']]

for (name, ber_arr) in methods_for_gain:
    snr_method = interp_snr_at_ber(SNR_RANGE, ber_arr, target_ber)
    if not np.isnan(snr_uncoded_ref) and not np.isnan(snr_method):
        gain = snr_uncoded_ref - snr_method
    else:
        gain = 0.0
    gain_names.append(name)
    gain_values.append(gain)

bars = ax_p3.barh(range(len(gain_names)), gain_values,
                  color=gain_bar_colors[:len(gain_names)], edgecolor='black',
                  linewidth=0.5)
ax_p3.set_yticks(range(len(gain_names)))
ax_p3.set_yticklabels(gain_names, fontsize=10)
ax_p3.set_xlabel(f"Coding Gain (dB) over Uncoded MMSE at BER={target_ber}",
                 fontsize=11)
ax_p3.set_title(f"Coding Gain at BER = {target_ber}", fontsize=13)
ax_p3.grid(True, axis='x', alpha=0.3)

for bar, val in zip(bars, gain_values):
    if val > 0:
        ax_p3.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                   f"{val:.1f} dB", va='center', fontsize=9, fontweight='bold')
    else:
        ax_p3.text(0.05, bar.get_y() + bar.get_height()/2,
                   "N/A", va='center', fontsize=9, color='gray')

ax_p4 = axes2[1, 1]
ax_p4.axis('off')

table_snrs = [8, 10, 12]
table_snr_idx = [list(SNR_RANGE).index(s) for s in table_snrs if s in SNR_RANGE]

row_labels = [
    "Uncoded MMSE",
    "MMSE + SPA",
    "ADMM + SPA",
    "EP + SPA",
    "AMP + SPA",
    "ADMM-IDD(1)",
    "ADMM-IDD(2)",
    "ADMM-IDD(4)",
    "EP-IDD(4)",
    "AMP-IDD(4)",
]

all_ber_arrs = [
    ber_uncoded_mmse,
    ber_mmse_spa,
    ber_admm_spa,
    ber_ep_spa,
    ber_amp_spa,
    ber_admm_idd[1],
    ber_admm_idd[2],
    ber_admm_idd[4],
    ber_ep_idd,
    ber_amp_idd,
]

col_labels = [f"SNR={s} dB" for s in table_snrs]
cell_text = []
for ber_arr in all_ber_arrs:
    row = []
    for idx in table_snr_idx:
        val = ber_arr[idx]
        if val < 1e-5:
            row.append("<1e-5")
        else:
            row.append(f"{val:.5f}")
    cell_text.append(row)

table = ax_p4.table(cellText=cell_text, rowLabels=row_labels,
                     colLabels=col_labels, loc='center',
                     cellLoc='center', rowLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.4)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white', fontweight='bold')
    elif col == -1:
        cell.set_facecolor('#D9E2F3')
        cell.set_text_props(fontweight='bold')

ax_p4.set_title("BER Summary at Key SNR Points", fontsize=13, pad=20)

fig2.suptitle(
    f"Day 14: IDD Convergence Analysis — "
    f"{Nr}×{Nt} BPSK, LDPC(64, 0.5), N={N_TRIALS}",
    fontsize=14, y=1.01
)
fig2.tight_layout()
fig2.savefig("data/day14_idd_convergence.png", dpi=150, bbox_inches='tight')
print("  Saved: data/day14_idd_convergence.png")

# =============================================================================
# SUMMARY & VERIFICATION
# =============================================================================
print("\n" + SEP)
print("SUMMARY: BER vs SNR — All Methods")
print(SEP)
header = (f"  {'SNR':>5} | {'Uncoded':>10} | {'MMSE+SPA':>10} | "
          f"{'ADMM+SPA':>10} | {'EP+SPA':>10} | {'AMP+SPA':>10} | "
          f"{'ADMM-1':>10} | {'ADMM-2':>10} | {'ADMM-4':>10} | "
          f"{'EP-IDD4':>10} | {'AMP-IDD4':>10}")
print(header)
print("  " + "-" * (len(header) - 2))
for j, snr_db in enumerate(SNR_RANGE):
    print(f"  {snr_db:4.0f}dB | "
          f"{ber_uncoded_mmse[j]:>10.5f} | "
          f"{ber_mmse_spa[j]:>10.5f} | "
          f"{ber_admm_spa[j]:>10.5f} | "
          f"{ber_ep_spa[j]:>10.5f} | "
          f"{ber_amp_spa[j]:>10.5f} | "
          f"{ber_admm_idd[1][j]:>10.5f} | "
          f"{ber_admm_idd[2][j]:>10.5f} | "
          f"{ber_admm_idd[4][j]:>10.5f} | "
          f"{ber_ep_idd[j]:>10.5f} | "
          f"{ber_amp_idd[j]:>10.5f}")

print("\n  Monotonicity check:")
mono_methods = [
    ("Uncoded MMSE", ber_uncoded_mmse),
    ("MMSE+SPA",     ber_mmse_spa),
    ("ADMM+SPA",     ber_admm_spa),
    ("EP+SPA",       ber_ep_spa),
    ("AMP+SPA",      ber_amp_spa),
    ("ADMM-IDD(1)",  ber_admm_idd[1]),
    ("ADMM-IDD(2)",  ber_admm_idd[2]),
    ("ADMM-IDD(4)",  ber_admm_idd[4]),
    ("EP-IDD(4)",    ber_ep_idd),
    ("AMP-IDD(4)",   ber_amp_idd),
]
all_pass = True
for name, arr in mono_methods:
    violations = sum(1 for i in range(1, len(arr)) if arr[i] > arr[i-1] + 1e-4)
    status = 'PASS (monotonic)' if violations == 0 else f'FAIL ({violations} violations)'
    if violations > 0:
        all_pass = False
    print(f"    {name:>15}: {status}")

print(f"\n  Coding gain over uncoded MMSE at BER = {target_ber}:")
print(f"    Uncoded MMSE reference SNR: {snr_uncoded_ref:.1f} dB" if not np.isnan(snr_uncoded_ref)
      else "    Uncoded MMSE: BER never crosses target")
for name, val in zip(gain_names, gain_values):
    if val > 0:
        print(f"    {name:>15}: {val:.1f} dB gain")
    else:
        print(f"    {name:>15}: N/A (target BER not reached)")

print(SEP)
total_time = time.time() - t_start
print(f"\nTotal runtime: {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"Monotonicity: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
print(f"\nPlots saved:")
print(f"  data/day14_main_comparison.png")
print(f"  data/day14_idd_convergence.png")

