"""
test_idd_ber.py
---------------
Runs a BER sweep comparing:
  1. MMSE + SPA-LDPC
  2. ADMM + SPA-LDPC (no iterations)
  3. ADMM-IDD (iterative turbo loop)

System: Nr x Nt BPSK MIMO, LDPC(n_ldpc, rate).
Multi-channel-use: one codeword spans T = n_ldpc // Nt MIMO slots.
Using n_ldpc=648 (IEEE 802.11n short block), Nt=4, Nr=4 -> T=162 slots.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.idd_receiver import simulate_idd_ber

# ── System parameters ──────────────────────────────────────────────────────────
Nr        = 4           # receive antennas
Nt        = 4           # transmit antennas
N_LDPC    = 648         # LDPC codeword length  (648 // 4 = 162 slots/codeword)
RATE      = 0.5         # LDPC code rate  -> k=324 info bits, m=324 check bits
RHO       = 1.0         # ADMM penalty parameter
ADMM_ITER = 20          # ADMM inner iterations
BP_ITER   = 20          # Belief-propagation decoder iterations
IDD_ITER  = 4           # outer turbo IDD iterations
N_TRIALS  = 10          # Monte Carlo codewords per SNR point
SEED      = 42

SNR_DB = np.arange(0, 13, 2, dtype=float)   # 0, 2, 4, 6, 8, 10, 12 dB

print("=" * 70)
print(f"  ADMM-IDD vs MMSE+SPA  |  {Nr}x{Nt} BPSK MIMO")
print(f"  LDPC({N_LDPC}, R={RATE})  ->  {N_LDPC//Nt} slots/codeword")
print(f"  rho={RHO}  |  ADMM_iter={ADMM_ITER}  |  BP_iter={BP_ITER}  |  IDD_iter={IDD_ITER}")
print(f"  {N_TRIALS} codewords/SNR")
print("=" * 70)

results = simulate_idd_ber(
    Nr=Nr, Nt=Nt,
    n_ldpc=N_LDPC, rate=RATE,
    snr_db_range=SNR_DB,
    rho=RHO,
    admm_iter=ADMM_ITER,
    bp_iter=BP_ITER,
    idd_iter=IDD_ITER,
    n_trials=N_TRIALS,
    seed=SEED,
)

print()
print("-" * 70)
print(f"{'SNR':>6} | {'MMSE+SPA':>10} | {'ADMM+SPA':>10} | {'ADMM-IDD':>10} | {'vs MMSE':>9}")
print("-" * 70)
for i, snr in enumerate(results['snr']):
    m  = results['ber_mmse_spa'][i]
    a  = results['ber_admm_spa'][i]
    d  = results['ber_idd'][i]
    gain_vs_mmse = m - d
    gain_vs_admm = a - d
    if d < m and d < a:
        verdict = " *** IDD BEST ***"
    elif d < m:
        verdict = " <-- IDD beats MMSE"
    elif m < d:
        verdict = " <-- MMSE beats IDD"
    else:
        verdict = ""
    print(f"{snr:>5.0f}  | {m:>10.5f} | {a:>10.5f} | {d:>10.5f} | {gain_vs_mmse:>+9.5f}{verdict}")
print("-" * 70)
