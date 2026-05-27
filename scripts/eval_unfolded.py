

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

from src.channel import generate_channel, awgn_channel
from src.admm_detector import admm_detect
from src.deep_unfolding import (
    ADMMUnfoldedNumpy, generate_training_batch,
    TORCH_AVAILABLE
)

if TORCH_AVAILABLE:
    from src.deep_unfolding import (
        ADMMUnfoldedNet, train_admm_unfolded, evaluate_unfolded_ber
    )
    import torch

Nr, Nt       = 4, 4
K            = 10
N_TRIALS     = 1000
SNR_RANGE    = np.arange(0, 16, 2)
LLR_CLIP     = 30.0
RHO_MAX      = 50.0

SNR_DB_TRAIN = 10.0
N_EPOCHS     = 1000
BATCH_SIZE   = 128
LR           = 5e-3
SEED         = 42

rng = np.random.default_rng(SEED)
SEP = "=" * 70

# =============================================================================
# HELPER FUNCTIONS
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

    A_inv  = np.linalg.inv(A)
    s2_eff = np.maximum(np.real(np.diag(A_inv)), 1e-9)
    llr    = 2.0 * x_hat / s2_eff
    return np.clip(llr, -LLR_CLIP, LLR_CLIP)

def get_rho(sigma2):
    """
    Computes the adaptive penalty parameter rho based on noise variance.
    
    Args:
        sigma2 (float): Noise variance.
        
    Returns:
        float: Computed rho value.
    """

    return min(1.0 / max(sigma2, 1e-6), RHO_MAX)

# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================
def eval_mmse_ber(snr_db_range, n_trials, rng_seed=99):
    """
    Evaluates Bit Error Rate (BER) for the MMSE detector over a range of SNRs.
    
    Args:
        snr_db_range (list or np.ndarray): SNRs to evaluate (in dB).
        n_trials (int): Number of channel realizations per SNR point.
        rng_seed (int): Random seed for channel generation.
        
    Returns:
        np.ndarray: BER values corresponding to each SNR.
    """

    rng_e = np.random.default_rng(rng_seed)
    ber = np.zeros(len(snr_db_range))
    for i, snr_db in enumerate(snr_db_range):
        errs, total = 0, 0
        for _ in range(n_trials):
            bits = rng_e.integers(0, 2, size=Nt)
            x = 1.0 - 2.0 * bits.astype(float)
            H_t = generate_channel(Nr, Nt)
            y_t, s2_t = awgn_channel(H_t, x, snr_db, rng_e)
            llr = compute_mmse_llr(y_t, H_t, s2_t)
            b_hat = (llr < 0).astype(int)
            errs  += int(np.sum(bits != b_hat))
            total += Nt
        ber[i] = errs / total
    return ber

def eval_admm_fixed_ber(snr_db_range, n_trials, rho_fixed=1.0, rng_seed=99):
    """
    Evaluates BER for the ADMM detector using a fixed penalty parameter rho.
    
    Args:
        snr_db_range (list or np.ndarray): SNRs to evaluate (in dB).
        n_trials (int): Number of channel realizations per SNR point.
        rho_fixed (float): Fixed ADMM penalty parameter.
        rng_seed (int): Random seed for channel generation.
        
    Returns:
        np.ndarray: BER values corresponding to each SNR.
    """

    rng_e = np.random.default_rng(rng_seed)
    ber = np.zeros(len(snr_db_range))
    for i, snr_db in enumerate(snr_db_range):
        errs, total = 0, 0
        for _ in range(n_trials):
            bits = rng_e.integers(0, 2, size=Nt)
            x = 1.0 - 2.0 * bits.astype(float)
            H_t = generate_channel(Nr, Nt)
            y_t, s2_t = awgn_channel(H_t, x, snr_db, rng_e)
            z_hard, _ = admm_detect(y_t, H_t, s2_t, M=2,
                                    rho=rho_fixed, max_iter=K,
                                    return_soft=False)
            b_hat = (np.real(z_hard) < 0).astype(int)
            errs  += int(np.sum(bits != b_hat))
            total += Nt
        ber[i] = errs / total
    return ber

def eval_admm_adaptive_ber(snr_db_range, n_trials, rng_seed=99):
    """
    Evaluates BER for the ADMM detector using adaptive rho based on noise variance.
    
    Args:
        snr_db_range (list or np.ndarray): SNRs to evaluate (in dB).
        n_trials (int): Number of channel realizations per SNR point.
        rng_seed (int): Random seed for channel generation.
        
    Returns:
        np.ndarray: BER values corresponding to each SNR.
    """

    rng_e = np.random.default_rng(rng_seed)
    ber = np.zeros(len(snr_db_range))
    for i, snr_db in enumerate(snr_db_range):
        errs, total = 0, 0
        for _ in range(n_trials):
            bits = rng_e.integers(0, 2, size=Nt)
            x = 1.0 - 2.0 * bits.astype(float)
            H_t = generate_channel(Nr, Nt)
            y_t, s2_t = awgn_channel(H_t, x, snr_db, rng_e)
            rho_t = get_rho(s2_t)
            z_hard, _ = admm_detect(y_t, H_t, s2_t, M=2,
                                    rho=rho_t, max_iter=K,
                                    return_soft=False)
            b_hat = (np.real(z_hard) < 0).astype(int)
            errs  += int(np.sum(bits != b_hat))
            total += Nt
        ber[i] = errs / total
    return ber

# =============================================================================
# PHASE 0: ISOLATION TESTS
# =============================================================================
print("\n" + SEP)
print("PHASE 0: Isolation tests")
print(SEP)

print(f"\n  Test 1: {Nr}x{Nt} MIMO MMSE uncoded (BER must decrease)")
ber_prev = 1.0
test_ok = True
for snr_db in [0, 4, 8, 12]:
    errs, total = 0, 0
    for _ in range(2000):
        bits = rng.integers(0, 2, size=Nt)
        x = 1.0 - 2.0 * bits.astype(float)
        H_t = generate_channel(Nr, Nt)
        y_t, s2_t = awgn_channel(H_t, x, snr_db, rng)
        llr = compute_mmse_llr(y_t, H_t, s2_t)
        b_hat = (llr < 0).astype(int)
        errs  += int(np.sum(bits != b_hat))
        total += Nt
    ber = errs / total
    mono = ber <= ber_prev + 0.005
    test_ok &= mono
    print(f"    SNR={snr_db:2d} dB: BER={ber:.4f} {'OK' if mono else 'NOT MONOTONIC'}")
    ber_prev = ber
print(f"  Test 1: {'PASSED' if test_ok else 'FAILED'}")

print(f"\n  Test 2: {Nr}x{Nt} ADMM fixed ρ=1.0 uncoded (BER must decrease)")
ber_prev = 1.0
test_ok = True
for snr_db in [0, 4, 8, 12]:
    errs, total = 0, 0
    for _ in range(2000):
        bits = rng.integers(0, 2, size=Nt)
        x = 1.0 - 2.0 * bits.astype(float)
        H_t = generate_channel(Nr, Nt)
        y_t, s2_t = awgn_channel(H_t, x, snr_db, rng)
        z_hard, _ = admm_detect(y_t, H_t, s2_t, M=2, rho=1.0,
                                max_iter=K, return_soft=False)
        b_hat = (np.real(z_hard) < 0).astype(int)
        errs  += int(np.sum(bits != b_hat))
        total += Nt
    ber = errs / total
    mono = ber <= ber_prev + 0.005
    test_ok &= mono
    print(f"    SNR={snr_db:2d} dB: BER={ber:.4f} {'OK' if mono else 'NOT MONOTONIC'}")
    ber_prev = ber
print(f"  Test 2: {'PASSED' if test_ok else 'FAILED'}")

print(f"\n  Test 3: ADMMUnfoldedNumpy forward pass sanity")
model_np = ADMMUnfoldedNumpy(K=K, rho_init=1.0)
H_test = generate_channel(Nr, Nt)
bits_test = rng.integers(0, 2, size=Nt)
x_test = 1.0 - 2.0 * bits_test.astype(float)
y_test, s2_test = awgn_channel(H_test, x_test, 10.0, rng)
z_test = model_np.forward(y_test, H_test, s2_test, M=2)
in_bpsk = np.all(np.isin(z_test, [-1.0, 1.0]))
print(f"    z output: {z_test}")
print(f"    z ∈ {{-1,+1}}: {in_bpsk}")
print(f"    x_true:  {x_test}")
match = np.array_equal(z_test, x_test)
print(f"    Correct: {match} ({np.sum(z_test == x_test)}/{Nt} symbols)")
print(f"  Test 3: {'PASSED' if in_bpsk else 'FAILED'}")

if TORCH_AVAILABLE:

    print("\n" + SEP)
    print("PyTorch detected — running full training + evaluation")
    print(SEP)

    # =============================================================================
    # DEEP UNFOLDING TRAINING (PYTORCH)
    # =============================================================================

    print(f"\n  Training at fixed SNR = {SNR_DB_TRAIN} dB ...")
    print(f"    K={K} layers, {N_EPOCHS} epochs, batch_size={BATCH_SIZE}, lr={LR}")
    t_train_start = time.time()

    model_fixed_snr, loss_hist_fixed = train_admm_unfolded(
        Nr=Nr, Nt=Nt, K=K,
        snr_db_train=SNR_DB_TRAIN,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        seed=SEED,
        verbose=True
    )
    t_fixed = time.time() - t_train_start
    print(f"  Fixed-SNR training done in {t_fixed:.1f}s")

    rho_fixed_snr = model_fixed_snr.get_rho_values()
    print(f"  Learned ρ schedule (fixed SNR): {np.array2string(rho_fixed_snr, precision=3)}")

    print(f"\n  Training with mixed SNR ∈ [2, 14] dB (random per batch) ...")
    t_mixed_start = time.time()

    torch.manual_seed(SEED + 100)
    rng_mixed = np.random.default_rng(SEED + 100)

    model_mixed_snr = ADMMUnfoldedNet(K=K, rho_init=1.0).double()
    optimiser = torch.optim.Adam(model_mixed_snr.parameters(), lr=LR)
    loss_fn   = torch.nn.MSELoss()
    loss_hist_mixed = []

    for epoch in range(N_EPOCHS):

        snr_batch = rng_mixed.uniform(2.0, 14.0)
        y_np, H_np, x_np, sigma2 = generate_training_batch(
            BATCH_SIZE, Nr, Nt, snr_batch, rng_mixed
        )

        y_t = torch.tensor(y_np, dtype=torch.complex128)
        H_t = torch.tensor(H_np, dtype=torch.complex128)
        x_t = torch.tensor(x_np, dtype=torch.float64)

        optimiser.zero_grad()
        z_out = model_mixed_snr(y_t, H_t, sigma2)
        loss  = loss_fn(z_out, x_t)
        loss.backward()
        optimiser.step()
        loss_hist_mixed.append(float(loss.item()))

        if (epoch + 1) % 200 == 0:
            rho_vals = model_mixed_snr.get_rho_values()
            print(f"    Epoch {epoch+1:4d}/{N_EPOCHS} | "
                  f"Loss={loss.item():.6f} | SNR={snr_batch:.1f} dB | "
                  f"ρ: [{', '.join(f'{r:.3f}' for r in rho_vals[:3])}...]")

    t_mixed = time.time() - t_mixed_start
    rho_mixed_snr = model_mixed_snr.get_rho_values()
    print(f"  Mixed-SNR training done in {t_mixed:.1f}s")
    print(f"  Learned ρ schedule (mixed SNR): {np.array2string(rho_mixed_snr, precision=3)}")

    model_eval = model_mixed_snr
    loss_hist  = loss_hist_mixed
    rho_learned = rho_mixed_snr

    # =============================================================================
    # PHASE 1: BER EVALUATION (PYTORCH)
    # =============================================================================
    print("\n" + SEP)
    print("PHASE 1: BER Evaluation — 4 methods")
    print(SEP)

    t_eval_start = time.time()

    EVAL_SEED = 99

    print("\n  Evaluating MMSE ...")
    ber_mmse = eval_mmse_ber(SNR_RANGE, N_TRIALS, rng_seed=EVAL_SEED)

    print("  Evaluating ADMM (fixed ρ=1.0) ...")
    ber_admm_fixed = eval_admm_fixed_ber(SNR_RANGE, N_TRIALS,
                                          rho_fixed=1.0, rng_seed=EVAL_SEED)

    print("  Evaluating ADMM (adaptive ρ=1/σ²) ...")
    ber_admm_adapt = eval_admm_adaptive_ber(SNR_RANGE, N_TRIALS,
                                             rng_seed=EVAL_SEED)

    print("  Evaluating Deep-Unfolded ADMM ...")
    model_eval.eval()
    rng_eval = np.random.default_rng(EVAL_SEED)
    ber_unfolded = np.zeros(len(SNR_RANGE))

    with torch.no_grad():
        for i, snr_db in enumerate(SNR_RANGE):
            y_np, H_np, x_np, sigma2 = generate_training_batch(
                N_TRIALS, Nr, Nt, snr_db, rng_eval
            )
            y_t = torch.tensor(y_np, dtype=torch.complex128)
            H_t = torch.tensor(H_np, dtype=torch.complex128)
            z_out = model_eval(y_t, H_t, sigma2).numpy()

            bits_tx  = ((1 - x_np) / 2).astype(int)
            bits_hat = (z_out < 0).astype(int)
            ber_unfolded[i] = float(np.mean(bits_tx != bits_hat))

    t_eval = time.time() - t_eval_start
    print(f"\n  Evaluation done in {t_eval:.1f}s")

    print(f"\n  {'SNR':>5} | {'MMSE':>10} | {'ADMM-fix':>10} | "
          f"{'ADMM-adapt':>10} | {'Unfolded':>10}")
    print("  " + "-" * 58)
    for j, snr_db in enumerate(SNR_RANGE):
        print(f"  {snr_db:4.0f}dB | {ber_mmse[j]:>10.5f} | "
              f"{ber_admm_fixed[j]:>10.5f} | "
              f"{ber_admm_adapt[j]:>10.5f} | "
              f"{ber_unfolded[j]:>10.5f}")

    def interpolate_snr_at_ber(snr_arr, ber_arr, target_ber):

        ber_arr = np.maximum(ber_arr, 1e-10)
        log_ber = np.log10(ber_arr)
        target_log = np.log10(target_ber)

        for j in range(len(log_ber) - 1):
            if (log_ber[j] >= target_log >= log_ber[j+1]) or               (log_ber[j] <= target_log <= log_ber[j+1]):

                frac = (target_log - log_ber[j]) / (log_ber[j+1] - log_ber[j] + 1e-15)
                return snr_arr[j] + frac * (snr_arr[j+1] - snr_arr[j])
        return np.nan

    ber_targets = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]
    snr_gain_unfolded = []
    snr_gain_labels   = []

    print("\n  SNR gain of Deep-Unfolded over fixed-ρ ADMM:")
    for target in ber_targets:
        snr_fix = interpolate_snr_at_ber(SNR_RANGE, ber_admm_fixed, target)
        snr_unf = interpolate_snr_at_ber(SNR_RANGE, ber_unfolded, target)
        if not (np.isnan(snr_fix) or np.isnan(snr_unf)):
            gain = snr_fix - snr_unf
            snr_gain_unfolded.append(gain)
            snr_gain_labels.append(f"BER={target:.0e}")
            print(f"    At BER={target:.0e}: fixed needs {snr_fix:.1f} dB, "
                  f"unfolded needs {snr_unf:.1f} dB → gain = {gain:.2f} dB")
        else:
            print(f"    At BER={target:.0e}: not reachable by both methods in SNR range")

    # =============================================================================
    # PHASE 2: PLOTTING (PYTORCH)
    # =============================================================================
    print("\n" + SEP)
    print("PHASE 2: Generating diagnostic plots")
    print(SEP)

    os.makedirs("data", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    eps = 1e-6

    ax1 = axes[0, 0]
    epochs_arr = np.arange(1, len(loss_hist) + 1)
    ax1.semilogy(epochs_arr, loss_hist, 'b-', lw=1, alpha=0.5, label='Raw loss')

    if len(loss_hist) > 50:
        window = 50
        smoothed = np.convolve(loss_hist, np.ones(window)/window, mode='valid')
        ax1.semilogy(np.arange(window, len(loss_hist)+1), smoothed,
                     'r-', lw=2, label=f'Smoothed (win={window})')
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("MSE Loss", fontsize=12)
    ax1.set_title("Training Loss (mixed-SNR training)", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, which='both', alpha=0.4)

    ax2 = axes[0, 1]
    layers = np.arange(1, K + 1)
    bar_width = 0.35
    ax2.bar(layers - bar_width/2, rho_learned, bar_width,
            color='steelblue', alpha=0.8, label='Learned ρ_k (mixed SNR)')
    ax2.bar(layers + bar_width/2, rho_fixed_snr, bar_width,
            color='darkorange', alpha=0.8,
            label=f'Learned ρ_k (fixed SNR={SNR_DB_TRAIN}dB)')

    for ref_snr in [4, 10]:
        s2_ref = Nt / (Nr * 10**(ref_snr / 10.0))
        rho_ref = get_rho(s2_ref)
        ax2.axhline(rho_ref, ls='--', lw=1.5, alpha=0.7,
                    label=f'Adaptive ρ at {ref_snr}dB = {rho_ref:.2f}')
    ax2.set_xlabel("Layer k", fontsize=12)
    ax2.set_ylabel("ρ_k", fontsize=12)
    ax2.set_title("Learned ρ Schedule", fontsize=13)
    ax2.set_xticks(layers)
    ax2.legend(fontsize=8, loc='best')
    ax2.grid(True, alpha=0.4, axis='y')

    ax3 = axes[1, 0]
    ax3.semilogy(SNR_RANGE, np.maximum(ber_mmse, eps),
                 'k-o', lw=2, ms=6, label='MMSE')
    ax3.semilogy(SNR_RANGE, np.maximum(ber_admm_fixed, eps),
                 'b--s', lw=2, ms=6, label='ADMM (fixed ρ=1)')
    ax3.semilogy(SNR_RANGE, np.maximum(ber_admm_adapt, eps),
                 'g--^', lw=2, ms=6, label='ADMM (adaptive ρ=1/σ²)')
    ax3.semilogy(SNR_RANGE, np.maximum(ber_unfolded, eps),
                 'r-D', lw=2.5, ms=8, label='Deep-Unfolded ADMM')
    ax3.set_xlabel("SNR (dB)", fontsize=12)
    ax3.set_ylabel("BER", fontsize=12)
    ax3.set_title(f"BER vs SNR — {Nr}×{Nt} BPSK, K={K} layers", fontsize=13)
    ax3.legend(fontsize=9, loc='lower left')
    ax3.grid(True, which='both', alpha=0.4)
    ax3.set_ylim([eps, 0.5])

    ax4 = axes[1, 1]
    if snr_gain_unfolded:
        x_pos = np.arange(len(snr_gain_unfolded))
        bars = ax4.bar(x_pos, snr_gain_unfolded, color='steelblue', alpha=0.8)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(snr_gain_labels, fontsize=9, rotation=15)
        ax4.set_ylabel("SNR Gain (dB)", fontsize=12)
        ax4.set_title("SNR gain: Unfolded over fixed-ρ ADMM", fontsize=13)
        for bar, val in zip(bars, snr_gain_unfolded):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f"{val:.2f}", ha='center', va='bottom', fontsize=10)
        ax4.axhline(0, color='black', lw=0.8)
        ax4.grid(True, alpha=0.4, axis='y')
    else:
        ax4.text(0.5, 0.5, "No overlapping BER range\nfor gain computation",
                 ha='center', va='center', fontsize=14, transform=ax4.transAxes)

    plt.suptitle(
        f"Day 13 v2: Deep-Unfolded ADMM — {Nr}×{Nt} BPSK, K={K} layers, "
        f"{N_TRIALS} trials/SNR",
        fontsize=14, y=1.01
    )
    plt.tight_layout()
    plt.savefig("data/day13_v2_diagnostic.png", dpi=150, bbox_inches='tight')
    print("\nPlot saved to data/day13_v2_diagnostic.png")

else:

    print("\n" + SEP)
    print("PyTorch NOT available — running NumPy fallback demo")
    print(SEP)

    # =============================================================================
    # NUMPY FALLBACK: DEMO AND OPTIMIZATION
    # =============================================================================

    print("\n  Numerical gradient demo:")
    model_np = ADMMUnfoldedNumpy(K=K, rho_init=1.0)

    rng_grad = np.random.default_rng(SEED)
    H_g = generate_channel(Nr, Nt)
    bits_g = rng_grad.integers(0, 2, size=Nt)
    x_g = 1.0 - 2.0 * bits_g.astype(float)
    y_g, s2_g = awgn_channel(H_g, x_g, 10.0, rng_grad)

    loss_before = model_np.loss_mse(y_g, H_g, s2_g, x_g)
    grad = model_np.numerical_gradient(y_g, H_g, s2_g, x_g)
    print(f"    Initial ρ = {model_np.rho}")
    print(f"    MSE loss  = {loss_before:.6f}")
    print(f"    Gradient  = {np.array2string(grad, precision=5)}")

    lr_np = 0.1
    n_gd_steps = 50
    loss_trace_np = [loss_before]
    for step in range(n_gd_steps):
        grad = model_np.numerical_gradient(y_g, H_g, s2_g, x_g)
        model_np.rho -= lr_np * grad
        model_np.rho = np.maximum(model_np.rho, 0.01)
        loss_val = model_np.loss_mse(y_g, H_g, s2_g, x_g)
        loss_trace_np.append(loss_val)
    print(f"    After {n_gd_steps} GD steps:")
    print(f"      ρ = {np.array2string(model_np.rho, precision=3)}")
    print(f"      MSE = {loss_trace_np[-1]:.6f}")

    print("\n  MSE vs ρ sweep (uniform ρ for all layers):")
    rho_sweep = np.logspace(-1, 1, 30)
    mse_sweep = np.zeros(len(rho_sweep))

    rng_sw = np.random.default_rng(SEED + 1)
    n_sweep_trials = 200
    for ri, rho_val in enumerate(rho_sweep):
        model_sw = ADMMUnfoldedNumpy(K=K, rho_init=rho_val)
        mse_accum = 0.0
        for _ in range(n_sweep_trials):
            bits_s = rng_sw.integers(0, 2, size=Nt)
            x_s = 1.0 - 2.0 * bits_s.astype(float)
            H_s = generate_channel(Nr, Nt)
            y_s, s2_s = awgn_channel(H_s, x_s, 10.0, rng_sw)
            mse_accum += model_sw.loss_mse(y_s, H_s, s2_s, x_s)
        mse_sweep[ri] = mse_accum / n_sweep_trials

    best_rho_idx = np.argmin(mse_sweep)
    print(f"    Best ρ = {rho_sweep[best_rho_idx]:.3f} "
          f"(MSE = {mse_sweep[best_rho_idx]:.6f})")

    print("\n  BER evaluation:")
    EVAL_SEED = 99

    print("    Evaluating MMSE ...")
    ber_mmse = eval_mmse_ber(SNR_RANGE, N_TRIALS, rng_seed=EVAL_SEED)

    print("    Evaluating ADMM (fixed ρ=1.0) ...")
    ber_admm_fixed = eval_admm_fixed_ber(SNR_RANGE, N_TRIALS,
                                          rho_fixed=1.0, rng_seed=EVAL_SEED)

    print("    Evaluating ADMM (adaptive ρ) ...")
    ber_admm_adapt = eval_admm_adaptive_ber(SNR_RANGE, N_TRIALS,
                                             rng_seed=EVAL_SEED)

    print("    Evaluating NumPy Unfolded (best fixed ρ from sweep) ...")
    model_np_best = ADMMUnfoldedNumpy(K=K, rho_init=rho_sweep[best_rho_idx])
    rng_npu = np.random.default_rng(EVAL_SEED)
    ber_np_unfolded = np.zeros(len(SNR_RANGE))
    for i, snr_db in enumerate(SNR_RANGE):
        errs, total = 0, 0
        for _ in range(N_TRIALS):
            bits = rng_npu.integers(0, 2, size=Nt)
            x = 1.0 - 2.0 * bits.astype(float)
            H_t = generate_channel(Nr, Nt)
            y_t, s2_t = awgn_channel(H_t, x, snr_db, rng_npu)
            z_out = model_np_best.forward(y_t, H_t, s2_t, M=2)
            b_hat = (z_out < 0).astype(int)
            errs  += int(np.sum(bits != b_hat))
            total += Nt
        ber_np_unfolded[i] = errs / total

    print(f"\n  {'SNR':>5} | {'MMSE':>10} | {'ADMM-fix':>10} | "
          f"{'ADMM-adapt':>10} | {'NP-unfold':>10}")
    print("  " + "-" * 58)
    for j, snr_db in enumerate(SNR_RANGE):
        print(f"  {snr_db:4.0f}dB | {ber_mmse[j]:>10.5f} | "
              f"{ber_admm_fixed[j]:>10.5f} | "
              f"{ber_admm_adapt[j]:>10.5f} | "
              f"{ber_np_unfolded[j]:>10.5f}")

    ber_unfolded = ber_np_unfolded
    loss_hist = loss_trace_np
    rho_learned = model_np.rho

    # =============================================================================
    # PHASE 2: PLOTTING (NUMPY)
    # =============================================================================
    print("\n" + SEP)
    print("PHASE 2: Generating diagnostic plots (NumPy fallback)")
    print(SEP)

    os.makedirs("data", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    eps = 1e-6

    ax1 = axes[0, 0]
    ax1.semilogx(rho_sweep, mse_sweep, 'b-o', lw=2, ms=4)
    ax1.axvline(rho_sweep[best_rho_idx], color='r', ls='--', lw=1.5,
                label=f'Best ρ={rho_sweep[best_rho_idx]:.3f}')
    ax1.set_xlabel("ρ (uniform for all layers)", fontsize=12)
    ax1.set_ylabel("Mean MSE", fontsize=12)
    ax1.set_title("MSE vs ρ (single-layer sweep, SNR=10dB)", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.4)

    ax2 = axes[0, 1]
    layers = np.arange(1, K + 1)
    ax2.bar(layers, rho_learned, color='steelblue', alpha=0.8,
            label='GD-optimised ρ_k')
    s2_ref = Nt / (Nr * 10**(10.0/10.0))
    rho_ref = get_rho(s2_ref)
    ax2.axhline(rho_ref, ls='--', color='red', lw=1.5,
                label=f'Adaptive ρ at 10dB = {rho_ref:.2f}')
    ax2.set_xlabel("Layer k", fontsize=12)
    ax2.set_ylabel("ρ_k", fontsize=12)
    ax2.set_title("ρ schedule after numerical GD (single sample)", fontsize=13)
    ax2.set_xticks(layers)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.4, axis='y')

    ax3 = axes[1, 0]
    ax3.semilogy(SNR_RANGE, np.maximum(ber_mmse, eps),
                 'k-o', lw=2, ms=6, label='MMSE')
    ax3.semilogy(SNR_RANGE, np.maximum(ber_admm_fixed, eps),
                 'b--s', lw=2, ms=6, label='ADMM (fixed ρ=1)')
    ax3.semilogy(SNR_RANGE, np.maximum(ber_admm_adapt, eps),
                 'g--^', lw=2, ms=6, label='ADMM (adaptive ρ=1/σ²)')
    ax3.semilogy(SNR_RANGE, np.maximum(ber_np_unfolded, eps),
                 'r-D', lw=2.5, ms=8, label='NumPy Unfolded (best ρ)')
    ax3.set_xlabel("SNR (dB)", fontsize=12)
    ax3.set_ylabel("BER", fontsize=12)
    ax3.set_title(f"BER vs SNR — {Nr}×{Nt} BPSK, K={K}", fontsize=13)
    ax3.legend(fontsize=9, loc='lower left')
    ax3.grid(True, which='both', alpha=0.4)
    ax3.set_ylim([eps, 0.5])

    ax4 = axes[1, 1]
    ax4.plot(range(len(loss_trace_np)), loss_trace_np, 'b-o', lw=2, ms=3)
    ax4.set_xlabel("GD Step", fontsize=12)
    ax4.set_ylabel("MSE Loss", fontsize=12)
    ax4.set_title("Numerical GD loss (single sample)", fontsize=13)
    ax4.grid(True, alpha=0.4)

    plt.suptitle(
        f"Day 13 v2 (NumPy fallback): {Nr}×{Nt} BPSK, K={K} layers, "
        f"{N_TRIALS} trials/SNR",
        fontsize=14, y=1.01
    )
    plt.tight_layout()
    plt.savefig("data/day13_v2_diagnostic.png", dpi=150, bbox_inches='tight')
    print("\nPlot saved to data/day13_v2_diagnostic.png")

# =============================================================================
# SUMMARY & VERIFICATION
# =============================================================================
print("\n" + SEP)
print("SUMMARY: BER vs SNR")
print(SEP)

print(f"  {'SNR':>5} | {'MMSE':>10} | {'ADMM-fix':>10} | "
      f"{'ADMM-adapt':>10} | {'Unfolded':>10}")
print("  " + "-" * 58)
for j, snr_db in enumerate(SNR_RANGE):
    print(f"  {snr_db:4.0f}dB | {ber_mmse[j]:>10.5f} | "
          f"{ber_admm_fixed[j]:>10.5f} | "
          f"{ber_admm_adapt[j]:>10.5f} | "
          f"{ber_unfolded[j]:>10.5f}")

print("\n  Monotonicity check:")
for name, arr in [("MMSE", ber_mmse),
                  ("ADMM-fixed", ber_admm_fixed),
                  ("ADMM-adapt", ber_admm_adapt),
                  ("Unfolded", ber_unfolded)]:
    violations = sum(1 for i in range(1, len(arr))
                     if arr[i] > arr[i-1] + 1e-4)
    status = 'PASS (monotonic)' if violations == 0             else f'FAIL ({violations} violations)'
    print(f"    {name:>12}: {status}")

print(f"\n  Learned ρ schedule (K={K} layers):")
print(f"    {np.array2string(rho_learned, precision=3)}")

print(SEP)
total_time = time.time() - time.time()
print(f"\nDay 13 v2 complete.")

