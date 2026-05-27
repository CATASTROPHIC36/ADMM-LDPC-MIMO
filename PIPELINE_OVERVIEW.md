# ADMM MIMO Receiver Project — Complete Pipeline Overview

## 📋 Project Summary
This project implements an **Iterative Detection and Decoding (IDD)** receiver for MIMO channels using the **ADMM algorithm** (Alternating Direction Method of Multipliers). It progressively builds from simple channel simulation through classical signal processing to learning-based deep unfolded architectures.

---

## 🔧 Core System Architecture

### System Model
```
TRANSMIT SIDE:
  Bit stream u (k bits)
      ↓ LDPC encode
  Codeword c (n bits)
      ↓ BPSK modulate (or QAM)
  Symbol vector x ∈ {±1}^Nt (or QAM constellation)

CHANNEL:
  y = H * x + n
  where:
    H  ∈ C^(Nr × Nt)  — flat-fading MIMO channel matrix
    n  ∈ C^Nr        — AWGN noise, n ~ CN(0, σ²I)
    y  ∈ C^Nr        — received signal

RECEIVER SIDE:
  Received y
      ↓ [Detector: ADMM/MMSE/EP/AMP]
  Soft symbol estimates (or LLRs)
      ↓ [Channel decoder: SPA]
  Decoded bits û
```

---

## 📦 Module Architecture

```
src/
├── channel.py           — MIMO channel simulator (DAY 1)
│   ├── generate_channel()
│   ├── bpsk_modulate/demodulate_hard()
│   ├── awgn_channel()
│   └── mmse_detect()     [baseline linear detector]
│
├── modulation.py        — QAM modulation + LLRs (DAY 2)
│   ├── qam_modulate()
│   ├── compute_llrs_mmse()
│   └── compute_llrs_qam()
│
├── ldpc.py              — LDPC codec (DAY 3-4)
│   ├── make_ldpc_matrix()
│   ├── ldpc_encode()
│   ├── spa_decode_soft()  [belief propagation]
│   └── verify_codeword()
│
├── admm_detector.py     — ADMM detector (DAY 5)
│   ├── admm_detect()
│   ├── admm_llrs()
│   └── admm_effective_sigma2()
│
├── idd_receiver.py      — IDD loop manager (DAY 6)
│   ├── simulate_idd_ber()
│   └── clip_llrs()
│
└── deep_unfolding.py    — Neural network unrolling (DAY 7)
    ├── ADMMUnfoldedNumpy
    ├── ADMMUnfoldedNet()  [PyTorch]
    └── train_admm_unfolded()
```

---

## 🔄 Data Pipeline & Signal Flow

### Phase 1: Transmission & Channel
```
1. LDPC Encoding
   u_msg (32 bits) → [LDPC Encode] → c (64 bits, systematic)

2. Modulation
   c → [BPSK Map: 0→+1, 1→-1] → x ∈ {-1,+1}^4

3. Channel + Noise
   x → [MIMO Channel H] → Hx (4×4 channel mix)
           ↓
   Hx + n → y (received, SNR-dependent)
```
 
### Phase 2: Detection (Classical ADMM-IDD)
```
OUTER LOOP: Iterative Detection-Decoding (K_idd = 1..4 iterations)
│
└─→ Iteration t:
    
    A) Detector (with decoder prior)
       Input:  y, H, σ², prior LLR from previous SPA iteration
       Output: Detector LLR
       
       ADMM x-update:  (H^H H/σ² + ρI) x = H^H y/σ² + ρ(z-u)
       ADMM z-update:  z = sign(x + u)                [hard projection]
       ADMM u-update:  u = u + x - z
       
       LLR = 2*x_final / σ²_eff_diagonal
    
    B) Extrinsic Subtraction
       LLR_ext = LLR_detector - LLR_prior
       
    C) SPA Decoder (K_bp = 20-30 iterations)
       Input:  LLR_ext
       Process: Tanner graph message passing on LDPC factor graph
       Output: Decoded bits û, posterior LLR_post
    
    D) Feedback
       LLR_prior = LLR_post - LLR_ext  [extrinsic principle]
       
    E) Check Convergence
       if H @ c_hat = 0 (mod 2): BREAK  [valid codeword found]
```

### Phase 3: Alternative Detectors
```
┌─ MMSE Linear Filter (baseline)
│  x̂ = (H^H H + σ² I)^{-1} H^H y
│  
├─ ADMM (fixed ρ = 1/σ²)
│  Convex relaxation via splitting: min_x ||y-Hx||² s.t. z∈A
│  (this project's primary focus)
│  
├─ EP (Expectation Propagation)
│  Iterative Gaussian moment matching
│  Refines posterior covariance per stream
│  
└─ AMP (Approximate Message Passing)
   Iterative algorithm with provable convergence on random matrices
```

---

## 📊 Script Execution Pipeline

### Validation Tier (Scripts: run_day1-7, debug_pipeline)

**run_day1.py** — Channel Simulator Validation
```
Tests: BPSK modulation, channel matrix generation, noise scaling
Output: BER curve (MMSE baseline)
Passes if: BER decreases monotonically with SNR
```

**run_day2.py** — QAM Modulation & LLRs
```
Tests: 16-QAM constellation mapping, Gray coding, LLR computation
Output: LLR statistics, constellation diagram
Passes if: LLR increases with SNR, no outliers
```

**run_day3_day4.py** — LDPC Codec Verification
```
Tests: LDPC matrix generation, systematic encoding, BP decoding
Output: Frame error rate, convergence plots
Passes if: Codewords verify, decoder converges
```

**run_day5.py** — ADMM Convergence Study
```
Tests: ADMM detector on uncoded MIMO
Output: Symbol error rate vs ADMM iterations
Passes if: SER plateaus (convergence verified)
```

**debug_pipeline.py** — 5-Step Isolation
```
Step 1: SISO BPSK AWGN (must match Q(√(2·SNR)) theory)
Step 2: SISO LDPC coded (validates encoder/decoder)
Step 3: Uncoded 4×4 MIMO (validates ADMM)
Step 4: One-shot MIMO+LDPC (validates LLR chain)
Step 5: ADMM-IDD full loop (validates prior exchange)
```

### Performance Tier (Scripts: run_day6, run_day9_v2, run_day10-11)

**run_day6.py** — Classical IDD Receiver
```
Compares:
  - MMSE + SPA (no ADMM)
  - ADMM + SPA (one-shot, no IDD loop)
  - ADMM-IDD (with iterative feedback)
Output: BER vs SNR (3 curves)
Expected: ADMM-IDD gains ~1 dB over MMSE+SPA at high SNR
```

**run_day9_v2.py** — ADMM-IDD vs EP-IDD
```
Compares:
  - ADMM-IDD (fixed ρ)
  - EP-IDD (learned covariance per stream)
  - MMSE+SPA baseline
Parameters: 1500 trials/SNR, N_LDPC=64, K_idd=4
Output: 4-panel diagnostic:
  Panel 1: BER curve comparison
  Panel 2: BER before/after decoder
  Panel 3: Mean |LLR| vs IDD iteration
  Panel 4: Symbol error rate convergence
```

**run_day10.py** — AMP Detector Benchmark
```
Adds: AMP-IDD (Approximate Message Passing detector)
Compares: ADMM-IDD vs AMP-IDD vs MMSE baseline
Output: BER comparison, AMP convergence analysis
Expected: AMP comparable to ADMM, better than MMSE
```

**run_day11_v2.py** — 16-QAM Extension
```
Modulation upgrade: BPSK → 16-QAM (4 bits/symbol)
Adjustments:
  - Uses qam_modulate() instead of bpsk_modulate()
  - Higher SNR range (4-20 dB vs 0-14 dB)
  - Fewer trials per SNR (fewer bits per trial)
Output: 4-panel:
  Panel 1: BER vs SNR
  Panel 2: Before/after decoder comparison
  Panel 3: 16-QAM constellation at reference SNR
  Panel 4: ADMM iteration sweep study
```

### Learning-Based Tier (Scripts: run_day7, run_day13, run_day14)

**run_day7.py** — Deep Unfolding
```
Concept: Unroll K=10 ADMM iterations → 10 neural network layers
Learns: {ρ_k} parameters per layer (small (10,) vector)

Phase 1: NumPy architecture walkthrough
  - Forward pass: y, H → 10 ADMM layers → x̂
  - Compute MSE w.r.t. true x
  
Phase 2: Numerical gradient check
  - Verify ∂MSE/∂ρ_k non-zero (parameters are meaningful)
  
Phase 3: PyTorch training (if torch available)
  - Loss: MSE between final x̂ and true symbols
  - Optimizer: Adam, lr=5e-3
  - Epochs: 300
  - Batch size: 64
  
Output: Loss history, learned ρ_k schedule, BER comparison

Expected: Unfolded network BER ≤ classical ADMM at training SNR
```

**run_day14_v2.py** — Grand Final Comparison (Publication Quality)
```
All 8 detectors on SAME channel realizations:
  1. Uncoded MMSE baseline
  2. MMSE + SPA (one-shot)
  3. ADMM + SPA (one-shot)
  4. EP + SPA (one-shot)
  5. AMP + SPA (one-shot)
  6. ADMM-IDD (1, 2, 4 outer iterations)
  7. EP-IDD (4 iterations)
  8. AMP-IDD (4 iterations)

High fidelity:
  - 1500 trials/SNR (48k bits per point)
  - Same H and n for all methods per trial (fair comparison)
  - Apply all 8 bug fixes
  
Output: 2 high-quality PNG figures
  - Figure 1: Main comparison (BER vs SNR, all 8 methods)
  - Figure 2: IDD convergence (2×2 panel: iteration progression)
```

**run_rho_sweep.py** — Hyperparameter Study
```
Sweep ρ ∈ [0.1, 1.0, 10, 100] at each SNR
Find: Optimal ρ(SNR) schedule for ADMM convergence
Output: Heatmap of BER vs (ρ, SNR)
```

---

## 🐛 Bug Fixes (Days 9-14 Cleanup)

| Bug # | Issue | Fix |
|-------|-------|-----|
| 1 | BPSK mapping inconsistent | Use `x = 1 - 2*bit` uniformly (bit 0→+1, bit 1→-1) |
| 2 | ADMM returns hard `z`, not soft `x` | Fixed `admm_detect()` return to continuous `x` from x-update |
| 3 | Wrong LLR from soft symbol | Use `LLR = 2*x / σ²_eff_k`, not scalar average |
| 4 | σ²_eff scalar, not per-stream | Extract diagonal of `(H^H H/σ² + ρI)^{-1}` per stream |
| 5 | LDPC n too small (n=32) | Use n=64, verified encoder works up to 64 |
| 6 | MMSE uses wrong noise variance | MMSE computes its own effective σ²_eff |
| 7 | Different channels per method | Use identical H and n for all methods in each trial |
| 8 | Low trial count (100-300) | Increase to 1500 trials/SNR for smooth curves |

---

## 📈 Expected Performance

### Typical SNR=8 dB Results (4×4 BPSK, LDPC n=64, rate=0.5)

```
Method                    BER (approx)
────────────────────────────────────
Uncoded MMSE              0.050
MMSE + SPA (1-shot)       0.010
ADMM + SPA (1-shot)       0.008
ADMM-IDD (4 iter)         0.003
EP-IDD (4 iter)           0.002
AMP-IDD (4 iter)          0.003
Deep-Unfolded ADMM        0.0025
────────────────────────────────────
```

### Gain Progression
```
Baseline MMSE                       0 dB reference
→ Add LDPC (no ADMM)               ~2 dB gain
→ Replace MMSE with ADMM           ~0.5 dB extra
→ Add IDD loop (4 iterations)      ~1.5 dB extra
→ Learn ρ per layer (deep unfold)  ~0.5 dB extra
────────────────────────────────────
Total gain from full pipeline:     ~4.5 dB
```

---

## 🧪 Diagnostic Tests

### Sanity Check: SISO BPSK AWGN
```
Theory: BER = Q(√(2·SNR_linear)) for SISO BPSK
Runs:   5000 trials at SNR = 0, 4, 8 dB
Passes: If simulated BER ≈ theoretical Q-function
```

### LDPC Encoder Validation
```
Generate random u_msg (k bits)
Encode → c (n bits)
Verify: H @ c = 0 (mod 2)
Passes: If verify succeeds
```

### IDD Convergence
```
Track BER before/after each IDD iteration
Passes: If BER decreases monotonically
```

### Constellation Diagram
```
At reference SNR, plot 1000 detected symbols
Overlay: true constellation points
Passes: If clusters tight around correct points
```

---

## 📁 Output & Diagnostics

### Typical Run Output
```
==================================================================
DAY 9 v2 -- EP-IDD vs ADMM-IDD (all bugs fixed)
  4x4 BPSK | LDPC(n=64, R=0.5) | 16 uses/block | 1500 trials/SNR
  LLR clip=30.0 | IDD damp=0.75
==================================================================

PHASE 0: Isolation tests
  Test 1: SISO BPSK AWGN
    SNR=0dB: BER=0.0796 (expect 0.0793)  [OK]
    SNR=4dB: BER=0.0144 (expect 0.0142)  [OK]
    ...
  
PHASE 1: Main BER comparison (SNR=0..14 dB, Δ=2)
  SNR    Uncoded  MMSE+SPA  ADMM+SPA  ADMM-IDD  EP-IDD
  ─────  ────────  ──────  ──────   ────────  ──────
  0 dB   0.500     0.150    0.140    0.080     0.075
  2 dB   0.420     0.095    0.085    0.035     0.030
  ...
  
Total time: 120 sec
Output: data/day9_v2_ber_comparison.png [saved]
```

### Generated Plots
```
day1_mmse_ber.png              ← MMSE baseline BER curve
day3_ldpc_convergence.png      ← BP convergence vs iterations
day6_idd_comparison.png        ← MMSE+SPA vs ADMM+SPA vs ADMM-IDD
day9_v2_main_comparison.png    ← 4-panel comprehensive
day10_amp_benchmark.png        ← ADMM vs AMP vs MMSE
day11_16qam_comparison.png     ← 16-QAM: MMSE vs ADMM
day14_main_comparison.png      ← ALL 8 METHODS (publication)
day14_idd_convergence.png      ← IDD iteration analysis
```

---

## 🚀 Quick Start Guide

### Step 1: Verify Installation
```bash
cd c:\Users\Lenovo\ADMM
python -c "import numpy; import matplotlib; print('OK')"
```

### Step 2: Run Validation Pipeline
```bash
# Channel simulator
python scripts/run_day1.py

# LDPC codec test
python scripts/run_day3_day4.py

# Debug pipeline (5 steps)
python scripts/debug_pipeline.py
```

### Step 3: Run Main Detector Comparison
```bash
# Classical IDD
python scripts/run_day6.py

# ADMM-IDD vs EP-IDD (comprehensive)
python scripts/run_day9_v2.py

# Full comparison (all 8 methods)
python scripts/run_day14_v2.py
```

### Step 4: Explore Learning (if PyTorch available)
```bash
# Deep unfolding training
python scripts/run_day7.py
```

---

## 📚 Key References

- **ADMM Algorithm**: Boyd et al. "Distributed Optimization and Statistical Learning" (2011)
- **LDPC Codes**: MacKay & Neal (1996), Gallager (1963)
- **Sum-Product Algorithm**: Pearl (1988), Kschischang et al. (2001)
- **IDD Framework**: Hagenauer et al. (1996)
- **Deep Unfolding**: Gregor & LeCun (2010), Metzler et al. (2017)

---

## ✅ Current Status (Latest: Day 14)

✓ **Complete**: BPSK MIMO detection with ADMM + LDPC + IDD  
✓ **Complete**: 16-QAM extension (Day 11)  
✓ **Complete**: AMP detector alternative (Day 10)  
✓ **Complete**: EP detector alternative (Days 9-10)  
✓ **Complete**: Deep unfolding learning framework (Day 7)  
✓ **Complete**: All 8 bug fixes applied (Days 9-14)  
✓ **Complete**: Publication-quality final comparison (Day 14)  

📊 **Last Run**: day14_v2 generated 8-method grand comparison  
🐛 **Known Issues**: None (all critical bugs fixed)  
⚡ **Performance**: ~4.5 dB gain over uncoded MMSE at BER=10^-3

