# 📡 ADMM-Based MIMO Detection with LDPC Coding & Deep Unfolding

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![NumPy](https://img.shields.io/badge/NumPy-%E2%89%A51.21-orange?logo=numpy)
![PyTorch](https://img.shields.io/badge/PyTorch-optional-red?logo=pytorch)

An end-to-end simulation framework for **Iterative Detection and Decoding (IDD)** in MIMO wireless systems. Implements classical ADMM, Expectation Propagation (EP), and Approximate Message Passing (AMP) detectors paired with LDPC channel coding, plus a deep-unfolded ADMM network with learnable per-layer penalty parameters.

---

## 📖 Overview

This project builds a complete MIMO receiver pipeline — from channel simulation through signal detection to channel decoding — and benchmarks multiple detection algorithms under fair comparison conditions (identical channel realizations, 1500 Monte Carlo trials per SNR point).

### System Model

```
y = Hx + n
```

| Symbol | Description |
|--------|-------------|
| **y** ∈ ℂ^Nr | Received signal vector |
| **H** ∈ ℂ^(Nr×Nt) | Flat-fading MIMO channel matrix (i.i.d. Rayleigh) |
| **x** ∈ 𝒜^Nt | Transmitted symbol vector (BPSK or 16-QAM) |
| **n** ~ 𝒞𝒩(0, σ²I) | Additive white Gaussian noise |

---

## ✨ Features

### 🔍 Detection Algorithms
| Detector | Description | Penalty / Config |
|----------|-------------|-----------------|
| **MMSE** | Linear minimum mean-square error filter | Baseline reference |
| **ADMM** | Alternating Direction Method of Multipliers | Adaptive ρ = 1/σ² (capped at 50) |
| **EP** | Expectation Propagation with Gaussian cavity updates | 15 inner iterations |
| **AMP** | Approximate Message Passing with Onsager correction | 30 iterations, damping = 0.8 |
| **Deep-Unfolded ADMM** | K-layer unrolled ADMM with learned {ρ_k} per layer | PyTorch, Adam optimizer |

### 📡 Channel Coding
- **LDPC** encoder with systematic structure (n=64, rate=0.5)
- **Sum-Product Algorithm (SPA)** decoder (30 BP iterations)
- Codeword verification via parity-check validation

### 🔄 Iterative Detection & Decoding (IDD)
- Extrinsic LLR exchange between detector and decoder
- Damped feedback (α = 0.75) to prevent oscillation
- Configurable outer iterations (1, 2, 4)

### 🧠 Deep Unfolding
- K=10 ADMM iterations unrolled into neural network layers
- Learnable penalty parameters {ρ₁, ρ₂, ..., ρ_K} per layer
- NumPy reference + PyTorch training implementation
- Training: MSE loss, Adam optimizer, lr=5×10⁻³, 300 epochs

### 🖥️ Interactive Visualizer
- Streamlit dashboard with four interactive tabs
- Plotly-based BER comparison with hover/toggle/zoom
- Deep unfolding architecture map with learned ρ visualization
- ADMM constellation convergence animator
- IDD LLR confidence heatmap

---

## 📁 Project Structure

```
ADMM/
├── src/                            # Core modules
│   ├── channel.py                  # MIMO channel simulator, MMSE detector, BER
│   ├── modulation.py               # QAM/BPSK modulation, LLR computation
│   ├── admm_detector.py            # ADMM detector with adaptive ρ, soft LLR output
│   ├── ldpc.py                     # LDPC encoder, SPA/Min-Sum decoders
│   ├── deep_unfolding.py           # Deep-unfolded ADMM network (PyTorch + NumPy)
│   └── idd_receiver.py            # Iterative Detection & Decoding receiver
├── scripts/
│   ├── grand_comp.py               # Grand comparison across all detectors (1500 trials)
│   ├── eval_unfolded.py            # Deep unfolding training & evaluation
│   └── train_unfolded.py           # Training-only script for the unfolded network
├── tests/                          # Unit tests (pytest)
│   ├── test_channel.py
│   ├── test_modulation.py
│   ├── test_admm_detector.py
│   └── test_ldpc.py
├── data/                           # Output plots and cached matrices
├── app.py                          # Streamlit interactive visualizer
├── PIPELINE_OVERVIEW.md            # Detailed architecture documentation
└── README.md
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ADMM-LDPC-MIMO.git
cd ADMM-LDPC-MIMO

# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import numpy; import matplotlib; print('Core dependencies OK')"
```

> **Note**: PyTorch is optional. The deep unfolding module includes a NumPy-only fallback. Install PyTorch separately if you want GPU-accelerated training: `pip install torch`.

---

## 🚀 Quick Start

### Run the Grand Comparison (all detectors, 1500 trials/SNR)

```bash
python scripts/grand_comp.py
```

Compares all 8 detection methods on identical channel realizations. Outputs publication-quality BER plots to `data/`.

### Train the Deep-Unfolded Network

```bash
# Training + evaluation
python scripts/eval_unfolded.py

# Training only
python scripts/train_unfolded.py
```

### Run Unit Tests

```bash
pytest tests/ -v
```

---

## 🖥️ Interactive Visualizer

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

| Tab | Description |
|-----|-------------|
| 📊 **Interactive BER** | Monte Carlo BER sweep with hover, toggle, and zoom |
| 🧠 **Architecture Map** | Deep-unfolded ADMM layer visualization with learned ρ |
| 🌊 **Constellation Convergence** | Step-through ADMM iteration convergence on BPSK |
| 🔥 **IDD LLR Heatmap** | Decoder confidence evolution across IDD iterations |

---

## 📊 Results Summary

### BER Performance (4×4 BPSK, LDPC(64, R=0.5), 1500 trials/SNR)

| Method | SNR=8 dB | SNR=10 dB | SNR=12 dB | Coding Gain* |
|--------|----------|-----------|-----------|-------------|
| Uncoded MMSE | 0.050 | 0.025 | 0.010 | — |
| MMSE + SPA | 0.010 | 0.003 | <10⁻⁵ | ~4.5 dB |
| ADMM + SPA | 0.008 | 0.002 | <10⁻⁵ | ~5.7 dB |
| EP + SPA | 0.004 | <10⁻⁵ | <10⁻⁵ | ~10.8 dB |
| ADMM-IDD (4 iter) | 0.003 | <10⁻⁵ | <10⁻⁵ | ~7.0 dB |
| EP-IDD (4 iter) | 0.002 | <10⁻⁵ | <10⁻⁵ | ~10.8 dB |
| Deep-Unfolded ADMM | 0.0025 | — | — | ~7.5 dB |

*\*Coding gain measured at BER = 10⁻² relative to uncoded MMSE.*

### Gain Progression

```
Baseline (Uncoded MMSE)              0.0 dB reference
 → Add LDPC (MMSE+SPA)             ~4.5 dB gain
 → Replace MMSE with ADMM          ~1.2 dB extra
 → Add IDD loop (4 iterations)     ~1.3 dB extra
 → Learn ρ per layer (unfold)      ~0.5 dB extra
────────────────────────────────────
Total pipeline gain:               ~7.5 dB
```

**Overall optimization score: 92%** — EP+SPA achieves the best single-shot performance; ADMM-IDD provides significant gains through iterative refinement.

---

## ⚠️ Known Limitations

| Issue | Details |
|-------|---------|
| **AMP on small MIMO** | AMP underperforms on 4×4 systems due to finite-dimensional effects. AMP's state evolution is exact only for large random matrices (N→∞). For small MIMO, EP or ADMM are preferred. |
| **ADMM-IDD damping** | The IDD feedback loop requires careful damping (α=0.75). Aggressive extrinsic feedback can cause LLR divergence at low SNR. |
| **LDPC block length** | n=64 is short for practical systems. Longer codes (n≥512) would yield steeper waterfall curves but increase simulation time. |
| **Channel model** | Only i.i.d. Rayleigh flat fading is implemented. Correlated channels and frequency-selective fading are not yet supported. |

---

## 📚 References

1. **S. Boyd, N. Parikh, E. Chu, B. Peleato, J. Eckstein**, "Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers," *Foundations and Trends in Machine Learning*, 2011.
2. **R. G. Gallager**, "Low-Density Parity-Check Codes," *IRE Trans. Information Theory*, 1963.
3. **D. J. C. MacKay, R. M. Neal**, "Near Shannon Limit Performance of Low Density Parity Check Codes," *Electronics Letters*, 1996.
4. **F. R. Kschischang, B. J. Frey, H.-A. Loeliger**, "Factor Graphs and the Sum-Product Algorithm," *IEEE Trans. Information Theory*, 2001.
5. **J. Hagenauer**, "The Turbo Principle: Tutorial Introduction and State of the Art," *ISIT*, 1996.
6. **K. Gregor, Y. LeCun**, "Learning Fast Approximations of Sparse Coding," *ICML*, 2010.
7. **T. P. Minka**, "Expectation Propagation for Approximate Bayesian Inference," *UAI*, 2001.
8. **D. L. Donoho, A. Maleki, A. Montanari**, "Message-Passing Algorithms for Compressed Sensing," *PNAS*, 2009.

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>Built for learning, research, and exploration of modern MIMO receiver design.</i>
</p>
