

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.channel import (
    generate_channel,
    bpsk_modulate,
    bpsk_demodulate_hard,
    awgn_channel,
    mmse_detect,
    compute_ber,
)

class TestChannelGeneration:
    def test_shape(self):
        H = generate_channel(Nr=4, Nt=4)
        assert H.shape == (4, 4)

    def test_complex_dtype(self):
        H = generate_channel(4, 4)
        assert np.iscomplexobj(H)

    def test_power_normalisation(self):

        Nr, Nt = 100, 8
        H = generate_channel(Nr, Nt, seed=0)
        mean_power = np.mean(np.abs(H)**2)
        assert abs(mean_power - 1.0 / Nt) < 0.01, f"Mean entry power {mean_power:.4f} ≠ 1/{Nt}"

    def test_reproducibility(self):
        H1 = generate_channel(4, 4, seed=42)
        H2 = generate_channel(4, 4, seed=42)
        np.testing.assert_array_equal(H1, H2)

class TestBPSK:
    def test_mapping(self):
        bits = np.array([0, 1, 0, 1])
        syms = bpsk_modulate(bits)
        np.testing.assert_array_equal(syms, [-1, 1, -1, 1])

    def test_hard_decision(self):
        syms = np.array([0.8, -0.3, 1.2, -0.1])
        bits = bpsk_demodulate_hard(syms)
        np.testing.assert_array_equal(bits, [1, 0, 1, 0])

    def test_roundtrip_no_noise(self):

        bits = np.array([1, 0, 1, 1, 0])
        syms = bpsk_modulate(bits)
        recovered = bpsk_demodulate_hard(syms)
        np.testing.assert_array_equal(bits, recovered)

class TestAWGNChannel:
    def test_output_shape(self):
        Nr, Nt = 4, 4
        H = generate_channel(Nr, Nt, seed=0)
        x = bpsk_modulate(np.array([1, 0, 1, 0]))
        y, sigma2 = awgn_channel(H, x, snr_db=10.0)
        assert y.shape == (Nr,)
        assert np.iscomplexobj(y)

    def test_sigma2_decreases_with_snr(self):

        H = generate_channel(4, 4, seed=0)
        x = bpsk_modulate(np.ones(4, dtype=int))
        _, sig_low  = awgn_channel(H, x, snr_db=0.0)
        _, sig_high = awgn_channel(H, x, snr_db=20.0)
        assert sig_high < sig_low

class TestMMSEDetector:
    def test_noiseless_recovery(self):

        Nr, Nt = 4, 4
        rng = np.random.default_rng(0)
        H = generate_channel(Nr, Nt, seed=0)
        bits = np.array([1, 0, 1, 0])
        x = bpsk_modulate(bits)

        y, sigma2 = awgn_channel(H, x, snr_db=40.0, rng=rng)
        x_hat = mmse_detect(y, H, sigma2)
        bits_rx = bpsk_demodulate_hard(x_hat)

        assert compute_ber(bits, bits_rx) == 0.0

    def test_ber_improves_with_snr(self):

        Nr, Nt = 4, 4
        rng = np.random.default_rng(1)
        n_trials = 200

        def run_ber(snr_db):
            errors = 0
            for _ in range(n_trials):
                H = generate_channel(Nr, Nt)
                bits = rng.integers(0, 2, size=Nt)
                x = bpsk_modulate(bits)
                y, sigma2 = awgn_channel(H, x, snr_db, rng)
                x_hat = mmse_detect(y, H, sigma2)
                errors += np.sum(bits != bpsk_demodulate_hard(x_hat))
            return errors / (n_trials * Nt)

        ber_low_snr  = run_ber(snr_db=0.0)
        ber_high_snr = run_ber(snr_db=15.0)
        assert ber_high_snr < ber_low_snr, (
            f"BER did not improve: low={ber_low_snr:.4f}, high={ber_high_snr:.4f}"
        )

