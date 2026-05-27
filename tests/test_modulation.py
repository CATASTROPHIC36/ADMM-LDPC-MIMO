

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.modulation import (
    qam_modulate, qam_demodulate_hard,
    compute_llrs_mmse, bpsk_llr,
    compute_llrs_qam, compute_llr_awgn,
    _pam_gray_bit_table, _pam_levels,
)

class TestGrayCoding:
    def test_pam4_levels(self):

        levels = _pam_levels(4)
        np.testing.assert_array_equal(levels, [-3, -1, 1, 3])

    def test_gray_table_4pam(self):

        table = _pam_gray_bit_table(4)
        assert table.shape == (4, 2)
        for i in range(3):
            diff = np.sum(table[i] != table[i + 1])
            assert diff == 1, f"Rows {i} and {i+1} differ in {diff} bits (expected 1)"

    def test_all_symbols_unique(self):

        from itertools import product
        bits = np.array(list(product(range(2), repeat=4))).ravel()

        syms = []
        for pat in np.array(list(product(range(2), repeat=4))):
            s = qam_modulate(pat, M=16)
            syms.append(s[0])
        assert len(set(syms)) == 16, "16-QAM symbols are not all unique"

class TestQAMModulator:
    def test_output_shape(self):
        bits = np.zeros(40, dtype=int)
        syms = qam_modulate(bits, M=16)
        assert syms.shape == (10,)

    def test_complex_output(self):
        bits = np.zeros(8, dtype=int)
        syms = qam_modulate(bits, M=16)
        assert np.iscomplexobj(syms)

    def test_average_power_unity(self):

        rng = np.random.default_rng(42)
        bits = rng.integers(0, 2, size=4000)
        syms = qam_modulate(bits, M=16)
        avg_power = np.mean(np.abs(syms) ** 2)
        assert abs(avg_power - 1.0) < 0.05, f"Avg power {avg_power:.4f} ≠ 1"

    def test_roundtrip_noiseless(self):

        rng = np.random.default_rng(7)
        bits_tx = rng.integers(0, 2, size=400)
        syms    = qam_modulate(bits_tx, M=16)
        bits_rx = qam_demodulate_hard(syms, M=16)
        np.testing.assert_array_equal(bits_tx, bits_rx)

    def test_bpsk_special_case(self):

        bits = np.array([0, 0])
        sym  = qam_modulate(bits, M=4)
        assert np.abs(np.abs(sym[0]) - 1.0) < 1e-9

    def test_bad_M_raises(self):
        with pytest.raises(AssertionError):
            qam_modulate(np.zeros(4, dtype=int), M=8)

class TestLLRComputation:
    def test_bpsk_llr_sign_bit0(self):

        llr = bpsk_llr(np.array([-2.0]), sigma2_eq=1.0)
        assert llr[0] > 0

    def test_bpsk_llr_sign_bit1(self):

        llr = bpsk_llr(np.array([2.0]), sigma2_eq=1.0)
        assert llr[0] < 0

    def test_bpsk_llr_magnitude(self):

        llr_low  = bpsk_llr(np.array([1.0]), sigma2_eq=2.0)
        llr_high = bpsk_llr(np.array([1.0]), sigma2_eq=0.1)
        assert abs(llr_high[0]) > abs(llr_low[0])

    def test_qam_llr_shape(self):

        n_sym = 5
        y_eq  = np.random.default_rng(0).standard_normal(n_sym) +                1j * np.random.default_rng(1).standard_normal(n_sym)
        llrs  = compute_llrs_qam(y_eq, sigma2_eq=0.5, M=16)
        assert llrs.shape == (n_sym * 4,)

    def test_llr_correct_sign_high_snr(self):

        rng   = np.random.default_rng(42)
        bits  = np.zeros(40, dtype=int)
        syms  = qam_modulate(bits, M=16)

        noisy = syms + 1e-4 * (rng.standard_normal(10) + 1j * rng.standard_normal(10))
        llrs  = compute_llrs_qam(noisy, sigma2_eq=1e-6, M=16)

        assert np.all(llrs > 0), f"Expected all LLRs > 0, got min={llrs.min():.4f}"

class TestMMSELLR:

    def test_output_shape(self):
        from src.channel import generate_channel, awgn_channel
        Nr, Nt = 4, 4
        H  = generate_channel(Nr, Nt, seed=0)
        bits = np.zeros(Nt * 4, dtype=int)
        x    = qam_modulate(bits, M=16)
        y, s2 = awgn_channel(H, x, snr_db=15.0)
        llrs = compute_llrs_mmse(y, H, s2, M=16)
        assert llrs.shape == (Nt * 4,)

    def test_ber_improves_with_snr(self):
        from src.channel import generate_channel, awgn_channel
        Nr, Nt = 4, 4
        rng = np.random.default_rng(99)
        n_trials = 100

        def get_ber(snr_db):
            errs, total = 0, 0
            for _ in range(n_trials):
                bits = rng.integers(0, 2, size=Nt * 4)
                x    = qam_modulate(bits, M=16)
                H    = generate_channel(Nr, Nt)
                y, s2 = awgn_channel(H, x, snr_db, rng)
                llrs  = compute_llrs_mmse(y, H, s2, M=16)
                bits_rx = (llrs <= 0).astype(int)
                errs  += np.sum(bits != bits_rx)
                total += len(bits)
            return errs / total

        ber_low  = get_ber(0.0)
        ber_high = get_ber(20.0)
        assert ber_high < ber_low, f"BER not improving: low={ber_low:.4f}, high={ber_high:.4f}"

