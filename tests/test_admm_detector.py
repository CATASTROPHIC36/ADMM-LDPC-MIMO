
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.admm_detector import (
    project_bpsk, project_qam, project_constellation,
    admm_detect, admm_llrs, admm_effective_sigma2,
    soft_project_bpsk
)
from src.channel import generate_channel, awgn_channel

class TestProjections:
    def test_bpsk_values(self):
        v = np.array([-2.0, -0.5, 0.0, 0.3, 1.5])
        z = project_bpsk(v)
        assert set(np.unique(z)).issubset({-1.0, 1.0})

    def test_bpsk_sign(self):
        v = np.array([-3.0, 3.0])
        z = project_bpsk(v)
        assert z[0] == -1.0 and z[1] == 1.0

    def test_bpsk_tie(self):

        assert project_bpsk(np.array([0.0]))[0] == 1.0

    def test_qam16_on_constellation(self):

        rng = np.random.default_rng(0)
        v   = (rng.standard_normal(50) + 1j * rng.standard_normal(50)) * 1.5
        z   = project_qam(v, M=16)
        m_pam  = 4
        levels = np.arange(-(m_pam - 1), m_pam, 2, dtype=float) / np.sqrt(10.0)
        for sym in z:
            assert any(np.isclose(np.real(sym), l, atol=1e-9) for l in levels)
            assert any(np.isclose(np.imag(sym), l, atol=1e-9) for l in levels)

    def test_qam4_on_constellation(self):
        v = np.array([0.5 + 0.5j, -0.5 + 0.5j, -0.5 - 0.5j, 0.5 - 0.5j])
        z = project_qam(v, M=4)
        m_pam = 2
        levels = np.arange(-(m_pam - 1), m_pam, 2, dtype=float) / np.sqrt(2.0)
        for sym in z:
            assert any(np.isclose(np.real(sym), l, atol=1e-9) for l in levels)
            assert any(np.isclose(np.imag(sym), l, atol=1e-9) for l in levels)

    def test_dispatch_bpsk(self):
        v = np.array([0.5, -0.5])
        z = project_constellation(v, M=2)
        np.testing.assert_array_equal(z, [1.0, -1.0])

    def test_soft_project_prior(self):

        v         = np.array([-0.1])
        llr_prior = np.array([5.0])
        z         = soft_project_bpsk(v, llr_prior, rho=1.0)
        assert z[0] == 1.0, "Strong prior should win over weak ADMM estimate"

class TestADMMDetect:
    def _setup(self, seed=0, snr_db=20.0, Nt=4, Nr=4):
        rng  = np.random.default_rng(seed)
        H    = generate_channel(Nr, Nt, seed=seed)
        bits = rng.integers(0, 2, size=Nt)
        x    = 1.0 - 2.0 * bits.astype(float)
        y, s2 = awgn_channel(H, x, snr_db, rng)
        return H, x, bits, y, s2

    def test_high_snr_bpsk(self):

        H, x, bits, y, s2 = self._setup(snr_db=30.0)
        z, _ = admm_detect(y, H, s2, M=2, rho=1.0, max_iter=20)
        bits_rx = (np.real(z) < 0).astype(int)
        assert np.sum(bits != bits_rx) == 0, "Should recover perfectly at 30 dB"

    def test_output_shape_bpsk(self):
        H, x, bits, y, s2 = self._setup()
        z, _ = admm_detect(y, H, s2, M=2, max_iter=5)
        assert z.shape == (4,), f"Expected shape (4,), got {z.shape}"

    def test_return_soft(self):
        H, x, bits, y, s2 = self._setup()
        z_hard, z_soft = admm_detect(y, H, s2, M=2, return_soft=True)
        assert z_soft is not None

    def test_return_no_soft(self):
        H, x, bits, y, s2 = self._setup()
        z_hard, z_soft = admm_detect(y, H, s2, M=2, return_soft=False)
        assert z_soft is None

    def test_more_iters_not_worse_on_avg(self):

        rng = np.random.default_rng(99)
        n_trials = 100
        Nr, Nt = 4, 4
        snr_db = 8.0

        def run(K):
            errs = 0
            for _ in range(n_trials):
                H    = generate_channel(Nr, Nt)
                bits = rng.integers(0, 2, size=Nt)
                x    = 1.0 - 2.0 * bits.astype(float)
                y, s2 = awgn_channel(H, x, snr_db, rng)
                z, _  = admm_detect(y, H, s2, M=2, rho=1.0, max_iter=K)
                errs += int(np.sum(bits != (np.real(z) < 0).astype(int)))
            return errs / (n_trials * Nt)

        ber_5  = run(5)
        ber_20 = run(20)
        assert ber_20 <= ber_5 + 0.05, (
            f"BER with K=20 ({ber_20:.4f}) should not be much worse than K=5 ({ber_5:.4f})"
        )

    def test_effective_sigma2_positive(self):
        H  = generate_channel(4, 4, seed=0)
        s2 = admm_effective_sigma2(H, 0.1, rho=1.0)
        assert s2 > 0, "Effective noise variance must be positive"

    def test_llrs_bpsk_sign(self):

        z_soft = np.array([2.0, -2.0, 0.5, -0.5])
        llrs   = admm_llrs(z_soft, sigma2_eff=0.5, M=2)
        assert llrs[0] > 0, "Positive z → LLR > 0 (likely bit 0 = symbol +1)"
        assert llrs[1] < 0, "Negative z → LLR < 0 (likely bit 1 = symbol -1)"

    def test_prior_shifts_decision(self):

        rng  = np.random.default_rng(5)
        H    = generate_channel(4, 4, seed=5)
        bits = np.zeros(4, dtype=int)
        x    = np.ones(4)
        y, s2 = awgn_channel(H, x, 0.0, rng)

        strong_prior = np.array([10.0, 10.0, 10.0, 10.0])
        z, _ = admm_detect(y, H, s2, M=2, rho=1.0, max_iter=20,
                           llr_prior=strong_prior)

        bits_rx = (np.real(z) < 0).astype(int)
        assert np.sum(bits_rx) == 0, "Strong prior should force all bits to 0"

