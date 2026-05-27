

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ldpc import (
    make_ldpc_matrix, ldpc_encode, verify_codeword,
    build_tanner_graph, spa_decode, min_sum_decode, spa_decode_soft
)

@pytest.fixture(scope="module")
def small_ldpc():

    H = make_ldpc_matrix(n=100, rate=0.5, seed=0)
    vn_to_cn, cn_to_vn = build_tanner_graph(H)
    return H, vn_to_cn, cn_to_vn

class TestLDPCMatrix:
    def test_shape(self, small_ldpc):
        H, _, _ = small_ldpc
        m, n = H.shape
        assert n == 100
        assert m == 50

    def test_binary(self, small_ldpc):
        H, _, _ = small_ldpc
        assert set(np.unique(H)).issubset({0, 1})

    def test_variable_degree(self, small_ldpc):

        H, _, _ = small_ldpc
        col_sums = H.sum(axis=0)
        assert np.all(col_sums == 3), f"Variable degrees not all 3: {np.unique(col_sums)}"

    def test_sparsity(self, small_ldpc):

        H, _, _ = small_ldpc
        density = H.mean()
        assert density < 0.1, f"Matrix too dense: {density:.3f}"

class TestLDPCEncoder:
    def test_codeword_length(self, small_ldpc):
        H, _, _ = small_ldpc
        m, n = H.shape
        k = n - m
        u = np.zeros(k, dtype=int)
        c = ldpc_encode(u, H)
        assert len(c) == n

    def test_systematic_part_preserved(self, small_ldpc):

        H, _, _ = small_ldpc
        m, n = H.shape; k = n - m
        rng = np.random.default_rng(1)
        u = rng.integers(0, 2, size=k)
        c = ldpc_encode(u, H)

        assert verify_codeword(c, H), "Encoded codeword fails parity check"

    def test_parity_check(self, small_ldpc):

        H, _, _ = small_ldpc
        m, n = H.shape; k = n - m
        rng = np.random.default_rng(2)
        for _ in range(20):
            u = rng.integers(0, 2, size=k)
            c = ldpc_encode(u, H)
            assert verify_codeword(c, H), "Parity check failed!"

    def test_zero_message(self, small_ldpc):

        H, _, _ = small_ldpc
        m, n = H.shape; k = n - m
        c = ldpc_encode(np.zeros(k, dtype=int), H)
        assert verify_codeword(c, H)

    def test_binary_output(self, small_ldpc):
        H, _, _ = small_ldpc
        m, n = H.shape; k = n - m
        c = ldpc_encode(np.ones(k, dtype=int), H)
        assert set(np.unique(c)).issubset({0, 1})

class TestTannerGraph:
    def test_vn_to_cn_length(self, small_ldpc):
        H, vn_to_cn, _ = small_ldpc
        assert len(vn_to_cn) == H.shape[1]

    def test_cn_to_vn_length(self, small_ldpc):
        H, _, cn_to_vn = small_ldpc
        assert len(cn_to_vn) == H.shape[0]

    def test_edge_consistency(self, small_ldpc):

        H, vn_to_cn, cn_to_vn = small_ldpc
        for v, checks in enumerate(vn_to_cn):
            for c in checks:
                assert v in cn_to_vn[c], f"Inconsistent edge v={v}, c={c}"

    def test_degree_matches_H(self, small_ldpc):
        H, vn_to_cn, cn_to_vn = small_ldpc
        for v in range(H.shape[1]):
            assert len(vn_to_cn[v]) == H[:, v].sum()
        for c in range(H.shape[0]):
            assert len(cn_to_vn[c]) == H[c, :].sum()

class TestSPADecoder:
    def _make_channel_llrs(self, c, sigma2, seed=0):

        rng = np.random.default_rng(seed)
        x = 1.0 - 2.0 * c.astype(float)
        y = x + rng.standard_normal(len(c)) * np.sqrt(sigma2)
        return 2.0 * y / sigma2

    def test_output_shape(self, small_ldpc):
        H, vn_to_cn, cn_to_vn = small_ldpc
        n = H.shape[1]
        llr = np.zeros(n)
        b, conv, nit = spa_decode(llr, H, max_iter=5,
                                   vn_to_cn=vn_to_cn, cn_to_vn=cn_to_vn)
        assert b.shape == (n,)
        assert isinstance(conv, (bool, np.bool_))
        assert isinstance(nit, int)

    def test_high_snr_convergence(self, small_ldpc):

        H, vn_to_cn, cn_to_vn = small_ldpc
        m, n = H.shape; k = n - m
        rng = np.random.default_rng(5)
        u = rng.integers(0, 2, size=k)
        c = ldpc_encode(u, H)

        llr_ch = self._make_channel_llrs(c, sigma2=0.01, seed=10)
        b_hat, converged, n_iter = spa_decode(llr_ch, H, max_iter=50,
                                               vn_to_cn=vn_to_cn, cn_to_vn=cn_to_vn)

        errors = np.sum(b_hat != c)
        assert errors == 0, f"Wrong bits at high SNR: {errors} errors, converged={converged}"

    def test_all_zero_codeword(self, small_ldpc):

        H, vn_to_cn, cn_to_vn = small_ldpc
        n = H.shape[1]

        llr_ch = np.full(n, 5.0)
        b_hat, conv, _ = spa_decode(llr_ch, H, max_iter=20,
                                     vn_to_cn=vn_to_cn, cn_to_vn=cn_to_vn)
        assert np.all(b_hat == 0), "All-positive LLRs should decode to all zeros"

    def test_converged_is_valid_codeword(self, small_ldpc):

        H, vn_to_cn, cn_to_vn = small_ldpc
        m, n = H.shape; k = n - m
        rng = np.random.default_rng(7)
        u = rng.integers(0, 2, size=k)
        c = ldpc_encode(u, H)

        llr_ch = self._make_channel_llrs(c, sigma2=0.1, seed=20)
        b_hat, converged, _ = spa_decode(llr_ch, H, max_iter=50,
                                          vn_to_cn=vn_to_cn, cn_to_vn=cn_to_vn)
        if converged:
            assert verify_codeword(b_hat, H), "Converged output is not a valid codeword!"

    def test_min_sum_matches_spa_direction(self, small_ldpc):

        H, vn_to_cn, cn_to_vn = small_ldpc
        m, n = H.shape; k = n - m
        rng = np.random.default_rng(99)
        n_trials = 20
        total_errors = 0
        for _ in range(n_trials):
            u = rng.integers(0, 2, size=k)
            c = ldpc_encode(u, H)
            llr_ch = self._make_channel_llrs(c, sigma2=0.01, seed=int(rng.integers(0,1000)))
            b_ms, _, _ = min_sum_decode(llr_ch, H, max_iter=50)
            total_errors += np.sum(b_ms != c)
        ber = total_errors / (n_trials * n)
        assert ber < 0.05, f"Min-sum BER too high at high SNR: {ber:.4f}"

    def test_soft_output_shape(self, small_ldpc):
        H, vn_to_cn, cn_to_vn = small_ldpc
        n = H.shape[1]
        llr_ch = np.random.default_rng(0).standard_normal(n)
        b, llr_out, conv, nit = spa_decode_soft(llr_ch, H, max_iter=10,
                                                  vn_to_cn=vn_to_cn, cn_to_vn=cn_to_vn)
        assert llr_out.shape == (n,)

    def test_extrinsic_differs_from_channel(self, small_ldpc):

        H, vn_to_cn, cn_to_vn = small_ldpc
        n = H.shape[1]
        rng = np.random.default_rng(3)
        llr_ch = rng.standard_normal(n)
        _, llr_out, _, _ = spa_decode_soft(llr_ch, H, max_iter=10,
                                            vn_to_cn=vn_to_cn, cn_to_vn=cn_to_vn)
        extrinsic = llr_out - llr_ch
        assert np.any(extrinsic != 0), "Extrinsic LLR is all zeros — SPA did nothing"

