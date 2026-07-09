

import numpy as np
from typing import Tuple, Optional

def make_ldpc_matrix(n: int = 648, rate: float = 0.5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = int(n * (1 - rate))
    k = n - m
    
    # Create a sparse P matrix (k columns, m rows)
    # To keep the LDPC graph sparse, we give each column of P weight d_v = 3
    P = np.zeros((m, k), dtype=np.int8)
    d_v = 3
    for v in range(k):
        rows = rng.choice(m, size=d_v, replace=False)
        P[rows, v] = 1
        
    # H = [P | I]
    I = np.eye(m, dtype=np.int8)
    H = np.hstack((P, I))
    return H

def load_or_build_ldpc(
    n: int = 648,
    rate: float = 0.5,
    cache_path: str = "data/ldpc_matrix.npz"
) -> np.ndarray:

    import os
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        H = data["H"]
        print(f"  Loaded LDPC matrix from {cache_path}: shape {H.shape}")
    else:
        print(f"  Building random (3,6)-LDPC matrix  n={n}, rate={rate} ...")
        H = make_ldpc_matrix(n, rate)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, H=H)
        print(f"  Saved to {cache_path}")
    return H

def ldpc_encode(u: np.ndarray, H: np.ndarray) -> np.ndarray:
    m, n = H.shape
    k = n - m
    assert len(u) == k, f"Expected {k} message bits, got {len(u)}"
    
    # H is [P | I], so P is the first k columns of H
    P = H[:, :k]
    p = (P @ u) % 2
    
    c = np.concatenate([u, p]).astype(np.int8)
    return c

def verify_codeword(c: np.ndarray, H: np.ndarray) -> bool:

    syndrome = (H.astype(int) @ c.astype(int)) % 2
    return bool(np.all(syndrome == 0))

def build_tanner_graph(H: np.ndarray):

    m, n = H.shape
    cn_to_vn = [list(np.where(H[c, :] == 1)[0]) for c in range(m)]
    vn_to_cn = [list(np.where(H[:, v] == 1)[0]) for v in range(n)]
    return vn_to_cn, cn_to_vn

def spa_decode(
    llr_ch: np.ndarray,
    H: np.ndarray,
    max_iter: int = 50,
    vn_to_cn: list = None,
    cn_to_vn: list = None
) -> Tuple[np.ndarray, bool, int]:

    m, n = H.shape

    if vn_to_cn is None or cn_to_vn is None:
        vn_to_cn, cn_to_vn = build_tanner_graph(H)

    L_v2c = [np.zeros(len(vn_to_cn[v])) for v in range(n)]
    L_c2v = [np.zeros(len(cn_to_vn[c])) for c in range(m)]

    for v in range(n):
        L_v2c[v][:] = llr_ch[v]

    EPS = 1e-10

    def phi(x):

        x = np.maximum(x, EPS)
        return -np.log(np.tanh(np.minimum(x, 20.0) / 2.0) + EPS)

    for iteration in range(max_iter):

        for c in range(m):
            neighbors_v = cn_to_vn[c]
            msgs = np.array([L_v2c[v][vn_to_cn[v].index(c)] for v in neighbors_v])

            signs = np.sign(msgs)
            signs[signs == 0] = 1

            prod_sign = np.prod(signs)

            mag = np.abs(msgs)
            phi_mag = phi(mag)

            sum_phi = np.sum(phi_mag)

            for i, v in enumerate(neighbors_v):

                sign_excl = prod_sign * signs[i]
                sum_excl  = sum_phi - phi_mag[i]

                magnitude = phi(np.maximum(sum_excl, EPS))
                v_idx_in_c = cn_to_vn[c].index(v)
                L_c2v[c][v_idx_in_c] = sign_excl * magnitude

        for v in range(n):
            neighbors_c = vn_to_cn[v]

            total_check = sum(
                L_c2v[c][cn_to_vn[c].index(v)] for c in neighbors_c
            )
            for j, c in enumerate(neighbors_c):

                excl_c_msg = L_c2v[c][cn_to_vn[c].index(v)]
                L_v2c[v][j] = llr_ch[v] + total_check - excl_c_msg

        L_total = llr_ch.copy()
        for v in range(n):
            for c in vn_to_cn[v]:
                L_total[v] += L_c2v[c][cn_to_vn[c].index(v)]

        b_hat = (L_total < 0).astype(int)

        syndrome = (H.astype(int) @ b_hat) % 2
        if np.all(syndrome == 0):
            return b_hat, True, iteration + 1

    return b_hat, False, max_iter

def min_sum_decode(
    llr_ch: np.ndarray,
    H: np.ndarray,
    max_iter: int = 30,
    scaling: float = 0.75
) -> Tuple[np.ndarray, bool, int]:

    m, n = H.shape
    vn_to_cn, cn_to_vn = build_tanner_graph(H)

    L_v2c = [np.full(len(vn_to_cn[v]), llr_ch[v]) for v in range(n)]
    L_c2v = [np.zeros(len(cn_to_vn[c])) for c in range(m)]

    for iteration in range(max_iter):

        for c in range(m):
            neighbors_v = cn_to_vn[c]
            msgs = np.array([L_v2c[v][vn_to_cn[v].index(c)] for v in neighbors_v])
            signs = np.sign(msgs)
            signs[signs == 0] = 1
            mags  = np.abs(msgs)
            prod_sign = np.prod(signs)

            sorted_idx = np.argsort(mags)
            min1 = mags[sorted_idx[0]]
            min2 = mags[sorted_idx[1]] if len(mags) > 1 else min1

            for i, v in enumerate(neighbors_v):
                sign_excl = prod_sign * signs[i]
                min_excl  = min2 if i == sorted_idx[0] else min1
                v_idx = cn_to_vn[c].index(v)
                L_c2v[c][v_idx] = scaling * sign_excl * min_excl

        for v in range(n):
            neighbors_c = vn_to_cn[v]
            total = sum(L_c2v[c][cn_to_vn[c].index(v)] for c in neighbors_c)
            for j, c in enumerate(neighbors_c):
                excl = L_c2v[c][cn_to_vn[c].index(v)]
                L_v2c[v][j] = llr_ch[v] + total - excl

        L_total = llr_ch.copy()
        for v in range(n):
            for c in vn_to_cn[v]:
                L_total[v] += L_c2v[c][cn_to_vn[c].index(v)]

        b_hat = (L_total < 0).astype(int)

        syndrome = (H.astype(int) @ b_hat) % 2
        if np.all(syndrome == 0):
            return b_hat, True, iteration + 1

    return b_hat, False, max_iter

def spa_decode_soft(
    llr_ch: np.ndarray,
    H: np.ndarray,
    max_iter: int = 50,
    vn_to_cn: list = None,
    cn_to_vn: list = None
) -> Tuple[np.ndarray, np.ndarray, bool, int]:

    m, n = H.shape
    if vn_to_cn is None or cn_to_vn is None:
        vn_to_cn, cn_to_vn = build_tanner_graph(H)

    L_v2c = [np.full(len(vn_to_cn[v]), llr_ch[v]) for v in range(n)]
    L_c2v = [np.zeros(len(cn_to_vn[c])) for c in range(m)]

    EPS = 1e-10

    def phi(x):
        x = np.maximum(x, EPS)
        return -np.log(np.tanh(np.minimum(x, 20.0) / 2.0) + EPS)

    L_total = llr_ch.copy()

    for iteration in range(max_iter):
        for c in range(m):
            neighbors_v = cn_to_vn[c]
            msgs  = np.array([L_v2c[v][vn_to_cn[v].index(c)] for v in neighbors_v])
            signs = np.sign(msgs)
            signs[signs == 0] = 1
            prod_sign = np.prod(signs)
            phi_mag   = phi(np.abs(msgs))
            sum_phi   = np.sum(phi_mag)

            for i, v in enumerate(neighbors_v):
                sign_excl = prod_sign * signs[i]
                sum_excl  = sum_phi - phi_mag[i]
                v_idx = cn_to_vn[c].index(v)
                L_c2v[c][v_idx] = sign_excl * phi(np.maximum(sum_excl, EPS))

        for v in range(n):
            neighbors_c = vn_to_cn[v]
            total = sum(L_c2v[c][cn_to_vn[c].index(v)] for c in neighbors_c)
            for j, c in enumerate(neighbors_c):
                excl = L_c2v[c][cn_to_vn[c].index(v)]
                L_v2c[v][j] = llr_ch[v] + total - excl

        L_total = llr_ch.copy()
        for v in range(n):
            for c in vn_to_cn[v]:
                L_total[v] += L_c2v[c][cn_to_vn[c].index(v)]

        b_hat = (L_total < 0).astype(int)
        syndrome = (H.astype(int) @ b_hat) % 2
        if np.all(syndrome == 0):
            return b_hat, L_total, True, iteration + 1

    b_hat = (L_total < 0).astype(int)
    return b_hat, L_total, False, max_iter

