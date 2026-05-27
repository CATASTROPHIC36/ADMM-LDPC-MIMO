

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.channel import generate_channel, awgn_channel
from src.admm_detector import admm_detect, project_bpsk
from src.ldpc import (make_ldpc_matrix, ldpc_encode, verify_codeword,
                      spa_decode_soft, build_tanner_graph)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="ADMM-LDPC-MIMO Visualizer",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown(   , unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_data
def run_ber_sweep(Nr, Nt, n_trials, snr_range_list, admm_iter, seed=42):
    """
    Simulates Bit Error Rate (BER) across a range of SNRs for MMSE, ADMM, and EP detectors.
    
    Args:
        Nr (int): Number of receive antennas.
        Nt (int): Number of transmit antennas.
        n_trials (int): Number of channel realizations per SNR point.
        snr_range_list (list): SNRs to evaluate (in dB).
        admm_iter (int): Number of ADMM iterations to run.
        seed (int): Random seed for reproducibility.
        
    Returns:
        tuple: (ber_mmse, ber_admm, ber_ep) as lists.
    """

    rng = np.random.default_rng(seed)
    snr_range = np.array(snr_range_list)
    n_snr = len(snr_range)

    ber_mmse = np.zeros(n_snr)
    ber_admm = np.zeros(n_snr)
    ber_ep   = np.zeros(n_snr)

    RHO_MAX = 50.0
    LLR_CLIP = 30.0

    for j, snr_db in enumerate(snr_range):
        e_mmse = e_admm = e_ep = 0
        total = 0

        for _ in range(n_trials):
            bits = rng.integers(0, 2, size=Nt)
            x = 1.0 - 2.0 * bits.astype(float)
            H_t = generate_channel(Nr, Nt)
            y_t, s2_t = awgn_channel(H_t, x, snr_db, rng)

            HH = H_t.conj().T @ H_t
            A = HH / s2_t + np.eye(Nt)
            Hhy = H_t.conj().T @ y_t / s2_t
            x_mmse = np.real(np.linalg.solve(A, Hhy))
            b_mmse = (x_mmse < 0).astype(int)
            e_mmse += int(np.sum(bits != b_mmse))

            rho_t = min(1.0 / max(s2_t, 1e-6), RHO_MAX)
            _, x_soft = admm_detect(y_t, H_t, s2_t, M=2, rho=rho_t,
                                     max_iter=admm_iter, return_soft=True)
            A_admm = HH / s2_t + rho_t * np.eye(Nt)
            A_inv = np.linalg.inv(A_admm)
            s2_eff = np.maximum(np.real(np.diag(A_inv)), 1e-9)
            llr_admm = np.clip(2.0 * np.real(x_soft) / s2_eff, -LLR_CLIP, LLR_CLIP)
            b_admm = (llr_admm < 0).astype(int)
            e_admm += int(np.sum(bits != b_admm))

            H_real = np.real(HH)
            Hhy_real = np.real(H_t.conj().T @ y_t)
            v_site = np.full(Nt, 1e6)
            m_site = np.zeros(Nt)
            for _ in range(15):
                Lambda = H_real / s2_t + np.diag(1.0 / np.maximum(v_site, 1e-9))
                try:
                    Sigma = np.linalg.inv(Lambda)
                except:
                    Sigma = np.linalg.pinv(Lambda)
                mu = Sigma @ (Hhy_real / s2_t + m_site / np.maximum(v_site, 1e-9))
                for k in range(Nt):
                    s_kk = max(Sigma[k, k], 1e-12)
                    v_sk = max(v_site[k], 1e-12)
                    inv_v_cav = 1.0 / s_kk - 1.0 / v_sk
                    if inv_v_cav <= 1e-12:
                        continue
                    v_cav = 1.0 / inv_v_cav
                    mu_cav = v_cav * (mu[k] / s_kk - m_site[k] / v_sk)
                    arg = np.clip(mu_cav / max(v_cav, 1e-9), -15, 15)
                    mu_new = np.tanh(arg)
                    v_new = max(1.0 - mu_new**2, 1e-6)
                    inv_vs_new = 1.0 / v_new - inv_v_cav
                    if inv_vs_new <= 1e-12:
                        continue
                    v_site[k] = 1.0 / inv_vs_new
                    m_site[k] = v_site[k] * (mu_new / v_new - mu_cav / v_cav)
            Lambda = H_real / s2_t + np.diag(1.0 / np.maximum(v_site, 1e-9))
            try:
                Sigma = np.linalg.inv(Lambda)
            except:
                Sigma = np.linalg.pinv(Lambda)
            mu = Sigma @ (Hhy_real / s2_t + m_site / np.maximum(v_site, 1e-9))
            s2_diag = np.maximum(np.real(np.diag(Sigma)), 1e-9)
            llr_ep = np.clip(2.0 * mu / s2_diag, -LLR_CLIP, LLR_CLIP)
            b_ep = (llr_ep < 0).astype(int)
            e_ep += int(np.sum(bits != b_ep))

            total += Nt

        ber_mmse[j] = e_mmse / total
        ber_admm[j] = e_admm / total
        ber_ep[j]   = e_ep   / total

    return ber_mmse.tolist(), ber_admm.tolist(), ber_ep.tolist()

@st.cache_data
def run_admm_iterations(Nr, Nt, snr_db, admm_iter, seed=42):
    """
    Runs a single channel realization and tracks the convergence of the ADMM detector.
    
    Args:
        Nr (int): Number of receive antennas.
        Nt (int): Number of transmit antennas.
        snr_db (float): Signal-to-Noise Ratio (dB).
        admm_iter (int): Number of ADMM iterations.
        seed (int): Random seed.
        
    Returns:
        tuple: (x_true, x_history, z_history, bits_true).
    """

    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=Nt)
    x_true = 1.0 - 2.0 * bits.astype(float)
    H = generate_channel(Nr, Nt)
    y, sigma2 = awgn_channel(H, x_true, snr_db, rng)

    RHO_MAX = 50.0
    rho = min(1.0 / max(sigma2, 1e-6), RHO_MAX)

    HH = H.conj().T @ H
    A = HH / sigma2 + rho * np.eye(Nt)
    Hhy = H.conj().T @ y / sigma2

    z = np.zeros(Nt, dtype=float)
    u = np.zeros(Nt, dtype=float)

    x_history = []
    z_history = []

    x_init = np.real(np.linalg.solve(HH / sigma2 + np.eye(Nt),
                                      H.conj().T @ y / sigma2))
    x_history.append(x_init.tolist())
    z_history.append(z.tolist())

    for k in range(admm_iter):
        rhs = Hhy + rho * (z - u)
        x = np.real(np.linalg.solve(A, rhs))
        v = x + u
        z = project_bpsk(np.real(v))
        u = u + x - z

        x_history.append(x.tolist())
        z_history.append(z.tolist())

    return x_true.tolist(), x_history, z_history, bits.tolist()

@st.cache_data
def run_idd_iterations(Nr, Nt, n_ldpc, rate, idd_iters, admm_iter, bp_iter,
                       snr_db, seed=42):
    """
    Performs Iterative Detection and Decoding (IDD) using ADMM and LDPC BP decoding.
    
    Args:
        Nr, Nt (int): Number of receive/transmit antennas.
        n_ldpc (int): LDPC block length.
        rate (float): LDPC code rate.
        idd_iters (int): Number of outer IDD loop iterations.
        admm_iter (int): Number of ADMM detector iterations.
        bp_iter (int): Number of Belief Propagation decoder iterations.
        snr_db (float): Signal-to-Noise Ratio (dB).
        seed (int): Random seed.
        
    Returns:
        tuple: (codeword_list, llr_history).
    """

    rng = np.random.default_rng(seed)

    H_ldpc = make_ldpc_matrix(n=n_ldpc, rate=rate, seed=0)
    vn_to_cn, cn_to_vn = build_tanner_graph(H_ldpc)
    m_ldpc = H_ldpc.shape[0]
    k_ldpc = n_ldpc - m_ldpc

    u = rng.integers(0, 2, size=k_ldpc)
    c = ldpc_encode(u, H_ldpc)
    USES_PER_BLOCK = n_ldpc // Nt
    c_2d = c.reshape(USES_PER_BLOCK, Nt)
    x_2d = 1.0 - 2.0 * c_2d.astype(float)

    RHO_MAX = 50.0
    LLR_CLIP = 30.0
    IDD_DAMP = 0.75

    channels = []
    for t in range(USES_PER_BLOCK):
        H_t = generate_channel(Nr, Nt)
        y_t, s2_t = awgn_channel(H_t, x_2d[t], snr_db, rng)
        rho_t = min(1.0 / max(s2_t, 1e-6), RHO_MAX)
        channels.append((H_t, y_t, s2_t, rho_t))

    llr_history = []
    llr_prior = np.zeros(n_ldpc)

    for idd_t in range(idd_iters):
        llr_det = np.zeros(n_ldpc)
        for t in range(USES_PER_BLOCK):
            H_t, y_t, s2_t, rho_t = channels[t]
            prior_t = llr_prior[t*Nt:(t+1)*Nt] if idd_t > 0 else None
            _, x_soft = admm_detect(y_t, H_t, s2_t, M=2, rho=rho_t,
                                     max_iter=admm_iter,
                                     llr_prior=prior_t,
                                     return_soft=True)
            HH = H_t.conj().T @ H_t
            A = HH / s2_t + rho_t * np.eye(Nt)
            A_inv = np.linalg.inv(A)
            s2_eff = np.maximum(np.real(np.diag(A_inv)), 1e-9)
            llr = np.clip(2.0 * np.real(x_soft) / s2_eff, -LLR_CLIP, LLR_CLIP)
            llr_det[t*Nt:(t+1)*Nt] = llr

        if idd_t == 0:
            llr_ext = llr_det
        else:
            llr_ext = np.clip(llr_det - llr_prior, -LLR_CLIP, LLR_CLIP)

        b_dec, llr_dec, _, _ = spa_decode_soft(
            llr_ext, H_ldpc, max_iter=bp_iter,
            vn_to_cn=vn_to_cn, cn_to_vn=cn_to_vn)

        llr_history.append(llr_dec.tolist())

        if idd_t < idd_iters - 1:
            llr_prior = IDD_DAMP * np.clip(llr_dec - llr_ext, -LLR_CLIP, LLR_CLIP)

    return c.tolist(), llr_history

# =============================================================================
# SIDEBAR CONFIGURATION
# =============================================================================
with st.sidebar:
    st.markdown("### 📡 System Configuration")

    Nr = st.slider("Receive Antennas (Nr)", 2, 8, 4)
    Nt = st.slider("Transmit Antennas (Nt)", 2, 8, 4)
    admm_iter = st.slider("ADMM Iterations", 5, 40, 20)

    st.markdown("---")
    st.markdown("### 🎯 BER Sweep Settings")
    n_trials_ber = st.slider("Trials per SNR", 50, 500, 200, step=50)
    snr_min = st.slider("SNR Min (dB)", -2, 6, 0)
    snr_max = st.slider("SNR Max (dB)", 8, 20, 14)

    st.markdown("---")
    st.markdown("### 🔄 IDD Settings")
    idd_iters = st.slider("IDD Outer Iterations", 1, 6, 4)
    snr_idd = st.slider("IDD SNR (dB)", 0, 14, 8)

    st.markdown("---")
    st.markdown("### 🌊 Convergence Settings")
    snr_conv = st.slider("Convergence SNR (dB)", 0, 14, 10)
    conv_seed = st.number_input("Random Seed", 1, 9999, 42)

# =============================================================================
# MAIN UI: HERO & METRICS
# =============================================================================
st.markdown('<div class="hero-title">ADMM-LDPC-MIMO Visualizer</div>',
            unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Deep-Unfolded ADMM Detection · '
            'Interactive Exploration · Publication-Quality Results</div>',
            unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{Nr}×{Nt}</div>
        <div class="metric-label">MIMO Config</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{admm_iter}</div>
        <div class="metric-label">ADMM Iterations</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{n_trials_ber}</div>
        <div class="metric-label">Monte Carlo Trials</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{idd_iters}</div>
        <div class="metric-label">IDD Iterations</div>
    </div>""", unsafe_allow_html=True)

# =============================================================================
# MAIN UI: TABS
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Interactive BER",
    "🧠 Architecture Map",
    "🌊 Constellation Convergence",
    "🔥 IDD LLR Heatmap"
])

# =============================================================================
# TAB 1: INTERACTIVE BER
# =============================================================================
with tab1:
    st.markdown('<div class="section-header">BER vs SNR — All Detectors</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="info-box">💡 Hover over data points for exact '
                'values. Click legend items to toggle detectors on/off. '
                'Drag to zoom into specific SNR regions.</div>',
                unsafe_allow_html=True)

    snr_range = list(range(snr_min, snr_max + 1, 2))

    with st.spinner("Running Monte Carlo BER sweep..."):
        ber_mmse, ber_admm, ber_ep = run_ber_sweep(
            Nr, Nt, n_trials_ber, snr_range, admm_iter)

    fig_ber = go.Figure()

    fig_ber.add_trace(go.Scatter(
        x=snr_range,
        y=[max(b, 1e-6) for b in ber_mmse],
        mode='lines+markers',
        name='MMSE',
        line=dict(color='#636EFA', width=2.5, dash='dash'),
        marker=dict(size=8, symbol='circle'),
        hovertemplate='SNR: %{x} dB<br>BER: %{y:.5f}<extra>MMSE</extra>'
    ))

    fig_ber.add_trace(go.Scatter(
        x=snr_range,
        y=[max(b, 1e-6) for b in ber_admm],
        mode='lines+markers',
        name='ADMM (adaptive ρ)',
        line=dict(color='#EF553B', width=3),
        marker=dict(size=9, symbol='diamond'),
        hovertemplate='SNR: %{x} dB<br>BER: %{y:.5f}<extra>ADMM</extra>'
    ))

    fig_ber.add_trace(go.Scatter(
        x=snr_range,
        y=[max(b, 1e-6) for b in ber_ep],
        mode='lines+markers',
        name='EP',
        line=dict(color='#00CC96', width=2.5, dash='dot'),
        marker=dict(size=8, symbol='triangle-up'),
        hovertemplate='SNR: %{x} dB<br>BER: %{y:.5f}<extra>EP</extra>'
    ))

    fig_ber.update_layout(
        template='plotly_dark',
        yaxis_type='log',
        yaxis_title='Bit Error Rate (BER)',
        xaxis_title='SNR (dB)',
        title=dict(
            text=f'{Nr}×{Nt} BPSK MIMO — Uncoded BER Comparison '
                 f'({n_trials_ber} trials/SNR)',
            font=dict(size=16)
        ),
        legend=dict(
            x=0.65, y=0.95,
            bgcolor='rgba(30,30,50,0.8)',
            bordercolor='rgba(102,126,234,0.3)',
            borderwidth=1,
            font=dict(size=12)
        ),
        yaxis=dict(range=[np.log10(1e-5), np.log10(0.5)]),
        height=550,
        margin=dict(t=60, b=40),
        hovermode='x unified'
    )

    st.plotly_chart(fig_ber, use_container_width=True)

    st.markdown("**Numerical Results**")
    import pandas as pd
    df_ber = pd.DataFrame({
        'SNR (dB)': snr_range,
        'MMSE': [f"{b:.5f}" for b in ber_mmse],
        'ADMM': [f"{b:.5f}" for b in ber_admm],
        'EP':   [f"{b:.5f}" for b in ber_ep],
    })
    st.dataframe(df_ber, use_container_width=True, hide_index=True)

# =============================================================================
# TAB 2: ARCHITECTURE MAP
# =============================================================================
with tab2:
    st.markdown('<div class="section-header">Deep-Unfolded ADMM Architecture</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="info-box">🧠 Each block represents one ADMM '
                'iteration / unfolded layer. The color intensity and size of '
                'the connections reflect the learned penalty parameter ρ_k. '
                'Larger ρ = stronger constraint enforcement.</div>',
                unsafe_allow_html=True)

    use_learned = st.checkbox("Use learned ρ schedule from training",
                               value=True)

    if use_learned:
        rho_values = [2.689, 4.098, 3.257, 2.101, 1.245,
                      0.868, 0.513, 0.467, 0.637, 1.096]
        K = len(rho_values)
    else:
        K = admm_iter
        rho_val = st.slider("Fixed ρ value", 0.1, 10.0, 1.0, 0.1)
        rho_values = [rho_val] * K

    K_display = min(K, 12)
    rho_display = rho_values[:K_display]

    fig_arch = go.Figure()

    layer_x = list(range(K_display + 2))
    node_y_center = 2.0
    node_labels = (["Input\ny, H, σ²"] +
                   [f"Layer {k+1}\nρ={rho_display[k]:.3f}"
                    for k in range(K_display)] +
                   ["Output\nz (soft)"])

    rho_arr = np.array(rho_display)
    rho_min, rho_max = rho_arr.min(), rho_arr.max()
    rho_norm = ((rho_arr - rho_min) / (rho_max - rho_min + 1e-9))

    for k in range(K_display + 1):
        if k < K_display:
            width = 2 + 6 * rho_norm[min(k, K_display - 1)]
            opacity = 0.4 + 0.6 * rho_norm[min(k, K_display - 1)]
        else:
            width = 3
            opacity = 0.8

        fig_arch.add_trace(go.Scatter(
            x=[layer_x[k], layer_x[k+1]],
            y=[node_y_center, node_y_center],
            mode='lines',
            line=dict(color=f'rgba(102,126,234,{opacity})', width=width),
            showlegend=False,
            hoverinfo='skip'
        ))

    node_colors = (['#2d3436'] +
                   [f'rgb({int(50+180*rn)},{int(80+100*(1-rn))},'
                    f'{int(200-150*rn)})'
                    for rn in rho_norm] +
                   ['#2d3436'])
    node_sizes = ([35] +
                  [25 + 25 * rn for rn in rho_norm] +
                  [35])

    fig_arch.add_trace(go.Scatter(
        x=layer_x,
        y=[node_y_center] * len(layer_x),
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(color='#667eea', width=2),
            symbol='square'
        ),
        text=node_labels,
        textposition='top center',
        textfont=dict(size=9, color='#b0bec5'),
        showlegend=False,
        hovertemplate='%{text}<extra></extra>'
    ))

    for k in range(K_display):
        fig_arch.add_annotation(
            x=layer_x[k+1], y=node_y_center - 0.8,
            text=f"x←solve(A,rhs)<br>z←Π(x+u)<br>u←u+x−z",
            showarrow=False,
            font=dict(size=7, color='#78909c'),
            align='center'
        )

    fig_arch.update_layout(
        template='plotly_dark',
        height=400,
        title=dict(
            text=f"K={K_display} Layer Deep-Unfolded ADMM — "
                 f"Learned ρ Schedule",
            font=dict(size=15)
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                    range=[-0.5, K_display + 1.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                    range=[0, 4]),
        margin=dict(t=60, b=20, l=20, r=20)
    )

    st.plotly_chart(fig_arch, use_container_width=True)

    st.markdown("**Learned ρ_k Values per Layer**")
    fig_rho = go.Figure()
    fig_rho.add_trace(go.Bar(
        x=[f"Layer {k+1}" for k in range(K_display)],
        y=rho_display,
        marker=dict(
            color=rho_display,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="ρ value")
        ),
        hovertemplate='Layer %{x}<br>ρ = %{y:.3f}<extra></extra>'
    ))
    fig_rho.update_layout(
        template='plotly_dark',
        height=350,
        yaxis_title='ρ_k',
        xaxis_title='Layer',
        margin=dict(t=30, b=40)
    )
    st.plotly_chart(fig_rho, use_container_width=True)

# =============================================================================
# TAB 3: CONSTELLATION CONVERGENCE
# =============================================================================
with tab3:
    st.markdown('<div class="section-header">ADMM Constellation Convergence</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="info-box">🌊 Watch how the ADMM detector\'s '
                'soft estimates (x) converge from noisy initial values towards '
                'the true BPSK constellation points (±1). Use the slider to '
                'step through iterations.</div>',
                unsafe_allow_html=True)

    with st.spinner("Running single-trial ADMM..."):
        x_true, x_hist, z_hist, bits_true = run_admm_iterations(
            Nr, Nt, snr_conv, admm_iter, seed=int(conv_seed))

    iter_select = st.slider(
        "ADMM Iteration",
        min_value=0,
        max_value=len(x_hist) - 1,
        value=0,
        help="0 = initial noisy estimate, last = final converged estimate"
    )

    x_current = np.array(x_hist[iter_select])
    z_current = np.array(z_hist[iter_select])
    x_true_arr = np.array(x_true)

    fig_conv = make_subplots(rows=1, cols=2,
                              subplot_titles=("Symbol Estimates (x)",
                                              "Convergence Trajectory"),
                              column_widths=[0.5, 0.5])

    antenna_labels = [f"Ant {k+1}" for k in range(Nt)]

    fig_conv.add_trace(go.Scatter(
        x=list(range(Nt)),
        y=x_true,
        mode='markers',
        marker=dict(size=16, color='#00CC96', symbol='star',
                    line=dict(width=2, color='white')),
        name='True x',
        hovertemplate='Ant %{x}<br>True: %{y}<extra></extra>'
    ), row=1, col=1)

    fig_conv.add_trace(go.Scatter(
        x=list(range(Nt)),
        y=x_current.tolist(),
        mode='markers',
        marker=dict(size=14, color='#EF553B', symbol='diamond',
                    line=dict(width=1.5, color='white')),
        name=f'x (iter {iter_select})',
        hovertemplate='Ant %{x}<br>Estimate: %{y:.4f}<extra></extra>'
    ), row=1, col=1)

    fig_conv.add_hline(y=1.0, line=dict(color='rgba(102,126,234,0.5)',
                                         width=1, dash='dash'),
                       row=1, col=1)
    fig_conv.add_hline(y=-1.0, line=dict(color='rgba(102,126,234,0.5)',
                                          width=1, dash='dash'),
                       row=1, col=1)

    colors_ant = px.colors.qualitative.Plotly[:Nt]
    for k in range(Nt):
        trajectory = [x_hist[i][k] for i in range(len(x_hist))]
        fig_conv.add_trace(go.Scatter(
            x=list(range(len(x_hist))),
            y=trajectory,
            mode='lines+markers',
            name=f'Ant {k+1} (true={x_true[k]:+.0f})',
            line=dict(color=colors_ant[k % len(colors_ant)], width=2),
            marker=dict(size=4),
            hovertemplate='Iter %{x}<br>x=%{y:.4f}<extra></extra>'
        ), row=1, col=2)

    fig_conv.add_vline(x=iter_select,
                       line=dict(color='yellow', width=2, dash='dot'),
                       row=1, col=2)

    for k in range(Nt):
        fig_conv.add_hline(y=x_true[k],
                           line=dict(color=colors_ant[k % len(colors_ant)],
                                     width=0.5, dash='dot'),
                           row=1, col=2)

    fig_conv.update_layout(
        template='plotly_dark',
        height=480,
        title=dict(
            text=f"ADMM Convergence — {Nr}×{Nt} BPSK, SNR={snr_conv} dB, "
                 f"Iteration {iter_select}/{admm_iter}",
            font=dict(size=14)
        ),
        margin=dict(t=70, b=40)
    )
    fig_conv.update_xaxes(title_text="Antenna Index", row=1, col=1)
    fig_conv.update_yaxes(title_text="Symbol Value", row=1, col=1)
    fig_conv.update_xaxes(title_text="ADMM Iteration", row=1, col=2)
    fig_conv.update_yaxes(title_text="Soft Estimate x_k", row=1, col=2)

    st.plotly_chart(fig_conv, use_container_width=True)

    mse_current = float(np.mean((x_current - x_true_arr) ** 2))
    correct = int(np.sum(np.sign(x_current) == x_true_arr))

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("MSE", f"{mse_current:.4f}")
    with col_m2:
        st.metric("Correct Symbols", f"{correct}/{Nt}")
    with col_m3:
        converged = "✅ Yes" if correct == Nt else "❌ No"
        st.metric("Fully Converged", converged)

# =============================================================================
# TAB 4: IDD LLR HEATMAP
# =============================================================================
with tab4:
    st.markdown('<div class="section-header">IDD Extrinsic LLR Heatmap</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="info-box">🔥 This heatmap shows the decoder\'s '
                'Log-Likelihood Ratios (LLRs) for each bit across IDD '
                'iterations. Blue = confident bit is 0 (+1 symbol), '
                'Red = confident bit is 1 (−1 symbol). Stronger colors = '
                'higher confidence. Watch the confidence grow!</div>',
                unsafe_allow_html=True)

    N_LDPC = 64
    RATE = 0.5

    with st.spinner("Running IDD loop..."):
        codeword, llr_history = run_idd_iterations(
            Nr, Nt, N_LDPC, RATE, idd_iters, admm_iter, 30,
            snr_idd, seed=int(conv_seed))

    llr_matrix = np.array(llr_history)
    codeword_arr = np.array(codeword)

    fig_heat = go.Figure()

    fig_heat.add_trace(go.Heatmap(
        z=llr_matrix,
        x=[f"Bit {i}" for i in range(N_LDPC)],
        y=[f"IDD Iter {t+1}" for t in range(idd_iters)],
        colorscale='RdBu',
        zmid=0,
        zmin=-15,
        zmax=15,
        colorbar=dict(title="LLR", titlefont=dict(size=12)),
        hovertemplate='%{x}<br>%{y}<br>LLR: %{z:.2f}<extra></extra>'
    ))

    fig_heat.update_layout(
        template='plotly_dark',
        height=350 + 30 * idd_iters,
        title=dict(
            text=f"Decoder LLR Confidence — LDPC({N_LDPC}, R={RATE}), "
                 f"SNR={snr_idd} dB, {idd_iters} IDD iterations",
            font=dict(size=14)
        ),
        xaxis=dict(title="Bit Position", tickangle=-45,
                    dtick=4, tickfont=dict(size=8)),
        yaxis=dict(title="IDD Iteration"),
        margin=dict(t=60, b=60)
    )

    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("**Mean Absolute LLR (Confidence) per IDD Iteration**")
    mean_conf = [float(np.mean(np.abs(llr_matrix[t])))
                 for t in range(idd_iters)]

    fig_conf = go.Figure()
    fig_conf.add_trace(go.Scatter(
        x=[f"Iter {t+1}" for t in range(idd_iters)],
        y=mean_conf,
        mode='lines+markers+text',
        text=[f"{c:.2f}" for c in mean_conf],
        textposition='top center',
        textfont=dict(size=11, color='#b0bec5'),
        line=dict(color='#667eea', width=3),
        marker=dict(size=10, color='#667eea',
                    line=dict(width=2, color='white')),
        hovertemplate='%{x}<br>Mean |LLR|: %{y:.2f}<extra></extra>'
    ))
    fig_conf.update_layout(
        template='plotly_dark',
        height=300,
        yaxis_title='Mean |LLR|',
        xaxis_title='IDD Iteration',
        margin=dict(t=30, b=40)
    )
    st.plotly_chart(fig_conf, use_container_width=True)

    st.markdown("**Bit Accuracy per IDD Iteration**")
    accuracies = []
    for t in range(idd_iters):
        decoded_bits = (np.array(llr_history[t]) < 0).astype(int)
        acc = float(np.mean(decoded_bits == codeword_arr))
        accuracies.append(acc)

    fig_acc = go.Figure()
    fig_acc.add_trace(go.Bar(
        x=[f"Iter {t+1}" for t in range(idd_iters)],
        y=[a * 100 for a in accuracies],
        marker=dict(
            color=[a * 100 for a in accuracies],
            colorscale=[[0, '#EF553B'], [0.5, '#FFA15A'],
                        [0.8, '#00CC96'], [1.0, '#636EFA']],
            showscale=False
        ),
        text=[f"{a*100:.1f}%" for a in accuracies],
        textposition='outside',
        textfont=dict(size=12, color='#b0bec5'),
        hovertemplate='%{x}<br>Accuracy: %{y:.1f}%<extra></extra>'
    ))
    fig_acc.update_layout(
        template='plotly_dark',
        height=300,
        yaxis_title='Bit Accuracy (%)',
        yaxis=dict(range=[0, 105]),
        xaxis_title='IDD Iteration',
        margin=dict(t=30, b=40)
    )
    st.plotly_chart(fig_acc, use_container_width=True)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#546e7a;font-size:0.85rem;">'
    'ADMM-LDPC-MIMO Visualizer · Built with Streamlit + Plotly · '
    f'{Nr}×{Nt} BPSK MIMO System</div>',
    unsafe_allow_html=True
)

