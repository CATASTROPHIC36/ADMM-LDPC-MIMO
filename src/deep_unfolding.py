"""
Deep Unfolded ADMM MIMO Detector

This module provides both NumPy-based and PyTorch-based implementations
of a deep unfolded Alternating Direction Method of Multipliers (ADMM)
detector. Unfolding allows the penalty parameters (rho) to be learned
via backpropagation to improve convergence and detection performance.
"""

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# =============================================================================
# NUMPY UNFOLDING IMPLEMENTATION
# =============================================================================

class ADMMUnfoldedNumpy:
    """
    NumPy-based deep unfolded ADMM detector.
    Allows for setting layer-wise penalty parameters manually or via
    numerical gradient calculation for testing.
    """

    def __init__(self, K: int = 10, rho_init: float = 1.0):

        self.K   = K
        self.rho = np.full(K, rho_init, dtype=float)

    def forward(
        self,
        y: np.ndarray,
        H: np.ndarray,
        sigma2: float,
        M: int = 2
    ) -> np.ndarray:

        from src.admm_detector import project_constellation

        Nr, Nt = H.shape
        HH     = H.conj().T @ H
        Hhy    = H.conj().T @ y / sigma2

        z = np.zeros(Nt, dtype=float if M == 2 else complex)
        u = np.zeros_like(z)

        for k in range(self.K):
            rho_k = float(self.rho[k])

            A   = HH / sigma2 + rho_k * np.eye(Nt)
            rhs = Hhy + rho_k * (z - u)
            x   = np.linalg.solve(A, rhs)
            if M == 2:
                x = np.real(x)

            z = project_constellation(x + u, M=M)

            u = u + x - z

        return z

    def loss_mse(
        self,
        y: np.ndarray,
        H: np.ndarray,
        sigma2: float,
        x_true: np.ndarray,
        M: int = 2
    ) -> float:

        z = self.forward(y, H, sigma2, M)
        return float(np.mean((np.real(z) - np.real(x_true)) ** 2))

    def numerical_gradient(
        self,
        y: np.ndarray,
        H: np.ndarray,
        sigma2: float,
        x_true: np.ndarray,
        M: int = 2,
        eps: float = 1e-5
    ) -> np.ndarray:

        grad = np.zeros(self.K)
        for k in range(self.K):
            rho_orig     = self.rho[k]
            self.rho[k]  = rho_orig + eps
            L_plus       = self.loss_mse(y, H, sigma2, x_true, M)
            self.rho[k]  = rho_orig - eps
            L_minus      = self.loss_mse(y, H, sigma2, x_true, M)
            grad[k]      = (L_plus - L_minus) / (2 * eps)
            self.rho[k]  = rho_orig
        return grad

# =============================================================================
# PYTORCH UNFOLDING IMPLEMENTATION
# =============================================================================

if TORCH_AVAILABLE:

    class ADMMUnfoldedLayer(nn.Module):
        """
        A single layer of the deep unfolded ADMM network.
        Learns the penalty parameter 'rho' for its specific iteration.
        """

        def __init__(self, rho_init: float = 1.0):
            super().__init__()

            self.log_rho = nn.Parameter(
                torch.tensor(float(np.log(rho_init)), dtype=torch.float64)
            )

        @property
        def rho(self) -> torch.Tensor:
            return torch.exp(self.log_rho)

        def forward(
            self,
            x: torch.Tensor,
            z: torch.Tensor,
            u: torch.Tensor,
            HH: torch.Tensor,
            Hhy: torch.Tensor,
            sigma2: float,
            Nt: int
        ) -> tuple:

            rho = self.rho.to(HH.dtype)

            I_batch = torch.eye(Nt, dtype=HH.dtype, device=HH.device).unsqueeze(0)
            A = HH / sigma2 + rho * I_batch

            rhs = Hhy + rho * (z - u).to(HH.dtype)

            x_complex = torch.linalg.solve(A, rhs)
            x_new     = x_complex.real

            v = x_new + u
            z_hard = torch.sign(v)
            z_hard = torch.where(z_hard == 0, torch.ones_like(z_hard), z_hard)

            z_new = v + (z_hard - v).detach()

            u_new = u + x_new - z_hard.detach()

            return x_new, z_new, u_new

    class ADMMUnfoldedNet(nn.Module):
        """
        Complete deep unfolded ADMM network constructed from multiple
        ADMMUnfoldedLayer instances.
        """

        def __init__(self, K: int = 10, rho_init: float = 1.0):

            super().__init__()
            self.K      = K
            self.layers = nn.ModuleList([
                ADMMUnfoldedLayer(rho_init=rho_init) for _ in range(K)
            ])

        def get_rho_values(self) -> np.ndarray:

            return np.array([
                float(layer.rho.item()) for layer in self.layers
            ])

        def forward(
            self,
            y: torch.Tensor,
            H: torch.Tensor,
            sigma2: float
        ) -> torch.Tensor:

            batch, Nr, Nt = H.shape

            H_conj_T = H.conj().transpose(-2, -1)
            HH       = torch.bmm(H_conj_T, H)
            Hhy      = torch.bmm(H_conj_T, y.unsqueeze(-1)).squeeze(-1) / sigma2

            x = torch.zeros(batch, Nt, dtype=torch.float64, device=y.device)
            z = torch.zeros(batch, Nt, dtype=torch.float64, device=y.device)
            u = torch.zeros(batch, Nt, dtype=torch.float64, device=y.device)

            for k, layer in enumerate(self.layers):
                x, z, u = layer(x, z, u, HH, Hhy, sigma2, Nt)

            return z

# =============================================================================
# TRAINING AND EVALUATION ROUTINES
# =============================================================================

def generate_training_batch(
    batch_size: int,
    Nr: int,
    Nt: int,
    snr_db: float,
    rng: np.random.Generator = None
):

    from src.channel import generate_channel

    if rng is None:
        rng = np.random.default_rng()

    snr_lin  = 10.0 ** (snr_db / 10.0)
    sigma2   = Nt / (Nr * snr_lin)
    noise_std = np.sqrt(sigma2 / 2.0)

    scale    = 1.0 / np.sqrt(2 * Nt)
    H_r      = rng.standard_normal((batch_size, Nr, Nt)) * scale
    H_i      = rng.standard_normal((batch_size, Nr, Nt)) * scale
    H_batch  = H_r + 1j * H_i

    bits     = rng.integers(0, 2, size=(batch_size, Nt))
    x_batch  = (2.0 * bits - 1.0).astype(float)

    Hx       = np.einsum('bij,bj->bi', H_batch, x_batch)
    noise    = noise_std * (
        rng.standard_normal((batch_size, Nr)) +
        1j * rng.standard_normal((batch_size, Nr))
    )
    y_batch  = Hx + noise

    return y_batch, H_batch, x_batch, sigma2

def train_admm_unfolded(
    Nr: int,
    Nt: int,
    K: int = 10,
    snr_db_train: float = 10.0,
    n_epochs: int = 500,
    batch_size: int = 64,
    lr: float = 5e-3,
    seed: int = 42,
    verbose: bool = True
):

    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for training. "
            "Install with: pip install torch"
        )

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model     = ADMMUnfoldedNet(K=K, rho_init=1.0).double()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()
    loss_hist = []

    for epoch in range(n_epochs):

        y_np, H_np, x_np, sigma2 = generate_training_batch(
            batch_size, Nr, Nt, snr_db_train, rng
        )

        y_t = torch.tensor(y_np, dtype=torch.complex128)
        H_t = torch.tensor(H_np, dtype=torch.complex128)
        x_t = torch.tensor(x_np, dtype=torch.float64)

        optimiser.zero_grad()

        z_out = model(y_t, H_t, sigma2)
        loss  = loss_fn(z_out, x_t)
        loss.backward()
        optimiser.step()

        loss_hist.append(float(loss.item()))

        if verbose and (epoch + 1) % 50 == 0:
            rho_vals = model.get_rho_values()
            print(f"  Epoch {epoch+1:4d}/{n_epochs} | "
                  f"Loss={loss.item():.6f} | "
                  f"ρ: [{', '.join(f'{r:.3f}' for r in rho_vals[:3])}...]")

    return model, loss_hist

def evaluate_unfolded_ber(
    model,
    Nr: int,
    Nt: int,
    snr_db_range: np.ndarray,
    n_trials: int = 500,
    seed: int = 99
):

    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required.")

    from src.admm_detector import admm_detect

    rng            = np.random.default_rng(seed)
    model.eval()
    K              = model.K
    ber_unfolded   = np.zeros(len(snr_db_range))
    ber_classical  = np.zeros(len(snr_db_range))

    with torch.no_grad():
        for i, snr_db in enumerate(snr_db_range):
            y_np, H_np, x_np, sigma2 = generate_training_batch(
                n_trials, Nr, Nt, snr_db, rng
            )

            y_t  = torch.tensor(y_np, dtype=torch.complex128)
            H_t  = torch.tensor(H_np, dtype=torch.complex128)
            z_out = model(y_t, H_t, sigma2).numpy()
            bits_tx   = ((1 - x_np) / 2).astype(int)
            bits_unfold = (z_out < 0).astype(int)
            ber_unfolded[i] = float(np.mean(bits_tx != bits_unfold))

            err_cls = 0
            for b in range(n_trials):
                z_cls, _ = admm_detect(
                    y_np[b], H_np[b], sigma2, M=2,
                    rho=1.0, max_iter=K, return_soft=True
                )
                bits_cls = (np.real(z_cls) < 0).astype(int)
                err_cls += int(np.sum(bits_tx[b] != bits_cls))
            ber_classical[i] = err_cls / (n_trials * Nt)

            print(f"  SNR={snr_db:4.1f} dB | "
                  f"Unfolded={ber_unfolded[i]:.5f} | "
                  f"Classical={ber_classical[i]:.5f}")

    return ber_unfolded, ber_classical


