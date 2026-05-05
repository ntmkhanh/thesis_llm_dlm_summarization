import torch


class DiffusionSchedule:
    """DDPM schedule + q/p samplers for continuous latent diffusion."""

    def __init__(self, timesteps: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02, device: str = "cpu"):
        self.timesteps = timesteps
        self.device = torch.device(device)

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32, device=self.device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    def to(self, device: torch.device):
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: q(z_t|z_0)."""
        a_bar = self.alpha_bars[t].view(-1, 1, 1)
        return torch.sqrt(a_bar) * z0 + torch.sqrt(1.0 - a_bar) * noise

    def predict_z0_from_eps(self, zt: torch.Tensor, t: torch.Tensor, eps_hat: torch.Tensor) -> torch.Tensor:
        a_bar = self.alpha_bars[t].view(-1, 1, 1)
        return (zt - torch.sqrt(1.0 - a_bar) * eps_hat) / torch.sqrt(a_bar)

    def p_sample_step(self, zt: torch.Tensor, t_scalar: int, eps_hat: torch.Tensor) -> torch.Tensor:
        """Deterministic reverse step using predicted epsilon."""
        t = torch.tensor([t_scalar], device=zt.device)
        z0_hat = self.predict_z0_from_eps(zt, t, eps_hat)
        if t_scalar > 0:
            a_bar_prev = self.alpha_bars[t - 1].view(1, 1, 1)
            return torch.sqrt(a_bar_prev) * z0_hat + torch.sqrt(1.0 - a_bar_prev) * eps_hat
        return z0_hat
