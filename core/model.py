import numpy as np


class LatentValueModel:
    """
    Latent value process and signal generation

    V_t = V_{t-1} + eps_t
    eps_t ~ N(0, sigma_V^2)

    Agents observe delayed noisy versions of V_t
    """

    def __init__(self, T, sigma, start_price=100.0, seed=None):
        self.T = T
        self.sigma = sigma
        self.start_price = start_price

        self.rng = np.random.default_rng(seed or 65)

        self.V = self._generate_latent_value()

    def _generate_latent_value(self):
        """Generate latent value signals"""
        V = np.empty(self.T)
        V[0] = self.start_price

        eps = self.rng.normal(0.0, self.sigma, size=self.T)

        for t in range(1, self.T):
            V[t] = V[t - 1] + eps[t]

        return V
    
    def observe(self, t, latency, noise_std):
        """Observe delayed noisy signal"""
        idx = max(0, t - latency)
        return self.V[idx] + self.rng.normal(0.0, noise_std)