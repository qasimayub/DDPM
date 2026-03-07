import torch
import torch.nn.functional as F
from utils import make_beta_schedule

class DiffusionProcess:
    def __init__(self, L, device="cpu", beta_schedule_type="linear"):
        self.L = L
        self.device = device

        self.betas = make_beta_schedule(L, type=beta_schedule_type).to(device)

        self.alphas = (1.0 - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(device)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas).to(device)

        # beta_tilde_i = (1 - alpha_bar_{i-1}) / (1 - alpha_bar_i) * beta_i
        alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(device) # pad with a 0 at the start because for i=0, there is no i-1.

        self.posterior_variance = self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod).to(device)

        # mu_tilde = (sqrt(alpha_bar_{i-1}) * beta_i / (1 - alpha_bar_i)) * x_0 + (sqrt(alpha_{i}) * (1 - alpha_bar_{i-1}) / (1 - alpha_bar_i)) * x_i
        self.posterior_mean_coef1 = self.betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - self.alphas_cumprod).to(device)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod).to(device)
        
    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.L, (batch_size,), device=self.device)

    def q_sample(self, x0, t, eps=None):
        if eps is None:
            eps = torch.randn_like(x0)

        sqrt_alpha_bar = self.extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * eps

    def extract(self, v, t, x_shape):
        """Helper to extract t-index values and reshape for broadcasting"""
        # Pull the values for the specific timesteps in 't'
        out = v.gather(-1, t)
        # Add (1, 1, 1) to the end of the shape
        # len(x_shape) - 1 calculates how many '1s' we need (usually 3 for images)
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))
    
    def q_posterior_mean_var(self, x0, xi, t):
        """Computes the mean and variance of the true posterior q(xi-1 | xi, x0)"""
        coef1 = self.extract(self.posterior_mean_coef1, t, xi.shape)
        coef2 = self.extract(self.posterior_mean_coef2, t, xi.shape)
        
        mu_tilde = coef1 * x0 + coef2 * xi
        posterior_var = self.extract(self.posterior_variance, t, xi.shape)
        
        return mu_tilde, posterior_var

    def predict_x0_from_eps(self, xi, t, eps_hat):
        """Predicts x0 from current noisy image and predicted noise"""
        sqrt_recip_alpha_bar = self.extract(torch.sqrt(1.0 / self.alphas_cumprod), t, xi.shape)
        sqrt_eps_coef = self.extract(torch.sqrt(1.0 / self.alphas_cumprod - 1), t, xi.shape)
        
        return sqrt_recip_alpha_bar * xi - sqrt_eps_coef * eps_hat

    def p_mean_from_eps(self, xi, t, eps_hat):
        """Computes the mean of p(xi-1 | xi) using the model's noise prediction (Equation 8)"""
        coef = self.extract(self.betas / self.sqrt_one_minus_alphas_cumprod, t, xi.shape)
        mu = self.extract(self.sqrt_recip_alphas, t, xi.shape) * (xi - coef * eps_hat)
        return mu

    def p_sample_step(self, xi, t, eps_hat):
        """The core reverse step: samples xi-1 given xi and the model's noise prediction"""
        mu = self.p_mean_from_eps(xi, t, eps_hat)

        if (t > 0).all():
            z = torch.randn_like(xi)
            sigma = torch.sqrt(self.extract(self.posterior_variance, t, xi.shape))
            return mu + sigma * z
        else:
            return mu