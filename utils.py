import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Subset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from models.position_embedding import SinusoidalPositionEmbeddings
from models.unet import UNet


def visualise_loader(imgs, count, grid=3):
    plt.figure(figsize=(5, 4))
    for _ in range(count):
        plt.subplot(grid, grid, _ + 1)
        plt.imshow(imgs[_].permute(1, 2, 0), cmap="gray")
        plt.axis("off")
    plt.show() 

def make_beta_schedule(L, type='linear', beta_min=1e-4, beta_max=0.02):
    if type == 'linear':
        return torch.linspace(beta_min, beta_max, L)
    else:
        return torch.linspace(beta_min**0.5, beta_max**0.5, L) ** 2

def periodic_sample_grids(model, diff, count, device, grid=3):
    model.eval()
    model.to(device)
    with torch.no_grad():
        xi = torch.randn((count, 1, 28, 28), device=device)
        for i in reversed(range(L)):
            t_tensor = torch.full((count,), i, device=device, dtype=torch.long)
            eps_hat = model(xi, t_tensor)
            xi = diff.p_sample_step(xi, t_tensor, eps_hat)
    model.train()
    plt.figure(figsize=(5, 4))
    for _ in range(count):
        img_to_plot = (xi[_] + 1.0) / 2.0 
        img_to_plot = torch.clamp(img_to_plot, 0.0, 1.0)
        plt.subplot(grid, grid, _ + 1)
        plt.imshow(img_to_plot.to("cpu").permute(1, 2, 0), cmap="gray")
        plt.axis("off")
    plt.show() 

def ancestral_sample(model, diff, count, device):
    model.eval()
    xi = torch.randn((count, 1, 28, 28), device=device)
    L = diff.L
    norms = []
    intermediates = {}
    save_points = {L-1, 3*L//4, L//4, L//8, L//16, 0}

    with torch.no_grad():
        for idx in reversed(range(L)):
            t_tensor = torch.full((count,), idx, device=device, dtype=torch.long)

            eps_hat = model(xi, t_tensor)
            xi = diff.p_sample_step(xi, t_tensor, eps_hat)

            norms.append(torch.norm(xi).item())
            if idx in save_points:
                intermediates[idx] = xi.clone()

    return xi, intermediates, norms

def plot_ancestral_samples(intermediates, count):
    timesteps = sorted(intermediates.keys(), reverse=True)
    num_cols = len(timesteps)

    plt.figure(figsize=(2 * num_cols, 2 * count))

    for row in range(count):
        for col, t in enumerate(timesteps):
            plt.subplot(count, num_cols, row * num_cols + col + 1)
            img = intermediates[t][row].squeeze().cpu().numpy()

            img = (img + 1.0) / 2.0
            img = img.clip(0, 1)
            
            plt.imshow(img, cmap="gray")
            plt.axis("off")

            if row == 0:
                if t == 1:
                    plt.title("Final (t=1)")
                else:
                    plt.title(f"t={t}")

    plt.tight_layout()
    plt.show()

def plot_sampling_norms(norms, L):
    timesteps = list(range(L, 0, -1))
    
    plt.figure(figsize=(8, 4))
    plt.plot(timesteps, norms, color="blue", linewidth=2, label="$||x_i||$")

    plt.gca().invert_xaxis()
    
    plt.title("Norm of $x_i$ During Reverse Denoising")
    plt.xlabel("Timestep $i$ (L $\\rightarrow$ 1)")
    plt.ylabel("L2 Norm")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def check_posterior_logic(diff, loader, device, i_val=100):
    x0, _ = next(iter(loader))
    x0 = x0.to(device)
    batch_size = x0.shape[0]
    
    i = torch.full((batch_size,), i_val, device=device, dtype=torch.long)
    eps = torch.randn_like(x0)
    xi = diff.q_sample(x0, i, eps)

    mu_tilde, var_tilde = diff.q_posterior_mean_var(x0, xi, i)
    xi_minus_1 = mu_tilde + torch.sqrt(var_tilde) * torch.randn_like(xi)

    dist_i = torch.mean((xi - x0)**2).item()
    dist_prev = torch.mean((xi_minus_1 - x0)**2).item()
    
    print(f"Distance at i={i_val}: {dist_i:.4f}")
    print(f"Distance at i-1: {dist_prev:.4f}")
    return dist_prev < dist_i

def verify_training_inputs(diff, batch_size=10000):
    t = diff.sample_timesteps(batch_size).cpu().numpy()
    plt.hist(t, bins=50)
    plt.title("Timestep Distribution (Should be flat/Uniform)")
    plt.show()

def check_noise_correlation(model, diff, x0):
    t = diff.sample_timesteps(x0.shape[0]).to(device)
    model.eval()
    with torch.no_grad():
        eps = torch.randn_like(x0)
        xi = diff.q_sample(x0, t, eps)
        eps_hat = model(xi, t)
        correlation = torch.corrcoef(torch.stack([eps.flatten(), eps_hat.flatten()]))[0, 1]
    model.train()
    return correlation.item()

def strided_ancestral_sample(model, diff, count, device, stride=1):
    model.eval()
    xi = torch.randn((count, 1, 28, 28), device=device)
    timesteps = list(range(diff.L - 1, -1, -stride))
    with torch.no_grad():
        for idx in timesteps:
            t_tensor = torch.full((count,), idx, device=device, dtype=torch.long)
            eps_hat = model(xi, t_tensor)
            xi = diff.p_sample_step(xi, t_tensor, eps_hat)
    model.train()
    return xi

def ddim_sample_step(diff, xi, t_tensor, t_prev_tensor, eps_hat):
    alpha_bar_t = diff.extract(diff.alphas_cumprod, t_tensor, xi.shape)
    valid_mask = (t_prev_tensor >= 0).view(-1, 1, 1, 1)
    alpha_bar_t_prev = torch.where(
        valid_mask,
        diff.extract(diff.alphas_cumprod, t_prev_tensor.clamp(min=0), xi.shape),
        torch.ones_like(alpha_bar_t)
    )
    pred_x0 = (xi - torch.sqrt(1.0 - alpha_bar_t) * eps_hat) / torch.sqrt(alpha_bar_t)
    dir_xt = torch.sqrt(1.0 - alpha_bar_t_prev) * eps_hat
    
    # sqrt(alpha_bar_t_prev) * x0 + sqrt(1-alpha_bar_t_prev) * eps_hat
    # we ignore sigma term to make it deterministic
    x_prev = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt
    return x_prev

def strided_ddim_sample(model, diff, count, device, stride=1):
    model.eval()
    xi = torch.randn((count, 1, 28, 28), device=device)
    timesteps = list(range(diff.L - 1, -1, -stride))
    with torch.no_grad():
        for i, idx in enumerate(timesteps):
            t_tensor = torch.full((count,), idx, device=device, dtype=torch.long)
            if i < len(timesteps) - 1:
                idx_prev = timesteps[i + 1]
            else:
                idx_prev = -1
            t_prev_tensor = torch.full((count,), idx_prev, device=device, dtype=torch.long)
            eps_hat = model(xi, t_tensor)
            xi = ddim_sample_step(diff, xi, t_tensor, t_prev_tensor, eps_hat)
    return xi