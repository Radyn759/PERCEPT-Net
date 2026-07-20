import numpy as np
import torch


def kspace_phase_perturbation(img_tensor, perturb_strength=0.15, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    C, H, W = img_tensor.shape
    kspace = torch.fft.fft2(img_tensor, dim=(-2, -1))

    rand_phase = (torch.rand(1, H, W, device=img_tensor.device) * 2 - 1) * np.pi * perturb_strength
    phase_mask = torch.exp(1j * rand_phase)

    perturbed_kspace = kspace * phase_mask

    perturbed_img = torch.fft.ifft2(perturbed_kspace, dim=(-2, -1)).real

    perturbed_img = torch.clamp(perturbed_img, 0.0, 1.0)
    return perturbed_img