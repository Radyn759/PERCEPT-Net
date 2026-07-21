import numpy as np
import torch


def kspace_phase_perturbation(
        img_tensor,
        severity="medium",
        seed=None):

    """
    Simulate MRI motion artifact using k-space phase perturbation.

    Args:
        img_tensor:
            Tensor [C,H,W], intensity normalized to [0,1]

        severity:
            light / medium / heavy

    Returns:
        corrupted image
    """

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)


    C,H,W = img_tensor.shape

    severity_range = {

        "light":
            (0.02, 0.08),

        "medium":
            (0.08, 0.20),

        "heavy":
            (0.20, 0.40)

    }


    if severity not in severity_range:
        raise ValueError(
            "severity must be light/medium/heavy"
        )


    min_strength,max_strength = severity_range[severity]


    strength = torch.empty(1).uniform_(
        min_strength,
        max_strength
    ).item()

    kspace = torch.fft.fft2(
        img_tensor,
        dim=(-2,-1)
    )

    phase_noise = (
        torch.rand(
            1,H,W,
            device=img_tensor.device
        )
        *2-1
    )


    phase_noise *= np.pi * strength


    phase_mask = torch.exp(
        1j*phase_noise
    )


    corrupted_kspace = (
        kspace *
        phase_mask
    )


    corrupted_img = torch.fft.ifft2(
        corrupted_kspace,
        dim=(-2,-1)
    ).real


    corrupted_img=torch.clamp(
        corrupted_img,
        0,
        1
    )


    return corrupted_img, strength
