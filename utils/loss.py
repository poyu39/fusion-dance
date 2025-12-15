import numpy as np
import torch
import torch.nn as nn


def rgb_to_hsv_torch(img, eps=1e-6):
    """
    自行實作 RGB→HSV，避免依賴 torchvision 版本；確保梯度可傳遞。
    """
    if img.shape[1] != 3:
        raise ValueError("RGB Tensor must have 3 channels.")
    r, g, b = img[:, 0], img[:, 1], img[:, 2]
    maxc, _ = img.max(dim=1)
    minc, _ = img.min(dim=1)
    deltac = maxc - minc

    hue = torch.zeros_like(maxc)
    sat = torch.zeros_like(maxc)
    val = maxc

    mask = deltac > eps
    safe_denom = torch.where(mask, deltac, torch.ones_like(deltac))

    # Hue 根據最大通道決定偏移
    hue_r = torch.remainder((g - b) / safe_denom, 6)
    hue_g = ((b - r) / safe_denom) + 2
    hue_b = ((r - g) / safe_denom) + 4

    hue = torch.where(mask & (maxc == r), hue_r, hue)
    hue = torch.where(mask & (maxc == g), hue_g, hue)
    hue = torch.where(mask & (maxc == b), hue_b, hue)
    hue = torch.remainder(hue / 6.0, 1.0)

    sat = torch.where(
        maxc <= eps,
        torch.zeros_like(maxc),
        deltac / (maxc + eps),
    )

    return torch.stack([hue, sat, val], dim=1)


def bits_per_dimension_loss(x_pred, x):
    nll = nn.functional.cross_entropy(x_pred, x, reduction="none")
    bpd = nll.mean(dim=[1, 2, 3]) * np.log2(np.exp(1))
    return bpd.mean()


def rmse_loss(reconstructed_x, x, use_sum=False, epsilon=1e-8):
    """
    We use epsilon to avoid NaN during backprop if mse = 0.
    Ref: https://discuss.pytorch.org/t/rmse-loss-function/16540/6
    """
    if use_sum:
        mse = nn.functional.mse_loss(reconstructed_x, x, reduction="sum")
    else:
        mse = nn.functional.mse_loss(reconstructed_x, x, reduction="mean")
    return torch.sqrt(mse + epsilon)


def mse_loss(reconstructed_x, x, use_sum=False):
    if use_sum:
        return nn.functional.mse_loss(reconstructed_x, x, reduction="sum")
    else:
        return nn.functional.mse_loss(reconstructed_x, x, reduction="mean")


def bce_loss(reconstructed_x, x, use_sum=False):
    if use_sum:
        return nn.functional.binary_cross_entropy(reconstructed_x, x, reduction="sum")
    else:
        return nn.functional.binary_cross_entropy(reconstructed_x, x, reduction="mean")


def crossentropy_loss(reconstructed_x, x, use_sum=False):
    if use_sum:
        return nn.functional.cross_entropy(reconstructed_x, x, reduction="sum")
    else:
        return nn.functional.cross_entropy(reconstructed_x, x, reduction="mean")


def kl_divergence(mu, log_var, use_sum=False):
    # Assumes a standard normal distribution for the 2nd gaussian
    inner_element = 1 + log_var - mu.pow(2) - log_var.exp()
    if use_sum:
        return -0.5 * torch.sum(inner_element)
    else:
        return -0.5 * torch.sum(inner_element, dim=1).mean()


def kl_divergence_two_gaussians(mu1, log_var1, mu2, log_var2, use_sum=False):
    # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    # Verify by setting mu2=torch.zeros((shape)), log_var2=torch.zeros((shape))
    # We use 0s for logvar since log 1 = 0.
    term1 = log_var1 - log_var2
    term2 = (log_var1.exp() + (mu1 - mu2).pow(2)) / log_var2.exp()
    if use_sum:
        kl_d = -0.5 * torch.sum(term1 - term2 + 1)
    else:
        kl_d = -0.5 * torch.sum(term1 - term2 + 1, dim=1).mean()
    return kl_d, {"KL Divergence": kl_d.item()}


def mse_ssim_loss(
    reconstructed_x, x, use_sum=False, ssim_module=None, mse_weight=1, ssim_weight=1
):
    mse = mse_weight * mse_loss(reconstructed_x, x, use_sum)
    if ssim_module:
        # ssim gives a score from 0-1 where 1 is the highest.
        # So we do 1 - ssim in order to minimize it.
        ssim = ssim_weight * (1 - ssim_module(reconstructed_x, x))
    else:
        ssim = torch.tensor(0.0, device=reconstructed_x.device)
    return mse + ssim, {"MSE": mse.item(), "SSIM": ssim.item()}


def palette_aware_loss(
    reconstructed_x,
    x,
    use_sum=False,
    hue_weight=1.0,
    saturation_weight=0.5,
    value_weight=0.25,
):
    """
    將 RGB 映射到 HSV 後比較色相、飽和度與明度，避免純粹用 L2 無法感知色票差異。
    hue_weight / saturation_weight / value_weight 用來控制三個通道的重要性。
    """
    reconstructed = torch.clamp(reconstructed_x, 0.0, 1.0)
    target = torch.clamp(x, 0.0, 1.0)
    reconstructed_hsv = rgb_to_hsv_torch(reconstructed)
    target_hsv = rgb_to_hsv_torch(target)
    hue_diff = torch.remainder(
        reconstructed_hsv[:, 0:1] - target_hsv[:, 0:1] + 0.5, 1.0
    ) - 0.5  # 讓色相差落在 [-0.5, 0.5]
    sat_diff = reconstructed_hsv[:, 1:2] - target_hsv[:, 1:2]
    val_diff = reconstructed_hsv[:, 2:3] - target_hsv[:, 2:3]
    color_distance = (
        hue_weight * torch.abs(hue_diff)
        + saturation_weight * torch.abs(sat_diff)
        + value_weight * torch.abs(val_diff)
    )
    if use_sum:
        return color_distance.sum()
    return color_distance.mean()


def pixel_aware_reconstruction_loss(
    reconstructed_x,
    x,
    use_sum=False,
    ssim_module=None,
    mse_weight=1.0,
    ssim_weight=1.0,
    palette_weight=1.0,
    palette_kwargs=None,
):
    """
    Pixel-aware loss = α·MSE + β·(1-SSIM) + γ·Palette loss。
    palette_kwargs 允許針對色票距離設定不同權重，方便後續做消融。
    """
    loss_total = torch.tensor(0.0, device=reconstructed_x.device)
    palette_kwargs = palette_kwargs or {}

    mse = mse_weight * mse_loss(reconstructed_x, x, use_sum)
    loss_total = loss_total + mse

    if ssim_module and ssim_weight != 0:
        ssim = ssim_weight * (1 - ssim_module(reconstructed_x, x))
    else:
        ssim = torch.tensor(0.0, device=reconstructed_x.device)
    loss_total = loss_total + ssim

    if palette_weight != 0:
        palette = palette_weight * palette_aware_loss(
            reconstructed_x,
            x,
            use_sum=use_sum,
            **palette_kwargs,
        )
    else:
        palette = torch.tensor(0.0, device=reconstructed_x.device)
    loss_total = loss_total + palette

    return loss_total, {
        "MSE": mse.item(),
        "SSIM": ssim.item(),
        "Palette": palette.item(),
    }


def VAE_loss(
    reconstructed_x,
    x,
    mu,
    log_var,
    use_sum=True,
    ssim_module=None,
    mse_weight=1,
    ssim_weight=1,
    reconstruction_weight=1,
    kl_weight=1,
):
    mse_ssim, loss_dict = mse_ssim_loss(
        reconstructed_x,
        x,
        use_sum,
        ssim_module=ssim_module,
        mse_weight=mse_weight,
        ssim_weight=ssim_weight,
    )
    KL_d = kl_divergence(mu, log_var, use_sum)
    weighted_loss = (reconstruction_weight * mse_ssim) + (kl_weight * KL_d)
    return weighted_loss, {
        "MSE": loss_dict["MSE"],
        "SSIM": loss_dict["SSIM"],
        "KL Divergence": KL_d.item(),
    }
