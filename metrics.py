import torch


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 1]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return -10 * torch.log10(mse)


class ResidualLoss:
    def __init__(self):
        self.name = "ResidualLoss"

    @staticmethod
    def __call__(low_res, predicted, ground_truth):
        residual = ground_truth - low_res
        loss = torch.mean((residual - predicted)**2)
        return loss
