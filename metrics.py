import numpy as np
from skimage.metrics import structural_similarity as ssim


def calculate_ssim(image1, image2):
    return -ssim(image1, image2, channel_axis=-1)


def calculate_mse(image1, image2):
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)

    mse = np.mean((image1 - image2) ** 2)

    return mse
