import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid


def show_image(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation="nearest")
    plt.show()


def show_images(imgs: list[torch.Tensor], n=6):
    grid = make_grid(imgs)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.tight_layout()
    plt.show()
