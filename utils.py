import numpy as np 
import torch

from PIL import Image
from torchvision.transforms import ToPILImage

to_PIL = ToPILImage()

def to_Image(tensor, show=None, normalize=False):
    if normalize:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    tensor = tensor.cpu()
    image = to_PIL(tensor)
    if show:
        image.save(show + ".png")
    return image

def show_heatmap(tensor, Radius):
    tensor = tensor.cpu()
    n, c, h, w = tensor.shape
    out, coords = tensor.view(n, c, -1).topk(dim=-1, \
        k=int(3.14 * Radius * Radius))
    import ipdb; ipdb.set_trace()