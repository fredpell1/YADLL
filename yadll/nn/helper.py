from ..autodiff import *
import numpy as np


def compute_out_dims_for_pooling_ops(*dims, padding: tuple[tuple], kernel_size, stride):
    pad = [sum(pad) for pad in padding]
    out_dims = tuple(
        int(np.floor((dim + pad[i] - kernel_size[i]) / stride[i + 2] + 1))
        for i, dim in enumerate(dims)
    )
    return out_dims
