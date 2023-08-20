from yadll.autodiff import Tensor
from .helper import compute_out_dims_for_pooling_ops
from .module import Module
from typing import Callable, Any
import numpy as np

class Pool(Module):
    def __init__(self, kernel_size: tuple, stride: tuple, padding: tuple[tuple], pad_value=0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,)
        self.stride = stride
        self.padding = padding
        self.pad_value = pad_value

    def forward(self, x: Tensor, op: Callable = None,*args, **kwargs) -> Tensor:
        padded_x = x.pad(((0,0), (0,0), *self.padding), self.pad_value) if any(pad > 0 for tup in self.padding for pad in tup) else x
        input_shape = tuple(x.shape[-i] for i in reversed(range(1, len(x.shape) - 1)))
        out_dim = compute_out_dims_for_pooling_ops(*input_shape, padding=self.padding, kernel_size=self.kernel_size, stride=self.stride)
        window = padded_x.rolling_window((padded_x.shape[0], padded_x.shape[1], *self.kernel_size), self.stride)
        out = op(window, *args, **kwargs).reshape(out_dim + (x.shape[0], x.shape[1]))
        out_order = tuple(i for i in range(len(out.shape) -2, len(out.shape))) + tuple(i for i in range(len(out.shape) -2))
        return out.permute(out_order)
    
def max_pool(x: Tensor, *args, **kwargs):
    n = kwargs['n']
    out = x
    for _ in range(n):
        out = out.max(-1)
    return out

def avg_pool(x: Tensor, *args, **kwargs):
    kernel_size = kwargs['kernel_size']
    dim = kwargs['dim']
    return x.sum(dim) / float(np.prod(kernel_size))

class AvgPool1d(Pool):
    def __init__(self, kernel_size: tuple, stride: tuple = None, padding: tuple[tuple] = ((0,0),)) -> None:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,)
        stride = stride if stride else kernel_size
        if isinstance(stride, int):
            stride = (stride,)
        if len(stride) != 3:
            stride = (1,1) + stride
        super().__init__(kernel_size, stride, padding)

    def forward(self, x: Tensor, op: Callable = None, *args, **kwargs) -> Tensor:
        return super().forward(x, op=avg_pool,dim=(-1,), kernel_size = self.kernel_size)
    
class AvgPool2d(Pool):
    def __init__(self, kernel_size: tuple, stride: tuple = None, padding: tuple[tuple] = ((0,0), (0,0))) -> None:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        stride = stride if stride else kernel_size
        if isinstance(stride, int):
            stride = (stride,)
        if len(stride) != 4:
            stride = (1,1) + stride
        super().__init__(kernel_size, stride, padding)

    def forward(self, x: Tensor, op: Callable[..., Any] = None, *args, **kwargs) -> Tensor:
        return super().forward(x, avg_pool,dim=(-1,-2), kernel_size = self.kernel_size)

class AvgPool3d(Pool):
    def __init__(self, kernel_size: tuple, stride: tuple = None, padding: tuple[tuple] = ((0,0), (0,0), (0,0))) -> None:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        stride = stride if stride else kernel_size
        if isinstance(stride, int):
            stride = (stride,)
        if len(stride) != 5:
            stride = (1,1) + stride
        super().__init__(kernel_size, stride, padding)

    def forward(self, x: Tensor, op: Callable[..., Any] = None, *args, **kwargs) -> Tensor:
        return super().forward(x, avg_pool,dim=(-1,-2, -3), kernel_size = self.kernel_size)

class MaxPool1d(Pool):
    def __init__(self, kernel_size: tuple, stride: tuple = None, padding: tuple[tuple] = ((0,0),)) -> None:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,)
        stride = stride if stride else kernel_size
        if isinstance(stride, int):
            stride = (stride,) * 3
        if len(stride) < 3:
            stride = (1,1) + stride
        super().__init__(kernel_size, stride, padding, -np.inf)

    def forward(self, x: Tensor, op: Callable = None, *args, **kwargs) -> Tensor:
        return super().forward(x, op=max_pool,n=1)
    
class MaxPool2d(Pool):
    def __init__(self, kernel_size: tuple, stride: tuple = None, padding: tuple[tuple] = ((0,0),(0,0))) -> None:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,)
        stride = stride if stride else kernel_size
        if isinstance(stride, int):
            stride = (stride,) * 4
        if len(stride) != 4:
            stride = (1,1) + stride
        super().__init__(kernel_size, stride, padding, -np.inf)

    def forward(self, x: Tensor, op: Callable = None, *args, **kwargs) -> Tensor:
        return super().forward(x, op=max_pool,n=2)
    
class MaxPool3d(Pool):
    def __init__(self, kernel_size: tuple, stride: tuple = None, padding: tuple[tuple] = ((0,0),(0,0), (0,0))) -> None:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,)
        stride = stride if stride else kernel_size
        if isinstance(stride, int):
            stride = (stride,)
        if len(stride) != 5:
            stride = (1,1) + stride
        super().__init__(kernel_size, stride, padding, -np.inf)

    def forward(self, x: Tensor, op: Callable = None, *args, **kwargs) -> Tensor:
        return super().forward(x, op=max_pool,n=3)