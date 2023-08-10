from .module import Module
from ..autodiff import *
from .helper import compute_out_dims_for_pooling_ops
class Conv(Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int],
        stride: tuple[int],
        padding: tuple[tuple],
        bias: bool = True
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.weight = Tensor.random((out_channels, in_channels, *kernel_size))
        self.params.append(self.weight)
        if bias:
            self.b = Tensor.random((out_channels,))
            self.params.append(self.b)

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        #NOTE this is a general implementation and works for 1d,2d,3d
        padded_x = x.pad(((0,0), (0,0), *self.padding)) if any(pad > 0 for tup in self.padding for pad in tup) else x
        input_shape = tuple(x.shape[-i] for i in reversed(range(1, len(x.shape) - 1)))
        out_dim = compute_out_dims_for_pooling_ops(*input_shape, padding=self.padding, kernel_size=self.kernel_size, stride=self.stride)
        window = padded_x.rolling_window((padded_x.shape[0], padded_x.shape[1], *self.kernel_size), self.stride)
        out = window.reshape(out_dim + (padded_x.shape[0], padded_x.shape[1] * np.prod(self.kernel_size))) @ self.weight.flatten(1).T
        out_order = tuple(i for i in range(len(out.shape) -2, len(out.shape))) + tuple(i for i in range(len(out.shape) -2))
        out = out.permute((out_order))
        return out + self.b.reshape((-1,) + tuple(1 for i in range(len(out.shape) - 2 ))) if self.bias else out
    
#NOTE: All Conv1d, Conv2d, Conv3d are aliases for Conv with some checks in their constructor to support syntactic sugar
class Conv1d(Conv):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int], stride: tuple[int] = (1,1,1), padding: tuple[tuple] = ((0,0),), bias: bool = True) -> None:
        if isinstance(stride, int):
            stride = (stride,)
        if len(stride) != 3:
            stride = (1,1) + stride
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,)
        if isinstance(padding, int):
            padding = ((padding, padding), )
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias)

class Conv2d(Conv):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int], stride: tuple[int] = (1,1,1,1), padding: tuple[tuple] = ((0,0), (0,0)), bias: bool = True) -> None:
        if isinstance(stride, int):
            stride = (stride,) * 2
        if len(stride) != 4:
            stride = (1,1) + stride
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias)

class Conv3d(Conv):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int], stride: tuple[int] = (1,1,1,1,1), padding: tuple[tuple] = ((0,0), (0,0), (0,0)), bias: bool = True) -> None:
        if isinstance(stride, int):
            stride = (stride,) * 3
        if len(stride) != 5:
            stride = (1,1) + stride
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias)