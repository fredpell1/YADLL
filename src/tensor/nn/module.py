from src.tensor.autodiff import Tensor
from src.tensor.nn.functional import Tensor
from ..autodiff import *
from .functional import *
from typing import Any, List, Generator


class Module:
    def __init__(self) -> None:
        self.params: List[Tensor] = []

    def parameters(self) -> Generator[Tensor, Any, Any]:
        for p in self.params:
            yield p

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        raise NotImplementedError("You should override this method in a subclass")


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = Tensor.random((out_features, in_features), True)
        self.params.append(self.weight)
        if bias:
            self.b = Tensor.random((1, out_features), True)
            self.params.append(self.b)

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == self.in_features
        output = x @ self.weight.T + self.b if self.bias else x @ self.weight.T
        return output

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple=(1,1),
        padding: tuple[tuple]=((0,0), (0,0)),
        bias=True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.weight = Tensor.random((out_channels, in_channels, kernel_size[0],kernel_size[1]), True, 'weight')
        self.params.append(self.weight)
        if bias:
            self.b = Tensor.random((out_channels,), True, 'bias')
            self.params.append(self.b)

    def __compute_out_dim(self, Hin, Win) -> tuple[int, int]:
        assert self.padding[0][0] == self.padding[0][1]
        assert self.padding[1][0] == self.padding[1][1]
        Hout = np.floor(
            (Hin + 2*self.padding[0][0] - self.kernel_size[0])/self.stride[0] + 1
        )
        Wout = np.floor(
            (Win + 2*self.padding[1][0] - self.kernel_size[1])/self.stride[1] + 1
        )
        return int(Hout), int(Wout)

    def forward(self, x: Tensor) -> Tensor:
        padded_x = x.pad(((0,0),(0,0),*self.padding)) if any(pad > 0 for tup in self.padding for pad in tup) else x
        Hout, Wout = self.__compute_out_dim(x.shape[-2], x.shape[-1]) #could optimize and only compute this once
        Hin, Win = padded_x.shape[-2], padded_x.shape[-1]
        out = Tensor.zeros((x.shape[0], self.out_channels, Hout, Wout), name="out")
        for batch in range(padded_x.shape[0]):
            for out_index in range(self.out_channels):
                for in_index  in range(self.in_channels):
                    for i,h_kernel in enumerate(range(0, (Hin - self.kernel_size[0])+ 1, self.stride[0])):
                        for j,w_kernel in enumerate(range(0, (Win - self.kernel_size[1]) + 1, self.stride[1])):
                            count += 6
                            out[batch,out_index,i,j] += (self.weight[out_index, in_index, :, :] * padded_x[batch, in_index, h_kernel:h_kernel+self.kernel_size[0], w_kernel:w_kernel + self.kernel_size[1]]).sum()

                if self.bias:
                    out[batch, out_index,:, :] += self.b[out_index] #adding the bias out of in_channels loop to avoid over-adding it if multiple channels
        return out


class Sum(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.sum()

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class Mean(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.mean()

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class Max(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.max()

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class ReLU(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class Exp(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.exp()

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class Log(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.log()

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class Sequential(Module):
    def __init__(self, *args) -> None:
        super().__init__()
        for arg in args:
            assert isinstance(arg, Module)
            self.params.append(arg)

    def parameters(self) -> Generator[Tensor, Any, Any]:
        for module in self.params:
            for param in module.parameters():
                yield param

    def append(self, module: Module) -> None:
        self.params.append(module)

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for layer in self.params:
            out = layer(out)
        return out

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
