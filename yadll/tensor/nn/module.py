from yadll.tensor.autodiff import Tensor
from yadll.tensor.nn.functional import Tensor
from ..autodiff import *
from .functional import *
from typing import Any, List, Generator


class Module:
    def __init__(self) -> None:
        self.params: List[Tensor] = []
        self.eval_mode = False

    def parameters(self) -> Generator[Tensor, Any, Any]:
        for p in self.params:
            yield p

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        raise NotImplementedError("You should override this method in a subclass")
    
    def __call__(self,x:Tensor, *args: Any, **kwds: Any) -> Any:
        return self.forward(x)

    def train(self):
        self.eval_mode = False
    
    def eval(self):
        self.eval_mode = True

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

class Conv1d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding : int = 0,
        bias = True
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.weight = Tensor.random((out_channels, in_channels,kernel_size))
        self.params.append(self.weight)
        if bias:
            self.b = Tensor.random((out_channels,))
            self.params.append(self.b)

    def __compute_out_length(self,Lin):
        Lout = np.floor(
            (Lin + 2*self.padding - self.kernel_size) / self.stride + 1
        )
        return int(Lout)
    
    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 3, "conv1d is currently only supporting batched inputs"
        padded_x = x.pad(((0,0), (0,0), (self.padding,self.padding))) if self.padding > 0 else x
        Lout = self.__compute_out_length(x.shape[-1])
        expanded_view = padded_x.rolling_window((padded_x.shape[0], padded_x.shape[1], self.kernel_size), self.stride)
        out = (expanded_view.reshape((Lout, padded_x.shape[0], padded_x.shape[1] * self.kernel_size)) @ self.weight.flatten(1).T).permute((1,2,0))
        return out + self.b.reshape((-1,1)) if self.bias else out

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
        assert len(x.shape) == 4, "conv2d currently only supports batched inputs"
        padded_x = x.pad(((0,0),(0,0),*self.padding)) if any(pad > 0 for tup in self.padding for pad in tup) else x
        Hout, Wout = self.__compute_out_dim(x.shape[-2], x.shape[-1]) #could optimize and only compute this once
        expanded_view = padded_x.rolling_window((padded_x.shape[0], padded_x.shape[1], self.kernel_size[0], self.kernel_size[1]), stride = self.stride[0])
        out = (expanded_view.reshape((Hout,Wout, padded_x.shape[0],padded_x.shape[1] * self.kernel_size[0] * self.kernel_size[1])) @ self.weight.flatten(1).T).permute((2,3,0,1))
        return out + self.b.reshape((-1,1,1)) if self.bias else out


class Conv3d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple=(1,1,1),
        padding:tuple[tuple]=((0,0), (0,0), (0,0)),
        bias=True
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

    def __compute_out_dim(self, Din, Hin, Win) -> tuple[int,int,int]:
        pad0 = self.padding[0][0]
        pad1 = self.padding[1][0]
        pad2 = self.padding[2][0]
        Dout = np.floor(
            (Din + 2*pad0 - self.kernel_size[0])/ self.stride[0] + 1
        )
        Hout = np.floor(
            (Hin + 2*pad1 - self.kernel_size[1])/ self.stride[1] + 1
        )
        Wout = np.floor(
            (Win + 2*pad2 - self.kernel_size[2])/ self.stride[2] + 1
        )
        return int(Dout), int(Hout), int(Wout)
    
    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 5, "conv3d currently only supports batched inputs"
        padded_x = x.pad(((0,0),(0,0),*self.padding)) if any(pad > 0 for tup in self.padding for pad in tup) else x
        Dout, Hout, Wout = self.__compute_out_dim(x.shape[-3], x.shape[-2], x.shape[-1])
        expanded_view = padded_x.rolling_window((padded_x.shape[0], padded_x.shape[1], *self.kernel_size), self.stride[0])
        out = (expanded_view.reshape((Dout,Hout,Wout, padded_x.shape[0], padded_x.shape[1] * np.prod(self.kernel_size))) @ self.weight.flatten(1).T).permute((3,4,0,1,2))
        return out + self.b.reshape((-1,1,1,1)) if self.bias else out
    
class MaxPool1d(Module):
    def __init__(
        self,
        kernel_size:int,
        stride:int = None,
        padding: tuple[tuple] = ((0,0),)
        ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.padding = padding


    def __compute_out_length(self, Lin):
        Lout = np.floor(
            (Lin + 2*self.padding[0][0] - self.kernel_size) / self.stride + 1
        )
        return int(Lout)
    

    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 3, "MaxPool1d only supports batched inputs"
        padded_x = x.pad(((0,0),(0,0),*self.padding), -np.inf) if any(pad > 0 for tup in self.padding for pad in tup) else x
        Lout = self.__compute_out_length(x.shape[-1])
        expanded_view = padded_x.rolling_window((padded_x.shape[0], padded_x.shape[1], self.kernel_size), self.stride)
        return expanded_view.max(-1).reshape((Lout, x.shape[0], x.shape[1])).permute((1,2,0))

class MaxPool2d(Module):
    def __init__(
        self,
        kernel_size: tuple[int],
        stride: tuple[int] = None,
        padding: tuple[tuple] = ((0,0), (0,0)),

        ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.padding = padding

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
        assert len(x.shape) == 4, "MaxPool2d only supports batched inputs"
        padded_x = x.pad(((0,0),(0,0),*self.padding), -np.inf) if any(pad > 0 for tup in self.padding for pad in tup) else x
        Hout, Wout = self.__compute_out_dim(x.shape[-2], x.shape[-1])
        expanded_view = padded_x.rolling_window((padded_x.shape[0], padded_x.shape[1], self.kernel_size[0], self.kernel_size[1]), stride = self.stride[0])
        return expanded_view.max(-1).max(-1).reshape((Hout,Wout,x.shape[0], x.shape[1])).permute((2,3,0,1))

class MaxPool3d(Module):
    def __init__(
        self,
        kernel_size: tuple,
        stride: tuple=None,
        padding:tuple[tuple]=((0,0), (0,0), (0,0)),
        ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.padding = padding

    def __compute_out_dim(self, Din, Hin, Win) -> tuple[int,int,int]:
        pad0 = self.padding[0][0]
        pad1 = self.padding[1][0]
        pad2 = self.padding[2][0]
        Dout = np.floor(
            (Din + 2*pad0 - self.kernel_size[0])/ self.stride[0] + 1
        )
        Hout = np.floor(
            (Hin + 2*pad1 - self.kernel_size[1])/ self.stride[1] + 1
        )
        Wout = np.floor(
            (Win + 2*pad2 - self.kernel_size[2])/ self.stride[2] + 1
        )
        return int(Dout), int(Hout), int(Wout)    

    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 5, "MaxPool3d only supports batched inputs"
        padded_x = x.pad(((0,0),(0,0),*self.padding), -np.inf) if any(pad > 0 for tup in self.padding for pad in tup) else x
        Dout, Hout, Wout = self.__compute_out_dim(x.shape[-3], x.shape[-2], x.shape[-1])
        expanded_view = padded_x.rolling_window((padded_x.shape[0], padded_x.shape[1], *self.kernel_size), self.stride[0])
        return expanded_view.max(-1).max(-1).max(-1).reshape((Dout, Hout, Wout, x.shape[0], x.shape[1])).permute((3,4,0,1,2))


class AvgPool1d(Module):
    def __init__(
        self,
        kernel_size:int,
        stride:int = None,
        padding: tuple[tuple] = ((0,0),)
        ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.padding = padding


    def __compute_out_length(self, Lin):
        Lout = np.floor(
            (Lin + 2*self.padding[0][0] - self.kernel_size) / self.stride + 1
        )
        return int(Lout)
    

    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 3, "AvgPool1d only supports batched inputs"
        padded_x = x.pad(((0,0),(0,0),*self.padding),0) if any(pad > 0 for tup in self.padding for pad in tup) else x
        Lout = self.__compute_out_length(x.shape[-1])
        expanded_view = padded_x.rolling_window((padded_x.shape[0], padded_x.shape[1], self.kernel_size), self.stride)
        return (expanded_view.sum(-1).reshape((Lout, x.shape[0], x.shape[1])) / float(np.prod(self.kernel_size))).permute((1,2,0))

class AvgPool2d(Module):
    def __init__(
        self,
        kernel_size:tuple[int],
        stride:int = None,
        padding: tuple[tuple] = ((0,0),(0,0))
        ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.padding = padding

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
        assert len(x.shape) == 4, "AvgPool2d only supports batched inputs"
        padded_x = x.pad(((0,0),(0,0),*self.padding), 0) if any(pad > 0 for tup in self.padding for pad in tup) else x
        Hout, Wout = self.__compute_out_dim(x.shape[-2], x.shape[-1])
        expanded_view = padded_x.rolling_window((padded_x.shape[0], padded_x.shape[1], self.kernel_size[0], self.kernel_size[1]), stride = self.stride[0])
        return (expanded_view.sum((-1,-2)).reshape((Hout,Wout,x.shape[0], x.shape[1])) / float(np.prod(self.kernel_size))).permute((2,3,0,1))


class AvgPool3d(Module):
    def __init__(
        self,
        kernel_size: tuple,
        stride: tuple=None,
        padding:tuple[tuple]=((0,0), (0,0), (0,0)),
        ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.padding = padding

    def __compute_out_dim(self, Din, Hin, Win) -> tuple[int,int,int]:
        pad0 = self.padding[0][0]
        pad1 = self.padding[1][0]
        pad2 = self.padding[2][0]
        Dout = np.floor(
            (Din + 2*pad0 - self.kernel_size[0])/ self.stride[0] + 1
        )
        Hout = np.floor(
            (Hin + 2*pad1 - self.kernel_size[1])/ self.stride[1] + 1
        )
        Wout = np.floor(
            (Win + 2*pad2 - self.kernel_size[2])/ self.stride[2] + 1
        )
        return int(Dout), int(Hout), int(Wout)    

    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 5, "MaxPool3d only supports batched inputs"
        padded_x = x.pad(((0,0),(0,0),*self.padding)) if any(pad > 0 for tup in self.padding for pad in tup) else x
        Dout, Hout, Wout = self.__compute_out_dim(x.shape[-3], x.shape[-2], x.shape[-1])
        expanded_view = padded_x.rolling_window((padded_x.shape[0], padded_x.shape[1], *self.kernel_size), self.stride[0])
        return (expanded_view.sum((-3,-2,-1)).reshape((Dout, Hout, Wout, x.shape[0], x.shape[1])) / float(np.prod(self.kernel_size))).permute((3,4,0,1,2))


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


class Max(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.max()



class ReLU(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()



class Exp(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.exp()



class Log(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.log()


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

