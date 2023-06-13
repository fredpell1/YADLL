from src.tensor.autodiff import Tensor
from src.tensor.nn.functional import Tensor
from ..autodiff import *
from .functional import *
from typing import Any, List, Generator

class Module:
    
    def __init__(self) -> None:
        self.params : List[Tensor] = []

    def parameters(self) -> Generator[Tensor, Any, Any]:
        for p in self.params:
            yield p

    def forward(self, x:Tensor, *args, **kwargs) -> Tensor:
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
            self.b = Tensor.random((1,out_features), True)
            self.params.append(self.b)

    def forward(self,x: Tensor) -> Tensor:
        assert x.shape[-1] == self.in_features
        output = x @ self.weight.T + self.b if self.bias else x @ self.weight.T
        return output 

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


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

    def append(self, module : Module) -> None:
        self.params.append(module)

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for layer in self.params:
            out = layer(out)
        return out    
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)