from ..autodiff import *
from .functional import *
from typing import List, Generator

class Module:
    
    def __init__(self) -> None:
        self.params : List[Tensor] = []

    def parameters(self) -> Generator[Tensor, any, any]:
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
        output = x @ self.weight.transpose(0,1) + self.b if self.bias else x @ self.weight.T
        return output 

class Sequential(Module):
    pass