from typing import Any
from src.scalar.autodiff import Scalar
import numpy as np
from abc import abstractmethod, ABCMeta

class Module(metaclass=ABCMeta):
    "Base class for all neural nets. Named to match Pytorch's api"
    
    @abstractmethod
    def parameters(self):
        raise NotImplementedError("Method parameters not implememented")
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0


class Neuron(Module):

    def __init__(self, nin, activation_func = None) -> None:
        super().__init__()
        self.nin = nin
        self.activation_func = activation_func
        self.weights = [Scalar(np.random.randn()) for i in range(nin)]
        self.bias = Scalar(np.random.randn())

    def parameters(self):
        return self.weights + [self.bias]
    
    def __call__(self,x):
        assert len(x) == self.nin
        out = Scalar(np.dot(self.weights, x).data + self.bias.data)
        if self.activation_func:
            if hasattr(Scalar, self.activation_func):
                return getattr(out, self.activation_func)()
            else:
                raise ValueError(f"activation function: {self.activation_func} is not supported")
        else:
            return out
        

class Layer(Module):

    def __init__(self, nin, nout, **kwargs) -> None:
        super().__init__()
        self.nin = nin
        self.nout = nout
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]


    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]
    

    def __call__(self,x):
        return [neuron(x) for neuron in self.neurons]
    

class Sequential(Module):

    def __init__(self, *args) -> None:
        super().__init__()
        self.layers = [*args]


    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

    def add_layer(self, layer):
        self.layers.append(layer)    
    