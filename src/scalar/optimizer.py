from src.scalar.nn import Module
from abc import abstractmethod

class Optimizer(Module):
    
    def __init__(self, parameters):
        self.params = parameters

    def parameters(self):
        return self.params
    
    @abstractmethod
    def step(self):
        raise NotImplementedError("Method step not implememented")
    

class SGD(Optimizer):

    def __init__(self, parameters, learning_rate):
        super().__init__(parameters)
        self.lr = learning_rate

    def step(self):
        for p in self.parameters():
            p.data -= self.lr * p.grad
    

