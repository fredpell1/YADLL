from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    def __init__(self, params):
        self.params = params

    @abstractmethod
    def step(self):
        pass

    def zero_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr: float, momentum: float):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * p.grad
