from abc import ABC, abstractmethod
import numpy as np
from ..autodiff import Tensor

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
    def __init__(self, params, lr: float, momentum: float = 0, dampening: float = 0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.velocities = {}

    def step(self):
        for p in self.params:
            if p.grad is not None:
                if self.momentum == 0:
                    p.data -= self.lr * p.grad
                else:
                    if p not in self.velocities:
                        self.velocities[p] = Tensor.zeros(p.shape)
                    velocity = (self.momentum * self.velocities[p]).data + (1 - self.dampening) * p.grad
                    # Update the velocity
                    self.velocities[p] = velocity
                    p.data -= self.lr *velocity
        

