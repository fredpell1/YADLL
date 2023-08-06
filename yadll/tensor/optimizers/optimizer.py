from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    def __init__(self, params) -> None:
        self.params = params

    @abstractmethod
    def step(self):
        pass

    def zero_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr: float, momentum: float = 0) -> None:
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [np.zeros_like(p) for p in self.params]

    def step(self):
        # Implement step for SGD with momentum
        for i, p in enumerate(self.params):
            if p.grad is not None:
                if self.momentum > 0:
                    # Update velocity
                    self.velocities[i] = self.momentum * self.velocities[i] + (1 - self.momentum) * p.grad
                    # Apply momentum in the update
                    p.data -= self.lr * self.velocities[i]
                else:
                    # Basic SGD update without momentum
                    p.data -= self.lr * p.grad
