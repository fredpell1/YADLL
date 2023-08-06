class Optimizer:
    def __init__(self, params) -> None:
        self.params = params

    def step(self):
        raise NotImplementedError("You should override this method in a subclass")

    def zero_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr: float, momentum: float = 0) -> None:
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum

    def step(self):
        # TODO: implement step for SGD
        for p in self.params:
            continue
