from ..autodiff import *
from .module import Module
import numpy as np

class BatchNorm2d(Module):
    def __init__(
        self,
        num_features: int, 
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True
        ) -> None:
        super().__init__()
        self.num_features = num_features    
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if affine:
            self.gamma = Tensor.ones((num_features,), name='gamma')
            self.beta = Tensor.zeros((num_features,), name='beta')
            self.params.append(self.gamma)
            self.params.append(self.beta)
        self.running_mean = Tensor.zeros((num_features,), False, 'running_mean') if track_running_stats else None
        self.running_var = Tensor.ones((num_features,), False, 'running_var') if track_running_stats else None

    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 4, "BatchNorm2d only takes batched inputs"
        current_mean = x.mean((0,2,3))
        current_var = x.var((0,2,3), unbiased=False)
        mean = current_mean.reshape((1,self.num_features, 1,1))
        var = current_var.reshape((1,self.num_features, 1, 1))
        if self.track_running_stats:
            if self.eval_mode:
                mean = self.running_mean.reshape((1,self.num_features,1,1))
                var = self.running_var.reshape((1,self.num_features,1,1))
            else:
                self.running_mean = (1.0-self.momentum) * self.running_mean + self.momentum * current_mean
                self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * x.var((0,2,3), unbiased=True)
        gamma = self.gamma if self.affine else Tensor.ones((self.num_features,))
        beta = self.beta if self.affine else Tensor.zeros((self.num_features,))
        return gamma.reshape((1,self.num_features,1,1)) * (x- mean) / (var + self.eps) ** (1/2) + beta.reshape((1,self.num_features,1,1))