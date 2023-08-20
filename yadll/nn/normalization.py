from ..autodiff import *
from .module import Module


class BatchNorm(Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if affine:
            self.gamma = Tensor.ones((num_features,), name="gamma")
            self.beta = Tensor.zeros((num_features,), name="beta")
            self.params.append(self.gamma)
            self.params.append(self.beta)
        self.running_mean = (
            Tensor.zeros((num_features,), False, "running_mean")
            if track_running_stats
            else None
        )
        self.running_var = (
            Tensor.ones((num_features,), False, "running_var")
            if track_running_stats
            else None
        )

    def norm(self, x, axis):
        current_mean = x.mean(axis)
        current_var = x.var(axis, unbiased=False)
        shape = (1, self.num_features) + tuple(1 for i in range(len(x.shape) - 2))
        mean = current_mean.reshape(shape)
        var = current_var.reshape(shape)
        if self.track_running_stats:
            if self.eval_mode:
                mean = self.running_mean.reshape(shape)
                var = self.running_var.reshape(shape)
            else:
                self.running_mean = (
                    1.0 - self.momentum
                ) * self.running_mean + self.momentum * current_mean
                self.running_var = (
                    1.0 - self.momentum
                ) * self.running_var + self.momentum * x.var(axis, unbiased=True)
        gamma = self.gamma if self.affine else Tensor.ones((self.num_features,))
        beta = self.beta if self.affine else Tensor.zeros((self.num_features,))
        return gamma.reshape(shape) * (x - mean) / (var + self.eps) ** (
            1 / 2
        ) + beta.reshape(shape)


class BatchNorm1d(BatchNorm):
    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 3, "BatchNorm1d only takes batched inputs"
        return self.norm(x, (0, 2))


class BatchNorm2d(BatchNorm):
    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 4, "BatchNorm2d only takes batched inputs"
        return self.norm(x, (0, 2, 3))


class BatchNorm3d(BatchNorm):
    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 5, "BatchNorm3d only takes batched inputs"
        return self.norm(x, (0, 2, 3, 4))


class LayerNorm(Module):
    def __init__(
        self,
        normalized_shape: tuple[int],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.affine = elementwise_affine

        self.gamma = Tensor.ones(normalized_shape, name="gamma")
        self.beta = Tensor.zeros(normalized_shape, name="beta")
        if elementwise_affine:
            self.params.append(self.gamma)
            self.params.append(self.beta)

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        axis = tuple(
            reversed([len(x.shape) - i - 1 for i in range(len(self.normalized_shape))])
        )
        shape = tuple(s if i not in axis else 1 for i, s in enumerate(x.shape))
        param_shape = (1,) * (
            len(x.shape) - len(self.normalized_shape)
        ) + self.normalized_shape
        mean = x.mean(axis).reshape(shape)
        var = x.var(axis).reshape(shape)
        return self.gamma.reshape(param_shape) * (x - mean) / (var + self.eps) ** (
            1 / 2
        ) + self.beta.reshape(param_shape)
