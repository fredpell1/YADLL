# YADLL
yadll is a deep learning library based on top of [numpy](https://numpy.org/doc/stable/index.html).

## Design Philosophy
yadll is designed with the goal of being the most readable and easy to browse deep learning library. 

Pytorch is great, but if you're wondering how their Conv2d is implemented, you're in for a wild ride across many layers of abstraction and a lot of cpp files. Therefore, the core design principle of yadll is to minimize the number of layers of abstraction. Currently, there are only two layers: the [tensor layer](https://github.com/fredpell1/YADLL/blob/main/yadll/autodiff.py) containing the code for automatic differentiation and the [neural net (nn)](https://github.com/fredpell1/YADLL/tree/main/yadll/nn) layer containing modules to build neural networks.

As mentionned above, Pytorch is great! So yadll aims to match as close as possible the torch api so that your yadll code can be migrated to torch and vice versa.

Tests, tests, tests. Everything in yadll is matched against Pytorch so we can ensure that the implementation is correct, or at least as correct as pytorch. 

yaddl currently does not support gpu acceleration because of its numpy backend, but eventually it will as more backends are integrated.

## Installation
If you wish to install yadll, the only currently supported way is from source:
```sh
git clone https://github.com/fredpell1/YADLL.git
cd YADLL
python -m pip install -e .
```
If you wish to also install dependencies for the tests, change the last line to
```sh
python -m pip install -e .['testing']
```
If you wish to use our pre-commit hooks you will first need to install [ruff](https://github.com/astral-sh/ruff).
```sh
pip install ruff
```
Then you can activate the pre-commit with the following command
```sh
pre-commit install
```
## Contributing
If you wish to contribute to yadll do the following: 

1. Read the [design philosophy](#design-philosophy)
2. Install yadll with the [test dependencies](#installation)
3. Look in issues for features that need to be added, bugs to be fixed, etc.
4. If you want a new feature in yadll, submit a new issue!

**All PRs that add functionalities or fix bugs need to come with tests that match against the torch api.**

## Features

### Tensor operations
You can do a lot of operations with yadll's [tensors](https://github.com/fredpell1/YADLL/blob/main/yadll/autodiff.py) similarly to what you can do with pytorch. Every operation also defines its own backward pass, e.g.
```python
def exp(self) -> Tensor:
    output = Tensor(
        np.exp(self.data),
        requires_grad=True if self.requires_grad else False,
        parent = (self,),
        op="exp"
    )
    def _backward():
        self.grad += np.exp(self.data) * output.grad
    output._backward = _backward

    return output
```

### Neural networks
yadll supports 
- [x] Linear Layers 
- [x] Activation layers: ReLU, Max, Mean, etc.
- [x] Convolution layers
- [x] Pooling layers: max and average
- [x] Normalization layers: Batch and Layer norm
- [ ] RNNs
- [ ] Transformers

## Examples
Here's an example on how to use yadll, as you can see it's almost identical to torch:
```python
from yadll.autodiff import *
from yadll.nn import *
a = Tensor.random((10,20,20))
b = Tensor.random((10,20,20))
c = (a @ b).sum()
c.backward()
model = Linear(10,1)
...
```
