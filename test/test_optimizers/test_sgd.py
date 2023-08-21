from yadll.tensor.autodiff import *
from yadll.tensor.optimizers import *
from yadll.tensor.nn import *
import numpy
import torch


def test_sgd_step_no_momentum():
    x = Tensor.random((2, 2))
    torch_x = torch.tensor(x.data)
    model = Linear(2, 1, False)
    torch_model = torch.nn.Linear(2, 1, False)
    torch_model.weight = torch.nn.Parameter(torch.tensor(model.weight.data))
    optim = SGD(model.parameters(), 0.1, 0)
    torch_optim = torch.optim.SGD(torch_model.parameters(), 0.1, 0)
    out = model(x).sum()
    torch_out = torch_model(torch_x).sum()
    out.backward()
    torch_out.backward()
    assert np.all(model.weight.data == torch_model.weight.detach().numpy()), "weight not equal before step"
    optim.step()
    torch_optim.step()
    assert np.all(model.weight.data == torch_model.weight.detach().numpy()), "weight not equal after step"


def test_sgd_with_momentum():
    model = Linear(2, 1, False)
    torch_model = torch.nn.Linear(2, 1, False)
    torch_model.weight = torch.nn.Parameter(torch.tensor(model.weight.data))
    optim = SGD(model.parameters(), 0.1, 0.9)
    torch_optim = torch.optim.SGD(torch_model.parameters(), 0.1, 0.9)
    assert np.all(model.weight.data == torch_model.weight.detach().numpy()), "weight not equal before step"
    for _ in range(3):
        x = Tensor.random((2, 2))
        torch_x = torch.tensor(x.data)
        out = model(x).sum()
        torch_out = torch_model(torch_x).sum()
        out.backward()
        torch_out.backward()
        optim.step()
        torch_optim.step()
    assert np.all(model.weight.data == torch_model.weight.detach().numpy()), "weight not equal after step"
