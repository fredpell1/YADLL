from src.tensor.nn.module import *
import torch
import numpy as np

def test_linear_layer_output():
    with torch.no_grad():
        x = Tensor(np.array([1.0,2,3,4]))
        layer = Linear(4,10)
        torch_x = torch.tensor([1.0,2,3,4], dtype=torch.float64).reshape(1,-1)
        torch_layer = torch.nn.Linear(4,10, dtype=torch.float64)
        torch_layer.weight = torch.nn.Parameter(torch.tensor(layer.weight.data, dtype=torch.float64))
        torch_layer.bias = torch.nn.Parameter(torch.tensor(layer.b.data, dtype=torch.float64))
        
        output = layer.forward(x)
        torch_output = torch_layer(torch_x)

        assert np.all(abs(output.data - torch_output.detach().numpy()[0]) < 0.000001)

def test_linear_layer_backward():
        x = Tensor(np.array([1.0,2,3,4]).reshape(1,-1), requires_grad=True)
        layer = Linear(4,10)
        torch_x = torch.tensor([1.0,2,3,4], dtype=torch.float64).reshape(1,-1)
        torch_layer = torch.nn.Linear(4,10, dtype=torch.float64)
        torch_layer.weight = torch.nn.Parameter(torch.tensor(layer.weight.data, dtype=torch.float64))
        torch_layer.bias = torch.nn.Parameter(torch.tensor(layer.b.data, dtype=torch.float64))
        
        output = layer.forward(x).relu().sum()
        torch_output = torch_layer(torch_x).relu().sum()
        output.backward()
        torch_output.backward()
        assert np.all(layer.weight.grad == torch_layer.weight.grad.detach().numpy())

def test_sequential_output():
    with torch.no_grad():
        x = Tensor(np.array([1.0,2,3,4]).reshape(1,-1), requires_grad=True)
        net = Sequential(
            Linear(4,10),
            ReLU(), 
            Linear(10,1)
        )
        torch_x = torch.tensor([1.0,2,3,4], dtype=torch.float64).reshape(1,-1)
        torch_net = torch.nn.Sequential(
             torch.nn.Linear(4,10, dtype=torch.float64),
             torch.nn.ReLU(),
             torch.nn.Linear(10,1, dtype=torch.float64)
        )
        for param, torch_param in zip(net.parameters(), torch_net.parameters()):
            torch_param.data.copy_(torch.tensor(param.data.reshape(torch_param.data.shape), dtype=torch.float64))

        assert abs(net(x).data[0,0] - torch_net(torch_x).numpy()[0,0]) < 0.00000001

def test_sequential_backward():
    x = Tensor(np.array([1.0,2,3,4]).reshape(1,-1), requires_grad=True)
    net = Sequential(
        Linear(4,10),
        ReLU(), 
        Linear(10,1)
    )
    torch_x = torch.tensor([1.0,2,3,4], dtype=torch.float64).reshape(1,-1)
    torch_net = torch.nn.Sequential(
            torch.nn.Linear(4,10, dtype=torch.float64),
            torch.nn.ReLU(),
            torch.nn.Linear(10,1, dtype=torch.float64)
    )
    for param, torch_param in zip(net.parameters(), torch_net.parameters()):
        torch_param.data.copy_(torch.tensor(param.data.reshape(torch_param.data.shape), dtype=torch.float64))
    out = net(x)
    torch_out = torch_net(torch_x)
    out.backward()
    torch_out.backward()
    for param, torch_param in zip(net.parameters(), torch_net.parameters()):
        assert np.all(abs(param.grad - torch_param.grad.detach().numpy()) < 0.0000001)
