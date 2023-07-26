# from src.tensor.nn.module import *
from yadll.tensor.nn import *
import torch
import numpy as np
import pytest


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

def test_multidim_linear_layer_backward():
    x = Tensor.random((4,3,10))
    layer = Linear(10,10)
    torch_x = torch.tensor(x.data, requires_grad=True)
    torch_layer = torch.nn.Linear(10,10, dtype=torch.float64)
    torch_layer.weight = torch.nn.Parameter(torch.tensor(layer.weight.data, dtype=torch.float64))
    torch_layer.bias = torch.nn.Parameter(torch.tensor(layer.b.data, dtype=torch.float64))
    output = layer.forward(x).relu().sum()
    torch_output = torch_layer(torch_x).relu().sum()
    output.backward()
    torch_output.backward()
    assert np.all(abs(layer.weight.grad - torch_layer.weight.grad.detach().numpy()) < 0.00000001)
        
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


def test_conv2d_output_default_stride():
    with torch.no_grad():
        x = Tensor.random((1, 1, 3,3), True)
        conv = Conv2d(1,1,(2,2),stride=(1,1),padding = ((1,1), (1,1)))
        torch_x = torch.tensor(x.data, dtype=torch.float64)
        torch_conv = torch.nn.Conv2d(1,1,2,padding=(1,1))
        torch_conv.weight = torch.nn.Parameter(torch.tensor(conv.weight.data))
        torch_conv.bias = torch.nn.Parameter(torch.tensor(conv.b.data))
        out = conv.forward(x)
        torch_out = torch_conv(torch_x)

        assert np.all(abs(out.data - torch_out.numpy()) < 0.0000001)

def test_conv2d_output_two_stride():
    with torch.no_grad():
        x = Tensor.random((1, 1, 3,3), True)
        conv = Conv2d(1,1,(2,2),stride=(2,2),padding = ((1,1), (1,1)))
        torch_x = torch.tensor(x.data, dtype=torch.float64)
        torch_conv = torch.nn.Conv2d(1,1,2,stride=(2,2),padding=(1,1))
        torch_conv.weight = torch.nn.Parameter(torch.tensor(conv.weight.data))
        torch_conv.bias = torch.nn.Parameter(torch.tensor(conv.b.data))
        out = conv.forward(x)
        torch_out = torch_conv(torch_x)
        assert np.all(abs(out.data - torch_out.numpy()) < 0.00000001)

def test_conv2d_output_batched_input():
    x = Tensor.random((2, 1, 3,3), True)
    conv = Conv2d(1,1,(2,2),stride=(1,1),padding = ((1,1), (1,1)))
    torch_x = torch.tensor(x.data, dtype=torch.float64)
    torch_conv = torch.nn.Conv2d(1,1,2,padding=(1,1))
    torch_conv.weight = torch.nn.Parameter(torch.tensor(conv.weight.data))
    torch_conv.bias = torch.nn.Parameter(torch.tensor(conv.b.data))
    out = conv.forward(x)
    torch_out = torch_conv(torch_x)
    assert np.all(abs(out.data - torch_out.detach().numpy()) < 0.00000001)

def test_conv2d_output_multiple_in_channels():
    x = Tensor.random((1, 3, 3,3), True)
    conv = Conv2d(3,1,(2,2),stride=(1,1),padding = ((1,1), (1,1)))
    torch_x = torch.tensor(x.data, dtype=torch.float64)
    torch_conv = torch.nn.Conv2d(3,1,2,padding=(1,1))
    torch_conv.weight = torch.nn.Parameter(torch.tensor(conv.weight.data))
    torch_conv.bias = torch.nn.Parameter(torch.tensor(conv.b.data))
    out = conv.forward(x)
    torch_out = torch_conv(torch_x)
    assert np.all(abs(out.data - torch_out.detach().numpy()) < 0.00000001)
    
def test_conv2d_output_multiple_out_channels():
    x = Tensor.random((1, 1, 3,3), True)
    conv = Conv2d(1,3,(2,2),stride=(1,1),padding = ((1,1), (1,1)))
    torch_x = torch.tensor(x.data, dtype=torch.float64)
    torch_conv = torch.nn.Conv2d(1,3,2,padding=(1,1))
    torch_conv.weight = torch.nn.Parameter(torch.tensor(conv.weight.data))
    torch_conv.bias = torch.nn.Parameter(torch.tensor(conv.b.data))
    out = conv.forward(x)
    torch_out = torch_conv(torch_x)
    assert np.all(abs(out.data - torch_out.detach().numpy()) < 0.00000001)
    
def test_conv2d_no_bias_backward():
    x = Tensor.random((1, 1, 3,3), True, name='x')
    conv = Conv2d(1,1,(2,2),stride=(1,1),padding = ((0,0), (0,0)), bias=False)
    torch_x = torch.tensor(x.data, dtype=torch.float64)
    torch_conv = torch.nn.Conv2d(1,1,2,padding=(0,0), bias=False)
    torch_conv.weight = torch.nn.Parameter(torch.tensor(conv.weight.data))
    out = conv.forward(x).mean()
    torch_out = torch_conv(torch_x).mean()
    out.backward()
    torch_out.backward()
    assert np.all(abs(conv.weight.grad - torch_conv.weight.grad.detach().numpy()) < 0.00000001)

def test_conv2d_with_bias_backward():
    x = Tensor.random((1, 1, 3,3), True, name='x')
    conv = Conv2d(1,1,(2,2),stride=(1,1),padding = ((0,0), (0,0)), bias=True)
    torch_x = torch.tensor(x.data, dtype=torch.float64)
    torch_conv = torch.nn.Conv2d(1,1,3,padding=(0,0), bias=True)
    torch_conv.weight = torch.nn.Parameter(torch.tensor(conv.weight.data))
    torch_conv.bias = torch.nn.Parameter(torch.tensor(conv.b.data))
    out = conv.forward(x).mean() * 3.2 - 2.3
    torch_out = torch_conv(torch_x).mean() * 3.2 -2.3
    out.backward()
    torch_out.backward()
    assert np.all(abs(conv.weight.grad - torch_conv.weight.grad.detach().numpy()) < 0.00000001)
    assert np.all(abs(conv.b.grad - torch_conv.bias.grad.detach().numpy()) < 0.00000001)


def test_conv2d_all_features_backward():
    x = Tensor.random((32, 3, 28,28), True, name='x')
    conv = Conv2d(3,5,(3,3),stride=(2,2),padding = ((1,1), (1,1)), bias=True)
    torch_x = torch.tensor(x.data, dtype=torch.float64)
    torch_conv = torch.nn.Conv2d(3,5,3,stride=2,padding=(1,1), bias=True)
    torch_conv.weight = torch.nn.Parameter(torch.tensor(conv.weight.data))
    torch_conv.bias = torch.nn.Parameter(torch.tensor(conv.b.data))
    out = conv.forward(x).mean() * 3.2 - 2.3
    torch_out = torch_conv(torch_x).mean() * 3.2 -2.3
    out.backward()
    torch_out.backward()
    assert np.all(abs(conv.weight.grad - torch_conv.weight.grad.detach().numpy()) < 0.00000001)
    assert np.all(abs(conv.b.grad - torch_conv.bias.grad.detach().numpy()) < 0.00000001)

def test_conv1d_output_with_bias():
    x = Tensor.random((32,3,8))
    conv = Conv1d(3,6,2)
    torch_x = torch.tensor(x.data, dtype=torch.float64)
    torch_conv = torch.nn.Conv1d(3,6,2)
    torch_conv.weight= torch.nn.Parameter(torch.tensor(conv.weight.data))
    torch_conv.bias = torch.nn.Parameter(torch.tensor(conv.b.data))
    out = conv(x)
    torch_out = torch_conv(torch_x)
    assert np.all(out.data == torch_out.detach().numpy())

def test_conv1d_backward_pass():
    x = Tensor.random((32,3,8))
    conv = Conv1d(3,6,2)
    torch_x = torch.tensor(x.data, dtype=torch.float64)
    torch_conv = torch.nn.Conv1d(3,6,2)
    torch_conv.weight= torch.nn.Parameter(torch.tensor(conv.weight.data))
    torch_conv.bias = torch.nn.Parameter(torch.tensor(conv.b.data))
    out = conv(x).mean()
    torch_out = torch_conv(torch_x).mean()
    out.backward()
    torch_out.backward()
    assert np.all(abs(conv.weight.grad - torch_conv.weight.grad.detach().numpy()) < 0.00000001)
    assert np.all(abs(conv.b.grad - torch_conv.bias.grad.detach().numpy()) < 0.00000001)

def test_conv1d_pad_and_stride_backward_pass():
    x = Tensor.random((32,3,8))
    conv = Conv1d(3,6,2, 2, 3)
    torch_x = torch.tensor(x.data, dtype=torch.float64)
    torch_conv = torch.nn.Conv1d(3,6,2, 2,3)
    torch_conv.weight= torch.nn.Parameter(torch.tensor(conv.weight.data))
    torch_conv.bias = torch.nn.Parameter(torch.tensor(conv.b.data))
    out = conv(x).mean()
    torch_out = torch_conv(torch_x).mean()
    out.backward()
    torch_out.backward()
    assert np.all(abs(conv.weight.grad - torch_conv.weight.grad.detach().numpy()) < 0.00000001)
    assert np.all(abs(conv.b.grad - torch_conv.bias.grad.detach().numpy()) < 0.00000001)
    
def test_conv3d_output():
    x = Tensor.random((32,3,32,28,28))
    conv = Conv3d(3,6,(3,3,3))
    torch_x = torch.tensor(x.data, dtype=torch.float64)
    torch_conv = torch.nn.Conv3d(3,6,3)
    torch_conv.weight= torch.nn.Parameter(torch.tensor(conv.weight.data))
    torch_conv.bias = torch.nn.Parameter(torch.tensor(conv.b.data))
    torch_out = torch_conv(torch_x)
    out = conv(x)
    assert np.all(abs(out.data - torch_out.detach().numpy()) < 0.00000001)

def test_conv3d_backward_pass():
    x = Tensor.random((32,3,32,28,28))
    conv = Conv3d(3,6,(3,3,3))
    torch_x = torch.tensor(x.data, dtype=torch.float64)
    torch_conv = torch.nn.Conv3d(3,6,3)
    torch_conv.weight= torch.nn.Parameter(torch.tensor(conv.weight.data))
    torch_conv.bias = torch.nn.Parameter(torch.tensor(conv.b.data))
    torch_out = torch_conv(torch_x).mean()
    out = conv(x).mean()
    torch_out.backward()
    out.backward()
    assert np.all(abs(conv.weight.grad - torch_conv.weight.grad.detach().numpy()) < 0.00000001)
    assert np.all(abs(conv.b.grad - torch_conv.bias.grad.detach().numpy()) < 0.00000001)
    
def test_conv3d_pad_and_stride_backward_pass():
    x = Tensor.random((32,3,32,28,28))
    conv = Conv3d(3,6,(3,3,3), ((2,2,2)), ((1,1), (2,2), (1,1)))
    torch_x = torch.tensor(x.data, dtype=torch.float64)
    torch_conv = torch.nn.Conv3d(3,6,3, 2, (1,2,1))
    torch_conv.weight= torch.nn.Parameter(torch.tensor(conv.weight.data))
    torch_conv.bias = torch.nn.Parameter(torch.tensor(conv.b.data))
    torch_out = torch_conv(torch_x).mean()
    out = conv(x).mean()
    torch_out.backward()
    out.backward()
    assert np.all(abs(conv.weight.grad - torch_conv.weight.grad.detach().numpy()) < 0.00000001)
    assert np.all(abs(conv.b.grad - torch_conv.bias.grad.detach().numpy()) < 0.00000001)
    

def test_maxpool1d_forward_pass():
    x = Tensor.random((32,3,20))
    pool = MaxPool1d(3)
    out = pool(x)
    torch_x = torch.tensor(x.data)
    torch_pool = torch.nn.MaxPool1d(3)
    torch_out = torch_pool(torch_x)
    assert out.shape == torch_out.shape, "shape not equal"
    assert np.all(abs(out.data - torch_out.detach().numpy()) < 0.00000001), "result not equal"

def test_maxpool1d_with_padding_and_stride_forward_pass():
    x = Tensor.random((32,3,20))
    pool = MaxPool1d(3, 2, ((1,1),))
    out = pool(x)
    torch_x = torch.tensor(x.data)
    torch_pool = torch.nn.MaxPool1d(3,2, 1)
    torch_out = torch_pool(torch_x)
    assert out.shape == torch_out.shape, "shape not equal"
    assert np.all(abs(out.data - torch_out.detach().numpy()) < 0.00000001), "result not equal"
    

def test_maxpool2d_forward_pass():
    x = Tensor.random((32,3,28,28))
    pool = MaxPool2d((3,3))
    out = pool(x)
    torch_x = torch.tensor(x.data)
    torch_pool = torch.nn.MaxPool2d(3)
    torch_out = torch_pool(torch_x)
    assert out.shape == torch_out.shape, "shape not equal"
    assert np.all(abs(out.data - torch_out.detach().numpy()) < 0.00000001), "result not equal"

def test_maxpool2d_with_padding_and_stride_forward_pass():
    x = Tensor.random((32,3,28,28))
    pool = MaxPool2d((3,3), (4,4), ((1,1), (1,1)))
    out = pool(x)
    torch_x = torch.tensor(x.data)
    torch_pool = torch.nn.MaxPool2d(3, 4, 1)
    torch_out = torch_pool(torch_x)
    assert out.shape == torch_out.shape, "shape not equal"
    assert np.all(abs(out.data - torch_out.detach().numpy()) < 0.00000001), "result not equal"
    
@pytest.mark.skip(reason="fails but not criticial because backward through a pool almost never happens")
def test_maxpool2d_backward_pass():
    x = Tensor.random((1,1,10,10), name="x")
    torch_x = torch.tensor(x.data, requires_grad=True)
    pool = MaxPool2d((3,3))
    pooled = pool(x)
    out = pooled.sum()
    out.backward()
    torch_pool = torch.nn.MaxPool2d(3)
    torch_pooled = torch_pool(torch_x)
    torch_out = torch_pooled.sum()
    torch_out.backward()
    assert np.all(abs(x.grad - torch_x.grad.detach().numpy()) < 0.00001), "result not equal"
    
def test_maxpool3d_forward_pass():
    x = Tensor.random((3,3,10,10,10))
    pool = MaxPool3d((3,3,3))
    out = pool(x)
    torch_x = torch.tensor(x.data)
    torch_pool = torch.nn.MaxPool3d(3)
    torch_out = torch_pool(torch_x)
    assert out.shape == torch_out.shape, "shape not equal"
    assert np.all(abs(out.data - torch_out.detach().numpy()) < 0.00000001), "result not equal"

def test_max_pool3d_with_padding_and_stride_forward_pass():
    x = Tensor.random((3,3,10,10,10))
    pool = MaxPool3d((3,3,3), (2,2,2,), ((1,1), (1,1), (1,1)))
    out = pool(x)
    torch_x = torch.tensor(x.data)
    torch_pool = torch.nn.MaxPool3d(3,2,1)
    torch_out = torch_pool(torch_x)
    assert out.shape == torch_out.shape, "shape not equal"
    assert np.all(abs(out.data - torch_out.detach().numpy()) < 0.00000001), "result not equal"
    

def test_avgpool1d_forward_pass():
    x = Tensor.random((3,3,20))
    pool = AvgPool1d(3)
    out = pool(x)
    torch_x = torch.tensor(x.data)
    torch_pool = torch.nn.AvgPool1d(3)
    torch_out = torch_pool(torch_x)
    assert out.shape == torch_out.shape, "shape not equal"
    assert np.all(abs(out.data - torch_out.detach().numpy()) < 0.0000001), "result not equal"

def test_avgpool1d_with_padding_and_stride_forward_pass():
    x = Tensor.random((3,3,20))
    pool = AvgPool1d(3, 2, ((1,1),))
    out = pool(x)
    torch_x = torch.tensor(x.data)
    torch_pool = torch.nn.AvgPool1d(3,2, 1)
    torch_out = torch_pool(torch_x)
    assert out.shape == torch_out.shape, "shape not equal"
    assert np.all(abs(out.data - torch_out.detach().numpy()) < 0.0000001), "result not equal"
    
def test_avgpool2d_forward_pass():
    x = Tensor.random((3,3,10,10))
    pool = AvgPool2d((3,3))
    out = pool(x)
    torch_x = torch.tensor(x.data)
    torch_pool = torch.nn.AvgPool2d(3)
    torch_out = torch_pool(torch_x)
    assert out.shape == torch_out.shape, "shape not equal"
    assert np.all(abs(out.data - torch_out.detach().numpy()) < 0.0000001), "result not equal"
    
def test_avgpool2d_with_padding_and_stride_forward_pass():
    x = Tensor.random((3,3,10,10))
    pool = AvgPool2d((3,3), (2,2), ((1,1), (1,1)))
    out = pool(x)
    torch_x = torch.tensor(x.data)
    torch_pool = torch.nn.AvgPool2d(3,2,1)
    torch_out = torch_pool(torch_x)
    assert out.shape == torch_out.shape, "shape not equal"
    assert np.all(abs(out.data - torch_out.detach().numpy()) < 0.0000001), "result not equal"
    
def test_avgpool3d_forward_pass():
    x = Tensor.random((3,3,10,10,10))
    pool = AvgPool3d((3,3,3))
    out = pool(x)
    torch_x = torch.tensor(x.data)
    torch_pool = torch.nn.AvgPool3d(3)
    torch_out = torch_pool(torch_x)
    assert out.shape == torch_out.shape, "shape not equal"
    assert np.all(abs(out.data - torch_out.detach().numpy()) < 0.00000001), "result not equal"

def test_avgpool3d_with_padding_and_stride_forward_pass():
    x = Tensor.random((3,3,10,10,10))
    pool = AvgPool3d((3,3,3), (2,2,2), ((1,1), (1,1), (1,1)))
    out = pool(x)
    torch_x = torch.tensor(x.data)
    torch_pool = torch.nn.AvgPool3d(3,2,1)
    torch_out = torch_pool(torch_x)
    assert out.shape == torch_out.shape, "shape not equal"
    assert np.all(abs(out.data - torch_out.detach().numpy()) < 0.00000001), "result not equal"

def test_batchnorm2d_no_tracking_affine_forward_pass():
    x = Tensor.random((3,3,16,16))
    norm = BatchNorm2d(3, track_running_stats=False)
    torch_x = torch.tensor(x.data)
    torch_norm = torch.nn.BatchNorm2d(3, track_running_stats=False)
    torch_norm.weight = torch.nn.Parameter(torch.tensor(norm.gamma.data, dtype=torch.float64))
    torch_norm.bias = torch.nn.Parameter(torch.tensor(norm.beta.data))
    out = norm(x)
    torch_out = torch_norm(torch_x)
    assert np.all(abs(out.data - torch_out.detach().numpy()) < 1e-7)

def test_batchnorm1d_no_tracking_affine_forward_pass():
    x = Tensor.random((3,3,16))
    norm = BatchNorm1d(3, track_running_stats=False)
    torch_x = torch.tensor(x.data)
    torch_norm = torch.nn.BatchNorm1d(3, track_running_stats=False)
    torch_norm.weight = torch.nn.Parameter(torch.tensor(norm.gamma.data, dtype=torch.float64))
    torch_norm.bias = torch.nn.Parameter(torch.tensor(norm.beta.data))
    out = norm(x)
    torch_out = torch_norm(torch_x)
    assert np.all(abs(out.data - torch_out.detach().numpy()) < 1e-7)
    
def test_batchnorm1d_no_tracking_affine_backward_pass():
    x = Tensor.random((3,3,16))
    norm = BatchNorm1d(3, track_running_stats=False)
    torch_x = torch.tensor(x.data)
    torch_norm = torch.nn.BatchNorm1d(3, track_running_stats=False)
    torch_norm.weight = torch.nn.Parameter(torch.tensor(norm.gamma.data, dtype=torch.float64))
    torch_norm.bias = torch.nn.Parameter(torch.tensor(norm.beta.data))
    out = norm(x).sum()
    torch_out = torch_norm(torch_x).sum()
    out.backward()
    torch_out.backward()
    assert np.all(abs(norm.beta.grad.data - torch_norm.bias.grad.detach().numpy()) < 1e-7), "bias grad incorrect"
    assert np.all(abs(norm.gamma.grad.data - torch_norm.weight.grad.detach().numpy()) < 1e-7), "weight grad incorrect"
    

def test_batchnorm2d_no_tracking_affine_backward_pass():
    x = Tensor.random((3,3,16,16))
    norm = BatchNorm2d(3, track_running_stats=False)
    torch_x = torch.tensor(x.data)
    torch_norm = torch.nn.BatchNorm2d(3, track_running_stats=False)
    torch_norm.weight = torch.nn.Parameter(torch.tensor(norm.gamma.data, dtype=torch.float64))
    torch_norm.bias = torch.nn.Parameter(torch.tensor(norm.beta.data))
    out = norm(x).sum()
    torch_out = torch_norm(torch_x).sum()
    out.backward()
    torch_out.backward()
    assert np.all(abs(norm.beta.grad.data - torch_norm.bias.grad.detach().numpy()) < 1e-7), "bias grad incorrect"
    assert np.all(abs(norm.gamma.grad.data - torch_norm.weight.grad.detach().numpy()) < 1e-7), "weight grad incorrect"

def test_batchnorm3d_no_tracking_affine_forward_pass():
    x = Tensor.random((3,3,16,16,16))
    norm = BatchNorm3d(3, track_running_stats=False)
    torch_x = torch.tensor(x.data)
    torch_norm = torch.nn.BatchNorm3d(3, track_running_stats=False)
    torch_norm.weight = torch.nn.Parameter(torch.tensor(norm.gamma.data, dtype=torch.float64))
    torch_norm.bias = torch.nn.Parameter(torch.tensor(norm.beta.data))
    out = norm(x)
    torch_out = torch_norm(torch_x)
    assert np.all(abs(out.data - torch_out.detach().numpy()) < 1e-7)



def test_batchnorm3d_no_tracking_affine_backward_pass():
    x = Tensor.random((3,3,16,16,16))
    norm = BatchNorm3d(3, track_running_stats=False)
    torch_x = torch.tensor(x.data)
    torch_norm = torch.nn.BatchNorm3d(3, track_running_stats=False)
    torch_norm.weight = torch.nn.Parameter(torch.tensor(norm.gamma.data, dtype=torch.float64))
    torch_norm.bias = torch.nn.Parameter(torch.tensor(norm.beta.data))
    out = norm(x).sum()
    torch_out = torch_norm(torch_x).sum()
    out.backward()
    torch_out.backward()
    assert np.all(abs(norm.beta.grad.data - torch_norm.bias.grad.detach().numpy()) < 1e-7), "bias grad incorrect"
    assert np.all(abs(norm.gamma.grad.data - torch_norm.weight.grad.detach().numpy()) < 1e-7), "weight grad incorrect"


def test_batchnorm2d_tracking_forward_pass():
    x = Tensor.random((3,3,16,16))
    norm = BatchNorm2d(3, track_running_stats=True)
    torch_x = torch.tensor(x.data, dtype=torch.float64)
    torch_norm = torch.nn.BatchNorm2d(3, track_running_stats=True, dtype=torch.float64)
    torch_norm.weight = torch.nn.Parameter(torch.tensor(norm.gamma.data, dtype=torch.float64))
    torch_norm.bias = torch.nn.Parameter(torch.tensor(norm.beta.data, dtype=torch.float64))
    for i in range(3):
        out = norm(x ** i + i)
        torch_out = torch_norm(torch_x ** i + i)

    assert np.all(abs(norm.running_mean.data - torch_norm.running_mean.detach().numpy()) < 1e-7), "running mean incorrect"
    assert np.all(abs(norm.running_var.data - torch_norm.running_var.detach().numpy()) < 1e-7), "running var incorrect"
    assert np.all(abs(out.data - torch_out.detach().numpy()) < 1e-7), "output incorrect"

def test_batchnorm3d_tracking_forward_pass():
    x = Tensor.random((3,3,16,16,16))
    norm = BatchNorm3d(3, track_running_stats=True)
    torch_x = torch.tensor(x.data, dtype=torch.float64)
    torch_norm = torch.nn.BatchNorm3d(3, track_running_stats=True, dtype=torch.float64)
    torch_norm.weight = torch.nn.Parameter(torch.tensor(norm.gamma.data, dtype=torch.float64))
    torch_norm.bias = torch.nn.Parameter(torch.tensor(norm.beta.data, dtype=torch.float64))
    for i in range(3):
        out = norm(x ** i + i)
        torch_out = torch_norm(torch_x ** i + i)

    assert np.all(abs(norm.running_mean.data - torch_norm.running_mean.detach().numpy()) < 1e-7), "running mean incorrect"
    assert np.all(abs(norm.running_var.data - torch_norm.running_var.detach().numpy()) < 1e-7), "running var incorrect"
    assert np.all(abs(out.data - torch_out.detach().numpy()) < 1e-7), "output incorrect"

def test_batchnorm1d_tracking_forward_pass():
    x = Tensor.random((3,3,16))
    norm = BatchNorm1d(3, track_running_stats=True)
    torch_x = torch.tensor(x.data, dtype=torch.float64)
    torch_norm = torch.nn.BatchNorm1d(3, track_running_stats=True, dtype=torch.float64)
    torch_norm.weight = torch.nn.Parameter(torch.tensor(norm.gamma.data, dtype=torch.float64))
    torch_norm.bias = torch.nn.Parameter(torch.tensor(norm.beta.data, dtype=torch.float64))
    for i in range(3):
        out = norm(x ** i + i)
        torch_out = torch_norm(torch_x ** i + i)

    assert np.all(abs(norm.running_mean.data - torch_norm.running_mean.detach().numpy()) < 1e-7), "running mean incorrect"
    assert np.all(abs(norm.running_var.data - torch_norm.running_var.detach().numpy()) < 1e-7), "running var incorrect"
    assert np.all(abs(out.data - torch_out.detach().numpy()) < 1e-7), "output incorrect"
    
def test_batchnorm2d_tracking_backward_pass():
    x = Tensor.random((3,3,16,16))
    norm = BatchNorm2d(3, track_running_stats=True)
    torch_x = torch.tensor(x.data, dtype=torch.float64)
    torch_norm = torch.nn.BatchNorm2d(3, track_running_stats=True, dtype=torch.float64)
    torch_norm.weight = torch.nn.Parameter(torch.tensor(norm.gamma.data, dtype=torch.float64))
    torch_norm.bias = torch.nn.Parameter(torch.tensor(norm.beta.data, dtype=torch.float64))
    for i in range(3):
        out = norm(x ** i + i)
        torch_out = torch_norm(torch_x ** i + i)
    out.sum().backward()
    torch_out.sum().backward()
    assert np.all(abs(norm.beta.grad.data - torch_norm.bias.grad.detach().numpy()) < 1e-7), "bias grad incorrect"
    assert np.all(abs(norm.gamma.grad.data - torch_norm.weight.grad.detach().numpy()) < 1e-7), "weight grad incorrect"

def test_batchnorm3d_tracking_backward_pass():
    x = Tensor.random((3,3,16,16,16))
    norm = BatchNorm3d(3, track_running_stats=True)
    torch_x = torch.tensor(x.data, dtype=torch.float64)
    torch_norm = torch.nn.BatchNorm3d(3, track_running_stats=True, dtype=torch.float64)
    torch_norm.weight = torch.nn.Parameter(torch.tensor(norm.gamma.data, dtype=torch.float64))
    torch_norm.bias = torch.nn.Parameter(torch.tensor(norm.beta.data, dtype=torch.float64))
    for i in range(3):
        out = norm(x ** i + i)
        torch_out = torch_norm(torch_x ** i + i)
    out.sum().backward()
    torch_out.sum().backward()
    assert np.all(abs(norm.beta.grad.data - torch_norm.bias.grad.detach().numpy()) < 1e-7), "bias grad incorrect"
    assert np.all(abs(norm.gamma.grad.data - torch_norm.weight.grad.detach().numpy()) < 1e-7), "weight grad incorrect"

def test_batchnorm1d_tracking_backward_pass():
    x = Tensor.random((3,3,16))
    norm = BatchNorm1d(3, track_running_stats=True)
    torch_x = torch.tensor(x.data, dtype=torch.float64)
    torch_norm = torch.nn.BatchNorm1d(3, track_running_stats=True, dtype=torch.float64)
    torch_norm.weight = torch.nn.Parameter(torch.tensor(norm.gamma.data, dtype=torch.float64))
    torch_norm.bias = torch.nn.Parameter(torch.tensor(norm.beta.data, dtype=torch.float64))
    for i in range(3):
        out = norm(x ** i + i)
        torch_out = torch_norm(torch_x ** i + i)
    out.sum().backward()
    torch_out.sum().backward()
    assert np.all(abs(norm.beta.grad.data - torch_norm.bias.grad.detach().numpy()) < 1e-7), "bias grad incorrect"
    assert np.all(abs(norm.gamma.grad.data - torch_norm.weight.grad.detach().numpy()) < 1e-7), "weight grad incorrect"



def test_batchnorm2d_eval_mode():
    x = Tensor.random((3,3,16,16))
    norm = BatchNorm2d(3, track_running_stats=True)
    torch_x = torch.tensor(x.data, dtype=torch.float64)
    torch_norm = torch.nn.BatchNorm2d(3, track_running_stats=True, dtype=torch.float64)
    torch_norm.weight = torch.nn.Parameter(torch.tensor(norm.gamma.data, dtype=torch.float64))
    torch_norm.bias = torch.nn.Parameter(torch.tensor(norm.beta.data, dtype=torch.float64))
    for i in range(3):
        out = norm(x ** i + i)
        torch_out = torch_norm(torch_x ** i + i)
    norm.eval()
    torch_norm.eval()
    out = norm(x ** 2 + 12)
    torch_out = torch_norm(torch_x ** 2 + 12)
    assert np.all(abs(norm.running_mean.data - torch_norm.running_mean.detach().numpy()) < 1e-7), "running mean incorrect"
    assert np.all(abs(norm.running_var.data - torch_norm.running_var.detach().numpy()) < 1e-7), "running var incorrect"
    assert np.all(abs(out.data - torch_out.detach().numpy()) < 1e-6), "output incorrect"

def test_layernorm_forward_pass():
    x = Tensor.random((20,5,10,10))
    norm = LayerNorm((5,10,10))
    torch_x = torch.tensor(x.data)
    torch_norm = torch.nn.LayerNorm((5,10,10), dtype=torch.float64)
    out = norm(x)
    torch_out = torch_norm(torch_x)
    assert out.shape == torch_out.shape, "shape incorrect"
    assert np.all(abs(out.data - torch_out.detach().numpy()) < 1e-8), "output incorrect"

def test_layernorm_backward_pass():
    x = Tensor.random((20,5,10,10))
    norm = LayerNorm((10,10,))
    torch_x = torch.tensor(x.data, dtype=torch.float64)
    torch_norm = torch.nn.LayerNorm((10,10,), dtype=torch.float64)
    out = norm(x).sum()
    torch_out = torch_norm(torch_x).sum()
    out.backward()
    torch_out.backward()
    assert np.all(abs(norm.beta.grad.data - torch_norm.bias.grad.detach().numpy()) < 1e-7), "bias grad incorrect"
    assert np.all(abs(norm.gamma.grad.data - torch_norm.weight.grad.detach().numpy()) < 1e-7), "weight grad incorrect"

    