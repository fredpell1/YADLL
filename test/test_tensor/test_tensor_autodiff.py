from src.tensor.autodiff import *
import numpy as np
import torch
def test_mul_scalar_backward_pass():
    a = Tensor(np.array([[1,2],[2,1]]), requires_grad = True)
    b = a * 2
    b.backward()
    c = torch.tensor([[1.0,2],[2,1]], requires_grad=True)
    d = (c * 2)
    d.backward(torch.ones_like(d)) #need to pass a tensor because d has more than one element
    print(a.grad, c.grad)
    assert np.all(a.grad == c.grad.numpy())
    
def test_mul_matrix_backward_pass():
    a = Tensor(np.array([[1,2],[2,1]]), requires_grad = True)
    b = Tensor(np.array([[2,2],[3,4]]), requires_grad = True)
    c = a * b
    c.backward()
    d = torch.tensor([[1.0,2],[2,1]], requires_grad = True)
    e = torch.tensor([[2.0,2],[3,4]], requires_grad = True)
    f = d * e
    f.backward(torch.ones_like(f))
    assert np.all(a.grad == d.grad.numpy())

def test_matmul_backward_pass():
    a = Tensor(np.array([[1,2],[2,1]]), requires_grad = True)
    b = Tensor(np.array([[2,2],[3,4]]), requires_grad = True)
    c = a @ b
    c.backward()
    d = torch.tensor([[1.0,2],[2,1]], requires_grad = True)
    e = torch.tensor([[2.0,2],[3,4]], requires_grad = True)
    f = d @ e
    f.backward(torch.ones_like(f))
    assert np.all(a.grad == d.grad.numpy()) and np.all(b.grad == e.grad.numpy())

def test_rmatmul_backward_pass():
    a = Tensor(np.array([[1,2],[2,1]]), requires_grad = True)
    b = Tensor(np.array([[2,2],[3,4]]), requires_grad = True)
    c = b @ a
    c.backward()
    d = torch.tensor([[1.0,2],[2,1]], requires_grad = True)
    e = torch.tensor([[2.0,2],[3,4]], requires_grad = True)
    f = e @ d
    f.backward(torch.ones_like(f))
    assert np.all(a.grad == d.grad.numpy()) and np.all(b.grad == e.grad.numpy())

def test_pow_backward_pass():
    a = Tensor(np.array([[1,2],[2,1]]), requires_grad=True)
    b = a**2
    b.backward()
    c = torch.tensor([[1.0,2],[2,1]], requires_grad=True)
    d = c**2
    d.backward(torch.ones_like(d))
    assert np.all(a.grad == c.grad.numpy())

def test_div_backward_pass():
    a = Tensor(np.array([[1.0,2],[2,1]]), requires_grad=True)
    b = a/2
    b.backward()
    c = torch.tensor([[1.0,2],[2,1]], requires_grad=True)
    d = c/2
    d.backward(torch.ones_like(d))
    assert np.all(a.grad == c.grad.numpy())
    
def test_transpose_backward_pass():
    a = Tensor(np.array([[1.0,2],[2,1]]), requires_grad=True)
    b = a.transpose(0,1)
    b.backward()
    c = torch.tensor([[1.0,2],[2,1]], requires_grad=True)
    d = c.transpose(0,1)
    d.backward(torch.ones_like(d))
    print(a.grad, c.grad)
    assert np.all(a.grad == c.grad.numpy())

def test_sum_backward_pass():
    a = Tensor(np.array([[1.0,2],[2,1]]), requires_grad=True)
    b = a.sum()
    b.backward()
    c = torch.tensor([[1.0,2], [2,1]], requires_grad=True)
    d = c.sum()
    d.backward()
    assert np.all(a.grad == c.grad.numpy())

def test_mean_backward_pass():
    a = Tensor(np.array([[1.0,2],[2,1]]), requires_grad=True)
    b = a.mean()
    b.backward()
    c = torch.tensor([[1.0,2],[2,1]], requires_grad=True)
    d = c.mean()
    d.backward(torch.ones_like(d))
    assert np.all(a.grad == c.grad.numpy())

def test_max_backward_pass():
    a = Tensor(np.array([[1.0,2],[2,1]]), requires_grad=True)
    b = a.max() * 3
    b.backward()
    c = torch.tensor([[1.0,2],[2,1]], requires_grad=True)
    d = c.max() * 3
    d.backward()
    assert np.all(a.grad == c.grad.numpy())
