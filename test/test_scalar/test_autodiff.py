import pytest
from src.scalar.autodiff import Scalar
import numpy as np


def test_add_grad_compute():
    a = Scalar(2)
    b = Scalar(3)
    c = a+b
    c.backward()
    assert a.grad == 1 and b.grad == 1

def test_mul_grad_compute():
    a = Scalar(2)
    b = Scalar(3)
    c = a*b
    c.backward()
    assert a.grad == 3 and b.grad == 2


def test_pow_grad_compute():
    a = Scalar(2)
    b = a**2
    b.backward()
    assert a.grad == 4


def test_exp_grad_compute():
    a = Scalar(2)
    b = a.exp()
    b.backward()
    assert np.abs(a.grad - np.exp(a).data) < 0.00000001

def test_relu_grad_compute_positive():
    a = Scalar(2)
    b = a.relu()
    b.backward()
    assert a.grad == 1

def test_relu_grad_compute_negative():
    a = Scalar(-1)
    b = a.relu()
    b.backward()
    assert a.grad == 0

def test_sigmoid_grad_compute():
    a = Scalar(2)
    b = a.sigmoid()
    b.backward()
    t = b.data
    assert a.grad == t*(1-t)

def test_tanh_grad_compute():
    a = Scalar(2)
    b = a.tanh()
    b.backward()
    t = b.data
    assert a.grad == 1-t**2
    