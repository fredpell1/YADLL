#set the backend to numpy
import os
os.environ['backend'] = 'numpy'

from yadll.ir import *
import numpy as np

def test_type():
    a = Data([1,2])
    assert isinstance(a, np.ndarray)

def test_add():
    a = np.random.randn(1,2,3)
    b = np.random.randn(1,2,3)
    assert add == np.add, "operation not the same"
    assert np.all(add(a,b) == (a+b)), "result not the same"

def test_mul():
    a = np.random.randn(1,2,3)
    b = np.random.randn(1,2,3)
    assert mul == np.multiply, "operation not the same"
    assert np.all(mul(a,b) == (a*b)), "result not the same"

def test_matmul():
    a = np.random.randn(1,2,3)
    b = np.random.randn(1,3,2)
    assert matmul == np.matmul, "operation not the same"
    assert np.all(matmul(a,b) == (a @ b))

def test_pow_with_number():
    a = np.random.randn(2,2)
    power = 2.0
    assert pow == np.power, "operation not the same"
    assert np.all(pow(a,power) == a ** power)    

def test_pow_with_data():
    a = np.random.random((1,2))
    power = np.random.random((1,2))
    assert np.all(pow(a,power) == a ** power)    
