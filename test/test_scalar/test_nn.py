import pytest
from src.scalar.nn import *
import numpy as np

def test_linear_neuron():
    x = [1,1]
    neuron = Neuron(2)
    assert abs(neuron(x).data - (np.sum(neuron.weights) +  neuron.bias).data) < 0.000001

def test_relu_neuron():
    x = [1,1]
    neuron = Neuron(2, 'relu')
    output = neuron(x).data
    expected_output = (np.sum(neuron.weights) +  neuron.bias).relu().data
    assert abs(output - expected_output) < 0.0000001

def test_linear_layer_output_size():
    x = [1,1]
    layer = Layer(2,3)
    output = layer(x)
    assert len(output) == 3

def test_linear_layer_activation_function():
    x = [1,1]
    layer = Layer(2,1, activation_func = "tanh")
    output = layer(x)[0].data
    neuron = layer.neurons[0]
    expected_output = (np.sum(neuron.weights) + neuron.bias).tanh().data
    assert abs(output - expected_output) < 0.0000001

def test_sequential_output_size():
    x = [1,1]
    model = Sequential(
        Layer(2,30),
        Layer(30,30),
        Layer(30,1,activation_func = "sigmoid")
    )
    output = model(x)
    assert len(output) == 1