import numpy as np
class Scalar:
    
    def __init__(self, data, parent = ()) -> None:
        self.data = data
        self.grad = 0.0
        self.parent = parent
        self._backward = lambda : None

    def __neg__(self):
        return self * -1
    
    def __add__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        output = Scalar(self.data + other.data, (self, other))

        def _backward():
            self.grad += output.grad
            other.grad += output.grad
        output._backward = _backward

        return output
    
    def __radd__(self,other):
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)        
        output = Scalar(self.data * other.data, (self,other))

        def _backward():
            self.grad += output.grad * other.data
            other.grad += output.grad * self.data
        output._backward = _backward
        return output
    
    def __rmul__(self,other):
        return self * other
    
    def __sub__(self,other):
        return self + (-other)
    
    def __rsub__(self,other):
        return other + (-self)

    def __pow__(self, power):
        assert isinstance(power, (int,float))
        output = Scalar(self.data ** power, (self,))
        def _backward():
            self.grad += power * self.data ** (power - 1) * output.grad
        output._backward = _backward
        
        return output
    
    def __truediv__(self,other):
        return self * (other)**(-1)
    
    def __rtruediv__(self,other):
        return other * self**(-1)
    
    
    def exp(self):
        output = Scalar(np.exp(self.data), (self,))
        def _backward():
            self.grad += np.exp(self.data) * output.grad
        output._backward = _backward
        
        return output
    
    def __repr__(self) -> str:
        return f"data: {self.data}, gradient: {self.grad}"
    
    def backward(self):
        topo_order = []
        visited = set()
        self.__build_topological_sort(self, visited, topo_order)
        self.grad = 1
        for v in reversed(topo_order):
            v._backward()
        
    def __build_topological_sort(self, v, visited, topo_order):
        if v not in visited:
            visited.add(v)
            for parent in v.parent:
                self.__build_topological_sort(parent, visited, topo_order)
            topo_order.append(v)


    # activation functions

    def relu(self):
        output = Scalar(0 if self.data < 0 else self.data, (self,))
        def _backward():
            self.grad += (output.data > 0) * output.grad #gradient is passed only if output is positive, otherwise 0
        output._backward = _backward
        return output
    

    def sigmoid(self):
        output = Scalar(1/(1 + (-self).exp().data), (self,))
        def _backward():
            self.grad += output.data * (1 - output.data) * output.grad
        output._backward = _backward
        return output
    
    def tanh(self):
        output = Scalar(((2*self).exp().data - 1) / ((2*self).exp().data + 1), (self,))

        def _backward():
            self.grad += (1 - output.data**2) * output.grad
        output._backward = _backward
        return output