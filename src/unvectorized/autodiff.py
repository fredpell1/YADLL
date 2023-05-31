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
        output = Scalar(self.data + other.data, (self, other))

        def _backward():
            self.grad += output.grad
            other.grad += output.grad
        output._backward = _backward

        return output
    
    def __radd__(self,other):
        return self + other
    
    def __mul__(self, other):
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
        output.grad = _backward
        
        return output
    
    def __truediv__(self,other):
        return self * (other)**(-1)
    
    def __rtruediv__(self,other):
        return other * self**(-1)
    
    
    def exp(self):
        output = Scalar(np.exp(self.data), (self,))
        def _backward():
            self.grad += np.exp(self.data) * output.grad
        output.grad = _backward
        
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