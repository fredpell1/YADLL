class Scalar:
    
    def __init__(self, data, parent = ()) -> None:
        self.data = data
        self.grad = 0.0
        self.parent = parent
        self._backward = lambda : None

    def __add__(self, other):
        output = Scalar(self.data + other.data, (self, other))

        def _backward():
            self.grad += output.grad
            other.grad += output.grad
        output._backward = _backward

        return output
    
    def __mul__(self, other):
        output = Scalar(self.data * other.data, (self,other))

        def _backward():
            self.grad += output.grad * other.data
            other.grad += output.grad * self.data
        output._backward = _backward
        return output
    
    def __pow__(self, power):
        assert isinstance(power, (int,float))
        output = Scalar(self.data ** power, (self,))
        def _backward():
            self.grad += power * self.data ** (power - 1) * output.grad
        output.grad = _backward
        
        return output