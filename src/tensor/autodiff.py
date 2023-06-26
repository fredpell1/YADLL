from __future__ import annotations
from typing import Union, Tuple
import numpy as np

def add_dimensions(old_shape, new_shape):
    if len(old_shape) <= len(new_shape):
        minimal_length = old_shape
        other_shape = new_shape
    else:
        minimal_length = new_shape
        other_shape = old_shape
    
    return minimal_length + (1,) * (len(other_shape) - len(minimal_length))

def shape_to_axis(old_shape,new_shape):
    # taken from https://github.com/geohot/tinygrad/blob/master/tinygrad/runtime/ops_cpu.py
    if len(old_shape) == 0 or len(new_shape) == 0: return None
    if len(old_shape) < len(new_shape):
        old_shape = add_dimensions(old_shape, new_shape)
    elif len(old_shape) > len(new_shape):
        new_shape = add_dimensions(old_shape, new_shape)
    size = tuple(i for i,(a,b) in enumerate(zip(old_shape, new_shape)) if a != b)
    print(size)
    return size

class Tensor():

    def __init__(self, data: np.array, requires_grad: bool = False, parent = (), op='', name='') -> None:
        self.data : np.array = data
        self.requires_grad : bool = requires_grad
        self.grad : np.array = np.zeros_like(data) if requires_grad else None
        self._backward = lambda: None
        self.parent = parent
        self.op = op
        self.name = name
        self.init_name = name

    def __repr__(self):
        return f"Tensor({self.data})"

    def __getitem__(self,val):
        output = Tensor(
            self.data[val],
            requires_grad=True,
            parent=(self,),
            op="getitem",
            name=f"{self.init_name}[{val}]"
        )
        def _backward():
            self.grad[val] += output.grad
        output._backward = _backward
        return output
    

    def __setitem__(self, index, value):
        self.data[index] = value.data
        self.parent = (*self.parent, value)
        self.op = "setitem"
        self.name = f"{self.init_name}[{index}]"

        def _backward():
            value._backward()
            value.grad += self.grad[index]  # Accumulate the gradients in `self.value.grad` instead of assigning directly

        if not hasattr(self, "_backward"):
            self._backward = _backward
        else:
            old_backward = self._backward
            def new_backward():
                old_backward()
                _backward()
            self._backward = new_backward



    def __neg__(self):
        return self * -1
    
    def __add__(self, other: Tensor) -> Tensor:
        other = other if isinstance(other,Tensor) else Tensor(other, False)
        output = Tensor(
            self.data + other.data,
            requires_grad= True if self.requires_grad else False,
            parent = (self,other),
            op = "add",
            name=f"{self.name} + {other.name}"
            )
        def _backward():
            self.grad += output.grad
            if other.requires_grad:
                other.grad += output.grad if other.shape == output.shape else \
                    output.grad.sum(axis=shape_to_axis(self.shape, other.shape), keepdims=True).reshape(other.grad.shape)
        output._backward = _backward
        return output
    

    def __radd__(self,other: Tensor) -> Tensor:
        return self + other
    
    def __sub__(self,other: Tensor) -> Tensor:
        return self + (-other)
    
    def __rsub__(self, other: Tensor) -> Tensor:
        return other + (-self)
    
    def __mul__(self,other: Union[Tensor, int, float]) -> Tensor:
        """Element-wise multiplication

        Args:
            other (Union[Tensor, int, float]): If a float or an int, perform regular scalar
            multiplication. If a tensor, perform element-wise multiplication.

        Returns:
            Tensor: New Tensor
        """
        if isinstance(other, (int,float)):
            output =  Tensor(
                other * self.data,
                requires_grad= True if self.requires_grad else False,
                parent = (self,),
                op="mul",
                name=f"{self.name} * {other}"            
                )
        elif isinstance(other, Tensor):
            output = Tensor(
                self.data * other.data,
                requires_grad=True if self.requires_grad else False,
                parent = (self,other),
                op="mul",
                name=f"{self.name} * {other.name}"            
            )
        else:
            raise ValueError(f"Cannot multiply a tensor with a {type(other)}")

        def _backward():
            if isinstance(other, (int,float)):
                self.grad += other * output.grad
            if isinstance(other, Tensor):
                self.grad += other.data * output.grad
                other.grad += self.data * output.grad
        
        output._backward = _backward
        return output
    
    def __rmul__(self,other):
        return self * other
    

    def __matmul__(self,other: Tensor) -> Tensor:
        output = Tensor(
            self.data @ other.data,
            requires_grad = True if self.requires_grad else False,
            parent = (self,other),
            op = "matmul"
        )
        def _backward():
            self.grad += output.grad @ other.data.T 
            other.grad += self.data.T @ output.grad
        output._backward = _backward
        return output

    def __rmatmul__(self,other: Tensor) -> Tensor:
        return other @ self
    
    def __pow__(self,power: Union[int,float]) -> Tensor:
        assert isinstance(power, (int,float))
        output = Tensor(
            self.data ** power,
            requires_grad= True if self.requires_grad else False,
            parent = (self,),
            op="pow"
        )
        def _backward():
            #works because of numpy's broadcasting
            self.grad += power * self.data ** (power - 1) * output.grad
        output._backward = _backward
        return output

    def __truediv__(self, other: Union[int,float]) ->Tensor:
        assert isinstance(other, (int,float))
        return self * other ** (-1)
    
    def transpose(self, dim0:int, dim1: int) -> Tensor:
        output = Tensor(
            np.transpose(self.data, (dim1,dim0)),
            requires_grad=True if self.requires_grad else False,
            parent = (self,),
            op="transpose"
        )
        def _backward():
            self.grad += output.grad.T
        output._backward = _backward
        return output

    def pad(self, pad: Union[tuple[tuple], int], value = None) -> Tensor:
        output = Tensor(
            np.pad(self.data, pad, constant_values = value if value else 0),
            True,
            parent = (self,),
            op = "pad"
        )
        def _backward():
            slices = [slice(p[0], -p[1] if p[1] != 0 else self.shape[i], None) for i,p in enumerate(pad)]
            self.grad += output.grad[np.s_[tuple(slices)]]
        output._backward = _backward
        return output        
    
    


    def sum(self) -> Tensor:
        output = Tensor(
            np.sum(self.data),
            requires_grad=True if self.requires_grad else False,
            parent = (self,),
            op="sum",
            name=f"sum({self.name})"
        )
        def _backward():
            assert output.grad.shape == ()
            self.grad += np.full(self.data.shape, output.grad.item())
        output._backward = _backward
        return output

    def mean(self) -> Tensor:
        return self.sum() / self.data.size

    def max(self) -> Tensor:
        max_locations = np.argwhere(self.data == np.max(self.data))  
        output = Tensor(
            self.data[max_locations[0,0], max_locations[0,1]],
            requires_grad=True if self.requires_grad else False,
            parent = (self,),
            op="max"
        )
        def _backward():
            grad_matrix = np.zeros(self.data.shape)
            div = np.sum(max_locations)
            grad_matrix[max_locations[:,0], max_locations[:,1]] = 1/div
            self.grad += grad_matrix * output.grad
        output._backward = _backward
        return output

    def relu(self) -> Tensor:
        output = Tensor(
            np.where(self.data > 0, self.data, 0),
            requires_grad=True if self.requires_grad else False,
            parent = (self,),
            op="relu"
        )
        def _backward():
            self.grad += np.where(self.data > 0, output.grad, 0)
        output._backward = _backward
        return output

    def exp(self) -> Tensor:
        output = Tensor(
            np.exp(self.data),
            requires_grad=True if self.requires_grad else False,
            parent = (self,),
            op="exp"
        )
        def _backward():
            self.grad += np.exp(self.data) * output.grad
        output._backward = _backward

        return output

    def log(self) -> Tensor:
        output = Tensor(
            np.log(self.data),
            requires_grad=True if self.requires_grad else False,
            parent = (self,),
            op="log"
        )
        def _backward():
            self.grad += self.data ** (-1) * output.grad
        output._backward = _backward
        return output

    def backward(self):
        topo_order = []
        visited = set()
        self.__build_topological_sort(self, visited, topo_order)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo_order):
            v._backward()
    
    def __build_topological_sort(self, v, visited, topo_order):
        if v not in visited:
            visited.add(v)
            for parent in v.parent:
                self.__build_topological_sort(parent, visited, topo_order)
            topo_order.append(v)

    @property
    def shape(self) -> Tuple:
        return self.data.shape
    
    @property
    def T(self) -> Tensor:
        return self.transpose(0,1)

    @staticmethod
    def random(dim: Tuple, requires_grad : bool = True, name='')->Tensor:
        return Tensor(
            np.random.randn(*dim),
            requires_grad,
            op="random",
            name=name
        )
    
    @staticmethod
    def zeros(dim: tuple, requires_grad : bool = True, name='') -> Tensor:
        return Tensor(
            np.zeros(dim),
            requires_grad,
            op="zeros",
            name = name
        )