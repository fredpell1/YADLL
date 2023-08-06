from yadll.tensor.autodiff import *
import numpy as np
import torch

def test_getitem_backward_pass():
    x = Tensor(np.array([[1.0,2,3],
                    [4,5,6],
                    [7,8,9]]), requires_grad=True)
    y = x[:1,:]
    z = y.mean() + x[0,0] + x[0,0]
    z.backward()
    torch_x = torch.tensor([[1,2,3],
                    [4,5,6],
                    [7,8,9]], dtype=torch.float64, requires_grad=True)
    torch_y = torch_x[:1, :] + torch_x[0,0] + torch_x[0,0]
    torch_z = torch_y.mean()
    torch_z.backward()
    assert np.all(x.grad == torch_x.grad.detach().numpy())

def test_setitem_backward_pass():
    x = Tensor(np.array([[1.0,2,3],
                  [4,5,6],
                  [7,8,9]]), requires_grad=True)
    y = Tensor.zeros(x.shape, True)
    y[0,0] = x[0,0]
    y = y**2
    y[0,1] = x[1,1]
    z = y.mean()
    z.backward()
    torch_x = torch.tensor([[1,2,3],
                  [4,5,6],
                  [7,8,9]], dtype=torch.float64, requires_grad=True)
    torch_y = torch.zeros_like(torch_x)
    torch_y[0,0] = torch_x[0,0]
    torch_y = torch_y ** 2
    torch_y[0,1] = torch_x[1,1]
    torch_z = torch_y.mean()
    torch_x.retain_grad()
    torch_z.backward()
    print(x.grad, torch_x.grad)
    assert np.all(x.grad == torch_x.grad.detach().numpy())

def test_add_scalar_backward_pass():
    x = Tensor(np.array([[1.0,2], [3,4]]), requires_grad=True)
    b = Tensor(np.array([2.0]), requires_grad=True)
    y = (x + b).sum() * 3
    y.backward()
    torch_x = torch.tensor(x.data, requires_grad=True)
    torch_b = torch.tensor(b.data, requires_grad=True)
    torch_y = (torch_x + torch_b).sum() * 3
    torch_y.backward()
    assert np.all(b.grad == torch_b.grad.detach().numpy())

def test_add_tensor_same_shape_backward_pass():
    x = Tensor(np.array([[1.0,2], [2,3]]), requires_grad=True)
    y = Tensor(np.array([[1.0,1], [3.2, 4.3]]), requires_grad=True)
    z = (x + y).sum() ** 2
    z.backward()
    
    torch_x = torch.tensor([[1.0,2],[2,3]], requires_grad=True)
    torch_y = torch.tensor([[1.0,1], [3.2, 4.3]], requires_grad=True)
    torch_x.retain_grad()
    torch_y.retain_grad()
    torch_z = (torch_x + torch_y).sum() ** 2
    torch_z.backward()
    assert np.all(x.grad == torch_x.grad.detach().numpy())
    assert np.all(y.grad == torch_y.grad.detach().numpy())

def test_add_tensor_row_broadcast_backward_pass():
    x = Tensor(np.array([[1.0,2], [2,3]]), requires_grad=True)
    y = Tensor(np.array([1.0,1]), requires_grad=True)
    z = (x + y).sum() ** 2
    z.backward()
    
    torch_x = torch.tensor([[1.0,2],[2,3]], requires_grad=True)
    torch_y = torch.tensor([1.0,1], requires_grad=True)
    torch_x.retain_grad()
    torch_y.retain_grad()
    torch_z = (torch_x + torch_y).sum() ** 2
    torch_z.backward()
    assert np.all(x.grad == torch_x.grad.detach().numpy())
    assert np.all(y.grad == torch_y.grad.detach().numpy())

def test_add_tensor_col_broadcast_backward_pass():
    x = Tensor(np.array([[1.0,2], [2,3]]), requires_grad=True)
    y = Tensor(np.array([[1.0],[1]]), requires_grad=True)
    z = (x + y).sum() ** 2
    z.backward()
    
    torch_x = torch.tensor([[1.0,2],[2,3]], requires_grad=True)
    torch_y = torch.tensor([[1.0],[1]], requires_grad=True)
    torch_x.retain_grad()
    torch_y.retain_grad()
    torch_z = (torch_x + torch_y).sum() ** 2
    torch_z.backward()
    assert np.all(x.grad == torch_x.grad.detach().numpy())
    assert np.all(y.grad == torch_y.grad.detach().numpy())


def test_mul_scalar_backward_pass():
    a = Tensor(np.array([[1,2],[2,1]]), requires_grad = True)
    b = (a * 2).sum() ** 2
    b.backward()
    c = torch.tensor([[1.0,2],[2,1]], requires_grad=True)
    d = (c * 2).sum() ** 2
    d.backward() #need to pass a tensor because d has more than one element
    assert np.all(a.grad == c.grad.numpy())
    
#TODO: test mul with broadcasting
def test_mul_tensor_same_shape_backward_pass():
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

def test_batch_matmul_backward_pass():
    a = Tensor.random((4,3,2,2))
    b = Tensor.random((4,3,2,3))
    c = (a @ b).sum()
    torch_a = torch.tensor(a.data, requires_grad=True)
    torch_b = torch.tensor(b.data, requires_grad=True)
    torch_c = (torch_a @ torch_b).sum()
    c.backward()
    torch_c.backward()
    assert np.all(a.grad == torch_a.grad.detach().numpy())

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
    a = Tensor(np.random.rand(3,4,2), True)
    b = a.transpose(0,1)
    c = b.transpose(0,2)
    d = c.sum()
    d.backward()
    torch_a = torch.tensor(a.data, requires_grad = True)
    torch_b = torch_a.transpose(0,1)
    torch_c = torch_b.transpose(0,2)
    torch_d = torch_c.sum()
    torch_a.retain_grad()
    torch_b.retain_grad()
    torch_c.retain_grad()
    torch_d.backward()
    assert np.all(a.grad == torch_a.grad.detach().numpy()), "a.grad incorrect"
    assert np.all(b.grad == torch_b.grad.detach().numpy()), "b.grad incorrect"
    assert np.all(c.grad == torch_c.grad.detach().numpy()), "c.grad incorrect"

def test_sum_backward_pass():
    a = Tensor(np.array([[1.0,2],[2,1]]), requires_grad=True)
    b = a.sum()
    b.backward()
    c = torch.tensor([[1.0,2], [2,1]], requires_grad=True)
    d = c.sum()
    d.backward()
    assert np.all(a.grad == c.grad.numpy())

def test_sum_with_axis_forward_pass():
    a = Tensor.random((3,3,28,28))
    b = a.sum(1)
    torch_a = torch.tensor(a.data)
    torch_b = torch_a.sum(1)
    assert b.shape == torch_b.shape, "shape not equal"
    assert np.all(abs(b.data - torch_b.detach().numpy()) < 0.00000001)


def test_sum_with_multiple_axis_forward_pass():
    a = Tensor.random((3,3,28,28))
    b = a.sum((1,3))
    torch_a = torch.tensor(a.data)
    torch_b = torch_a.sum((1,3))
    assert b.shape == torch_b.shape, "shape not equal"
    assert np.all(abs(b.data - torch_b.detach().numpy()) < 0.00000001)


def test_sum_with_axis_backward_pass():
    a = Tensor.random((3,3,28,28))
    b = a.sum((1,3))
    torch_a = torch.tensor(a.data, requires_grad=True)
    torch_b = torch_a.sum((1,3))
    b.backward()
    torch_b.backward(torch.ones_like(torch_b))
    assert np.all(abs(a.grad - torch_a.grad.detach().numpy()) < 0.00000001)

def test_mean_backward_pass():
    a = Tensor(np.array([[1.0,2],[2,1]]), requires_grad=True)
    b = a.mean()
    b.backward()
    c = torch.tensor([[1.0,2],[2,1]], requires_grad=True)
    d = c.mean()
    d.backward(torch.ones_like(d))
    assert np.all(a.grad == c.grad.numpy())

def test_mean_with_keepdim():
    a = Tensor.random((3,3,28,28))
    b = a.mean(2, keepdim=True)
    torch_a = torch.tensor(a.data)
    torch_b = torch_a.mean(2, keepdim=True)
    assert np.all(abs(b.data - torch_b.numpy()) < 1e-7)

def test_mean_with_keepdim_backward_pass():
    a = Tensor.random((3,3,28,28))
    b = a.mean((1,3), keepdim=True)
    torch_a = torch.tensor(a.data, requires_grad=True)
    torch_b = torch_a.mean((1,3), keepdim=True)
    b.backward()
    torch_b.retain_grad()
    torch_b.backward(torch.ones_like(torch_b))
    assert np.all(abs(a.grad - torch_a.grad.detach().numpy()) < 1e-7)

def test_mean_with_axis_forward_pass():
    a = Tensor.random((1,1,10,10))
    b = a.mean(2)
    torch_a = torch.tensor(a.data)
    torch_b = torch_a.mean(2)
    assert b.shape == torch_b.shape, "shape not equal"
    assert np.all(abs(b.data - torch_b.detach().numpy()) < 0.0000001)

def test_mean_with_multiple_axis_forward_pass():
    a = Tensor.random((3,3,28,28))
    b = a.mean((1,3))
    torch_a = torch.tensor(a.data, requires_grad=True)
    torch_b = torch_a.mean((1,3))
    assert b.shape == torch_b.shape, "shape not equal"
    assert np.all(abs(b.data - torch_b.detach().numpy()) < 0.0000001)


def test_mean_with_axis_backward_pass():
    a = Tensor.random((3,3,28,28))
    b = a.mean((1,3))
    torch_a = torch.tensor(a.data, requires_grad=True)
    torch_b = torch_a.mean((1,3))
    b.backward()
    torch_b.backward(torch.ones_like(torch_b))
    assert np.all(abs(a.grad - torch_a.grad.numpy()) < 0.00000001)

def test_var_no_dim_forward_pass():
    a = Tensor.random((3,3,28,28))
    b = a.var()
    torch_a = torch.tensor(a.data)
    torch_b = torch_a.var(unbiased=False)
    assert b.shape == torch_b.shape, "shapes not equal"
    assert np.all(abs(b.data - torch_b.numpy()) < 1e-8)

def test_var_dim_forward_pass():
    a = Tensor.random((3,3,28,28))
    b = a.var((0,2,3))
    torch_a = torch.tensor(a.data)
    torch_b = torch_a.var((0,2,3),unbiased=False)
    assert b.shape == torch_b.shape, "shapes not equal"
    assert np.all(abs(b.data - torch_b.numpy()) < 1e-7)

def test_var_dim_backward_pass():
    a = Tensor.random((3,3,28,28))
    b = a.var(2)
    torch_a = torch.tensor(a.data, requires_grad=True)
    torch_b = torch_a.var(2,unbiased=False)
    b.backward()
    torch_b.backward(torch.ones_like(torch_b))
    assert np.all(abs(a.grad - torch_a.grad.numpy()) < 0.0000001)

def test_max_forward_pass():
    a = Tensor.random((2,3,4,4))
    torch_a = torch.tensor(a.data)
    assert a.max().data == torch_a.max().detach().numpy()

def test_max_backward_pass():
    a = Tensor(np.array([[1.0,2],[2,1]]), requires_grad=True)
    b = a.max() * 3
    b.backward()
    c = torch.tensor([[1.0,2],[2,1]], requires_grad=True)
    d = c.max() * 3
    d.backward()
    assert np.all(a.grad == c.grad.numpy())

def test_max_with_axis_forward_pass():
    a = Tensor.random((3,3,28,28))
    b = a.max(2)
    torch_a = torch.tensor(a.data, dtype=torch.float64)
    torch_b = torch_a.max(2)[0]
    assert b.shape == torch_b.shape, "shape not equal"
    assert np.all(abs(b.data - torch_b.detach().numpy()) < 0.00000001), "result not equal"

def test_max_with_axis_backward_pass():
    a = Tensor.random((3,3,28,28))
    b = a.max(-1).max(-1)
    b.backward()
    torch_a = torch.tensor(a.data, requires_grad=True)
    torch_b = torch_a.max(-1)[0].max(-1)[0]
    torch_b.backward(torch.ones_like(torch_b))
    assert np.all(abs(a.grad - torch_a.grad.numpy()) < 0.00000001)

def test_relu_backward_pass():
    a = Tensor(np.array([[-1.0,2], [-3,2]]), requires_grad=True)
    b = a.relu() * 2
    b.backward()
    c = torch.tensor([[-1.0,2], [-3,2]], requires_grad=True)
    d = c.relu() * 2
    d.backward(torch.ones_like(d))
    assert np.all(a.grad == c.grad.numpy())

def test_exp_backward_pass():
    a = Tensor(np.array([[-1.0,2], [-3,2]]), requires_grad=True)
    b = a.exp() * 2
    b.backward()
    c = torch.tensor([[-1.0,2], [-3,2]], requires_grad=True, dtype=torch.float64)
    d = c.exp() * 2.0
    d.backward(torch.ones_like(d))
    assert np.all(abs(a.grad - c.grad.numpy()) < 0.00000001)

def test_log_backward_pass():
    a = Tensor(np.array([[1.0,2], [3,2]]), requires_grad=True)
    b = a.log() * 2
    b.backward()
    c = torch.tensor([[1.0,2], [3,2]], requires_grad=True, dtype=torch.float64)
    d = c.log() * 2
    d.backward(torch.ones_like(d))
    assert np.all(a.grad == c.grad.numpy())

def test_pad_backward_pass():
    a = Tensor(np.array([[1.0,2], [3,2]]), requires_grad=True)
    b = a.pad(((1,1),(2,2)))
    c = b.mean() + a[0,0]
    c.backward()
    torch_a = torch.tensor([[1.0,2], [3,2]], requires_grad=True, dtype=torch.float64)
    torch_b = torch.nn.functional.pad(torch_a, (1,1,2,2))
    torch_c = torch_b.mean() + torch_a[0,0]
    torch_c.backward()
    assert np.all(a.grad ==  torch_a.grad.detach().numpy())


def test_permute_backward_pass():
    a = Tensor(np.random.rand(3,4,2), True)
    b = a.permute((1,2,0))
    c = b.permute((2,0,1)) * 3
    d = c.mean()
    d.backward()
    torch_a = torch.tensor(a.data, requires_grad = True)
    torch_b = torch_a.permute((1,2,0))
    torch_c = torch_b.permute((2,0,1)) * 3
    torch_d = torch_c.mean()
    torch_a.retain_grad()
    torch_b.retain_grad()
    torch_c.retain_grad()
    torch_d.backward()
    assert np.all(a.grad == torch_a.grad.detach().numpy()), "a.grad incorrect"
    assert np.all(b.grad == torch_b.grad.detach().numpy()), "b.grad incorrect"
    assert np.all(c.grad == torch_c.grad.detach().numpy()), "c.grad incorrect"

def test_reshape_backward_pass():
    a = Tensor(np.random.rand(3,4,2), True)
    b = a.reshape((3,8))
    c = b.reshape((3,2,4))
    d = c.sum()
    d.backward()
    torch_a = torch.tensor(a.data, requires_grad = True)
    torch_b = torch_a.reshape((3,8))
    torch_c = torch_b.reshape((3,2,4))
    torch_d = torch_c.sum()
    torch_a.retain_grad()
    torch_b.retain_grad()
    torch_c.retain_grad()
    torch_d.backward()
    assert np.all(a.grad == torch_a.grad.detach().numpy()), "a.grad incorrect"
    assert np.all(b.grad == torch_b.grad.detach().numpy()), "b.grad incorrect"
    assert np.all(c.grad == torch_c.grad.detach().numpy()), "c.grad incorrect"
    
def test_flatten_backward_pass():
    a = Tensor.random((2,1,4,3,4,2,4))
    b = a.flatten(3).mean()
    b.backward()
    torch_a = torch.tensor(a.data, requires_grad=True)
    torch_b = torch_a.flatten(3).mean()
    torch_b.backward()
    assert np.all(a.grad == torch_a.grad.detach().numpy()), "a.grad incorrect"

def test_expand_backward_pass():
    a = Tensor.random((3,1))
    b = a.expand((3,3,3))
    c = b.mean()
    c.backward()
    torch_a = torch.tensor(a.data, requires_grad=True)
    torch_b = torch_a.expand(3,3,3)
    torch_c = torch_b.mean()
    torch_b.retain_grad()
    torch_c.backward()
    assert np.all(a.grad == torch_a.grad.detach().numpy()), "a.grad incorrect"
    assert np.all(b.grad == torch_b.grad.detach().numpy()), "b.grad incorrect"
    

def test_squeeze_backward_pass():
    a = Tensor.random((1,3,1,4,2,1))
    b = a.squeeze((0,2))
    c = b.mean()
    c.backward()
    torch_a = torch.tensor(a.data, requires_grad=True)
    torch_b = torch_a.squeeze(0,2)
    torch_c = torch_b.mean()
    torch_b.retain_grad()
    torch_c.backward()
    assert np.all(a.grad == torch_a.grad.detach().numpy()), "a.grad incorrect"
    assert np.all(b.grad == torch_b.grad.detach().numpy()), "b.grad incorrect"
    
def test_as_strided_forward_pass():
    x = Tensor.random((1,1,5,5), dtype = np.float32)
    torch_x = torch.from_numpy(x.data)
    new_shape = (1,1,3,3,1,1,3,3)
    strides = (100,100,20,4,100,100,20,4)
    torch_strides = tuple(s//4 for s in strides) #needed because of the different mechanism numpy and torch use to store data
    out = x.as_strided(new_shape, strides)
    torch_out = torch_x.as_strided(new_shape, torch_strides)
    assert np.all(out.shape == torch_out.shape), "shapes not equal"
    assert np.all(out.data == torch_out.numpy()), "output not equal" 

def test_as_strided_backward_pass():
    x = Tensor.random((1,1,5,5), dtype = np.float32)
    torch_x = torch.tensor(x.data, requires_grad=True)
    new_shape = (1,1,3,3,1,1,3,3)
    strides = (100,100,20,4,100,100,20,4)
    torch_strides = tuple(s//4 for s in strides) #needed because of the different mechanism numpy and torch use to store data
    out = (x.as_strided(new_shape, strides) ** 2).max() * 10
    torch_out = (torch_x.as_strided(new_shape, torch_strides) ** 2).max() * 10
    out.backward()
    torch_out.backward()
    print(x.grad)
    print(torch_x.grad)
    assert(np.all(abs(x.grad - torch_x.grad.detach().numpy()) < 1e-6))

def test_cat_forward_pass():
    tensors = [Tensor.random((1,1,5+i,5)) for i in range(3)]
    big_tensor = Tensor.cat(tensors, 2)
    torch_tensors = [torch.tensor(tensor.data) for tensor in tensors]
    torch_big_tensor = torch.cat(torch_tensors, 2)
    assert big_tensor.shape == torch_big_tensor.shape, "shape is wrong"
    assert np.all(big_tensor.data == torch_big_tensor.numpy()), "output is wrong"

def test_cat_backward_pass():
    tensors = [Tensor.random((1,1,5+i,5)) for i in range(3)]
    big_tensor = Tensor.cat(tensors, 2)
    torch_tensors = [torch.tensor(tensor.data, requires_grad=True) for tensor in tensors]
    torch_big_tensor = torch.cat(torch_tensors, 2)
    out = big_tensor.mean()
    torch_out = torch_big_tensor.mean()
    out.backward()
    torch_out.backward()

def test_unsqueeze_forward_pass():
    x = Tensor.random((3,3,2,2))
    x_unsqueezed = x.unsqueeze(1)
    assert x_unsqueezed.shape == (3,1,3,2,2), "shape incorrect"
    assert np.all(x_unsqueezed[:,0,:,:,:].data == x.data), "data incorrect"

def test_unsqueeze_backward_pass():
    x = Tensor.random((3,3,2,2))
    torch_x = torch.tensor(x.data, requires_grad=True)
    out = x.unsqueeze(1).sum()
    torch_out = torch_x.unsqueeze(1).sum()
    out.backward()
    torch_out.backward()
    assert np.all(x.grad == torch_x.grad.detach().numpy()), "grad incorrect"

def test_unfold_forward_pass():
    x = Tensor.random((4,4,5,5))
    torch_x = torch.tensor(x.data)
    unfolded = x.unfold(1,2,1)
    torch_unfolded = torch_x.unfold(1,2,1)
    assert unfolded.shape == torch_unfolded.shape, "shape incorrect"
    assert np.all(unfolded.data == torch_unfolded.numpy()), "output incorrect"

def test_unfold_backward_pass():
    x = Tensor.random((4,4,5,5))
    torch_x = torch.tensor(x.data, requires_grad=True)
    unfolded = x.unfold(1,2,1)
    torch_unfolded = torch_x.unfold(1,2,1)
    out = unfolded.sum()
    torch_out = torch_unfolded.sum()
    out.backward()
    torch_out.backward()
    assert np.all(x.grad == torch_x.grad.detach().numpy()), "grad incorrect"