# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 11:57:31 2021

@author: CC-i7-1065G7
"""
# 《动手学深度学习》 http://tangshusen.me/Dive-into-DL-PyTorch/#/
# https://pytorch.org/docs/stable/torch.html
# 数学运算整理到 digamma 
import torch
import numpy
"""
1.list,numpy,tensor相互转化

                             tensor
                         ↗          ↘↖
torch.as_tensor(list1) ↗              ↘↖torch.as_tensor(arr1)
torch.Tensor(list1)  ↗                  ↘↖torch.from_numpy(arr1)
torch.tensor(list1)↗       tensor1.numpy()↘↖torch.Tensor(arr1) 
                 ↗                          ↘↖torch.tensor(arr1)
               ↗       array1.tolist()        ↘↖
            list      <-----------------        numpy
                      ----------------->
                      numpy.array(list1)
"""
# 1.1.numpy <-> tensor
# !! Tensors on the CPU and NumPy arrays can share their underlying memory location,
# and changing one will change the other. !!
arr1 = numpy.array([1,1,1])
arr2tensor1 = torch.from_numpy(arr1)    #共享内存
arr2tensor2 = torch.as_tensor(arr1)     #共享内存
arr2tensor3 = torch.tensor(arr1)        #深拷贝
arr2tensor4 = torch.Tensor(arr1)        #深拷贝
tensor2arr1 = arr2tensor1.numpy()       #共享内存
print('Before modify:', arr2tensor1)
print('Before modify:', arr2tensor2)
print('Before modify:', arr2tensor3)
print('Before modify:', arr2tensor4)
print('Before modify:', tensor2arr1)
arr1 += 1
print('After modify:', arr2tensor1)
print('After modify:', arr2tensor2)
print('After modify:', arr2tensor3)
print('After modify:', arr2tensor4)
print('After modify:', tensor2arr1)

# 1.2. list -> tensor
list1 = [1,1,1]
list2tensor1 = torch.as_tensor(list1)   #保留原数据类型
list2tensor2 = torch.Tensor(list1)      #保存为float
list2tensor3 = torch.tensor(list1)      #保留原数据类型
print(list2tensor1, list2tensor1.dtype)
print(list2tensor2, list2tensor2.dtype)
print(list2tensor3, list2tensor3.dtype)

# torch.tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False)
tensor1 = torch.tensor([[1,2,3],[4,5,6],[7,8,9]], dtype=torch.float)
print(tensor1)
print( type(tensor1) )
print('is a tensor?', torch.is_tensor(tensor1) )
print( tensor1.dtype )
"""
Data type                | dtype                         | Legacy Constructors
32-bit floating point    | torch.float32 or torch.float  | torch.*.FloatTensor
64-bit floating point    | torch.float64 or torch.double | torch.*.DoubleTensor
64-bit complex           | torch.complex64 or torch.cfloat
128-bit complex          | torch.complex128 or torch.cdouble
16-bit floating point 1  | torch.float16 or torch.half   | torch.*.HalfTensor
16-bit floating point 2  | torch.bfloat16                | torch.*.BFloat16Tensor
8-bit integer (unsigned) | torch.uint8                   | torch.*.ByteTensor
8-bit integer (signed)   | torch.int8                    | torch.*.CharTensor
16-bit integer (signed)  | torch.int16 or torch.short    | torch.*.ShortTensor
32-bit integer (signed)  | torch.int32 or torch.int      | torch.*.IntTensor
64-bit integer (signed)  | torch.int64 or torch.long     | torch.*.LongTensor
Boolean                  | torch.bool                    | torch.*.BoolTensor
"""
print(torch.tensor([1], dtype = torch.complex64), 'is complex?', torch.is_complex(torch.tensor([1], dtype = torch.complex64)))
print(torch.tensor([1], dtype = torch.complex128), 'is complex?', torch.is_complex(torch.tensor([1], dtype = torch.complex128)))
print(torch.tensor([1], dtype = torch.float32), 'is float?', torch.is_floating_point(torch.tensor([1], dtype = torch.float32)))
print(torch.tensor([1], dtype = torch.float64), 'is float?', torch.is_floating_point(torch.tensor([1], dtype = torch.float64)))
print(torch.tensor([1], dtype = torch.float16), 'is float?', torch.is_floating_point(torch.tensor([1], dtype = torch.float16)))
print(torch.tensor([1], dtype = torch.bfloat16), 'is float?', torch.is_floating_point(torch.tensor([1], dtype = torch.bfloat16)))

print( tensor1.device )
#-----------------------------------------------------------------------------#
# Creation Ops
tensor1 = torch.tensor([[1,2,3],[4,5,6],[7,8,9]], dtype=torch.float)

# 1.基本
"""
torch.zeros(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
"""
print( torch.zeros(5, 3) )
print( '<=>', torch.zeros((5,3)) )
"""
torch.zeros_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)
"""
print( torch.zeros_like(tensor1) )
"""
torch.ones(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
"""
print( torch.ones(5, 3) )
"""
torch.ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)
"""
print( torch.ones_like(tensor1) )
"""
torch.eye(n, m=None, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
对角线为1
"""
print( torch.eye(5, 3) )
"""
torch.empty(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False) 
生成一个空的tensor(每个元素为随机的极其小的正值，从而不影响数值计算）
"""
print( torch.empty(5, 3) )
"""
torch.empty_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)
"""
print( torch.empty_like(tensor1) )
"""
torch.full(size, fill_value, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
Creates a tensor of size size filled with fill_value
"""
print( torch.full(size=(2, 3), fill_value=3.141592) )
"""
torch.full_like(input, fill_value, *, dtype=None, layout=torch.strided, device=None, requires_grad=False, memory_format=torch.preserve_format)
Creates a tensor of size size filled with fill_value
"""
print( torch.full_like(tensor1, fill_value=3.141592) )


# 2.生成序列
"""
2.1.
torch.arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
torch.range() will be replaced by torch.arange()
"""
print( torch.arange(0, 10, 1) )
"""
2.2.
torch.linspace(start, end, steps, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
"""
print( torch.linspace(start=0, end=10, steps=11) )
"""
2.3.
torch.logspace(start, end, steps, base=10.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
from base**start to base**end
"""
print( torch.logspace(start=0, end=10, steps=11) )
print( torch.logspace(start=0, end=10, steps=11, base=2) )
"""
2.4.
torch.randperm(n, *, generator=None, out=None, dtype=torch.int64, layout=torch.strided, device=None, requires_grad=False, pin_memory=False)
0～n-1 的随机排列
"""
print( torch.randperm(4) ) 

 
# 3.均分分布抽样
"""
torch.rand(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
[0,1）随机小数
"""
print( torch.rand(5, 3) )
"""
torch.rand_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)
[0,1）随机小数
"""
print( torch.rand_like(tensor1) )
"""
torch.randint(low=0, high, size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
size is tuple
[low, high）随机整数
"""
print( torch.randint(low=5,high=10,size=(2,2)) )
print( torch.randint(high=10,size=(2,2)) )
"""
torch.randint_like(input, low=0, high, *, dtype=None, layout=torch.strided, device=None, requires_grad=False, memory_format=torch.preserve_format)
size is tuple
[low, high）随机整数
"""
print( torch.randint_like(tensor1,low=5,high=10) ) #[low, high）随机整数，均匀分布
print( torch.randint_like(tensor1,high=10) ) #[low, high）随机整数，均匀分布


# 4.正态分布抽样
"""
4.1.
torch.randn(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
标准正态分布，均值为0，方差为1
"""
print( torch.randn(5, 3) )
"""
4.2.
torch.randn_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)
标准正态分布，均值为0，方差为1
"""
print( torch.rand_like(tensor1) )
"""
4.3.
torch.normal(mean, std, *, generator=None, out=None)
case 1: mean is float and std is float
    torch.normal(mean, std, size, *, out=None)
case 2: mean is float and std is tensor
    torch.normal(mean=0.0, std, *, out=None)
    mean (float, optional) – the mean for all distributions
    std (Tensor) – the tensor of per-element standard deviations
case 3: mean is tensor and std is float
    torch.normal(mean, std=1.0, *, out=None)
    mean (Tensor) – the tensor of per-element means
    std (float, optional) – the standard deviation for all distributions
case 4: mean is tensor and std is tensor
    torch.normal(mean, std, *, generator=None, out=None)
    mean (Tensor) – the tensor of per-element means
    std (Tensor) – the tensor of per-element standard deviations
"""
print('case 1:', torch.normal(mean=2, std=3, size=(1, 4)) )
print('case 2:', torch.normal(mean=0.5, std=torch.arange(1., 6.)) )
print('case 3:', torch.normal(mean=torch.arange(1., 6.)) )
print('case 4:', torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1, 0, -0.1)) )


# 5.多项式分布抽样
"""
torch.multinomial(input, num_samples, replacement=False, *, generator=None, out=None)
基于多项式分布，从input中随机抽取num_samples元素，并返回去索引
input的每行之和不需一定为1，但必须是大于0的有限数
0 < num_samples <= len(input[0])
replacement=False 表示无重复抽取
replacement=True  表示可重复抽取
"""
print( torch.multinomial(torch.tensor([[0., 10., 3.],[4., 2., 8.]]), 2) )
print( torch.multinomial(torch.tensor([[0., 10., 3.],[4., 2., 8.]]), 3) )
print( torch.multinomial(torch.tensor([[0., 10., 3.],[4., 2., 8.]]), 3, replacement=True) )


# 6.0-1分布抽样
"""
torch.bernoulli(input, *, generator=None, out=None)
input的每个元素都必须在[0,1]内
"""
print( torch.bernoulli(torch.rand(5, 3)) )


# 7.泊松分布抽样
"""
torch.poisson(input, generator=None)
"""
print( torch.poisson(torch.rand(5, 3)*5) )
#-----------------------------------------------------------------------------#
# Indexing, Slicing, Joining, Mutating Ops
# 1.形状
tensor1 = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.float)
print('total number of elements:', tensor1.numel() )
print('<=>', torch.numel(tensor1) )

print('total number of dimensions:', tensor1.dim() )

print('size:', tensor1.shape)
print('<=>', tensor1.size())

print('size of dim=0:',tensor1.shape[0])
print('<=>',tensor1.size(0))
print('<=>',tensor1.size()[0])


# 2.查找/索引(不会开辟新内存)
print('first element:', tensor1[0][0].item())
print('<=>', tensor1[0,0].item())
print('First row: ',tensor1[0])
print('First column: ', tensor1[:,0])
print('Last column:', tensor1[:,-1])
print('<=>', tensor1[...,-1])
# 索引出来的结果与原数据共享内存，也即修改一个，另一个会跟着修改
tensor1[:,1] = 0
print(tensor1)
"""
2.1.
torch.take(input, index)
indices (LongTensor)
"""
print( torch.take(tensor1, torch.tensor([0,2,5,7])) )
"""
2.2.
torch.masked_select(input, mask, *, out=None)
Returns a new 1-D tensor which indexes the input tensor according to the boolean mask mask which is a BoolTensor.
返回一个新的一维张量，该张量根据布尔模板掩码（即BoolTensor）对输入张量进行索引。
mask (BoolTensor)
"""
mask = torch.tensor([[True,False,True,False],[False,True,False,True]])
print( torch.masked_select(tensor1, mask) )
"""
2.3.
torch.nonzero(input, *, out=None, as_tuple=False)
非0位置索引
"""
print( torch.nonzero(torch.tensor([1, 1, 1, 0, 1])) )
print( torch.nonzero(torch.tensor([[0.6, 0.0],[0.0, 0.4],[0.0, 0.0],[0.0, -0.4]])) )
"""
2.4.
torch.where(condition, x, y)
condition (BoolTensor); x (Tensor or Scalar); y (Tensor or Scalar)
out[i] = { x[i], if condition[i] = True
         | y[i], otherwise
"""
tensorX = torch.tensor([[-0.462, 0.3139],[0.3898, -0.7197]], dtype=torch.float64)
tensorY = torch.zeros_like(tensorX)
print( torch.where(tensorX > 0, tensorX, tensorY) )
print( torch.where(tensorX > 0, tensorX, -2.) )


# 3.变形
"""
torch.reshape(input, shape) <=> tensorX.view(shape)
shape (tuple of python:ints)
"""
tensor1 = torch.arange(1, 25)
print( torch.reshape(tensor1,(2,3,4)).size() )
print( '<=>', tensor1.view(2,3,4).size() )
print( torch.reshape(tensor1,(8,-1)).size() ) #-1 表示24/8
print( '<=>', tensor1.view(8,-1).size() )
print( torch.reshape(tensor1,(-1,4)).size() ) #-1 表示24/4
print( '<=>', tensor1.view(-1,4).size() )
print( torch.reshape(tensor1,(2,-1,4)).size() ) #-1 表示24/2/4
print( '<=>', tensor1.view(2,-1,4).size() )
print( torch.reshape(tensor1,(24,)).size() ) #一行
print( '<=>', tensor1.view(24,).size() )
print( torch.reshape(tensor1,(-1,)).size() ) #一行
print( '<=>', tensor1.view(-1,).size() )

# 变形虽然生成新的tensor，但是与源Tensor共享数据
print(tensor1, 'id=', id(tensor1))
tensor2 = torch.reshape(tensor1,(2,3,4))
print(tensor2, 'id=', id(tensor2))
tensor3 = tensor1.view(4,6) 
print(tensor3, 'id=', id(tensor3))
tensor2 -= 12
print('改变一个变形数据，所有tensor共享数据，一起发生变化:\n',tensor1,'\n',tensor2,'\n',tensor3)

# 如果不想修改源tensor，需要将源tensor深拷贝后再变形
print(tensor1, 'id=', id(tensor1))
tensor4 = tensor1.clone().view(3,8)
print(tensor4, 'id=', id(tensor4))
tensor4 += 12
print('深拷贝后再变形修改数据，则不会影响源数据:\n',tensor1,'\n',tensor2,'\n',tensor3,'\n',tensor4)
"""
深拷贝 
torch.clone(input, *, memory_format=torch.preserve_format)
This function is differentiable, so gradients will flow back from the result of this operation to input.
此函数是可微的，因此梯度将反向转播给input
"""
print(tensor1, id(tensor1))
tensor2 = torch.clone(tensor1)
print(tensor2, id(tensor2))
# data_tensor.copy_(data_tensor)


# 4.降维
"""
4.1.a
torch.squeeze(input, dim=None, *, out=None)
a tensor with all the dimensions of input of size 1 removed
"""
tensor1 = torch.zeros(2, 1, 3, 1, 2)
print(tensor1.size())
print( torch.squeeze(tensor1).size() )
print( torch.squeeze(tensor1, dim=0).size() )
print( torch.squeeze(tensor1, dim=1).size() )
"""
4.1.b
torch.unsqueeze(input, dim)
A dim value within the range [-input.dim() - 1, input.dim() + 1) can be used. Negative dim will correspond to unsqueeze() applied at dim = dim + input.dim() + 1.
"""
for ii in range(-tensor1.dim()-1,tensor1.dim()+1):
    print('dim='+str(ii)+':', torch.unsqueeze(tensor1, dim=ii).size() )
"""
4.2.
torch.movedim(input, source, destination)
torch.moveaxis(input, source, destination) 同上
move source-th dim to destination-th dim
source (int or tuple of python:ints); destination (int or tuple of python:ints)
"""
print( torch.movedim(tensor1, 1, 0).size() )
print( '<=>', torch.moveaxis(tensor1, 1, 0).size() )
print( torch.movedim(tensor1, (1, 2), (0, 1)).size() )
print( '<=>', torch.moveaxis(tensor1, (1, 2), (0, 1)).size() )
"""
torch.unbind(input, dim=0)
Removes a tensor dimension and Returns a tuple of all slices along a given dimension, already without it
删除tensor的一个维度,并返回一个tuple
"""
tensor1 = torch.tensor([[1,2,3],[4,5,6]])
print( torch.unbind(tensor1, dim=0) )
print( torch.unbind(tensor1, dim=1) )


# 5.转置
"""
5.1.
torch.transpose(input, dim0, dim1)
torch.swapaxes(input, axis0, axis1) 同上
torch.swapdims(input, dim0, dim1) 同上
"""
tensor1 = torch.arange(1,25).reshape(2,3,4)
print(tensor1)
print( torch.transpose(tensor1, dim0=0, dim1=1) )
print( '<=>', torch.swapaxes(tensor1, axis0=0, axis1=1) )
print( '<=>', torch.swapdims(tensor1, dim0=0, dim1=1) )

print( torch.transpose(tensor1, dim0=0, dim1=2) )
print( torch.transpose(tensor1, dim0=1, dim1=2) )
"""
5.2.
torch.t(input)
维度>=2 转置; 维度<2 不变
"""
print( torch.t(torch.tensor([1,2,3])) )
print( torch.t(torch.tensor([[1,2,3],[4,5,6]])) )
print( '<=>', torch.transpose(torch.tensor([[1,2,3],[4,5,6]]), 0, 1) )


# 6.矩阵拼接
# 6.1 cat/hstack/column_stack/vstack/row_stack/stack/dstack
tensor1 = torch.tensor([[1,2,3],[4,5,6]])
tensor2 = torch.tensor([[7,8,9],[10,11,12]])

print( torch.cat((tensor1, tensor2), dim=1) )
print( '<=>', torch.hstack((tensor1,tensor2)) ) # 水平拼接，二位或多维张量沿着dim=1拼接，一维张量沿着dim=0拼接
print( '<=>', torch.column_stack((tensor1,tensor2)) ) # 二位或多维张量沿着dim=1拼接，一维张量转置再沿着dim=1拼接

print( torch.cat((tensor1, tensor2), dim=0) )
print( '<=>', torch.vstack((tensor1,tensor2)) ) # 垂直拼接
print( '<=>', torch.row_stack((tensor1,tensor2)) ) # 同上

print( torch.stack((tensor1, tensor2), dim=0) ) #沿新维度拼接矩阵，新维度是dim=0
print( torch.stack((tensor1, tensor2), dim=1) ) #沿新维度拼接矩阵，新维度是dim=1

print( torch.stack((tensor1, tensor2), dim=2) ) #与下行命令效果一样
print( '<=>', torch.dstack((tensor1,tensor2)) ) #沿第三轴拼接
"""
                                -> ->
                              [[[1, 7],
                                [2, 8],
[[ ↓ 1,2,3],[4,5,6]]    --->    [3, 9]],
[[ ↓ 7,8,9],[10,11,12]         [[4, 10],
                                [5, 11],
                                [6, 12]]]
"""
tensor1 = torch.tensor([1,2,3])
tensor2 = torch.tensor([4,5,6])

print( torch.cat((tensor1, tensor2), dim=0) )
print( '<=>', torch.hstack((tensor1,tensor2)) ) # 水平拼接，一维张量沿着dim=0拼接，二位或多维张量沿着dim=1拼接
print( torch.column_stack((tensor1,tensor2)) ) # 一维张量转置再沿着dim=1拼接，二位或多维张量沿着dim=1拼接
print( '<=>', torch.cat((tensor1.view(3,1), tensor2.view(3,1)), dim=1) )

print( torch.vstack((tensor1,tensor2)) ) # 垂直拼接
print( '<=>', torch.row_stack((tensor1,tensor2)) ) # 同上
print( '<=>', torch.stack((tensor1, tensor2), dim=0) ) #一维tensor时，效果同上；二维或多维tensor，会增加一维度

print( torch.stack((tensor1, tensor2), dim=1) ) #比下行命令少一维

print( torch.dstack((tensor1,tensor2)) ) #沿第三轴拼接
"""
                    -> ->
[ ↓ 1,2,3]       [[[1, 4],
[ ↓ 4,5,6]  --->   [2, 5],
                   [3, 6]]]
"""
"""
6.2.
torch.tile(input, reps)
Constructs a tensor by repeating the elements of input
reps (tuple) – the number of repetitions per dimension
"""
tensor1 = torch.tensor([[1,2,3],[4,5,6]])
print( torch.tile(tensor1, (2, 1)) )
print( '<=>', torch.vstack((tensor1, tensor1)) )

print( torch.tile(tensor1, (2,)) )
print( '<=>', torch.tile(tensor1, (1, 2)) )
print( '<=>', torch.hstack((tensor1, tensor1)) )

print( torch.tile(tensor1, (2, 2)) )

# 7.切片
"""
7.1.
torch.index_select(input, dim, index, *, out=None)
dim维度的index全部
input (Tensor); dim (int); index (IntTensor or LongTensor)
"""
tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print( torch.index_select(tensor1, dim=0, index=torch.tensor([1])) )
print( torch.index_select(tensor1, dim=0, index=torch.tensor([0,1,1])) )
print( torch.index_select(tensor1, dim=1, index=torch.tensor([2])) )
print( torch.index_select(tensor1, dim=1, index=torch.tensor([1,2])) )
"""
7.2.
torch.narrow(input, dim, start, length)
dim维度的[start,start+length)
dim (int); start (int); length (int)
"""
print( torch.narrow(tensor1, dim=0, start=1, length=2) )
print( torch.narrow(tensor1, dim=1, start=0, length=2) )
"""
7.3.
按切片大小切
torch.split(tensor, split_size_or_sections, dim=0)
split_size_or_sections (int) or (list(int))
sum(split_size_or_sections) = 第dim维的长度
"""
tensor1 = torch.arange(1,23).reshape(11,2)
print( torch.split(tensor1, 2) )
tensor2 = torch.arange(1,23).view(2,11)
print( torch.split(tensor2, [1,2,3,5], dim=1) )
"""
7.4.
按数量或索引切
torch.tensor_split(input, indices_or_sections, dim=0)
indices_or_sections (Tensor, int or list or tuple of python:ints)
If indices_or_sections is an integer n or a zero dimensional long tensor with value n, input is split into n sections along dimension dim.
If indices_or_sections is a list or tuple of ints, or a one-dimensional long tensor, then input is split along dimension dim at each of the indices in the list, tuple or tensor. 
"""
print( torch.tensor_split(tensor1, 2) )                 #indices_or_sections是整数，切成两个
print( '<=>', torch.tensor_split(tensor1, torch.tensor(2)) )   #indices_or_sections是0维tensor，切成两个
print( torch.tensor_split(tensor1, torch.tensor([2])) ) #indices_or_sections是1维tensor，按索引切
print( torch.tensor_split(tensor2, (0,3,5,7), dim=1) )  #indices_or_sections是tuple，按索引切
print( '<=>', torch.tensor_split(tensor2, [0,3,5,7], dim=1) )  #indices_or_sections是list，按索引切
print( '<=>', torch.tensor_split(tensor2, torch.tensor([0,3,5,7]), dim=1) ) #indices_or_sections是1维tensor，按索引切
"""
7.5.
torch.chunk(input, chunks, dim=0)
chunks (int); dim (int)
"""
# tensor1 = torch.tensor([[1,2,3],[4,5,6]])
print('====沿着dim=0切片====')
for chunks in range(1,tensor1.size(0)+1):
    print('chunks=',chunks,':', torch.chunk(tensor1,chunks,dim=0) )
print('====沿着dim=1切片====')
for chunks in range(1,tensor1.size(1)+1):
    print('chunks=',chunks,':', torch.chunk(tensor1,chunks,dim=1) )
#-----------------------------------------------------------------------------#
# 数学运算
data_tensor = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float)

# 加法
tensorX = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float)
tensorY = torch.ones_like(tensorX)

# X数值不变，内存也不便，开辟新的内存地址存储结果Z  (1)
print('Before: X=', tensorX, 'id=', id(tensorX))
tensorZ = tensorX + tensorY
print('After: X=', tensorX, 'id=', id(tensorX))
print('After: Z=', tensorZ, 'id=', id(tensorZ))
# X数值不变，内存也不便，开辟新的内存地址存储结果Z  (2)
print('Before: X=', tensorX, 'id=', id(tensorX))
tensorZ = torch.add(tensorX, tensorY)
print('After: X=', tensorX, 'id=', id(tensorX))
print('After: Z=', tensorZ, 'id=', id(tensorZ))
# X数值不变，内存也不便，开辟新的内存地址存储结果Z  (3)
print('Before: X=', tensorX, 'id=', id(tensorX))
tensorZ = tensorX.add(tensorY)
print('After: X=', tensorX, 'id=', id(tensorX))
print('After: Z=', tensorZ, 'id=', id(tensorZ))

# X数值改变，内存也改变 (1)
print('Before: X=', tensorX, 'id=', id(tensorX))
tensorX = tensorX + tensorY
print('After: X=', tensorX, 'id=', id(tensorX))

# !! X数值改变，内存不变 (1) - 节约内存开销 !!
print('Before: X=', tensorX, 'id=', id(tensorX))
tensorX[:] = tensorX + tensorY
print('After: X=', tensorX, 'id=', id(tensorX))
# X数值改变，内存不变 (2)
print('Before: X=', tensorX, 'id=', id(tensorX))
torch.add(tensorX, tensorY, out=tensorX)
print('After: X=', tensorX, 'id=', id(tensorX))
# X数值改变，内存不变 (3)
print('Before: X=', tensorX, 'id=', id(tensorX))
tensorX.add_(tensorY)
print('After: X=', tensorX, 'id=', id(tensorX))
# X数值改变，内存不变 (4)
print('Before: X=', tensorX, 'id=', id(tensorX))
tensorX += tensorY
print('After: X=', tensorX, 'id=', id(tensorX))

# !! 广播机制:先适当复制元素使这两个Tensor形状相同后再按元素运算 !!
print('X:', torch.tensor([1,2,3]))
print('Y:', torch.tensor([[1],[2],[3]]))
print('X + Y =', torch.tensor([1,2,3]) + torch.tensor([[1],[2],[3]]))






# 矩阵乘法
data_tensor = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float)
print(data_tensor @ data_tensor.T)
print(data_tensor.matmul(data_tensor.T))
print(torch.matmul(data_tensor, data_tensor.T, out=torch.zeros((data_tensor.shape[0], data_tensor.T.shape[1]), dtype=data_tensor.dtype)))
print(data_tensor)


# 矩阵点乘
print(data_tensor * data_tensor)
print(data_tensor.mul(data_tensor))
print(torch.mul(data_tensor, data_tensor, out=torch.zeros((data_tensor.shape[0], data_tensor.shape[1]), dtype=data_tensor.dtype)))

# 点除
"""
torch.div(input, other, *, rounding_mode=None, out=None)
torch.divide(input, other, *, rounding_mode=None, out=None) 同上
input (Tensor) – the dividend; other (Tensor or Number) – the divisor
rounding_mode=None 表示四舍五入
rounding_mode="trunc" 表示向0取整
rounding_mode="floor" 表示向下取整
"""
print( torch.div(torch.tensor([-4., 1., 4.]), 3) )
print( torch.div(torch.tensor([-4., 1., 4.]), 3, rounding_mode="trunc") )
print( torch.divide(torch.tensor([-4., 1., 4.]), 3, rounding_mode="floor") )


# 取值


print( torch.abs(torch.tensor([-4.2, -0.1, 0.2, 4.6])) )# 绝对值
print( torch.absolute(torch.tensor([-4.2, -0.1, 0.2, 4.6])) )# 绝对值
print( torch.ceil(torch.tensor([-4.2, -0.1, 0.2, 4.6])) )#向上取整
print( torch.round(torch.tensor([-4.2, -0.1, 0.2, 4.6])) )#四舍五入
print( torch.floor(torch.tensor([-4.2, -0.1, 0.2, 4.6])) )#向下取整
print( torch.trunc(torch.tensor([-4.2, -0.1, 0.2, 4.6])) )#向0取整
print( torch.fix(torch.tensor([-4.2, -0.1, 0.2, 4.6])) )#向0取整


"""
torch.heaviside(input, values, *, out=None) #input 和 values 都必须是tensor
阶跃函数                   
                          | 0,      if input < 0
heaviside(input,values) = { values, if input = 0
                          | 1,      if input > 0
"""
print( torch.heaviside(input=torch.tensor([-1.5, 0, 2.0]), values=torch.tensor([3.])) )
print( torch.heaviside(input=torch.tensor([-1.5, 0, 2.0]), values=torch.tensor([1.2, -2.0, 3.5])) )


"""
torch.clamp(input, min, max, *, out=None)
torch.clip(input, min, max, *, out=None) 同上
out[i] = min(max(input[i],min_value),max_value)
               | min,      if input[i] < min
i.e., out[i] = { input[i], if min <= input[i] <= max
               | max,      if input[i] > max
"""
print( torch.clamp(input=torch.tensor([-1.5, 0, 2.0]), min=-0.5, max=0.5) )


"""
torch.copysign(input, other, *, out=None)
out[i] = { -|input[i]|, if other[i] < -0.0
         |  |input[i]|, if other[i] >= 0.0
"""
print( torch.copysign(input=torch.tensor([-1.5,-1.5,-1.5,2.0,2.0,2.0]), other=torch.tensor([-1.5,0,2.0,-1.5,0,2.0]) ) )


"""
torch.addcdiv(input, tensor1, tensor2, *, value=1, out=None)
out[i] = input[i] + value*tensor1[i]/tensor2[i]
"""
print( torch.addcdiv(input=torch.tensor([4.,5.]),
                     tensor1=torch.tensor([1.,1.]),
                     tensor2=torch.tensor([2.,2.]),
                     value=2) )


"""
# torch.addcmul(input, tensor1, tensor2, *, value=1, out=None)
# out[i] = input[i] + value*tensor1[i]*tensor2[i]
"""
print( torch.addcmul(input=torch.tensor([4.,5.]),
                     tensor1=torch.tensor([1.,1.]),
                     tensor2=torch.tensor([2.,2.]),
                     value=2) )




"""
torch.bitwise_and(input, other, *, out=None)
按位与
"""
print( torch.bitwise_and(torch.tensor([-1, -2, 3]), torch.tensor([1, 0, 3])) )
print( torch.bitwise_and(torch.tensor([True, True, False]), torch.tensor([False, True, False])) )
"""
torch.bitwise_or(input, other, *, out=None)
按位或
"""
print( torch.bitwise_or(torch.tensor([-1, -2, 3]), torch.tensor([1, 0, 3])) )
print( torch.bitwise_or(torch.tensor([True, True, False]), torch.tensor([False, True, False])) )
"""
torch.bitwise_xor(input, other, *, out=None)
按位异或
"""
print( torch.bitwise_xor(torch.tensor([-1, -2, 3]), torch.tensor([1, 0, 3])) )
print( torch.bitwise_xor(torch.tensor([True, True, False]), torch.tensor([False, True, False])) )
"""
torch.bitwise_not(input, *, out=None)
按位取反，即 ~x = -（x+1）
"""
print( torch.bitwise_not(torch.tensor([-1, -2, 3])) )


"""
三角函数
torch.sin() # sin()
torch.asin() # sin()逆运算
torch.arcsin() # sin()逆运算
torch.cos() # cos()
torch.acos() # cos()逆运算
torch.arccos() # cos()逆运算
torch.tan() # tan()
torch.atan() # tan()逆运算
torch.arctan() # tan()逆运算
torch.sinh() # sinh()
torch.asinh() # sinh()逆运算
torch.arcsinh() # sinh()逆运算
torch.cosh() # cosh()
torch.acosh() # cosh()逆运算
torch.arccosh() # cosh()逆运算
torch.tanh() # tanh()
torch.atanh() # tanh()逆运算
torch.arctanh() # tanh()逆运算
"""
# 角度（angles in degrees）-> 弧度（radians）
print( torch.deg2rad(torch.tensor([180., 90.])) )
# 弧度（radians）-> 角度（angles in degrees）
print( torch.rad2deg(torch.tensor([3.1416, 1.5708])) )
# if x < 0: return pi; if x >= 0: return 0
print( torch.angle(torch.tensor(-1)) )
print( torch.angle(torch.tensor(0)) )
print( torch.angle(torch.tensor(1)) )



#-----------------------------------------------------------------------------#

# Reduction Ops

tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]],dtype = torch.float)
"""
torch.sum(input, dim, keepdim=False, *, dtype=None)
torch.mean(input, dim, keepdim=False, *, out=None)
torch.max(input, dim, keepdim=False, *, out=None) -> max + max_indices
torch.min(input, dim, keepdim=False, *, out=None) -> min + min_indices
"""
print(torch.sum(tensor1))
print(tensor1.mean())
print(torch.max(tensor1))
print(tensor1.min())


print(torch.sum(tensor1, dim=0, keepdim=False)) #对列求和
print(tensor1.mean(dim=1, keepdim=False))       #对行求平均值
print(torch.max(tensor1, dim=0, keepdim=True))  #对列求最大值
print(tensor1.min(dim=1, keepdim=True))         #对行求最小值

# torch.maximum() <=> torch.max(input, other, *, out=None)
# torch.minimum() <=> torch.min(input, other, *, out=None) 

#-----------------------------------------------------------------------------#
#复数
"""
torch.complex(real, imag, *, out=None)  'out = real + imag*j'
"""
real = torch.tensor([1, 2], dtype=torch.float32)
imag = torch.tensor([3, 4], dtype=torch.float32)
print( torch.complex(real, imag) )
"""
torch.polar(abs, angle, *, out=None)  'out = abs*cos(angle)+abs*sin(angle)*j
"""
abs = torch.tensor([1, 2], dtype=torch.float64)
angle = torch.tensor([3.1415926/2, 5*3.1415926/4], dtype=torch.float64)
print( torch.polar(abs, angle) ) 
"""
torch.conj(input, *, out=None)
共轭复数，两个实部相等，虚部互为相反数
"""
torch.conj(torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j]))


