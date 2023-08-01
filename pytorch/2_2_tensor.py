# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 11:57:31 2021

@author: CC-i7-1065G7
"""
# 《动手学深度学习》 http://tangshusen.me/Dive-into-DL-PyTorch/#/
# 数学运算整理到 digamma 
import torch
import numpy


#-----------------------------------------------------------------------------#
# Creation Ops
tensor1 = torch.tensor([[1,2,3],[4,5,6],[7,8,9]], dtype=torch.float)

#-----------------------------------------------------------------------------#
# Indexing, Slicing, Joining, Mutating Ops
# 1.形状

# 2.查找/索引(不会开辟新内存)

# 3.变形

# 4.降维

# 5.转置

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


