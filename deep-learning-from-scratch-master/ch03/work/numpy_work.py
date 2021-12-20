from typing import BinaryIO
import numpy as np


A = np.array([[1,2],[3,4]])
print(A)

B = np.array([[5,6,7],[7,8,9]])
print(B)

C= np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print(C)


#==============================
# numpy.ndim() : 配列の次元を出力
#==============================

def print_ndim(A):
    print(f"{A} の次元：{np.ndim(A)}")

print_ndim(A)
print_ndim(B)
print_ndim(C)


#====================================
# numpy.array.shape : 配列の形状を出力
#====================================

def print_shape(A):
    print(f"{A} の形状：{A.shape}")

print_shape(A)
print_shape(B)
print_shape(C)

#==========================================
# numpy.dot() : 配列の内積（いわゆる行列の積）
#==========================================

def print_dot(A,B):
    print(f"{A} * {B} = {np.dot(A,B)}")

print_dot(A,B)