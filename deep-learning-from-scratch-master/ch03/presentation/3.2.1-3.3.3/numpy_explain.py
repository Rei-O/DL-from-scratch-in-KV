import sys, os
from matplotlib.pyplot import xlabel
sys.path.append(os.path.join(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'), '..'))  # 親の親の親ディレクトリのファイルをインポートするための設定
import numpy as np

import common.presentation.debug as debug

debug.isDebugMode = True

A = np.array([[1,2],[3,4]])
# debug.debugprt(A)

B = np.array([[5,6,7],[7,8,9]])

# debug.debugprt(B)

C= np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
# debug.debugprt(C)


#==============================
# numpy.ndim() : 配列の次元を出力
#==============================

def print_ndim(A):
    debug.debugprt(f"{A} の次元：{np.ndim(A)}")

# debug.debugprt("=====numpy.ndim() : 配列の次元を出力)=====")
# print_ndim(A)
# print_ndim(B)
# print_ndim(C)



#====================================
# numpy.array.shape : 配列の形状を出力
#====================================

def print_shape(A):
    debug.debugprt(f"{A} の形状：{A.shape}")

# debug.debugprt("=====numpy.array.shape : 配列の形状を出力=====")
# print_shape(A)
# print_shape(B)
# print_shape(C)

#=======================================
# numpy.array.reshape : 配列の形状を変更する
#=======================================
reA = A.reshape(1,4)
reB = B.reshape(6,1)
reC = C.reshape(1,2,-1)
def print_reshape(A, reA):
    debug.debugprt(f"{A} の変更後：\n{reA}")

# debug.debugprt("=====numpy.array.reshape : 配列の形状を変更=====")
# print_reshape(A, reA)
# print_reshape(B, reB)
# print_reshape(C, reC)

#==========================================
# numpy.dot() : 配列の内積（いわゆる行列の積）
#==========================================
def print_dot(A,B):
    debug.debugprt(f"{A} * {B} = {np.dot(A,B)}")

# debug.debugprt("=====numpy.dot() : 配列の内積（いわゆる行列の積）=====")
# print_dot(A,B)

#==========================================
# numpy.sum() : 配列の成分の和
#==========================================

def print_sum(A):
    debug.debugprt(f"{A} の成分の和：{np.sum(A)}")

# debug.debugprt("=====numpy.sum() : 配列の成分の和=====")
# print_sum(A)
# print_sum(B)
# print_sum(C)


#==========================================
# 配列の要素を複数指定する
#==========================================
# debug.debugprt("=====配列の要素を複数指定する=====")

# debug.debugprt(f"B : {B}")
# debug.debugprt(f"Bの0~1行目×1列目 : {B[[0,1],[1]]}")


#================================================================================================
# 演習問題
# D+EとD+Fは実行結果が異なります。
# これはEが横ベクトル、Fが縦ベクトルという違いから起こります。
# 入力が横ベクトル、縦ベクトルに関わらずベクトル同士の足し算を行うためのaddVector関数を作成してください。
#================================================================================================
D = np.array([1,2,3,4])
# print(D)

E = np.array([5,6,7,8])
# print(E)

F= np.array([[5],[6],[7],[8]])
# print(F)

# 模範解答①
def addVector(A, B):

    if len(B.shape) != 1:    
        B = B.reshape(-1)

    val = A + B

    return val

# 模範解答②
# def addVector(A, B):

#     if np.ndim(B) != 1:    
#         B = B.reshape(-1)

#     val = A + B

#     return val

# 模範解答③
# def addVector(A, B):
#     val = A.reshape(-1) + B.reshape(-1)

#     return val

# 皆さんからの解答も一部記載

# チャネットさん
# def addVector(x, y):
#     val = x.flatten() + y.flatten()
#     return val

# N.M さん（一部修正）
# import itertools
# def addVector(A, B):
#     _B = np.array([])
#     for x in itertools.chain(B):
#         _B = np.append(_B, x) 
#     val = A+_B

#     return val


print(addVector(D, E))
# 期待値 > [ 6  8 10 12]

print(addVector(D, F))
# 期待値 > [ 6  8 10 12]
