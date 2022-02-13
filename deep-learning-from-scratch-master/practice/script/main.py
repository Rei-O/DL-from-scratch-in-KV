# coding: utf-8
import os
import sys
from typing import OrderedDict
from xmlrpc.server import DocXMLRPCRequestHandler
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'))  # 親ディレクトリの親ディレクトリのファイルをインポートするための設定
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from dataset.presentation.mnist import load_mnist
from layers import *
import common.presentation.debug as de

# デバッグモード
de.isDebugMode = True

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 動確用にデータ数を限定
x_train = x_train[:10, :]
t_train = t_train[:10]

de.debugprt(x_train.shape, 'x_train_size')
de.debugprt(t_train.shape, 't_train_size')

batch_size = 64

input_size = x_train.shape[1]
hiddenLayerSizeList = np.array([200, 100])
output_size = len(t_train)

class NuralNetwork:
    def __init__(self, input_size, hiddenLayerSizeList, output_size):
        ########################
        # レイヤサイズを設定 #
        ########################
        self.input_size = input_size
        self.hiddenLayerSizeList = hiddenLayerSizeList
        self.output_size = output_size
        self.layerSizeList = np.concatenate([np.array([self.input_size]), self.hiddenLayerSizeList, np.array([self.output_size])]) 
    
        de.debugprt(self.layerSizeList, 'layerSizeList')

        ###################################
        # パラメータの初期値設定：Heの初期値 #
        ###################################
        self.paramDict = {}
        for i in range(1, len(self.layerSizeList)):
            self.paramDict['W' + str(i)] = np.random.normal(loc=0, scale=2 / np.sqrt(self.layerSizeList[i-1]), size=[self.layerSizeList[i-1], self.layerSizeList[i]])  # 標準偏差を2/(前層のニューロン数の平方根)で設定する
            de.debugprt(self.paramDict['W' + str(i)].shape , f"self.paramDict[{'W' + str(i)}].shape")
            self.paramDict['b' + str(i)] = np.zeros(self.layerSizeList[i])
            de.debugprt(self.paramDict['b' + str(i)].shape , f"self.paramDict[{'b' + str(i)}].shape")

        ###############
        # レイヤ定義 #
        ###############
        self.layerDict = OrderedDict()  # 順序付きディクショナリ

    def forward(self, X):
        self.Y = X

        for i in range(1, len(self.layerSizeList)):
            if i < len(self.layerSizeList)-1:
                self.__forward(i)
            else:
                self.__forwardLastLayer(i)
        
        return self.Y
    

    def __forward(self, i):
        de.debugprt(f'====={i}層目=====')

        # Affineレイヤ
        affine = Affine(self.paramDict['W'+str(i)],  self.paramDict['b'+str(i)])
        de.debugprt(self.Y.shape, "self.Y.shape")
        de.debugprt(self.paramDict['W'+str(i)].shape, f"self.paramDict[{'W'+str(i)}].shape")
        A = affine.forward(self.Y)
        self.layerDict['Affine' + str(i)] = affine

        # ReLUレイヤ (活性化レイヤ)
        relu = ReLU()
        self.Y = relu.forward(A)
        self.layerDict['ReLU' + str(i)] = relu
        

    def __forwardLastLayer(self, i):
        de.debugprt(f'====={i}層目（出力層）=====')

        # Affineレイヤ
        affine = Affine(self.paramDict['W'+str(i)],  self.paramDict['b'+str(i)])
        de.debugprt(self.Y.shape, "self.Y.shape")
        de.debugprt(self.paramDict['W'+str(i)].shape, f"self.paramDict[{'W'+str(i)}].shape")
        self.Y = affine.forward(self.Y)
        self.layerDict['Affine' + str(i)] = affine
 


network = NuralNetwork(input_size, hiddenLayerSizeList, output_size)

y = network.forward(x_train)
print(y)