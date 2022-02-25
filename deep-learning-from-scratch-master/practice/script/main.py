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
    def __init__(self, input_size, hiddenLayerSizeList, output_size, weight_decay_lambda=0, dropout_ratio=0.5):
        ########################
        # レイヤサイズを設定 #
        ########################
        self.input_size = input_size
        self.hiddenLayerSizeList = hiddenLayerSizeList
        self.output_size = output_size
        self.allLayerSizeList = np.concatenate([np.array([self.input_size]), self.hiddenLayerSizeList, np.array([self.output_size])]) 
    
        de.debugprt(self.allLayerSizeList, 'allLayerSizeList')

        ###################################
        # パラメータの初期設定 #
        ###################################
        self.paramDict = {}
        self.weight_decay_lambda = weight_decay_lambda

        #############
        # レイヤ生成 #
        #############
        self.layerDict = OrderedDict()  # 順序付きディクショナリ
        for idx in range(1, len(self.hiddenLayerSizeList)+1):
            #######################
            # Affine : Heの初期値 #
            #######################
            self.paramDict['W' + str(idx)] = np.random.normal(loc=0, scale=2 / np.sqrt(self.allLayerSizeList[idx-1]), size=[self.allLayerSizeList[idx-1], self.allLayerSizeList[idx]])  # 標準偏差を2/(前層のニューロン数の平方根)で設定する
            de.debugprt(self.paramDict['W' + str(idx)].shape , f"self.paramDict[{'W' + str(idx)}].shape")
            self.paramDict['b' + str(idx)] = np.zeros(self.allLayerSizeList[idx])
            de.debugprt(self.paramDict['b' + str(idx)].shape , f"self.paramDict[{'b' + str(idx)}].shape")
            self.layerDict['Affine'+str(idx)] = Affine(self.paramDict['W' + str(idx)], self.paramDict['b' + str(idx)])

            #############
            # BatchNorm #
            #############
            self.paramDict['gamma'+str(idx)] = np.ones(hiddenLayerSizeList[idx-1])
            self.paramDict['beta'+str(idx)] = np.zeros(hiddenLayerSizeList[idx-1])
            self.layerDict['BatchNorm'+str(idx)] = BatchNorm(self.paramDict['gamma'+str(idx)], self.paramDict['beta'+str(idx)])

            ########
            # ReLU #
            ########
            self.layerDict['ReLU'+str(idx)] = ReLU()

            ###########
            # Dropout #            
            ###########
            self.layerDict['Dropout'+str(idx)] = Dropout(dropout_ratio)

        ##############
        # Last layer #
        ##############
        idx = len(self.hiddenLayerSizeList) + 1

        # Affine : Heの初期値
        self.paramDict['W' + str(idx)] = np.random.normal(loc=0, scale=2 / np.sqrt(self.allLayerSizeList[idx-1]), size=[self.allLayerSizeList[idx-1], self.allLayerSizeList[idx]])  # 標準偏差を2/(前層のニューロン数の平方根)で設定する
        de.debugprt(self.paramDict['W' + str(idx)].shape , f"self.paramDict[{'W' + str(idx)}].shape")
        self.paramDict['b' + str(idx)] = np.zeros(self.allLayerSizeList[idx])
        de.debugprt(self.paramDict['b' + str(idx)].shape , f"self.paramDict[{'b' + str(idx)}].shape")
        self.layerDict['Affine'+str(idx)] = Affine(self.paramDict['W' + str(idx)], self.paramDict['b' + str(idx)])

        # SoftmaxWithLoss
        self.lastLayerDict = {"train" : SoftmaxWithLoss(), "predict" : Softmax()}
        # self.lastLayer = SoftmaxWithLoss()

    def __forward(self, X, train_flg=True):
        """
        学習・予測時におけるforwardを行う. 
        """
        for key, layer in self.layerDict.items():
            de.debugprt(f"forward : {key}")
            X = layer.forward(X, train_flg)
        return X
    
    def predict(self, X, train_flg=False):
        """
        予測を行う. 
        """
        self.__forward(X, train_flg)
        Y = self.lastLayerDict['predict'].forward(X)
        return Y

    def __loss(self, X, t, train_flg=False):
        # 荷重減衰の初期化
        weight_decay = 0
        # 各層の荷重減衰を積上げ
        for idx in range(1, len(self.allLayerSizeList)):
            W = self.paramDict['W'+str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)
        
        return self.lastLayerDict['train'].forward(X, t) + weight_decay
    
    def gradient(self, X, t):
        """
        学習を行う. 
        """
        # forward
        self.__forward(X, train_flg=True)
        self.__loss(X, t)

        # backward
        dout = 1
        dout = self.lastLayerDict["train"].bachward(dout)

        # レイヤを取得
        layerList = list(self.layerDict.values())
        # 逆順に並べ替え
        layerList.reverse()

        # 逆伝播
        for layer in layerList:
            dout = layer.backward(dout)  # ここで各レイヤのインスタンスに微分値を格納する
        
        # 微分値を格納（backwardで格納した微分値を取得していく）
        grads = {}
        for idx in range(1, len(self.allLayerSizeList)):
            # Affine layer
            grads['W'+str(idx)] = self.layerDict['Affine'+str(idx)].dW + self.weight_decay_lambda * self.paramDict['W' + str(idx)]
            grads['b'+str(idx)] = self.layerDict['Affine'+str(idx)].db

            # BatchNorm layer
            grads['gamma'+str(idx)] = self.layerDict['BatchNorm'+str(idx)].dgamma
            grads['beta'+str(idx)] = self.layerDict['BatchNorm'+str(idx)].dbeta

        return grads
    #################### 以下廃止予定 ####################

    # def __forward(self, i):
    #     de.debugprt(f'====={i}層目=====')

    #     # Affineレイヤ
    #     affine = Affine(self.paramDict['W'+str(i)],  self.paramDict['b'+str(i)])
    #     de.debugprt(self.Y.shape, "self.Y.shape")
    #     de.debugprt(self.paramDict['W'+str(i)].shape, f"self.paramDict[{'W'+str(i)}].shape")
    #     A = affine.forward(self.Y)
    #     self.layerDict['Affine' + str(i)] = affine

    #     # ReLUレイヤ (活性化レイヤ)
    #     relu = ReLU()
    #     self.Y = relu.forward(A)
    #     self.layerDict['ReLU' + str(i)] = relu
        

    # def __forwardLastLayer(self, i):
    #     de.debugprt(f'====={i}層目（出力層）=====')

    #     # Affineレイヤ
    #     affine = Affine(self.paramDict['W'+str(i)],  self.paramDict['b'+str(i)])
    #     de.debugprt(self.Y.shape, "self.Y.shape")
    #     de.debugprt(self.paramDict['W'+str(i)].shape, f"self.paramDict[{'W'+str(i)}].shape")
    #     self.Y = affine.forward(self.Y)
    #     self.layerDict['Affine' + str(i)] = affine
 


network = NuralNetwork(input_size, hiddenLayerSizeList, output_size)

grads = network.gradient(x_train, x_test)
print(grads)

# y = network.forward(x_train)
y = network.predict(x_train)
print(y)

