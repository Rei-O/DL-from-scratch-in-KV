# coding: utf-8
import sys, os
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'))  # 親ディレクトリの親ディレクトリのファイルをインポートするための設定
import numpy as np

from common.presentation.functions import *
from common.presentation.gradient import numerical_gradient
from common.presentation.layers import *
import common.presentation.debug as debug
from collections import OrderedDict
from dataset.presentation.mnist import load_mnist

####################################################################
# 5章の総まとめとして誤差逆伝播法による学習を実装する
# 
# <<方針>>
# 1. ミニバッチ
# 2. 勾配の算出
# 3. パラメータの更新
# 4. 1~3の繰り返し
####################################################################


# 2層ニューラルネットワーク
class TwoLayerNet:
    """
    ### TwoLayerNet class
    #### ■概要
    誤差逆伝播法による「学習」を実装するための2層ニューラルネットワーク
    #### ■メソッド
    predict : (self, x) -> Array
    loss : (self, x, t) -> float
    """
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):

        self.params = {}

        # 1層のパラメータを初期化
        # 重みの初期化
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) # 各要素がN(0, 0.01)に従う(input_size,hidden_size)行列を生成
        # バイアスの初期化
        self.params['b1'] = np.zeros(hidden_size)

        # 2層のパラメータを初期化
        # 重みの初期化
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) # 各要素がN(0, 0.01)に従う(hidden_size, output_size)行列を生成
        # バイアスの初期化
        self.params['b2'] = np.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict() # 格納順を記憶するディクショナリ
        # レイヤ1
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu'] = Relu()
        # レイヤ2
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        # 出力層
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        """
        入力データに対してニューラルネットワークの予測結果を返却する
        param : x=Array
        return : Array
        """
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    # x:入力データ, t:教師データ
    def loss(self, x, t):
        """
        入力データ、正解ラベルに対してモデルによる予測を行い、クロスエントロピーによる損失結果を返却する
        param : x=Array(入力データ), t=Array(教師データ)
        return : Array
        """
        return self.lastLayer.forward(self.predict(x), t)
    
    def accuracy(self, x, t):
        """
        入力データ、正解ラベルに対してモデルによる予測を行い、正解率(Accuracy)を返却する
        param : x=Array(入力データ), t=Array(教師データ)
        return : Array
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1) # axis=1は縦方向で最大値のインデックスを返却
        if t.ndim != 1 : t = np.argmax(t, axis=1) # one-hot, ラベル表現両方対応

        # Accuracyの計算
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x, t):
        """
        入力データ、正解ラベルに対してモデルによる予測を行い、数値微分によって重みとバイアスの勾配を計算する
        param : x=Array(入力データ), t=Array(教師データ)
        return : Array
        """        
        loss_W = lambda W : self.loss(x,t)

        # 重み、バイアスの勾配を格納するディクショナリ
        grads = {}

        grads['W1'] = numerical_gradient(loss_W, self.params['W1']) # 各要素がN(0, 0.01)に従う(input_size,hidden_size)行列を生成
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2']) # 各要素がN(0, 0.01)に従う(hidden_size, output_size)行列を生成
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
    
    def gradient(self, x, t):
        """
        入力データ、正解ラベルに対してモデルによる予測を行い、誤差逆伝播法によって重みとバイアスの勾配を計算する
        param : x=Array(入力データ), t=Array(教師データ)
        return : Array
        """   
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        # 設定
        grads = {}
        for i in (1, 2):
            grads[f'W{i}'] = self.layers[f'Affine{i}'].dW
            grads[f'b{i}'] = self.layers[f'Affine{i}'].db
        
        return grads

if __name__ == '__main__':

    ########################################################################
    # 勾配確認（gradient check）
    # 誤差逆伝播法の実装が正しいことを数値微分で勾配を求めた結果との差分で確認する
    ########################################################################

    # 画像データ読込み
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10) # 入力は28*28ピクセルの画像データ、手書き数字0~9の識別を想定

    x_batch = x_train[:3]
    t_batch = t_train[:3]
    grad_numerical = net.numerical_gradient(x_batch, t_batch)
    grad_backprop = net.gradient(x_batch, t_batch)

    for key in grad_numerical.keys():
        diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
        debug.debugprt(diff, key)
