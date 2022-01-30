# coding: utf-8
import sys, os
sys.path.append(os.pardir) # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from collections import OrderedDict
from common.presentation.layers import *
from common.presentation.gradient import numerical_gradient

class MultiLayerNetExtend:
    """拡張版の全結合による多層ニューラルネットワーク
    
    Weiht Decay、Dropout、Batch Normalizationの機能を持つ

    Parameters
    ----------
    input_size : 入力サイズ（MNISTの場合は784）
    hidden_size_list : 隠れ層のニューロンの数のリスト（e.g. [100, 100, 100]）
    output_size : 出力サイズ（MNISTの場合は10）
    activation : 'relu' or 'sigmoid'
    weight_init_std : 重みの標準偏差を指定（e.g. 0.01）
        'relu'または'he'を指定した場合は「Heの初期値」を設定
        'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
    weight_decay_lambda : Weight Decay（L2ノルム）の強さ
    use_dropout: Dropoutを使用するかどうか
    dropout_ration : Dropoutの割り合い
    use_batchNorm: Batch Normalizationを使用するかどうか
    """
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0, 
                 use_dropout = False, dropout_ration = 0.5, use_batchnorm=False):
        self.input_size = input_size                    # 入力データサイズ（入力層ノード数）
        self.output_size = output_size                  # 出力データサイズ（出力層ノード数）
        self.hidden_size_list = hidden_size_list        # 隠れ層のニューロンの数のリスト
        self.hidden_layer_num = len(hidden_size_list)   # 隠れ層の数
        self.use_dropout = use_dropout                  # Dropoutを使用するかどうか
        self.weight_decay_lambda = weight_decay_lambda  # Weight Decay（L2ノルム）の強さ
        self.use_batchnorm = use_batchnorm              # Batch Normalizationを使用するかどうか
        self.params = {}                                # parameterを格納するディクショナリ

        # 重みの初期化
        self.__init_weight(weight_init_std)

        # レイヤの生成
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu} # インスタンス生成時の引数でどちらかを指定（初期値はReLU）
        self.layers = OrderedDict()                           # レイヤを格納する順序付きディクショナリ
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)]) # アフィンレイヤ生成

            # Batch Normalizationの場合
            if self.use_batchnorm:
                self.params['gamma' + str(idx)] = np.ones(hidden_size_list[idx-1])                              # ガンマ(傾き・1)
                self.params['beta' + str(idx)] = np.zeros(hidden_size_list[idx-1])                              # ベータ(切片・0)
                self.layers['BatchNorm' + str(idx)] = BatchNormalization(self.params['gamma' + str(idx)]
                    , self.params['beta' + str(idx)])                                                           # Batch Normレイヤ生成
                
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()                      # 活性化レイヤ生成
            
            # dropoutの場合
            if self.use_dropout:
                self.layers['Dropout' + str(idx)] = Dropout(dropout_ration)                                     # dropoutレイヤ生成

        # 出力レイヤの生成
        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])

        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """重みの初期値設定

        Parameters
        ----------
        weight_init_std : 重みの標準偏差を指定（e.g. 0.01）
            'relu'または'he'を指定した場合は「Heの初期値」を設定
            'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
        """
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        # 各層の重みとバイアスの初期値を設定
        for idx in range(1, len(all_size_list)):
            #############################
            # スケール（標準偏差）の設定　#
            #############################
            # 直接指定する場合
            scale = weight_init_std
            # Heの初期値の場合
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])  # ReLUを使う場合に推奨される初期値のスケール（標準偏差）
            # Xavierの初期値の場合            
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])  # sigmoidを使う場合に推奨される初期値のスケール（標準偏差）

            ########################
            # 初期パラメータの設定　#
            ########################
            # 重み
            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            # バイアス
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            # DropoutレイヤまたはBatchNormレイヤの場合
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flg) 
            # それ以外のレイヤの場合
            else:
                x = layer.forward(x)

        return x

    def loss(self, x, t, train_flg=False):
        """損失関数を求める
        引数のxは入力データ、tは教師ラベル
        """
        y = self.predict(x, train_flg)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x, train_flg=False)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """勾配を求める（数値微分）

        Parameters
        ----------
        x : 入力データ
        t : 教師ラベル

        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        """
        loss_W = lambda W: self.loss(x, t, train_flg=True)

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])
            
            if self.use_batchnorm and idx != self.hidden_layer_num+1:
                grads['gamma' + str(idx)] = numerical_gradient(loss_W, self.params['gamma' + str(idx)])
                grads['beta' + str(idx)] = numerical_gradient(loss_W, self.params['beta' + str(idx)])

        return grads
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t, train_flg=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            # 重みパラメータを更新
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.params['W' + str(idx)]
            # バイアスパラメータを更新
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

            # BatchNormalization使用時かつ出力層以外の場合
            if self.use_batchnorm and idx != self.hidden_layer_num+1:
                # ガンマ（傾き）を更新
                grads['gamma' + str(idx)] = self.layers['BatchNorm' + str(idx)].dgamma
                # ベータ（切片）を更新
                grads['beta' + str(idx)] = self.layers['BatchNorm' + str(idx)].dbeta

        return grads