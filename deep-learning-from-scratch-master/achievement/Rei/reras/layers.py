from nntplib import NNTPPermanentError
import os, sys
from turtle import forward
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from abc import ABC, abstractmethod
from reras.functions import *
from reras.utils import *

########################### Abstract Class ###########################

class AbstractLayer(ABC):
    """
    通常レイヤの抽象クラス
    """
    @abstractmethod
    def __init__(self, input_units, output_units, activation_key):
        """
        共通項目の初期化を行う。

        Params
        ---------------
        input_units(int,ndarray) : 入力データの1ユニットサイズ(個別指定する場合に使用、前層のoutput_unitsの場合はNone)
        output_units(int,ndarray) : 出力データの1ユニットサイズ
        activation_key(str) : 活性化関数名

        Return
        ---------------
        None
        """
        self.input_units = input_units if type(input_units) is tuple else (input_units,)
        self.output_units = output_units if type(output_units) is tuple else (output_units,)
        self.activation_key = activation_key
        self.activation = Activation(self.activation_key)

    @abstractmethod
    def forward(self, x, forward, activation, train_flg):
        y = forward(x, train_flg)
        out = activation.forward(y)

        return out

    @abstractmethod
    def backward(self, dout, backward, activation):
        dy = activation.backward(dout)

        return backward(dy)

    @abstractmethod
    def compile(self):
        pass


class AbstractWithLossLayer(ABC):
    """
    損失の計算を行うレイヤの抽象クラス
    """
    @abstractmethod
    def forward(self, x, t, forward, loss, train_flg):
        self.t = t
        self.y = forward(x)

        if train_flg:
            return loss(self.y, t)
        else:
            return self.y

    @abstractmethod
    def backward(self, dout=1):
        pass

############################## Sample Class ##############################

class Sample:
    def __init__(self, output_units, activation=None, batch_size=None):
        """
        Abstract
        ---------------
        レイヤのforward, backwardを行う. 

        Params
        ---------------
        """
        # forward

        # backward

    def forward(self, X, train_flg):
        pass

    def backward(self, dout):
        pass


########################### Layer Class ###########################

class Input(AbstractLayer):
    def __init__(self, output_units):
        super().__init__(0, output_units, 'identify')
        
    def forward(self, x, train_flg):
        return super().forward(x, self.__forward, self.activation, train_flg)

    def __forward(self, x, train_flg=None):
        return x
    
    def backward(self, dout):
        return super().backward(dout, self.__backward, self.activation)

    def __backward(self, dout):
        return dout

    def compile(self, model, batch_size, input_units, output_units, idx):
        pass

class Dense(AbstractLayer):
    def __init__(self, output_units, activation):
        # レイヤクラス共通の初期化
        super().__init__(None, output_units, activation)

        # レイヤ個別の初期化
        # パラメータ
        self.W = None
        self.b = None

        # backward時に使用する値を保持する変数
        self.x = None

        # gradientを保持する変数
        self.dW = None
        self.db = None

    def forward(self, x, train_flg=None):
        return super().forward(x, self.__forward, self.activation, train_flg)

    def __forward(self, x, train_flg):
        # backward時に使用する値を保存
        self.x = x 
        return  np.dot(self.x, self.W) + self.b

    def backward(self, dout):
        return super().backward(dout, self.__backward, self.activation)

    def __backward(self, dout):
        # gradientを保存
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = np.dot(dout, self.W.T)

        return dx
    
    def compile(self, model, batch_size, input_units, output_units, idx):
        scale = 1.0
        if self.activation_key.lower() is 'relu' :
            scale = np.sqrt(2.0 / input_units)
        elif self.activation_key.lower() is 'sigmoid' :
            scale = np.sqrt(1.0 / input_units)

        model.params['W' + str(idx)] =  np.random.normal(loc=0.0, scale=scale, size=[input_units[0], output_units[0]])
        model.params['b' + str(idx)] =  np.zeros(output_units[0])

        self.W = model.params['W' + str(idx)]
        self.b = model.params['b' + str(idx)]


class Flatten(AbstractLayer):
    """
    Abstract
    ---------------
    平坦化レイヤのforward, backwardを行う. 

    Params
    ---------------
    None
    """

    def __init__(self):
        super().__init__(None, None, 'identify')
        self.original_x_shape = None
    
    def forward(self, x, train_flg):
        return super().forward(x, self.__forward, self.activation, train_flg)

    def __forward(self, x, train_flg=None):
        self.original_x_shape = x.shape  # インプットのオリジナルの形状を保存（backwardで使用）
        x = x.reshape(x.shape[0], -1)  # バッチサイズ×1つ分のデータサイズに整形

        return x

    def backward(self, dout):
        return super().backward(dout, self.__backward, self.activation)

    def __backward(self, dout):
        dx = dout.reshape(*self.original_x_shape)

        return dx

    def compile(self, model, batch_size, input_units, output_units, idx):
        pass


class Conv2D(AbstractLayer):
    def __init__(self, filter_num, filter_size, activation='identify', stride=1, pad=0):
        super().__init__(None, None, activation)
        self.filter_num = filter_num
        self.filter_size =  (filter_size, filter_size) if type(filter_size) is int else filter_size
        self.filter_height = self.filter_size[0]
        self.filter_width = self.filter_size[1]
        self.output_height = None
        self.output_width = None        

        # レイヤ個別の初期化
        # パラメータ
        self.W = None
        self.b = None
        self.stride = stride
        self.pad = pad

        # backward時に使用する値を保持する変数
        self.x = None
        self.col = None
        self.col = None
        self.col_W = None

        # gradientを保持する変数
        self.dW = None
        self.db = None
    
    def forward(self, x, train_flg):
        return super().forward(x, self.__forward, self.activation, train_flg)

    def __forward(self, x, train_flg=None):
        col = im2col(x, self.filter_size[0], self.filter_size[1], self.stride, self.pad)
        col_W = self.W.reshape(self.filter_num, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(x.shape[0], self.output_height, self.output_width, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out
    
    def backward(self, dout):
        return super().backward(dout, self.__backward, self.activation)

    def __backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, self.filter_num)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1,0).reshape(self.filter_num, -1, self.filter_size[0], self.filter_size[1])

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, self.filter_size[0], self.filter_size[1], self.stride, self.pad)

        return dx

    def compile(self, model, batch_size, input_units, output_units, idx):
        scale = 1.0
        if self.activation_key.lower() is 'relu' :
            scale = np.sqrt(2.0 / input_units)
        elif self.activation_key.lower() is 'sigmoid' :
            scale = np.sqrt(1.0 / input_units)

        model.params['W' + str(idx)] =  np.random.normal(loc=0.0, scale=scale, size=[self.filter_num, input_units[0], self.filter_height, self.filter_width])
        model.params['b' + str(idx)] =  np.zeros(output_units[0])

        self.W = model.params['W' + str(idx)]
        self.b = model.params['b' + str(idx)]


class MaxPooling2D(AbstractLayer):
    def __init__(self, filter_size, stride=2, pad=0):
        super().__init__(None, None, 'identify')
        self.filter_size =  (filter_size, filter_size) if type(filter_size) is int else filter_size
        self.pool_height = self.filter_size[0]
        self.pool_width = self.filter_size[1]
        self.output_height = None
        self.output_width = None
        self.stride = stride
        self.pad = pad  # パディング未実装

        self.x = None
        self.arg_max = None

    def forward(self, x, train_flg):
        return super().forward(x, self.__forward, self.activation, train_flg)

    def __forward(self, x, train_flg=None):
        col = im2col(x, self.pool_height, self.pool_width, self.stride, self.pad)
        col = col.reshape(-1, self.pool_height*self.pool_width)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(x.shape[0], self.output_height, self.output_width, x.shape[1]).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        return super().backward(dout, self.__backward, self.activation)

    def __backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_height * self.pool_width
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_height, self.pool_width, self.stride, self.pad)

        return dx

    def compile(self, model, batch_size, input_units, output_units, idx):
        pass

class Categorical_crossEntropy(AbstractWithLossLayer):
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    
    def forward(self, x, t, train_flg):
        return super().forward(x, t, softmax, cross_entropy_error, train_flg)

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        # 正解データがone-hot-vector表現の場合
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        # 正解データがラベル表現の場合
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx


lastLayer_class_dict = {'categorical_crossentropy':Categorical_crossEntropy}

class Activation():
    def __init__(self, activation):
        self.activation_key = activation
        self.activation = activationLayer_class_dict[self.activation_key.lower()]()
    
    def forward(self, x):
        return self.activation.forward(x)

    def backward(self, dout):
        return self.activation.backward(dout)

class Relu():
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

class Identify():
    def __init__(self):
        pass
    
    def forward(self, x):
        return x
    
    def backward(self, dout):
        return dout

activationLayer_class_dict = {'relu':Relu, 'identify':Identify}