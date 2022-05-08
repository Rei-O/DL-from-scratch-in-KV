import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import abc
from reiras.functions import *

########################### Abstract Class ###########################

class AbstractLayer(metaclass=abc.ABCMeta):
    """
    通常レイヤの抽象クラス
    """
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def forward(self, x, train_flg=True):
        pass

    @abc.abstractmethod
    def backward(self, dout):
        pass


class AbstractWithLossLayer(metaclass=abc.ABCMeta):
    """
    損失の計算を行うレイヤの抽象クラス
    """
    @abc.abstractmethod
    def forward(self, X, t):
        pass

    @abc.abstractmethod
    def backward(self, dout=1):
        pass

############################## Sample Class ##############################

class Sample:
    def __init__(self, units, activation=None, ):
        """
        Abstract
        ---------------
        レイヤのforward, backwardを行う. 

        Params
        ---------------
        """
        # forward

        # backward

    def forward(self, X, train_flg=True):
        pass

    def backward(self, dout):
        pass


########################### Layer Class ###########################

class Input(AbstractLayer):
    def __init__(self, output_units, batch_size=None):
        self.input_units = 0
        self.output_units = output_units if type(output_units) is tuple else (output_units,)
        

    def forward(self, x, train_flg=True):
        return x
    
    def backward(self, dout):
        return dout

    def compile(self, model, batch_size, input_units, output_units, idx):
        pass

class Dense(AbstractLayer):
    def __init__(self, output_units, activation):
        self.input_units = None
        self.output_units = output_units if type(output_units) is tuple else (output_units,)
        self.activation_key = activation
        self.activation = Activation(activation)
        self.W = None
        self.b = None

        # backward時に使用する値を保持する
        self.x = None

        # gradientを保持する
        self.dW = None
        self.db = None
    
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

    def forward(self, x):
        # backward時に使用する値を保存
        self.x = x 
        y = np.dot(self.x, self.W) + self.b

        out = self.activation.forward(y)

        return out
    
    def backward(self, dout):

        dy = self.activation.backward(dout)

        # gradientを保存
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = np.dot(dout, self.W.T)

        return dx


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
        self.original_x_shape = None
    
    def forward(self, x):
        self.original_x_shape = x.shape  # インプットのオリジナルの形状を保存（backwardで使用）
        x = x.reshape(x.shape[0], -1)  # バッチサイズ×1つ分のデータサイズに整形

        return x

    def backward(self, dout):
        dx = dout.reshape(*self.original_x_shape)

        return dx

class Categorical_crossentropy():
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t, train_flg):
        self.t = t
        self.y = softmax(x)

        if train_flg:
            self.loss = cross_entropy_error(self.y, self.t)
            return self.loss
        else:
            return self.y

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


lastLayer_class_dict = {'categorical_crossentropy':Categorical_crossentropy}

class Activation():
    def __init__(self, activation):
        self.activation_key = activation
        self.activation = activationLayer_class_dict[activation.lower()]()
    
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

activationLayer_class_dict = {'relu':Relu}