import numpy as np
import abc
from functions import softmax

############################## Sample ##############################

class Sample:
    def __init__(self):
        """
        Abstract
        ---------------
        レイヤのforward, backwardを行う. 

        Params
        ---------------
        """
        # forward

        # backward

    def forward(self, X):
        pass

    def backward(self, dout):
        pass

####################################################################

########################### Abstract Class ###########################

class Layer(metaclass=abc.ABCMeta):
    """
    通常レイヤの抽象クラス
    """
    @abc.abstractmethod
    def forward(self, X):
        pass

    @abc.abstractmethod
    def backward(self, dout):
        pass


class LossLayer(metaclass=abc.ABCMeta):
    """
    損失の計算を行うレイヤの抽象クラス
    """
    @abc.abstractmethod
    def forward(self, X, t
    
    ):
        pass

    @abc.abstractmethod
    def backward(self, dout=1):
        pass
  


########################### Layer Class ###########################


class Affine(Layer):
    def __init__(self, W, b):
        """
        Abstract
        ---------------
        Affineレイヤのforward, backwardを行う. 

        Params
        ---------------
        W : 重み
        b : バイアス
        """
        # forward
        self.W = W
        self.b = b
        self.X = None
        self.org_X_shape = None  # テンソル対応

        # backward
        self.dW = None
        self.db = None

    def forward(self, X):
        self.org_X_shape = X.shape
        self.X = X.reshape(X.shape[0], -1)  # テンソル対応
        self.X = X
        Z = np.dot(self.X, self.W) + self.b

        return Z

    def backward(self, dout):
        self.dW = np.dot(self.X.T,dout)
        self.db = dout
        dX = np.dot(dout, self.W.T)
        dX = dX.reshape(*self.org_X_shape)  # テンソル対応で形状変更したため元に戻す

        return dX


class ReLU(Layer):
    def __init__(self):
        """
         Abstract
        ---------------
        ReLUレイヤのforward, backwardを行う. 


        Params
        ---------------
        None
        """
        # forward
        self.mask = None

        # backward

    def forward(self, X):
        self.mask = (X > 0)
        Z = self.mask * X

        return Z

    def backward(self, dout):
        dX = self.mask * dout

        return dX

class Softmax(Layer):
    def __init__(self):
        """
        Abstract
        ---------------
        Softmaxレイヤのforward, backwardを行う. 

        Params
        ---------------
        """
        # forward
        self.Y = None

        # backward

    def forward(self, X):
        Z = softmax(X)

    def backward(self, dout):
        pass

class Loss(LossLayer):
    def __init__(self):
        """
        Abstract
        ---------------
        Lossレイヤのforward, backwardを行う. 
        ※現状では交差エントロピー関数のみ対応

        Params
        ---------------
        """
        # forward

        # backward

    def forward(self, X):
        pass

    def backward(self, dout):
        pass

