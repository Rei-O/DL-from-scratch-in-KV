import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import abc
from collections import OrderedDict
from reiras.optimizers import *
from reiras.layers import *
from reiras.metrics import *

########################### Abstract Class ###########################

class AbstractModel(metaclass=abc.ABCMeta):
    """
    学習モデルの抽象クラス
    """
    @abc.abstractmethod
    def add(self):
        """
        レイヤを追加する
        """
        pass

    @abc.abstractmethod
    def summary(self):
        """
        レイヤ構成を表示する
        """
        pass

    @abc.abstractmethod
    def compile(self):
        """
        モデルのコンパイルを行う
        """
        pass

    @abc.abstractmethod
    def fit(self):
        """
        モデルの学習を行う
        """
        pass

    @abc.abstractmethod
    def evalute(self):
        """
        学習済みモデルの精度評価を行う
        """
        pass

    @abc.abstractmethod
    def predict(self):
        """
        学習済みモデルによる予測を行う
        """
        pass

    @abc.abstractmethod
    def predict_classes(self):
        """
        学習済みモデルによる予測を行う
        分類クラスを出力する
        """
        pass

    @abc.abstractmethod
    def predict_proba(self):
        """
        学習済みモデルによる予測を行う
        クラスごとの確率を出力する
        """
        pass


########################### Model Class ###########################

class NuralNet(AbstractModel):

    def __init__(self):
        self.layersOrderDict = OrderedDict()
        self.input_layer_size_list = []
        self.output_layer_size_list = []
        self.params = {}

    def add(self, layer):
        """
        レイヤを追加する
        """
        self.layersOrderDict[type(layer).__name__ + '_' + str(len(self.layersOrderDict))] = layer
        self.input_layer_size_list.append(self.output_layer_size_list[-1] if layer.input_units is None else layer.input_units)
        self.output_layer_size_list.append(layer.output_units)

    def summary(self):
        """
        レイヤ構成を表示する
        """
        pass

    def compile(self, optimizer, loss, metrics, output_units, batch_size=1, **optimizer_param):
        """
        モデルのコンパイルを行う
        """
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)

        self.last_layer = lastLayer_class_dict[loss.lower()]()

        self.metrics = metric_class_dict[metrics.lower()]

        self.output_units = output_units if type(output_units) is tuple else (output_units,)
        self.input_layer_size_list.append(self.output_layer_size_list[-1])
        self.output_layer_size_list.append(self.output_units)

        self.batch_size = batch_size

        idx = 0
        for layer in self.layersOrderDict.values():
            layer.compile(self, self.batch_size, self.input_layer_size_list[idx], self.output_layer_size_list[idx], idx)
            idx += 1
            

    def fit(self, x_train, t_train, x_test, t_test, epochs=10):
        """
        モデルの学習を行う
        """
        self.train_loss_list = []
        self.train_evalute_list = []
        self.test_evalute_list = []
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test

        self.epochs = epochs
        self.train_size = self.x_train.shape[0]
        self.iter_per_epoch = max(self.train_size/self.batch_size, 1)
        self.max_iter = int(self.epochs * self.iter_per_epoch)

        for i in range(self.max_iter):
            batch_mask = np.random.choice(self.train_size, self.batch_size if self.batch_size < self.train_size else self.train_size)
            x_batch = self.x_train[batch_mask]
            t_batch = self.t_train[batch_mask]

            grads = self.__gradient(x_batch, t_batch)
            self.optimizer.update(self.params, grads)

            # TODO: lossを実装するか要検討
            train_loss = self.__predict(x_batch, t_batch)
            self.train_loss_list.append(train_loss)

            # 1エポック開始時の場合
            if i % self.iter_per_epoch == 0:
                print(f'=========== epoch :{int(i/self.iter_per_epoch)} ===========')
                train_evalute = self.evalute(self.x_train, self.t_train)
                self.train_evalute_list.append(train_evalute)
                print(f'train score : {train_evalute}')
                test_evalute = self.evalute(x_test, t_test)
                self.test_evalute_list.append(test_evalute)
                print(f'test score : {test_evalute}')
        
    def __gradient(self, x, t):
        # forward
        self.__predict(x, t, train_flg=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)
        
        layers = list(self.layersOrderDict.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grads = {}

        idx = len(self.layersOrderDict)-1
        for key, layer in self.layersOrderDict.items().__reversed__():
            if 'dense' in key.lower():
                grads['W' + str(idx)] = layer.dW
                grads['b' + str(idx)] = layer.db                
            idx -= 1

        return grads

    def evalute(self, x, t):
        """
        学習済みモデルの精度評価を行う
        """
        y = self.__predict(x, t, train_flg=False)
        return self.metrics(y, t)

    def __predict(self, x, t, train_flg=True):
        """
        学習済みモデルによる予測を行う
        """
        for key, layer in self.layersOrderDict.items():
            if 'dropout' in key.lower():
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        
        return self.last_layer.forward(x, t, train_flg)

    def predict(self, x):
        """
        学習済みモデルによる予測を行う
        """
        y = self.__predict(x, None, train_flg=False)

        return y

    def predict_classes(self):
        """
        学習済みモデルによる予測を行う
        分類クラスを出力する
        """
        pass

    def predict_proba(self):
        """
        学習済みモデルによる予測を行う
        クラスごとの確率を出力する
        """
        pass