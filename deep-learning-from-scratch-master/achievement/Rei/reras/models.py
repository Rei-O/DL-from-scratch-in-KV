import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import abc
from collections import OrderedDict
from reras.optimizers import *
from reras.layers import *
from reras.metrics import *

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
        モデルにレイヤを追加する
        
        Params
        ---------------
        layer(layer class) : レイヤ

        Return
        ---------------
        None
        """
        # レイヤを格納
        self.layersOrderDict[type(layer).__name__ + '_' + str(len(self.layersOrderDict))] = layer
        

    # TODO : Step3実装予定
    def summary(self):
        """
        レイヤ構成を表示する
        Params
        ---------------
        None

        Return
        ---------------
        None
        """
        __layer_print_length = 40
        __input_print_length = 40
        __output_print_length = 40
        __line_print_length = __layer_print_length + __input_print_length + __output_print_length
        # __param_print_length = 10
        # __line_print_length = __layer_print_length + __input_print_length + __output_print_length + __param_print_length

        print('='*__line_print_length)
        print('Layer'.ljust(__layer_print_length) + 'Input'.ljust(__input_print_length) + 'Output'.ljust(__output_print_length))
        # print('Layer'.ljust(__layer_print_length) + 'Input'.ljust(__input_print_length) + 'Output'.ljust(__output_print_length) + 'Param'.ljust(__param_print_length))
        print('='*__line_print_length)
        idx = 0
        for key, layer in self.layersOrderDict.items():
            __str = str(key).ljust(__layer_print_length)
            __str += str((self.batch_size,) + self.input_layer_size_list[idx]).ljust(__input_print_length)
            __str += str((self.batch_size,) + self.output_layer_size_list[idx]).ljust(__output_print_length)
            print(__str)
            print('-'*__line_print_length)
            idx += 1            


    def compile(self, optimizer, loss, metrics, output_units, batch_size=1, **optimizer_param):
        """
        モデルのコンパイルを行う

        Params
        ---------------
        optimizer(str) : オプティマイザ名
        loss(str) : loss関数名
        metrics(str) : score関数名
        output_units(int) : 出力層のノード数
        batch_size(int) : ミニバッチサイズ
        **optimizer_param(dict) : オプティマイザのハイパーパラメータ

        Return
        ---------------
        None
        """
        # オプティマイザ初期化
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        # 出力層初期化
        self.last_layer = lastLayer_class_dict[loss.lower()]()
        # score関数定義
        self.metrics = metric_class_dict[metrics.lower()]

        # 各層の重み形状を算出
        for layer in self.layersOrderDict.values():
            self.__setLayerSize(layer)

        # 出力層のノード数を格納
        self.output_units = output_units if type(output_units) is tuple else (output_units,)
        # 出力層のインプットサイズを格納
        self.input_layer_size_list.append(self.output_layer_size_list[-1])
        # 出力層のアウトプットサイズを格納
        self.output_layer_size_list.append(self.output_units)

        # ミニバッチサイズを格納
        self.batch_size = batch_size

        # 各レイヤのコンパイル（パラメータ初期化）
        idx = 0
        for layer in self.layersOrderDict.values():
            layer.compile(self, self.batch_size, self.input_layer_size_list[idx], self.output_layer_size_list[idx], idx)
            idx += 1

    def __setLayerSize(self, layer):    
        # インプットサイズを格納
        __input_units = self.output_layer_size_list[-1] if layer.input_units[0] is None else layer.input_units
        self.input_layer_size_list.append(__input_units)

        # アウトプットサイズを格納
        __output_units = 1
        #　Flattenの場合
        if type(layer) is Flatten :
            # 平坦化後のoutput_unitsに更新
            for size in self.output_layer_size_list[-1]:
                __output_units *= size
        # Conv2Dの場合
        elif type(layer) is Conv2D:
            layer.output_height = int(1 + (self.input_layer_size_list[-1][1] + 2*layer.pad - layer.filter_size[0]) / layer.stride)
            layer.output_width = int(1 + (self.input_layer_size_list[-1][2] + 2*layer.pad - layer.filter_size[1]) / layer.stride)
            __output_units = (layer.filter_num, layer.output_height, layer.output_width)
        # MaxPooling2Dの場合
        elif type(layer) is MaxPooling2D:
            layer.output_height = int(1 + (self.input_layer_size_list[-1][1] - layer.pool_height) / layer.stride)
            layer.output_width = int(1 + (self.input_layer_size_list[-1][2] - layer.pool_width) / layer.stride)
            __output_units = (self.input_layer_size_list[-1][0], layer.output_height, layer.output_width)
        else:
            __output_units = layer.output_units

        self.output_layer_size_list.append(__output_units if type(__output_units) is tuple else (__output_units,))

    def fit(self, x_train, x_test, t_train, t_test, epochs=10):
        """
        モデルの学習を行う

        Params
        ---------------
        x_train(numpy.ndarray) : 学習データ
        x_test(numpy.ndarray) : 学習用正解データ
        t_train(numpy.ndarray) : テストデータ
        t_test(numpy.ndarray) : テスト用正解データ
        epochs(int) : エポック数

        Return
        ---------------
        None
        """
        self.train_loss_list = []
        self.train_evalute_list = []
        self.test_evalute_list = []
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        # 学習データサイズ
        self.train_size = self.x_train.shape[0]
        # 1エポックあたりの学習繰返し回数
        self.iter_per_epoch = max(int(self.train_size/self.batch_size), 1)
        # 全学習繰返し回数
        self.max_iter = int(self.epochs * self.iter_per_epoch)

        # 1学習ごとの繰返し処理
        for i in range(1, self.max_iter+1):
            # ミニバッチ学習用データ作成
            batch_mask = np.random.choice(self.train_size, self.batch_size if self.batch_size < self.train_size else self.train_size)
            x_batch = self.x_train[batch_mask]
            t_batch = self.t_train[batch_mask]

            # 学習（forward → loss →backward）
            grads = self.__gradient(x_batch, t_batch)
            # パラメータ更新
            self.optimizer.update(self.params, grads)

            # loss算出
            train_loss = self.__predict(x_batch, t_batch)
            self.train_loss_list.append(train_loss)

            # 1エポック開始時の場合
            if i % self.iter_per_epoch == 0:
                print(f'=========== epoch : {int(i/self.iter_per_epoch)} ===========')
                train_evalute = self.evalute(self.x_train, self.t_train)
                self.train_evalute_list.append(train_evalute)
                print(f'train score : {train_evalute}')
                test_evalute = self.evalute(x_test, t_test)
                self.test_evalute_list.append(test_evalute)
                print(f'test score : {test_evalute}')

    def __gradient(self, x, t):
        """
        モデルの学習を行う

        Params
        ---------------
        x(numpy.ndarray) : 学習データ
        t(numpy.ndarray) : 正解データ

        Return
        ---------------
        grads(dictionary) : gradient結果
        """
        # forward #######################################
        self.__predict(x, t, train_flg=True)

        # backward ######################################
        # dout初期化
        dout = 1
        # 出力層の逆伝播
        dout = self.last_layer.backward(dout)
        # 逆伝播用にレイヤを逆順に保持したリストを作成
        layers = list(self.layersOrderDict.values())
        layers.reverse()
        # 中間層の逆伝播
        for layer in layers:
            dout = layer.backward(dout)

        # 逆伝播結果を格納 ###############################
        # 格納用dictionary初期化
        grads = {}
        # 
        idx = len(self.layersOrderDict)-1
        for key, layer in self.layersOrderDict.items().__reversed__():  # レイヤを逆順でパラメータ格納
            # 各レイヤのパラメータを格納
            # Denseレイヤの場合
            if 'dense' in key.lower():
                grads['W' + str(idx)] = layer.dW
                grads['b' + str(idx)] = layer.db                
            # Conv2Dレイヤの場合
            elif 'conv2d' in key.lower():
                grads['W' + str(idx)] = layer.dW
                grads['b' + str(idx)] = layer.db                
            idx -= 1

        return grads

    def evalute(self, x, t):
        """
        学習済みモデルの精度評価を行う

        Params
        ---------------
        x(numpy.ndarray) : 学習データ
        t(numpy.ndarray) : 正解データ

        Return
        ---------------
        score(float) : 予測精度
        """
        y = self.__predict(x, t, train_flg=False)
        score = self.metrics(y, t)

        return score

    def __predict(self, x, t, train_flg=True):
        """
        学習済みモデルによる予測を行う

        Params
        ---------------
        x(numpy.ndarray) : 学習データ
        t(numpy.ndarray) : 正解データ
        train_flg(boolean) : 学習フラグ

        Return
        ---------------
        last_layer.forward(numpy.ndarray) : 予測結果
        """
        for key, layer in self.layersOrderDict.items():
            x = layer.forward(x, train_flg)
        
        return self.last_layer.forward(x, t, train_flg)

    def predict(self, x):
        """
        学習済みモデルによる予測を行う

        Params
        ---------------
        x(numpy.ndarray) : 予測データ

        Return
        ---------------
        y(numpy.ndarray) : 予測結果
        """
        y = self.__predict(x, None, train_flg=False)

        return y

    # TODO : Step3実装予定
    def predict_classes(self):
        """
        学習済みモデルによる予測を行う
        分類クラスを出力する

        Params
        ---------------
        x(numpy.ndarray) : 予測データ

        Return
        ---------------
        y(numpy.ndarray) : 予測結果
        """
        pass

    # TODO : Step3実装予定
    def predict_proba(self):
        """
        学習済みモデルによる予測を行う
        クラスごとの確率を出力する

        Params
        ---------------
        x(numpy.ndarray) : 予測データ

        Return
        ---------------
        y(numpy.ndarray) : 予測結果
        """
        pass