import sys, os
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'))  # 親の親ディレクトリのファイルをインポートするための設
sys.path.append(os.path.join(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'), '..'))  # 親の親の親ディレクトリのファイルをインポートするための設定

import numpy as np
from common.presentation.util import im2col, col2im  # image to column, column to image

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        # 初期値
        self.W = W              # フィルター
        self.b = b              # バイアス
        self.stride = stride    # ストライド幅
        self.pad = pad          # パディング幅

        # backward時に使用する中間データ
        self.X = None
        self.col_X = None
        self.col_W = None

        # gradientを格納
        self.db = None
        self.dW = None


    def forward(self, X):
        self.X = X
        filter_batch_size, filter_channel_num , filter_height, filter_width = self.W.shape
        input_batch_size, input_channel_num, input_height, input_width = X.shape

        # output_sizeの計算
        output_batch_size = input_batch_size
        output_height = int((input_height - filter_height + self.pad*2)/self.stride + 1)
        output_width = int((input_width - filter_width + self.pad*2)/self.stride + 1)

        # インプット、フィルターを2次元配列化
        self.col_X = im2col(X, filter_height, filter_width, stride=self.stride, pad=self.pad)
        self.col_W = self.W.reshape(filter_batch_size, -1).T  

        out = np.dot(self.col_X, self.col_W) + self.b
        out = out.reshape(output_batch_size, output_height, output_width, -1).transpose(0, 3, 1, 2)  # 指定した順に軸を変更する

        return out
    
    def backward(self, dout):
        # Affineと同じbackwardの後に形状変換の逆変換をするだけ
        filter_batch_size, filter_channel_num , filter_height, filter_width = self.W.shape        
        dout = dout.transpose(0,2,3,1).reshape(-1, filter_batch_size)

        # Affineと同じbackward計算
        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col_X.T, dout)
        dcol_X = np.dot(dout, self.col_W.T)

        # 形状変換の逆変換
        self.dW = self.dW.transpose(1, 0).reshape(filter_batch_size, filter_channel_num , filter_height, filter_width)
        dx = col2im(dcol_X, self.X.shape, filter_height, filter_width, self.stride, self.pad)

        return dx

class Pooling:
    def __init__(self, pool_height, pool_width, stride=2, pad=0):
        # 初期値
        self.pool_height = pool_height
        self.pool_width = pool_width
        self.stride = stride
        self.pad = pad

        # backward時に使用する中間データ
        self.X = None
        self.arg_max = None

    def forward(self, X):
        input_batch_size, input_channel_num, input_height, input_width = X.shape
        output_height = int((input_height - self.pool_height + self.pad*2)/self.stride + 1)
        output_width = int((input_width - self.pool_width + self.pad*2)/self.stride + 1)

        # 展開
        col = im2col(X, self.pool_height, self.pool_width, self.stride, self.pad)
        col = col.reshape(-1, self.pool_height*self.pool_width)

        # 最大値取得
        out = np.max(col, axis=1)

        # 整形
        out = out.reshape(input_batch_size, output_height, output_width, input_channel_num).transpose(0, 3, 1, 2)

        # 中間データを格納
        self.X = X
        self.arg_max = np.argmax(col, axis=1)

        return out
    
    def backward(self, dout):
        # MAXプーリングの場合、ReLUのようにMAXの要素だけ1、それ以外は0を返す
        dout = dout.transpose(0, 2, 3, 1)  #バッチサイズ, height, width, channel数の順に変換

        pool_size = self.pool_height * self.pool_width  # プーリングフィルターの要素数
        dmax = np.zeros((dout.size, pool_size))  # doutの要素数×プーリングフィルターの要素数のゼロ配列を生成(返り値を入れておく受け皿)
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()  # 各ウィンドウでのmaxの要素に後層からの連携値を渡す
        dmax = dmax.reshape(dout.shape + (pool_size, ))  # タプル同士の結合で5次元にreshape

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)  # バッチサイズ×height×width, channel数×プーリングサイズに変換（←これがim2colの出力結果、上の変換は徐々にこの形に近づけるため？？）
        dx = col2im(dcol, self.X.shape, self.pool_height, self.pool_width, self.stride, self.pad)

        return dx

if __name__ == '__main__':
    #################
    # im2colの使用例 #
    #################
    x1 = np.random.rand(1,3,7,7)  # [0, 1)のランダム値を(データ数, チャンネル数, height, width)で生成
    col1 = im2col(x1, 5, 5, stride=1, pad=0)  # イメージデータ, filter_height, filter_width, ストライド幅, パディング幅
    print(col1.shape)  # height(width)は(7+0*2)-5+1=3なのでoutput_sizeは(3, 3). フィルターサイズは(3, 5, 5). よってim2colのサイズは(3*3, 3*5*5)

    x2 = np.random.rand(10, 3, 7, 7)  # [0, 1)のランダム値を(データ数, チャンネル数, height, width)で生成
    col2 = im2col(x2, 5, 5, stride=1, pad=1)  # イメージデータ, filter_height, filter_width, ストライド幅, パディング幅
    print(col2.shape)  # バッチサイズが10、height(width)は(7+1*2)-5+1=5なのでoutput_sizeは(10, 5, 5). フィルターサイズは(3, 5, 5). よってim2colのサイズは(10*5*5, 3*5*5)


    ###############################
    # Convolution Class の動作確認 #
    ###############################
    # 重み(フィルター)、バイアスを生成
    filter_num = 3
    channel_num = 3
    filter_height = 5
    filter_width = 5
    W = np.random.rand(filter_num, channel_num, filter_height, filter_width)
    b = np.zeros(filter_num)

    # Convolution Layerを生成
    convLayer = Convolution(W, b)

    # インプットデータ生成
    input_num = 10
    input_height = 7
    input_width = 7
    x3 = np.random.rand(input_num, channel_num, input_height, input_width)

    # forward
    out = convLayer.forward(x3)
    print(out.shape)

    # backward
    dx = convLayer.backward(out)  # ホントは微分値を渡すので違うが、、
    print(dx.shape)