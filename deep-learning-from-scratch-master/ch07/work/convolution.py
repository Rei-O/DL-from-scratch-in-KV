import sys, os
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'))  # 親の親ディレクトリのファイルをインポートするための設
sys.path.append(os.path.join(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'), '..'))  # 親の親の親ディレクトリのファイルをインポートするための設定

import numpy as np
from common.presentation.util import im2col  # image to column

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W              # フィルター
        self.b = b              # バイアス
        self.stride = stride    # ストライド幅
        self.pad = pad          # パディング幅

    def forward(self, X):
        filter_batch_size, filter_channel_num , filter_height, filter_width = self.W.shape
        input_batch_size, input_channel_num, input_height, input_width = X.shape

        # output_sizeの計算
        output_batch_size = input_batch_size
        output_height = int((input_height - filter_height + self.pad*2)/self.stride + 1)
        output_width = int((input_width - filter_width + self.pad*2)/self.stride + 1)

        # インプット、フィルターを2次元配列化
        col_X = im2col(X, filter_height, filter_width, stride=self.stride, pad=self.pad)
        col_W = self.W.reshape(filter_batch_size, -1).T  

        out = np.dot(col_X, col_W) + self.b
        out = out.reshape(output_batch_size, output_height, output_width, -1).transpose(0, 3, 1, 2)

        return out
    
    # def backward(self, dout):

if __name__ == '__main__':
    # im2colの使用例
    x1 = np.random.rand(1,3,7,7)  # [0, 1)のランダム値を(データ数, チャンネル数, height, width)で生成
    col1 = im2col(x1, 5, 5, stride=1, pad=0)  # イメージデータ, filter_height, filter_width, ストライド幅, パディング幅
    print(col1.shape)  # height(width)は(7+0*2)-5+1=3なのでoutput_sizeは(3, 3). フィルターサイズは(3, 5, 5). よってim2colのサイズは(3*3, 3*5*5)

    x2 = np.random.rand(10, 3, 7, 7)  # [0, 1)のランダム値を(データ数, チャンネル数, height, width)で生成
    col2 = im2col(x2, 5, 5, stride=1, pad=1)  # イメージデータ, filter_height, filter_width, ストライド幅, パディング幅
    print(col2.shape)  # バッチサイズが10、height(width)は(7+1*2)-5+1=5なのでoutput_sizeは(10, 5, 5). フィルターサイズは(3, 5, 5). よってim2colのサイズは(10*5*5, 3*5*5)

    # Convolution Class の動作確認
    filter = np.random.rand(10, 3, 3, 3)
    b = np.random.rand(3, 1, 1)
    convLayer = Convolution(filter, b)

    out = convLayer.forward(x2)
    print(out.shape)
