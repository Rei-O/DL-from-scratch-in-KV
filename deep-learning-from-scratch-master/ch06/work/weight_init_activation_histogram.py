import sys, os
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'))  # 親ディレクトリの親ディレクトリのファイルをインポートするための設定
import numpy as np
from matplotlib import pyplot as plt
from common.presentation.functions import sigmoid

###############################################
# アクティベーション（活性化関数後の出力値）分布 #
###############################################

stds  = [1, 0.1, 0.01] # 標準偏差

x = np.random.randn(1000, 100) # 1000*100のndarrayを生成(100次元のデータを1000個生成)
node_num = 100                 # 各中間層のノード数
hidden_layer_size = 5          # 中間層の数
activations = {}               # アクティベーションを格納

for j in range(len(stds)):
    ######################### フォワード #########################
    for i in range(hidden_layer_size):
        # 2層目以降は前層のアクティベーションを取得
        if i != 0:
            x = activations[i-1]
        
        # 標準正規分布に従う重みを生成
        w = np.random.randn(node_num, node_num) * stds[j]

        # 線形和を計算
        z = np.dot(x, w)
        # アクティベーションを計算
        a = sigmoid(z)
        # アクティベーションを格納
        activations[i] = a

    ###################### ヒストグラム描画 ######################
    for i, a in activations.items():
        plt.subplot(len(stds), len(activations), j*len(activations)+i+1)
        plt.title(str(i+1) + "-layer (std :" + str(stds[j]) + ")")
        plt.hist(a.flatten(), 30, range=(0,1))
plt.show()