import sys, os
sys.path.append(os.path.join(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'), '..'))  # 親の親の親ディレクトリのファイルをインポートするための設定
import numpy as np

import common.presentation.debug as debug

debug.isDebugMode = True

def step_function(x):
    return np.array(x > 0, dtype=np.int)

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

if __name__ == '__main__':
    x = np.array([-10,-1,0,1,10])
    debug.debugprt(x,"入力")
    debug.debugprt(step_function(x),"ステップ関数")
    debug.debugprt(sigmoid(x),"シグモイド関数")
    debug.debugprt(relu(x),"ReLU関数")