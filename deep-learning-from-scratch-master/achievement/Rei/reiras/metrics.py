import numpy as np

def accuracy(x, t):
    # 正解データがラベル表現でない場合
    if t.ndim != 1 : t = np.argmax(t, axis=1)

    # 初期化
    acc = 0.0

    x = np.argmax(x, axis=1)
    acc += np.sum(x == t)
    
    return acc / x.shape[0]


metric_class_dict = {'accuracy':accuracy}