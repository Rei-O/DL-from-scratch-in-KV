import numpy as np

#================================
# 活性化関数：
#   Step関数
#
# 説明：
#   しきい値以上で1、未満で0を返す
#================================

def step_fnc(X):
    # 重み
    W = np.array([0.5,0.5])
    # バイアス
    b = -0.7
    y = np.sum(W*X)+b
    if y >= 0:
        return 1
    else:
        return 0
X = np.array([1,1])

print(step_fnc(X))