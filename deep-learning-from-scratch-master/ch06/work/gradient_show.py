# 参考: https://watlab-blog.com/2020/02/29/gradient-descent/#1-2
# 最小値方向と勾配方向が一致しない場合学習が非効率になる例

from pickletools import optimize
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from optimizers import SGD

# 初期値設定
max_iteration = 100                 # 最大反復回数
x0 = 5.0                            # 初期値x0
y0 = 5.0                            # 初期値y0
x_pred = [x0]                       # 描画用x0軌跡リスト(初期値をプリセット)
y_pred = [y0]                       # 描画用y0軌跡リスト(初期値をプリセット)

# パラメータ最適化を行う関数
def f(x, y):
    z = (1/20)*x**2 + y**2
    return z

# パラメータ最適化を行う関数の微分
def df(x):
    dzdx = (1/10)*x[0]
    dzdy = 2*x[1]
    dz = np.array([dzdx, dzdy])
    return dz

# optimizerに渡す引数を定義
params = {}
params["init"] =  np.array([x0, y0])

# オプティマイザーのインスタンス生成
optimizer = SGD()

# 最大反復回数まで計算する
for i in range(max_iteration):
    grads = {}
    grads["init"] = df(params["init"])
    optimizer.update(params, grads)
    x_pred.append(params["init"][0])               # x0の軌跡をリストに追加
    y_pred.append(params["init"][1])               # y0の軌跡をリストに追加
    print(i, params["init"][0], params["init"][1])

x_pred = np.array(x_pred)           # 描画用にx0をnumpy配列変換
y_pred = np.array(y_pred)           # 描画用にx0をnumpy配列変換
z_pred = f(x_pred, y_pred)          # 軌跡のz値を計算

# 基準関数の表示用
x = np.arange(0, 6, 0.2)
y = np.arange(-5, 5, 0.2)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# ここからグラフ描画----------------------------------------------------------------
# フォントの種類とサイズを設定する。
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'Times New Roman'

#  グラフの入れ物を用意する。
fig = plt.figure()
ax1 = Axes3D(fig)

# 軸のラベルを設定する。
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

# データプロットする。
ax1.plot_wireframe(X, Y, Z, label='f(x, y)')
ax1.scatter3D(x_pred, y_pred, z_pred, label='gd', color='red', s=100)

# グラフを表示する。
plt.show()
plt.close()
# ---------------------------------------------------------------------------------