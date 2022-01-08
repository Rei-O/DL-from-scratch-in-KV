import numpy as np
from layer_naive import *

# 初期値
apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# Layerの生成
# appleの小計を計算する乗算レイヤ
mul_apple_layer = MulLayer()
# orangeの小計を計算する乗算レイヤ
mul_orange_layer = MulLayer()
# appleとorrangeの合計金額を計算する加算レイヤ
add_apple_orange_layer = AddLayer()
# 消費税を乗じる乗算レイヤ
mul_tax_layer = MulLayer()


# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
sum_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(sum_price, tax)

print(f"price : {price}")


# backward
dprice = 1 # 合計金額をそのまま出力しているので活性化関数は恒等関数とみなす
dsum_price , dtax = mul_tax_layer.backward(dprice) # mul_tax_layer.forwardの出力の微分を受け取り、下流の微分に渡す
dapple_price, dorange_price = add_apple_orange_layer.backward(dsum_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(f"dtax : {dtax}, dsum_price : {dsum_price}")
print(f"dorange_price : {dorange_price}, dapple_price : {dapple_price}")
print(f"dorange : {dorange}, dorange_num : {dorange_num}")
print(f"dapple : {dapple}, dapple_num : {dapple_num}")



relu_layer = Relu()
forward = relu_layer.forward(np.array([-0.1, 0.2, 1.0, -2.0]))
print(forward)
d = 1
backward = relu_layer.backward(d)
print(backward)