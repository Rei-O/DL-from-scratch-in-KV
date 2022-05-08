import numpy as np
from numpy import arange
from reiras.models import NuralNet
from reiras.layers import *
from collections import OrderedDict
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'))  # 親ディレクトリの親ディレクトリのファイルをインポートするための設定
from dataset.presentation.mnist import load_mnist


# # input_size_list = (1, 2, 5, 5)  # 画像データを想定
# input_size_list = (5, 5)  # テーブルデータを想定
# input_size = 1
# for input in input_size_list:
#     input_size *= input
# x_train = np.arange(input_size).reshape(input_size_list)
# print('x_train' + str(x_train.shape))
# t_train = np.arange(input_size_list[0])


# # test_size_list = (1, 2, 5, 5)  # 画像データを想定
# test_size_list = (5, 5)  # テーブルデータを想定
# test_size = 1
# for test in test_size_list:
#     test_size *= test
# x_test = np.arange(test_size).reshape(test_size_list)
# t_test = np.arange(test_size_list[0])


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)

# print(x)

# Flatten layer のテスト
# flat = Flatten()
# y = flat.forward(x)
# print(y)
# dx = flat.backward(y)
# print(dx)

# model = NuralNet()
# model.add(Flatten())
# print(model.layersOrderDict.items())


# x_shape = (1,2,3)
# batch_size = 4
# joined_x_shape = ()
# if batch_size != None: 
#     joined_x_shape = (batch_size,) + x_shape
# else:
#     joined_x_shape = x_shape

# print(joined_x_shape)

# units = (1,)
# if type(units) == tuple : 
#     list = (batch_size,) + x_shape
# else:
#     joined_x_shape = x_shape

# print(joined_x_shape)

# print(type(units) == tuple)
# print(type(t_train))
# print(type(t_test))

model = NuralNet()
model.add(Input(x_train.shape[1]))
model.add(Dense(output_units=10, activation='relu'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy', output_units=10, batch_size=64)
model.fit(x_train=x_train, t_train=t_train, x_test=x_test, t_test=t_test, epochs=100)
y = model.predict(x_test)
print(y)
print(model.train_loss_list[-1])
print(model.train_evalute_list[-1])
print(model.test_evalute_list[-1])

# print(model.layer_size_list)
# print(model.params.keys())
# print(model.layersOrderDict)
# print(len(model.layersOrderDict))
# model.add(Flatten())
# print(model.layersOrderDict.items())

# layerDict = OrderedDict()

# layerDict['key1'] = 'value1'
# layerDict['key2'] = 'value2'

# for key, value in layerDict.items().__reversed__():
#     print('1' in key)

# for i in np.flipud(arange(len(model.layersOrderDict))):
#     print(i)