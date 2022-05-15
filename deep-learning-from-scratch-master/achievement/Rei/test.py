from reiras.models import NuralNet
from reiras.layers import *
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'))  # 親ディレクトリの親ディレクトリのファイルをインポートするための設定
from dataset.presentation.mnist import load_mnist

input_size_list = (1, 2, 5, 5)  # 画像データを想定
# input_size_list = (5, 5)  # テーブルデータを想定
input_size = 1
for input in input_size_list:
    input_size *= input
x_train = np.arange(input_size).reshape(input_size_list)
print('x_train' + str(x_train.shape))
t_train = np.arange(input_size_list[0])


test_size_list = (8, 2, 5, 5)  # 画像データを想定
# test_size_list = (5, 5)  # テーブルデータを想定
test_size = 1
for test in test_size_list:
    test_size *= test
x_test = np.arange(test_size).reshape(test_size_list)
t_test = np.arange(test_size_list[0])


# model = NuralNet()
# model.add(Input(output_units=input_size_list[1:]))
# model.add(Flatten())
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy', output_units=10, batch_size=1)
# model.fit(x_train=x_train, x_test=x_test, t_train=t_train, t_test=t_test, epochs=2)
# y = model.predict(x_test)
# print(y.shape)


model = NuralNet()
model.add(Input(x_train.shape[1:]))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(output_units=16, activation='relu'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy', output_units=10, batch_size=4)
model.fit(x_train=x_train, x_test=x_test, t_train=t_train, t_test=t_test, epochs=5)
y = model.predict(x_test)
print(y)
print(model.train_loss_list[-1])
print(model.train_evalute_list[-1])
print(model.test_evalute_list[-1])


