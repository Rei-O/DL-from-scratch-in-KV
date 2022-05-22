from reiras.models import NuralNet
from reiras.layers import *
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'))  # 親ディレクトリの親ディレクトリのファイルをインポートするための設定
from dataset.presentation.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=False, one_hot_label=True)

print("input : " + str(x_train.shape[1:]))

model = NuralNet()
model.add(Input(x_train.shape[1:]))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(filter_size=(2,2),stride=2))
model.add(Flatten())
model.add(Dense(output_units=10, activation='relu'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy', output_units=10, batch_size=128)
# TODO 現状出力層とその前層のアウトプットunitsが合ってないとエラーになる（denseとかしてないのでcompileのoutput_unitsが意味ない）
print("model.input_layer_size_list : " + str(model.input_layer_size_list))
print("model.output_layer_size_list : " + str(model.output_layer_size_list))
model.fit(x_train=x_train, x_test=x_test, t_train=t_train, t_test=t_test, epochs=100)
y = model.predict(x_test)
print(y)
print(model.train_loss_list[-1])
print(model.train_evalute_list[-1])
print(model.test_evalute_list[-1])


