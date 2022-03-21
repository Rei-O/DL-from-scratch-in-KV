import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'))  # 親の親ディレクトリのファイルをインポートするための設
sys.path.append(os.path.join(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'), '..'))  # 親の親の親ディレクトリのファイルをインポートするための設定

import numpy as np
from common.presentation.util import im2col, col2im  # image to column, column to image


x = np.array([[[[1,2,3],[4,5,6],[7,8,9]],[[10,11,12],[13,14,15],[16,17,18]]]
            ,[[[101,102,103],[104,105,106],[107,108,109]],[[110,111,112],[113,114,115],[116,117,118]]]])

print(x.shape)  # batch_size:3, channel:2, height:2, width:2
print(x)

# im2col
col_x = im2col(x, 2, 2, stride=1, pad=0)
print(col_x.shape)
print(col_x)


####################### ここから下は動きません　##################

# # col2im
# batch_size = 2
# output_size = 4  # (in:3 + 2*pad:0 - filter:2)/stride:1 + 1 = 2　が高さ×幅
# channel_num = 2
# filter_size = 4

# col_x_trns = col_x.transpose(1, 0).reshape(batch_size, output_size , channel_num, filter_size).transpose(0, 2, 3, 1)  #(batch, channel, output_height, output_width)に変換
# im_x = col2im(col_x_trns, col_x_trns.shape, 2, 2, stride=1, pad=0)
# print(im_x.shape)
# print(im_x) # 2倍とか4倍になっているのはなんでだろうか

# l = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
# print(l[1:6])