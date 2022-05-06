from trainUnet import train_unet
from trainCnn import train_cnn

# train_unet()  # 训练后得到unet.h5,用于车牌定位
train_cnn()  # 训练后得到cnn.h5，用于车牌识别
