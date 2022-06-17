import os
import shutil

import cv2
import numpy as np
from tkinter import *
from PIL import Image
from tensorflow import keras
from tool import locate_and_correct
from trainUnet import unet_predict
from trainCnn import cnn_predict


def modeltest(d):
    # 加载模型
    unet = keras.models.load_model('model\\unet.h5')
    cnn = keras.models.load_model('model\\cnn.h5')
    cnn_predict(cnn, [np.zeros((80, 240, 3))])
    name1 = os.listdir(d)
    re = 0
    # with open("text.txt", "w") as f:
    res = []
    for n in name1:
        # print(n)
        str_image_file = d + n  # 图像
        # print(d + n)
        # img_open = cv2.imread(str_image_file)
        img_open = cv2.imdecode(np.fromfile(str_image_file, dtype=np.uint8), -1)
        h, w = img_open.shape[0], img_open.shape[1]
        if h * w <= 240 * 80 and 2 <= w / h <= 5:  # 满足该条件说明可能整个图片就是一张车牌,无需定位,直接识别即可
            lic = cv2.resize(img_open, dsize=(240, 80), interpolation=cv2.INTER_AREA)[:, :, :3]  # 直接resize为(240,80)
            img_src_copy, Lic_img = img_open, [lic]
        else:  # 否则就需通过unet对img_src原图预测,得到img_mask,实现车牌定位,然后进行识别
            img_src, img_mask = unet_predict(unet, str_image_file)
            img_src_copy, Lic_img = locate_and_correct(img_src, img_mask)  # 利用tool中的locate_and_correct函数进行车牌定位和矫正
        Lic_prediction = cnn_predict(cnn, Lic_img)  # 利用cnn进行车牌的识别预测,Lic_prediction中存的是元祖(车牌图片,识别结果)
        if Lic_prediction:
            img = Image.fromarray(img_src_copy[:, :, ::-1])  # img_src_copy[:, :, ::-1]将BGR转为RGB
            # cv2.imshow('plate position', img_src_copy)

            for i, lic_prediction in enumerate(Lic_prediction):
                if n[:7] == lic_prediction[1]:
                    re += 1
                    # print(n, '识别成功')
                else:
                    res.append((n, lic_prediction[1]))
                    # print(n, '识别失败')
                    # f.write(n[:7] + '->' + lic_prediction[1] + '\n')
                    # shutil.copy(d + n, 'C:\\Users\\desktop\\LicensePlateRecognition\\test1\\')
                # os.rename(d + n, d + lic_prediction[1][0]+n[1:7] + n[-4:])
                # print(lic_prediction)
    cv2.waitKey(0)
    for i in res:
        print(i[0]+'识别失败'+'正确结果：'+str(i[0][:7])+'识别结果：'+i[1])
    print('测试集中正确率为{:.2%}'.format(re / len(name1)))


# def renametool():
#     d = r'C:\\Users\\desktop\\LicensePlateRecognition\\test_data\\'
#     name1 = os.listdir(d)
#     # i = 0
#     with open("text.txt", "w") as f:
#         for n in name1:
#             # os.rename(d + n, d + str(i) + n[1:])
#             f.write(n[:7]+'\n')


if __name__ == '__main__':
    modeltest(r'test_data\\')
    # renametool()
