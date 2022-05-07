import sys
from tkinter import *
from tkinter.filedialog import askopenfilename

import cv2
import numpy as np
from PIL import Image, ImageTk
from tensorflow import keras

from tool import locate_and_correct
from trainCnn import cnn_predict
from trainUnet import unet_predict


class Window:
    def __init__(self, win, ww, wh):
        self.win = win
        self.ww = ww
        self.wh = wh
        self.win.geometry("%dx%d+%d+%d" % (ww, wh, 200, 50))  # 界面启动时的初始位置
        self.win.title("车牌定位、矫正与识别")
        self.img_src_path = None

        Label(self.win, text='原图:', font=('微软雅黑', 16)).place(x=0, y=0)
        # Label(self.win, text='车牌区域:', font=('微软雅黑', 18)).place(x=600, y=125)
        Label(self.win, text='识别结果:', font=('微软雅黑', 18)).place(x=600, y=220)

        self.can_src = Canvas(self.win, width=512, height=512, bg='white', relief='solid', borderwidth=1)  # 原图画布
        self.can_src.place(x=50, y=0)
        # self.can_lic1 = Canvas(self.win, width=245, height=85, bg='white', relief='solid', borderwidth=1)  # 车牌区域1画布
        # self.can_lic1.place(x=710, y=100)
        self.can_pred1 = Canvas(self.win, width=245, height=65, bg='white', relief='solid', borderwidth=1)  # 车牌识别1画布
        self.can_pred1.place(x=710, y=200)

        self.button1 = Button(self.win, text='选择文件', width=15, height=2, command=self.load_show_img)  # 选择文件按钮
        self.button1.place(x=580, y=wh - 140)
        self.button2 = Button(self.win, text='识别车牌', width=15, height=2, command=self.display)  # 识别车牌按钮
        self.button2.place(x=730, y=wh - 140)
        self.button3 = Button(self.win, text='清空', width=15, height=2, command=self.clear)  # 清空所有按钮
        self.button3.place(x=880, y=wh - 140)
        self.unet = keras.models.load_model('model\\unet.h5')
        self.cnn = keras.models.load_model('model\\cnn.h5')
        print('正在启动中,请稍等...')
        cnn_predict(self.cnn, [np.zeros((80, 240, 3))])
        print("已启动,开始识别吧！")

    def load_show_img(self):
        self.clear()
        sv = StringVar()
        sv.set(askopenfilename())
        # print(type(sv))
        self.img_src_path = Entry(self.win, state='readonly', textvariable=sv).get()  # 获取到所打开的图片
        # print(self.img_src_path)
        img_open = Image.open(self.img_src_path)
        if img_open.size[0] * img_open.size[1] > 240 * 80:
            img_open = img_open.resize((512, 512), Image.ANTIALIAS)
        self.img_Tk = ImageTk.PhotoImage(img_open)
        self.can_src.create_image(258, 258, image=self.img_Tk, anchor='center')

    def display(self):
        if self.img_src_path is None:  # 还没选择图片就进行预测
            self.can_pred1.create_text(32, 15, text='请选择图片', anchor='nw', font=('黑体', 28))
        else:
            img_src = cv2.imdecode(np.fromfile(self.img_src_path, dtype=np.uint8), -1)  # 从中文路径读取时用
            h, w = img_src.shape[0], img_src.shape[1]
            if h * w <= 240 * 80 and 2 <= w / h <= 5:  # 满足该条件说明可能整个图片就是一张车牌,无需定位,直接识别即可
                lic = cv2.resize(img_src, dsize=(240, 80), interpolation=cv2.INTER_AREA)[:, :, :3]  # 直接resize为(240,80)
                img_src_copy, Lic_img = img_src, [lic]
            else:  # 否则就需通过unet对img_src原图预测,得到img_mask,实现车牌定位,然后进行识别
                img_src, img_mask = unet_predict(self.unet, self.img_src_path)
                img_src_copy, Lic_img = locate_and_correct(img_src,
                                                           img_mask)  # 利用core.py中的locate_and_correct函数进行车牌定位和矫正

            Lic_pred = cnn_predict(self.cnn, Lic_img)  # 利用cnn进行车牌的识别预测,Lic_pred中存的是元祖(车牌图片,识别结果)
            if Lic_pred:
                img = Image.fromarray(img_src_copy[:, :, ::-1])  # img_src_copy[:, :, ::-1]将BGR转为RGB
                self.img_Tk = ImageTk.PhotoImage(img)
                self.can_src.delete('all')  # 显示前,先清空画板
                self.can_src.create_image(258, 258, image=self.img_Tk,
                                          anchor='center')  # img_src_copy上绘制出了定位的车牌轮廓,将其显示在画板上
                self.lic_Tk1 = ImageTk.PhotoImage(Image.fromarray(Lic_pred[0][0][:, :, ::-1]))
                # self.can_lic1.create_image(5, 5, image=self.lic_Tk1, anchor='nw')
                self.can_pred1.create_text(35, 15, text=Lic_pred[0][1], anchor='nw', font=('黑体', 28))

            else:  # Lic_pred为空说明未能识别
                self.can_pred1.create_text(47, 15, text='未能识别', anchor='nw', font=('黑体', 27))

    def clear(self):
        self.can_src.delete('all')
        # self.can_lic1.delete('all')
        self.can_pred1.delete('all')
        self.img_src_path = None

    @staticmethod
    def closeEvent():  # 关闭前清除session(),防止'NoneType' object is not callable
        keras.backend.clear_session()
        sys.exit()


if __name__ == '__main__':
    win = Tk()
    ww = 1000  # 窗口宽设定1000
    wh = 600  # 窗口高设定600
    Window(win, ww, wh)
    win.protocol()
    win.mainloop()
