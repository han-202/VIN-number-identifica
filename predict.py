import cv2
import numpy as np
from numpy.linalg import norm
from itertools import groupby
# from skimage import data,color,morphology
import sys
import os
import json
from svmutil import *
import math

SZ = 20  # 训练图片长宽
MAX_WIDTH = 1000  # 原始图片最大宽度
Min_Area = 1000  # 车牌区域允许最大面积
PROVINCE_START = 1000


# 读取图片文件
def imreadex(filename):
    return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)


def point_limit(point):
    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0


def separate_color_red(img):
    # 颜色提取
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
    #lower_hsv = np.array([0, 123, 100])  # 提取颜色的低值
    lower_hsv = np.array([0, 43, 46])  # 提取颜色的低值
    high_hsv = np.array([10, 255, 255])  # 提取颜色的高值
    mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=high_hsv)

    # print("颜色提取完成")
    return mask

#图像预处理
def max_min_value_filter(image, ksize=3):
    """
    :param image: 原始图像
    :param ksize: 卷积核大小
    :param mode:  最大值：1 或最小值：2
    :return:
    """
    img = image.copy()
    rows, cols = img.shape
    padding = (ksize - 1) // 2
    new_img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=255)
    for i in range(rows):
        for j in range(cols):
            roi_img = new_img[i:i + ksize, j:j + ksize].copy()
            min_val, max_val, min_index, max_index = cv2.minMaxLoc(roi_img)
            img[i, j] = min_val
    return img

def Sum_part_card(matrix):
    sm = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            sm += matrix[i][j]
    return sm


# 字符矫正
def rotate(image, angle, center=None, scale=1.0):
    (w, h) = image.shape[0:2]
    if center is None:
        center = (w // 2, h // 2)
    wrapMat = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, wrapMat, (h, w))


# 绘制直方图
def caleGrayHist(image):
    # 灰度图像的高、宽
    rows, cols = image.shape
    # 存储灰度直方图
    grayHist = np.zeros([256], np.uint64)  # 图像的灰度级范围是0~255
    for r in range(rows):
        for c in range(cols):
            grayHist[image[r][c]] += 1
    return grayHist


# 根据设定的阈值和图片直方图，找出波峰，用于分隔字符
def find_waves(threshold, histogram):
    up_point = -1  # 上升点
    is_peak = False
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    wave_peaks = []
    for i, x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    return wave_peaks


# 根据找出的波峰，分隔图片，从而得到逐个字符图片
def seperate_card(img, waves):
	part_cards = []
	for wave in waves:
		part_cards.append(img[:, wave[0]:wave[1]])
	return part_cards

# def seperate_card(img, waves):
#     part_cards = []
#     for wave in waves: #waves是由数组组成的列表
#         img_seg = img[:, wave[0]:wave[1]]
#         img_seg_col = img_seg.shape[1]
#         if img_seg_col > 60:  # 35 40,38
#             img_seg1 = img_seg[:,0:img_seg_col//2]
#             img_seg2 = img_seg[:,img_seg_col//2:img_seg_col]
#             part_cards.append(img_seg1)
#             part_cards.append(img_seg2)
#         elif img_seg_col <= 60:  # 35 40,38
#             part_cards.append(img_seg)
#     return part_cards
# 分割粘连字符
def seperate_card_connect(mean,par_car, img_seg_col, part_cards_pre):
    image_num = math.ceil((img_seg_col - 11) / mean)
    if image_num > 1:
        for i in range(image_num):
            wide = int(img_seg_col / image_num)
            img_seg1 = par_car[:, wide * i: wide * (i + 1)]
            part_cards_pre.append(img_seg1)
    else:
        part_cards_pre.append(par_car)
        # if img_seg_col > 2*mean + 10:
    #     img_seg1 = par_car[:, 0:img_seg_col // 3]
    #     img_seg2 = par_car[:,img_seg_col // 3:2*img_seg_col//3]
    #     img_seg3 = par_car[:,2*img_seg_col//3:img_seg_col]
    #     part_cards_pre.append(img_seg1)
    #     part_cards_pre.append(img_seg2)
    #     part_cards_pre.append(img_seg3)
    # else:
    # # elif img_seg_col > mean + 10 and img_seg_col < 2*mean + 10:
    #     img_seg1 = par_car[:,0:img_seg_col//2]
    #     img_seg2 = par_car[:,img_seg_col//2:img_seg_col]
    #     part_cards_pre.append(img_seg1)
    #     part_cards_pre.append(img_seg2)
    #return img_seg1,img_seg2
    return part_cards_pre

# 来自opencv的sample，用于svm训练
def deskew(img):
    m = cv2.moments(img) #图像矩（图像的几何特征），M中包含了很多轮廓的特征信息，以帮助我们计算图像的质心，面积等，返回字典
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


# 来自opencv的sample，用于svm训练
def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n * ang / (2 * np.pi))
        bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]  #降维并统计次数
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)

class StatModel(object):
    def load(self, fn):
        self.model = self.model.load(fn)


    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    def __init__(self, C=1, gamma=0.5):
        self.model = cv2.ml.SVM_create() #创建分类器
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF) #使用高斯核
        self.model.setType(cv2.ml.SVM_C_SVC) #SVM类型

    # 训练svm
    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)  #cv2.ml.ROW_SAMPLE代表每一行是一个样本

    # 字符识别
    def predict(self, samples):
        r = self.model.predict(samples)
        return r[1].ravel()


class CardPredictor:
    def __init__(self):
        # 车牌识别的部分参数保存在js中，便于根据图片分辨率做调整
        f = open('config.js')
        #f = open('C:\国交空间\车牌号相关\License-Plate-Recognition-master\config.js')
        j = json.load(f)
        for c in j["config"]:
            if c["open"]:
                self.cfg = c.copy()
                break
        else:
            raise RuntimeError('没有设置有效配置参数')
        self.model = None

    def load_model(self):
        self.model = SVM(C=5, gamma=0.5)
        self.model.load("svm.dat")
        #self.model.load("C:\国交空间\车牌号相关\License-Plate-Recognition-master\svm.dat")

    def __del__(self):
        self.save_traindata()

    def train_svm(self):
        # 识别英文字母和数字
        self.model = SVM(C=1, gamma=0.5)
        # 识别中文
        self.modelchinese = SVM(C=1, gamma=0.5)
        if os.path.exists("svm.dat"):
            self.model.load("svm.dat")
        else:
            chars_train = []
            chars_label = []
            # root 所指的是当前正在遍历的这个文件夹的本身的地址；dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
            #files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
            for root, dirs, files in os.walk("train\\chars2"): # "train\\chars2"
                if len(os.path.basename(root)) > 1:
                    continue
                root_int = ord(os.path.basename(root))
                for filename in files:
                    filepath = os.path.join(root, filename)
                    digit_img = cv2.imread(filepath)
                    digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                    chars_train.append(digit_img)
                    # chars_label.append(1)
                    chars_label.append(root_int)

            chars_train = list(map(deskew, chars_train))
            chars_train = preprocess_hog(chars_train)
            # chars_train = chars_train.reshape(-1, 20, 20).astype(np.float32)
            chars_label = np.array(chars_label)

            print('chars_train:',chars_train)
            print('chars_label:', chars_label)

            print(chars_train.shape)
            self.model.train(chars_train, chars_label)

    # if os.path.exists("svmchinese.dat"):
    # 	self.modelchinese.load("svmchinese.dat")
    # else:
    # 	chars_train = []
    # 	chars_label = []
    # 	for root, dirs, files in os.walk("train\\charsChinese"):
    # 		if not os.path.basename(root).startswith("zh_"):
    # 			continue
    # 		pinyin = os.path.basename(root)
    # 		index = provinces.index(pinyin) + PROVINCE_START + 1 #1是拼音对应的汉字
    # 		for filename in files:
    # 			filepath = os.path.join(root,filename)
    # 			digit_img = cv2.imread(filepath)
    # 			digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
    # 			chars_train.append(digit_img)
    # 			#chars_label.append(1)python
    # 			chars_label.append(index)
    # 	chars_train = list(map(deskew, chars_train))
    # 	chars_train = preprocess_hog(chars_train)
    # 	#chars_train = chars_train.reshape(-1, 20, 20).astype(np.float32)
    # 	chars_label = np.array(chars_label)
    # 	print(chars_train.shape)
    # 	self.modelchinese.train(chars_train, chars_label)

    def save_traindata(self):
        if not os.path.exists("svm.dat"):
            self.model.save("svm.dat")

    # if not os.path.exists("svmchinese.dat"):
    # 	self.modelchinese.save("svmchinese.dat")

    def accurate_place(self, card_img_hsv, limit1, limit2, color):
        row_num, col_num = card_img_hsv.shape[:2]
        xl = col_num
        xr = 0
        yh = 0
        yl = row_num
        # col_num_limit = self.cfg["col_num_limit"]
        row_num_limit = self.cfg["row_num_limit"]
        col_num_limit = col_num * 0.8 if color != "green" else col_num * 0.5  # 绿色有渐变
        for i in range(row_num):
            count = 0
            for j in range(col_num):
                H = card_img_hsv.item(i, j, 0)
                S = card_img_hsv.item(i, j, 1)
                V = card_img_hsv.item(i, j, 2)
                if limit1 < H <= limit2 and 34 < S and 46 < V:
                    count += 1
            if count > col_num_limit:
                if yl > i:
                    yl = i
                if yh < i:
                    yh = i
        for j in range(col_num):
            count = 0
            for i in range(row_num):
                H = card_img_hsv.item(i, j, 0)
                S = card_img_hsv.item(i, j, 1)
                V = card_img_hsv.item(i, j, 2)
                if limit1 < H <= limit2 and 34 < S and 46 < V:
                    count += 1
            if count > row_num - row_num_limit:
                if xl > j:
                    xl = j
                if xr < j:
                    xr = j
        return xl, xr, yh, yl

    def predict(self, car_pic):
        if type(car_pic) == type(""):
            img = imreadex(car_pic)
        else:
            img = car_pic
            #img = cv2.resize(img, (588, 237), interpolation=cv2.INTER_AREA) #待用
        pic_hight, pic_width = img.shape[:2]

        # if pic_width > MAX_WIDTH:
        # 	resize_rate = MAX_WIDTH / pic_width
        # 	img = cv2.resize(img, (MAX_WIDTH, int(pic_hight*resize_rate)), interpolation=cv2.INTER_AREA)
        #
        # blur = self.cfg["blur"]
        # #高斯去噪
        # if blur > 0:
        # 	img = cv2.GaussianBlur(img, (blur, blur), 0)#图片分辨率调整
        # oldimg = img
        # img = imreadex(car_pic)

        # cv2.imshow('img_original', img)
        # cv2.waitKey()
        oldimg = img
        """提取ROI区域"""

        # img = separate_color_red(img)  # 提取红色（roi）区域
        # cv2.imshow('HSV', img)
        # cv2.waitKey(0)  # weitKey: 等待键盘输入
        """提取ROI区域完"""

        # 准确定位,从原图中截取红色方框区域
        # opencv 3.X返回三个参数
        # opencv 4.X返回两个参数

        """==========在原图中提取ROI区域"""

        # _, cnts, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找矩形轮廓
        # contours = [cnt for cnt in cnts if cv2.contourArea(cnt) > 100]
        # car_contours = []
        # Marea = []
        # M = 0
        # for cnt in contours:
        #     area = cv2.contourArea(cnt)
        #     Marea.append(area)
        #
        # M = Marea.index(max(Marea))
        # rect = cv2.minAreaRect(contours[M])
        # car_contours.append(rect)
        # # print("len(car_contours):", len(car_contours))
        # card_imgs = []
        # for rect in car_contours:
        #     box = cv2.boxPoints(rect)  # 画矩形要有四个角，这四个角用 此函数 得到
        #     # print(box[1][1])
        #     # card_img = img[204:249,130:439]
        #     card_img = oldimg[int(box[1][1])+15:int(box[3][1])-16, int(box[1][0])+23:int(box[3][0])-15]
        #     #card_img = oldimg[int(box[1][1]):int(box[3][1]), int(box[1][0]):int(box[3][0])]
        #     card_imgs.append(card_img)
        #     #cv2.imwrite('./ROI_images/' +str(111)+ '.jpg',card_img)
        #     cv2.imshow('card_img', card_img)
        #     cv2.waitKey()
        # #img1 = cv2.bilateralFilter(card_img,10,100,100)
        # img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)

        """============在原图中提取ROI区域完"""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #临时用
        cv2.imshow("img",img)
        cv2.waitKey()


        #-------ROI图像预处理-------
        x = img.shape[0]
        k_size = math.ceil(x / 5)  # 参数可以选：或者除以10
        img_min = max_min_value_filter(image=img, ksize=k_size)
        cv2.imshow("img_min",img_min)
        cv2.waitKey()
        img_mean = cv2.blur(img_min, (k_size, k_size))
        cv2.imshow("img_mean",img_mean)
        cv2.waitKey()
        img_new = img - img_mean
        cv2.imshow("img_new",img_new)
        cv2.waitKey()
        ret, gray_img = cv2.threshold(img_new, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  #ret：True或False，代表有没有读到图片
        ret, gray_img = cv2.threshold(img_new, math.floor(ret * 0.85), 255, cv2.THRESH_BINARY)
        cv2.imshow("gray_img",gray_img)
        cv2.waitKey()

        # equ = cv2.equalizeHist(img)
        # img = np.hstack((img, equ))
        # ------------------------去掉图像中不会是车牌的区域----------------------------
        # kernel = np.ones((20, 20), np.uint8)
        # img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        # img_opening = cv2.addWeighted(img, 1, img_opening, -1, 0);
        # -------------------------------去掉图像中不会是车牌的区域完成----------------------------

        # ------------------------------------找到图像边缘--------------------------------------
        # ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # img_edge = cv2.Canny(img_thresh, 100, 200)
        # 使用开运算和闭运算让图像边缘成为一个整体
        # kernel = np.ones((self.cfg["morphologyr"], self.cfg["morphologyc"]), np.uint8)
        # img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
        # img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)
        # -----------------------------------找到图像边缘完成--------------------------------

        # 查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
        # try:
        # 	contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # except ValueError:
        # 	image, contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #		contours = [cnt for cnt in contours if cv2.contourArea(cnt) > Min_Area]
        #		print('len(contours)', len(contours))
        # 一一排除不是车牌的矩形区域
        # car_contours = []
        # for cnt in contours:
        # 	rect = cv2.minAreaRect(cnt)
        # 	area_width, area_height = rect[1]
        # 	if area_width < area_height:
        # 		area_width, area_height = area_height, area_width
        # 	wh_ratio = area_width / area_height
        # 	#print(wh_ratio)
        # 	#要求矩形区域长宽比在2到5.5之间，2到5.5是车牌的长宽比，其余的矩形排除
        # 	if wh_ratio > 2 and wh_ratio < 5.5:
        # 		car_contours.append(rect)
        # 		box = cv2.boxPoints(rect)
        # 		box = np.int0(box)
        # oldimg = cv2.drawContours(oldimg, [box], 0, (0, 0, 255), 2)
        # cv2.imshow("edge4", oldimg)
        # print(rect)

        #		print(len(car_contours))
        # --------------------------------提取红色区域-------------------------------------
        # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
        # lower_hsv = np.array([0, 43, 46])  # 提取颜色的低值
        # high_hsv = np.array([10, 255, 255])  # 提取颜色的高值
        # mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=high_hsv)
        # kernel = np.ones((20, 20), np.uint8)
        # img_opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # img_opening = cv2.addWeighted(mask, 1, img_opening, -1, 0)
        # ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # img_edge = cv2.Canny(img_thresh, 100, 200)
        # cv2.imshow('original', img_edge)
        # cv2.waitKey()
        # ------------------------------提取红色区域完成---------------------------------------------

        #		print("精确定位")
        #
        # 矩形区域可能是倾斜的矩形，需要矫正，以便使用颜色定位
        # for rect in car_contours:
        # 	if rect[2] > -1 and rect[2] < 1:#创造角度，使得左、高、右、低拿到正确的值
        # 		angle = 1
        # 	else:
        # 		angle = rect[2]
        # 	rect = (rect[0], (rect[1][0]+5, rect[1][1]+5), angle)#扩大范围，避免车牌边缘被排除
        #
        # 	box = cv2.boxPoints(rect)
        # 	heigth_point = right_point = [0, 0]
        # 	left_point = low_point = [pic_width, pic_hight]
        # 	for point in box:
        # 		if left_point[0] > point[0]:
        # 			left_point = point
        # 		if low_point[1] > point[1]:
        # 			low_point = point
        # 		if heigth_point[1] < point[1]:
        # 			heigth_point = point
        # 		if right_point[0] < point[0]:
        # 			right_point = point
        #
        # 	if left_point[1] <= right_point[1]:#正角度
        # 		new_right_point = [right_point[0], heigth_point[1]]
        # 		pts2 = np.float32([left_point, heigth_point, new_right_point])#字符只是高度需要改变
        # 		pts1 = np.float32([left_point, heigth_point, right_point])
        # 		M = cv2.getAffineTransform(pts1, pts2)
        # 		dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
        # 		point_limit(new_right_point)
        # 		point_limit(heigth_point)
        # 		point_limit(left_point)
        # 		card_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]
        # 		card_imgs.append(card_img)
        # 		#cv2.imshow("card", card_img)
        # 		#cv2.waitKey(0)
        # 	elif left_point[1] > right_point[1]:#负角度
        #
        # 		new_left_point = [left_point[0], heigth_point[1]]
        # 		pts2 = np.float32([new_left_point, heigth_point, right_point])#字符只是高度需要改变
        # 		pts1 = np.float32([left_point, heigth_point, right_point])
        # 		M = cv2.getAffineTransform(pts1, pts2)
        # 		dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
        # 		point_limit(right_point)
        # 		point_limit(heigth_point)
        # 		point_limit(new_left_point)
        # 		card_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]
        # 		card_imgs.append(card_img)
        # 		cv2.imshow("card", card_img)
        # 		cv2.waitKey(0)
        # 开始使用颜色定位，排除不是车牌的矩形，目前只识别蓝、绿、黄车牌
        # 颜色提取

        # 以上为车牌定位
        # 以下为识别车牌中的字符
        predict_result = []
        roi = None
        card_color = None
        # for i, color in enumerate(colors):
        # 	print('i:',i)
        # 	print('color',color)
        # if color in ("blue", "yello", "green"):
        # 	card_img = card_imgs[i]
        # 	gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
        # 黄、绿车牌字符比背景暗、与蓝车牌刚好相反，所以黄、绿车牌需要反向
        # if color == "green" or color == "yello":
        # 	gray_img = cv2.bitwise_not(gray_img)
        # ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 查找水平直方图波峰
        # ---------------------图像阈值、膨胀处理(自己)---------------
        # #ret, gray_img = cv2.threshold(img, 179, 255, cv2.THRESH_BINARY)
        # #ret, gray_img = cv2.threshold(img, 179, 255, cv2.THRESH_BINARY)
        # ret, gray_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # cv2.imshow("gray_img",gray_img)
        # cv2.waitKey()
        #
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))  # (2,3)
        # gray_img = cv2.erode(gray_img,kernel)
        # # ----gray_img = cv2.remove_small_objects(img, min_size=100,connectivity=1)
        # # cv2.imshow("erode",gray_img)
        # # cv2.waitKey()
        #---------------图像阈值、膨胀处理(自己)完------------------

        #-----------------------优化二值图--------------
        #gray_img, contours1, hierarchy1 = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 找矩形轮廓,cv2.CHAIN_APPROX_SIMPLE
        gray_img, contours1, hierarchy1 = cv2.findContours(gray_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        n = len(contours1)
        Area = 0
        Area_A = []
        for ct in range(n):
            tmp = np.zeros(gray_img.shape,np.uint8)
            #contoursimg.append(tmp)
            area = cv2.contourArea(np.array(contours1[ct]))
            Area_A.append(area)
            #print("area:",Area_A)
            Area += area
            Area_mean = Area / n
            # if area < Area_mean-: #25
            #     cv2.drawContours(gray_img, [contours1[ct]], 0,0, -1)

            if area < 0 : #25
                cv2.drawContours(gray_img, [contours1[ct]], 0,0, -1)
        # -----------------------优化二值图完--------------

        """空洞填充"""
        thresh_not = cv2.bitwise_not(gray_img)  # 二值图像的补集
        kernel_0 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 3×3结构元
        F = np.zeros(gray_img.shape, np.uint8)
        F[:, 0] = thresh_not[:, 0]
        F[:, -1] = thresh_not[:, -1]
        F[0, :] = thresh_not[0, :]
        F[-1, :] = thresh_not[-1, :]
        for i in range(200):
            F_dilation = cv2.dilate(F, kernel_0, iterations=1)
            F = cv2.bitwise_and(F_dilation, thresh_not)
        result = cv2.bitwise_not(F)  # 对结果执行not
        # cv2.imshow('p', result)
        # cv2.waitKey(0)
        #
        # _, contours, hierarchy = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # length = []
        # Area = []
        # for c in contours:
        #     L = cv2.arcLength(c, True)
        #     length.append(L)
        #     (x, y), radius = cv2.minEnclosingCircle(c)
        #     radius = int(radius)
        #     S = math.pow(radius,2) * math.pi
        #     Area.append(S)
        #     # print('Area = % d'% S)
        # remove_L = sum(length)/len(length)
        # remove_A =  sum(Area)/len(Area)
        # for c in contours:
        #     # 找到边界坐标
        #     x, y, w, h = cv2.boundingRect(c)  # 计算点集最外面的矩形边界
        #     cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #
        #     # 找面积最小的矩形
        #     rect = cv2.minAreaRect(c)
        #     a = rect[1][0]
        #     b = rect[1][1]
        #     area1 = a*b
        #     L_c = cv2.arcLength(c,True)
        #     (x1, y1), r = cv2.minEnclosingCircle(c)
        #     r = int(r)
        #     S_c = math.pow(r, 2) * math.pi
        #     # print('L=%d'%L)
        #     # # 得到最小矩形的坐标
        #     # box = cv2.boxPoints(rect)
        #     # # 标准化坐标到整数
        #     # box = np.int0(box)
        #     #if L_c < remove_L: #25
        #     if S_c < remove_A :
        #         cv2.drawContours(result, [c], 0,0, -1)
        #
        # gray_img = cv2.bitwise_and(result, gray_img)
        # cv2.imshow('y', gray_img)
        # cv2.waitKey(0)

        _, contours1, hierarchy1 = cv2.findContours(result, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        n = len(contours1)
        Area = 0
        Area_A = []
        for ct in range(n):
            tmp = np.zeros(gray_img.shape,np.uint8)
            #contoursimg.append(tmp)
            area = cv2.contourArea(np.array(contours1[ct]))
            Area_A.append(area)
            Area_A.sort(reverse=True)
            #print("area:",Area_A,len(Area_A))
            Area += area
            Area_mean = Area / n
        add_A = [0,0,0]
        Area_A = Area_A + add_A
        Re_thresh = Area_A[18]
        for ct in range(n):
            area = cv2.contourArea(np.array(contours1[ct]))
            if area < Re_thresh: #25
                cv2.drawContours(result, [contours1[ct]], 0,0, -1)
        gray_img = cv2.bitwise_and(result, gray_img)
        """空洞填充完"""

        # cv2.imshow('T_img',gray_img)
        # cv2.waitKey()
        #cv2.imwrite('./test_photo/'+ '.jpg', gray_img)

        #---------优化二值图完-------------------
        # gray_img = cv2.dilate(gray_img, kernel) #膨胀处理（与图像阈值、膨胀处理同用）
        # cv2.imshow('erode:', gray_img)
        # cv2.waitKey()
        # gray_img = cv2.dilate(gray_img, (2,3))
        # cv2.imshow('dilate:', gray_img)
        # cv2.waitKey()

        # ekernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        # gray_img = cv2.erode(gray_img,ekernel)

        # ---------------------无缩放旋转---------------
        # shape = gray_img.shape
        # h = shape[0]
        # w = shape[1]
        # gray_img_h = shape[0]
        # gray_img_w = shape[1]
        # matRotate = cv2.getRotationMatrix2D((gray_img_h, gray_img_w), -1, 1) #获得仿射变化矩阵
        # cos = np.abs(matRotate[0, 0])
        # sin = np.abs(matRotate[0, 1])
        # nw = int((gray_img_h * sin) + (gray_img_w * cos))
        # nh = int((gray_img_h * cos) + (gray_img_w * sin))
        # matRotate[0, 2] += (nw / 2) - gray_img_w // 2
        # matRotate[1, 2] += (nh / 2) - gray_img_h // 2
        # gray_img = cv2.warpAffine(gray_img, matRotate, (nw, nh)) #进行放射变换
        # cv2.imshow('rotate', gray_img)
        # cv2.waitKey()

        # ----------------------无缩放旋转完成-------------------

        # -------------剪裁图像-------------
        # cop = gray_img.shape
        # cop_row = cop[0]
        # cop_col = cop[1]
        #
        # a = int(cop_row / 2)  # x start
        # b = int(cop_row / 2 - 100)  # x end
        # # c = int(cop_col / 2 - 10)  # y start
        # # d = int(cop_col / 2 + 10)  # y end
        # print(a, b)
        # gray_img = gray_img[a:h, 0:w]  # 裁剪图像
        # print('cropImg.shape:', gray_img.shape)
        # cv2.imshow('cropimage', gray_img)
        # cv2.waitKey()
        # ------------剪裁图像完成-----------

        # -----------------水平直方图---------------------
        # cv2.imshow('gray_img',gray_img)
        # cv2.waitKey()
        x_histogram = np.sum(gray_img, axis=1) # x_histogram是一个列表；axis=1每一行元素相加之和。
        # print('x_histogram:',len(x_histogram))
        x_min = np.min(x_histogram)
        x_average = np.sum(x_histogram) / x_histogram.shape[0]
        x_threshold = (x_min + x_average) / 2
        wave_peaks = find_waves(x_threshold, x_histogram)
        # print('x_threshold:',x_threshold)
        # print('x_histogram:',x_histogram)
        # print('wave_peaks1:', wave_peaks)
        # -------------阈值、膨胀处理完-------------

        # '----------------阈值与直方图找波峰---改----------'
        # ret, gray_img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
        # histogram = caleGrayHist(gray_img )
        # wave_peaks = find_waves(x_threshold, x_histogram)

        # 		if len(wave_peaks) == 0:
        # #					print("peak less 0:")
        # 			continue
        # 认为水平方向，最大的波峰为车牌区域
        wave = max(wave_peaks, key=lambda x: x[1] - x[0])
        # -------------------------查找垂直直方图波峰----------------------------
        row_num, col_num = gray_img.shape[:2]
        #		去掉车牌上下边缘1个像素，避免白边影响阈值判断
        gray_img = gray_img[1:row_num - 1]
        y_histogram = np.sum(gray_img, axis=0)
        # print('y_histogram:',y_histogram)
        y_min = np.min(y_histogram)
        y_average = np.sum(y_histogram) / y_histogram.shape[0]
        # print('y_histogram.shape',y_histogram.shape)
        # y_threshold = (y_min + y_average)/5#U和0要求阈值偏小，否则U和0会被分成两半
        y_threshold = (y_min + y_average) / 5 #3,2.1
        # print('y_threshold:',y_threshold)
        wave_peaks = find_waves(y_threshold, y_histogram)
        # print('wave_peaks2:', wave_peaks)
        # -----------------------------查找垂直直方图波峰完成-------------------------------------
        # for wave in wave_peaks:
        #	cv2.line(card_img, pt1=(wave[0], 5), pt2=(wave[1], 5), color=(0, 0, 255), thickness=2)
        # 车牌字符数应大于6
        # 		if len(wave_peaks) <= 6:
        # #					print("peak less 1:", len(wave_peaks))
        # 			conti

        wave = max(wave_peaks, key=lambda x: x[1] - x[0])
        max_wave_dis = wave[1] - wave[0]
        # 判断是否是左侧车牌边缘
        # if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis/3 and wave_peaks[0][0] == 0:
        # 	wave_peaks.pop(0)

        # 组合分离汉字
        # cur_dis = 0
        # for i,wave in enumerate(wave_peaks):
        # 	if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
        # 		break
        # 	else:
        # 		cur_dis += wave[1] - wave[0]
        # if i > 0:
        # 	wave = (wave_peaks[0][0], wave_peaks[i][1])
        # 	wave_peaks = wave_peaks[i+1:]
        # 	wave_peaks.insert(0, wave)

        # 去除车牌上的分隔点
        # point = wave_peaks[2]
        # if point[1] - point[0] < max_wave_dis/3:
        # 	point_img = gray_img[:,point[0]:point[1]]
        # 	if np.mean(point_img) < 255/5:
        # 		wave_peaks.pop(2)

        # 				if len(wave_peaks) <= 6:
        # #					print("peak less 2:", len(wave_peaks))
        # 					continue
        # 分割图像
        part_cards = seperate_card(gray_img, wave_peaks)
        # ---------------显示分割图像与剪裁图像-----------
        # print(len(part_cards))
        part_cards_crop = []
        for pn in range(len(part_cards)):
            #print(part_cards[pn])
            # cv2.imshow('pn',part_cards[pn])
            # cv2.waitKey()

            cop = part_cards[pn].shape
            cop_row = cop[0]
            # print('cop is:', cop)
            cop_col = cop[1]
            #----=======---改进------------------------
            cop_sum = np.sum(part_cards[pn], axis=1)
            #print('cop_sum', cop_sum)

            pk_list = []
            for ck_i, ck in enumerate(cop_sum):
                #print(ck_i,ck)
                if ck != 0:
                    #print(ck_i)
                    pk_list.append(ck_i)
            #print("pk_list",pk_list)
            index_with_fault = []
            fun = lambda x: x[1] - x[0]
            for k, g in groupby(enumerate(pk_list), fun):
                l1 = [j for i, j in g]
                if len(l1) > 10: #20,10
                    index_with_fault = index_with_fault + l1
                    #print("index_with_fault:",index_with_fault)
                    lis_end = len(index_with_fault)-1
                    a = index_with_fault[0]
                    b = index_with_fault[-1]
                    #print('a,b:',b)
                    #a = int(cop_row)  # x start原来是30,改后是17
                    #b = int(cop_row / 2 + 25)  # x end原来是32，改后是16
                    c = int(cop_col / 2 - 200)  # y start原来是200
                    d = int(cop_col / 2 + 200)  # y end 原来是200
                    cropImg = part_cards[pn][a:b+1, :]  # 裁剪图像
                    part_cards_crop.append(cropImg)
                    # cv2.imshow('cropimage', cropImg)
                    # cv2.waitKey()
                   # ==============改进完成------------------
            # a = int(cop_row / 2 - 20)  # x start原来是30,改后是17
            # b = int(cop_row / 2 + 25)  # x end原来是32，改后是16
            # c = int(cop_col / 2 - 200)  # y start原来是200
            # d = int(cop_col / 2 + 200)  # y end 原来是200
            # cropImg = part_cards[pn][a:b, c:d]  # 裁剪图像
            # part_cards_crop.append(cropImg)
            # cv2.imshow('cropimage', cropImg)
            # cv2.waitKey()
        # print('part_cards_crop :', len(part_cards_crop))
        # ------------------显示分割图像与剪裁图像完成-----------

        # ----------------剔除噪点--------------
        part_cards_crop1 = []
        if len(part_cards_crop) <=17:
            part_cards_crop1 = part_cards_crop
        else:
            for i in part_cards_crop:  # part_cards_crop；part_cards
                s1 = 0
                s1 = np.sum(i, axis=1)
                #print('s1:',s1)
                # print('sa:',s1)
                s1 = [k for k in s1 if k > 0]
                if len(s1) > 15: #原来是30,10,15
                    part_cards_crop1.append(i)
        #print('part_cards_crop1:', len(part_cards_crop1))
        #----------分割粘连字符--------
        part_cards_len_cols = []
        for pre_part_card in part_cards_crop1:
            part_cards_len_col = pre_part_card.shape[1]
            part_cards_len_cols.append(part_cards_len_col)
        part_cards_mean = sum(part_cards_len_cols)/len(part_cards_crop1)

        # part_cards_len_cols.sort()
        # part_cards_len_cols = part_cards_len_cols[3:len(part_cards_len_cols)-3]
        # part_cards_mean = sum(part_cards_len_cols) / len(part_cards_crop1)

        #part_cards_mean = min(part_cards_len_cols)
        print('min_part_cards_len_cols:',part_cards_len_cols,part_cards_mean)
        if len(part_cards_crop1) < 17:
            extra_image = []
            for pre_part_card in part_cards_crop1:
                part_cards_len_col = pre_part_card.shape[1]
                # part_card_seg = [pre_part_card]

                seperate_card_connect(part_cards_mean, pre_part_card, part_cards_len_col, extra_image)
            part_cards_crop1 = extra_image
            # if part_cards_len_col > part_cards_mean + 10:
            #     part_card_seg = seperate_card_connect(part_cards_mean,pre_part_card, part_cards_len_col)
            # extra_image += part_card_seg
        #print('part_cards_len_cols:',part_cards_len_cols,len(part_cards_len_cols),part_cards_mean)

        #----------分割粘连字符完----------

        # ----------------再次剔除噪点--------------
        part_cards_crop_end = []
        part_cards_crop_row_end = []
        if len(part_cards_crop1) > 17:
            for j in part_cards_crop1:  # part_cards_crop；part_cards
                s1 = 0
                s1 = np.sum(j, axis=1)
                j_row = j.shape[0]
                part_cards_crop_row_end.append(j_row)
                row_end_mean = np.sum(part_cards_crop_row_end)
                thrsh_row = (row_end_mean/len(part_cards_crop1))-20
                s1 = [k for k in s1 if k > 0]
                if len(s1) > thrsh_row:  # 原来是30,10,15
                    part_cards_crop_end.append(j)
            part_cards_crop1 = part_cards_crop_end

        for i, part_card in enumerate(part_cards_crop1):  # part_cards_crop ;
            #-----画出每张字符轮廓查看噪声并去除噪声-----
            _,contours,h = cv2.findContours(part_card,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            n = len(contours)
            # contoursImg = []
            # for k in range(n):
            #     temp = np.zeros(part_card.shape,np.uint8)
            #     contoursImg.append(temp)
            #     cv2.drawContours(contoursImg[k],contours,k,255,2)
            #     cv2.imshow('contour-'+str(k),contoursImg[k])
            #     cv2.waitKey()
            #
            #     area = cv2.contourArea(np.array(contours[k]))
            #     if area < 80:  # 25
            #         cv2.drawContours(part_card, [contours[k]], 0, 0, -1)
            cv2.imshow('part_card',part_card)
            cv2.waitKey()

            part_card_connect = []
            part_card_h, part_card_w = part_card.shape
            img_seg_col = part_card.shape[1]
            # if len(part_cards_crop1) < 17:
            #     #if img_seg_col > part_cards_mean + 10:
            #         #part_card_connect1,part_card_connect2 = seperate_card_connect(part_card,img_seg_col)  # part_card_connect1,part_card_connect2不能放在列表里
            #     part_card_seg = seperate_card_connect(part_cards_mean,part_card, img_seg_col)
            #     for part_card_end in part_card_seg:
            #         #if part_card_end.shape[1] > part_cards_mean + 10:
            #         part_card_h_1, part_card_w_1 = part_card_end.shape
            #         arr = np.array([[20 / part_card_h_1, 0, 5], [0, 20 / part_card_h_1, 0]], np.float32)
            #         part_card_1 = cv2.warpAffine(part_card_end, arr, (20, 20), borderValue=(0, 0))
            #         part_card_1 = preprocess_hog([part_card_1])
            #         resp = self.model.predict(part_card_1)
            #         charactor = chr(resp[0])
            #         predict_result.append(charactor)
            #         cv2.imshow('part_card_connect1', part_card_end)
            #         cv2.waitKey()
            #         # part_card_h_1, part_card_w_1 = part_card_connect1.shape
            #         # arr = np.array([[20 / part_card_h_1, 0, 5], [0, 20 / part_card_h_1, 0]], np.float32)
            #         # part_card_1 = cv2.warpAffine(part_card_connect1, arr, (20, 20), borderValue=(0, 0))
            #         # part_card_1 = preprocess_hog([part_card_1])
            #         # resp = self.model.predict(part_card_1)
            #         # charactor = chr(resp[0])
            #         # predict_result.append(charactor)
            #         # cv2.imshow('part_card_connect1', part_card_connect1)
            #         # cv2.waitKey()
            #         #
            #         # part_card_h_2, part_card_w_2 = part_card_connect2.shape
            #         # arr = np.array([[20 / part_card_h_2, 0, 5], [0, 20 / part_card_h_2, 0]], np.float32)
            #         # part_card_2 = cv2.warpAffine(part_card_connect2, arr, (20, 20), borderValue=(0, 0))
            #         # part_card_2 = preprocess_hog([part_card_2])
            #         # resp = self.model.predict(part_card_2)
            #         # charactor = chr(resp[0])
            #         # predict_result.append(charactor)
            #         # cv2.imshow('part_card_connect2', part_card_connect2)
            #         # cv2.waitKey()
            #         continue

                # for k in range(len(part_card_connect)):
                #     #print('len(part_card):', len(part_card))
                #     part_card = part_card_connect[k]
                #     cv2.imshow('part_card', part_card_connect[k])
                #     cv2.waitKey()
                #     arr = np.array([[20 / part_card_h, 0, 5], [0, 20 / part_card_h, 0]], np.float32)
                #     part_card = cv2.warpAffine(part_card, arr, (20, 20), borderValue=(0, 0))
                #     part_card = preprocess_hog([part_card])
                #     resp = self.model.predict(part_card)
                #     charactor = chr(resp[0])
                #     predict_result.append(charactor)
                #     break
            #print('img_seg_col:',img_seg_col)
            # sum =Sum_part_card(part_card)

            # cv2.imshow('part_card',part_card)
            # cv2.waitKey()
            # #--------找字符轮廓并画出轮廓-------------
            # pic_hight, pic_width = part_card.shape[:2]
            # cont, hie = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # if len(cont) > 1:
            # 	contimg = []
            # 	for k in range(len(cont)):
            # 		temp = np.zeros(binary.shape,np.uint8)
            # 		contimg.append(temp)
            # 		cv2.drawContours(contimg[k],cont,k,255,2)
            # 		cv2.imshow('cont-'+str(k),contimg[k])
            # 		cv2.waitKey()
            #
            # print('len(cont):',len(cont))
            # cnt = cont[0]
            #
            # img_color1 = cv2.cvtColor(part_card, cv2.COLOR_GRAY2BGR)
            # img_color2 = np.copy(img_color1)
            # #cv2.drawContours(img_color1, [cont], 0, (0, 0, 255), 2)
            # rect = cv2.minAreaRect(cnt)  # 最小外接矩形
            # angle = rect[-1]
            #
            # if angle > 0:
            # 	if abs(angle) > 45:
            # 		angle = 90 - abs(angle)
            # else:
            # 	if abs(angle) > 45:
            # 		angle = 90 - abs(angle)
            #
            # part_card = rotate(part_card, angle)
            # print('angle:', angle)
            #
            # box = cv2.boxPoints(rect)
            #
            # box = np.int0(cv2.boxPoints(rect))  # 矩形的四个角点取整
            #
            # cv2.drawContours(img_color1, [box], 0, (255, 0, 0), 2)
            # cv2.imshow('rectbox',img_color1)
            # cv2.waitKey()
            # -----------------找字符轮廓并画出轮廓-------------
            # 可能是固定车牌的铆钉
            # 			if np.mean(part_card) < 255/5:
            # #						print("a point")
            # 				continue
            part_card_old = part_card
            w = abs(part_card.shape[1] - SZ) // 2
            # part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value = [0,0,0])
            # part_card = cv2.resize(part_card, (20, 20), interpolation=cv2.INTER_AREA)

            arr = np.array([[20/part_card_h, 0, 5], [0, 20/part_card_h, 0]], np.float32)
            part_card = cv2.warpAffine(part_card, arr, (20, 20), borderValue=(0, 0))
            #cv2.imwrite('test'+str(i) + '.jpg', part_card)
            #print('pend=',part_card.shape)
            # part_card = cv2.resize(part_card, (20, 50), None)
            #			cv2.namedWindow('part_card_show', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('part_card', part_card)
            # cv2.waitKey()
            if i == 14:
                cv2.imwrite('./test_pictures/part_card/'+ str(3.1)+'.jpg', part_card)
            # part_card = deskew(part_card)
            part_card = preprocess_hog([part_card])
            # if i == 0:
            # 	resp = self.modelchinese.predict(part_card)
            # 	charactor = provinces[int(resp[0]) - PROVINCE_START]
            # else:
            resp = self.model.predict(part_card)
            charactor = chr(resp[0])
            # 判断最后一个数是否是车牌边缘，假设车牌边缘被认为是1
            # if charactor == "1" and i == len(part_cards)-1:
            # 	if part_card_old.shape[0]/part_card_old.shape[1] >= 7:#1太细，认为是边缘
            # 		continue
            predict_result.append(charactor)
        print('len(predict_result):',len(predict_result))
        # roi = card_img
        # card_color = color
        # break

        return ''.join(predict_result) #predict_result #''.join(predict_result)   # , roi, card_color#识别到的字符、定位的车牌图像、车牌颜色

if __name__ == '__main__':
    c = CardPredictor()
    c.train_svm()
    #source_path = "D:\\TaxiDismantlingVerificationSystem\\CollectingPictures\\"
    #source_path = "C:\\opencv_image\\new\\"
    source_path = './test_pictures/test/' #C:\国交空间\车牌号相关\License-Plate-Recognition-master\test_pictures\test
    image_list = os.listdir(source_path)
    r = c.predict('./test_pictures/test/w1.jpg')  # r, roi, color = c.predict("3.png")
    i = 0
    # for file in image_list:
    #     i = i + 1
    #     print('file:',file)
    #     r = c.predict(source_path+file)
    #     print(r)

    print(r)






# C:\国交空间\车牌号相关\test-chepai\License-Plate-Recognition-master\predict