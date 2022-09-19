import cv2
import os
# 按指定图像大小调整尺寸C:\国交空间\车牌号相关\License-Plate-Recognition-master\train\untitled0.py

"author:youngkun date:20180615 function:change the size of pictures in one folder"

import cv2
import os

image_size=20                         #设定尺寸
source_path="C:\\opencv_image\\new\\resize-8\\"                     #源文件路径
target_path="C:\\opencv_image\\resize\\8\\"        #输出目标文件路径

if not os.path.exists(target_path):

    os.makedirs(target_path)
image_list=os.listdir(source_path)      #获得文件名
i=0
for file in image_list:

    i=i+1

    image_source=cv2.imread(source_path+file)#读取图片

    image = cv2.resize(image_source, (20, 20),0,0, cv2.INTER_LINEAR)#修改尺寸

    cv2.imwrite(target_path+str(i)+".jpg",image)           #重命名并且保存

print("批量处理完成")

