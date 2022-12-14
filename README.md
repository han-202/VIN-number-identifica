# VIN-number-identifica
vehicle frame VIN number identifica
一、ROI区域提取（车架号定位）
首先，将输入图像进行色彩空间转换，转换成HSV色彩空间，以便于提取红色方框区域。通过设置颜色提取低值和高值（红色的低值lower_hsv = np.array([0, 43, 46])，红色高值high_hsv = np.array([10, 255, 255])），去除背景部分，得到只有红色方框的二值图像。如下图所示。

然后，在原图中定位并截取车架号区域。通过查找该二值图像的轮廓（cv2.findContours），计算所有轮廓的面积，找到最大的轮廓即为需要提取的红色方框轮廓。接着找到该矩形轮廓的四个顶点的坐标，最后通过该坐标定位到原图中，并截取车架号区域。如下图所示。

二、ROI区域图像预处理并转成二值图像
将ROI区域图像预处理，是为了得到清晰的二值图像，转换成清晰的二值图像后，能够准确的分割图像。
首先，将ROI区域图像转换成灰度图像（cv2.cvtColor()）；然后，利用最小值滤波的方法对ROI区域进行预处理，再对得到的图像进行均值滤波；最后，将ROI区域灰度图像与均值滤波后的图像相减，得到需要进行二值化的图像。
利用大律法将上述预处理过的图像转化成二值图像。

三、二值图像去噪后处理（也给示意图）
由于ROI区域二值化后，会存在很多或大或小噪点，不利于后续进行字符分割，需要将这些噪点剔除。
利用查找轮廓的方法，剔除字符外的其它轮廓。由于像A、B这种内部有封闭轮廓的字符，在查找轮廓时，除了外部的整体轮廓外，内部封闭轮廓也会被单独作为轮廓输出，容易被当作噪点轮廓剔除，因此需要先将这些字符内部空洞进行填充。
空洞填充。首先，取ROI区域二值图像的补集；然后，构建一个与ROI区域二值图像同样大小的矩阵，并将补集的边界值写入，循环迭代，对该矩阵进行膨胀操作，结果与二值图像的补集执行and操作；最后，对经过上述处理过的图像取非（not）,得到ROI区域二值化的空洞填充的图像。（弄一个示意图）

剔除噪点，净化二值图像。查找空洞填充后的二值图像的轮廓，求出所有轮廓的最小外接圆的面积，并将其从大到小排序。将小于前17（车架号共17位）个面积中最小值的所有轮廓剔除，最后，将该二值图像与ROI区域二值化的图像进行与（and）操作，得到待分割的二值图像。

四、字符分割
利用投影法进行字符分割。首先，在垂直方向对上述二值化的图像中的白点像素进行投影统计，并根据统计结果的均值和最小值计算出分割阈值，根据阈值和统计结果找出每个字符的波峰和波谷，从而能够判断出每个字符在垂直方向上的实际占位范围，实现从二值图像中分割出每个字符。（给个示意图）
然后，在水平方向对上述二值化的图像中的白点像素和黑点像素进行投影统计，根据统计结果找出每个字符水平方向上的波峰（判断是否是字符）和波谷（该字符结束），判断出每个字符在水平方向上的实际占位范围，从而实现将上述分割出的每个字符的中上下非字符部分裁剪掉，以便减少后续进行字符图像尺寸变换（需要把分割的字符图像尺寸缩小到训练集字符图像的大小）时对识别的影响。
最后，分割粘连字符。由于图像中噪声的影响，进行字符分割时不可避免的造成把两个或者以上字符分割在一个图像中，从而影响识别结果，因此，需要将其进一步分割。计算上述分割出的每幅字符图像的宽度，根据其中的最小值与均值计算出新的分割阈值，将粘连字符分割。

五、字符识别
1、特征提取（来自opencv的sample）
采用方向梯度直方图（Histogram of Oriented Gradient, HOG）提取字符
特征。HOG特征的基本思想是：在一副图像中，局部目标的表象和形状能够被梯度或边缘的方向密度分布很好地描述，通过计算和统计图像局部区域的梯度方向直方图来构成特征。Hog特征结合SVM分类器已经被广泛应用于图像识别中（本质：梯度的统计信息，而梯度主要存在于边缘的地方）。
具体过程如下：
（1）标准化gamma空间和颜色空间
为了减少光照因素的影响，首先需要将整个图像进行规范化（归一化）。在图像的纹理强度中，局部的表层曝光贡献的比重较大，所以，这种压缩处理能够有效地降低图像局部的阴影和光照变化。因为颜色信息作用不大，通常先转化为灰度图。

（2）计算图像梯度
计算图像横坐标和纵坐标方向的梯度，并据此计算每个像素位置的梯度方向值；求导操作不仅能够捕获轮廓，还能进一步弱化光照的影响。

（3）为每个细胞单元构建梯度方向直方图
将图像分成若干个“单元格cell”，例如每个cell为8*8个像素。假设我们采用9个bin的直方图来统计这8*8个像素的梯度信息。也就是将cell的梯度方向360度分成9个方向块，对cell内每个像素用梯度方向在直方图中进行加权投影（映射到固定的角度范围），就可以得到这个cell的梯度方向直方图了，就是该cell对应的9维特征向量（因为有9个bin）。

（4）把细胞单元组合成大的块（block），块内归一化梯度直方图
由于局部光照的变化以及前景-背景对比度的变化，使得梯度强度的变化范围非常大。这就需要对梯度强度做归一化。归一化能够进一步地对光照、阴影和边缘进行压缩。

（5）收集HOG特征
最后一步就是将检测窗口中所有重叠的块进行HOG特征的收集，并将它们结合成最终的特征向量供分类使用。

（6）一个图像的HOG特征维数
根据Dalal提出的Hog特征提取的过程：把样本图像分割为若干个像素的单元（cell），把梯度方向平均划分为9个区间（bin），在每个单元里面对所有像素的梯度方向在各个方向区间进行直方图统计，得到一个9维的特征向量，每相邻的4个单元构成一个块（block），把一个块内的特征向量联起来得到36维的特征向量，用块对样本图像进行扫描，扫描步长为一个单元。最后将所有块的特征串联起来，就得到了人体的特征。例如，对于64*128的图像而言，每8*8的像素组成一个cell，每2*2个cell组成一个块，因为每个cell有9个特征，所以每个块内有4*9=36个特征，以8个像素为步长，那么，水平方向将有7个扫描窗口，垂直方向将有15个扫描窗口。也就是说，64*128的图片，总共有36*7*15=3780个特征。
2、数据集训练
首先将训练样本集中的字符图像用HOG特征提取方法提取每个类别中所有图像的特征，然后转换成libsvm的支持格式，输入到libsvm进行训练。Libsvm的标准数据格式如下：
<label1>  <index1>:<value1>  <index2>:<value2>     .....  <index L>:<valueL>
label：是分类的类别的标识
index：特征值的序列号
value：就是要训练的数据，即特征值，数据之间用空格隔开，如果特征值为0，特征冒号前面的序号可以不连续。
