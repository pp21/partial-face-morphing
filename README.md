# partial-face-morphing

# 依赖库
numpy
cv2
dlib
from scipy.spatial import Delaunay
shape_predictor_68_face_landmarks.dat

# 输入图像说明
本代码仅适用于常见的证件人脸图像(1024*1024像素以上)

# 使用步骤示例

## 1 预处理
imgpath = 'xxx.jpg' 图像路径
from face_nor import fimg_nor
norimg = fimg_nor(imgpath)

## 2 融合

imgpath1 = 'xxx.jpg' 图像路径
imgpath2 = 'xxx.jpg' 图像路径

### 融合权重(取值范围0.1-0.9)
alpha1 = 0.5 位置融合权重
alpha2 = 0.5 像素融合权重

### 鼻子融合
from pfmor_nose import mor_nose
morimg_nose = mor_nose(imgpath1, imgpath2, alpha1, alpha2)

### 眼睛鼻子融合
from pfmor_eyes_nose_c import mor_eyes_nose
morimg_eyes_nose = mor_eyes_nose(imgpath1, imgpath2, alpha1, alpha2)

### 中心脸区域融合
from pfmor_cen import mor_cen
morimg_cen = mor_cen(imgpath1, imgpath2, alpha1, alpha2)
