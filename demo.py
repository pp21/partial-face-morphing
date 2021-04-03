# face normalization
imgpath = 'xxx.jpg' # your img path
from face_nor import fimg_nor
norimg = fimg_nor(imgpath)

# img path
imgpath1 = 'xxx.jpg' # your img (after face normalization) path
imgpath2 = 'xxx.jpg' # your img (after face normalization) path

# morphing weight (from 0.1 to 0.9)
alpha1 = 0.5
alpha2 = 0.5

## nose based morph
from pfmor_nose import mor_nose
morimg_nose = mor_nose(imgpath1, imgpath2, alpha1, alpha2)

# eyes and nose based morph
from pfmor_eyes_nose_c import mor_eyes_nose
morimg_eyes_nose = mor_eyes_nose(imgpath1, imgpath2, alpha1, alpha2)

# central face morph
from pfmor_cen import mor_cen
morimg_cen = mor_cen(imgpath1, imgpath2, alpha1, alpha2)
