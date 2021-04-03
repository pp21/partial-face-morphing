import cv2
import dlib

# eyes based rotation
def warp_affine(image, dx, dy, cen, scale):

    # get rotation degree
    bdval = [ image[0][0][0], image[0][0][1], image[0][0][2] ]
    bdval = tuple( int(x) for x in bdval )
    angle = cv2.fastAtan2(dy, dx)
    rot = cv2.getRotationMatrix2D(cen, angle, scale=scale)
    rot_img = cv2.warpAffine( image, rot, dsize=(image.shape[1], image.shape[0]), borderValue=(bdval[0], bdval[1], bdval[2]) )
    return rot_img

# face normalization
# input - imgpath (image path)
# output - a normalized image
def fimg_nor(imgpath):
    # paras
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('D:/Kits/shape_predictor_68_face_landmarks.dat')
    sdfw = 413
    sdfh = 531
    perc = 0.7
    
    # read img
    fimg = cv2.imread(imgpath)
    
    # rotation
    try:
        dets = detector(fimg, 1)[0]
    except:
        print('No ori face detected')
    points = [[p.x, p.y] for p in predictor(fimg, dets).parts()]
    eyedx = points[46][0] - points[41][0]
    eyedy = points[46][1] - points[41][1]
    img_rt = warp_affine(fimg, eyedx, eyedy, tuple(points[27]), 1.0)   
    
    # resize
    try:
        dets = detector(img_rt, 1)[0]
    except:
        print('No rot face detected')
    points = [[p.x, p.y] for p in predictor(img_rt, dets).parts()]
    fw = points[16][0] - points[0][0]
    sc = (sdfw * perc) / fw
    fw_sc = round( img_rt.shape[1]*sc )
    fh_sc = round( img_rt.shape[0]*sc )
    fimg_sc = cv2.resize(img_rt, (fw_sc, fh_sc))
    
    # crop
    try:
        dets = detector(fimg_sc, 1)[0]
    except:
        print('No scale face detected')
    points = [[p.x, p.y] for p in predictor(fimg_sc, dets).parts()]
    cen_x = points[27][0]
    cen_y = points[27][1]
    fimg_psd = fimg_sc[ cen_y-sdfh//2:cen_y+sdfh//2+1, cen_x-sdfw//2:cen_x+sdfw//2+1, : ]
    return fimg_psd

