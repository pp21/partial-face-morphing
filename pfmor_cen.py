# Ref.[1] https://learnopencv.com/face-morph-using-opencv-cpp-python/
# Ref.[2] https://learnopencv.com/face-swap-using-opencv-c-python/

import numpy as np
import cv2
import dlib
from scipy.spatial import Delaunay

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst

# Warps and alpha blends triangular regions from img1 and img2 to img
def morphTriangle(img1, img2, img, t1, t2, t, alpha):

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = alpha * warpImage1 + (1.0 - alpha) * warpImage2
    
    # nd fd size check
    tpimg = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]
    if tpimg.shape[0] != imgRect.shape[0]:
        imgRect = cv2.resize(imgRect, (tpimg.shape[1], tpimg.shape[0]))
        mask = cv2.resize(mask, (tpimg.shape[1], tpimg.shape[0]))
        print('****** tri size error ******')

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask

# get points for complete face morphing    
def getpotins(img):
    
    # get dlib 68 face landmarks
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    try:
        dets = detector(img, 1)[0]
    except:
        print('No face detected')
    points = [[p.x, p.y] for p in predictor(img, dets).parts()]
    
    # auxiliary points
    x = img.shape[1] - 1
    y = img.shape[0] - 1
    points.append([0, 0])
    points.append([x, 0])
    points.append([0, y])
    points.append([x, y])
    for i in range(1, 5):
        points.append( [x // 5 * i, 0] )
        points.append( [x, y // 5 * i] )
        points.append( [x // 5 * i, y] )
        points.append( [0, y // 5 * i] )
        
    return points
    
# complete face morph
# input: face1, face2 - source faces for morphing
#        alpha1 - location fusion weight for face1, cv2.imread img
#        alpha2 - intensity fusion weight for face1
# output: mface - a complete morphed face
def morphface_c(face1, face2, alpha1, alpha2):
    
    # get morphing points
    points_f1 = getpotins(face1)
    points_f2 = getpotins(face2)
    points_mf = []
    
    # Compute weighted average point coordinates
    for i in range(0, len(points_f1)):
        x = alpha1 * points_f1[i][0] + ( 1 - alpha1 ) * points_f2[i][0]
        y = alpha1 * points_f1[i][1] + ( 1 - alpha1 ) * points_f2[i][1]
        points_mf.append( (x, y) )
    
    # Delaunay triangulation
    tri = Delaunay(points_mf).simplices
    
    # Convert Mat to float data type
    face1 = np.float32(face1)
    face2 = np.float32(face2)
    
    # Allocate space for final output
    mface = np.zeros(face1.shape, dtype = face1.dtype)
    
    # foe each triangle
    for i in range(0, len(tri)):
        x = int(tri[i][0])
        y = int(tri[i][1])
        z = int(tri[i][2])
        
        t1 = [points_f1[x], points_f1[y], points_f1[z]]
        t2 = [points_f2[x], points_f2[y], points_f2[z]]
        t = [ points_mf[x], points_mf[y], points_mf[z] ]
        
        # Morph one triangle at a time.
        morphTriangle(face1, face2, mface, t1, t2, t, alpha2)
        
    return mface

# get points for eyes and nose swap   
def getpotins_sw(img):
    
    # get dlib 68 face landmarks
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    try:
        dets_sw = detector(img, 1)[0]
    except:
        print('No face detected')
    points = [[p.x, p.y] for p in predictor(img, dets_sw).parts()]
    
    # get points
    cenpoints = []
    cenpoints.append( [ points[17][0], points[18][1] ] )
    cenpoints.append( points[19] )
    cenpoints.append( [ round( ( points[19][0]+points[24][0] )/2 ), round( ( points[19][1]+points[24][1] )/2 ) ] )
    cenpoints.append( points[24] )
    cenpoints.append( [ points[26][0], points[25][1] ] )
    cenpoints.append( [ round( ( points[16][0]+points[45][0] )/2 ), round( ( points[16][1]+points[45][1] )/2 ) ] )
    cenpoints.append( [ points[26][0], points[15][1] ] )
    cenpoints.append( [ points[26][0], points[14][1] ] )
    cenpoints.append( [ round( ( points[35][0]+points[54][0] )/2 ), round( ( points[35][1]+points[54][1] )/2 ) ] )
    cenpoints.append( [ round( ( points[33][0]+points[51][0] )/2 ), round( ( points[33][1]+points[51][1] )/2 ) ] )
    cenpoints.append( [ round( ( points[31][0]+points[48][0] )/2 ), round( ( points[31][1]+points[48][1] )/2 ) ] )
    cenpoints.append( [ points[17][0], points[2][1] ] )
    cenpoints.append( [ points[17][0], points[1][1] ] )
    cenpoints.append( [ round( ( points[0][0]+points[36][0] )/2 ), round( ( points[0][1]+points[36][1] )/2 ) ] )
        
    bb = min( points[49][1], points[51][1], points[53][1] )
      
    return cenpoints, bb

# face swap
# input: face1, face2 - source faces for swap, face1 is target
# output: swface - a face with component swap, from face2 to face1
def facedswap(img1, img2):       
    
    # get array of corresponding points
    points1, bb1 = getpotins_sw(img1)
    points2, bb2 = getpotins_sw(img2)
    
    # get component center in target face
    tl = min(points1)[0]
    tr = max(points1)[0]
    tt = min(points1, key=lambda x:x[1])[1]
    tb = max(points1, key=lambda x:x[1])[1]
    t_cen = ( round( (tl + tr) / 2 ), round( (tt + tb) / 2 ) )   
    
    # Create a rough mask around the component
    src_mask = np.zeros(img2.shape, img2.dtype)
    poly = np.array(points2, np.int32)
    cv2.fillPoly(src_mask, [poly], (255, 255, 255))
    
    swface = cv2.seamlessClone(img2, img1, src_mask, t_cen, cv2.NORMAL_CLONE)
    
    return swface

# central face region morphing
# input:  imgpath1 - path of face img), imgpath2 - path of face img
#         alpha1 - location morphing weight for imgpath1
#         alpha2 - intensity morphing weight for imgpath2
# output: pfmorimg - partial morphed face
def mor_cen(imgpath1, imgpath2, alpha1, alpha2):
    # Read images
    img1 = cv2.imread(imgpath1)
    img2 = cv2.imread(imgpath2)        
    mface = morphface_c(img1, img2, alpha1, alpha2)
    pfmorimg = facedswap( img1, np.uint8(mface) )
    return pfmorimg
 