import cv2
import numpy as np
from matplotlib import pyplot as plt

def corner(source_image):
	img = cv2.imread(source_image)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	# find Harris corners
	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray,2,3,0.04)
	dst = cv2.dilate(dst,None)
	ret, dst = cv2.threshold(dst,0.001*dst.max(),255,0)

	dst = np.uint8(dst)
	#cv2.imshow("dst", dst)
	#cv2.waitKey()
	# find centroids
	ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

	# define the criteria to stop and refine the corners
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
	corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
	return corners
	# Now draw them
	#res = np.hstack((centroids,corners))
	#res = np.int0(res)
	#img[res[:,1],res[:,0]]=[0,0,255]
	#img[res[:,3],res[:,2]] = [0,255,0]

	#cv2.imshow("corner", img)
	#cv2.waitKey()
	#cv2.destroyAllWindows()

def blobs(source_image):
	im = cv2.imread(source_image, cv2.IMREAD_GRAYSCALE)
	
	params = cv2.SimpleBlobDetector_Params()
	# Change thresholds
	params.minThreshold = 200;
	params.maxThreshold = 255;
 
	# Filter by Area.
	params.filterByArea = True
	params.minArea = 16
 
	# Filter by Circularity
	params.filterByCircularity = True
	params.minCircularity = 0.1
 
	# Filter by Convexity
	params.filterByConvexity = True
	params.minConvexity = 0.87
 
	# Filter by Inertia
	params.filterByInertia = True
	params.minInertiaRatio = 0.1
 
	# Create a detector with the parameters
	ver = (cv2.__version__).split('.')
	if int(ver[0]) < 3 :
		detector = cv2.SimpleBlobDetector(params)
	else : 
		detector = cv2.SimpleBlobDetector_create(params)
		# Detect blobs.
	keypoints = detector.detect(im) 
	# Draw detected blobs as red circles.
	# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
	im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
	# Show keypoints
	#cv2.imshow("Keypoints", im_with_keypoints)
	#cv2.waitKey()
	#cv2.destroyAllWindows()

def ShowImage(image):
	cv2.imshow('123',image)
	cv2.waitKey()
	cv2.destroyAllWindows()

def RotateImage(source_image,matrix):
	im=cv2.imread(source_image)
	dst = cv2.warpAffine(im,matrix)
	return dst
	



filename = 'P:\\Programm\\Pyton\SearchForMark\\Mark\\qr.png'
filename_gomo = 'P:\\Programm\\Pyton\SearchForMark\\Mark\\scene.bmp'
#source_corner = corner(filename)
#gomo_corner = corner(filename_gomo)


FLANN_INDEX_LSH = 6
MIN_MATCH_COUNT = 6

img1 = cv2.imread(filename,0)          # queryImage
img2 = cv2.imread(filename_gomo,0)          # trainImage

# Initiate ORB detector
sift = cv2.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
search_params = dict(checks=100)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe ratio test.
good = []

# Need to draw only good matches, so create a mask
#matchesMask = [[0,0] for i in xrange(len(matches))]

# ratio test as per Lowe paper
for m_n in matches:
    if len(m_n) != 2:
        continue
    (m,n) = m_n
    if m.distance < 0.9*n.distance:
        #matchesMask[i]=[1,0]
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    
    ShowImage(img2)
    img2 = cv2.polylines(img2,[np.int32(dst)],True,0,3, cv2.LINE_AA)
    ShowImage(img2)
else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None


draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = None,
                   matchesMask = matchesMask,
                   flags = 0)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
ShowImage(cv2.warpPerspective(img1,M,(h,w)))
size = img1.shape

perspectiveM = cv2.getPerspectiveTransform(np.float32(dst),pts)

img_dst = cv2.warpPerspective(img2,perspectiveM,size)

plt.imshow(img3, 'gray')
plt.show()

#transform_matrix = cv2.getPerspectiveTransform(gomo_corner,source_corner) 
#ShowImage(RotateImage(filename,transform_matrix))
