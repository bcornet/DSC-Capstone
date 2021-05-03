# code from Adrian Rosebrock at pyimagesearch.com:
# https://www.pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/#download-the-code

# "image", "template" should be numpy arrays from cv2.imread()
# note that cv2 uses BGR while PIL uses RGB; may need to swap

# import the necessary packages
import numpy as np
import imutils
import cv2
def align_images(image, template, maxFeatures=500, keepPercent=0.2,
	debug=True):
	# convert both the input image and template to grayscale
	imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
	if False:
        imageGray = auto_canny(imageGray)
        templateGray = auto_canny(templateGray)
	# use ORB to detect keypoints and extract (binary) local
	# invariant features
	orb = cv2.ORB_create(maxFeatures)
	(kpsA, descsA) = orb.detectAndCompute(imageGray, None)
	(kpsB, descsB) = orb.detectAndCompute(templateGray, None)
	
	# match the features
	method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
	matcher = cv2.DescriptorMatcher_create(method)
	matches = matcher.match(descsA, descsB, None)
	
	# sort the matches by their distance (the smaller the distance,
	# the "more similar" the features are)
	matches = sorted(matches, key=lambda x:x.distance)
	
	# keep only the top matches
	keep = int(len(matches) * keepPercent)
	matches = matches[:keep]
	
	# check to see if we should visualize the matched keypoints
	if debug:
		matchedVis = cv2.drawMatches(image, kpsA, template, kpsB,
			matches, None)
		matchedVis = imutils.resize(matchedVis, width=1000)
		#cv2.imshow("Matched Keypoints", matchedVis)
        cv2.imwrite("align_debug.png",matchedVis)
		#cv2.waitKey(0)
	
	# allocate memory for the keypoints (x, y)-coordinates from the
	# top matches -- we'll use these coordinates to compute our
	# homography matrix
	ptsA = np.zeros((len(matches), 2), dtype="float")
	ptsB = np.zeros((len(matches), 2), dtype="float")
	
	# loop over the top matches
	for (i, m) in enumerate(matches):
		# indicate that the two keypoints in the respective images
		# map to each other
		ptsA[i] = kpsA[m.queryIdx].pt
		ptsB[i] = kpsB[m.trainIdx].pt
	
	# compute the homography matrix between the two sets of matched
	# points
	(H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
	# use the homography matrix to align the images
	(h, w) = template.shape[:2]
	aligned = cv2.warpPerspective(image, H, (w, h))
	# return the aligned image
	return aligned

# this is also from Rosebrock; didn't see much success compared to other canny method
def auto_canny(image, sigma=0.33): # cv2 canny is pretty rough
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged
    
# return with cv2:
# cv2.imwrite("filename.png",aligned)
dy1 = cv2.imread("donghui.jpg")
dy2 = cv2.imread("donghui_base.png")
dalign = align_images(dy1,dy2)

# none of these worked
mh = cv2.imread("mhgu.jpg")
mh_photo = cv2.imread("bri_mhgu_photo.jpg")
mh_clean = cv2.imread("bri_mhgu_clean.jpg")

mhc_otherc = cv2.imread("mhgu_canny.png")
mhc_photoc = cv2.imread("bri_mhgu_photo_canny.png")
mhc_cleanc = cv2.imread("bri_mhgu_clean_canny.png")

mhc_photo1 = cv2.imread("bri_mhgu_photo_crop1.png")
mhc_clean1 = cv2.imread("bri_mhgu_clean_crop1.png")
crop1_align = align_images(mhc_photo1,mhc_clean1)
cv2.imwrite("bri_mhgu_align8.png",crop1_align)

mhc_photo2 = cv2.imread("bri_mhgu_photo_crop2.png")
mhc_clean2 = cv2.imread("bri_mhgu_clean_crop2.png")
crop2_align = align_images(mhc_photo2,mhc_clean2)
cv2.imwrite("bri_mhgu_align9.png",crop2_align)

# if using a canny image, can just use red component
mhc_photo_rect = mhc_photoc[:,:,0]
# for matching on a regular rectangle
blank = np.zeros(mhc_photo_rect.shape)
for i,j in enumerate(mhc_photo_rect.argmax(axis=1)):
    blank[i,j] = 255
for j,i in enumerate(mhc_photo_rect.argmax(axis=0)):
    blank[i,j] = 255
mhc_photo_rect = np.flip(mhc_photo_rect)
blank = np.flip(blank)
for i,j in enumerate(mhc_photo_rect.argmax(axis=1)):
    blank[i,j] = 255
for j,i in enumerate(mhc_photo_rect.argmax(axis=0)):
    blank[i,j] = 255
blank = np.flip(blank)
Image.fromarray(blank).show()

# clears the "interior" of a canny image
def rectangle(image):
    # should canny this first
    rect = image[:,:,0] # can use a gray image, but this assumes RGB
    blank = np.zeros(rect.shape)
    for i,j in enumerate(rect.argmax(axis=1)):
        blank[i,j] = 255
    for j,i in enumerate(rect.argmax(axis=0)):
        blank[i,j] = 255
    rect = np.flip(rect)
    blank = np.flip(blank)
    for i,j in enumerate(rect.argmax(axis=1)):
        blank[i,j] = 255
    for j,i in enumerate(rect.argmax(axis=0)):
        blank[i,j] = 255
    # clean up the image edges
    blank[:,(0,-1)] = 0
    blank[(0,-1),:] = 0
    blank = np.flip(blank)
    #Image.fromarray(blank).show()
    return blank

