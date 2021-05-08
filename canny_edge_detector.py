# Originally by Sofiane Sahir from Towards Data Science:
# https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
# (the download here has scripts and notebooks to use, does this on facial recognition!)

# added some notes by Bri

from scipy import ndimage
from scipy.ndimage.filters import convolve

from scipy import misc
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

class cannyEdgeDetector:
    def __init__(self, imgs, sigma=1, kernel_size=5, weak_pixel=75, strong_pixel=255, lowthreshold=0.05, highthreshold=0.15):
        self.imgs = imgs # list of original images to use; should be numpy matrices full o' float64s from 0.0 to 255.0
        self.imgs_final = [] # list of canny edge images; should be numpy matrices full o' int32s from 0 to 255
        self.img_smoothed = None # 
        self.gradientMat = None
        self.thetaMat = None
        self.nonMaxImg = None
        self.thresholdImg = None
        self.weak_pixel = weak_pixel
        self.strong_pixel = strong_pixel
        self.sigma = sigma # Gaussian filter sigma value
        self.kernel_size = kernel_size # Gaussian filter kernel size
        self.lowThreshold = lowthreshold # thresholding lower cutoff percentage; values below are 0
        self.highThreshold = highthreshold # thresholding upper cutoff; values above
        return 
    
    def gaussian_kernel(self, size=0, sigma=1): # should assume kernel_size is default
        if not size: # zero
            size = self.kernel_size
        size = int(abs(size)) // 2 # size is always odd; truncate decimals, then add 1 if size is even
        x, y = np.mgrid[-size:size+1, -size:size+1] # x = y.T and y = x.T
        # x rows and y cols are a range from -n//2 to n//2,
        # x cols and y rows are the same value repeated n times
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        # larger filter size means lower sensitivity to noise; 5 is a good size in most cases
        return g
    
    def sobel_filters(self, img):
        # kernels to use in convolution process; Wikipedia flips Kx in their example but it shouldn't matter
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32) # horizontal kernel
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32) # vertical kernel

        # applies the 2D signal processing convolution operation (* symbol; this is NOT matrix multiplication!): 
        # Ix = Kx*A and Iy = Ky*A
        Ix = ndimage.filters.convolve(img, Kx)
        Iy = ndimage.filters.convolve(img, Ky)

        G = np.hypot(Ix, Iy) # G = sqrt(Ix^2 + Iy^2)
        G = G / G.max() * 255 # scale to 255 range
        theta = np.arctan2(Iy, Ix) # theta angles for non-max suppression direction
        return (G, theta)
    

    def non_max_suppression(self, img, D):
        M, N = img.shape # dimensions
        Z = np.zeros((M,N), dtype=np.int32) # new matrix with 0 as default state
        angle = D * 180. / np.pi # radians to degrees
        angle[angle < 0] += 180 # all angles should be between 0 and 180

        # start iteratin' over every interior pixel (anything not part of the border)
        for i in range(1,M-1): 
            for j in range(1,N-1):
                try:
                    # these should be overwritten in every case
                    q = 255
                    r = 255

                    # switch cases for four possible angle ranges (hori, vert, both diagonals)
                    #angle 0
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180): # horizontal
                        q = img[i, j+1] # pixel to the right
                        r = img[i, j-1] # pixel to the left
                    #angle 45
                    elif (22.5 <= angle[i,j] < 67.5): # diagonal along y=-x (NW to SE)
                        q = img[i+1, j-1] # pixel above and to the left
                        r = img[i-1, j+1] # pixel below and to the right
                    #angle 90
                    elif (67.5 <= angle[i,j] < 112.5): # vertical
                        q = img[i+1, j] # pixel above
                        r = img[i-1, j] # pixel below
                    #angle 135
                    elif (112.5 <= angle[i,j] < 157.5): # diagonal along y=x (SW to NE); could be an else statement
                        q = img[i-1, j-1] # pixel below and to the left
                        r = img[i+1, j+1] # pixel above and to the right

                    if (img[i,j] >= q) and (img[i,j] >= r): # pixel is stronger than both adjacent pixels
                        Z[i,j] = img[i,j] # keep it!
                    else:
                        Z[i,j] = 0 # suppress it


                except IndexError as e: # oopsies
                    pass

        return Z

    def threshold(self, img):

        highThreshold = img.max() * self.highThreshold; # high cutoff value; if max is 255 and HT is 0.15, this is 38.25
        lowThreshold = highThreshold * self.lowThreshold; # low cutoff value; if above is 38.25 and LT is 0.05, this is 1.9125

        M, N = img.shape
        res = np.zeros((M,N), dtype=np.int32) # default of 0 for suppressed pixel locations

        # don't think these two need to be assigned but whatever
        weak = np.int32(self.weak_pixel) # weak_pixel default is 75
        strong = np.int32(self.strong_pixel) # strong_pixel default is 255

        strong_i, strong_j = np.where(img >= highThreshold) # strong where pixels are above the high threshold
        #zeros_i, zeros_j = np.where(img < lowThreshold) # gonna comment this out since it's unnecessary

        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold)) # weak between high and low thresholds

        res[strong_i, strong_j] = strong # assign the strong pixel value to all strong indices
        res[weak_i, weak_j] = weak # assign the weak pixel value to all weak indices

        return (res)


    def hysteresis(self, img): # suppresses any weak pixels not adjacent to strong pixels

        M, N = img.shape
        weak = self.weak_pixel
        strong = self.strong_pixel

        for i in range(1, M-1):
            for j in range(1, N-1):
                if (img[i,j] == weak):
                    try:
                        if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                            or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                            or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                            img[i, j] = strong # if a weak pixel has a strong in any adjacent position, make it strong
                        else:
                            img[i, j] = 0 # otherwise suppress it
                    except IndexError as e: # oopsies
                        pass

        return img


    def detect(self): # actually do the whole thing
        imgs_final = [] # I really don't like how this assumes a list but I'm too lazy to change it
        for i, img in enumerate(self.imgs): # don't know why we're enumerating
            self.img_smoothed = convolve(img, self.gaussian_kernel(self.kernel_size, self.sigma))
            self.gradientMat, self.thetaMat = self.sobel_filters(self.img_smoothed)
            self.nonMaxImg = self.non_max_suppression(self.gradientMat, self.thetaMat)
            self.thresholdImg = self.threshold(self.nonMaxImg)
            img_final = self.hysteresis(self.thresholdImg)
            self.imgs_final.append(img_final)

        return self.imgs_final




def rgb2gray(rgb): # converts MxNx3 RGB matrix to MxN grayscale
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    #gray = 0.2989 * r + 0.5870 * g + 0.1140 * b # doesn't equal to 1; might be a typo
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray

def rgb2gray_part(rgb): # just scales the MxNx3 matrix colors without combining them
    rgb[:,:,0] = 0.299*rgb[:,:,0]
    rgb[:,:,1] = 0.587*rgb[:,:,0]
    rgb[:,:,2] = 0.114*rgb[:,:,0]
    return rgb

# can preview both of the above with this:
if False:
    gray = np.arange(256).reshape([16,16])
    rgb_orig = np.array([gray,gray,gray]).astype(float).swapaxes(0,2)
    rgb_part = rgb2gray_part(rgb_orig.copy())
    rgb_gray = rgb2gray(rgb_orig.copy())

def visualize(imgs, format=None, gray=False):
    plt.figure(figsize=(20, 40))
    for i, img in enumerate(imgs):
        if img.shape[0] == 3:
            img = img.transpose(1,2,0)
        plt_idx = i+1
        plt.subplot(2, 2, plt_idx)
        plt.imshow(img, format)
    plt.show()
    

#dy = Image.open('donghui.jpg')
#dynp = np.array(dy)
#canny = ['donghui.jpg','donghui_base.png']
#canny = ['mhgu.jpg','bri_mhgu_clean.jpg','bri_mhgu_photo.jpg']
canny = ['bri_acnh_clean.jpg','bri_acnh_photo.jpg']

#for i in ['mhgu.jpg','mhgu2.png','acnh.jpg','acnh2.png']:
imgArray = []
for i in canny:
    imgArray.append(rgb2gray(np.array(Image.open(i))))

detector = cannyEdgeDetector(imgArray, sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.17, weak_pixel=100)
imgs_final = detector.detect()
for i in canny:
    imgs_final.append(np.array(img[i]))
visualize(imgs_final, 'gray')


# without something like a CNN or input from the user, 
# it'll be very difficult to determine table boundaries in images like acnh.jpg
# BUT! for something like mhgu.jpg, it should be easy to detect
# on the other hand, anch.jpg has some fairly obvious row separators,
# whereas mhgu.jpg has many row boundaries cut off
# also acnh.jpg's actual font is perfectly legible just from this method, mhgu.jpg's is not


# I'm pretty sure I accidentally forgot to save my code for finding the exterior edges

# NO I DIDN'T! it was just in image_align.py, whew!

# clears the "interior" of a canny image
def rectangle(image):
    # input should be a canny image
    if image.ndim == 3:
        rect = image[:,:,0].copy() # RGB
    else: # just gonna assume nobody's putting in anything other than 2D or 3D images
        rect = image.copy() # grayscale
    blank = np.zeros(rect.shape) # black image
    for i,j in enumerate(rect.argmax(axis=1)): # argmax gets the first on each array, which is the first from the top here
        blank[i,j] = 255
    for j,i in enumerate(rect.argmax(axis=0)): # and again but first from the left
        blank[i,j] = 255
    rect = np.flip(rect) # gotta flip it so we get the LAST
    blank = np.flip(blank) # and gotta flip this too to get the right one
    for i,j in enumerate(rect.argmax(axis=1)): # first from the bottom (of the original, anyway)
        blank[i,j] = 255
    for j,i in enumerate(rect.argmax(axis=0)): # first from the right (again, from the original's perspective)
        blank[i,j] = 255
    # clean up the image edges
    blank[:,(0,-1)] = 0 # for any columns or rows full of 0s, the above sections make their edges 255
    blank[(0,-1),:] = 0 # can assume we don't want those
    blank = np.flip(blank)
    #Image.fromarray(blank).show()
    return blank # in theory, we could crop to the borders here, but we actually want the padding

