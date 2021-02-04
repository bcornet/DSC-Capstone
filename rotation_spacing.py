# -*- coding: utf-8 -*-
"""
Automatically detect rotation and line spacing of an image of text using
Radon transform
If image is rotated by the inverse of the output, the lines will be
horizontal (though they may be upside-down depending on the original image)
It doesn't work with black borders

ALSO!
lots of modifications by Bri!
retain original image for rotation through scipy.ndimage.rotate
matplotlib.mlab.rms_flat is deprecated as of version 2.2
parabolic module is from 2015 so we'll drop that too
need to test on color images!
"""
from __future__ import division, print_function
from skimage.transform import radon
from PIL import Image
import numpy as np
from numpy.fft import rfft
import matplotlib.pyplot as plt
from scipy.ndimage import rotate


def rms_flat(a):
    """
    Return the root mean square of all the elements of *a*, flattened out.
    Copied the deprecated version from matplotlib version 2.2.4.
    """
    return np.sqrt(np.mean(np.abs(a) ** 2))


detailed = False
#filename = 'skew-linedetection.png'
from random import randint
filename = 'test.png'
set_rotation = randint(-45,45)
print("Rotating \"%s\" by %d degrees!"%(filename,set_rotation))


# Load file, converting to grayscale
I = np.asarray(Image.open(filename).convert('L'))
I = rotate(I,set_rotation)
I_orig = I.copy()
I = I - np.mean(I)  # Demean; make the brightness extend above and below zero
#print(type(I),I.shape)

if detailed:
    plt.subplot(2, 3, 1)
    plt.imshow(I_orig)
    plt.subplot(2,3,2)
    plt.imshow(I)
else:
    plt.subplot(1, 2, 1)
    plt.imshow(I_orig)


# Do the radon transform and display the result

# C:\Users\bcorn\Anaconda3\lib\site-packages\skimage\transform\radon_transform.py:87: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
#  coords = np.array(np.ogrid[:image.shape[0], :image.shape[1]])
# C:\Users\bcorn\Anaconda3\lib\site-packages\skimage\transform\radon_transform.py:91: UserWarning: Radon transform: image must be zero outside the reconstruction circle
#  warn('Radon transform: image must be zero outside the '


sinogram = radon(I)

if detailed:
    plt.subplot(2, 3, 4)
    plt.imshow(sinogram.T, aspect='auto')
#plt.gray()

# Find the RMS value of each row and find "busiest" rotation,
# where the transform is lined up perfectly with the alternating dark
# text and white lines
r = np.array([rms_flat(line) for line in sinogram.transpose()])
rotation = np.argmax(r)


# Plot the busy row
row = sinogram[:, rotation]
N = len(row)
if detailed:
    plt.axhline(rotation, color='r')
    plt.subplot(2, 3, 5)
    plt.plot(row)

# Take spectrum of busy row and find line spacing
window = np.blackman(N)
spectrum = rfft(row * window)

frequency = np.argmax(abs(spectrum))
line_spacing = N / frequency  # pixels

if detailed:
    plt.plot(row * window)
    plt.subplot(2, 3, 6)
    plt.plot(abs(spectrum))
    plt.axvline(frequency, color='r')
    plt.yscale('log')
#plt.show()

    plt.subplot(2, 3, 3)
else:
    plt.subplot(1,2,2)
    
rotation = 90-rotation
print('Rotation: {:.2f} degrees'.format(rotation))
print('Line spacing: {:.2f} pixels'.format(line_spacing))

diff = set_rotation+rotation
print("Old:  %4d\nNew:  %4d\nDiff: %4d"%(set_rotation,rotation,diff))
if diff==0:
    print("Image was perfectly rotated back into position!")
else:
    print("oop, this didn't work right")
plt.imshow(rotate(I_orig,rotation))
plt.show()