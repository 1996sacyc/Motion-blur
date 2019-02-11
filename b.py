from __future__ import division
import numpy as np
import cv2
import scipy as sp
import math
from matplotlib import pyplot as plt

temp = cv2.imread('Q1.tif')
img = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
xSize, ySize = img.shape

def applyFilter(img, func):
    image = np.copy(img)
    for u in range(0,xSize):
        for v in range(0,ySize):
            image[u,v] = func(u-xSize/2,v-ySize/2)

    return image*img

def H(x,y):
    a = 0.1
    b = 0.1
    T = 1
    C = math.pi*(a*x+b*y)

    if(C == 0):
        return 1

    return (T/C)*math.sin(C)*math.e**(-1j*C)

def toRealnum(img):
    realImg = np.zeros(img.shape)

    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            realImg[i,j] = np.absolute(img[i,j])

    return realImg

def normalize(image):
    img = image.copy()
    img = toRealnum(img)
    img -= img.min()
    img *= 255.0/img.max()
    return img.astype(np.uint8)

f = np.fft.fft2(img.astype(np.int32))
fft_img = np.fft.fftshift(f)
filtered_fft = applyFilter(fft_img, H)
f_fft_img = np.fft.ifftshift(filtered_fft)
filtered_img = np.fft.ifft2(f_fft_img)
filtered_img = normalize(filtered_img)
cv2.imwrite('blurred_b.tif', filtered_img.astype(np.uint8))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(filtered_img, cmap = 'gray')
plt.title('Blurred Image'), plt.xticks([]), plt.yticks([])
plt.show()