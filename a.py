import scipy.misc as s
import numpy as np
import matplotlib.pyplot as plt
import scipy
import cv2

temp = cv2.imread('Q1.tif')
img = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
temp = cv2.imread('psf.tif')
psf = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
H = np.fft.fft2(img)
P = np.fft.fft2(psf)

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

blr = (1.0/(H.shape[0]*H.shape[1]))*np.fft.fftshift(np.fft.ifft2(H*P).real)
# add noise with standard deviation of 0.1
blr = blr+np.random.randn(blr.shape[0],blr.shape[1])*0.1
blr = normalize(blr)
cv2.imwrite('blurred_a.tif', blr.astype(np.uint8))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blr, cmap = 'gray')
plt.title('Blurred Image'), plt.xticks([]), plt.yticks([])
plt.show()
