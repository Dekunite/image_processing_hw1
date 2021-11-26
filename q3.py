import cv2
import numpy as np
import random
from skimage.util import random_noise
import matplotlib.pyplot as plt

def d_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            multiplier = np.random.rand()
            if rdn < prob:
                output[i][j] = image[i][j] * multiplier
            elif rdn > thres:
                output[i][j] = image[i][j] / multiplier
            else:
                output[i][j] = image[i][j]
    return output

def start():
    image = cv2.imread("wiki.jpg",0)
    equalizer(image)
    ski_equalizer(image)


#too similar to skimage,change it
def equalizer(image):
    cdf = plot_hist(image)

    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')

    img2 = cdf[image]
    histogram_equ,bin_edges_equ = np.histogram(img2.flatten(),256,[0,256])
    cdf_equ = histogram_equ.cumsum()
    cdf_normalized_equ = cdf_equ * histogram_equ.max()/ cdf_equ.max()
    plt.plot(cdf_normalized_equ, color = 'b')
    plt.hist(img2.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()

    #show normal image
    cv2.imshow("aksda",image)
    cv2.waitKey(0)
    #show equilazed image
    cv2.imshow("aksda",img2)
    cv2.waitKey(0)

def plot_hist(image):
    histogram, bin_edges = np.histogram(image.flatten(), 256, [0, 256])
    cdf = histogram.cumsum()
    cdf_normalized = cdf * histogram.max() / cdf.max()
    plt.plot(cdf_normalized, color='b')
    plt.hist(image.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()
    return cdf


def ski_equalizer(image):
    plot_hist(image)
    equ = cv2.equalizeHist(image)
    plot_hist(equ)
    res = np.hstack((image,equ)) #stacking images side-by-side
    cv2.imwrite('res.png',res)



if __name__ == '__main__':
    start()

