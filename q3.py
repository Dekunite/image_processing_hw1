import cv2
import numpy as np
import matplotlib.pyplot as plt
import timeit

# Muhammet Derviş kopuz
# 504201531

def start():
    image = cv2.imread("wiki.jpg",0)
    starttime = timeit.default_timer()
    print("The start time is :",starttime)
    equalizer(image)
    print("The time difference is :", timeit.default_timer() - starttime)

    starttime = timeit.default_timer()
    print("The start time is :",starttime)
    cv2_equalizer(image)
    print("The time difference is :", timeit.default_timer() - starttime)

def equalizer(image):
    #show normal image
    show_image(image, "Original image")
    #plot hist original
    cum_sum = plot_hist(image, "Original image")

    #maskes_equal method from numpy is used to, mask values which are equal to the given value, in this case 0
    masked_cum_sum = np.ma.masked_equal(cum_sum,0)
    #normalize the masked array between values 0 and 255
    masked_cum_sum = (masked_cum_sum - masked_cum_sum.min())*255/(masked_cum_sum.max()-masked_cum_sum.min())
    #fill the masked values in the array with given value
    cum_sum = np.ma.filled(masked_cum_sum,0)
    cum_sum = cum_sum.astype(np.uint8)

    #equilaze image
    equilazed_image = [cum_sum[p] for p in list(image.flatten())]
    #reshape list in to image array
    eq_img_array = np.reshape(np.asarray(equilazed_image), image.shape)
    plot_equ_hist(eq_img_array,"Equilazed image")

    #show equilazed image
    show_image(eq_img_array, "Equilazed image method")


def plot_equ_hist(equilazed_image, title):
    # flat the quilazed image in to !D array
    flat_equilazed = equilazed_image.flatten()
    range = [0, 255]
    # plot histogram using numpy
    histogram_equ, bin_edges_equ = np.histogram(flat_equilazed, 256, range)
    # cummulative summation of histogram values
    cum_sum_equ = histogram_equ.cumsum()
    # normalize cummulative summation values
    cum_sum_equ_normal = cum_sum_equ * histogram_equ.max() / cum_sum_equ.max()
    plt.plot(cum_sum_equ_normal, color='r')
    plt.title(title)
    plt.hist(flat_equilazed, 256, range)
    plt.legend(('cdf', 'hist'))
    plt.show()

def show_image(image, title):
    plt.imshow(image, "gray", aspect='auto')
    plt.title(title)
    plt.show()

def plot_hist(image,title):
    range = [0, 255]
    #flat the array in to 1 dimension
    flat_image = image.flatten()
    #use np.shitogram method to draw the histogram of the picture, in the range of 0 and 256
    histogram, bin_edges = np.histogram(flat_image, 256, range)
    #cummulative summation of the histogram values using numpy
    cum_sum = histogram.cumsum()
    #normalize cummulative sum values
    cum_sum_normal = cum_sum * histogram.max() / cum_sum.max()
    #plot cdf
    plt.hist(flat_image, 256, range)
    plt.title(title)
    plt.plot(cum_sum_normal, color='r')
    plt.legend(('cdf','hist'))
    plt.show()
    return cum_sum


def cv2_equalizer(image):
    plot_hist(image,"Original image")
    equ = cv2.equalizeHist(image)
    plot_hist(equ,"Equilazed by cv2")
    show_image(equ, "Cv2 equilazer")

if __name__ == '__main__':
    start()

