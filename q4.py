import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import timeit

# Muhammet Derviş kopuz
# 504201531

def start():
    # read image in gray scale
    image = cv2.imread("Frequency_Filter.jpg", 0)

    numpy_butterworth(image)

    starttime = timeit.default_timer()
    print("The start time is :",starttime)
    blurred = cv2_2d_filter(image)
    print("The time difference is :", timeit.default_timer() - starttime)

    show_image(blurred, "Cv2 2D Filter")


def numpy_butterworth(image):
    # use fast sourier transform method in np library to transform image to frequency domain
    original = np.fft.fft2(image)
    # usisng fftshift function we can carry the origin to the center of the domain
    centered = np.fft.fftshift(original)
    # show image
    plt.imshow(image, "gray", aspect='auto')
    plt.title("Original Image")
    plt.show()
    # log of the values are taken to make it easier to see, since values are quite small
    plt.imshow(np.log(0.1 + np.abs(original)), "gray", aspect='auto')
    plt.title("Spectrum")
    plt.show()
    # log is taken for display
    plt.imshow(np.log(0.1 + np.abs(centered)), "gray", aspect='auto')
    plt.title("Spectrum Centered")
    plt.show()
    starttime = timeit.default_timer()
    print("The start time is :",starttime)
    d = 40
    n = 10
    plot_butterworth(d, image, n)
    filter_image(centered, d, image, n)
    print("The time difference is :", timeit.default_timer() - starttime)

    d = 40
    n = 1
    plot_butterworth(d, image, n)
    filter_image(centered, d, image, n)
    d = 20
    n = 10
    plot_butterworth(d, image, n)
    filter_image(centered, d, image, n)


def plot_butterworth(d, image, n):
    buttered = butterworth_low_pass(d, n, image)
    plt.imshow(buttered, "gray", aspect='auto')
    plt.title("Butterworth Low Pass Filter with n=" + str(n) + " D=" + str(d))
    plt.show()

#return back to image with inverse operations
def filter_image(center, d, image, n):
    buttered_center = center * butterworth_low_pass(d, n, image)
    # inverse shifting the center
    buttered = np.fft.ifftshift(buttered_center)
    # inverse fast fourier transform
    inversed = np.fft.ifft2(buttered)
    plt.imshow(np.abs(inversed), "gray", aspect='auto')
    plt.title("Butterworth Low Pass Filter with n=" + str(n) + " D=" + str(d))
    plt.show()


def show_image(image, title):
    plt.imshow(image, "gray", aspect='auto')
    plt.title(title)
    plt.show()


def cv2_2d_filter(image):
    #1/25 blurring kernel
    kernel = np.ones((5, 5), np.float32) / 25
    new_image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    return new_image


def butterworth_low_pass(d_0, n, image):
    shape = image.shape
    rows, columns = shape[:2]
    #get origin of image
    origin = (rows / 2, columns / 2)
    temp = np.zeros(shape[:2])
    for x in range(columns):
        for y in range(rows):
            #apply butterworth function
            temp[y, x] = butterworth_func(x, y, origin, d_0, n)
    return temp


def butterworth_func(x, y, l, d_0, n):
    result = 1 / (1 + (sqrt((y - l[0]) ** 2 + (x - l[1]) ** 2) / d_0) ** (2 * n))
    return result


if __name__ == '__main__':
    start()
