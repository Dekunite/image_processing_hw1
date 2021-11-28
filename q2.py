import math
import cv2
import numpy as np
import random
from skimage.util import random_noise
import matplotlib.pyplot as plt
import timeit

# Muhammet Derviş kopuz
# 504201531

def start():
    #read image
    image = cv2.imread("wiki.jpg",0)
    show_image(image, "original image")

    starttime = timeit.default_timer()
    print("The start time is :",starttime)
    #apply uniform noise
    filter1 = uniform(image)
    show_image(filter1, "uniform noise")
    #apply 3x3, 5x5 and 9x9 median filter
    median_filters(filter1, "median uniform")

    #apply s%p noise
    filter1 = salt_pepper(image,0.1)
    show_image(filter1, "salt and pepper")
    #apply 3x3, 5x5 and 9x9 median filter
    median_filters(filter1, "median salt and pepper")

    #apply gaussian noise
    filter1 = gaussian_noise(image)
    show_image(filter1, "gaussian noise")
    #apply 3x3, 5x5 and 9x9 median filter
    median_filters(filter1, "median gaussian")
    print("The time difference is :", timeit.default_timer() - starttime)

    starttime = timeit.default_timer()
    print("The start time is :",starttime)
    #apply s&p noise by skimage
    filter1 = ski_salt_pepper(image)
    show_image(filter1, "Skimage salt and pepper noise")
    #apply 3x3, 5x5 and 9x9 median filter by cv2
    median_filters_cv2(filter1, "median blurred s&p")

    #apply gaussian noise by skimage
    filter1 = ski_gauss(image)
    show_image(filter1, "Skimage gaussian noise")
    #apply 3x3, 5x5 and 9x9 median filter by cv2
    median_filters_cv2(filter1, "median blurred gaussian")

    #apply speckle noise by skimage
    filter1 = ski_speckle(image)
    show_image(filter1, "Skimage speckle noise")
    #apply 3x3, 5x5 and 9x9 median filter by cv2
    median_filters_cv2(filter1, "median blurred speckle")
    print("The time difference is :", timeit.default_timer() - starttime)


def median_filters(image,name):
    filter1 = median_filter(image,3)
    show_image(filter1, name + " kernel size 3x3")
    filter1 = median_filter(image,5)
    show_image(filter1, name + " kernel size 5x5")
    filter1 = median_filter(image,9)
    show_image(filter1, name + " kernel size 9x9")

def median_filters_cv2(image,name):
    filter1 = cv2_median_blur(image,3)
    show_image(filter1, "Cv2 "+ name + " kernel size 3x3")
    filter1 = cv2_median_blur(image,5)
    show_image(filter1, "Cv2 "+ name + " kernel size 5x5")
    filter1 = cv2_median_blur(image,9)
    show_image(filter1, "Cv2 "+ name + " kernel size 9x9")


def show_image(image, title):
    plt.imshow(image, "gray", aspect='auto')
    plt.title(title)
    plt.show()

def gaussian_noise(image):
    row, column = image.shape
    #create zero matic
    result = np.zeros(image.shape, dtype=np.uint8)
    mean = 1
    var = 50
    sigma = var**0.5
    #gaussin distribution
    gaussian = np.random.normal(mean, sigma, (row, column))
    gaussian = gaussian.reshape(row, column)
    result = image + gaussian
    result = result.astype(np.uint8)
    return result

def uniform(image):
    row, col = image.shape
    #uniform distribution
    normal = np.random.randn(row, col)
    normal = normal.reshape(row, col)
    #lower noise level for clearity
    result = image + (image * normal)*0.1
    result = (result).astype(np.uint8)
    return result

def median_filter(image, kernel):
    rows = len(image)
    columns = len(image[0])
    result = []
    result = np.zeros((len(image), columns))
    offset = kernel / 2
    offset = math.floor(offset)
    zero_pixels = [0]
    zero_pixels = zero_pixels * kernel
    temp = []

    for i in range(rows):
        for j in range(columns):
            for z in range(kernel):
                row_limit = i + z - offset
                top_limit = j + z - offset
                bottom_limit = j + offset
                if row_limit < 0 or row_limit > rows - 1:
                    #non existing pixels inside the kernel are taken as 0
                    temp.extend(zero_pixels)
                elif top_limit < 0 or bottom_limit > columns - 1:
                    #non existing pixels inside the kernel are taken as 0
                    temp.append(0)
                else:
                    for k in range(kernel):
                        #append neighbouring values
                        temp.append(image[row_limit][k - offset + j])
            #sort the temp array to find the middle value
            temp.sort()
            middle = math.floor(len(temp) / 2)
            result[i][j] = temp[middle]
            temp = []
    result = result.astype(np.uint8)
    return result

def salt_pepper(image,prob):
    #create a 0's matrix with size of image, in uint8 type
    result = np.zeros(image.shape,np.uint8)
    row = image.shape[0]
    column = image.shape[1]
    for i in range(row):
        for j in range(column):
            random_prob = random.random()
            if random_prob < prob:
                result[i][j] = random.choice((0, 255))
            else:
                result[i][j] = image[i][j]
    return result

def ski_salt_pepper(image):
    result = random_noise(image, mode='s&p')
    #function returns between [0,1] range it is multiplied by 255
    result = np.array(255*result, dtype = 'uint8')
    return result

def ski_gauss(image):
    result = random_noise(image, mode='gaussian')
    #function returns between [0,1] range it is multiplied by 255
    result = np.array(255*result, dtype = 'uint8')
    return result

def ski_speckle(image):
    result = random_noise(image, mode='speckle')
    #function returns between [0,1] range it is multiplied by 255
    result = np.array(255*result, dtype = 'uint8')
    return result

def cv2_median_blur(image, kernel):
    result = cv2.medianBlur(image, kernel)
    return result


if __name__ == '__main__':
    start()

