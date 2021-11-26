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
    filter1 = uniform(image)
    cv2.imshow("aksda",filter1)
    cv2.waitKey(0)
    filter1 = median_filter(filter1,3)
    cv2.imshow("aksda",filter1)
    cv2.waitKey(0)
    filter1 = ski_salt_pepper(image)
    cv2.imshow("aksda",filter1)
    cv2.waitKey(0)
    filter1 = median_filter(filter1,3)
    cv2.imshow("aksda",filter1)
    cv2.waitKey(0)
    filter1 = ski_gauss(image)
    cv2.imshow("aksda",filter1)
    cv2.waitKey(0)
    filter1 = median_filter(filter1,3)
    cv2.imshow("aksda",filter1)
    cv2.waitKey(0)
    filter1 = ski_speckle(image)
    cv2.imshow("aksda",filter1)
    cv2.waitKey(0)
    filter1 = median_filter(filter1,3)
    cv2.imshow("aksda",filter1)
    cv2.waitKey(0)
    #filter2 = geek(filter1,5)
    #filter3 = median_filter(filter1,5)
    #filter4 = geek(filter1,5)
    #display = [filter1,filter2,filter3,filter4]

    fig = plt.figure(figsize = (12, 10))
    for i in range(4):
        fig.add_subplot(2, 2, i+1)
        #plt.imshow(display[i], cmap = 'gray')
        plt.imshow(filter1, cmap = 'gray')
    plt.show()

def gaussian_noise(image):
    row,col= image.shape
    noisy = np.zeros(image.shape, dtype=np.uint8)
    mean = 1
    var = 100
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col))
    #gauss = gauss.astype(np.uint8)
    gauss = gauss.reshape(row,col)
    noisy = image + gauss
    #cv2.normalize(noisy, noisy, 0, 255, cv2.NORM_MINMAX, dtype = -1)
    noisy = noisy.astype(np.uint8)
    return noisy

def uniform(image):
    row,col = image.shape
    gauss = np.random.randn(row,col)
    gauss = gauss.reshape(row,col)
    #alter noise level for more understandable image
    noisy = image +(image * gauss)*0.1
    noisy = (noisy).astype(np.uint8)
    return noisy

def median_filterD():
    image = cv2.imread("mountain.jpg")
    cv2.imshow("aksda",image)
    cv2.waitKey(0)

def median_filter(data, kernel_size):
    temp = []
    indexer = kernel_size // 2
    data_final = []
    data_final = np.zeros((len(data),len(data[0])))
    rowRange = len(data)
    colRange = len(data[0])

    for i in range(rowRange):

        for j in range(colRange):

            for z in range(kernel_size):
                topLeft = i + z - indexer
                if topLeft < 0 or topLeft > rowRange - 1:
                    for c in range(kernel_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > colRange - 1:
                        temp.append(0)
                    else:
                        for k in range(kernel_size):
                            temp.append(data[topLeft][j + k - indexer])

            temp.sort()
            if (i == j==1):
                print("git")
                print(temp)
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    data_final = data_final.astype(np.uint8)
    return data_final

def geek(img_noisy1, kernel_size):
    # Obtain the number of rows and columns
    # of the image
    m, n = img_noisy1.shape
    #kernel_size = 3 #n x n
    offset = kernel_size // 2 #1
    median = (kernel_size * kernel_size) //2

    # Traverse the image. For every 3X3 area,
    # find the median of the pixels and
    # replace the center pixel by the median
    img_new1 = np.zeros([m, n])

    for i in range(1, m-offset):
        for j in range(1, n-offset):
            temp = []
            for k in range(offset+1):
                for l in range(offset+1):
                    if(k==0 and l==0):
                        temp.append(img_noisy1[i][j])
                    elif(k!=0 and l==0):
                        temp.append(img_noisy1[i-k][j])
                        temp.append(img_noisy1[i+k][j])
                    elif(k==0 and l!=0):
                        temp.append(img_noisy1[i][j-l])
                        temp.append(img_noisy1[i][j+l])
                    else:
                        temp.append(img_noisy1[i-k][j-l])
                        #temp.append(img_noisy1[i-k][j])
                        temp.append(img_noisy1[i-k][j+l])
                        #temp.append(img_noisy1[i][j-k])
                        #temp.append(img_noisy1[i][j+k])
                        temp.append(img_noisy1[i+k][j-l])
                        #temp.append(img_noisy1[i+k][j])
                        temp.append(img_noisy1[i+k][j+l])
            #temp.append(img_noisy1[i][j])

            temp = sorted(temp)
            if (i == j==1):
                print("derv")
                print(temp)
            img_new1[i, j]= temp[median]

    img_new1 = img_new1.astype(np.uint8)
    return img_new1

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
    noise_img = random_noise(image, mode='s&p')

    # The above function returns a floating-point image
    # on the range [0, 1], thus we changed it to 'uint8'
    # and from [0,255]
    noise_img = np.array(255*noise_img, dtype = 'uint8')
    return noise_img

def ski_gauss(image):
    noise_img = random_noise(image, mode='gaussian')

    # The above function returns a floating-point image
    # on the range [0, 1], thus we changed it to 'uint8'
    # and from [0,255]
    noise_img = np.array(255*noise_img, dtype = 'uint8')
    return noise_img

def ski_speckle(image):
    noise_img = random_noise(image, mode='speckle')

    # The above function returns a floating-point image
    # on the range [0, 1], thus we changed it to 'uint8'
    # and from [0,255]
    noise_img = np.array(255*noise_img, dtype = 'uint8')
    return noise_img

if __name__ == '__main__':
    start()

