import cv2
import numpy as np
import random
from skimage.util import random_noise
import matplotlib.pyplot as plt


def print_hi(name):
    image = cv2.imread("wiki.jpg")
    random_numbers = (1,2,3,4,5,6,7,8)
    N = random.choice(random_numbers)
    fig = plt.figure()
    testo = image
    for i in range(N):
        fig.add_subplot(4, 2, i+1)
        testo = salt_pepper(testo,0.03)
        plt.imshow(testo, cmap = 'gray')
        plt.title(i)

        #plt.imshow(display[i], cmap = 'gray')
    plt.show()

    #cv2.imshow("aksda",testo)
    #cv2.waitKey(0)
    test2 = skiNoise(image)
    cv2.imshow("aksda",test2)
    cv2.waitKey(0)
    print(type(image))


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

def skiNoise(image):
    # Add salt-and-pepper noise to the image.
    noise_img = random_noise(image, 's&p')

    # The above function returns a floating-point image
    # on the range [0, 1], thus we changed it to 'uint8'
    # and from [0,255]
    noise_img = 255*noise_img
    noise_img = noise_img.astype(np.uint8)
    return noise_img

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
