import cv2
import numpy as np
import random
from skimage.util import random_noise

# Muhammet Derviş kopuz
# 504201531

def start():
    #read image
    image = cv2.imread("wiki.jpg",0)
    if image is None:
        print("No image available")
        return "No image available"
    salt_pepper_prob = 0.03
    random_numbers = (1,2,3,4,5,6,7,8)
    #choose random number
    N = random.choice(random_numbers)

    testo = image
    for i in range(N):
        #apply salt pepper filter
        testo = salt_pepper(testo,salt_pepper_prob)
        show_image(testo,i)

    testo = image
    for i in range(N):
        #apply ski salt pepper filter
        testo = ski_salt_pepper(testo)
        show_image_ski(testo,i)


def salt_pepper(image,prob):
    #create a 0's matrix with size of image, in uint8 type
    result = np.zeros(image.shape,np.uint8)
    row = image.shape[0]
    column = image.shape[1]
    for i in range(row):
        for j in range(column):
            random_prob = random.random()
            if random_prob < prob:
                #choose between white or black
                result[i][j] = random.choice((0, 255))
            else:
                result[i][j] = image[i][j]
    return result

def show_image(image, number):
    name = "S&p applied " + str((number+1)) + " time/s"
    cv2.imshow(name, image)
    cv2.waitKey(0)

def show_image_ski(image, number):
    name = "Ski S&p applied " + str((number+1)) + " time/s"
    cv2.imshow(name, image)
    cv2.waitKey(0)

def skiNoise(image):
    result = random_noise(image, 'gaussian')
    #function returns between [0,1] range it is multiplied by 255
    result = 255*result
    result = result.astype(np.uint8)
    return result

def ski_salt_pepper(image):
    result = random_noise(image, mode='s&p')
    #function returns between [0,1] range it is multiplied by 255
    result = np.array(255*result, dtype = 'uint8')
    return result

if __name__ == '__main__':
    start()

