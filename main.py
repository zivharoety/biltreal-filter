import numpy as np
import math
import sys
import cv2


distances = []
colors = []


def distance(x, y):
    return np.sqrt(x ** 2 + y ** 2)


def gaussian(x, y, sigma):
    if np.isnan(distances[x][y]):
        distances[x][y] = math.exp(- (1 / 2) * ((distance(x, y) / sigma) ** 2))
    return distances[x][y]


def similarity(x, sigma_range):
    if np.isnan(colors[x]):
        colors[x] = math.exp(-(1 / 2) * (x / sigma_range) ** 2)
    return colors[x]


def bff(image, radius, sigma_domain, sigma_range):
    print('bff')
    filtered_image = np.zeros(image.shape)
    for i in range(0, int(len(image))):
        for j in range(0, int(len(image[0]))):
            print(i, j)
            filtered_image[i][j] = bilateral_filter(image, radius, i, j, sigma_domain, sigma_range)
    return filtered_image


def bilateral_filter(image, radius, i, j, sigma_domain, sigma_range):
    k, sum = 0, 0
    for x in range(i - radius, i + radius + 1):
        if 0 <= x < len(image):
            for y in range(j - radius, j + radius + 1):
                if 0 <= y < len(image[0]):
                    g = gaussian(np.absolute(x-i), np.absolute(y-j), sigma_domain)
                    s = similarity(np.absolute(image[x][y] - image[i][j]), sigma_range)
                    sum += (image[x][y] * g * s)
                    k += (g * s)
    if k == 0 or sum == 0:
        return 0
    return sum / k


def main():
    global distances, colors
    x = sys.argv[1]
    sigma_domain = int(sys.argv[2])
    sigma_range = int(sys.argv[3])
    radius = int(sys.argv[4])
    image = cv2.imread(x, 0)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    distances = np.empty(image.shape)
    distances[:] = np.NAN
    colors = np.empty(256)
    colors[:] = np.NAN
    filtered_image = bff(image, radius, sigma_domain, sigma_range)
    cv2.imwrite("output_image1.jpeg", filtered_image)


if __name__ == "__main__":
    main()


