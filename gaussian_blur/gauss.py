import sys
import cv2
import numpy as np
import math
import os

input_image_path = sys.argv[1]
image = cv2.imread(input_image_path)

# kernel = np.array([[0.00000067, 0.00002292, 0.00019117, 0.00038771, 0.00019117, 0.00002292, 0.00000067],
#                   [0.00002292, 0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633, 0.00002292],
#                   [0.00019117, 0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965, 0.00019117],
#                   [0.00038771, 0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373, 0.00038771],
#                   [0.00019117, 0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965, 0.00019117],
#                   [0.00002292, 0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633, 0.00002292],
#                   [0.00000067, 0.00002292, 0.00019117, 0.00038771, 0.00019117, 0.00002292, 0.00000067]])

sigma = 0.84089642 # standard
sigma = 10
n = math.ceil(6 * sigma)
if not n % 2: n += 1

kernel_array = [[(1 / (2 * math.pi * (sigma ** 2))) * (math.e ** -(((i - (n / 2)) ** 2 + (j - (n / 2)) ** 2) / (2 * (sigma ** 2)))) for j in range(n)]
                for i in range(n)]
kernel = np.array(kernel_array)
kernel /= np.sum(kernel)
convolved = cv2.filter2D(image, -1, kernel)

filename, extension = os.path.splitext(input_image_path)
cv2.imwrite(f'{filename}_gauss{extension}', convolved)
