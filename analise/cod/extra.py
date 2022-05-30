import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from PIL import Image, ImageFilter
import cv2


def adjust(vector, alpha, beta):
    aux = np.zeros(255)
    for i in range(0, 255):
        k = int(i * alpha + beta)
        if k > 254:
            k = 254
        elif k < 0:
            k = 0
        aux[k] = aux[k] + vector[i]

    aux[0] = 0
    aux[254] = 0
    return aux


alpha = 1
beta = 0

negative = pd.read_csv("Fcolorednegative.csv")
positive = pd.read_csv("Fcoloredpositive.csv")

negative = np.asarray(negative)
positive = np.asarray(positive)

for channel in range(0, 1):
    negative[:, channel] = (1 / np.sum(negative[:, channel])) * negative[:, channel]
    positive[:, channel] = (1 / np.sum(positive[:, channel])) * positive[:, channel]
    # plt.plot(adjust(negative[:, 1], alpha, beta))
    negative[:, channel] = adjust(negative[:, channel], alpha, beta)
    plt.plot(negative[:, channel])
    plt.show()

    # plt.plot(adjust(positive[:, 1], alpha, beta))
    positive[:, channel] = adjust(positive[:, channel], alpha, beta)
    plt.plot(positive[:, channel])
    plt.show()


# plt.plot(adjust(positive[:, 1], alpha, beta) - adjust(negative[:, 1], alpha, beta))
# plt.plot(positive-negative)
# plt.show()

#
# def plot_image(image_1, image_2, title_1="Orignal", title_2="New Image"):
#     plt.figure(figsize=(10, 10))
#     plt.plot()
#     plt.imshow(image_1)
#     plt.title(title_1)
#     plt.show()
#
#     plt.figure(figsize=(10, 10))
#     plt.plot()
#     plt.imshow(image_2)
#     plt.title(title_2)
#     plt.show()
#
#
# def plot_histograms(image_1, image_2, title_1="Orignal", title_2="New Image"):
#     gray = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
#     gray_scale = cv2.calcHist([gray], [0], None, [256], [0, 256])
#     intensity_values = np.array([x for x in range(gray_scale.shape[0])])
#     plt.bar(intensity_values, gray_scale[:, 0], width=5)
#     plt.title("Bar histogram gray" + title_1)
#     plt.show()
#
#     gray = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
#     gray_scale = cv2.calcHist([gray], [0], None, [256], [0, 256])
#     intensity_values = np.array([x for x in range(gray_scale.shape[0])])
#     plt.bar(intensity_values, gray_scale[:, 0], width=5)
#     plt.title("Bar histogram gray" + title_2)
#     plt.show()
#
#     color_scales = np.transpose(np.array([np.zeros(256), np.zeros(256), np.zeros(256)]))
#     # Red, Green and Blue
#     cv2image = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
#     for channel in range(0, 3):
#         color_scales[:, channel] = color_scales[:, channel] + cv2.calcHist([cv2image], [channel], None, [256],
#                                                                            [0, 256])[:, 0]
#         intensity_values = np.array([x for x in range(color_scales[:, channel].shape[0])])
#         plt.bar(intensity_values, color_scales[:, channel], width=5)
#         plt.title("Bar histogram colors " + str(channel) + title_1)
#         plt.show()
#
#     color_scales = np.transpose(np.array([np.zeros(256), np.zeros(256), np.zeros(256)]))
#     # Red, Green and Blue
#     cv2image = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
#     for channel in range(0, 3):
#         color_scales[:, channel] = color_scales[:, channel] + cv2.calcHist([cv2image], [channel], None, [256],
#                                                                            [0, 256])[:, 0]
#         intensity_values = np.array([x for x in range(color_scales[:, channel].shape[0])])
#         plt.bar(intensity_values, color_scales[:, channel], width=5)
#         plt.title("Bar histogram colors " + str(channel) + title_2)
#         plt.show()

patches = "../heridal/patches"

dimensions = np.array([[0, 0, 0]])

count = 0
fpCount = 0
fnCount = 0
tpCount = 0
tnCount = 0

for label in os.listdir(patches):
    labelPath = os.path.join(patches, label)
    colorScales = np.transpose(np.array([np.zeros(255), np.zeros(255), np.zeros(255)]))
    for file in os.listdir(labelPath):
        count = count + 1

        imagePath = os.path.join(labelPath, file)
        image = Image.open(imagePath)

        (nRows, nCols) = image.size

        # gray = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        # gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        # grayScale = cv2.calcHist([gray], [0], None, [256], [0, 256]) / (81 * 81)
        # grayScale[0] = 0
        # grayScale[255] = 0
        # # plt.plot(grayScale)
        # # plt.show()

        # Red, Green and Blue
        cv2image = cv2.imread(imagePath)
        cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
        for channel in range(0, 3):
            colorScales[:, channel] = colorScales[:, channel] + cv2.calcHist([cv2image], [channel], None, [255],
                                                                             [0, 255])[:, 0] / (nRows * nCols)

        errorPositive = (colorScales - positive)
        errorPositive = np.sum(np.square(errorPositive, errorPositive))
        errorNegative = (colorScales - negative)
        errorNegative = np.sum(np.square(errorNegative, errorNegative))

        if str(label) == "negative":
            if errorPositive < errorNegative:
                tnCount = tnCount + 1
            else:
                fpCount = fpCount + 1
        else:
            if errorPositive < errorNegative:
                fnCount = fnCount + 1
            else:
                tpCount = tpCount + 1

        if count > 1000:
            count = 0
            break


print(tnCount)
print(fpCount)
print(fnCount)
print(tpCount)
