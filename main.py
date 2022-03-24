import numpy as np
import os
from PIL import Image
import cv2

import matplotlib.pyplot as plt

patches = "../heridal/patches"

dimensions = np.array([[0, 0, 0]])
generalGrayScale = np.transpose(np.array([np.zeros(256)]))
generalColorScales = np.transpose(np.array([np.zeros(256), np.zeros(256), np.zeros(256)]))
# count = 0

for label in os.listdir(patches):
    labelPath = os.path.join(patches, label)
    grayScale = np.transpose(np.array([np.zeros(256)]))
    colorScales = np.transpose(np.array([np.zeros(256), np.zeros(256), np.zeros(256)]))
    for file in os.listdir(labelPath):
        # count = count + 1

        imagePath = os.path.join(labelPath, file)
        image = Image.open(imagePath)

        (nRows, nCols) = image.size

        # Check dimensions
        found = False
        for stored in range(0, len(dimensions)):
            if nRows == dimensions[stored, 0] and nCols == dimensions[stored, 1]:
                dimensions[stored, 2] = dimensions[stored, 2] + 1
                found = True
                break
        if not found:
            dimensions = np.vstack([dimensions, np.array([nRows, nCols, 1])])

        # Gray Scale
        gray = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        grayScale = grayScale + cv2.calcHist([gray], [0], None, [256], [0, 256])
        np.savetxt("gray" + str(label) + ".csv", grayScale, delimiter=",")

        # Red, Green and Blue
        cv2image = cv2.imread(imagePath)
        cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
        for channel in range(0, 3):
            colorScales[:, channel] = colorScales[:, channel] + cv2.calcHist([cv2image], [channel], None, [256],
                                                                             [0, 256])[:, 0]
        np.savetxt("colored" + str(label) + ".csv", colorScales, delimiter=",")

    intensity_values = np.array([x for x in range(grayScale.shape[0])])
    plt.bar(intensity_values, grayScale[:, 0], width=5)
    plt.title("Bar histogram gray" + str(label))
    plt.show()
    plt.savefig("gray" + str(label))

    for channel in range(0, 3):
        intensity_values = np.array([x for x in range(colorScales[:, channel].shape[0])])
        plt.bar(intensity_values, colorScales[:, channel], width=5)
        plt.title("Bar histogram colors " + str(channel) + str(label))
        plt.show()
        plt.savefig(str(channel) + str(label))

    generalGrayScale = generalGrayScale + grayScale
    generalColorScales = generalColorScales + colorScales

# intensity_values = np.array([x for x in range(grayScale.shape[0])])
# plt.bar(intensity_values, grayScale[:, 0], width=5)
# plt.title("Bar histogram final gray")
# plt.show()
#
# for channel in range(0, 3):
#     intensity_values = np.array([x for x in range(colorScales[:, channel].shape[0])])
#     plt.bar(intensity_values, colorScales[:, channel], width=5)
#     plt.title("Bar histogram final colors " + str(channel))
#     plt.show()
