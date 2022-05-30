import numpy as np
import os
from PIL import Image, ImageFilter
import cv2
import matplotlib.pyplot as plt

# Plota uds imagens, uma ao lado da outra para verifficar a mudança por meio de convolução
def plot_image(image_1, image_2,title_1="Orignal",title_2="New Image"):
    plt.figure(figsize=(10,10))
    plt.plot()
    plt.imshow(image_1)
    plt.title(title_1)
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.plot()
    plt.imshow(image_2)
    plt.title(title_2)
    plt.show()


def plot_histograms(image_1, image_2,title_1="Orignal",title_2="New Image"):
    gray = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    gray_scale = cv2.calcHist([gray], [0], None, [256], [0, 256])
    intensity_values = np.array([x for x in range(gray_scale.shape[0])])
    plt.bar(intensity_values, gray_scale[:, 0], width=5)
    plt.title("Bar histogram gray" + title_1)
    plt.show()

    gray = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    gray_scale = cv2.calcHist([gray], [0], None, [256], [0, 256])
    intensity_values = np.array([x for x in range(gray_scale.shape[0])])
    plt.bar(intensity_values, gray_scale[:, 0], width=5)
    plt.title("Bar histogram gray" + title_2)
    plt.show()

    color_scales = np.transpose(np.array([np.zeros(256), np.zeros(256), np.zeros(256)]))
    # Red, Green and Blue
    cv2image = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
    for channel in range(0, 3):
        color_scales[:, channel] = color_scales[:, channel] + cv2.calcHist([cv2image], [channel], None, [256],
                                                                         [0, 256])[:, 0]
        intensity_values = np.array([x for x in range(color_scales[:, channel].shape[0])])
        plt.bar(intensity_values, color_scales[:, channel], width=5)
        plt.title("Bar histogram colors " + str(channel) + title_1)
        plt.show()

    color_scales = np.transpose(np.array([np.zeros(256), np.zeros(256), np.zeros(256)]))
    # Red, Green and Blue
    cv2image = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
    for channel in range(0, 3):
        color_scales[:, channel] = color_scales[:, channel] + cv2.calcHist([cv2image], [channel], None, [256],
                                                                           [0, 256])[:, 0]
        intensity_values = np.array([x for x in range(color_scales[:, channel].shape[0])])
        plt.bar(intensity_values, color_scales[:, channel], width=5)
        plt.title("Bar histogram colors " + str(channel) + title_2)
        plt.show()


patches = "../heridal/patches"

dimensions = np.array([[0, 0, 0]])
generalGrayScale = np.transpose(np.array([np.zeros(256)]))
generalColorScales = np.transpose(np.array([np.zeros(256), np.zeros(256), np.zeros(256)]))
count = 0

for label in os.listdir(patches):
    labelPath = os.path.join(patches, label)
    # grayScale = np.transpose(np.array([np.zeros(256)]))
    # colorScales = np.transpose(np.array([np.zeros(256), np.zeros(256), np.zeros(256)]))
    for file in os.listdir(labelPath):
        count = count + 1

        imagePath = os.path.join(labelPath, file)
        image = Image.open(imagePath)

        (nRows, nCols) = image.size

        # # Check dimensions
        # found = False
        # for stored in range(0, len(dimensions)):
        #     if nRows == dimensions[stored, 0] and nCols == dimensions[stored, 1]:
        #         dimensions[stored, 2] = dimensions[stored, 2] + 1
        #         found = True
        #         break
        # if not found:
        #     dimensions = np.vstack([dimensions, np.array([nRows, nCols, 1])])

        # # Gray Scale
        # gray = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        # grayScale = grayScale + cv2.calcHist([gray], [0], None, [256], [0, 256])
        # np.savetxt("gray" + str(label) + ".csv", grayScale, delimiter=",")
        #
        # # Red, Green and Blue
        # cv2image = cv2.imread(imagePath)
        # cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
        # for channel in range(0, 3):
        #     colorScales[:, channel] = colorScales[:, channel] + cv2.calcHist([cv2image], [channel], None, [256],
        #                                                                      [0, 256])[:, 0]
        # np.savetxt("colored" + str(label) + ".csv", colorScales, delimiter=",")

        # gray = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        # grayScale = grayScale + cv2.calcHist([gray], [0], None, [256], [0, 256]) / (81 * 81)
        # np.savetxt("Fgray" + str(label) + ".csv", grayScale, delimiter=",")
        #
        # # Red, Green and Blue
        # cv2image = cv2.imread(imagePath)
        # cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
        # for channel in range(0, 3):
        #     colorScales[:, channel] = colorScales[:, channel] + cv2.calcHist([cv2image], [channel], None, [256],
        #                                                                      [0, 256])[:, 0] / (81 * 81)
        # np.savetxt("Fcolored" + str(label) + ".csv", colorScales, delimiter=",")

        image = np.asarray(image)

        # image2 = cv2.imread(imagePath)
        # new_image = cv2.equalizeHist(image2)
        # plot_image(image2, new_image, "Orignal", "Histogram Equalization")

        # Common Kernel for image sharpening
        # kernel = np.array([[-2, -2, -2],
        #                    [-2, 17, -2],
        #                    [-2, -2, -2]])
        kernel = np.array([[1, 0, -1],
                           [1, 0, -1],
                           [1, 0, -1]])
        # # Applys the sharpening filter using kernel on the original image without noise
        sharpened = cv2.filter2D(image, -1, kernel)
        # sharpened = cv2.convertScaleAbs(image, alpha=3.5, beta=-200)
        # Plots the sharpened image and the original image without noise
        plot_image(sharpened, image, title_1="Sharpened image"+str(label), title_2="Image"+str(label))
        plot_histograms(sharpened, image, title_1="Sharpened image" + str(label), title_2="Image" + str(label))

        print(file)

        if count > 0:
            count = 0
            break


    # grayScale = grayScale/count
    # colorScales = colorScales/count
    #
    # intensity_values = np.array([x for x in range(grayScale.shape[0])])
    # plt.bar(intensity_values, grayScale[:, 0], width=5)
    # plt.title("Bar histogram freq gray" + str(label))
    # plt.savefig("Fgray" + str(label))
    # plt.show()
    #
    # for channel in range(0, 3):
    #     intensity_values = np.array([x for x in range(colorScales[:, channel].shape[0])])
    #     plt.bar(intensity_values, colorScales[:, channel], width=5)
    #     plt.title("Bar histogram freq colors " + str(channel) + str(label))
    #     plt.savefig("F" + str(channel) + str(label))
    #     plt.show()
    #
    # generalGrayScale = generalGrayScale + grayScale
    # generalColorScales = generalColorScales + colorScales
