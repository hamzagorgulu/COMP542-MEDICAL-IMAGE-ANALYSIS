import cv2
import numpy as np


def show(orig, mask):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 800, 300)
    grey_3_channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    numpy_horizontal_concat = np.concatenate((orig, grey_3_channel), axis=1)
    cv2.imshow('image', numpy_horizontal_concat)
    cv2.waitKey(0)


def load(path, mode):
    return cv2.imread(path, mode)


def zoom(img, zoom):
    cy, cx = [i/2 for i in img.shape[:-1]]
    rot_mat = cv2.getRotationMatrix2D((cx, cy), 0, zoom)
    return cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)


def adapt_threshold(img):
    img = cv2.GaussianBlur(img, (7, 7), 0)
    # ret,thr = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    thr = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    return 255-thr


def thinning(img):
    # Structuring Element
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    # Create an empty output image to hold values
    thin = np.zeros(img.shape, dtype='uint8')

    # Loop until erosion leads to an empty set
    while (cv2.countNonZero(img) != 0):
        # Erosion
        erode = cv2.erode(img, kernel)
        # Opening on eroded image
        opening = cv2.morphologyEx(erode, cv2.MORPH_OPEN, kernel)
        # Subtract these two
        subset = erode - opening
        # Union of all previous sets
        thin = cv2.bitwise_or(subset, thin)
        # Set the eroded image for next iteration
        img = erode.copy()
    return thin


def sharpen_img(img, kernel_size):
    blurred = cv2.blur(img, (kernel_size, kernel_size))
    details = img - blurred
    return img + details


def auto_canny(image, sigma=0.05):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper, 7, L2gradient=True)
    # return the edged image
    return edged


def filtered_connected_components(img, connectivity, area_threshold):
    analysis = cv2.connectedComponentsWithStats(img,
                                                connectivity,
                                                cv2.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis

    # Initialize a new image to store
    # all the output components
    output = np.zeros(img.shape, dtype="uint8")

    # Loop through each component
    for i in range(1, totalLabels):

        # Area of the component
        area = values[i, cv2.CC_STAT_AREA]

        if (area > area_threshold):
            componentMask = (label_ids == i).astype("uint8") * 255
            output = cv2.bitwise_or(output, componentMask)
    return output


def clip_outside(original, mask):
    zoomed = cv2.cvtColor(zoom(original, 0.95), cv2.COLOR_RGB2GRAY)
    th1 = cv2.threshold(zoomed, 1, 255, cv2.THRESH_BINARY)[1]
    return cv2.bitwise_and(th1, mask)


def evaluate(mask, gold):
    
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]
    if np.max(gold) == 255:
        gold = cv2.threshold(gold, 0, 1, cv2.THRESH_BINARY)[1]
        
    show(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)*255, gold*255)
    TP = np.sum(np.logical_and(mask == 1, gold == 1))
    TN = np.sum(np.logical_and(mask == 0, gold == 0))
    FP = np.sum(np.logical_and(mask == 1, gold == 0))
    FN = np.sum(np.logical_and(mask == 0, gold == 1))
    
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    fscore = 2*TP/(2*TP + FP + FN)
    return {"precision": precision, "recall": recall, "fscore":fscore}