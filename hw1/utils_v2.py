import cv2
import numpy as np


def show(orig, mask):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 800, 300)
    #breakpoint()

    grey_3_channel = cv2.cvtColor(cv2.convertScaleAbs(mask), cv2.COLOR_GRAY2BGR)

    print(grey_3_channel.shape)
    numpy_horizontal_concat = np.concatenate((orig, grey_3_channel), axis=1)
    cv2.imshow('image', numpy_horizontal_concat)
    cv2.waitKey(0)

def show_multiple(imgs):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 800, 300)
    for i in range(len(imgs)):
        imgs[i] = imgs[i].astype(np.uint8)
        if len(imgs[i].shape) != 3 and np.max(imgs[i]) > 1:
            imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_GRAY2BGR)
        elif len(imgs[i].shape) != 3 and np.max(imgs[i]) <= 1:
            imgs[i] = imgs[i]*255
            imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_GRAY2BGR)
    numpy_horizontal_concat = np.concatenate((imgs), axis=1)
    cv2.imshow('image', numpy_horizontal_concat)
    cv2.waitKey(0)


def load(path, mode):
    # mode = 0 for grayscale
    # mode = 1 for rgb
    return cv2.imread(path, mode)

def load_txt(path):
    return np.loadtxt(path)


def zoom(img, zoom):
    #breakpoint()
    cy, cx = [i/2 for i in img.shape[:-1]] # : this line halves the shape of the image.
    rot_mat = cv2.getRotationMatrix2D((cx, cy), 0, zoom)
    return cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)


def adapt_threshold(img):
    img = cv2.GaussianBlur(img, (7, 7), 0)
    # ret,thr = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    #breakpoint()
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # drop channel size to 1
    thr = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
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
    zoomed = cv2.cvtColor(zoom(original, 0.95), cv2.COLOR_RGB2GRAY) # the channel is already grayscale
    th1 = cv2.threshold(zoomed, 1, 255, cv2.THRESH_BINARY)[1]
    return cv2.bitwise_and(th1, mask)


def evaluate(mask, gold):
    
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]
    if np.max(gold) == 255:
        gold = cv2.threshold(gold, 0, 1, cv2.THRESH_BINARY)[1]
    #breakpoint()
    show(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)*255, gold*255)
    TP = np.sum(np.logical_and(mask == 1, gold == 1))
    TN = np.sum(np.logical_and(mask == 0, gold == 0))
    FP = np.sum(np.logical_and(mask == 1, gold == 0))
    FN = np.sum(np.logical_and(mask == 0, gold == 1))
    
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    fscore = 2*TP/(2*TP + FP + FN)
    return {"precision": precision, "recall": recall, "fscore":fscore}

def evaluate_part2_v2(ground_truth, found):
    #ground_truth = cv2.threshold(ground_truth, 1, 256, cv2.THRESH_BINARY)[1]
    TP = np.sum(np.logical_and(found == 1, ground_truth == 1))
    TN = np.sum(np.logical_and(found == 0, ground_truth == 0))
    FP = np.sum(np.logical_and(found == 1, ground_truth == 0))
    FN = np.sum(np.logical_and(found == 0, ground_truth == 1))

    #breakpoint()

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    fscore = 2*TP/(2*TP + FP + FN)
    return {"precision": precision, "recall": recall, "fscore":fscore}

def check_set(img):
    lst = []
    if len(img.shape)==3:     
        for i in img:
            for j in i:
                for k in j:
                    lst.append(k)
    else:
        for i in img:
            for j in i:
                lst.append(j)
    print(set(lst))
    if len(set(lst))>2:
        print("image is not binary")
    else:
        print("image is binary")

def calc_cell_tp(ground_truth, found):
    nm = int(np.max(ground_truth))
    res = 0
    for i in range(1, nm + 1):
        one_count = np.sum(np.logical_and(ground_truth == i, found == 1))
    if one_count == 1:
        res += 1
    print("one count:", one_count)
    return res


def clip_img(img):
    img = img.copy()
    return np.clip(img, 0, 255).astype("uint8")

def find_center(c):
  M = cv2.moments(c)
  cX = int(M["m10"] / M["m00"])
  cY = int(M["m01"] / M["m00"])
  return (cY, cX)

def xy_image(xy, shape):
  image = np.zeros(shape, 'uint8')
  for center in xy:
    image[center] = 1
  return image