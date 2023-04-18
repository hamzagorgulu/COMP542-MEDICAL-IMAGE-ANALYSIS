import cv2
import numpy as np
import matplotlib.pyplot as plt

def readtxt(path):
    with open(path) as file:
        img = file.readlines()
    return np.loadtxt(img, dtype=np.uint8)

def get_cells_as_singel_objects(cells):
    cell_objects = [(cells == i+1)*1 for i in range(int(np.max(cells)))]
    return cell_objects

def get_seeds(cells, img):
    seeds = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    
    for cell in cells:
        coord = []
        cell = cv2.morphologyEx(cell.astype(np.uint8), cv2.MORPH_ERODE, kernel, iterations=5)
        activations = cell * cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        activations[activations==0] = 255
        argmin = np.unravel_index(activations.argmin(), activations.shape)
        #argmax = np.unravel_index(cell.argmax(), cell.shape)

        # for i in range(cell.shape[0]):
        #     for j in range(cell.shape[1]):
        #         if cell[i, j] == 1:
        #             coord.append([i, j])
        # cog = np.mean(coord, axis=0)
        # cog = cog.astype(np.uint16)
        seeds.append(argmin)
    return seeds

def get_mean_of_cell(img, cell):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(cell.dtype)
    img = img.flatten()
    cell = cell.flatten()
    activation = np.sum(img*cell)
    non_zero = len(np.nonzero(cell)[0])
    mean = activation/(non_zero + 0.0001)
    return mean
        

def show(imgs):
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
    return cv2.imread(path, mode)


def zoom(img, zoom):
    cy, cx = [i/2 for i in img.shape[:-1]]
    rot_mat = cv2.getRotationMatrix2D((cx, cy), 0, zoom)
    return cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

def increase_contrast(img, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
    topHat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    blackHat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    res = img + topHat - blackHat
    return res

def adapt_threshold(img, kernel_size, constant):
    img = cv2.GaussianBlur(img, (7, 7), 0)
    # ret,thr = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    thr = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, kernel_size, constant)
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


def filtered_connected_components(img, connectivity, area_threshold, seed=None):
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
            if seed:
                if componentMask[seed[0], seed[1]] == 0:
                    continue
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

def get4ngb(rows, cols, x, y):
    ngb = list()
    if x > 1:
        ngb += [[y, x-1]]
    if y < rows-1:
        ngb += [[y+1, x]]
    if x < cols-1:
        ngb += [[y, x+1]]
    if y > 1:
        ngb += [[y-1, x]]
    return ngb 

def is_similar(p1, p2, th):
    p1 = p1.astype(np.int16)
    p2 = p2.astype(np.int16)
    dist = np.mean(np.abs(p1 - p2))
    max_dist = (255 - np.mean(p2))*th
    return dist < max_dist


def region_grow(I, seedY, seedX, th, mask):
    if len(I.shape) == 3:
      rows, cols, ch = I.shape
    else:
      rows, cols = I.shape
    mask = 1 - mask
    R = -np.ones(shape=(rows, cols))
    R = R + mask
    queue = []
    queue.append([seedY, seedX])
    
    R[seedY, seedX] = 0
    while len(queue) > 0:
        v = queue[0]
        queue = queue[1:]
        
        if is_similar(I[v[0], v[1]], I[seedY, seedX], th): # 
            R[v[0], v[1]] = 1
            nbgs = get4ngb(rows, cols, v[1], v[0])
            for i in range(0,len(nbgs)):
                qq = nbgs[i]
                if (R[qq[0], qq[1]] < 0):
                    R[qq[0], qq[1]] = 0
                    queue.append([qq[0], qq[1]])
            
    return R

def IoU(pred, target):
    pred[pred > 0] = 1
    pred[target > 0] = 1
    TP = np.sum(np.logical_and(pred == 1, target == 1))
    TN = np.sum(np.logical_and(pred == 0, target == 0))
    FP = np.sum(np.logical_and(pred == 1, target == 0))
    FN = np.sum(np.logical_and(pred == 0, target == 1))
    return TP/(TP + FP + FN)

def evaluate_cells(predictions, targets):
    results = [] # dice0.5, iou0.5 dice0.75,iou0.75, dice0.9, iou0.9
    thresholds = [0.5, 0.75, 0.9]
    for thr in thresholds:
        TP = 0
        FP = 0
        FN = 0
        for i, pred in enumerate(predictions):
            print("pred", i)
            max_iou = 0
            for target in targets:
                if IoU(pred, target) > max_iou:
                    max_iou = IoU(pred, target)
            if max_iou > thr:
                TP += 1
            else:
                FP += 1

        FN = len(targets) - (TP + FP)
        dice = 2*TP / (2*TP + FP + FN)
        iou = TP / (TP + FP + FN)
        results.append(dice)
        results.append(iou)
    return results
        
        
            
def fuse_predictions_to_print(predictions):
    output = np.zeros(predictions[0].shape + (3,), dtype="uint8")
    for pred in predictions:
        pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
        color = list(np.random.random(size=3) * 256)
        pred = pred * color
        output = output + pred
    return output