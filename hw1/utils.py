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

def get_seeds(cells):
    seeds = []
    
    for cell in cells:
        coord = []
        argmax = np.unravel_index(cell.argmax(), cell.shape)

        # for i in range(cell.shape[0]):
        #     for j in range(cell.shape[1]):
        #         if cell[i, j] == 1:
        #             coord.append([i, j])
        # cog = np.mean(coord, axis=0)
        # cog = cog.astype(np.uint16)
        seeds.append(argmax)
    return seeds

def get_mean_of_cell(img, cell):
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

def get4ngb(rows, cols, x, y):
    # function ngb = get4ngb(rows, cols, x, y)
    # 
    # Input:
    # rows, cols    Number of rows and colums from the 2D Image
    # x,y           Coordinates of the center point
    #
    # Output:
    # ngb           list of possible neighbours

    # initialize empty result list
    ngb = list()

    # left
    if x > 1:
        ngb += [[y, x-1]]

    # down
    if y < rows-1:
        ngb += [[y+1, x]]

    # right
    if x < cols-1:
        ngb += [[y, x+1]]

    # up
    if y > 1:
        ngb += [[y-1, x]]

    return ngb 
# Our similarity metric. Given two points with shape=(3,), return true or false if the distance 
# is lower or biger than th. Use the l1 distance (mean of absolute values)
def is_similar(p1, p2, th):
    p1 = p1.astype(np.int16)
    p2 = p2.astype(np.int16)
    # TODO: your metric here.
    dist = np.mean(np.abs(p1 - p2))
    return dist < th


# The algorithm. Our root is the pixel in position (seedX; seedY)
def region_grow(I, seedY, seedX, th):
    if len(I.shape) == 3:
      rows, cols, ch = I.shape
    else:
      rows, cols = I.shape
    # our map for the nodes (pixels) that we have not discovered yet (-1),
    # we have discovered (0)
    # we have discovered and meet the similarity condition (1). This is our segmentation
    R = -np.ones(shape=(rows, cols))
    queue = []
    # enqueue the root
    queue.append([seedY, seedX])
    
    # Setting the root as discovered
    R[seedY, seedX] = 0
    
    #While our queue is not empty
    while len(queue) > 0:
        # we dequeue the first node of the queue
        v = queue[0]
        queue = queue[1:]
        
        # TODO: Use our similarity function to check the condition on the node v
        if is_similar(I[v[0], v[1]], I[seedY, seedX], th): # 
            # TODO: set the node v as "meet the condition", that is, R[vy, vx] = 1
            R[v[0], v[1]] = 1
            # TODO: determine neighbors
            nbgs = get4ngb(rows, cols, v[1], v[0])
            # for all the neighbors of v
            for i in range(0,len(nbgs)):
                qq = nbgs[i]
                # If it was not discovered yet:
                if (R[qq[0], qq[1]] < 0):
                    #TODO: label as discovered
                    R[qq[0], qq[1]] = 0
                    #TODO: enqueue it 
                    queue.append([qq[0], qq[1]])
            
    return R