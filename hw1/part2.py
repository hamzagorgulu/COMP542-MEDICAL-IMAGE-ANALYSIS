import cv2
import pickle
import numpy as np
from skimage import color
import matplotlib.pyplot as plt
from utils_v2 import clip_img, find_center, xy_image, evaluate_part2_v2, evaluate, check_set

def enhance_sharpness(log_image, g_image):
    log_image = np.absolute(log_image)
    sharp = np.float32(g_image)
    sharp_im = sharp + 2 * log_image
    return sharp_im, log_image

def demonstrate_results(dists, cells, ims):
    horizontals = []
    for distance, cell_gold in zip(dists, cells):
        distance[distance==1] = 255
        cell_gold[cell_gold==1] = 255
        # turn to bgr
        distance = cv2.cvtColor(distance, cv2.COLOR_GRAY2BGR)
        cell_gold = cv2.cvtColor(cell_gold, cv2.COLOR_GRAY2BGR)
        #cv2.imshow("distance", distance)
        #cv2.imshow("cell_gold", cell_gold)
        #breakpoint()
        horizontal = np.concatenate([cell_gold, distance], axis=1)
        cv2.imshow("horizontal", horizontal)
        cv2.waitKey(0)
        horizontals.append(horizontal)

    vertical = np.concatenate(horizontals, axis=0)



def FindCellLocations(image, mask):
    #cv2.imshow("image", image)
    #cv2.imshow("mask", mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    #cv2.imshow("gray mask", mask)
    mask[mask == 1] = 255
    #cv2.imshow("8 bit mask", mask)

    image = cv2.GaussianBlur(image, (5,5), 0)
    #cv2.imshow("Gaussian blur", image)
    masked_image = cv2.bitwise_and(image, image, mask = mask) # this is where mask is used
    #cv2.imshow("bitwise and", masked_image)
    
    red_image = masked_image[:,:,0] # take only the first(red) channel
    #cv2.imshow("red channel", red_image)

    #laplacian gaussian
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    log_image = cv2.filter2D(red_image, cv2.CV_32F, kernel)
    #cv2.imshow("Laplacian Gaussian", log_image)
    
    log_image = np.absolute(log_image)
    #cv2.imshow("np absolute: negative values->positive", log_image)
    sharp = np.float32(red_image)
    #cv2.imshow("np float: should be sharpened", sharp)
    sharp_im = sharp - log_image
    #cv2.imshow("absolute-sharp", sharp_im)
    
    # cv2_imshow(sharp_im)
    sharp_im = clip_img(sharp_im)
    log_image = clip_img(log_image)
    #cv2.imshow("sharp_im after np clipped", sharp_im)
    #cv2.imshow("log_image after np clipped", log_image)


    im_th = cv2.threshold(sharp_im, 160, 255, cv2.THRESH_BINARY)[1]
    #cv2.imshow("sharp_im after thresholding:imt_th", im_th)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    im_th = cv2.morphologyEx(im_th, cv2.MORPH_OPEN, kernel, iterations = 2)
    #cv2.imshow("sharp_im after morphological opening", im_th)


    cells = cv2.bitwise_not(im_th, mask = mask)
    #cv2.imshow("cells after bitwise_not", cells)

    #breakpoint()
    distance = cv2.distanceTransform(cells, cv2.DIST_L2, 3)
    cv2.normalize(distance, distance, 0, 255, cv2.NORM_MINMAX)
    #cv2.imshow("after distance transform to cells", distance)

    distance = cv2.erode(distance, kernel, iterations =2)
    #cv2.imshow("after eroding", distance)

    distance = cv2.morphologyEx(distance, cv2.MORPH_OPEN, kernel, iterations = 1)
    #cv2.imshow("after morphological opening", distance)


    im_th = cv2.threshold(distance, 1, 255, cv2.THRESH_BINARY)[1].astype("uint8")
    #cv2.imshow("after thresholding", im_th)

    # Create the marker image
    markers = np.zeros(distance.shape, dtype=np.int32)

    # find contours
    contours, hierarchies = cv2.findContours(im_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
    print(f'{len(contours)} contour(s) found!')

    # Draw the foreground markers
    for i in range(len(contours)):
        cv2.drawContours(markers, contours, i, (i+1), -1)

    xy = [find_center(c) for c in contours]

    return xy, distance

def main():
    im1_file = "data/im1.jpg"
    im2_file = "data/im2.jpg"
    im3_file = "data/im3.jpg"
    mask_file1 = "data/mask_0.jpg" # foreground mask
    mask_file2 = "data/mask_1.jpg"
    mask_file3 = "data/mask_2.jpg"
    cell1_file = "data/im1_gold_cells.txt"
    cell2_file = "data/im2_gold_cells.txt"
    cell3_file = "data/im3_gold_cells.txt"


    im_files = [im1_file, im2_file, im3_file]
    mask_files = [mask_file1, mask_file2, mask_file3]
    cell_files = [cell1_file, cell2_file, cell3_file]

    ims = []
    masks = []
    cells = []
    locations = []
    dists = []

    for im_file, mask_file, cell_file in zip(im_files, mask_files, cell_files):
        masks.append(cv2.imread(mask_file)) # reads bgr by default
        ims.append(cv2.imread(im_file))
        cells.append(np.loadtxt(cell_file, "uint8"))
    
    for im, mask in zip(ims, masks):
        xy, distance = FindCellLocations(im, mask)
        locations.append(xy)
        dists.append(distance)
    
    for idx, location in enumerate(locations):
        with open(f"results/part2_location_{idx+1}", "wb") as f:
            pickle.dump(location, f)
    
    dist_shape = dists[0].shape

    # save dists
    for idx, distance in enumerate(dists):
        with open(f"results/part2_dist_{idx+1}", "wb") as f:
            pickle.dump(distance, f)
    
    out_lst = []

    for location, cell_gold, distance, im in zip(locations, cells, dists, ims):
        xy = xy_image(location, dist_shape)
        distance[distance>1]=1
        distance = np.array(distance, "uint8")
        cell_gold[cell_gold>1]=1

        out = evaluate_part2_v2(cell_gold, distance)
        out_lst.append(out)

    # visualize the results
    demonstrate_results(dists, cells, ims)
    print(out_lst)

        
    
if __name__ == "__main__":
    main()




