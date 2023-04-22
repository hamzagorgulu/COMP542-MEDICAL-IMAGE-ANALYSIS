import cv2
import numpy as np
from utils import show, load, readtxt, get_cells_as_singel_objects, get_seeds, get_mean_of_cell, region_grow, filtered_connected_components, evaluate_cells, fuse_predictions_to_print, read_pickle


def segment(img, mask, seed, cell):
    segments = []
    for seed_idx in range(len(seed)):
        print(seed_idx)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blured = cv2.medianBlur(gray, 3)
        out = region_grow(blured, seed[seed_idx]
                          [0], seed[seed_idx][1], 0.38, mask)
        out[out < 1] = 0
        out = out.astype(np.uint8)
        
        if np.sum(out) > 800:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            out = cv2.morphologyEx(out, cv2.MORPH_ERODE, kernel, iterations=2)
            out = filtered_connected_components(out, 8, 1, seed[seed_idx])
            out = cv2.morphologyEx(out, cv2.MORPH_DILATE, kernel, iterations=2)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
        out = cv2.morphologyEx(out, cv2.MORPH_DILATE, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        #show([blured, out, cell[cell_idx]])
        segments.append(out)
    map = fuse_predictions_to_print(segments)
    cell[seed_idx] = cell[seed_idx].astype(np.uint8) # whx
    cell_map = fuse_predictions_to_print(cell)
    show([img, cell_map.astype(np.uint8), map])
    return segments



def main():
    img1_path = "hw1\\data\\im1.jpg"
    gold1_cells_path = "hw1\\data\\im1_gold_cells.txt"
    dist1_cells_path = "hw1\\results\\part2_dist_1"
    gold1_mask_path = "hw1\\data\\im1_gold_mask.txt"
    img2_path = "hw1\\data\\im2.jpg"
    gold2_cells_path = "hw1\\data\\im2_gold_cells.txt"
    dist2_cells_path = "hw1\\results\\part2_dist_2"
    gold2_mask_path = "hw1\\data\\im2_gold_mask.txt"
    img3_path = "hw1\\data\\im3.jpg"
    gold3_cells_path = "hw1\\data\\im3_gold_cells.txt"
    dist3_cells_path = "hw1\\results\\part2_dist_3"
    gold3_mask_path = "hw1\\data\\im3_gold_mask.txt"

    img_paths = [img1_path, img2_path, img3_path]
    gold_cell_paths = [gold1_cells_path, gold2_cells_path, gold3_cells_path]
    gold_mask_paths = [gold1_mask_path, gold2_mask_path, gold3_mask_path]
    dist_cells = [dist1_cells_path, dist2_cells_path, dist3_cells_path]
    imgs = []
    masks = []
    cells = []
    outputs = []
    seeds = []
    dists = []
    cell_objects = []

    for img, mask, dist, cell in zip(img_paths, gold_mask_paths, dist_cells, gold_cell_paths):
        imgs.append(load(img, 1))
        masks.append(readtxt(mask))
        cells.append(readtxt(cell))
        dists.append(read_pickle(dist))
   # show([seeds[0]])

    for img, dist, cell in zip(imgs, dists, cells):
        dist = cv2.threshold(dist, 1, 255, cv2.THRESH_BINARY)[1]
        dist = dist.astype(np.uint8)
        dist = filtered_connected_components(dist, 8, 1, numerate=True)
        dist_object = get_cells_as_singel_objects(dist) #get list of images containting 1 dist in each of them
        cell_objects.append(get_cells_as_singel_objects(cell))

        seed_objects = get_seeds(dist_object, img) # pass object, make it 255 each and remove morph erode
        seeds.append(seed_objects)
        for i in range(len(dist_object)):
            if dist_object[i][seed_objects[i][0], seed_objects[i][1]] != 1:
                print("Seed and cell does not match")


    for img, mask, seed, cell in zip(imgs, masks, seeds, cell_objects):
        output = segment(img, mask, seed, cell)
        outputs.append(output)
        
        

    for img, out, cell in zip(imgs, outputs, cell_objects):
        score = evaluate_cells(out, cell)
        print(score)
        # map = fuse_predictions_to_print(out)
        # show([map])
        


if __name__ == "__main__":
    main()
