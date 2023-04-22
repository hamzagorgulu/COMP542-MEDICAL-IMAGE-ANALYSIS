import cv2
import numpy as np
from utils import show, load, readtxt, get_cells_as_singel_objects, get_seeds, region_grow, filtered_connected_components, evaluate_cells, fuse_predictions_to_print, read_pickle


def segment(img, mask, seed, cell):
    segments = []
    for seed_idx in range(len(seed)):
        # image preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blured = cv2.medianBlur(gray, 3)
        # regiongrowing from seed
        out = region_grow(blured, seed[seed_idx]
                          [0], seed[seed_idx][1], 0.38, mask)
        out[out < 1] = 0
        out = out.astype(np.uint8)
        # check for leakages
        if np.sum(out) > 800:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            out = cv2.morphologyEx(out, cv2.MORPH_ERODE, kernel, iterations=2)
            out = filtered_connected_components(out, 8, 1, seed[seed_idx])
            out = cv2.morphologyEx(out, cv2.MORPH_DILATE, kernel, iterations=2)
        # postprocess
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
        out = cv2.morphologyEx(out, cv2.MORPH_DILATE, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        segments.append(out)
    # give each cell object a color for showing
    map = fuse_predictions_to_print(segments)
    cell_map = fuse_predictions_to_print(cell)
    show([img, cell_map.astype(np.uint8), map])
    return segments


def main():
    img1_path = "data\\im1.jpg"
    gold1_cells_path = "data\\im1_gold_cells.txt"
    dist1_cells_path = "results\\part2_dist_1"
    mask1_path = "data\\mask_0"
    img2_path = "data\\im2.jpg"
    gold2_cells_path = "data\\im2_gold_cells.txt"
    dist2_cells_path = "results\\part2_dist_2"
    mask2_path = "data\\mask_1"
    img3_path = "data\\im3.jpg"
    gold3_cells_path = "data\\im3_gold_cells.txt"
    dist3_cells_path = "results\\part2_dist_3"
    mask3_path = "data\\mask_2"

    img_paths = [img1_path, img2_path, img3_path]
    gold_cell_paths = [gold1_cells_path, gold2_cells_path, gold3_cells_path]
    mask_paths = [mask1_path, mask2_path, mask3_path]
    dist_cells = [dist1_cells_path, dist2_cells_path, dist3_cells_path]
    imgs = []
    masks = []
    cells = []
    outputs = []
    seeds = []
    dists = []
    cell_objects = []

    # load from files
    for img, mask, dist, cell in zip(img_paths, mask_paths, dist_cells, gold_cell_paths):
        imgs.append(load(img, 1))
        masks.append(readtxt(mask))
        cells.append(readtxt(cell))
        dists.append(read_pickle(dist))

    # create seeds from distance transforms. Cells is processed just for visualization and score caluclation
    for img, dist, cell in zip(imgs, dists, cells):
        dist = cv2.threshold(dist, 1, 255, cv2.THRESH_BINARY)[1]
        dist = dist.astype(np.uint8)
        # connected components to get each distance transform as seperate object
        dist = filtered_connected_components(dist, 8, 1, numerate=True)
        dist_object = get_cells_as_singel_objects(dist)
        cell_objects.append(get_cells_as_singel_objects(cell))

        # take the seed im minimum intensity in the image
        seed_objects = get_seeds(dist_object, img)
        seeds.append(seed_objects)
        for i in range(len(dist_object)):
            if dist_object[i][seed_objects[i][0], seed_objects[i][1]] != 1:
                print("Seed and cell does not match")

    # run segmentation algorithm
    for img, mask, seed, cell in zip(imgs, masks, seeds, cell_objects):
        output = segment(img, mask, seed, cell)
        outputs.append(output)

    # run evaluation
    for img, out, cell in zip(imgs, outputs, cell_objects):
        score = evaluate_cells(out, cell)
        print(score)


if __name__ == "__main__":
    main()
